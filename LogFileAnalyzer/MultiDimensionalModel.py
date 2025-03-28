import tensorflow as tf
import numpy as np
from tensorflow import keras
import TokenParser
import Dataset
import math 
import ModelHash


def Custom_loss_function(parameter, maskValueList, DataTypeForTensors):
    verbose = False
    counter = 0
    counterBadStuff = 0
    featureIndex = 0
    numberOfFeatures = len(parameter)
    modelFeatureList = parameter

    maskValueList = tf.convert_to_tensor(maskValueList, dtype=DataTypeForTensors)
    embeddedList  = np.array([modelFeatureList[featureIndex]["embedded"] for featureIndex in range(numberOfFeatures)])
    embeddedList = tf.convert_to_tensor(embeddedList, dtype=tf.bool)
    print(embeddedList)

    # We krijgen elke keer alle features van output true, maar maar 1 feature per call van output predicted. De totale loss wordt buiten deze functie berekend. 
    # Voor LSTM: 
    # output_true.shape                = (32,32,4)
    # feature_output_predicted.shape   = (32,32,12530)
    #
    # voor Transformer:
    # output_true.shape                = (32,32,4)
    # feature_output_predicted.shape   = (32, 12530)

    @tf.function
    def custom_multi_output_loss_jents(output_true, feature_output_predicted):
        nonlocal counterBadStuff
        nonlocal featureIndex
        nonlocal numberOfFeatures
        nonlocal modelFeatureList
        nonlocal counter
        nonlocal maskValueList
        counter += 1
        
        feature_output_true_masked      = None
        feature_output_predicted_masked = None

        feature_output_true = tf.gather(output_true, featureIndex, axis=-1)
        feature_output_true = tf.expand_dims(feature_output_true,  axis=-1 )

        if (modelFeatureList[featureIndex]["embedded"] == True):
            feature_output_loss = tf.keras.losses.sparse_categorical_crossentropy(feature_output_true, feature_output_predicted, from_logits=True)
        else:
            mask_value = maskValueList[featureIndex]
         
            mask = tf.not_equal(feature_output_true, mask_value)
            mask = tf.cast(mask, dtype=tf.float32)
            
            feature_output_true_masked      = feature_output_true      * mask
            feature_output_predicted_masked = feature_output_predicted * mask

            feature_output_loss = tf.keras.losses.mean_squared_error(feature_output_true_masked, feature_output_predicted_masked)
                
        featureIndex = (featureIndex + 1) % numberOfFeatures

        return feature_output_loss
    return custom_multi_output_loss_jents

class CustomIdentityLayer(tf.keras.layers.Layer):
    def __init__(self, datatype):
        super().__init__()
        self.factor = 0
        self.datatype = datatype
        self.normalisationLayer = tf.keras.layers.BatchNormalization()
    def get_config(self):
        return {"factor": self.factor}
     
    @tf.function
    def call(self, input, training=False):
        input = tf.cast(input, dtype=self.datatype)
        input = self.normalisationLayer(input, training=training)
        input = tf.expand_dims(input = input, axis=-1)
        return input
    
class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, numberOfInputs, numberOfOutputs):
        super().__init__()
        self.numberOfCells = 0
        self.dense_output_layer1 = tf.keras.layers.Dense(units=20, input_shape=(numberOfInputs,), activation='relu')
        self.dense_output_layer2 = tf.keras.layers.Dense(units=1, input_shape=(numberOfOutputs,))
    def get_config(self):
        return {"layer1": self.dense_output_layer1, "layer2": self.dense_output_layer2}
    
    @tf.function     
    def call(self, input, training=False):
        input = self.dense_output_layer1(input, training=training)
        input = self.dense_output_layer2(input, training=training)
        return input

def round_up_to_power_of_two(num):
    # If the number is already a power of two, return it
    if num & (num - 1) == 0:
        return num
    
    # Otherwise, find the next power of two
    return 2 ** math.ceil(math.log2(num))

def CalculateEmbeddingDimensions(featureList):
    embeddingDimension = []
    total_output_dimensions = 0

    for i, feature in enumerate(featureList):
        if (feature["embedded"]):
            output_dimension = math.ceil(feature["vocab_size"]**0.25)
            if output_dimension % 2 != 0:
                output_dimension += 1  # Ensure depth is even
        else:
            output_dimension = 1  # Maybe change this from 1 to 2 because otherwhise the positional embedding does not work.
            
        embeddingDimension.append(output_dimension)
        total_output_dimensions += output_dimension
    print("Embedding dimensions:")
    print(embeddingDimension)
    return embeddingDimension

class LSTMPrintfLogMultiModel(tf.keras.Model):
    def __init__(self, featureList, rnn_units, sequence_length, return_sequences, datatype, hashAtCreation):
        super().__init__()

        self.hashAtCreation = ModelHash.ModelHash(hashAtCreation) # if hashAtCreation is None, a new hash is created, else hashAtCreation is used
            
        activationFunctionForDenseLayerAfterLstm = None #"elu"  ## half as slow as None
        self.numberOfFeatures = len(featureList)
        self.numberOfInputsToLstm = 0
        self.numberOfInputsToTotalEmbedding = 0
        self.return_states    = return_sequences
        self.featureList      = featureList
        self.datatype         = datatype
        self.rnn_units        = rnn_units
        self.sequence_length  = sequence_length
        self.return_sequences = return_sequences
        
        embeddingDimensionList = CalculateEmbeddingDimensions(featureList)

        self.input_layerList  = []


        for i, feature in enumerate(featureList):

            output_dimension = embeddingDimensionList[i]
        
            if (feature["embedded"] == True):
                input_layer = tf.keras.layers.Embedding(input_dim=feature["vocab_size"], output_dim=output_dimension, dtype=datatype)
            else:
                input_layer = CustomIdentityLayer(datatype=datatype) 
            
            self.numberOfInputsToLstm += output_dimension
            
            self.input_layerList.append(input_layer)    

     #   import Transformer_DecoderOnly
     #   self.decoderOnlyTransformerLayer = Transformer_DecoderOnly.DecoderOnlyTransformerLayer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
    #                         vocab_size=input_vocab_size, featureList=featureList, seq_length=seq_length, dropout_rate=0.1)
    
        self.lstm_layer       = tf.keras.layers.LSTM(units=rnn_units, input_shape=(None, sequence_length, self.numberOfInputsToLstm), return_sequences=return_sequences, dropout=0.1)
        
        self.output_layerList  = []

        for i, feature in enumerate(featureList):

            layer_name = feature["layer_name"]

            if (feature["embedded"] == True):
                dense_output_layer = tf.keras.layers.Dense(units=feature["vocab_size"], name=layer_name, activation=activationFunctionForDenseLayerAfterLstm)
            else:
                dense_output_layer = CustomDenseLayer(rnn_units, 1) 

            self.output_layerList.append(dense_output_layer)  

    def GetHashAtCreation(self):
        return self.hashAtCreation.getHash()

    def get_config(self):
        config = super().get_config()
        config.update({
            "featureList": self.featureList,
            "rnn_units": self.rnn_units,
            "return_sequences": self.return_sequences,
            "datatype": self.datatype,
            "sequence_length": self.sequence_length,
            "hashAtCreation": self.hashAtCreation.getHash()
        })
        return config

    @classmethod
    def from_config(cls, config):
        hashAtCreation = config.pop("hashAtCreation")
        model = cls(
            featureList=config['featureList'],
            rnn_units=config['rnn_units'],
            return_sequences=config['return_sequences'],
            datatype=config['datatype'],
            sequence_length=config['sequence_length'],
            hashAtCreation=config['hashAtCreation'],
        )

    def call(self, inputs, training=False):
        input_outputs = []
        
        for i, input_layer in enumerate(self.input_layerList):
            input_to_layer = inputs[:, :, i]
            output = input_layer(input_to_layer, training = training)
            input_outputs.append(output)

        concatenated_inputs = tf.keras.layers.concatenate(input_outputs, axis=-1)
        #concatenated_input_to_lstm = self.Total_embedding(concatenated_inputs, training = training)
        concatenated_input_to_lstm = concatenated_inputs
        lstm_out = self.lstm_layer(concatenated_input_to_lstm, training=training)

        output_List = []

        for i, output_layer in enumerate(self.output_layerList):
            layer_output = output_layer(lstm_out, training=training)
            output_List.append(layer_output)
        
        return output_List    
    
if __name__ == '__main__':
    ##################### Settings ######################
    #tf.config.run_functions_eagerly(True)  
    #tf.data.experimental.enable_debug_mode()

    datatype        = tf.float64
    sequence_length = 16
    rnn_units       = 64
    return_sequences = True
    dense_layer      = True

    featureList = []
    featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100, "name": "output_1"})
    featureList.append({"embedded": False, "name": "output_2"})
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model = LSTMPrintfLogMultiModel(featureList, rnn_units, sequence_length, return_sequences = return_sequences, datatype = datatype)

    name0 = featureList[0]["name"]
    name1 = featureList[1]["name"]
    model.compile(optimizer='adam', loss={name0: 'SparseCategoricalCrossentropy', name1: 'mean_squared_error'}, metrics=['accuracy'])

    num_samples = 1000
    vocab_size_1 = featureList[0]["vocab_size"]
    vocab_size_2 = 1  

    ##################### inference ######################

    embedded_input = tf.ones((32, sequence_length))
    direct_lstm_input = tf.ones((32, sequence_length))

    datasetInputData_tensor = tf.stack((embedded_input, direct_lstm_input), axis = -1)

    modelPrediction = model(datasetInputData_tensor)
    print("modelPrediction: ")
    print(modelPrediction[0].shape)
    print(modelPrediction[1].shape)

    modelPredictionShape = model.compute_output_shape(datasetInputData_tensor.shape)
    print("modelPredictionShape: ")
    print(modelPredictionShape)
    ##################### training ######################
    embedded_input_data    = np.random.randint(1000, 1000 + vocab_size_1, size=(num_samples))
    direct_lstm_input_data = np.random.randint(100, 105,                  size=(num_samples ))

    if (dense_layer):
        if (not return_sequences):
            regression_labels      = np.random.randint(0, 3,              size=(num_samples, featureList[0]["vocab_size"] ))
        else:
            regression_labels      = np.random.randint(0, 3,              size=(num_samples, sequence_length, featureList[0]["vocab_size"] ))
    else:
        if return_sequences:
            regression_labels      = np.random.randint(0, 3,              size=(num_samples, sequence_length, rnn_units))
        else:
            regression_labels      = np.random.randint(0, 3,              size=(num_samples))

    embedded_input_data_tensor    = tf.convert_to_tensor(embedded_input_data)
    direct_lstm_input_data_tensor = tf.convert_to_tensor(direct_lstm_input_data)
    regression_labels_tensor      = tf.convert_to_tensor(regression_labels)


    datasetInputData_tensor = tf.stack((embedded_input_data_tensor, direct_lstm_input_data_tensor), axis = 1)
    dataset_tensor = tf.data.Dataset.from_tensor_slices(datasetInputData_tensor)

    dataset_tensor = dataset_tensor.window(sequence_length + 1, shift=1, drop_remainder=True)
    dataset_tensor = dataset_tensor.flat_map(lambda window2: window2)
    dataset_tensor = dataset_tensor.batch(sequence_length + 1, drop_remainder = True)
    dataset_tensor = dataset_tensor.map(TokenParser.split_input_target_multidim) 
    dataset = dataset_tensor.batch(2, drop_remainder = True)
    dataset_iterator = iter(dataset)
    print(next(dataset_iterator))
    print(next(iter(dataset)))
    
    model.fit(dataset, epochs=5)


    ##################### inference ######################

    embedded_input = tf.ones((32, sequence_length))
    direct_lstm_input = tf.ones((32, sequence_length))

    datasetInputData_tensor = tf.stack((embedded_input, direct_lstm_input), axis = 1)

    modelPrediction = model(datasetInputData_tensor)
    print("modelPrediction: ")
    print(modelPrediction.shape)




