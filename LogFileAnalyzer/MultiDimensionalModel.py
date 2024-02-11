import tensorflow as tf
import numpy as np
from tensorflow import keras
import TokenParser
def Custom_loss_function(parameter, DataTypeForTensors):
    verbose = False
    counter = 0
    counterBadStuff = 0
    featureIndex = 0
    numberOfFeatures = len(parameter)
    modelFeatureList = parameter
    DataTypeForTensors = DataTypeForTensors
    maskValueList = np.array([modelFeatureList[featureIndex].get("MaskValue", None) for featureIndex in range(numberOfFeatures)])
    maskValueList = tf.convert_to_tensor(maskValueList, dtype=tf.float32)
  #  print(maskValueList)
    embeddedList  = np.array([modelFeatureList[featureIndex]["embedded"] for featureIndex in range(numberOfFeatures)])
    embeddedList = tf.convert_to_tensor(embeddedList, dtype=tf.bool)
    print(embeddedList)

    # we krijgen elke keer alle features van output true, maar maar 1 feature per call van output predicted. De totale loss wordt buiten deze functie berekend. 
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
            
#            scalar_loss = tf.reduce_mean(feature_output_loss)
#            with open('output_log.txt', 'a') as file:
##                file.write("\n*********************************")
#                file.write("\ncounterBadStuff = {}\n".format(counterBadStuff))
#                file.write("scalar_loss = {}\n".format(scalar_loss))
#                #file.write("feature_output_loss = {}\n".format(np.array2string(feature_output_loss.numpy(), threshold=np.inf)   ))
#                file.write("featureIndex = {}\n".format(featureIndex))
#                #file.write("output_true = {}\n".format(np.array2string(output_true.numpy(), threshold=np.inf)   ))
#                file.write("nfeature_output_predicted_masked = {}\n".format(np.array2string(feature_output_predicted_masked.numpy(), threshold=np.inf)   ))
#                file.write("feature_output_true_masked = {}\n".format(np.array2string(feature_output_true_masked.numpy(), threshold=np.inf)   ))
#                #file.write("mask = {}\n".format(np.array2string(mask.numpy(), threshold=np.inf)   ))
#            
#                file.write("End scalar loss.\n")
#            
#            if (scalar_loss > 1000000):
#                counterBadStuff +=1
#                print("\ncounterBadStuff = ", counterBadStuff)
#                print("\nscalar_loss = ", scalar_loss)
#                print("\nfeature_output_loss = ", feature_output_loss.numpy())
#                print("\nfeatureIndex = ", featureIndex)
##                print("\nmask = ", mask.numpy())
#                print("\noutput_true = ", output_true.numpy())
#                print("\nfeature_output_predicted = ", feature_output_predicted.numpy())
#                print("\nfeature_output_true_masked = ", feature_output_true_masked.numpy())
#                print("\nfeature_output_predicted_masked = ", feature_output_predicted_masked.numpy())
#                print("\nEnd scalar loss.\n")
                


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
         
    def call(self, input, training=False):
        input = self.dense_output_layer1(input, training=training)
        input = self.dense_output_layer2(input, training=training)
        return input

class LSTMPrintfLogMultiModel(tf.keras.Model):

    def __init__(self, featureList, rnn_units, sequence_length, dense_layer, return_sequences, datatype):
        super().__init__()
        self.numberOfFeatures = len(featureList)

        self.numberOfInputsToTotalEmbedding = 0
        self.return_states    = return_sequences
        self.featureList      = featureList
        self.datatype         = datatype
        
        self.input_layerList  = []
        inputList             = []

        for feature in featureList:
            if (feature["embedded"] == True):
                input_layer = tf.keras.layers.Embedding(input_dim=feature["vocab_size"], output_dim=feature["embedding_dim"], dtype=datatype)
                lstm_input  = tf.keras.layers.Input(shape=(sequence_length, feature["vocab_size"]))
                self.numberOfInputsToTotalEmbedding += feature["vocab_size"]
            else:
                input_layer = CustomIdentityLayer(datatype=datatype) 
                
                lstm_input  = tf.keras.layers.Input(shape=(sequence_length, 1))
                self.numberOfInputsToTotalEmbedding += 1
            
            self.input_layerList.append(input_layer)    
            inputList.append(lstm_input)
        
        
        self.concatenated = tf.keras.layers.concatenate(inputList)
        #self.Total_embedding = tf.keras.layers.Embedding(input_dim=self.numberOfInputsToTotalEmbedding, output_dim=self.numberOfFeatures, dtype=datatype)
        self.lstm_layer   = tf.keras.layers.LSTM(units=rnn_units, input_shape=(None, sequence_length, self.numberOfFeatures), return_sequences=return_sequences, dropout=0.1)
        
        self.output_layerList  = []

        for i, feature in enumerate(featureList):

            layer_name = feature["layer_name"]

            if (feature["embedded"] == True):
                dense_output_layer = tf.keras.layers.Dense(units=feature["vocab_size"], name=layer_name)
            else:
                dense_output_layer = CustomDenseLayer(rnn_units, 1) 

            self.output_layerList.append(dense_output_layer)  



    def call(self, inputs, training=False):
        input_outputs = []
        
        for i, input_layer in enumerate(self.input_layerList):
            input_to_layer = inputs[:, :, i]
            output = input_layer(input_to_layer, training = training)
            input_outputs.append(output)

        concatenated_inputs = tf.keras.layers.concatenate(input_outputs, axis=-1)
        #concatenated_inputs = self.Total_embedding(concatenated_inputs, training = training)
        lstm_out = self.lstm_layer(concatenated_inputs, training=training)

        output_List = []

        for i, output_layer in enumerate(self.output_layerList):
            layer_output = output_layer(lstm_out, training=training)
            output_List.append(layer_output)
        
        return output_List
    
if __name__ == '__main__':
    ##################### Settings ######################
    tf.config.run_functions_eagerly(True)  
    tf.data.experimental.enable_debug_mode()

    datatype        = tf.float64
    sequence_length = 16
    rnn_units       = 64
    return_sequences = True
    dense_layer      = True

    featureList = []
    featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100, "name": "output_1"})
    featureList.append({"embedded": False, "name": "output_2"})
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model = LSTMPrintfLogMultiModel(featureList, rnn_units, sequence_length, dense_layer = dense_layer, return_sequences = return_sequences, datatype = datatype)

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



