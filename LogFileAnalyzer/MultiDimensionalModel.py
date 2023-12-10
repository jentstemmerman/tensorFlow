import tensorflow as tf
import numpy as np
from tensorflow import keras
import TokenParser

class CustomIdentityLayer(tf.keras.layers.Layer):
    def __init__(self, datatype):
        super().__init__()
        self.factor = 0
        self.datatype = datatype
    def get_config(self):
        return {"factor": self.factor}
     
    
    def call(self, input, training=False):
        input = tf.cast(input, dtype=self.datatype)
        input = tf.expand_dims(input = input, axis=-1)
        return input

class LSTMPrintfLogMultiModel(tf.keras.Model):

    def __init__(self, featureList, rnn_units, sequence_length, datatype):
        super().__init__()
        self.numberOfFeatures = len(featureList)    
        self.featureList = featureList
        self.input_layerList= []
        inputList = []
        self.datatype = datatype

        for feature in featureList:
            if (feature["embedded"] == True):
                input_layer = tf.keras.layers.Embedding(input_dim=feature["vocab_size"], output_dim=feature["embedding_dim"], dtype=datatype)
                lstm_input  = tf.keras.layers.Input(shape=(sequence_length, feature["vocab_size"]))
            else:
                input_layer = CustomIdentityLayer(datatype=datatype) 
                lstm_input  = tf.keras.layers.Input(shape=(sequence_length, 1))

            self.input_layerList.append(input_layer)    
            inputList.append(lstm_input)

        self.concatenated = tf.keras.layers.concatenate(inputList)
        self.lstm_layer   = tf.keras.layers.LSTM(units=rnn_units, input_shape=(None, sequence_length), return_sequences=False)

    def call(self, inputs, training=False):
        input_outputs = []
        for i, input_layer in enumerate(self.input_layerList):
            output = input_layer(inputs[:, :, i], training=training)
            input_outputs.append(output)
        #print(inputs)
        concatenated_inputs = tf.keras.layers.concatenate(input_outputs, axis=-1)
        #print(concatenated_inputs.shape)
        lstm_output = self.lstm_layer(concatenated_inputs, training=training)
        
        return lstm_output
    

datatype = tf.float64
tf.config.run_functions_eagerly(True)  
tf.data.experimental.enable_debug_mode()
sequence_length = 16
rnn_units       = 64

featureList = []
featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100})
featureList.append({"embedded": False})

model = LSTMPrintfLogMultiModel(featureList, rnn_units, sequence_length, datatype = datatype)
model.compile(optimizer='adam', loss='mse')  # You might want to change the loss function based on your task

embedded_input = tf.ones((32, sequence_length))
direct_lstm_input = tf.ones((32, sequence_length, 1))
inputList = [embedded_input, direct_lstm_input]


num_samples = 20
vocab_size_1 = featureList[0]["vocab_size"]
vocab_size_2 = 1  # For non-embedded feature


embedded_input_data    = np.random.randint(1000, 1000 + vocab_size_1, size=(num_samples, sequence_length))
direct_lstm_input_data = np.random.randint(100, 105, size=(num_samples, sequence_length))
regression_labels      = np.random.randint(0, 3, size=(num_samples))

embedded_input_data_tensor    = tf.convert_to_tensor(embedded_input_data)
direct_lstm_input_data_tensor = tf.convert_to_tensor(direct_lstm_input_data)
regression_labels_tensor      = tf.convert_to_tensor(regression_labels)

datasetInputData_tensor = tf.stack((embedded_input_data_tensor, direct_lstm_input_data_tensor), axis = 1)
dataset_tensor = tf.data.Dataset.from_tensor_slices((datasetInputData_tensor, regression_labels_tensor))
iterator = iter(dataset_tensor)
print(next(iterator))
print(next(iterator))
dataset = dataset_tensor.batch(1, drop_remainder = True)

model.fit(dataset, epochs=10)




