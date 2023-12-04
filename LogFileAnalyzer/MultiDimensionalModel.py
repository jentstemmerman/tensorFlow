import tensorflow as tf
import numpy as np
from tensorflow import keras


class LSTMPrintfLogMultiModel(tf.keras.Model):

    def __init__(self, featureList, rnn_units, sequence_length):
        super().__init__(self)
        self.numberOfFeatures = len(featureList)    
        self.featureList = featureList
        self.embedded_layer = []
        self.inputs = []

        for feature in featureList:
            if (feature["embedded"] == True):
                self.embedded_layer.append(tf.keras.layers.Embedding(input_dim=feature["vocab_size"], output_dim=feature["embedding_dim"]))
                direct_lstm_input = tf.keras.layers.Input(shape=(sequence_length, feature["vocab_size"]))
                self.inputs.append(direct_lstm_input)
            else:
                direct_lstm_input = tf.keras.layers.Input(shape=(sequence_length, 1))
                self.inputs.append(direct_lstm_input)
                input_layer = tf.keras.layers.Identity() 
                self.embedded_layer.append(input_layer)


        self.concatenated = tf.keras.layers.concatenate(self.inputs)
        self.lstm_layer   = tf.keras.layers.LSTM(units=rnn_units, input_shape=(None, sequence_length), return_sequences=False)

    def call(self, embedded_input, direct_lstm_input, training=False):
        
        embedded_output = self.embedded_layer[0](embedded_input)
        embedded_output2 = self.embedded_layer[1](direct_lstm_input)
        
        concatenated_input = tf.concat([embedded_output, embedded_output2], axis=-1)  # Adjust axis if needed
        
        # Process the concatenated input through the LSTM layers
        lstm_out = self.lstm_layer(concatenated_input)
        
        return lstm_out
    
   # def GenerateRandomInputData(numberOfFeatures, )


sequence_length = 16
rnn_units = 64

featureList = []
featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 129})
featureList.append({"embedded": False})

# Assuming sequence_length is defined
model = LSTMPrintfLogMultiModel(featureList, rnn_units, sequence_length)

# Generate sample input tensors
embedded_input = tf.ones((32, sequence_length))
direct_lstm_input = tf.ones((32, sequence_length, 2))

# Obtain the model output by calling the model object
output = model(embedded_input, direct_lstm_input)
print(output.shape)  # To check the shape of the output
