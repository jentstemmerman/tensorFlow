import tensorflow as tf
import numpy as np
from tensorflow import keras


class LSTMPrintfLogMultiModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units, numberOfFeatures, sequence_length):
        super().__init__(self)
        self.numberOfFeatures = numberOfFeatures    
        
        self.embedded_input    = tf.keras.layers.Input(shape=(sequence_length,))         # create keras tensor embedded input
        self.direct_lstm_input = tf.keras.layers.Input(shape=(sequence_length, 1)) 

        self.embedded_layer    = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        
        self.lstm_layer_for_embedded     = tf.keras.layers.LSTM(units=64, input_shape=(None, 32))
        self.lstm_layer_for_direct_input = tf.keras.layers.LSTM(units=64, input_shape=(None, 1))
        self.concatenated                = tf.keras.layers.Concatenate()

    def call(self, embedded_input, direct_lstm_input, training=False):
        embedded_output = self.embedded_layer(embedded_input)
        lstm_out_embedded = self.lstm_layer_for_embedded(embedded_output)
        lstm_out_direct = self.lstm_layer_for_direct_input(direct_lstm_input)
        concatenated = self.concatenated([lstm_out_embedded, lstm_out_direct])
        
        return concatenated

vocab_size = 1000
sequence_length = 10
embedding_dim = 129
rnn_units = 256
numberOfFeatures = 2




# Assuming sequence_length is defined
model = LSTMPrintfLogMultiModel(vocab_size, embedding_dim, rnn_units, numberOfFeatures, sequence_length)

# Generate sample input tensors
embedded_input = tf.ones((32, sequence_length))
direct_lstm_input = tf.ones((32, sequence_length, 1))

# Obtain the model output by calling the model object
output = model(embedded_input, direct_lstm_input)
print(output.shape)  # To check the shape of the output
