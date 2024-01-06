import tensorflow as tf
import numpy as np
from tensorflow import keras
import TokenParser

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size_1, vocab_size_2, embedding_dim, lstm_units, dense_units_1, dense_units_2):
        super(MyModel, self).__init__()
        
        self.embedding_layer_1 = tf.keras.layers.Embedding(input_dim=vocab_size_1, output_dim=embedding_dim)
        self.embedding_layer_2 = tf.keras.layers.Embedding(input_dim=vocab_size_2, output_dim=embedding_dim)
        self.lstm = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True)
        self.dense_1 = tf.keras.layers.Dense(units=dense_units_1)
        self.dense_2 = tf.keras.layers.Dense(units=dense_units_2)
    
    def call(self, inputs):
        input_1, input_2 = inputs
        
        embedded_1 = self.embedding_layer_1(input_1)
        embedded_2 = self.embedding_layer_2(input_2)
        
        concatenated = tf.concat([embedded_1, embedded_2], axis=-1)
        
        lstm_output = self.lstm(concatenated)
        
        output_1 = self.dense_1(tf.reduce_mean(lstm_output, axis=1))  # Dense for first feature
        output_2 = self.dense_2(tf.reduce_mean(lstm_output, axis=1))  # Dense for second feature
        
        return output_1, output_2

tf.config.run_functions_eagerly(True)  
#tf.data.experimental.enable_debug_mode()
# Example usage to create a dataset and train the model
vocab_size_1 = 10
vocab_size_2 = 20
embedding_dim = 32
lstm_units = 64
dense_units_1 = 32
dense_units_2 = 16

# Dummy data generation (replace this with your dataset)
data_size = 1000
input_1_data = tf.random.uniform((data_size, 1), maxval=vocab_size_1, dtype=tf.int32)
input_2_data = tf.random.uniform((data_size, 1), maxval=vocab_size_2, dtype=tf.int32)
target_1_data = tf.random.normal((data_size, dense_units_1))
target_2_data = tf.random.normal((data_size, dense_units_2))

dataset = tf.data.Dataset.from_tensor_slices(((input_1_data, input_2_data), (target_1_data, target_2_data)))
dataset = dataset.shuffle(buffer_size=data_size).batch(32)

# Instantiate the model
model = MyModel(vocab_size_1, vocab_size_2, embedding_dim, lstm_units, dense_units_1, dense_units_2)

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(dataset, epochs=5)