import tensorflow as tf
import numpy as np

class LSTMPrintfLogModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        # Embedding layer
        embedded = self.embedding(inputs, training=training)

        # LSTM layer
        lstm_output = self.lstm(embedded, training=training)

        # Fully connected layer to produce logits
        logits = self.dense(lstm_output, training=training)

        return logits
    
def SimpleMultiInputLSTM():
    # create a sample input
    input_data = np.random.rand(17, 16, 3)

    # load the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, input_shape=(16, 3)))   #  activation='relu' is not supported by cuda
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')


    # run inference
    output = model.predict(input_data)


SimpleMultiInputLSTM()