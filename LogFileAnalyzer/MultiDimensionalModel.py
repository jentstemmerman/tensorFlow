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

    def call(self, inputs, training=False):
        embedded_input, direct_lstm_input = inputs
        embedded_output = self.embedded_layer[0](embedded_input)
        embedded_output2 = self.embedded_layer[1](direct_lstm_input)
        
        concatenated_input = tf.concat([embedded_output, embedded_output2], axis=-1)  # Adjust axis if needed
        
        # Process the concatenated input through the LSTM layers
        lstm_out = self.lstm_layer(concatenated_input)
        
        return lstm_out
    
   # def GenerateRandomInputData(numberOfFeatures, )


sequence_length = 16
rnn_units       = 64

featureList = []
featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 129})
featureList.append({"embedded": False})

model = LSTMPrintfLogMultiModel(featureList, rnn_units, sequence_length)

embedded_input = tf.ones((32, sequence_length))
direct_lstm_input = tf.ones((32, sequence_length, 1))
inputList = [embedded_input, direct_lstm_input]
output = model(inputList)
print(output.shape)  # To check the shape of the output



# Generate sample data for demonstration purposes
num_samples = 1000
vocab_size_1 = featureList[0]["vocab_size"]
vocab_size_2 = 1  # For non-embedded feature
embedded_input_data = np.random.randint(0, vocab_size_1, size=(num_samples, sequence_length))
direct_lstm_input_data = np.random.randn(num_samples, sequence_length, vocab_size_2)

# Compile the model
model.compile(optimizer='adam', loss='mse')  # You might want to change the loss function based on your task

# Generating sample target labels
num_samples = 1000  # Same as the number of samples in input data

# Assuming a regression task
# Generating random regression values for demonstration
regression_labels = np.random.rand(num_samples, 1)

# Assuming a classification task with 3 classes
# Generating random classification labels for demonstration
num_classes = 3
classification_labels = np.random.randint(0, num_classes, size=(num_samples,))

# Using regression_labels or classification_labels as your target labels in model.fit()

# For regression task
model.fit([embedded_input_data, direct_lstm_input_data], regression_labels, epochs=10, batch_size=32)

# For classification task
# Convert labels to one-hot encoded format
#classification_labels_one_hot = tf.keras.utils.to_categorical(classification_labels, num_classes=num_classes)
#model.fit([embedded_input_data, direct_lstm_input_data], classification_labels_one_hot, epochs=10, batch_size=32)