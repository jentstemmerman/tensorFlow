import tensorflow as tf
import numpy as np
from tensorflow import keras

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
        super().__init__(self)
        self.numberOfFeatures = len(featureList)    
        self.featureList = featureList
        self.input_layer = []
        self.inputs = []
        self.datatype = datatype

        for feature in featureList:
            if (feature["embedded"] == True):
                self.input_layer.append(tf.keras.layers.Embedding(input_dim=feature["vocab_size"], output_dim=feature["embedding_dim"], dtype=datatype))
                direct_lstm_input = tf.keras.layers.Input(shape=(sequence_length, feature["vocab_size"]))
                self.inputs.append(direct_lstm_input)
            else:
                direct_lstm_input = tf.keras.layers.Input(shape=(sequence_length, 1))
                self.inputs.append(direct_lstm_input)
                input_layer = CustomIdentityLayer(datatype=datatype) 
                self.input_layer.append(input_layer)


        self.concatenated = tf.keras.layers.concatenate(self.inputs)
        self.lstm_layer   = tf.keras.layers.LSTM(units=rnn_units, input_shape=(None, sequence_length), return_sequences=False)

    def call(self, inputs, training=False):
        splitted_inputs = tf.split(inputs, num_or_size_splits=3, axis=0)

        splitted_inputs[0] = tf.squeeze(splitted_inputs[0], axis=0)    
        splitted_inputs[1] = tf.squeeze(splitted_inputs[1], axis=0) 
           

        output0 = self.input_layer[0](tf.cast(splitted_inputs[0], dtype=self.datatype))
        output1 = self.input_layer[1](tf.cast(splitted_inputs[1], dtype=self.datatype))
        
        concatenated_input = tf.concat([output0, output1], axis=-1)  # Adjust axis if needed
        
        # Process the concatenated input through the LSTM layers
        lstm_out = self.lstm_layer(concatenated_input)
        
        return lstm_out
    
   # def GenerateRandomInputData(numberOfFeatures, )

tf.config.run_functions_eagerly(True)  
sequence_length = 16
rnn_units       = 64

featureList = []
featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 129})
featureList.append({"embedded": False})

model = LSTMPrintfLogMultiModel(featureList, rnn_units, sequence_length, datatype = tf.float64)

embedded_input = tf.ones((32, sequence_length))
direct_lstm_input = tf.ones((32, sequence_length, 1))
inputList = [embedded_input, direct_lstm_input]
#output = model(inputList)
#print(output.shape)  # To check the shape of the output



# Generate sample data for demonstration purposes
num_samples = 1000
vocab_size_1 = featureList[0]["vocab_size"]
vocab_size_2 = 1  # For non-embedded feature

embedded_input_data    = np.random.randint(1000, 1000 + vocab_size_1, size=(num_samples, sequence_length))
direct_lstm_input_data = np.random.randint(100, 101, size=(num_samples, sequence_length))


model.compile(optimizer='adam', loss='mse')  # You might want to change the loss function based on your task

regression_labels = np.random.randint(0, 3, size=(num_samples, sequence_length))

dataset = tf.data.Dataset.from_tensor_slices([embedded_input_data, direct_lstm_input_data, regression_labels])
dataset = dataset.shuffle(buffer_size=num_samples).batch(32)  
datasetIterator = iter(dataset)
sample = next(datasetIterator)
model.fit(dataset, epochs=10, batch_size=32)

