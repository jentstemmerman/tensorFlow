
import tensorflow as tf
import numpy as np
import keras
from keras import layers


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

print(tf.__version__)
root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

n_classes = len(np.unique(y_train))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0


class TransformerEncoder(keras.layers.Layer):
    def __init__(self, input_shape, head_size, num_heads, ff_dim, dropout=0):
        super(TransformerEncoder, self).__init__()
        
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.dropout   = dropout

        self.attention     = layers.MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)
        self.dropout_layer = layers.Dropout(self.dropout)
        self.norm_layer    = layers.LayerNormalization(epsilon=1e-6)
        self.Conv1D_layer1 = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")
        self.Conv1D_layer2 = layers.Conv1D(filters=input_shape[-1], kernel_size=1)
    
    def call(self, inputs, training=False):
        x = self.attention(inputs, inputs, training=training)
        x = self.dropout_layer(x, training=training)
        x = self.norm_layer(x, training=training)
        res = x + inputs

        x = self.Conv1D_layer1(res, training=training)
        x = self.dropout_layer(x, training=training)
        x = self.Conv1D_layer2(x, training=training)
        x = self.norm_layer(x, training=training)
        return x + res


class TransformerModel(keras.Model):

    def __init__(self, featureList, sequence_length, datatype, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, mlp_dropout=0, dropout=0):
        super(TransformerModel, self).__init__()
#        self.input_dim = input_shape
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.mlp_dropout = mlp_dropout
        self.dropout = dropout
        self.numberOfFeatures = len(featureList)

        import MultiDimensionalModel
        embeddingDimensionList = MultiDimensionalModel.CalculateEmbeddingDimensions(featureList)

        self.input_layerList  = []
        self.numberOfInputsToLstm = 0

        for i, feature in enumerate(featureList):

            output_dimension = embeddingDimensionList[i]

            if (feature["embedded"] == True):
                input_layer = tf.keras.layers.Embedding(input_dim=feature["vocab_size"], output_dim=output_dimension, dtype=datatype)
            else:
                input_layer = MultiDimensionalModel.CustomIdentityLayer(datatype=datatype) 
            
            self.numberOfInputsToLstm += output_dimension
            
            self.input_layerList.append(input_layer)   

        self.transformerEncoder = []

        for i in range(self.num_transformer_blocks):
            self.transformerEncoder.append(TransformerEncoder(input_shape=(None, sequence_length, self.numberOfInputsToLstm), head_size=self.head_size, num_heads=self.num_heads, ff_dim=self.ff_dim, dropout=self.dropout))

        self.GlobalAveragePooling1DLayer = layers.GlobalAveragePooling1D(data_format="channels_first", name="TooSmallOutput")
        
        self.output_layerList  = []
        self.output_dropout_layerList  = []

        for i, feature in enumerate(featureList):

            layer_name = feature["layer_name"]

            if (feature["embedded"] == True):
                dense_output_layer = tf.keras.layers.Dense(units=feature["vocab_size"], name=layer_name, activation="relu")
            else:
                dense_output_layer = MultiDimensionalModel.CustomDenseLayer(100, 1) # TODO, fix the 100
            output_dropout_layer = layers.Dropout(mlp_dropout)
            self.output_dropout_layerList.append(output_dropout_layer)  
            self.output_layerList.append(dense_output_layer)  

    
    def call(self, inputs, training=False):

        x = inputs
        input_outputs = []
        
        for i, input_layer in enumerate(self.input_layerList):
            input_to_layer = inputs[:, :, i]
            output = input_layer(input_to_layer, training = training)
            input_outputs.append(output)

        concatenated_inputs = tf.keras.layers.concatenate(input_outputs, axis=-1)

        x = concatenated_inputs

        for i in range(self.num_transformer_blocks):
            x = self.transformerEncoder[i](x, training=training)

        x = self.GlobalAveragePooling1DLayer(x, training=training)

        output_List = []
        for index in range(self.numberOfFeatures):
            output = self.output_layerList[index](x)
            output = self.output_dropout_layerList[index](output)
            output_List.append(output)

        return output_List

#