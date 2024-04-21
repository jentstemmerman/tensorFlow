"""
Title: Timeseries classification with a Transformer model
Author: [Theodoros Ntakouris](https://github.com/ntakouris)
Date created: 2021/06/25
Last modified: 2021/08/05
Description: This notebook demonstrates how to do timeseries classification using a Transformer model.
Accelerator: GPU
"""

"""
## Introduction

This is the Transformer architecture from
[Attention Is All You Need](https://arxiv.org/abs/1706.03762),
applied to timeseries instead of natural language.

This example requires TensorFlow 2.4 or higher.

## Load the dataset

We are going to use the same dataset and preprocessing as the
[TimeSeries Classification from Scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch)
example.
"""
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
    def __init__(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, mlp_dropout=0, dropout=0):
        super(TransformerModel, self).__init__()
        self.input_dim = input_shape
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.mlp_dropout = mlp_dropout
        self.dropout = dropout

        self.transformerEncoder = []

        for i in range(self.num_transformer_blocks):
            self.transformerEncoder.append(TransformerEncoder(input_shape=input_shape, head_size=self.head_size, num_heads=self.num_heads, ff_dim=self.ff_dim, dropout=self.dropout))

        self.GlobalAveragePooling1DLayer = layers.GlobalAveragePooling1D(data_format="channels_first", name="TooSmallOutput")
        
        self.Dense = []
        self.DropoutLayerList = []
        for dim in self.mlp_units:
            self.Dense.append(layers.Dense(dim, activation="relu", name="TooFewWeights"))
            self.DropoutLayerList.append(layers.Dropout(self.mlp_dropout))
        
        self.lastDenseLayer = layers.Dense(10, activation="softmax")  # assuming 10 classes

    
    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.num_transformer_blocks):
            x = self.transformerEncoder[i](x, training=training)

        x = self.GlobalAveragePooling1DLayer(x, training=training)
        for index, dim in enumerate(self.mlp_units):
            x = self.Dense[index](x)
            x = self.DropoutLayerList[index](x)

        outputs = self.lastDenseLayer(x)  # assuming 10 classes
        return outputs


input_shape = x_train.shape[1:],

model = TransformerModel(
    input_shape=input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25
)

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
#tf.config.run_functions_eagerly(True)  
#tf.data.experimental.enable_debug_mode()
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=["sparse_categorical_accuracy"])
#model.build(input_shape=x_train.shape[1:])
#model.summary()
model.fit(x_train, y_train, validation_split=0.2, epochs=150, batch_size=64, callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
print(model.evaluate(x_test, y_test, verbose=1))
model.summary()