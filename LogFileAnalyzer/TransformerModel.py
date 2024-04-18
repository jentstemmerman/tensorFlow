import tensorflow as tf
import numpy as np
import keras
import math
from keras import layers


class TransformerModel(tf.keras.Model):
    def __init__(self, featureList, sequence_length, return_sequences, datatype):
        super().__init__()
        self.numberOfFeatures = len(featureList)
        self.embeddingDimension = 0
        self.return_states    = return_sequences
        self.featureList      = featureList
        self.datatype         = datatype
        self.embeddingDimensionList = self.CalculateEmbeddingDimensions(featureList)

        self.input_layerList  = []

    def CalculateEmbeddingDimensions(self, featureList):
        embeddingDimension = []
        total_output_dimensions = 0

        for i, feature in enumerate(featureList):
            if (feature["embedded"] == True):
                output_dimension = math.ceil(feature["vocab_size"]**0.25)
            else:
                output_dimension = 1

            embeddingDimension.append(output_dimension)
            total_output_dimensions += output_dimension
        print("Embedding dimensions:")
        print(embeddingDimension)
        return embeddingDimension


    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res