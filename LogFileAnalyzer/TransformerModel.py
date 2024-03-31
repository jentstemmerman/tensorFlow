import tensorflow as tf
from tensorflow.keras.layers import Input, LayerNormalization, MultiHeadAttention, Dense, Dropout


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, inner_dim, dropout_rate=0.1, attention_dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads =num_attention_heads,
                                                            key_dim=inner_dim // num_attention_heads,
                                                            dropout=attention_dropout_rate)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.norm1    = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1   = tf.keras.layers.Dense(inner_dim, activation='relu')
        self.dense2   = tf.keras.layers.Dense(inner_dim)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.norm2    = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = self.norm1(inputs + attention_output)

        dense_output = self.dense1(attention_output)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout2(dense_output, training=training)
        encoded_output = self.norm2(attention_output + dense_output)

        return encoded_output

class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, inner_dim, dropout_rate=0.1, attention_dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention1 = MultiHeadAttention(num_heads=num_attention_heads,
                                              key_dim=inner_dim // num_attention_heads,
                                              dropout=attention_dropout_rate)
        self.attention2 = MultiHeadAttention(num_heads=num_attention_heads,
                                              key_dim=inner_dim // num_attention_heads,
                                              dropout=attention_dropout_rate)
        self.dropout1 = Dropout(dropout_rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.dropout2 = Dropout(dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout3 = Dropout(dropout_rate)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.dense1 = Dense(inner_dim, activation='relu')
        self.dense2 = Dense(inner_dim)
        self.dropout4 = Dropout(dropout_rate)
        self.norm4 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, encoder_outputs, training=False):
        attention_output = self.attention1(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = self.norm1(inputs + attention_output)

        attention2_output = self.attention2(attention_output, encoder_outputs)
        attention2_output = self.dropout2(attention2_output, training=training)
        attention2_output = self.norm2(attention_output + attention2_output)

        dense_output = self.dense1(attention2_output)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout4(dense_output, training=training)
        encoded_output = self.norm4(attention2_output + dense_output)

        return encoded_output

if __name__ == "__main__":    
    # Define the input shapes
    encoder_input_shape = (None, 512)  # (batch_size, sequence_length)
    decoder_input_shape = (None, 64)   # (batch_size, sequence_length)

    # Create input layers for encoder and decoder
    encoder_inputs = Input(shape=encoder_input_shape)
    decoder_inputs = Input(shape=decoder_input_shape)

    # Create a TransformerEncoderBlock
    encoder_block = TransformerEncoderBlock(
        num_attention_heads=8,
        inner_dim=512,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    )

    # Pass encoder inputs through the encoder block
    encoder_outputs = encoder_block(encoder_inputs)

    # Create a TransformerDecoderBlock
    decoder_block = TransformerDecoderBlock(
        num_attention_heads=8,
        inner_dim=512,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    )

    # Pass decoder inputs and encoder outputs through the decoder block
    decoder_outputs = decoder_block(decoder_inputs, encoder_outputs)

    # Create encoder and decoder models
    encoder_model = tf.keras.Model(inputs=encoder_inputs, outputs=encoder_outputs)
    decoder_model = tf.keras.Model(inputs=[decoder_inputs, encoder_inputs], outputs=decoder_outputs)

    # Display the models summaries
    encoder_model.summary()
    decoder_model.summary()
