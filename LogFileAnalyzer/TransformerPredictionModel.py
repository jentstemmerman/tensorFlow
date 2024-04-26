import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                      np.arange(d_model)[np.newaxis, :],
                                      d_model)
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, units, dropout):
        super(EncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = self.point_wise_feed_forward_network(d_model, units)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training, mask):
        attn_output = self.mha(inputs, inputs, attention_mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def point_wise_feed_forward_network(self, d_model, units):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='relu'),  # (batch_size, seq_len, units)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, units, dropout):
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads, d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads, d_model)

        self.ffn = self.point_wise_feed_forward_network(d_model, units)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, enc_output, training, look_ahead_mask, padding_mask):
        attn1 = self.mha1(inputs, inputs, attention_mask=look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + inputs)

        attn2 = self.mha2(out1, enc_output, attention_mask=padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3

    def point_wise_feed_forward_network(self, d_model, units):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='relu'),  # (batch_size, seq_len, units)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

tf.config.run_functions_eagerly(True)  
tf.data.experimental.enable_debug_mode()    



# Dummy data for inference
dummy_input = tf.random.uniform((1, 10, 512))  # (batch_size, sequence_length, d_model)

# Instantiate our layers
pos_encoding = PositionalEncoding(1000, 512)
encoder_layer = EncoderLayer(512, 8, 2048, 0.1)
decoder_layer = DecoderLayer(512, 8, 2048, 0.1)

# Forward pass through the layers
x = pos_encoding(dummy_input)
enc_output = encoder_layer(x, training=False, mask=None)
dec_output = decoder_layer(x, enc_output, training=False, look_ahead_mask=None, padding_mask=None)

print("Output shape: ", dec_output.shape)



class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, units, dropout):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = PositionalEncoding(1000, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, units, dropout) 
                           for _ in range(num_layers)]
        self.dec_layers = [DecoderLayer(d_model, num_heads, units, dropout) 
                           for _ in range(num_layers)]

        self.final_layer = tf.keras.layers.Dense(units)

    def call(self, inputs, training):
        seq_len = tf.shape(inputs)[1]
        x = self.pos_encoding(inputs)

        for i in range(self.num_layers):
            enc_output = self.enc_layers[i](x, mask=None)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, look_ahead_mask=None, padding_mask=None)

        final_output = self.final_layer(x)

        return final_output

# Instantiate and compile model
model = Transformer(num_layers=2, d_model=512, num_heads=8, units=512, dropout=0.1)
model.compile(optimizer='adam', loss='mse')

dummy_output = model(dummy_input)


# Dummy data for training
x_train = tf.random.uniform((64, 10, 512))  # (batch_size, sequence_length, d_model)
y_train = tf.random.uniform((64, 10, 512))  # (batch_size, sequence_length, d_model)

# Train model
model.fit(x_train, y_train, epochs=1)

# Save model
model.save('my_model')

# Load model
loaded_model = tf.keras.models.load_model('my_model')

predict1 = model.predict(x_train)
predict2 = loaded_model.predict(x_train)

# Check that the loaded model gives the same output as the original model
same = np.allclose(predict1, predict2, atol=1e-5)

assert same