import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import MultiDimensionalModel


# JENTS: this looks like the classic positional encoding. 
@tf.function
def positional_encoding(length, depth): # lenght is the sequence lenght, depth the embedding dimension
  """
  Generates a matrix of position encodings for an input sequence.

  Args:
      length: An integer representing the length of the input sequence.
      depth: An integer representing the dimensionality of the encoding.

  Returns:
      A `tf.Tensor` of shape `(length, depth)` representing the position encoding matrix.
  """
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
  """
  This layer combines the input embedding with a positional encoding that helps the Transformer to understand
  the relative position of the tokens in a sequence. It takes an input sequence of tokens and converts it to
  a sequence of embedding vectors, then adds positional information to it.

  Attributes:
      vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens in the input sequence.
      d_model (int): The number of dimensions in the embedding vector.

  Methods:
      compute_mask(*args, **kwargs): This method computes the mask to be applied to the embeddings.
      call(x): This method performs the computation for the layer.

  """
  def __init__(self, vocab_size, d_model, seq_length):
    """
    Initializes the PositionalEmbedding layer.

    Args:
        vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens in the input sequence.
        d_model (int): The number of dimensions in the embedding vector.
    """
    super().__init__()
    self.seq_length = seq_length
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
    self.pos_encoding = positional_encoding(length=seq_length, depth=d_model)
    print(self.pos_encoding.shape)

  def compute_mask(self, *args, **kwargs):
    """
    Computes the mask to be applied to the embeddings.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Mask to be applied to the embeddings.
    """
    return self.embedding.compute_mask(*args, **kwargs)
  

  @tf.function
  def call(self, x):
    """
    Computes the output of the layer.

    Args:
        x (tf.Tensor): Input sequence of tokens.

    Returns:
        The output sequence of embedding vectors with added positional information.
    """
    
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    
    length = self.seq_length # tf.shape(x)[1]
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class BaseAttention(tf.keras.layers.Layer):
  """
  Base Attention layer class that contains a MultiHeadAttention, LayerNormalization and Add layer.

  Attributes:
  -----------
  kwargs: dict
      keyword arguments that will be passed to the MultiHeadAttention layer during initialization.

  Methods:
  --------
  call(inputs, mask=None, training=None):
      Performs a forward pass on the input and returns the output.

  """
  def __init__(self, **kwargs):
    """
    Initializes a new instance of the BaseAttention layer class.

    Parameters:
    -----------
    kwargs: dict
        keyword arguments that will be passed to the MultiHeadAttention layer during initialization.
    """
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class CausalSelfAttention(BaseAttention):
  """
  Call self attention on the input sequence, ensuring that each position in the
  output depends only on previous positions (i.e. a causal model).

  Args:
      x: Input sequence tensor of shape `(batch_size, seq_len, embed_dim)`.

  Returns:
      Output sequence tensor of the same shape as the input, after self-attention
      and residual connection with layer normalization applied.
  """
  def call(self, x, return_attention_scores=False):
    multiHeadAttentionOutput = self.mha(
        query=x,
        value=x,
        key=x,
        return_attention_scores=return_attention_scores,
        use_causal_mask=True)
    
    if return_attention_scores:
        attn_output, attn_scores = multiHeadAttentionOutput
    else:
        attn_output = multiHeadAttentionOutput
        attn_scores = None

    x = self.add([x, attn_output])
    x = self.layernorm(x)
    
    if return_attention_scores:
        return x, attn_scores
    else:
        return x

class FeedForward(tf.keras.layers.Layer):
  """
  Implements the feedforward sublayer of the transformer block.

  Parameters:
  -----------
  d_model: int
      The number of expected features in the input and output.
  dff: int
      The number of neurons in the first Dense layer.
  dropout_rate: float, optional (default=0.1)
      The dropout rate to use.

  Attributes:
  -----------
  seq: tf.keras.Sequential
      The sequential model that applies the two Dense layers and Dropout.
  add: tf.keras.layers.Add
      The addition layer that adds the residual connection.
  layer_norm: tf.keras.layers.LayerNormalization
      The normalization layer applied to the output.

  Methods:
  --------
  call(x):
      Computes the feedforward sublayer on the input tensor x and returns the output.

  """
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  @tf.function
  def call(self, x):
    """
    Passes the input tensor `x` through a feedforward network consisting of two
    dense layers with `dff` hidden units and a `relu` activation function.
    A `dropout_rate` is applied after the first dense layer to prevent overfitting.
    The output of the feedforward network is added to the original input `x` via the
    `Add()` layer. Finally, the output is normalized using the `LayerNormalization()` layer.

    Args:
        x (tf.Tensor): Input tensor with shape `(batch_size, seq_len, d_model)`.

    Returns:
        tf.Tensor: Output tensor with shape `(batch_size, seq_len, d_model)`.
    """
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x

class DecoderLayer(tf.keras.layers.Layer):
  """
  A single layer of the decoder in a transformer-based architecture.

  Args:
    d_model (int): The number of expected features in the input.
    num_heads (int): The number of attention heads.
    dff (int): The dimensionality of the feedforward network.
    dropout_rate (float): The dropout rate to be applied.

  Attributes:
    causal_self_attention: An instance of the `CausalSelfAttention` layer.
    cross_attention: An instance of the `CrossAttention` layer.
    ffn: An instance of the `FeedForward` layer.
    last_attn_scores: A tensor containing the last attention scores.

  """
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate
       )
    
    self.cross_attention = None

    self.FeedForwardLayer = FeedForward(d_model, dff)

  
  def call(self, x, training):
    """
    Forward pass of the `DecoderLayer`.

    Args:
      x (tf.Tensor): The input tensor of shape
      `(batch_size, target_seq_len, d_model)`.

    Returns:
      The output tensor of the `DecoderLayer` or a tuple with output and attention scores.
    """
    isInference = not training
    
    if isInference:
        x, attn_scores = self.causal_self_attention(x=x, return_attention_scores=True)
    else:
        x = self.causal_self_attention(x=x, return_attention_scores=False)
        attn_scores = None

    # Cache the last attention scores for plotting later
    self.last_attn_scores = attn_scores

    x = self.FeedForwardLayer(x)  # Shape `(batch_size, seq_len, d_model)`.
     
    if isInference:
        return x, attn_scores
    else:
        return x

class Decoder(tf.keras.layers.Layer):
  """A decoder model for sequence to sequence learning.

  This class implements a decoder layer for a transformer-based model used for sequence to sequence learning tasks. The decoder layer takes input embeddings, positional encodings, and attention masks as input, and returns the output of the decoder layer after applying a multi-head self-attention mechanism, followed by a cross-attention mechanism with the output from the encoder layers, and then applying a feed-forward neural network.

  Attributes:
    d_model (int): The number of output dimensions for each layer.
    num_layers (int): The number of layers in the decoder.
    pos_embedding (PositionalEmbedding): The positional embedding layer.
    dropout (Dropout): A dropout layer.
    dec_layers (list): A list of DecoderLayer objects.
    last_attn_scores (ndarray): The attention scores from the last decoder layer.

  Methods:
    call(x, context): Implements the forward pass for the decoder layer.
      Args:
        x (ndarray): A tensor of shape (batch_size, target_seq_len), representing the input token IDs.
        context (ndarray): A tensor of shape (batch_size, input_seq_len, d_model), representing the output from the encoder layers.
      Returns:
        ndarray: A tensor of shape (batch_size, target_seq_len, d_model), representing the output from the decoder layers.
  """
  def __init__(self, *, num_layers, num_heads, dff, featureList, datatype, seq_length,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.num_layers = num_layers

    output_dimension = 0

    embeddingDimensionList = self.CalculateEmbeddingDimensions(featureList)

    self.input_layerList  = []

    for i, feature in enumerate(featureList):

        output_dimension += embeddingDimensionList[i]

        if (feature["embedded"]):
            input_layer = PositionalEmbedding(vocab_size=feature["vocab_size"], d_model=embeddingDimensionList[i], seq_length=seq_length)
        else:
            input_layer = MultiDimensionalModel.CustomIdentityLayer(datatype=datatype)  # no positional encoding for non-embedded features, might be wrong to do this
        
       
        self.input_layerList.append(input_layer)    

    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [DecoderLayer(d_model=output_dimension, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate) for _ in range(num_layers)]

    self.output_layerList  = []

    for i, feature in enumerate(featureList):

        layer_name = feature["layer_name"]

        if (feature["embedded"]):
            dense_output_layer = tf.keras.layers.Dense(units=feature["vocab_size"], name=layer_name, activation=None)
        else:
            dense_output_layer = MultiDimensionalModel.CustomDenseLayer(seq_length, 1) 

        self.output_layerList.append(dense_output_layer)  


    self.last_attn_scores = None


  def CalculateEmbeddingDimensions(self, featureList):
    embeddingDimension = []
    total_output_dimensions = 0

    for i, feature in enumerate(featureList):
        if (feature["embedded"]):
            import math
            output_dimension = 128 # math.ceil(feature["vocab_size"]**0.25) * 2
        else:
            output_dimension = 1

        # the positional encoder heeft een getal deelbaar door 2 nodig
        if (output_dimension % 2 == 1):
            output_dimension += 1
        
        embeddingDimension.append(output_dimension)
        total_output_dimensions += output_dimension

    return embeddingDimension


  @tf.function
  def call(self, inputs, training=False):
    inputList = []
    
    for i, input_layer in enumerate(self.input_layerList):
        input_to_layer = inputs[:, :, i]
        output = input_layer(input_to_layer, training = training)
        inputList.append(output)

    concatenated_inputs = tf.keras.layers.concatenate(inputList, axis=-1)
    concatenated_inputs = self.dropout(concatenated_inputs)
    
    hidden_state = concatenated_inputs
    
    if training:
      for i in range(self.num_layers):
          hidden_state = self.dec_layers[i](hidden_state, training=training)
          
      attention_weights = None  
    else:
      for i in range(self.num_layers):
          hidden_state, attention_weights = self.dec_layers[i](hidden_state, training=training)
   
        
    output_List = []

    for i, output_layer in enumerate(self.output_layerList):
        layer_output = output_layer(hidden_state, training=training)
        output_List.append(layer_output)

  
    #self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return output_List, hidden_state, attention_weights
  
  
class DecoderOnlyTransformerLayer(tf.keras.layers.Layer):

  def __init__(self, *, num_layers, num_heads, dff, featureList, seq_length, datatype,
               dropout_rate=0.1):
    super().__init__()
    
    self.encoder = None
    
    self.decoder = Decoder(num_layers=num_layers, featureList=featureList,
                           num_heads=num_heads, dff=dff, datatype=datatype,
                           seq_length=seq_length,
                           dropout_rate=dropout_rate)



  @tf.function
  def call(self, inputs):
    """
    Forward pass of the transformer model.

    Args:
      inputs (tuple): A tuple of two tensors. The first tensor is the context input tensor of shape (batch_size, context_len).
                      The second tensor is the target input tensor of shape (batch_size, target_len).

    Returns:
      logits (tf.Tensor): Output tensor of the final dense layer. Shape (batch_size, target_len, target_vocab_size).
      hidden_states (tf.Tensor): Hidden states from the decoder. Shape (batch_size, target_len, d_model).
    """

    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    x  = inputs
  
    #x = tf.squeeze(x, axis=-1)
        
    logits, hidden_states, attention_weights = self.decoder(x)  

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    return logits, hidden_states, attention_weights

class Transformer(tf.keras.Model):
  """
  A Transformer_DecoderOnly model that consists of a decoder and a final dense layer.

  Args:
    num_layers (int): Number of layers in decoder.
    d_model (int): Hidden size of the model.
    num_heads (int): Number of attention heads used in the model.
    dff (int): Size of the feedforward layer in decoder.
    input_vocab_size (int): Size of the vocabulary of the input.
    target_vocab_size (int): Size of the vocabulary of the target.
    dropout_rate (float): Dropout rate applied to the output of each sub-layer.

  Attributes:
    encoder (Encoder): An instance of the Encoder class.
    decoder (Decoder): An instance of the Decoder class.
    final_layer (Dense): A Dense layer that converts the final transformer output to output token probabilities.

  Methods:
    call(inputs): Forward pass of the transformer model.

  Returns:
    logits (tf.Tensor): Output tensor of the final dense layer. Shape (batch_size, target_len, target_vocab_size).
  """
  def __init__(self, *, num_layers, num_heads, dff,
               input_vocab_size, featureList, datatype, seq_length, dropout_rate=0.1):
    super().__init__()

    self.decoderOnlyTransformerLayer = DecoderOnlyTransformerLayer(num_layers=num_layers, featureList=featureList, datatype=datatype, num_heads=num_heads, dff=dff,
               seq_length=seq_length, dropout_rate=0.1)

  
  @tf.function
  def call(self, inputs, training=False):
    logits, hidden_state, attention_weights = self.decoderOnlyTransformerLayer(inputs)
    
    if (training):
      returnValue = logits
    else:
      returnValue = logits, hidden_state, attention_weights
      
    # Return the final output and the attention weights.
    return returnValue
  


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """
  Custom learning rate schedule that implements the learning rate function
  described in the original Transformer paper. The learning rate is increased
  linearly for the first `warmup_steps` training steps, and then decreased
  proportionally to the inverse square root of the step number.

  Args:
    d_model (int): the dimensionality of the model.
    warmup_steps (int): the number of steps taken to increase the learning rate
      linearly. Default is 4000.

  Attributes:
    d_model (float): the dimensionality of the model as a float.
    warmup_steps (int): the number of steps taken to increase the learning rate
      linearly.

  Methods:
    __call__(step): returns the learning rate at the given step.

  Returns:
    The learning rate at the given step.
  """
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    """
    Returns the learning rate at the given step.

    Args:
      step (int): the current training step.

    Returns:
      The learning rate at the given step as a float32 tensor.
    """
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
  def get_config(self):
      config =  {
            "d_model": self.d_model.numpy(),
            "warmup_steps": self.warmup_steps,}

      return config  
  
  @classmethod
  def from_config(cls, config):
      return cls(**config)
  

#tf.config.run_functions_eagerly(True)  
#tf.data.experimental.enable_debug_mode()    
if __name__ == "__main__":
  num_layers = 4
  d_model = 128
  dff = 512
  num_heads = 8
  dropout_rate = 0.1

  # Parameters
  num_samples = 500
  max_seq_length = 50
  input_vocab_size = 1000
  target_vocab_size = 1100
  epochs = 10

  transformer = Transformer(
      num_layers=num_layers,
      num_heads=num_heads,
      dff=dff,
      input_vocab_size=input_vocab_size,       #tokenizers.pt.get_vocab_size().numpy(),
      target_vocab_size=target_vocab_size,     #tokenizers.en.get_vocab_size().numpy(),
      featureList=None,
      dropout_rate=dropout_rate)


  # Generate random sequences for input and target
  input_sequences = np.random.randint(0, input_vocab_size, (num_samples, max_seq_length))
  target_sequences = np.random.randint(0, target_vocab_size, (num_samples, max_seq_length))

  output_transformer = transformer((input_sequences, target_sequences))


  trainingData = ((input_sequences, target_sequences), (target_sequences))
  # Convert to TensorFlow datasets
  dataset = tf.data.Dataset.from_tensor_slices(trainingData)

  # Print some samples to verify
  for input_example, target_example in dataset.take(1):
      print("Input sequence: ", input_example)
      print("Target sequence:", target_example)

  # Prepare the dataset for training
  # (e.g., shuffling, batching)
  BATCH_SIZE = 64
  BUFFER_SIZE = 20000

  datasetForTraining = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
  transformer.compile(optimizer='adam', loss=loss)

  print(transformer.summary())

  history = transformer.fit(datasetForTraining, epochs=epochs)
  print("Done")



















