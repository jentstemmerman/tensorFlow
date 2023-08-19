import tensorflow as tf
from tensorflow import keras

class PrintfLogModel(tf.keras.Model):


  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x
    

    
    
class OneStep(tf.keras.Model):
  def GetTokenProbability(self, inputTensor, predictedToken):

    tokenId = self.ids_from_tokens(predictedToken).numpy()
    inputTensorExpanded = tf.expand_dims(inputTensor, 0)

    #print("inputTensorExpanded: ", inputTensorExpanded.shape)
    #tf.print(inputTensorExpanded)

    states = None
    predicted_logits = self.model(inputs=inputTensorExpanded, return_state=False) # -> shape=(2, 5, 12535)
    predicted_logits = predicted_logits[0, -1, :] # ik wil alleen de voorspelling na de laatste token

    probabilitiesSum = tf.math.reduce_sum(predicted_logits, axis=0, keepdims=False, name=None).numpy()
    predicted_logits = predicted_logits / probabilitiesSum
    probabilities = tf.sigmoid(predicted_logits)
    probability = probabilities[tokenId].numpy()
#    probability = probability / probabilitiesSum

    return probability
  
  def __init__(self, model, Tokens_from_ids, ids_from_tokens, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.Tokens_from_ids = Tokens_from_ids
    self.ids_from_tokens = ids_from_tokens

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_tokens(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_tokens.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

 # @tf.function #  JENTS: this is a decorator. tf.function gets generate_one_step as an argument ( including parameters ) and does optimisation 
  def generate_one_step(self, inputTensor, states=None):

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    inputTensorExpanded = tf.expand_dims(inputTensor, 0)
    predicted_logits = self.model(inputs=inputTensorExpanded, states=states, return_state=False)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    #predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_id = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_id = tf.squeeze(predicted_id, axis=-1).numpy()[0]

    # Convert from token ids to characters
    predicted_token = (self.Tokens_from_ids(predicted_id)).numpy()
   
    probabilitiesSum = tf.math.reduce_sum(predicted_logits, axis=1, keepdims=False, name=None).numpy()[0]
    predicted_logits = predicted_logits / probabilitiesSum
    probabilities = tf.sigmoid(predicted_logits)
    

    probability = probabilities[0, predicted_id].numpy() # ik wil alleen de voorspelling na de laatste token

    # Return the characters and model state.
    return predicted_token, probability