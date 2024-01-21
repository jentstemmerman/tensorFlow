import tensorflow as tf
import numpy as np
from tensorflow import keras

class GruPrintfLogModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units, return_state, numberOfInputs):
    super().__init__(self)
    self.return_state = return_state
   # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru1      = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=return_state, dropout=0.2, input_shape=(16, vocab_size))
    #self.gru2      = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=False, dropout=0.2)
    self.dense     = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    
    if states is None:
      initial_states = self.gru1.get_initial_state(x)

    if (self.return_state):
      x, states = self.gru1(x, initial_state=initial_states, training=training)
    else:
      x = self.gru1(x, initial_state=initial_states, training=training)
    
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x
    
class SimpleRnnPrintfLogModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units, numberOfInputs):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.simple_rnn1 = tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, return_state=True)
    self.simple_rnn2 = tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, return_state=True)
    self.dense     = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    
    if states is None:
      states1 = self.simple_rnn1.get_initial_state(x)

    x, states = self.simple_rnn1(x, initial_state=states1, training=training)
    x, states = self.simple_rnn2(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

class CustomSplitLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.factor = 0



    def get_config(self):
        return {"factor": self.factor}
     
    
    def call(self, inputs, training=False):
        number_of_splits = inputs.shape[inputs.ndim - 1]
        
        if number_of_splits > 2:
            if number_of_splits > 1:
                split_tensors = tf.split(inputs, num_or_size_splits=number_of_splits, axis=-1)
                splitted_inputs = split_tensors[0]  # Remove this for multiple inputs
            else:
                splitted_inputs = inputs

            splitted_inputs = tf.squeeze(splitted_inputs, axis=-1)
        else:
            splitted_inputs = tf.squeeze(inputs, axis=-1)

        return splitted_inputs

class LSTMPrintfLogModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units, numberOfInputs):
        super().__init__(self)

        self.customSplitLayer = CustomSplitLayer()

        if (embedding_dim != -1):
          self.use_embedding_layer = True
          inputDimensionLstm = embedding_dim
          self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        else:
          self.use_embedding_layer = False
          inputDimensionLstm = vocab_size
        
        self.lstmLayer = tf.keras.layers.LSTM(rnn_units, return_sequences=True, input_shape=(None, inputDimensionLstm))

        self.dense = tf.keras.layers.Dense(vocab_size)

    
    def call(self, inputs, states=None, return_state=False, training=False):
        #if (training):
        input_shape = tf.shape(inputs)
        
        if (input_shape is None ) :
          print("input_shape is None ")
          exit()
        
        indexable_input_shape = tf.get_static_value(input_shape)
        
        if (indexable_input_shape is None ) :
          print("indexable_input_shape is None. input_shape: ", input_shape, ".  inputs:  ", inputs)
          print("training: ", training)
          
        inputs = self.customSplitLayer(inputs, training=training)

        if (inputs is None):
          print("inputs is none")
          exit()

        input_shape2 = tf.shape(inputs)
        indexable_input_shape2 = tf.get_static_value(input_shape2)

        if (indexable_input_shape2 is None ) :
          print("indexable_input_shape2 is None. inputs: ", inputs, " input_shape2: ", input_shape2)
          
        #if (indexable_input_shape2[1] != 16) or (inputs.ndim != 2):
        #  print("inputs wrong shape ", type(inputs) , "   ", inputs.shape, " training:", training, " inputs.ndim = ", inputs.ndim)
          
     
        if (self.use_embedding_layer):
          inputs = self.embedding(inputs, training=training)
        #else:
        # inputs = tf.expand_dims(inputs, axis=2)
          
        # LSTM layer
        lstm_output = self.lstmLayer(inputs, training=training)

        # Fully connected layer to produce logits
        logits = self.dense(lstm_output, training=training)

        return logits

    
    
    
class OneStep(tf.keras.Model):
 
  def __init__(self, model, Tokens_from_idList, ids_from_tokensList, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.Tokens_from_idList = Tokens_from_idList
    ids_from_tokens = ids_from_tokensList[0]

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = ids_from_tokens(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_tokens.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  def GetProbabilityFromLogits(self, logits, id):
    probability = logits[id].numpy()
    return probability

  def GetProbabilitiesFromLogits(self, logits, id):

    probabilities = tf.gather(logits, id, axis=1, batch_dims=1)
    #indices = tf.argmax(logits, axis=1)

    # Gather values from params using indices
    #probabilities = tf.gather(params, indices)
    return probabilities
  
  #@tf.function
  def generate_steps_optimized(self, inputTensor, nexttokenIds, states=None):
    with tf.device('/GPU:0'):
      # Run the model.
      predicted_logits_list = self.model(inputs=inputTensor)
    
      predictions            = []
      probabilityOfPredictions = None
      probabilityOfActuals  = None

      for index, Tokens_from_ids in enumerate(self.Tokens_from_idList):
          predicted_logits = predicted_logits_list[index]
          predicted_logits = predicted_logits[:, -1, :]
          predicted_logits = predicted_logits / self.temperature

          softmaxLogits = tf.nn.softmax(predicted_logits, axis=1)
          predicted_id = tf.argmax(predicted_logits, axis=1)


          if (Tokens_from_ids is not None):
              prediction = (self.Tokens_from_idList[index])(predicted_id).numpy()

              for i, pred in enumerate(prediction):
                 prediction[i] = pred.decode('utf-8') 

          else:
              prediction = predicted_id.numpy()
          
          predictions.append(prediction)
          
          probabilityOfPrediction = self.GetProbabilitiesFromLogits(softmaxLogits, predicted_id).numpy()
          if (probabilityOfPredictions is None):
              probabilityOfPredictions = probabilityOfPrediction
          else:
              probabilityOfPredictions  = np.stack((probabilityOfPredictions, probabilityOfPrediction))


          probabilityOfActual = self.GetProbabilitiesFromLogits(softmaxLogits, tf.cast(nexttokenIds[:,index], dtype=tf.int32)).numpy()

          if (probabilityOfActuals is None):
              probabilityOfActuals = probabilityOfActual
          else:
              probabilityOfActuals  = np.stack((probabilityOfActuals, probabilityOfActual))

      return predictions, probabilityOfPredictions, probabilityOfActuals
      
