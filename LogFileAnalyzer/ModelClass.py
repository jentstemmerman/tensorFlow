import tensorflow as tf
import numpy as np
from Mask import Mask
from SimilarityMaxShifter import SimilarityMaxShifter
        
class OneStep(tf.keras.Model):
  
  def __init__(self, model, DataTypeForTensors, seq_length, itemList, token_from_idList, id_from_tokenList, temperature=1.0, hidden_stateWeightTemperature=50.0, signatureLengthList=None, similarityMaxShifter=None):
    super().__init__()
    self.temperature                   = temperature
    self.model                         = model
    self.token_from_idList             = token_from_idList
    self.id_from_tokenList             = id_from_tokenList
    self.hidden_stateWeightTemperature = hidden_stateWeightTemperature
    self.seq_length                    = seq_length  
    self.DataTypeForTensors            = DataTypeForTensors
    self.itemList                      = itemList  # This is not used in this class, but it is used in the Signature class.
    self.signatureLengthList           = signatureLengthList
    self.similarityMaxShifter          = similarityMaxShifter

  def GetProbabilityFromLogits(self, softmaxLogits, id):
    
    probability = softmaxLogits[id].numpy()
    return probability
  
  
  def GetProbabilitiesFromLogits(self, softmaxLogits, id):

    probabilities = tf.gather(softmaxLogits, id, axis=1, batch_dims=1)

    return probabilities
  
  
  def GetCrossEntropyromLogits(self, softmaxLogits, id):
    probabilities = tf.keras.losses.sparse_categorical_crossentropy(id, softmaxLogits, from_logits=True)
    
    return probabilities
  
  def CalculateHiddenVectorSize(self):
 
      tokenListList = []
        
      inputTensor = self.generate_inputTensor_from_tokenListList(tokenListList)

      _, pooledLogitsData = self.generate_hidded_states(inputTensor)       
      hiddenVectorSize = pooledLogitsData['hidden_state_prediction'].shape[1]
      
      return hiddenVectorSize
  
  
  def generate_inputTensor_from_tokenListList(self, tokenListList):
    assert isinstance(tokenListList, list), "tokenListList should be a list of lists"
    assert self.id_from_tokenList is not None, "id_from_tokenList should not be None"

    numberOfLines = len(tokenListList)

    tokenIdList = []

    requiredNumberOfMaskTokens = self.seq_length - numberOfLines

    mask_token_filler_list = [Mask.Universal_Mask_Token] * requiredNumberOfMaskTokens

    for columnIndex, id_from_token in enumerate(self.id_from_tokenList):
      columnList = [column[columnIndex] for column in tokenListList]
      columnList.extend(mask_token_filler_list)  # add the mask tokens to the end of the column

      if (id_from_token is None):
        idColumn = tf.convert_to_tensor(columnList, dtype=self.DataTypeForTensors)
      else:
        idColumn = id_from_token(columnList)

      tokenIdList.append(idColumn)
      # After the loop, stack all idColumns along the last dimension to form the input tensor
      tokenIdList = tf.stack(tokenIdList, axis=-1)

    # Add a batch dimension of 1 to tokenIdList
    tokenIdList = tf.expand_dims(tokenIdList, axis=0)

    tokenIdList = tf.cast(tokenIdList, self.DataTypeForTensors)

    return tokenIdList
        
    
  #@tf.function
  def generate_simularity_lists(self, inputTensor, signature_list):
    with tf.device('/GPU:0'):
      
      try:
        _, hidden_state, attention_weights, _ = self.model(inputs=(inputTensor), training=False)  
      except Exception as e:
        print("Error in generate_hidded_states: ", e)
        print("inputTensor shape: ", inputTensor.shape)
        print("seq_length: ", self.seq_length)
        print("inputTensor: ", inputTensor)  
        return None, None
      
      try:
        
        # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
        # The rows (axis=2) are the queries.
        # - Axis 2: query positions
        # - Axis 3: key positions
        # (matrix: (row,column) = (query, key))
        # To compute attention received by each token (i.e., how much attention each token gets from all others), we sum over the row (axis=2)
        # For each token, we can store the attention it received from all other tokens by storing the rows of the attention weights.
        # 
        # a column shows how important THIS token is to enrich the meaning of other tokens. These score are not reflected in the hidden state.
        # a row shows how important OTHER tokens enrich the meaning of this token.  These scores are reflected in the hidden state and the resulting logits.
        
        
        # Operation: Mean across all attention heads (axis=1).
        attention_mean_heads = tf.reduce_mean(attention_weights, axis=1)
        attention_received_first_query = attention_mean_heads[:, :, 0]  # this is the attention received by the first token in the sequence. How important was this token for the other tokens.
          
        # Since this is causal attention, the first token is attented by all tokens, the second token by all but the first one, and so on... 
        # We need to correct for this bias. We can do this by dividing the attention scores by the number of tokens that attended to this token. 
        contrib_counts = tf.range(hidden_state.shape[1], 0, -1, dtype=self.DataTypeForTensors)    
          
        attention_received_first_query_normalized = attention_received_first_query / contrib_counts  
        attention_received_first_query_normalized_centered = attention_received_first_query_normalized - tf.reduce_max(attention_received_first_query_normalized, axis=-1, keepdims=True)

        attention_received_first_query_normalized_centered_with_temperature = attention_received_first_query_normalized_centered * self.hidden_stateWeightTemperature 
        
        attention_received_first_query_normalized = tf.nn.softmax(attention_received_first_query_normalized_centered_with_temperature, axis=-1)  # [batch, seq_len]  
          
        # Vectorized similarity computation across all signatures on GPU
        # Assumptions:
        # - Each Signature has: numberOfLines (int) and a precomputed signature vector at signature.signature (shape [D])
        # - Cosine similarity is computed after L2-normalizing both pooled window vectors and signature vectors
        # - Pooling mimics the per-signature path: L2-normalize token vectors, multiply by attention weights,
        #   take an average over the first k tokens (k = numberOfLines)

        # Short-circuit if no signatures
        if len(signature_list) == 0:
          similarity_list_tensor = tf.zeros([0, tf.shape(hidden_state)[0]], dtype=attention_received_first_query_normalized.dtype)
        else:
          # Prepare shapes
          T = tf.shape(hidden_state)[1]   # sequence length

          # Collect signature lengths and cap to T to avoid over-slicing
          lengths_py = [int(sig.numberOfLines) for sig in signature_list]
          lengths = tf.convert_to_tensor(lengths_py, dtype=tf.int32)
          lengths = tf.minimum(lengths, T)

          # Ensure all signatures have an embedded vector; if any are missing, compute them now
          # Note: This happens on CPU once; the heavy math below remains on GPU.
          for sig in signature_list:
            if getattr(sig, 'signature', None) is None:
              # Lazily compute and cache signature vector
              sig.CalculateSignature(self)

          # Stack signature vectors into [S, D]
          signatures_py = [tf.convert_to_tensor(sig.signature) for sig in signature_list]
          signatures = tf.stack(signatures_py, axis=0)

          # Normalize signatures across feature dim
          signatures = tf.nn.l2_normalize(signatures, axis=-1)

          # Build a mask [S, T] where mask[i, t] = 1 if t < lengths[i] else 0
          mask_ST = tf.sequence_mask(lengths, maxlen=T, dtype=hidden_state.dtype)  # [S, T]

          # Normalize token vectors after applying attention weights (to mimic original semantics)
          # Shapes:
          #   hidden_state: [B, T, D]
          #   attention_received_first_query_normalized: [B, T]
          A = tf.expand_dims(attention_received_first_query_normalized, -1)  # [B, T, 1]
          H_att = hidden_state * A                                           # [B, T, D]
          Hn = tf.nn.l2_normalize(H_att, axis=-1)                            # [B, T, D]

          # Broadcast across signatures: add a leading S dimension
          Hn = tf.expand_dims(Hn, axis=0)  # [1, B, T, D]
          M  = tf.expand_dims(tf.expand_dims(mask_ST, axis=1), axis=-1)  # [S, 1, T, 1]

          # Masked token vectors (average without additional attention weighting to match original)
          weighted = Hn * M

          # Average over the first k tokens (mask ensures only first k contribute)
          numerator = tf.reduce_sum(weighted, axis=2)  # [S, B, D]
          denom = tf.cast(tf.reshape(tf.maximum(lengths, 1), [-1, 1, 1]), dtype=hidden_state.dtype)  # [S,1,1]
          pooled = numerator / denom  # [S, B, D]

          # Normalize pooled vectors per signature-batch
          pooled = tf.nn.l2_normalize(pooled, axis=-1)  # [S, B, D]

          # Cosine similarity (negated to match tf.keras.losses.cosine_similarity sign)
          # sim[s, b] = - dot(pooled[s, b, :], signatures[s, :])
          similarity_list_tensor = -tf.einsum('sbd,sd->sb', pooled, signatures)
          
                  
      except Exception as e:
        print("Error in generate_hidded_states: ", e)
        print("inputTensor shape: ", inputTensor.shape)
        print("seq_length: ", self.seq_length)
        print("inputTensor: ", inputTensor)
      
    return similarity_list_tensor
  


      
    
  #@tf.function
  def generate_hidded_states(self, inputTensor):
    with tf.device('/GPU:0'):
      try:
        
        try:
          predicted_logits_list, hidden_state, attention_weights, attentionScoresList = self.model(inputs=(inputTensor), training=False)  
        except Exception as e:
          print("Error in generate_hidded_states: ", e)
          print("inputTensor shape: ", inputTensor.shape)
          print("seq_length: ", self.seq_length)
          print("inputTensor: ", inputTensor)  
          return None, None
        
        lastItemInWindow = self.seq_length - 1
        
        hiddenStateOfFirstLineInWindow = tf.gather(hidden_state, indices = 0                   , axis=1)
        hiddenStateOfPrediction        = tf.gather(hidden_state, indices = lastItemInWindow    , axis=1)
        hiddenStateOfLastLineInWindow  = tf.gather(hidden_state, indices = lastItemInWindow - 1, axis=1)
        
        # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
        # The rows (axis=2) are the queries.
        # - Axis 2: query positions
        # - Axis 3: key positions
        # (matrix: (row,column) = (query, key))
        # To compute attention received by each token (i.e., how much attention each token gets from all others), we sum over the row (axis=2)
        # For each token, we can store the attention it received from all other tokens by storing the rows of the attention weights.
        # 
        # a column shows how important THIS token is to enrich the meaning of other tokens. These score are not reflected in the hidden state.
        # a row shows how important OTHER tokens enrich the meaning of this token.  These scores are reflected in the hidden state and the resulting logits.
        
        
        # Operation: Mean across all attention heads (axis=1).
        attention_mean_heads = tf.reduce_mean(attention_weights, axis=1)
        attention_received_first_query = attention_mean_heads[:, :, 0]  # this is the attention received by the first token in the sequence. How important was this token for the other tokens.
          
        # Since this is causal attention, the first token is attented by all tokens, the second token by all but the first one, and so on... 
        # We need to correct for this bias. We can do this by dividing the attention scores by the number of tokens that attended to this token. 
        contrib_counts = tf.range(hidden_state.shape[1], 0, -1)    
        contrib_counts = tf.cast(contrib_counts, hidden_state.dtype)  
        
        attention_received_first_query_normalized = attention_received_first_query / contrib_counts  
        attention_received_first_query_normalized_centered = attention_received_first_query_normalized - tf.reduce_max(attention_received_first_query_normalized, axis=-1, keepdims=True)

        attention_received_first_query_normalized_centered_with_temperature = attention_received_first_query_normalized_centered * self.hidden_stateWeightTemperature 
        
        attention_received_first_query_normalized = tf.nn.softmax(attention_received_first_query_normalized_centered_with_temperature, axis=-1)  # [batch, seq_len]  
          
        # flatten this hidden state here for performance reasons. We will have first the slices in the sequence: for instance [200, 64, 12] will be [200*64, 12]
        hidden_state_reshaped_for_file = tf.reshape(hidden_state, [hidden_state.shape[0] * hidden_state.shape[1], hidden_state.shape[2]])
        
        pooledLogitsData = {
                                'hidden_state_prediction': hiddenStateOfPrediction,
                                'hidden_state_first':      hiddenStateOfFirstLineInWindow,
                                'hidden_state_last':       hiddenStateOfLastLineInWindow,
                                'hidden_state_all_flattened': hidden_state_reshaped_for_file,
                                'hidden_state_all':        hidden_state,
                                'attention_received':      attention_received_first_query_normalized,
                           }
        
        
      except Exception as e:
        print("Error in generate_hidded_states: ", e)
        print("inputTensor shape: ", inputTensor.shape)
        print("seq_length: ", self.seq_length)
        print("inputTensor: ", inputTensor)
      
      return predicted_logits_list, pooledLogitsData
        
    

  #@tf.function
  def generate_steps_optimized(self, inputTensor, nexttokenIds, window_size, DataTypeForTensors):
    with tf.device('/GPU:0'):

      predicted_logits_list, pooledLogitsData = self.generate_hidded_states(inputTensor, window_size, DataTypeForTensors)
        
      numberOfFeatures       = len(self.token_from_idList)
      predictions            = []
      probabilityOfPredictions = None
      probabilityOfActuals     = None
      crossEntropyOfActuals    = None
      
      for index, Tokens_from_ids in enumerate(self.token_from_idList):
          predicted_logits = predicted_logits_list[index]
          predicted_logits = predicted_logits[:, -1, :]    # take the last logits, this is the one the model predicts
          predicted_logits = predicted_logits / self.temperature

          if (Tokens_from_ids is not None):
              predicted_id = tf.argmax(predicted_logits, axis=1)
              prediction = (self.token_from_idList[index])(predicted_id).numpy()

              for i, pred in enumerate(prediction):
                prediction[i] = pred.decode('utf-8') 

              softmaxLogits = tf.nn.softmax(predicted_logits, axis=1)
              probabilityOfPrediction = self.GetProbabilitiesFromLogits(softmaxLogits, predicted_id).numpy()

          else:
              predicted_id = predicted_logits
              predicted_id = tf.reshape(predicted_id, [-1])
              prediction = predicted_id.numpy()
              tensor_shape = (inputTensor.shape[0],)
              probabilityOfPrediction = tf.zeros(shape=tensor_shape) # dit heeft geen zin voor non-cathegorical, maar er moet iets ingevuld zijn

          
          predictions.append(prediction)
        
          
          if (probabilityOfPredictions is None):
              probabilityOfPredictions  = probabilityOfPrediction   
          else:
              probabilityOfPredictions   = np.append(probabilityOfPredictions, probabilityOfPrediction)

          probabilityOfActual = self.GetProbabilitiesFromLogits(softmaxLogits, tf.cast(nexttokenIds[:,index], dtype=tf.int32)).numpy()

          if (probabilityOfActuals is None):
              probabilityOfActuals = probabilityOfActual
          else:
              probabilityOfActuals   = np.append(probabilityOfActuals, probabilityOfActual)


          crossEntropyOfActual = self.GetCrossEntropyromLogits(predicted_logits, tf.cast(nexttokenIds[:,index], dtype=tf.int32)).numpy()

          if (crossEntropyOfActuals is None):
              crossEntropyOfActuals = crossEntropyOfActual
          else:
              crossEntropyOfActuals   = np.append(crossEntropyOfActuals, crossEntropyOfActual)
      
      probabilityOfPredictions = np.reshape(probabilityOfPredictions, [numberOfFeatures, int(len(probabilityOfPredictions) / numberOfFeatures)])
      probabilityOfActuals     = np.reshape(probabilityOfActuals,     [numberOfFeatures, int(len(probabilityOfActuals)     / numberOfFeatures)])
      crossEntropyOfActuals    = np.reshape(crossEntropyOfActuals,    [numberOfFeatures, int(len(crossEntropyOfActuals)     / numberOfFeatures)])
    

      return predictions, probabilityOfPredictions, probabilityOfActuals, crossEntropyOfActuals, pooledLogitsData
      
