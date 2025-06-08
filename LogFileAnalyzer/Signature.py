import Logfile
import tensorflow as tf
import ModelClass
import Dataset

class Signature:
    
    def __init__(self, 
                 logSnipped=None, 
                 logFile=None, 
                 linesIndexList=None, 
                 startLine=None,
                 endLine=None,
                 itemList=None, 
                 model = None,
                 id_from_tokenList = None,
                 temperature=None,
                 seq_length=None,
                 DataTypeForTensors=None):
        
        assert logSnipped is not None or logFile is not None, "Either logSnipped or logFile must be provided."
        assert linesIndexList is not None or (startLine is not None and endLine is not None), "Either linesIndexList or startLine and endLine must be provided."
        assert itemList is not None, "itemList must be provided."
        assert id_from_tokenList is not None, "id_from_tokenList must be provided."
        assert temperature is not None, "temperature must be provided."
        assert seq_length is not None, "seq_length must be provided."
        assert DataTypeForTensors is not None, "DataTypeForTensors must be provided."
        assert model is not None, "model must be provided."
        
        self.id_from_tokenList = id_from_tokenList
        self.model = model
        self.seq_length = seq_length
        self.temperature = temperature
        self.startLine = startLine
        self.endLine = endLine
        self.logFile = logFile
        self.linesIndexList = linesIndexList
        self.DataTypeForTensors = DataTypeForTensors
        self.logSnipped = logSnipped
        self.itemList = itemList
        
        if logSnipped is not None:
            self.logSnipped = logSnipped
        elif logFile is not None:
           
            if linesIndexList is not None:
                self.linesIndexList = linesIndexList                
            else:
                self.linesIndexList = []
                if startLine is not None and endLine is not None:
                    for i in range(startLine, endLine + 1):
                        self.linesIndexList.append(i)
        else:
            self.logSnipped = None
        
    def compare_with(self, other_signature):
        assert other_signature is not None, "other_signature must not be None."
        assert isinstance(other_signature, Signature), "other_signature must be an instance of Signature."
        
        this_signature = self.CalculateSignature()
        other_signature = other_signature.CalculateSignature()
        
        if this_signature is None or other_signature is None:
            return None
        assert isinstance(this_signature, tf.Tensor), "this_signature must be a Tensor."
        assert isinstance(other_signature, tf.Tensor), "other_signature must be a Tensor."
        assert this_signature.shape == other_signature.shape, "Signatures must have the same shape."
        
        # Calculate the cosine similarity between the two signatures
        cosine_similarity = tf.reduce_sum(tf.multiply(this_signature, other_signature), axis=-1) / (
            tf.norm(this_signature, axis=-1) * tf.norm(other_signature, axis=-1)
        )
        # Convert the cosine similarity to a scalar value
        cosine_similarity = tf.reduce_mean(cosine_similarity)
        # Convert the cosine similarity to a float value
        cosine_similarity = cosine_similarity.numpy().item()
        # Return the cosine similarity
        return cosine_similarity
        

    def CalculateSignature(self):
        
        if self.logSnipped is None:
            logFile = Logfile.Logfile(fileName=self.logFile, itemList=self.itemList, DataTypeForTensors=self.DataTypeForTensors)
            tokenListList = logFile.GetTokensFromLineIndexList(self.linesIndexList)
        else: 
            tokenListList = logFile.GetTokensFromLines(self.logSnipped)      
            
        Signature = self.CalculateSignatureFromTokenListList(tokenListList)
        
        return Signature
        
                   
    def CalculateSignatureFromTokenListList(self, tokenListList):
        assert tokenListList is not None, "tokenListList must not be None."
        assert isinstance(tokenListList, list), "tokenListList must be a list."
        assert len(tokenListList) > 0, "tokenListList must not be empty."
        
        numberOfLines = len(tokenListList)
                
        tokenIdList = []
        
        requiredNumberOfMaskTokens = self.seq_length - numberOfLines    
        
        mask_token_filler_list = [Dataset.Universal_Mask_Token] * requiredNumberOfMaskTokens
        
        
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
        # Cast tokenIdList to float32
        tokenIdList = tf.cast(tokenIdList, self.DataTypeForTensors)
        
        one_step_model = ModelClass.OneStep(self.model, Tokens_from_idList=None, temperature=self.temperature)
        
        _, pooledLogitsVectorData = one_step_model.generate_hidded_states(tokenIdList, self.seq_length, self.DataTypeForTensors)

        hiddenVectors = pooledLogitsVectorData['hidden_state_all']

        hiddenVectors = hiddenVectors[:, 0 : numberOfLines, :]
        
        signature = self.CalculateSignatureFromHiddenVectors(hiddenVectors, None)  # No received attention in this case

        return signature
    
    def CalculateSignatureFromHiddenVectors(self, batchedvectors, receivedAttention):
        usePooling = True
        useAveragePooling = True
        useMaxPooling = False
        
        if usePooling:
            
            if receivedAttention is not None:
                reshaped_attention = tf.expand_dims(receivedAttention, axis=-1)
                
                batchedvectors = batchedvectors * reshaped_attention
                    
            batchedvectors = tf.nn.l2_normalize(batchedvectors, axis=-1)
            
            if useAveragePooling:
                pooledAverageVector = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(batchedvectors)
            else:
                pooledAverageVector = None   # this might not work
            
            if useMaxPooling:
                pooledMaxVector = tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last")(batchedvectors)
                
                if pooledAverageVector is not None:
                    pooledVector = tf.concat([pooledAverageVector, pooledMaxVector], axis=-1)

            else:
                pooledVector = pooledAverageVector    
                
            # pooledVector = tf.nn.l2_normalize(pooledVector, axis=-1)
        else:        
            pooledVector = batchedvectors[:, -1, :]

        return pooledVector