import Logfile
import tensorflow as tf
import ModelClass  # noqa: F401
import Dataset     # noqa: F401
import Constants                    # type: ignore
# import DefaultTrainingSettings
import os
import json
from Mask import Mask

class Signature:
    
    def __init__(self, 
                 name,
                 logSnipped=None, 
                 logFile=None, 
                 linesIndexList=None, 
                 startLine=None,
                 endLine=None):
        
        assert name is not None, "Signature name must be provided."
        assert logSnipped is not None or logFile is not None, "Either logSnipped or logFile must be provided."

        self.name = name
        self.logFile = logFile
        self.linesIndexList = linesIndexList
        self.logSnipped = logSnipped
        self.signature = None
        
        if logSnipped is not None:
            self.logSnipped = logSnipped
            self.numberOfLines = len(logSnipped)
        elif logFile is not None:
            assert startLine is not None or endLine is not None or linesIndexList is not None
            
            if linesIndexList is not None:
                self.linesIndexList = linesIndexList         
                       
            else:
                self.linesIndexList = []
                if startLine is not None and endLine is not None:
                    for i in range(startLine, endLine + 1):
                        self.linesIndexList.append(i)
                        
            self.logSnipped = self.GetLogSnipped()
        else:
            self.logSnipped = None
        
    def compare_with(self, other_signature, one_step_model):
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
        

    def GetLogSnipped(self):
        if self.logSnipped is not None:
            return self.logSnipped
        
        if self.logFile is not None and self.linesIndexList is not None:
            logFile = Logfile.Logfile(fileName=self.logFile)
            logSnipped = logFile.GetTokensFromLineIndexList(self.linesIndexList)
            return logSnipped
        
        return None

    def CalculateSignature(self, one_step_model):
        if self.signature is None:
            assert one_step_model is not None, "one_step_model must not be None."
            assert self.logSnipped is not None, "logSnipped must not be None."
            logFile = Logfile.Logfile(fileName=self.logFile, itemList=one_step_model.itemList)
            tokenListList = logFile.GetTokensFromLines(self.logSnipped)      
                
            self.signature = self.CalculateSignatureFromTokenListList(tokenListList, one_step_model)
                        

                        
        return self.signature
        
    # the resulting signature is a vector of shape (batch, hidden_vectors ( contatenated over the features))                
    def CalculateSignatureFromTokenListList(self, tokenListList, one_step_model):
        assert tokenListList is not None, "tokenListList must not be None."
        assert isinstance(tokenListList, list), "tokenListList must be a list."
        assert len(tokenListList) > 0, "tokenListList must not be empty."
        
        numberOfLines = len(tokenListList)
                
        tokenIdList = []
        seq_length = one_step_model.seq_length
        
        
        requiredNumberOfMaskTokens = seq_length - numberOfLines    
        
        mask_token_filler_list = [Mask.Universal_Mask_Token] * requiredNumberOfMaskTokens
        
        
        for columnIndex, id_from_token in enumerate(one_step_model.id_from_tokenList):
            
            assert id_from_token is not None, "id_from_token must not be None, if it is, this means that the feature is not tokenized, which is not supported for signature calculation YET"

            columnList = [column[columnIndex] for column in tokenListList]
            columnList.extend(mask_token_filler_list)  # add the mask tokens to the end of the column
                
            if (id_from_token is None):
                idColumn = tf.convert_to_tensor(columnList, dtype=one_step_model.DataTypeForTensors)
            else:
                idColumn = id_from_token(columnList)
            
            tokenIdList.append(idColumn)
            # After the loop, stack all idColumns along the last dimension to form the input tensor
            tokenIdList = tf.stack(tokenIdList, axis=-1)


        # Add a batch dimension of 1 to tokenIdList
        tokenIdList = tf.expand_dims(tokenIdList, axis=0)
        # Cast tokenIdList to float32
        tokenIdList = tf.cast(tokenIdList, one_step_model.DataTypeForTensors)
                
        _, pooledLogitsVectorData = one_step_model.generate_hidded_states(tokenIdList)

        hiddenVectors = pooledLogitsVectorData['hidden_state_all']

        hiddenVectors = hiddenVectors[:, 0 : numberOfLines, :]
        
        signature = self.CalculateSignatureFromHiddenVectors(hiddenVectors, None)  # No received attention in this case

        assert signature is not None, "Signature must not be None."
        assert signature.shape[0] == 1, "Signature must have a batch dimension of 1."
        
        # we need to remove the batch dimension
        signature = tf.squeeze(signature, axis=0)
        
        return signature
    
    
    # This function expects batchedvectors with shape (batch, features, seq_length)
    # returns a vector of shape (batch, hidden vector). The features are already concatenated into one hidden vector.
    @tf.function
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
    
    @tf.function
    def CalculateSimularityBatched(self, tensor, receivedAttention):
        
        assert self.signature is not None, "signature must not be None."
        
        pooledVector = self.CalculateSignatureFromHiddenVectors(tensor, receivedAttention)
        pooledVector = tf.linalg.l2_normalize(pooledVector, axis=-1)
        
        signature = tf.linalg.l2_normalize(self.signature, axis=-1)
        
        batched_signature_tensor = tf.stack([signature] * tensor.shape[0], axis=0)
        
        
        similarity = tf.keras.losses.cosine_similarity(batched_signature_tensor, pooledVector)
    
        return similarity
    
    
    @staticmethod
    def LoadSignatureFromFile(signatureFilename):
        with open(signatureFilename, 'r') as file:
            signature_data = json.load(file)    
                        
            signature_name = signature_data['signature_name']
            signature_text = signature_data['signature_text']
            logFileName = signature_data['logFileName']

            # Create the Signature instance
            signature_instance = Signature(
                name=signature_name,
                logSnipped=signature_text,
                logFile=logFileName,
                linesIndexList=None,
            )

            return signature_instance
    
    @staticmethod
    def SaveSignatureFile(signature_name, logFileName, selectedTextForSignature, userId):
        try:
            linesIndexList = [e["line"] for e in selectedTextForSignature if "line" in e]

            logFile = Logfile.Logfile(fileName=logFileName, itemList=None)
            
            signature_text = logFile.GetLinesFromLineIndexList(linesIndexList=linesIndexList)
            
            signature_data = {
                "signature_name": signature_name,
                "logFileName": logFileName,
                "signature_text": signature_text,
            }

            file_path = Constants.GetSignatureFilePath(signature_name, userId)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w") as file:
                json.dump(signature_data, file, indent=4, sort_keys=True)

            print("Signature saved at ", file_path)
             
            return signature_name
            

        except Exception as e:
            print("Error saving signature:", e)
            return None