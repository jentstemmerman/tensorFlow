import multiprocessing
import numpy as np
import copy
import tensorflow as tf
import time

with tf.device('/CPU:0'):
    def GetNextTokenWindow(oldWindow, newToken):
        newWindow = oldWindow[1:]  
        newWindowr = np.append(newWindow, newToken)
        #newWindow.append(newToken)
        return newWindow

    def GetNextTokenIdWindow(oldTokenIdWindow, newTokenId):
        nextTokenIdWindow = oldTokenIdWindow[1:]
        newTokenId = tf.reshape(newTokenId, [1, 2])  # maak compatibel met concat van nextTokenIdWindow
        nextTokenIdWindow = tf.concat([nextTokenIdWindow, newTokenId], axis=0)

        return nextTokenIdWindow

    def GetFirstSequenceData(startIndex, tokenList, id_from_tokenList, window_size, DataTypeForTensors):

        if (tokenList is not None):
            tokenWindow       = tokenList[startIndex:startIndex + window_size]
            outputToken       = tokenList[startIndex + window_size]

            tokenIdWindow = copy.deepcopy(tokenWindow)
            outputTokenId = copy.deepcopy(outputToken)

            column = 0
            for id_from_token in id_from_tokenList:
                if (id_from_token is not None):
                    for tokenInWindow in tokenIdWindow:
                        tokenInWindow[column] = id_from_token(tokenInWindow[column])
                                                  
                    outputTokenId[column] = id_from_token(outputToken[column])
                column += 1  

            tokenIdWindow = tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors)
            outputTokenId = tf.convert_to_tensor(outputTokenId, dtype=DataTypeForTensors)
        else:
            tokenWindow = None 
            outputToken = None 
            tokenIdWindow = None  
            outputTokenId = None 

        return tokenWindow, outputToken, tokenIdWindow, outputTokenId

    def GetNextSequenceData(newToken, id_from_tokenList, prevWindow, prevInputTokens, prevOutputToken, prevOutputTokenId, DataTypeForTensors):
        
        newWindow = GetNextTokenWindow(prevWindow, prevOutputToken)
        newTokenId = copy.deepcopy(newToken)

        column = 0
        for id_from_token in id_from_tokenList:
            if (id_from_token is not None):
                newTokenId[column] = id_from_token(newToken[column])

            column += 1

        newTokenId          = tf.convert_to_tensor(newTokenId, dtype=DataTypeForTensors)
        nextTokenIdWindow   = GetNextTokenIdWindow(prevInputTokens, prevOutputTokenId)

        return newWindow, newToken, newTokenId, nextTokenIdWindow


    def AssembleInputsThread(workerParameter, threadId, DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, window_size):
        with tf.device('/CPU:0'):

            id_from_tokenList = workerParameter 
            inputTensor  = []
            nexttokenIds = []
            nextTokens   = []
            
            window, nextToken, tokenIdWindow, nexttokenId = GetFirstSequenceData(indexOfFirstPrediction - (window_size+1),  newTokenList, id_from_tokenList, window_size, DataTypeForTensors)
            
            inputTensor.append(tokenIdWindow)
            nexttokenIds.append(nexttokenId)  
            nextTokens.append(nextToken)
            index = 0

            
            for executionIndex in range(indexOfFirstPrediction, indexOfFirstPrediction + numberOfParallelExecutions - 1):
                
                if (threadId == 5) and (index == 20):
                     AssembleInputsnextTokenBatchStartTime = time.time()
                nextTokenBatch  = newTokenList[executionIndex]  
                
                if (threadId == 5) and (index == 20):
                    print("   AssembleInputsnextTokenBatchStartTime Time = {:2.3f}.".format((time.time() - AssembleInputsnextTokenBatchStartTime) * 8 * numberOfParallelExecutions))
                

                window, nextToken, nexttokenId, tokenIdWindow = GetNextSequenceData(nextTokenBatch, id_from_tokenList, window, tokenIdWindow, nextToken, nexttokenId, DataTypeForTensors)
                
                
                if (window is None):
                    break
                else:
                    tokenIdWindowTensor = tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors)
                    inputTensor.append(tokenIdWindowTensor)
                    nexttokenIdTensor = tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors)
                    nexttokenIds.append(nexttokenIdTensor)  
                    nextTokens.append(nextToken)

                index += 1
            
            inputTensor  = tf.stack(inputTensor,  axis=0)
            nexttokenIds = tf.stack(nexttokenIds, axis=0)


            result = (nextTokens, inputTensor, nexttokenIds)

            return result
        
        

            


