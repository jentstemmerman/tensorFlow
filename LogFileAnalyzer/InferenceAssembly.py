import multiprocessing
import numpy as np
import os
import tensorflow as tf
import time

with tf.device('/CPU:0'):
    def GetNextTokenWindow(oldWindow, newToken):
        newWindow = oldWindow[1:]  
        newWindow.append(newToken)
        return newWindow

    def GetNextTokenIdWindow(oldTokenIdWindow, newTokenId):
        nextTokenIdWindow = np.roll(oldTokenIdWindow, -1)
        nextTokenIdWindow[-1] = newTokenId
        return nextTokenIdWindow

    def GetFirstSequenceData(startIndex, tokenList, PrintfList, id_from_token, window_size):

        if (tokenList is not None):
            tokenWindow       = tokenList[startIndex:startIndex + window_size]

            outputToken       = tokenList[startIndex + window_size]

            nextTokenIdWindow = id_from_token(tokenWindow)
            outputTokenId     = id_from_token(outputToken)

            Printf            = PrintfList[startIndex + window_size]
        else:
            tokenWindow = None 
            outputToken = None 
            nextTokenIdWindow = None  
            outputTokenId = None 
            PrintfList = None 

        return tokenWindow, outputToken, nextTokenIdWindow, outputTokenId, Printf

    def GetNextSequenceData(newToken, newPrintf, id_from_token, prevWindow, prevInputTokens, prevOutputToken, prevOutputTokenId):
        
        newWindow           = GetNextTokenWindow(prevWindow, prevOutputToken)
        newTokenId          = id_from_token(newToken)
        nextTokenIdWindow   = GetNextTokenIdWindow(prevInputTokens, prevOutputTokenId)

        return newWindow, newToken, newTokenId, nextTokenIdWindow, newPrintf



    def AssembleInputsThread(threadId, DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, shared_id_from_token, window_size):
        with tf.device('/CPU:0'):

            #id_from_token = np.ndarray(shared_id_from_token.shape, dtype=shared_id_from_token.dtype)
            # Copy the data to shared memory
            #np.copyto(id_from_token, shared_id_from_token)
            VocabularyTensor  = tf.convert_to_tensor(shared_id_from_token)
            id_from_token     = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None)
            inputTensor  = []
            nexttokenIds = []
            nextTokens   = []
            PrintfList   = []

            
            window, nextToken, tokenIdWindow, nexttokenId, printf = GetFirstSequenceData(indexOfFirstPrediction - (window_size+1),  newTokenList, newPrintfList, id_from_token, window_size)
            
            PrintfList.append(printf)
            inputTensor.append(tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors))
            nexttokenIds.append(tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors))  
            nextTokens.append(nextToken)
            index = 0

            
            for executionIndex in range(indexOfFirstPrediction, indexOfFirstPrediction + numberOfParallelExecutions):
                
                if (threadId == 5) and (index == 20):
                     AssembleInputsnextTokenBatchStartTime = time.time()
                nextTokenBatch  = newTokenList[executionIndex]  
                
                if (threadId == 5) and (index == 20):
                    print("   AssembleInputsnextTokenBatchStartTime Time = {:2.3f}.".format((time.time() - AssembleInputsnextTokenBatchStartTime) * 8 * numberOfParallelExecutions))
                
                if (threadId == 5) and (index == 20):
                     AssembleInputsprintfBatchStartTime = time.time()

                printfBatch     = newPrintfList[executionIndex] 

                if (threadId == 5) and (index == 20):
                    print("   AssembleInputsprintfBatchStartTime Time = {:2.3f}.".format((time.time() - AssembleInputsprintfBatchStartTime) * 8 * numberOfParallelExecutions))

                window, nextToken, nexttokenId, tokenIdWindow, printf = GetNextSequenceData(nextTokenBatch, printfBatch, id_from_token, window, tokenIdWindow, nextToken, nexttokenId)
                

                
                if (window is None):
                    break
                else:
                    PrintfList.append(printf)
                    tokenIdWindowTensor = tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors)
                    inputTensor.append(tokenIdWindowTensor)
                    nexttokenIdTensor = tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors)
                    nexttokenIds.append(nexttokenIdTensor)  
                    nextTokens.append(nextToken)

                index += 1
            result = (nextTokens, PrintfList, inputTensor, nexttokenIds)

            return result
        
        

            


