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

    def GetNextTokenIdWindow(oldTokenIdWindow, newTokenId, numberOfFeatures):
        
        nextTokenIdWindow = oldTokenIdWindow[1:]
        newTokenId = tf.reshape(newTokenId, [1, numberOfFeatures])  # maak compatibel met concat van nextTokenIdWindow
        nextTokenIdWindow = tf.concat([nextTokenIdWindow, newTokenId], axis=0)

        return nextTokenIdWindow

    def GetFirstSequenceData(tokenList, id_from_tokenList, window_size, DataTypeForTensors, threadId, numberOfParallelExecutions):

        if (tokenList is not None):
            try:
                tokenWindow       = tokenList[0:window_size]
                outputToken       = tokenList[window_size]
            except IndexError:
                print(f"IndexError: threadId: {threadId} numberOfParallelExecutions {numberOfParallelExecutions}")     
                print(f"IndexError: len(tokenList) = {len(tokenList)} ")
                exit()

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
                if (type(newToken[column]) == str):
                    newTokenId[column] = id_from_token(newToken[column])
                else:
                    exit()

            column += 1

        newTokenId          = tf.convert_to_tensor(newTokenId, dtype=DataTypeForTensors)
        nextTokenIdWindow   = GetNextTokenIdWindow(prevInputTokens, prevOutputTokenId, column)

        return newWindow, newToken, newTokenId, nextTokenIdWindow



    def AssembleInputsThread(workerParameter, threadId, DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, origPrintfList, window_size):
        with tf.device('/CPU:0'):

            id_from_tokenList = workerParameter 
            inputTensor  = []
            nexttokenIds = []
            nextTokens   = []
            printfList   = []
            
            assert (indexOfFirstPrediction >= window_size)
            
            tokenListThatWillBeInputToThisFunction = newTokenList
            
            tokenListToUse = tokenListThatWillBeInputToThisFunction[0 : window_size + numberOfParallelExecutions]
                        
            window, nextToken, tokenIdWindow, nexttokenId = GetFirstSequenceData(tokenListToUse, id_from_tokenList, window_size, DataTypeForTensors, threadId, numberOfParallelExecutions)
            
            tokenListToUse = tokenListThatWillBeInputToThisFunction[window_size : window_size + numberOfParallelExecutions]
            
            inputTensor.append(tokenIdWindow)
            nexttokenIds.append(nexttokenId)  
            nextTokens.append(nextToken)
            index = 0
            
            if (origPrintfList is not None):
                printfList = origPrintfList[indexOfFirstPrediction:indexOfFirstPrediction + numberOfParallelExecutions]
            
            for executionIndex in range(numberOfParallelExecutions - 1):
                
                nextTokenBatch  = tokenListToUse[executionIndex]  
                
                if (type(nextTokenBatch) != str):
                    exit
                    
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


            result = (nextTokens, inputTensor, nexttokenIds, printfList)

            return result
        
        

            


