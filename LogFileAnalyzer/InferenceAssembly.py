import threading
import multiprocessing
import TokenParser
import time
import tensorflow as tf
import numpy as np
import ray

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


class AssembleInferenceDataClass:
    @ray.remote(num_returns=4)
    def AssembleInputsThread(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window_size):
        inputTensor  = []
        nexttokenIds = []
        nextTokens   = []
#        Windows      = []
        PrintfList   = []

        print("AssembleInputsThread")
        window, nextToken, tokenIdWindow, nexttokenId, printf = GetFirstSequenceData(indexOfFirstPrediction - (window_size+1),  newTokenList, newPrintfList, id_from_token, window_size)
        
        #Windows.append(window) 
        PrintfList.append(printf)
        inputTensor.append(tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors))
        nexttokenIds.append(tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors))  
        nextTokens.append(nextToken)

        for executionIndex in range(indexOfFirstPrediction, indexOfFirstPrediction + numberOfParallelExecutions):
            
            window, nextToken, nexttokenId, tokenIdWindow, printf = GetNextSequenceData(newTokenList[executionIndex], newPrintfList[executionIndex], id_from_token, window, tokenIdWindow, nextToken, nexttokenId)
            
            if (window is None):
                break
            else:
                #Windows.append(window) 
                PrintfList.append(printf)
                inputTensor.append(tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors))
                nexttokenIds.append(tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors))  
                nextTokens.append(nextToken)

        return nextTokens, PrintfList, inputTensor, nexttokenIds
    
    def StartAssembleInputs(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window_size):
#        AssembleInputsThreadResult_ref = AssembleInferenceDataClass.AssembleInputsThread2.remote(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window_size)
        AssembleInputsThreadResult_ref = AssembleInferenceDataClass.AssembleInputsThread2.remote(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, window_size)

        return AssembleInputsThreadResult_ref                                                  
    
    def WaitForCompletion(AssembleInputsThreadResult_ref):
        # Wait for the thread to finish 
        return ray.get(AssembleInputsThreadResult_ref)

    @ray.remote(num_returns=4)
    def AssembleInputsThread2(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, window_size):
        inputTensor  = []
        nexttokenIds = []
        nextTokens   = []
        PrintfList   = []

        return nextTokens, PrintfList, inputTensor, nexttokenIds

        
        