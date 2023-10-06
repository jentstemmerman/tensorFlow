import threading
import multiprocessing
import TokenParser
import time
import tensorflow as tf
import numpy as np
import ray
import ctypes
import os

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
    def __init__(self):
        self.result_queue = multiprocessing.Queue()
    
    def AssembleInputsThread(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, shared_id_from_token, window_size):
        #tf.config.set_visible_devices([], 'GPU')
        print("AssembleInputsThread")
        AssembleInputsThreadTime = time.time()
        id_from_token = np.ndarray(shared_id_from_token.shape, dtype=shared_id_from_token.dtype)
        # Copy the data to shared memory
        np.copyto(id_from_token, shared_id_from_token)
        VocabularyTensor  = tf.convert_to_tensor(id_from_token)
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
         

        result = (nextTokens, PrintfList, inputTensor, nexttokenIds)
        print("AssembleInputsThread done before queue put time {:2.3f}.".format(time.time() - AssembleInputsThreadTime))

        print("AssembleInputsThread done time {:2.3f}.".format(time.time() - AssembleInputsThreadTime))
        return result
    
    def StartAssembleInputs(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, shared_id_from_token, window_size):

        result_queue = multiprocessing.Queue()
        
        process = multiprocessing.Process(target=AssembleInferenceDataClass.AssembleInputsThread, args=(result_queue, DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, shared_id_from_token, window_size))
        process.start()
 
        returnValue = (process, result_queue)
        return returnValue                                                  
    
    def WaitForCompletion(processData):
        # Wait for the thread to finish 
        print("WaitForCompletion start")
        # time.sleep(5)
        print("WaitForCompletion start ( end sleep)")
        process      = processData[0]
        result_queue = processData[1]
        result = result_queue.get()
        print("WaitForCompletion queue done")
        process.join()
        print("WaitForCompletion done")

        return result[0], result[1], result[2], result[3]

    def AssembleInputsThread2(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, window_size):
        inputTensor  = []
        nexttokenIds = []
        nextTokens   = []
        PrintfList   = []

        return nextTokens, PrintfList, inputTensor, nexttokenIds

        


