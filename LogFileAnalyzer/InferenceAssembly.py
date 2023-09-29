import threading
import TokenParser
import time
import tensorflow as tf
import numpy as np


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
    def __init__(self, DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window_size):

        self.indexOfFirstPrediction = indexOfFirstPrediction
        self.DataTypeForTensors = DataTypeForTensors
        self.numberOfParallelExecutions = numberOfParallelExecutions
        self.newTokenList = newTokenList
        self.newPrintfList = newPrintfList
        self.id_from_token = id_from_token
        self.window_size = window_size
        
        self.inputTensor  = []
        self.nexttokenIds = []
        self.nextTokens   = []
        self.Windows      = []
        self.PrintfList   = []
        
    
    def AssembleInputsThread(self):
        
        window, nextToken, tokenIdWindow, nexttokenId, printf = GetFirstSequenceData(self.indexOfFirstPrediction - (self.window_size+1),  self.newTokenList, self.newPrintfList, self.id_from_token, self.window_size)
        
        self.Windows.append(window) 
        self.PrintfList.append(printf)
        self.inputTensor.append(tf.convert_to_tensor(tokenIdWindow, dtype=self.DataTypeForTensors))
        self.nexttokenIds.append(tf.convert_to_tensor(nexttokenId, dtype=self.DataTypeForTensors))  
        self.nextTokens.append(nextToken)
        for executionIndex in range(self.indexOfFirstPrediction, self.indexOfFirstPrediction + self.numberOfParallelExecutions):
            
            window, nextToken, nexttokenId, tokenIdWindow, printf = GetNextSequenceData(self.newTokenList[executionIndex], self.newPrintfList[executionIndex], self.id_from_token, window, tokenIdWindow, nextToken, nexttokenId)
            
            if (window is None):
                break
            else:
                self.Windows.append(window) 
                self.PrintfList.append(printf)
                self.inputTensor.append(tf.convert_to_tensor(tokenIdWindow, dtype=self.DataTypeForTensors))
                self.nexttokenIds.append(tf.convert_to_tensor(nexttokenId, dtype=self.DataTypeForTensors))  
                self.nextTokens.append(nextToken)


    def WaitForCompletion(self):
        # Wait for the thread to finish 
        self.generate_steps_thread.join()
        return self.nextTokens, self.PrintfList, self.inputTensor, self.nexttokenIds

    def StartAssembleInputs(self):

        self.generate_steps_thread = threading.Thread(target=self.AssembleInputsThread, args=())

        self.generate_steps_thread.start()


        
        