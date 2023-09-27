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

def GetFirstSequenceData(LogfileToAnalyze, id_from_token, window_size):
    tokenList, PrintfList = TokenParser.getNextLinesAndTokens(TokenParser.ReForTokens, window_size+1, LogfileToAnalyze)

    if (tokenList is not None):
        tokenWindow = tokenList[0:window_size]  
        outputToken = tokenList[window_size]

        nextTokenIdWindow = id_from_token(tokenWindow)
        outputTokenId     = id_from_token(outputToken)
    else:
        tokenWindow = None 
        outputToken = None 
        nextTokenIdWindow = None  
        outputTokenId = None 
        PrintfList = None 

    return tokenWindow, outputToken, nextTokenIdWindow, outputTokenId, PrintfList

def GetNextSequenceData(newToken, newPrintf, id_from_token, prevWindow, prevInputTokens, prevOutputToken, prevOutputTokenId):
    
    newWindow           = GetNextTokenWindow(prevWindow, prevOutputToken)
    newTokenId          = id_from_token(newToken)
    nextTokenIdWindow   = GetNextTokenIdWindow(prevInputTokens, prevOutputTokenId)

    return newWindow, newToken, newTokenId, nextTokenIdWindow, newPrintf


class AssembleInferenceDataClass:
    def __init__(self, DataTypeForTensors, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window, tokenIdWindow, nextToken, nexttokenId):
        self.DataTypeForTensors = DataTypeForTensors
        self.numberOfParallelExecutions = numberOfParallelExecutions
        self.newTokenList = newTokenList
        self.newPrintfList = newPrintfList
        self.id_from_token = id_from_token
        self.window = window 
        self.tokenIdWindow = tokenIdWindow
        self.nextToken = nextToken 
        self.nexttokenId = nexttokenId
        
        self.inputTensor  = []
        self.nexttokenIds = []
        self.nextTokens   = []
        self.Windows      = []
        self.PrintfList   = []
        
    
    def AssembleInputsThread(self):

        for executionIndex in range(self.numberOfParallelExecutions):
            
            self.window, self.nextToken, self.nexttokenId, self.tokenIdWindow, self.printf = GetNextSequenceData(self.newTokenList[executionIndex], self.newPrintfList[executionIndex], self.id_from_token, self.window, self.tokenIdWindow, self.nextToken, self.nexttokenId)
            
            if (self.window is None):
                break
            else:
                self.Windows.append(self.window) 
                self.PrintfList.append(self.printf)
                self.inputTensor.append(tf.convert_to_tensor(self.tokenIdWindow, dtype=self.DataTypeForTensors))
                self.nexttokenIds.append(tf.convert_to_tensor(self.nexttokenId, dtype=self.DataTypeForTensors))  
                self.nextTokens.append(self.nextToken)


    def WaitForCompletion(self):
        # Wait for the thread to finish 
        self.generate_steps_thread.join()
        return self.Windows, self.nextTokens, self.PrintfList, self.inputTensor, self.nexttokenIds, self.window, self.tokenIdWindow, self.nextToken, self.nexttokenId

    def StartAssembleInputs(self):

        self.generate_steps_thread = threading.Thread(target=self.AssembleInputsThread, args=())

        self.generate_steps_thread.start()


        
        