import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
import Dataset
import os
import re
from pathlib import Path
import sys

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

def GetNextSequenceData(LogfileToAnalyze, id_from_token, prevWindow, prevInputTokens, prevOutputToken, prevOutputTokenId):
    newToken, newPrintf = TokenParser.getNextLinesAndTokens(TokenParser.ReForTokens, 1, LogfileToAnalyze)
    newToken            = newToken[0]
    newPrintf           = newPrintf[0]
    
    newWindow           = GetNextTokenWindow(prevWindow, prevOutputToken)
    newTokenId          = id_from_token(newToken)
    nextTokenIdWindow   = GetNextTokenIdWindow(prevInputTokens, prevOutputTokenId)


    return newWindow, newToken, newTokenId, nextTokenIdWindow, newPrintf


