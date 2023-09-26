import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
from pathlib import Path
import threading
import queue

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

def AssembleInputs(DataTypeForTensors, startIndex, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window, tokenIdWindow, nextToken, nexttokenId):
    
    y = startIndex
    inputTensor  = []
    nexttokenIds = []
    nextTokens   = []
    Windows      = []
    PrintfList   = []

    while (y < numberOfParallelExecutions):
        
        window, nextToken, nexttokenId, tokenIdWindow, printf = GetNextSequenceData(newTokenList[y-startIndex], newPrintfList[y-startIndex], id_from_token, window, tokenIdWindow, nextToken, nexttokenId)
        
        if (window is None):
            break
        else:
            Windows.append(window) 
            PrintfList.append(printf)
            inputTensor.append(tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors))
            nexttokenIds.append(tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors))  
            nextTokens.append(nextToken)
        
        y += 1

    return Windows, nextTokens, PrintfList, inputTensor, nexttokenIds, window, tokenIdWindow, nextToken, nexttokenId

def AddProbabilitiesPrintfFileParallel(model, DataTypeForTensors, LogfileVerification, FileNameAnalyzed, token_from_id, id_from_token, PrintfFromToken, temperature, seq_length):

    numberOfIdenticals = 0
    averageNextTokenProbability = 0
    averagePredictedTokenProbability = 0
    accumulatedNextTokenProbability = 0.0
    accumulatedPredictedTokenProbability = 0.0

    one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token, temperature=temperature)
    window_size = seq_length

    
    LogfileToAnalyze = open(LogfileVerification, "r", errors='ignore')
    LogfileAnalyzed  = open(FileNameAnalyzed,    "w")

    LogfileAnalyzed.write("Analyzed file at temperature " + str(temperature) + "\n")

    window, nextToken, tokenIdWindow, nexttokenId, PrintfList = GetFirstSequenceData(LogfileToAnalyze, id_from_token, window_size)

    for line in PrintfList[0:-1]:
        LogfileAnalyzed.write(line)
    
    PrintfTemp = PrintfList[-1] 
    inputTensor  = []
    nexttokenIds = []
    nextTokens   = []
    Windows      = []
    PrintfList   = []

    Windows.append(window)
    nextTokens.append(nextToken)
    inputTensor.append(tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors))
    nexttokenIds.append(tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors))  # make it a list of tensors
    PrintfList.append(PrintfTemp)
    PrintfTemp = None
    numberOfAssembledItemsInThisCycle = 1

    maxNumberOfParallelExecutions     = 500
    numberOfAnalyzedLines             = 0

    while (True):
        startTime = time.time()
        numberOfParallelExecutions = maxNumberOfParallelExecutions
        if (numberOfParallelExecutions == 0):
            break
        
        newTokenList, newPrintfList = TokenParser.getNextLinesAndTokens(TokenParser.ReForTokens, numberOfParallelExecutions - numberOfAssembledItemsInThisCycle, LogfileToAnalyze)
        Windows2, nextTokens2, PrintfList2, inputTensor2, nexttokenIds2, window, tokenIdWindow, nextToken, nexttokenId = AssembleInputs(DataTypeForTensors, numberOfAssembledItemsInThisCycle, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window, tokenIdWindow, nextToken, nexttokenId)
                                                                                         
        inputTensor  = inputTensor + inputTensor2
        nexttokenIds = nexttokenIds + nexttokenIds2
        nextTokens   = nextTokens + nextTokens2
        Windows      = Windows + Windows2
        PrintfList   = PrintfList + PrintfList2


        
        if (window == None):
            break
        
        inputTensorForInference  = tf.stack(inputTensor, axis=0)
        nexttokenIdsForInference = tf.stack(nexttokenIds, axis=0)

        TimeNeededForAssembly = (time.time() - startTime) 
        print("     TimeNeededForAssembly: {:2.3f}".format(TimeNeededForAssembly) )
        startTimeGenerateSteps = time.time()
        nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens = one_step_model.generate_steps(inputTensorForInference, nexttokenIdsForInference, states=None)
        TimeNeededForGenerateSteps = (time.time() - startTimeGenerateSteps) 
        print("     TimeNeededForGenerateSteps: {:2.3f}".format(TimeNeededForGenerateSteps) )
        y = 0
        startTimeWriteAnalysis = time.time()
        while (y < numberOfParallelExecutions):
            nextToken = nextTokens[y] 
            nextPredictedToken = nextPredictedTokens[y].decode('utf-8')  
            
            if (nextToken == nextPredictedToken):
                nextPredictedToken = "Identical  "
                numberOfIdenticals += 1

            nextTokenProbability = (probabilityOfNextTokens[y]).numpy()
            lineToWrite = "Actual: " + nextToken  + "  {:2.3f}".format(nextTokenProbability) + ". Pred: " + nextPredictedToken + " {:2.3f}".format(probabilityPredictions[y].numpy()) + ". " + PrintfList[y]
            #print(lineToWrite)
            LogfileAnalyzed.write(lineToWrite)
            
            accumulatedNextTokenProbability += nextTokenProbability
            accumulatedPredictedTokenProbability += probabilityPredictions[y].numpy()

            y += 1
            numberOfAnalyzedLines += 1
        print("     TimeNeededWriteAnalysis: {:2.3f}".format(time.time() - startTimeWriteAnalysis) )
        
        inputTensor  = []
        nextTokens   = []
        nexttokenIds = []
        Windows      = []
        PrintfList   = []    
        numberOfAssembledItemsInThisCycle = 0

        y = 0
        Endtime = time.time()

        averageNextTokenProbability      = accumulatedNextTokenProbability / numberOfAnalyzedLines
        averagePredictedTokenProbability = accumulatedPredictedTokenProbability / numberOfAnalyzedLines

        print("  annotated line " + str(numberOfAnalyzedLines) + "."
              ". Number of lines per second: {:2.2f}.".format((numberOfParallelExecutions / (Endtime - startTime) )) + 
              "  Number Of Identicals: " + str(numberOfIdenticals) + "."
              "  Average next token probability = {:2.3f}.".format(averageNextTokenProbability) + 
              "  Average predicted token probability = {:2.3f}.".format(averagePredictedTokenProbability) + 
              "  Time: {:3.3f}\n".format(Endtime - startTime))


    averageNextTokenProbability      = accumulatedNextTokenProbability / numberOfAnalyzedLines
    averagePredictedTokenProbability = accumulatedPredictedTokenProbability / numberOfAnalyzedLines
    IdenticalsToNumberOfLinesRatio   = numberOfIdenticals / numberOfAnalyzedLines
    Result = "\n\nTotal number of annotated lines " + str(numberOfAnalyzedLines) + ".\n Number Of Identicals: " + str(numberOfIdenticals) + ", ({:2.3f})".format(IdenticalsToNumberOfLinesRatio) + ".\n Average next token probability = {:2.3f}".format(averageNextTokenProbability) + ".\n Average predicted token probability = {:2.3f}".format(averagePredictedTokenProbability) + "\n"
    print(Result)
    LogfileAnalyzed.write(Result)
    
    LogfileToAnalyze.close()
    LogfileAnalyzed.close()   





