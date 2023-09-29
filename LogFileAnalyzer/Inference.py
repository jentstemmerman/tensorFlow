import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
from pathlib import Path
import threading
import queue
import InferenceAssembly


def AssembleInputs(DataTypeForTensors, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window, tokenIdWindow, nextToken, nexttokenId):
    
    inputTensor  = []
    nexttokenIds = []
    nextTokens   = []
    Windows      = []
    PrintfList   = []

    for executionIndex in range(numberOfParallelExecutions):
        
        window, nextToken, nexttokenId, tokenIdWindow, printf = InferenceAssembly.GetNextSequenceData(newTokenList[executionIndex], newPrintfList[executionIndex], id_from_token, window, tokenIdWindow, nextToken, nexttokenId)
        
        if (window is None):
            break
        else:
            Windows.append(window) 
            PrintfList.append(printf)
            inputTensor.append(tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors))
            nexttokenIds.append(tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors))  
            nextTokens.append(nextToken)
    
    return Windows, nextTokens, PrintfList, inputTensor, nexttokenIds, window, tokenIdWindow, nextToken, nexttokenId




def ComposeLineList(numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions):
    
    linesToWrite = []
    numberOfIdenticals = 0
    accumulatedNextTokenProbability = 0
    accumulatedPredictedTokenProbability = 0

    for executionIndex in range(numberOfParallelExecutions):
        nextToken = nextTokens[executionIndex] 
        nextPredictedToken = nextPredictedTokens[executionIndex].decode('utf-8')  
        
        if (nextToken == nextPredictedToken):
            nextPredictedToken = "Identical  "
            numberOfIdenticals += 1

        nextTokenProbability = (probabilityOfNextTokens[executionIndex]).numpy()
        lineToWrite = "Actual: " + nextToken  + "  {:2.3f}".format(nextTokenProbability) + ". Pred: " + nextPredictedToken + " {:2.3f}".format(probabilityPredictions[executionIndex].numpy()) + ". " + PrintfList[executionIndex]
        linesToWrite.append(lineToWrite)
        accumulatedNextTokenProbability += nextTokenProbability
        accumulatedPredictedTokenProbability += probabilityPredictions[executionIndex].numpy()
    return linesToWrite, numberOfIdenticals, accumulatedNextTokenProbability, accumulatedPredictedTokenProbability



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
#    tokenList, PrintfList = TokenParser.getNextLinesAndTokens(TokenParser.ReForTokens, window_size + 1, LogfileToAnalyze)
 #   window, nextToken, tokenIdWindow, nexttokenId, PrintfList = InferenceAssembly.GetFirstSequenceData(tokenList, PrintfList, id_from_token, window_size)
#    LogfileToAnalyze.seek(0)
    
 ##   for line in PrintfList[:-1]:
  #      LogfileAnalyzed.write(line)


    PrintfTemp   = []
    inputTensor  = []
    nexttokenIds = []
    nextTokens   = []
    Windows      = []
    PrintfList   = []




    maxNumberOfParallelExecutions = 25
    numberOfAnalyzedLines         = 0

    newTokenList, newPrintfList = TokenParser.getLinesAndTokens(TokenParser.ReForTokens, -1, LogfileToAnalyze)
    
    indexOfFirstPrediction = window_size + 1
    
    while (True):
        startTime = time.time()
        numberOfParallelExecutions = maxNumberOfParallelExecutions
        if (numberOfParallelExecutions == 0):
            break
        
        assembleThread = InferenceAssembly.AssembleInferenceDataClass(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window_size)
        indexOfFirstPrediction += numberOfParallelExecutions
        assembleThread.StartAssembleInputs()

        nextTokens, PrintfList, inputTensor, nexttokenIds = assembleThread.WaitForCompletion() 
                                                                         

        
        TimeNeededForAssembly = (time.time() - startTime) 
        print("     TimeNeededForAssembly: {:2.3f}".format(TimeNeededForAssembly) )
 


        startTimeGenerateSteps = time.time()
        nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens = one_step_model.generate_steps(inputTensor, nexttokenIds, states=None)
        TimeNeededForGenerateSteps = (time.time() - startTimeGenerateSteps) 
        print("     TimeNeededForGenerateSteps: {:2.3f}".format(TimeNeededForGenerateSteps) )


        startTimeWriteAnalysis = time.time()
        
        linesToWrite, numberOfIdenticalsInThisCycle, accumulatedNextTokenProbability, accumulatedPredictedTokenProbability = ComposeLineList(numberOfParallelExecutions, nextPredictedTokens, nextTokens, PrintfList, probabilityOfNextTokens, probabilityPredictions)

        numberOfIdenticals    += numberOfIdenticalsInThisCycle
        numberOfAnalyzedLines += numberOfParallelExecutions

        for lineToWrite in linesToWrite:
            LogfileAnalyzed.write(lineToWrite)



        print("     TimeNeededWriteAnalysis: {:2.3f}".format(time.time() - startTimeWriteAnalysis) )
        
        inputTensor  = []
        nextTokens   = []
        nexttokenIds = []
        Windows      = []
        PrintfList   = []    
        numberOfAssembledItemsInThisCycle = 0

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





