import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
from pathlib import Path
#import threading
#import queue
import InferenceAssembly
import RunGeneration
import InferenceOutput
import ray

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




def AddProbabilitiesPrintfFileParallel(model, DataTypeForTensors, LogfileVerification, FileNameAnalyzed, token_from_id, id_from_token, PrintfFromToken, temperature, seq_length):
    ray.init()
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

    PrintfTemp   = []
    inputTensor  = []
    nexttokenIds = []
    nextTokens   = []
    Windows      = []
    PrintfList   = []

    numberOfThreads = 16
    maxNumberOfParallelExecutions = 250
    numberOfAnalyzedLines         = 0
    accumulatedNextTokenProbability = 0
    accumulatedPredictedTokenProbability = 0
    newTokenList, newPrintfList = TokenParser.getLinesAndTokens(TokenParser.ReForTokens, -1, LogfileToAnalyze)
    
    indexOfFirstPrediction = window_size + 1
    numberOfParallelExecutions = maxNumberOfParallelExecutions
    assembleThreadList = []
    assemblyStartTime = 0
    generationStartTime = 0
    outputStartTime = 0

    assemblySubmissionStartTime = time.time()
    for threadId in range(0,numberOfThreads):
       
        startassemblyStartTime = time.time()
        AssembleInputsThreadResult_ref = InferenceAssembly.AssembleInferenceDataClass.StartAssembleInputs(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window_size)
        assembleThreadList.append(AssembleInputsThreadResult_ref)
        indexOfFirstPrediction += numberOfParallelExecutions
        print("   startassemblyStart Time = {:2.3f}.".format(time.time() - startassemblyStartTime))
        
        if (threadId == 0):
              assemblyStartTime = time.time()

    print("   assemblySubmission Time = {:2.3f}.".format(time.time() - assemblySubmissionStartTime))

    while (True):
        cycleStartTime = time.time()
        numberOfParallelExecutions = maxNumberOfParallelExecutions
        
        if (numberOfParallelExecutions == 0):
            break

        numberOfAnalyzedLinesThisCycle = 0
        generationThreadList = []
        PrintfList = []
        nextTokensList = []

        for threadId in range(0,numberOfThreads):
            nextTokens, printf, inputTensor, nexttokenIds = InferenceAssembly.AssembleInferenceDataClass.WaitForCompletion(assembleThreadList[threadId]) 
            generationThread = RunGeneration.GenerationClass(one_step_model, inputTensor, nexttokenIds, states=None)     
            PrintfList.append(printf)
            nextTokensList.append(nextTokens)
            generationThreadList.append(generationThread)                                          
            generationThreadList[threadId].StartGeneration()
            if (threadId == 0):
                print("   AssembleTime = {:2.3f}.".format(time.time() - assemblyStartTime))
                generationStartTime = time.time()
        
        for threadId in range(0, numberOfThreads):
            nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens = generationThreadList[threadId].WaitForCompletion()  

        outputThreadList = []
        for threadId in range(0, numberOfThreads):
            nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens = generationThreadList[threadId].WaitForCompletion() 
            outputThread = InferenceOutput.InferenceOutputClass(numberOfParallelExecutions, nextPredictedTokens, nextTokensList[threadId], PrintfList[threadId], probabilityOfNextTokens, probabilityPredictions)
            outputThreadList.append(outputThread)                                          
            outputThreadList[threadId].StartInferenceOutput()
            if (threadId == 0):
                 print("   Generation Time = {:2.3f}.".format(time.time() - generationStartTime))
                 outputStartTime = time.time()


        assembleThreadList = []

        for threadId in range(0, numberOfThreads):
            linesToWrite, numberOfIdenticalsInThisCycle, accumulatedNextTokenProbabilityTemp, accumulatedPredictedTokenProbabilityTemp = outputThreadList[threadId].WaitForCompletion()
            assembleThread = InferenceAssembly.AssembleInferenceDataClass(DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window_size)
            assembleThreadList.append(assembleThread)

            assembleThreadList[threadId].StartAssembleInputs()
            accumulatedNextTokenProbability      += accumulatedNextTokenProbabilityTemp
            accumulatedPredictedTokenProbability += accumulatedPredictedTokenProbabilityTemp
            numberOfIdenticals                   += numberOfIdenticalsInThisCycle
            numberOfAnalyzedLines                += numberOfParallelExecutions
            numberOfAnalyzedLinesThisCycle       += numberOfParallelExecutions

            indexOfFirstPrediction += numberOfParallelExecutions
            if (threadId == 0):
                 print("   Output Time = {:2.3f}.".format(time.time() - outputStartTime))
                 assemblyStartTime = time.time()

            for lineToWrite in linesToWrite:
                 LogfileAnalyzed.write(lineToWrite)



        print("Generation")
        inputTensor  = []
        nextTokens   = []
        nexttokenIds = []
        Windows      = []
        PrintfList   = []

        Endtime = time.time()

        averageNextTokenProbability      = accumulatedNextTokenProbability / numberOfAnalyzedLines
        averagePredictedTokenProbability = accumulatedPredictedTokenProbability / numberOfAnalyzedLines

        print("  annotated line " + str(numberOfAnalyzedLines) + "."
              ". Number of lines per second: {:2.2f}.".format((numberOfAnalyzedLinesThisCycle / (Endtime - cycleStartTime) )) + 
              "  Number Of Identicals: " + str(numberOfIdenticals) + "."
              "  Average next token probability = {:2.3f}.".format(averageNextTokenProbability) + 
              "  Average predicted token probability = {:2.3f}.".format(averagePredictedTokenProbability) + 
              "  Time: {:3.3f}\n".format(Endtime - cycleStartTime))


    averageNextTokenProbability      = accumulatedNextTokenProbability / numberOfAnalyzedLines
    averagePredictedTokenProbability = accumulatedPredictedTokenProbability / numberOfAnalyzedLines
    IdenticalsToNumberOfLinesRatio   = numberOfIdenticals / numberOfAnalyzedLines
    Result = "\n\nTotal number of annotated lines " + str(numberOfAnalyzedLines) + ".\n Number Of Identicals: " + str(numberOfIdenticals) + ", ({:2.3f})".format(IdenticalsToNumberOfLinesRatio) + ".\n Average next token probability = {:2.3f}".format(averageNextTokenProbability) + ".\n Average predicted token probability = {:2.3f}".format(averagePredictedTokenProbability) + "\n"
    print(Result)
    LogfileAnalyzed.write(Result)
    
    LogfileToAnalyze.close()
    LogfileAnalyzed.close()   
    ray.shutdown()




