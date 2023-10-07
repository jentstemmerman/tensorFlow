import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
from pathlib import Path
#import threading
import multiprocessing
from multiprocessing import shared_memory
#import queue
import InferenceAssembly
import RunGeneration
import InferenceOutput
import sys 
import ctypes

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




def AddProbabilitiesPrintfFileParallel(model, numberOfThreads, Workers, DataTypeForTensors, LogfileVerification, FileNameAnalyzed, token_from_id, id_from_token, PrintfFromToken, temperature, seq_length, id_from_tokenNumpy):

    numberOfIdenticals = 0
    averageNextTokenProbability = 0
    averagePredictedTokenProbability = 0
    accumulatedNextTokenProbability = 0.0
    accumulatedPredictedTokenProbability = 0.0


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

    queueTimeout                         = 50
    maxNumberOfParallelExecutions        = 500
    numberOfAnalyzedLines                = 0
    accumulatedNextTokenProbability      = 0
    accumulatedPredictedTokenProbability = 0

    one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token, temperature=temperature)
    # Create a shared memory manager
    manager = multiprocessing.Manager()
    
    # Create a shared list
    sharedPrintfList     = manager.list()
    sharedTokenList      = manager.list()

    TokenList, PrintfList = TokenParser.getLinesAndTokens(TokenParser.ReForTokens, -1, LogfileToAnalyze)

    sharedPrintfList.extend(PrintfList)
    sharedTokenList.extend(TokenList)

        # Create shared memory object
    shm = shared_memory.SharedMemory(create=True, size=id_from_tokenNumpy.nbytes*10)

   
    shared_id_from_token = np.ndarray(id_from_tokenNumpy.shape, dtype=id_from_tokenNumpy.dtype, buffer=shm.buf)
    # Copy the data to shared memory
    np.copyto(shared_id_from_token, id_from_tokenNumpy)

    PrintfList = None
    TokenList  = None

    indexOfFirstPrediction     = window_size + 1
    numberOfParallelExecutions = maxNumberOfParallelExecutions
    assemblyStartTime          = 0
    generationStartTime        = 0
    outputStartTime            = 0


    for threadId in range(0,numberOfThreads):
        inputQueue  = Workers[threadId]["inputQueue"]
        outputQueue = Workers[threadId]["outputQueue"]
        inputQueue.put((InferenceAssembly.AssembleInferenceDataClass.AssembleInputsThread, (DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, sharedTokenList, sharedPrintfList, shared_id_from_token, window_size,)))
        if (threadId == 0):
            assemblySubmissionStartTime = time.time()  
        indexOfFirstPrediction += numberOfParallelExecutions
        
    while (True):
        cycleStartTime = time.time()
        numberOfParallelExecutions = maxNumberOfParallelExecutions
        
        if (numberOfParallelExecutions == 0):
            break

        numberOfAnalyzedLinesThisCycle = 0

        printfList                  = []
        nextTokensList              = []
        nextPredictedTokensList     = []
        probabilityPredictionsList  = []
        probabilityOfNextTokensList = []

        for threadId in range(0,numberOfThreads):
            inputQueue  = Workers[threadId]["inputQueue"]
            outputQueue = Workers[threadId]["outputQueue"]
            nextTokens, printf, inputTensor, nexttokenIds = outputQueue.get(timeout=queueTimeout)
            if (threadId == 0):
                 print("   Assembly Time = {:2.3f}.".format(time.time() - assemblySubmissionStartTime))
                 generationStartTime = time.time()

            nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens = one_step_model.generate_steps(inputTensor, nexttokenIds, states=None)

            if (threadId == 0):
                print("   Generation Time = {:2.3f}.".format(time.time() - generationStartTime))

            printfList.append(printf)
            nextTokensList.append(nextTokens)
            nextPredictedTokensList.append(nextPredictedTokens)
            probabilityPredictionsList.append(probabilityPredictions)
            probabilityOfNextTokensList.append(probabilityOfNextTokens)

       
        for threadId in range(0, numberOfThreads):
            inputQueue  = Workers[threadId]["inputQueue"]
            if (threadId == 0):
                outputStartTime = time.time()
            inputQueue.put((InferenceOutput.InferenceOutputClass.InferenceOutputThread, (numberOfParallelExecutions, nextPredictedTokensList[threadId], nextTokensList[threadId], printfList[threadId], probabilityOfNextTokensList[threadId], probabilityPredictionsList[threadId], )))



        
        for threadId in range(0, numberOfThreads):
            inputQueue  = Workers[threadId]["inputQueue"]
            outputQueue = Workers[threadId]["outputQueue"]

            linesToWrite, numberOfIdenticalsInThisCycle, accumulatedNextTokenProbabilityTemp, accumulatedPredictedTokenProbabilityTemp = outputQueue.get(timeout=queueTimeout)
            
            if (threadId == 0):
                 print("   Output thread Time = {:2.3f}.".format(time.time() - outputStartTime))

            inputQueue.put((InferenceAssembly.AssembleInferenceDataClass.AssembleInputsThread, (DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, sharedTokenList, sharedPrintfList, shared_id_from_token, window_size,)))
            
            if (threadId == 0):
                assemblySubmissionStartTime = time.time()

            accumulatedNextTokenProbability      += accumulatedNextTokenProbabilityTemp
            accumulatedPredictedTokenProbability += accumulatedPredictedTokenProbabilityTemp
            numberOfIdenticals                   += numberOfIdenticalsInThisCycle
            numberOfAnalyzedLines                += numberOfParallelExecutions
            numberOfAnalyzedLinesThisCycle       += numberOfParallelExecutions

            indexOfFirstPrediction += numberOfParallelExecutions

            fileWriteStartTime = time.time()
            for lineToWrite in linesToWrite:
                 LogfileAnalyzed.write(lineToWrite)
            if (threadId == 0):
                 print("   fileWrite Time = {:2.3f}.".format(time.time() - fileWriteStartTime))        


        inputTensor  = []
        nextTokens   = []
        nexttokenIds = []
        Windows      = []
        PrintfList   = []

        Endtime = time.time()

        averageNextTokenProbability      = accumulatedNextTokenProbability / numberOfAnalyzedLines
        averagePredictedTokenProbability = accumulatedPredictedTokenProbability / numberOfAnalyzedLines

        print("  annotated lines: " + str(numberOfAnalyzedLines) + "."
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





