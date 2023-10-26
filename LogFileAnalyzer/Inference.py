import time
import tensorflow as tf
#from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
#from pathlib import Path
import multiprocessing
from multiprocessing import shared_memory
import InferenceAssembly
import InferenceOutput


def AssembleInputs(DataTypeForTensors, numberOfParallelExecutions, newTokenList, newPrintfList, id_from_token, window, tokenIdWindow, nextToken, nexttokenId):
    
    inputTensor  = []
    nexttokenIds = []
    nextTokens   = []
    Windows      = []
    PrintfList   = []

    for executionIndex in range(numberOfParallelExecutions):
        
        window, nextToken, nexttokenId, tokenIdWindow, printf = InferenceAssembly.GetNextSequenceData(newTokenList[executionIndex], newPrintfList[executionIndex], id_from_token, window, tokenIdWindow, nextToken, nexttokenId, DataTypeForTensors)
        
        if (window is None):
            break
        else:
            Windows.append(window) 
            PrintfList.append(printf)
            inputTensor.append(tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors))
            nexttokenIds.append(tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors))  
            nextTokens.append(nextToken)
    
    return Windows, nextTokens, PrintfList, inputTensor, nexttokenIds, window, tokenIdWindow, nextToken, nexttokenId

def GetInferenceParameters(remainingNumberOfLines, maxNumberOfParallelExecutions, numberOfThreads):
    numberOfParallelExecutions  = maxNumberOfParallelExecutions
    numberOfLinesPerCycle       = numberOfThreads * maxNumberOfParallelExecutions
    reducedParameters           = False

    if (remainingNumberOfLines < numberOfLinesPerCycle):
        while(remainingNumberOfLines < numberOfLinesPerCycle):
            if (numberOfParallelExecutions > 1):
                numberOfParallelExecutions -= 1
                reducedParameters = True
            elif (numberOfThreads > 1):
                numberOfThreads -= 1
                reducedParameters = True
            else:
                break
            numberOfLinesPerCycle   = numberOfThreads * numberOfParallelExecutions
    else:
        numberOfParallelExecutions = maxNumberOfParallelExecutions

    return reducedParameters, numberOfParallelExecutions, numberOfThreads

def AddProbabilitiesPrintfFileParallel(model, numberOfThreads, Workers, itemList, DataTypeForTensors, LogfileVerification, FileNameAnalyzed, token_from_id, id_from_token, temperature, seq_length, id_from_tokenNumpy):
    with tf.device('/CPU:0'):
        numberOfIdenticals = 0
        averageNextTokenProbability = 0
        averagePredictedTokenProbability = 0
        accumulatedNextTokenProbability = 0.0
        accumulatedPredictedTokenProbability = 0.0

        window_size = seq_length
        
        LogfileToAnalyze = open(LogfileVerification, "r", errors='ignore')
        LogfileAnalyzed  = open(FileNameAnalyzed,    "w")

        LogfileAnalyzed.write("Analyzed file at temperature " + str(temperature) + "\n")

        inputTensor  = []
        nexttokenIds = []
        nextTokens   = []

        queueTimeout                         = 50
        maxNumberOfParallelExecutions        = 500
        numberOfAnalyzedLines                = 0
        accumulatedNextTokenProbability      = 0
        accumulatedPredictedTokenProbability = 0

        one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token, temperature=temperature)
        use_manager = False
        sharedTokenList  = []
        manager          = None
        TokenList, origPrintfList, numberOfFoundTokens = TokenParser.getLinesAndTokens(itemList, -1, LogfileToAnalyze)
        
        use_manager = True
        
        if (use_manager):
            # Create a shared memory manager
            manager = multiprocessing.Manager()
            
            # Create a shared list
            sharedTokenList = manager.list()
            sharedTokenList.extend(TokenList)
        else:
            # this is slower :-(
            manager = multiprocessing.Manager()

            TokenListbufferSize = len(TokenList) * len(TokenList[1])*8  ## 
            shmForTokens      = shared_memory.SharedMemory(create=True, size=TokenListbufferSize)
            sharedTokenList   = np.ndarray(len(TokenList), dtype='U{}'.format(len(TokenList[1])), buffer=shmForTokens.buf)

            for i in range(len(TokenList)):
                sharedTokenList[i] = TokenList[i]

        # Create shared memory object for id_from_token
        shm = shared_memory.SharedMemory(create=True, size=id_from_tokenNumpy.nbytes)
        shared_id_from_token = np.ndarray(id_from_tokenNumpy.shape, dtype=id_from_tokenNumpy.dtype, buffer=shm.buf)
        # Copy the data to shared memory
        np.copyto(shared_id_from_token, id_from_tokenNumpy)
    
        TokenList  = None

        indexOfFirstPrediction  = window_size + 1
        remainingNumberOfLines  = (numberOfFoundTokens - numberOfAnalyzedLines) - (window_size + 1)
        reducedParameters, numberOfParallelExecutions, numberOfThreads = GetInferenceParameters(remainingNumberOfLines, maxNumberOfParallelExecutions, numberOfThreads)
        
        generationStartTime        = 0
        outputStartTime            = 0
        analyzeStartTime           = time.time()
    
        with tf.device('/CPU:0'):
            printfList       = []
            nexttokenIdsList = []
            nextTokensList   = []
            inputTensorList  = []

            for threadId in range(0,numberOfThreads):
                inputQueue  = Workers[threadId]["inputQueue"]
                inputQueue.put((InferenceAssembly.AssembleInputsThread, (threadId, DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, sharedTokenList, shared_id_from_token, window_size,)))
                newPrint = origPrintfList[indexOfFirstPrediction - 1:indexOfFirstPrediction + numberOfParallelExecutions - 1]
                printfList.append(newPrint)
                if (threadId == 0):
                    assemblySubmissionStartTime = time.time()  
                indexOfFirstPrediction += numberOfParallelExecutions

            for threadId in range(0,numberOfThreads):
                outputQueue = Workers[threadId]["outputQueue"]
                nextTokens, inputTensor, nexttokenIds = outputQueue.get(timeout=queueTimeout)
                nexttokenIdsList.append(nexttokenIds)
                nextTokensList.append(nextTokens)
                inputTensorList.append(inputTensor)
                if (threadId == (numberOfThreads-1)):
                    print("   Assembly Time = {:2.3f}.".format(time.time() - assemblySubmissionStartTime))

            while (True):
                cycleStartTime = time.time()
                                
                if (numberOfParallelExecutions == 0):
                    break

                numberOfAnalyzedLinesThisCycle = 0
                nextPredictedTokensList        = []
                probabilityPredictionsList     = []
                probabilityOfNextTokensList    = []
                indexOfFirstPredictionTemp     = indexOfFirstPrediction

                assemblySubmissionStartTime    = time.time()

                if (not reducedParameters):
                    for threadId in range(0,numberOfThreads):
                        inputQueue  = Workers[threadId]["inputQueue"]
                        inputQueue.put((InferenceAssembly.AssembleInputsThread, (threadId, DataTypeForTensors, indexOfFirstPredictionTemp, numberOfParallelExecutions, sharedTokenList, shared_id_from_token, window_size,)))
                        indexOfFirstPredictionTemp  += numberOfParallelExecutions
                
                generationStartTime = time.time()
                for threadId in range(0,numberOfThreads):
                    nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens = one_step_model.generate_steps(inputTensorList[threadId], nexttokenIdsList[threadId], states=None)
                    
                    nextPredictedTokensList.append(nextPredictedTokens)
                    probabilityPredictionsList.append(probabilityPredictions)
                    probabilityOfNextTokensList.append(probabilityOfNextTokens)

                generationTime = time.time() - generationStartTime
                print("   Generation Time = {:2.3f}.".format(generationTime))
            
                for threadId in range(0, numberOfThreads):
                    inputQueue  = Workers[threadId]["inputQueue"]
                    if (threadId == 0):
                        outputStartTime = time.time()
                    inputQueue.put((InferenceOutput.InferenceOutputThread, (threadId, numberOfParallelExecutions, nextPredictedTokensList[threadId], nextTokensList[threadId], printfList[threadId], probabilityOfNextTokensList[threadId], probabilityPredictionsList[threadId], )))
                
                nexttokenIdsList            = []
                nextTokensList              = []
                inputTensorList             = []

                if (not reducedParameters):
                    for threadId in range(0,numberOfThreads):
                        outputQueue = Workers[threadId]["outputQueue"]
                        nextTokens, inputTensor, nexttokenIds = outputQueue.get(timeout=queueTimeout)
                        nexttokenIdsList.append(nexttokenIds)
                        nextTokensList.append(nextTokens)
                        inputTensorList.append(inputTensor)
                    
                print("   assembly Time = {:2.3f}.".format(time.time() - assemblySubmissionStartTime - generationTime))
                printfList = []

                for threadId in range(0, numberOfThreads):

                    outputQueue = Workers[threadId]["outputQueue"]
                    linesToWrite, numberOfIdenticalsInThisCycle, accumulatedNextTokenProbabilityTemp, accumulatedPredictedTokenProbabilityTemp = outputQueue.get(timeout=queueTimeout)
                    
                    if (threadId == (numberOfThreads-1)):
                        print("   Output thread Time = {:2.3f}.".format(time.time() - outputStartTime))

                    newPrint = origPrintfList[indexOfFirstPrediction - 1:indexOfFirstPrediction + numberOfParallelExecutions - 1]
                    printfList.append(newPrint)

                    accumulatedNextTokenProbability      += accumulatedNextTokenProbabilityTemp
                    accumulatedPredictedTokenProbability += accumulatedPredictedTokenProbabilityTemp
                    numberOfIdenticals                   += numberOfIdenticalsInThisCycle
                    numberOfAnalyzedLines                += numberOfParallelExecutions
                    numberOfAnalyzedLinesThisCycle       += numberOfParallelExecutions
                    indexOfFirstPrediction               += numberOfParallelExecutions

                    fileWriteStartTime = time.time()
                    for lineToWrite in linesToWrite:
                        LogfileAnalyzed.write(lineToWrite)
                    if (threadId == 0):
                        print("   fileWrite Time = {:2.3f}.".format(time.time() - fileWriteStartTime))        

                inputTensor  = []
                nextTokens   = []
                nexttokenIds = []
                
                Endtime = time.time()
                remainingNumberOfLines           = (numberOfFoundTokens - numberOfAnalyzedLines) - (window_size + 1)
                averageNextTokenProbability      = accumulatedNextTokenProbability / numberOfAnalyzedLines
                averagePredictedTokenProbability = accumulatedPredictedTokenProbability / numberOfAnalyzedLines
                analysedLinesPerSecond           = numberOfAnalyzedLinesThisCycle / (Endtime - cycleStartTime) 
                remainingTimeInMinutes           = remainingNumberOfLines/(numberOfAnalyzedLines/(time.time() - analyzeStartTime)) / 60
                
                reducedParameters, numberOfParallelExecutions, numberOfThreads = GetInferenceParameters(remainingNumberOfLines, maxNumberOfParallelExecutions, numberOfThreads)
                
                print("  annotated lines: " + str(numberOfAnalyzedLines) + "/" + str(numberOfFoundTokens) + " ({:2.2f}%)".format((numberOfAnalyzedLines * 100)/numberOfFoundTokens) +
                    ". Nr of lines/s: {:2.2f}.".format(analysedLinesPerSecond) + 
                    "  Nr Of Identicals: " + str(numberOfIdenticals) + "."
                    "  Av next token probab = {:2.3f}.".format(averageNextTokenProbability) + 
                    "  Av pred token prob = {:2.3f}.".format(averagePredictedTokenProbability) + 
                    "  Time: {:3.3f}. ".format(Endtime - cycleStartTime) + "Remaining time = {:2.2f} min".format(remainingTimeInMinutes))
                if (analysedLinesPerSecond < 2000):
                    print("Too long")        
                    
                if(reducedParameters):
                    break

        averageNextTokenProbability      = accumulatedNextTokenProbability      / numberOfAnalyzedLines
        averagePredictedTokenProbability = accumulatedPredictedTokenProbability / numberOfAnalyzedLines
        IdenticalsToNumberOfLinesRatio   = numberOfIdenticals                   / numberOfAnalyzedLines

        if(reducedParameters):
            print("\n\nEnded inference almost at end of file")

        Result =  "Total number of annotated lines: " + str(numberOfAnalyzedLines) + "/" + str(numberOfFoundTokens) + " ({:2.2f}%)".format((numberOfAnalyzedLines * 100)/numberOfFoundTokens) + ".\n Number Of Identicals: " + str(numberOfIdenticals) + ", ({:2.3f})".format(IdenticalsToNumberOfLinesRatio) + ".\n Average next token probability = {:2.3f}".format(averageNextTokenProbability) + ".\n Average predicted token probability = {:2.3f}".format(averagePredictedTokenProbability) + "\n"
        print(Result)
        LogfileAnalyzed.write(Result)
        
        LogfileToAnalyze.close()
        LogfileAnalyzed.close()   

        shm.close()
        shm.unlink()




