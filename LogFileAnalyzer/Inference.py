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
import BinaryFileHandler

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

    if (reducedParameters):
        print("reduced Parameters numberOfParallelExecutions = ", numberOfParallelExecutions, " numberOfThreads = ", numberOfThreads)

    return reducedParameters, numberOfParallelExecutions, numberOfThreads



def AddProbabilitiesPrintfFileParallel(model, numberOfThreads, Workers, itemList, DataTypeForTensors, LogfileVerification, FileNameAnalyzed, FileNameHumanReadablePooledVectors, FileNameMachineReadablePooledVectors, id_from_tokenList, token_from_idList, temperature, seq_length):
    with tf.device('/CPU:0'):

        numberOfIdenticals = 0
        averagePredictedTokenProbability = 0
        accumulatedNextTokenProbability = 0.0
        accumulatedPredictedTokenProbability = 0.0
        numberOfFeatures = len(id_from_tokenList)
        window_size = seq_length
        
        LogfileToAnalyze = open(LogfileVerification, "r", errors='ignore')
        
        from pathlib import Path
        file = Path(FileNameAnalyzed)
        file.parent.mkdir(parents=True, exist_ok=True)
        LogfileAnalyzed  = open(FileNameAnalyzed,    "w")


        LogfileAnalyzed.write("Analyzed file at temperature " + str(temperature) + "\n")

        file = Path(FileNameHumanReadablePooledVectors)
        file.parent.mkdir(parents=True, exist_ok=True)
        FileHumanReadablePooledVectors  = open(FileNameHumanReadablePooledVectors, "w")

        inputTensor  = []
        nexttokenIds = []
        nextTokens   = []

        queueTimeout                         = 50
        maxNumberOfParallelExecutions        = 200  # dit is het aantal samples dat de GPU inferes in 1 keer
        numberOfAnalyzedLines                = 0
        accumulatedNextTokenProbability      = 0
        accumulatedPredictedTokenProbability = 0

        one_step_model = ModelClass.OneStep(model, token_from_idList, temperature=temperature)

        sharedTokenList  = []
        manager          = None

        print("Retrieving tokens from files... ", LogfileVerification)
        TokenList, origPrintfList, numberOfLines = TokenParser.process_file_to_tokenFile(itemList, LogfileVerification, None)
        print("Retrieving tokens from files Done")

        print("Creating shared memory manager...")
        # Create a shared memory manager
        manager = multiprocessing.Manager()
        
        # Create a shared token list
        sharedTokenList = manager.list()
        sharedTokenList.extend(TokenList) # bug
        print("Creating shared memory manager Done")
        TokenList  = None

        indexOfFirstPrediction  = window_size + 1
        
        hiddenVectorSize = GetInitialInference(DataTypeForTensors, id_from_tokenList, window_size, one_step_model, sharedTokenList, indexOfFirstPrediction)
        
        hashAtCreation = "fdafdsvfdi" # model.GetHashAtCreation()
        print(hashAtCreation)
        
        binaryFileHandler = BinaryFileHandler.BinaryFileHandler(FileNameMachineReadablePooledVectors, data_type="float32", numbers_per_line=hiddenVectorSize, modelHash=hashAtCreation)   
        
        remainingNumberOfLines  = (numberOfLines - numberOfAnalyzedLines) - (window_size + 1)
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
                inputQueue.put((InferenceAssembly.AssembleInputsThread, (threadId, DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, sharedTokenList, window_size,)))
                newPrint = origPrintfList[indexOfFirstPrediction - 1:indexOfFirstPrediction + numberOfParallelExecutions - 1]
                printfList.append(newPrint)
                if (threadId == 0):
                    assemblySubmissionStartTime = time.time()  
                indexOfFirstPrediction += numberOfParallelExecutions

            for threadId in range(0,numberOfThreads):
                outputQueue = Workers[threadId]["outputQueue"]
                nextTokens, inputTensor, nexttokenIds = outputQueue.get(timeout=queueTimeout) #nextTokens is a list of a list of tokens
                nexttokenIdsList.append(nexttokenIds)
                nextTokensList.append(nextTokens)
                inputTensorList.append(inputTensor)
                if (threadId == (numberOfThreads-1)):
                    print("   Assembly Time = {:2.3f}.".format(time.time() - assemblySubmissionStartTime))

            print("stop")

            while (True):
                cycleStartTime = time.time()
                                
                if (numberOfParallelExecutions == 0):
                    break

                numberOfAnalyzedLinesThisCycle = 0
                nextPredictionsList            = []
                probabilityOfPredictionsList   = []
                probabilityOfActualsList       = []
                crossEntropyOfActualsList      = []
                pooledLogitsVectorDataList     = []
                indexOfFirstPredictionTemp     = indexOfFirstPrediction

                assemblySubmissionStartTime    = time.time()

                if (not reducedParameters):
                    for threadId in range(0,numberOfThreads):
                        inputQueue  = Workers[threadId]["inputQueue"]
                        inputQueue.put((InferenceAssembly.AssembleInputsThread, (threadId, DataTypeForTensors, indexOfFirstPredictionTemp, numberOfParallelExecutions, sharedTokenList, window_size,)))
                        indexOfFirstPredictionTemp  += numberOfParallelExecutions
                
                generationStartTime = time.time()
                
                for threadId in range(0, numberOfThreads):

                    predictions, probabilityOfPredictions, probabilityOfActuals, crossEntropyOfActuals, pooledLogitsVectorData = one_step_model.generate_steps_optimized(inputTensorList[threadId], nexttokenIdsList[threadId], states=None)
                    
                    # predictions is een lijst (1 voor elke numberOfParallelExecutions) van een lijst van next predicted tokens of scalars ( 1 voor elke feature )
                    nextPredictionsList.append(predictions)

                    # probabilityPredictionsList is a tensor of shape (numberOfParallelExecutions, number of features)
                    probabilityOfPredictionsList.append(probabilityOfPredictions)

                    # probabilityActuals is a tensor of shape (numberOfParallelExecutions, number of features)
                    probabilityOfActualsList.append(probabilityOfActuals)

                    # crossEntropyOfActuals is a tensor of shape (numberOfParallelExecutions, number of features)
                    crossEntropyOfActualsList.append(crossEntropyOfActuals)

                    pooledLogitsVectorDataList.append(pooledLogitsVectorData)

                generationTime = time.time() - generationStartTime
                print("   Generation Time = {:2.3f}.".format(generationTime))
            
                for threadId in range(0, numberOfThreads):
                    inputQueue  = Workers[threadId]["inputQueue"]
                    if (threadId == 0):
                        outputStartTime = time.time()
                    inputQueue.put((InferenceOutput.InferenceOutputThread, (threadId, numberOfParallelExecutions, nextPredictionsList[threadId], nextTokensList[threadId], printfList[threadId], probabilityOfActualsList[threadId], crossEntropyOfActualsList[threadId], probabilityOfPredictionsList[threadId], pooledLogitsVectorDataList[threadId],)))
                
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
                    linesToWrite, numberOfIdenticalsInThisCycle, accumulatedActualProbability, accumulatedPredictedProbability, pooledHumanReadableLogitsVectorlinesToWrite, pooledMachineReadableLogitsVectorlinesToWrite= outputQueue.get(timeout=queueTimeout)
                    
                    if (threadId == (numberOfThreads-1)):
                        print("   Output thread Time = {:2.3f}.".format(time.time() - outputStartTime))

                    newPrint = origPrintfList[indexOfFirstPrediction - 1:indexOfFirstPrediction + numberOfParallelExecutions - 1]
                    printfList.append(newPrint)

                    accumulatedNextTokenProbability      += accumulatedActualProbability
                    accumulatedPredictedTokenProbability += accumulatedPredictedProbability
                    numberOfIdenticals                   += numberOfIdenticalsInThisCycle
                    numberOfAnalyzedLines                += numberOfParallelExecutions
                    numberOfAnalyzedLinesThisCycle       += numberOfParallelExecutions
                    indexOfFirstPrediction               += numberOfParallelExecutions

                    fileWriteStartTime = time.time()

                    for lineToWrite in linesToWrite:
                        LogfileAnalyzed.write(lineToWrite)

                    for lineToWrite in pooledHumanReadableLogitsVectorlinesToWrite:
                        FileHumanReadablePooledVectors.write(lineToWrite)
                        
                    for lineToWrite in pooledMachineReadableLogitsVectorlinesToWrite:
                        binaryFileHandler.append(lineToWrite)
                        
                        
                    if (threadId == 0):
                        print("   fileWrite Time = {:2.3f}.".format(time.time() - fileWriteStartTime))        

                inputTensor  = []
                nextTokens   = []
                nexttokenIds = []
                
                Endtime = time.time()
                remainingNumberOfLines           = (numberOfLines - indexOfFirstPrediction) - (window_size + 1)
                averageNextTokenProbabilities    = accumulatedNextTokenProbability / (numberOfAnalyzedLines * numberOfFeatures)
                
                averageNextTokenProbabilities = TokenParser.TransformToListIfSingularType(averageNextTokenProbabilities)

                averagePredictedTokenProbability = accumulatedPredictedTokenProbability / (numberOfAnalyzedLines * numberOfFeatures)
                analysedLinesPerSecond           = numberOfAnalyzedLinesThisCycle / (Endtime - cycleStartTime) 
                remainingTimeInMinutes           = remainingNumberOfLines/(numberOfAnalyzedLines/(time.time() - analyzeStartTime)) / 60
                
                reducedParameters, numberOfParallelExecutions, numberOfThreads = GetInferenceParameters(remainingNumberOfLines, maxNumberOfParallelExecutions, numberOfThreads)
                
                print("  annotated lines: " + str(numberOfAnalyzedLines) + "/" + str(numberOfLines) + " ({:2.2f}%)".format((numberOfAnalyzedLines * 100)/numberOfLines) +
                    ". Nr of lines/s: {:2.2f}.".format(analysedLinesPerSecond) + 
                    "  Nr Of Identicals: " + str(numberOfIdenticals) + 
                    ".".join("  Av next token probab = {:2.3f}.".format(averageNextTokenProbability) for averageNextTokenProbability in averageNextTokenProbabilities) +
                     "  Av pred token prob = {:2.3f}.".format(averagePredictedTokenProbability) + 
                    "  Time: {:3.3f}. ".format(Endtime - cycleStartTime) + "Remaining time = {:2.2f} min".format(remainingTimeInMinutes))
                #if (analysedLinesPerSecond < 2000):
                #    print("Too long")        
                    
                if(reducedParameters):
                    break

        averageNextTokenProbabilities    = accumulatedNextTokenProbability      / numberOfAnalyzedLines
        averagePredictedTokenProbability = accumulatedPredictedTokenProbability / numberOfAnalyzedLines
        IdenticalsToNumberOfLinesRatio   = numberOfIdenticals                   / numberOfAnalyzedLines

        if(reducedParameters):
            print("\n\nEnded inference almost at end of file")
        averageNextTokenProbabilities = TokenParser.TransformToListIfSingularType(averageNextTokenProbabilities)
        Result = "Total number of annotated lines: " + str(numberOfAnalyzedLines) + "/" + str(numberOfLines) +                \
                    " ({:2.2f}%)".format((numberOfAnalyzedLines * 100) / numberOfLines) + ".\n Number Of Identicals: " +      \
                    str(numberOfIdenticals) + ", ({:2.3f})".format(IdenticalsToNumberOfLinesRatio) +                                \
                    ".\n".join("Average next token probability = {:2.3f}".format(averageNextTokenProbability) for averageNextTokenProbability in averageNextTokenProbabilities) +                            \
                    ".\n Average predicted token probability = {:2.3f}".format(averagePredictedTokenProbability) + "\n"
        
        print(Result)
        LogfileAnalyzed.write(Result)
        
        LogfileToAnalyze.close()
        LogfileAnalyzed.close()   
        binaryFileHandler.close()

def GetInitialInference(DataTypeForTensors, id_from_tokenList, window_size, one_step_model, sharedTokenList, indexOfFirstPrediction):
    result = InferenceAssembly.AssembleInputsThread(id_from_tokenList, 0, DataTypeForTensors, indexOfFirstPrediction, 1, sharedTokenList, window_size)

    inputTensor = result[1]   
    nexttokenIds = result[2]     
       
    _, _, _, _, pooledLogitsVectorData = one_step_model.generate_steps_optimized(inputTensor, nexttokenIds, states=None)
    numberOfVectors = pooledLogitsVectorData['vector'].shape[1]
    return numberOfVectors
      



