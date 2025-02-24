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
import Model


def GetInferenceParameters(remainingNumberOfLines, maxNumberOfParallelExecutions, numberOfThreads):
    
    numberOfParallelExecutions  = maxNumberOfParallelExecutions
    numberOfLinesPerCycle       = numberOfThreads * maxNumberOfParallelExecutions
    done                        = False
    reduced                     = False
    
    if (remainingNumberOfLines < numberOfLinesPerCycle):
        while(remainingNumberOfLines < numberOfLinesPerCycle):
            if (numberOfParallelExecutions > 1):
                numberOfParallelExecutions -= 1
                reduced = True
            elif (numberOfThreads > 1):
                numberOfThreads -= 1
                reduced = True
            else:
                break
            numberOfLinesPerCycle  = numberOfThreads * numberOfParallelExecutions
    else:
        numberOfParallelExecutions = maxNumberOfParallelExecutions

    newNumberOfLinesPerCycle       = numberOfThreads * numberOfParallelExecutions
    
    if (remainingNumberOfLines == 0):
        print(f"reduced Parameters remainingNumberOfLines = {remainingNumberOfLines} newNumberOfLinesPerCycle = {newNumberOfLinesPerCycle}")
        done = True
        
    return done, reduced, numberOfParallelExecutions, numberOfThreads



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
        LogfileCompared  = open(FileNameAnalyzed + "_OnlyPrintf",    "w")

        LogfileAnalyzed.write("Analyzed file at temperature " + str(temperature) + "\n")
        LogfileCompared.write("Analyzed file at temperature " + str(temperature) + "\n")

        file = Path(FileNameHumanReadablePooledVectors)
        file.parent.mkdir(parents=True, exist_ok=True)
        FileHumanReadablePooledVectors  = open(FileNameHumanReadablePooledVectors, "w")

        modelHash = Model.GetHashFromModel(model)
        LogfileAnalyzed.write("Model hash: " + modelHash + "\n")
        LogfileCompared.write("Model hash: " + modelHash + "\n")
        FileHumanReadablePooledVectors.write("Model hash: " + modelHash + "\n")
        
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

        indexOfFirstPrediction  = window_size
        
        hiddenVectorSize = GetInitialInference(DataTypeForTensors, id_from_tokenList, window_size, one_step_model, sharedTokenList, indexOfFirstPrediction)
                
        binaryFileHandler = BinaryFileHandler.BinaryFileHandler(FileNameMachineReadablePooledVectors, data_type="float32", numbers_per_line=hiddenVectorSize, modelHash=modelHash)   
                
        numberOfParallelExecutions = maxNumberOfParallelExecutions
        
        remainingNumberOfLines  = (numberOfLines - numberOfAnalyzedLines) - (window_size + 1)
        noMoreReductionPossible, reduced, numberOfParallelExecutions, numberOfThreads = GetInferenceParameters(remainingNumberOfLines, numberOfParallelExecutions, numberOfThreads)
        
        if (reduced):
            print("reduced parameters 1")
        
        newNumberOfParallelExecutions = numberOfParallelExecutions
        newNumberOfThreads = numberOfThreads
        analyzeStartTime   = time.time()
    
        printfList       = []
        nexttokenIdsList = []
        nextTokensList   = []
        inputTensorList  = []
        
        print(f"Start creating input data numberOfThreads: {numberOfThreads}, numberOfParallelExecutions {numberOfParallelExecutions}") 
        
        for threadId in range(0,numberOfThreads):
            inputQueue  = Workers[threadId]["inputQueue"]
          
            inputQueue.put((InferenceAssembly.AssembleInputsThread, (threadId, DataTypeForTensors, indexOfFirstPrediction, numberOfParallelExecutions, sharedTokenList, origPrintfList, window_size,)))
            
           # for line in newPrint:
           #     LogfileCompared.write(f"{line}")

            indexOfFirstPrediction += numberOfParallelExecutions

        LogfileCompared.flush()
        print(f"Start retrieving input data numberOfThreads: {numberOfThreads}") 
        
        for threadId in range(0,numberOfThreads):
            outputQueue = Workers[threadId]["outputQueue"]
            nextTokens, inputTensor, nexttokenIds, nextLogs = outputQueue.get(timeout=queueTimeout) #nextTokens is a list of a list of tokens
            nexttokenIdsList.append(nexttokenIds)
            nextTokensList.append(nextTokens)
            inputTensorList.append(inputTensor)
            printfList.append(nextLogs)
            
        ########################################################## Main loop starts here ##################################################	    
            
        while (True):
            cycleStartTime = time.time()
                            
            if (numberOfParallelExecutions == 0):
                break
            
            remainingNumberOfLines = (numberOfLines - indexOfFirstPrediction)
            noMoreReductionPossible, reduced, newNumberOfParallelExecutions, newNumberOfThreads = GetInferenceParameters(remainingNumberOfLines, numberOfParallelExecutions, numberOfThreads)

            if (reduced):
                print(f"reduced parameters: newNumberOfThreads: {newNumberOfThreads}, newNumberOfParallelExecutions {newNumberOfParallelExecutions}")
            
            if (noMoreReductionPossible):
                print("noMoreReductionPossible")

            numberOfAnalyzedLinesThisCycle = 0
            nextPredictionsList            = []
            probabilityOfPredictionsList   = []
            probabilityOfActualsList       = []
            crossEntropyOfActualsList      = []
            pooledLogitsVectorDataList     = []
            


            # Assemble inputs for the next cycle
            print(f"Start creating input data indexOfFirstPrediction: {indexOfFirstPrediction} numberOfThreads: {newNumberOfThreads}, numberOfParallelExecutions {newNumberOfParallelExecutions}") 
            if (not noMoreReductionPossible):
                indexOfFirstPredictionTemp = indexOfFirstPrediction
                
                for threadId in range(0, newNumberOfThreads):
                    inputQueue  = Workers[threadId]["inputQueue"]
                    inputQueue.put((InferenceAssembly.AssembleInputsThread, (threadId, DataTypeForTensors, indexOfFirstPredictionTemp, newNumberOfParallelExecutions, sharedTokenList, origPrintfList, window_size,)))
                    indexOfFirstPredictionTemp  += newNumberOfParallelExecutions
            
            
            # Generate inference
            print(f"generate_steps_optimized numberOfThreads: {numberOfThreads}, numberOfParallelExecutions {numberOfParallelExecutions}") 
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

        
            # Start output generation
            print(f"Start output generation numberOfThreads: {numberOfThreads}, numberOfParallelExecutions {numberOfParallelExecutions}") 
            for threadId in range(0, numberOfThreads):
                inputQueue  = Workers[threadId]["inputQueue"]
                inputQueue.put((InferenceOutput.InferenceOutputThread, (threadId, numberOfParallelExecutions, nextPredictionsList[threadId], nextTokensList[threadId], printfList[threadId], probabilityOfActualsList[threadId], crossEntropyOfActualsList[threadId], probabilityOfPredictionsList[threadId], pooledLogitsVectorDataList[threadId],)))
            
            nexttokenIdsList            = []
            nextTokensList              = []
            inputTensorList             = []
            printfList                  = []
            
            # Get inputs for the next inference cycle
            print(f"Start retrieving input data numberOfThreads: {newNumberOfThreads}") 
            if (not noMoreReductionPossible):
                for threadId in range(0,  newNumberOfThreads):
                    outputQueue = Workers[threadId]["outputQueue"]
                    nextTokens, inputTensor, nexttokenIds, nextLogs = outputQueue.get(timeout=queueTimeout)
                    nexttokenIdsList.append(nexttokenIds)
                    nextTokensList.append(nextTokens)
                    inputTensorList.append(inputTensor)
                    printfList.append(nextLogs)
                    

            # Get outputs to write to file
            print(f"Get output generation numberOfThreads: {numberOfThreads}, numberOfParallelExecutions {numberOfParallelExecutions}") 
            for threadId in range(0, numberOfThreads):

                outputQueue = Workers[threadId]["outputQueue"]
                linesToWrite, printfsToWrite, numberOfIdenticalsInThisCycle, accumulatedActualProbability, accumulatedPredictedProbability, pooledHumanReadableLogitsVectorlinesToWrite, pooledMachineReadableLogitsVectorlinesToWrite = outputQueue.get(timeout=queueTimeout)
                
                accumulatedNextTokenProbability      += accumulatedActualProbability
                accumulatedPredictedTokenProbability += accumulatedPredictedProbability
                numberOfIdenticals                   += numberOfIdenticalsInThisCycle
                numberOfAnalyzedLines                += numberOfParallelExecutions
                numberOfAnalyzedLinesThisCycle       += numberOfParallelExecutions
                
                for line in printfsToWrite:
                    LogfileCompared.write(f"{line}")
                
                LogfileCompared.flush()
                
                for lineToWrite in linesToWrite:
                    LogfileAnalyzed.write(''.join(lineToWrite))

                for lineToWrite in pooledHumanReadableLogitsVectorlinesToWrite:
                    FileHumanReadablePooledVectors.write(lineToWrite)
                    
                for lineToWrite in pooledMachineReadableLogitsVectorlinesToWrite:
                    binaryFileHandler.append(lineToWrite)
                    

            inputTensor  = []
            nextTokens   = []
            nexttokenIds = []
            
            Endtime = time.time()
            averageNextTokenProbabilities    = accumulatedNextTokenProbability / (numberOfAnalyzedLines * numberOfFeatures)
            
            averageNextTokenProbabilities    = TokenParser.TransformToListIfSingularType(averageNextTokenProbabilities)

            averagePredictedTokenProbability = accumulatedPredictedTokenProbability / (numberOfAnalyzedLines * numberOfFeatures)
            analysedLinesPerSecond           = numberOfAnalyzedLinesThisCycle / (Endtime - cycleStartTime) 
            remainingTimeInMinutes           = remainingNumberOfLines / (numberOfAnalyzedLines / (time.time() - analyzeStartTime)) / 60
            
                            
            print("  annotated lines: " + str(numberOfAnalyzedLines) + "/" + str(numberOfLines) + " ({:2.2f}%)".format((numberOfAnalyzedLines * 100) / numberOfLines) +
                ". Nr of lines/s: {:2.2f}.".format(analysedLinesPerSecond) + 
                "  Nr Of Ids: " + str(numberOfIdenticals) + 
                ".".join("  Next token prob = {:2.3f}.".format(averageNextTokenProbability) for averageNextTokenProbability in averageNextTokenProbabilities) +
                    " Pred token prob = {:2.3f}.".format(averagePredictedTokenProbability) + 
                "  Time: {:3.3f}. ".format(Endtime - cycleStartTime) + "Remaining time = {:2.2f} min".format(remainingTimeInMinutes))
                
            
            indexOfFirstPrediction    += (newNumberOfParallelExecutions * newNumberOfThreads)    
            numberOfParallelExecutions = newNumberOfParallelExecutions
            numberOfThreads = newNumberOfThreads
            
            
            
            if (noMoreReductionPossible):
                break

        averageNextTokenProbabilities    = accumulatedNextTokenProbability      / numberOfAnalyzedLines
        averagePredictedTokenProbability = accumulatedPredictedTokenProbability / numberOfAnalyzedLines
        IdenticalsToNumberOfLinesRatio   = numberOfIdenticals                   / numberOfAnalyzedLines
           
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

def GetInitialInference(DataTypeForTensors, id_from_tokenList, window_size, one_step_model, sharedTokenList, startIndex):
    result = InferenceAssembly.AssembleInputsThread(id_from_tokenList, 0, DataTypeForTensors, startIndex, 1, sharedTokenList, None, window_size)

    inputTensor = result[1]   
    nexttokenIds = result[2]     
       
    _, _, _, _, pooledLogitsVectorData = one_step_model.generate_steps_optimized(inputTensor, nexttokenIds, states=None)
    numberOfVectors = pooledLogitsVectorData['vector'].shape[1]
    return numberOfVectors
      



