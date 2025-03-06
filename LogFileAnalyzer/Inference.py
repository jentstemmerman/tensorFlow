import time
import tensorflow as tf
import TokenParser
import ModelClass
import InferenceAssembly
import InferenceOutput
import BinaryFileHandler
import Model
import os
import StreamingDataSet
    
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

def CreateDataSetFromVerificationFile(LogfileVerification, dataSetFile, itemList, id_from_tokenList, seq_length, DataTypeForTensors, dual_input):
    stride = 1
    import Dataset
    
    dataset = Dataset.CreateInferenceDatasetFromFile(   LogfileVerification, 
                                                        None, 
                                                        itemList, 
                                                        id_from_tokenList, 
                                                        stride, 
                                                        seq_length, 
                                                        DataTypeForTensors)
          
    try:
        Dataset.SaveDataset(dataset, dataSetFile)
    except Exception as e:
        print("Error in saving dataset: ", e)
        
    print(f"Dataset from file  {os.path.basename(LogfileVerification)} saved. to {dataSetFile}\n")


def AddProbabilitiesPrintfFileParallel(model, numberOfThreads, Workers, itemList, DataTypeForTensors, LogfileVerification, inferenceDataSetFile, FileNameAnalyzed, FileNameHumanReadablePooledVectors, FileNameMachineReadablePooledVectors, id_from_tokenList, token_from_idList, temperature, seq_length):
          
    with tf.device('/CPU:0'):
            
        maxNumberOfParallelExecutions        = 200  # dit is het aantal samples dat de GPU inferes in 1 keer
        
        numberOfFeatures = len(id_from_tokenList)
        queueTimeout                         = 50000
        numberOfIdenticals                   = 0
        averagePredictedTokenProbability     = 0
        accumulatedNextTokenProbability      = 0.0
        accumulatedPredictedTokenProbability = 0.0

        numberOfAnalyzedLines                = 0
        accumulatedNextTokenProbability      = 0
        accumulatedPredictedTokenProbability = 0
        
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
        
        one_step_model = ModelClass.OneStep(model, token_from_idList, temperature=temperature)
       
        LogfileVerificationTemp = os.path.splitext(LogfileVerification)[0] + ".temp"
        _ , origPrintfList, numberOfLines = TokenParser.process_file_to_tokenFile(itemList, LogfileVerification, human_readable_output_file=LogfileVerificationTemp)
        
        remainingNumberOfLines  = numberOfLines - (window_size + 1)
        
        numberOfParallelExecutions = maxNumberOfParallelExecutions
        
        noMoreReductionPossible, reduced, numberOfParallelExecutions, numberOfThreads = GetInferenceParameters(remainingNumberOfLines, numberOfParallelExecutions, numberOfThreads)
        
        datasetStreamWriter = StreamingDataSet.DatasetStreamWriter(inputFile=LogfileVerification, 
                                                                   dataSetFile=inferenceDataSetFile, 
                                                                   itemList=itemList, 
                                                                   id_from_tokenList=id_from_tokenList, 
                                                                   DataTypeForTensors=DataTypeForTensors, 
                                                                   numberOfShards=numberOfThreads,
                                                                   chardGranularity=numberOfParallelExecutions,
                                                                   windowSize=window_size)
        
        print("Start writing streaming dataset... This may take a long time")
        datasetStreamWriter.WriteStreamingDataset()
        print("Done writing streaming dataset")
        
        datasetStreamWriter = None
        
        indexOfFirstPrediction     = window_size
               
        hiddenVectorSize = GetInitialInference(DataTypeForTensors, inferenceDataSetFile, window_size, one_step_model, itemList)
               
        binaryFileHandler = BinaryFileHandler.BinaryFileHandler(FileNameMachineReadablePooledVectors, data_type="float32", numbers_per_line=hiddenVectorSize, modelHash=modelHash)   
        
        if (reduced):
            print("reduced parameters 1")
        
        newNumberOfParallelExecutions = numberOfParallelExecutions
        newNumberOfThreads = numberOfThreads
        
        analyzeStartTime   = time.time()
    
        printfList      = []
        nextIdList      = []
        nextTokenList   = []
        idList          = []
        tokenList       = []
        
        print(f"Start creating input data numberOfThreads: {numberOfThreads}, numberOfParallelExecutions {numberOfParallelExecutions}") 
        
        for threadId in range(0,numberOfThreads):
            inputQueue  = Workers[threadId]["inputQueue"]
            print("Before inputQueue.put AssembleInputsThread ", threadId)
            
            inputQueue.put((InferenceAssembly.AssembleInputsThread, (threadId,
                                                                     inferenceDataSetFile, 
                                                                     itemList, 
                                                                     DataTypeForTensors, 
                                                                     seq_length, 
                                                                     1, 
                                                                     numberOfThreads, 
                                                                     indexOfFirstPrediction, 
                                                                     numberOfParallelExecutions, 
                                                                     window_size,)))

            indexOfFirstPrediction += numberOfParallelExecutions

        LogfileCompared.flush()
        print(f"Start retrieving input data numberOfThreads: {numberOfThreads}") 
        
        indexOfFirstPredictionTemp = indexOfFirstPrediction
        
        for threadId in range(0,numberOfThreads):
            outputQueue = Workers[threadId]["outputQueue"]
            idInnerList, tokenInnerList, nextIdsInnerList, nextTokenInnerList = outputQueue.get(timeout=queueTimeout) #nextTokens is a list of a list of tokens

            if (origPrintfList is not None):
                nextLogs = origPrintfList[indexOfFirstPredictionTemp:indexOfFirstPredictionTemp + numberOfParallelExecutions]

            indexOfFirstPredictionTemp += numberOfParallelExecutions
            printfList.append(nextLogs)
            idList.append(idInnerList)
            tokenList.append(tokenInnerList)
            nextIdList.append(nextIdsInnerList)
            nextTokenList.append(nextTokenInnerList)
                         
            
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
           
                for threadId in range(0,numberOfThreads):
                    
                    inputQueue  = Workers[threadId]["inputQueue"]
                    print("Before inputQueue.put AssembleInputsThread ", threadId)
                    
                    inputQueue.put((InferenceAssembly.AssembleInputsThread, (threadId,
                                                                             inferenceDataSetFile, 
                                                                             itemList, 
                                                                             DataTypeForTensors, 
                                                                             seq_length, 
                                                                             1, 
                                                                             numberOfThreads, 
                                                                             indexOfFirstPredictionTemp, 
                                                                             numberOfParallelExecutions, 
                                                                             window_size,)))


                    indexOfFirstPredictionTemp += numberOfParallelExecutions
            
            # Generate inference
            print(f"generate_steps_optimized numberOfThreads: {numberOfThreads}, numberOfParallelExecutions {numberOfParallelExecutions}") 
            for threadId in range(0, numberOfThreads):

                predictions, probabilityOfPredictions, probabilityOfActuals, crossEntropyOfActuals, pooledLogitsVectorData = one_step_model.generate_steps_optimized(idList[threadId], nextIdList[threadId], states=None)
                
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
                inputQueue.put((InferenceOutput.InferenceOutputThread, (threadId, numberOfParallelExecutions, nextPredictionsList[threadId], nextTokenList[threadId], printfList[threadId], probabilityOfActualsList[threadId], crossEntropyOfActualsList[threadId], probabilityOfPredictionsList[threadId], pooledLogitsVectorDataList[threadId],)))
            
            printfList      = []
            nextIdList      = []
            nextTokenList   = []
            idList          = []
            tokenList       = []
            
            # Get inputs for the next inference cycle
            print(f"Start retrieving input data numberOfThreads: {newNumberOfThreads}") 
            indexOfFirstPredictionTemp = indexOfFirstPrediction
            
            if (not noMoreReductionPossible):
                for threadId in range(0,numberOfThreads):
                    outputQueue = Workers[threadId]["outputQueue"]

                    idInnerList, tokenInnerList, nextIdsInnerList, nextTokenInnerList = outputQueue.get(timeout=queueTimeout) #nextTokens is a list of a list of tokens
                    
                    if (origPrintfList is not None):
                        nextLogs = origPrintfList[indexOfFirstPredictionTemp:indexOfFirstPredictionTemp + numberOfParallelExecutions]
                    
                    indexOfFirstPredictionTemp += numberOfParallelExecutions
                    
                    printfList.append(nextLogs)
                    idList.append(idInnerList)
                    tokenList.append(tokenInnerList)
                    nextIdList.append(nextIdsInnerList)
                    nextTokenList.append(nextTokenInnerList)
                    

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

def GetInitialInference(DataTypeForTensors, inferenceDataSetFile, seq_length, one_step_model, itemList):
    datasetStreamReader = StreamingDataSet.DatasetStreamReader(dataSetFile=inferenceDataSetFile, itemList=itemList, DataTypeForTensors=DataTypeForTensors, seq_length=seq_length, stride=1, numberOfShards=1)
    datasetStreamer = datasetStreamReader.GetStreamingDataset()
    inputTensorList = []
    nexttokenIdsList = []
    
    batch = datasetStreamer.take(1)
    for (ids, _ , nextIds, _) in batch:
        inputTensor = ids
        nexttokenIds = nextIds
        inputTensorList.append(inputTensor)
        nexttokenIdsList.append(nexttokenIds)
        
    inputTensor = tf.stack(inputTensorList, axis=0)
    nexttokenIds = tf.stack(nexttokenIdsList, axis=0)
    
    _, _, _, _, pooledLogitsVectorData = one_step_model.generate_steps_optimized(inputTensor, nexttokenIds, states=None)
    numberOfVectors = pooledLogitsVectorData['vector'].shape[1]
    return numberOfVectors





