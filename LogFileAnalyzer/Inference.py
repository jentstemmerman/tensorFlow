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


def AddProbabilitiesPrintfFileParallel(model, 
                                       numberOfThreads, 
                                       Workers, 
                                       itemList, 
                                       DataTypeForTensors, 
                                       LogfileVerification, 
                                       inferenceDataSetFile, 
                                       FileNameAnalyzed, 
                                       FileNamesHumanReadablePooledVectors, 
                                       FileNamesMachineReadablePooledVectors, 
                                       id_from_tokenList,
                                       token_from_idList, 
                                       temperature, 
                                       seq_length):
          
    with tf.device('/CPU:0'):
            
        maxNumberOfParallelExecutions        = 200  # dit is het aantal samples dat de GPU inferes in 1 keer
        onlyLogLogitsVector                  = False
        
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

        file = Path(FileNamesHumanReadablePooledVectors["all"])
        file.parent.mkdir(parents=True, exist_ok=True)

        FileHumanReadablePooledFirstVectors =     open(FileNamesHumanReadablePooledVectors["first"],     "w")      
        FileHumanReadablePooledLastVectors =      open(FileNamesHumanReadablePooledVectors["last"],      "w")      
        FileHumanReadablePooledPredictedVectors = open(FileNamesHumanReadablePooledVectors["predicted"], "w")  
        FileHumanReadablePooledAllVectors =       open(FileNamesHumanReadablePooledVectors["all"],       "w")        


        modelHash = Model.GetHashFromModel(model)
        LogfileAnalyzed.write("Model hash: " + modelHash + "\n")
        LogfileCompared.write("Model hash: " + modelHash + "\n")
        
        FileHumanReadablePooledFirstVectors.write("Model hash: " + modelHash + "\n")
        FileHumanReadablePooledLastVectors.write("Model hash: " + modelHash + "\n")
        FileHumanReadablePooledPredictedVectors.write("Model hash: " + modelHash + "\n")
        FileHumanReadablePooledAllVectors.write("Model hash: " + modelHash + "\n")
       
        
        one_step_model = ModelClass.OneStep(model, token_from_idList, temperature=temperature)
       
        LogfileVerificationTemp = os.path.splitext(LogfileVerification)[0] + ".temp"
        _ , origPrintfList, numberOfLines = TokenParser.process_file_to_tokenFile(itemList, LogfileVerification, human_readable_output_file=LogfileVerificationTemp)
        
        remainingNumberOfLines  = numberOfLines - (window_size + 1)
        
        numberOfParallelExecutions = maxNumberOfParallelExecutions
   
             
        
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
        
        indexOfFirstPrediction = window_size
               
        hiddenVectorSize = CalculateHiddenVectorSize(DataTypeForTensors, inferenceDataSetFile, window_size, one_step_model, itemList)
          
        binaryFileHandlerFirstLogits     = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["first"], sequence_length=window_size, data_type=DataTypeForTensors, numbers_per_line=hiddenVectorSize, modelHash=modelHash)  
        binaryFileHandlerLastLogits      = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["last"],  sequence_length=window_size, data_type=DataTypeForTensors, numbers_per_line=hiddenVectorSize, modelHash=modelHash)        
        binaryFileHandlerPredictedLogits = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["predicted"], sequence_length=window_size, data_type=DataTypeForTensors, numbers_per_line=hiddenVectorSize, modelHash=modelHash)  
        binaryFileHandlerAllLogits       = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["all"],   sequence_length=window_size, data_type=DataTypeForTensors, numbers_per_line=hiddenVectorSize, modelHash=modelHash)   
        
        analyzeStartTime   = time.time()
    
        printfList      = []
        nextIdList      = []
        idList          = []
        
        print(f"Start creating input data numberOfThreads: {numberOfThreads}, numberOfParallelExecutions {numberOfParallelExecutions}") 
        
        numberOfExecutionsInProgress         = [0 for _ in range(numberOfThreads)]
        
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

        LogfileCompared.flush()
        print(f"Start retrieving input data numberOfThreads: {numberOfThreads}") 
        
        indexOfFirstPredictionTemp = indexOfFirstPrediction
        
        for threadId in range(0,numberOfThreads):
            outputQueue = Workers[threadId]["outputQueue"]
            idInnerList, nextIdsInnerList, numberOfSamples = outputQueue.get(timeout=queueTimeout) #nextTokens is a list of a list of tokens
            
            numberOfExecutionsInProgress[threadId] = numberOfSamples
            
            if (origPrintfList is not None):
                nextLogs = origPrintfList[indexOfFirstPrediction:indexOfFirstPrediction + numberOfSamples]

            printfList.append(nextLogs)
            idList.append(idInnerList)
            nextIdList.append(nextIdsInnerList)
            
            indexOfFirstPrediction += numberOfSamples
                         
        print("Start while loop")
          
        ########################################################## Main loop starts here ##################################################	    
            
        while (True):
            cycleStartTime = time.time()
                            
            if (numberOfParallelExecutions == 0):
                break
            
            remainingNumberOfLines = (numberOfLines - indexOfFirstPrediction)

            numberOfAnalyzedLinesThisCycle = 0
            nextPredictionsList            = []
            probabilityOfPredictionsList   = []
            probabilityOfActualsList       = []
            crossEntropyOfActualsList      = []
            allLogitsVectorDataList        = []
            lastLogitsVectorDataList       = []
            firstLogitsVectorDataList      = []
            predictedLogitsVectorsDataList = []
            
            # Assemble inputs for the next cycle

            indexOfFirstPredictionTemp = indexOfFirstPrediction

            for threadId in range(0,numberOfThreads):
                
                inputQueue  = Workers[threadId]["inputQueue"]
                
               
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
            for threadId in range(0, numberOfThreads):

                if (numberOfExecutionsInProgress[threadId] == 0):
                    if (onlyLogLogitsVector):
                        allLogitsVectorDataList.append([])
                        firstLogitsVectorDataList.append([])
                        lastLogitsVectorDataList.append([])
                        predictedLogitsVectorsDataList.append([])
                    else:
                        allLogitsVectorDataList.append([])
                        firstLogitsVectorDataList.append([])
                        lastLogitsVectorDataList.append([])
                        predictedLogitsVectorsDataList.append([])
                        
                        nextPredictionsList.append([])
                        probabilityOfPredictionsList.append([])
                        probabilityOfActualsList.append([])
                        crossEntropyOfActualsList.append([])
                else:

                    if (onlyLogLogitsVector):
                        _, pooledLogitsVectorData = one_step_model.generate_hidded_states(idList[threadId], window_size)
                        
                        allLogitsVectorsList = pooledLogitsVectorData['hidden_state'].numpy()
                        firstLogitsVectorsList = pooledLogitsVectorData['hidden_state_first'].numpy()   
                        lastLogitsVectorsList = pooledLogitsVectorData['hidden_state_last'].numpy()
                        predictedLogitsVectorsList = pooledLogitsVectorData['hidden_state_prediction'].numpy()  
                        
                        allLogitsVectorDataList.append(allLogitsVectorsList)
                        firstLogitsVectorDataList.append(firstLogitsVectorsList)
                        lastLogitsVectorDataList.append(lastLogitsVectorsList)
                        predictedLogitsVectorsDataList.append(predictedLogitsVectorsList)
                        
                        
                        nextPredictionsList.append([])
                        probabilityOfPredictionsList.append([])
                        probabilityOfActualsList.append([])
                        crossEntropyOfActualsList.append([])
                        
                    else:
                        predictions, probabilityOfPredictions, probabilityOfActuals, crossEntropyOfActuals, pooledLogitsVectorData = one_step_model.generate_steps_optimized(idList[threadId], nextIdList[threadId], window_size=window_size, states=None)
                        
                        allLogitsVectorsList = pooledLogitsVectorData['hidden_state'].numpy()
                        firstLogitsVectorsList = pooledLogitsVectorData['hidden_state_first'].numpy()   
                        lastLogitsVectorsList = pooledLogitsVectorData['hidden_state_last'].numpy() 
                        predictedLogitsVectorsList = pooledLogitsVectorData['hidden_state_prediction'].numpy()
                        
                        allLogitsVectorDataList.append(allLogitsVectorsList)
                        firstLogitsVectorDataList.append(firstLogitsVectorsList)
                        lastLogitsVectorDataList.append(lastLogitsVectorsList)
                        predictedLogitsVectorsDataList.append(predictedLogitsVectorsList)
                        
                        # predictions is een lijst (1 voor elke numberOfParallelExecutions) van een lijst van next predicted tokens of scalars ( 1 voor elke feature )
                        nextPredictionsList.append(predictions)

                        # probabilityPredictionsList is a tensor of shape (numberOfParallelExecutions, number of features)
                        probabilityOfPredictionsList.append(probabilityOfPredictions)

                        # probabilityActuals is a tensor of shape (numberOfParallelExecutions, number of features)
                        probabilityOfActualsList.append(probabilityOfActuals)

                        # crossEntropyOfActuals is a tensor of shape (numberOfParallelExecutions, number of features)
                        crossEntropyOfActualsList.append(crossEntropyOfActuals)


            # Start output generation
            for threadId in range(0, numberOfThreads):
                inputQueue  = Workers[threadId]["inputQueue"]
                
                if (onlyLogLogitsVector):
                    inputQueue.put((InferenceOutput.InferenceOutputThread, (threadId, 
                                                                            numberOfExecutionsInProgress[threadId], 
                                                                            None, 
                                                                            None, 
                                                                            None, 
                                                                            None, 
                                                                            None, 
                                                                            None, 
                                                                            None,
                                                                            None,
                                                                            window_size,
                                                                            onlyLogLogitsVector)))
                else:
                    inputQueue.put((InferenceOutput.InferenceOutputThread, (threadId, 
                                                    numberOfExecutionsInProgress[threadId], 
                                                    nextPredictionsList[threadId], 
                                                    nextIdList[threadId], 
                                                    printfList[threadId], 
                                                    probabilityOfActualsList[threadId], 
                                                    crossEntropyOfActualsList[threadId], 
                                                    probabilityOfPredictionsList[threadId], 
                                                    allLogitsVectorDataList[threadId],
                                                    lastLogitsVectorDataList[threadId], 
                                                    window_size,
                                                    onlyLogLogitsVector)))
            
            printfList      = []
            nextIdList      = []
            idList          = []
            
            totalNumberOfExecutionsWereInProgress = 0
            
            for executionInProgress in numberOfExecutionsInProgress:
                totalNumberOfExecutionsWereInProgress += executionInProgress
            
            numberOfExecutionsInProgress         = [0 for _ in range(numberOfThreads)]
            
            # Get inputs for the next inference cycles
    
            indexOfFirstPredictionTemp = indexOfFirstPrediction
            
            for threadId in range(0,numberOfThreads):
                outputQueue = Workers[threadId]["outputQueue"]

                idInnerList, nextIdsInnerList, numberOfSamples = outputQueue.get(timeout=queueTimeout) #nextTokens is a list of a list of tokens
                
                numberOfExecutionsInProgress[threadId] = numberOfSamples
                
                if (origPrintfList is not None):
                    nextLogs = origPrintfList[indexOfFirstPredictionTemp:indexOfFirstPredictionTemp + numberOfSamples]
            
                printfList.append(nextLogs)
                idList.append(idInnerList)
                nextIdList.append(nextIdsInnerList)
                indexOfFirstPredictionTemp += numberOfSamples

            # Get outputs to write to file
            for threadId in range(0, numberOfThreads):

                outputQueue = Workers[threadId]["outputQueue"]
                HumanReadableTextOutput, pooledHumanReadableLogitsLastVectorlinesToWrite, pooledHumanReadableLogitsAllVectorlinesToWrite = outputQueue.get(timeout=queueTimeout)
                
                if (len(firstLogitsVectorDataList) > 0):   
                    try:    
                        for lineToWrite in firstLogitsVectorDataList[threadId]:
                            binaryFileHandlerFirstLogits.append(lineToWrite)   
                    except Exception as e:
                        print("Error in writing to binary file: ", e)
                    
                    try:
                        for index, lineToWrite in enumerate(lastLogitsVectorDataList[threadId]):
                            binaryFileHandlerLastLogits.append(lineToWrite)   
                    except Exception as e:
                        print("Error in writing to binary file: ", e)
                    
                    try:    
                        for lineToWrite in predictedLogitsVectorsDataList[threadId]:
                            binaryFileHandlerPredictedLogits.append(lineToWrite)   
                    except Exception as e:
                        print("Error in writing to binary file: ", e)
                    
                    try:
                        for _, lineToWrite in enumerate(allLogitsVectorDataList[threadId]):   # this list is a flattened list of allLogitsVectorDataList. Each element is a timeslice of the sequence. Then the next seqenence and so on.
                            binaryFileHandlerAllLogits.append(lineToWrite)   
                    except Exception as e:
                        print("Error in writing to binary file: ", e)
                
                    numberOfAnalyzedLines                += (index + 1)
                    numberOfAnalyzedLinesThisCycle       += (index + 1)
                
                if (HumanReadableTextOutput is not None):
                    
                    linesToWrite, numberOfIdenticalsInThisCycle, accumulatedActualProbability, accumulatedPredictedProbability,  = HumanReadableTextOutput
                    for lineToWrite in linesToWrite:
                        LogfileAnalyzed.write(''.join(lineToWrite))
                        
                    accumulatedNextTokenProbability      += accumulatedActualProbability
                    accumulatedPredictedTokenProbability += accumulatedPredictedProbability
                    numberOfIdenticals                   += numberOfIdenticalsInThisCycle


                if (pooledHumanReadableLogitsLastVectorlinesToWrite is not None):
                    for lineToWrite in pooledHumanReadableLogitsLastVectorlinesToWrite:
                        FileHumanReadablePooledLastVectors.write(lineToWrite)
                
                if (pooledHumanReadableLogitsAllVectorlinesToWrite is not None):
                    for lineToWrite in pooledHumanReadableLogitsAllVectorlinesToWrite:
                        FileHumanReadablePooledAllVectors.write(lineToWrite)
                    
            Endtime = time.time()
            averageNextTokenProbabilities    = accumulatedNextTokenProbability / (numberOfAnalyzedLines * numberOfFeatures)
            
            averageNextTokenProbabilities    = TokenParser.TransformToListIfSingularType(averageNextTokenProbabilities)

            averagePredictedTokenProbability = accumulatedPredictedTokenProbability / (numberOfAnalyzedLines * numberOfFeatures)
            analysedLinesPerSecond           = numberOfAnalyzedLinesThisCycle / (Endtime - cycleStartTime) 
            remainingTimeInMinutes           = remainingNumberOfLines / (numberOfAnalyzedLines / (time.time() - analyzeStartTime)) / 60
            
                            
            print("Annotated lines: " + str(numberOfAnalyzedLines) + "/" + str(numberOfLines) + " ({:2.2f}%)".format((numberOfAnalyzedLines * 100) / numberOfLines) +
                ". Nr of lines/s: {:2.2f}.".format(analysedLinesPerSecond) + 
                "  Nr Of Ids: " + str(numberOfIdenticals) + 
                ".".join("  Next token prob = {:2.3f}.".format(averageNextTokenProbability) for averageNextTokenProbability in averageNextTokenProbabilities) +
                    " Pred token prob = {:2.3f}.".format(averagePredictedTokenProbability) + 
                "  Time: {:3.3f}. ".format(Endtime - cycleStartTime) + "Remaining time = {:2.2f} min".format(remainingTimeInMinutes))
                
            
            indexOfFirstPrediction  += totalNumberOfExecutionsWereInProgress
            
            if (totalNumberOfExecutionsWereInProgress == 0):
                print("No more executions")
                break

        averageNextTokenProbabilities    = accumulatedNextTokenProbability      / numberOfAnalyzedLines
        averagePredictedTokenProbability = accumulatedPredictedTokenProbability / numberOfAnalyzedLines
        IdenticalsToNumberOfLinesRatio   = numberOfIdenticals                   / numberOfAnalyzedLines
           
        averageNextTokenProbabilities = TokenParser.TransformToListIfSingularType(averageNextTokenProbabilities)
        Result = "\nTotal number of annotated lines: " + str(numberOfAnalyzedLines) + "/" + str(numberOfLines) +                \
                    " ({:2.2f}%)".format((numberOfAnalyzedLines * 100) / numberOfLines) + ".\n Number Of Identicals: " +      \
                    str(numberOfIdenticals) + ", ({:2.3f})".format(IdenticalsToNumberOfLinesRatio) +                                \
                    ".\n".join("Average next token probability = {:2.3f}".format(averageNextTokenProbability) for averageNextTokenProbability in averageNextTokenProbabilities) +                            \
                    ".\n Average predicted token probability = {:2.3f}".format(averagePredictedTokenProbability) + "\n"
        
        print(Result)
        LogfileAnalyzed.write(Result)
        
        binaryFileHandlerFirstLogits.close()
        binaryFileHandlerLastLogits.close()
        binaryFileHandlerAllLogits.close()
        binaryFileHandlerPredictedLogits.close()
        
        FileHumanReadablePooledLastVectors.close()
        FileHumanReadablePooledFirstVectors.close()
        FileHumanReadablePooledPredictedVectors.close()
        FileHumanReadablePooledAllVectors.close()
        
        
        LogfileToAnalyze.close()
        LogfileAnalyzed.close()   

def CalculateHiddenVectorSize(DataTypeForTensors, inferenceDataSetFile, seq_length, one_step_model, itemList):
    datasetStreamReader = StreamingDataSet.DatasetStreamReader(dataSetFile=inferenceDataSetFile, itemList=itemList, DataTypeForTensors=DataTypeForTensors, seq_length=seq_length, stride=1, numberOfShards=1)
    datasetStreamer = datasetStreamReader.GetStreamingDataset()
    inputTensorList = []
    nexttokenIdsList = []
    
    batch = datasetStreamer.take(1)
    for (ids, nextIds) in batch:
        inputTensor = ids
        nexttokenIds = nextIds
        inputTensorList.append(inputTensor)
        nexttokenIdsList.append(nexttokenIds)
        
    inputTensor = tf.stack(inputTensorList, axis=0)
    nexttokenIds = tf.stack(nexttokenIdsList, axis=0)
    
    _, _, _, _, pooledLogitsVectorData = one_step_model.generate_steps_optimized(inputTensor, nexttokenIds, window_size = seq_length, states=None)
    numberOfVectors = pooledLogitsVectorData['hidden_state_prediction'].shape[1]
    return numberOfVectors

def force_gpu_init():

    dummy_model = create_dummy_model()
    dummy_input = tf.constant([[1.0]])  # Some dummy input
    with tf.device('/GPU:0'):
        _ = dummy_model(dummy_input)



def create_dummy_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=(1,))
    ])
    return model