import time
import tensorflow as tf
import SimularityGraph
import TokenParser
import ModelClass
import InferenceAssembly
import InferenceOutput
import BinaryFileHandler
import Model
import os
import StreamingDataSet
import numpy as np
import Logfile
import SignatureCollection
import Constants
from   SimilarityMaxShifter import SimilarityMaxShifter

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




def InferSimilarities(  model, 
                        userId,
                        progress_cb,
                        numberOfThreads, 
                        Workers, 
                        itemList, 
                        DataTypeForTensors, 
                        LogfileVerification,
                        signatureInputDirectory,
                        SimilarityOutputDirectory,
                        inferenceDataSetFile, 
                        id_from_tokenList,
                        token_from_idList, 
                        temperature, 
                        seq_length):
          
    with tf.device('/CPU:0'):
            
        maxNumberOfParallelExecutions        = 200  # dit is het aantal samples dat de GPU inferes in 1 keer
        queueTimeout                         = 50000
        numberOfAnalyzedLines                = 0
        
        LogfileToAnalyze = open(LogfileVerification, "r", errors='ignore')
        
        from pathlib import Path

        SimilarityOutputDirectory = Path(SimilarityOutputDirectory)
        SimilarityOutputDirectory.parent.mkdir(parents=True, exist_ok=True)
                    

              
        logFile = Logfile.Logfile(LogfileVerification, itemList)
        
        numberOfLines = logFile.GetNumberOfLinesInFile()
        
        logFile = None
        
        remainingNumberOfLines  = numberOfLines - (seq_length + 1)
        
        numberOfParallelExecutions = maxNumberOfParallelExecutions
   
        signatureCollection = SignatureCollection.SignatureCollection(signatureInputDirectory)

        signatureLengthList = signatureCollection.GetSignatureLengthList()
        
        similarityMaxShifter  = SimilarityMaxShifter(number_of_simularities=len(signatureLengthList), max_similarity_length=seq_length, shift=-1, signatureLengthList=signatureLengthList, DataTypeForTensors=DataTypeForTensors)
        
        one_step_model = ModelClass.OneStep(model, DataTypeForTensors, seq_length, itemList, token_from_idList, id_from_tokenList, temperature=temperature, signatureLengthList=signatureLengthList, similarityMaxShifter=similarityMaxShifter)
                
        signatureList         = signatureCollection.get_signatures_and_calculateVectors(one_step_model)
                
        machineReadableOutputFileList = []
        humanReadableOutputFileList   = []
        simularityCalculationIndexes = np.ones(len(signatureList), dtype=int) # because the output of the model starts at line 1

        for signatureIndex, signature in enumerate(signatureList):
            signatureFile = Constants.GetSignatureFilePath(signature.name, userId)
            
            machineReadableOutputFile = Constants.GetMachineReadableSimularitiesFile(LogfileVerification, userId, signatureFile)
            humanReadableOutputFile   = Constants.GetHumanReadableSimularitiesFile(LogfileVerification, userId, signatureFile)
            
            # Delete output files if they exist
            if os.path.exists(machineReadableOutputFile):
                os.remove(machineReadableOutputFile)
            if os.path.exists(humanReadableOutputFile):
                os.remove(humanReadableOutputFile)
            
            # Ensure the directory for humanReadableOutputFile exists
            human_readable_dir = os.path.dirname(humanReadableOutputFile)
            if human_readable_dir and not os.path.exists(human_readable_dir):
                os.makedirs(human_readable_dir, exist_ok=True)
                
            machine_readable_dir = os.path.dirname(machineReadableOutputFile)
            if machine_readable_dir and not os.path.exists(machine_readable_dir):
                os.makedirs(machine_readable_dir, exist_ok=True)
            
            
            # Open the file for writing (creates if it does not exist)
            humanReadableOutputFileHandler = open(humanReadableOutputFile, 'w')
            
            outputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(machineReadableOutputFile, data_type=DataTypeForTensors, numbers_per_line=1, sequence_length=None, modelHash="AA")
            
            machineReadableOutputFileList.append(outputBinaryFileHandler)
            humanReadableOutputFileList.append(humanReadableOutputFileHandler)
           
        datasetStreamWriter = StreamingDataSet.DatasetStreamWriter(inputFile=LogfileVerification, 
                                                                   dataSetFile=inferenceDataSetFile, 
                                                                   itemList=itemList, 
                                                                   id_from_tokenList=id_from_tokenList, 
                                                                   DataTypeForTensors=DataTypeForTensors, 
                                                                   numberOfShards=numberOfThreads,
                                                                   chardGranularity=numberOfParallelExecutions,
                                                                   windowSize=seq_length)
        
        print("Start writing streaming dataset... This may take a long time")
        datasetStreamWriter.WriteStreamingDataset()
        print("Done writing streaming dataset")
        
        datasetStreamWriter = None
        
        indexOfFirstPrediction = seq_length
                
        
        analyzeStartTime   = time.time()
       
        idList          = []
              
        print(f"Start creating input data numberOfThreads: {numberOfThreads}, numberOfParallelExecutions {numberOfParallelExecutions}") 
        
        # set this list to 0. An entry for each thread
        numberOfExecutionsInProgressPerThread = [0 for _ in range(numberOfThreads)]
        
        indexOfFirstPredictionTemp = indexOfFirstPrediction
        
        
        
        for threadId in range(0,numberOfThreads):
            inputQueue  = Workers[threadId]["inputQueue"]
            print("Before inputQueue.put AssembleInputsThread ", threadId)
            
            inputQueue.put((InferenceAssembly.SimularityInputsThread, (threadId,
                                                                     inferenceDataSetFile, 
                                                                     itemList, 
                                                                     DataTypeForTensors, 
                                                                     seq_length, 
                                                                     1, 
                                                                     numberOfThreads, 
                                                                     indexOfFirstPredictionTemp, 
                                                                     numberOfParallelExecutions, 
                                                                     seq_length,)))

            indexOfFirstPredictionTemp += numberOfParallelExecutions


        print(f"Start retrieving input data numberOfThreads: {numberOfThreads}") 
        
        indexOfFirstPredictionTemp = indexOfFirstPrediction
        
        for threadId in range(0,numberOfThreads):
            outputQueue = Workers[threadId]["outputQueue"]
            idInnerList, numberOfSamples = outputQueue.get(timeout=queueTimeout) #nextTokens is a list of a list of tokens
            numberOfExecutionsInProgressPerThread[threadId] = numberOfSamples
            idList.append(idInnerList)
            indexOfFirstPrediction += numberOfSamples
                         
        print("Start while loop")
          
        ########################################################## Main loop starts here ##################################################	    
            
        while (True):
            cycleStartTime = time.time()
                            
            if (numberOfParallelExecutions == 0):
                break
            
            remainingNumberOfLines = numberOfLines - indexOfFirstPrediction

            numberOfAnalyzedLinesThisCycle   = 0
            similarityDataList               = []
            
            # Assemble inputs for the next cycle
            indexOfFirstPredictionTemp = indexOfFirstPrediction

            for threadId in range(0,numberOfThreads):
                
                inputQueue  = Workers[threadId]["inputQueue"]
                
               
                inputQueue.put((InferenceAssembly.SimularityInputsThread, (threadId,
                                                                            inferenceDataSetFile, 
                                                                            itemList, 
                                                                            DataTypeForTensors, 
                                                                            seq_length, 
                                                                            1, 
                                                                            numberOfThreads, 
                                                                            indexOfFirstPredictionTemp, 
                                                                            numberOfParallelExecutions, 
                                                                            seq_length,)))


                indexOfFirstPredictionTemp += numberOfParallelExecutions
            
            inputTimeTaken = time.time() - cycleStartTime 
            
            InferenceStartTime   = time.time()
            # Generate inference
            for threadId in range(0, numberOfThreads):

                if (numberOfExecutionsInProgressPerThread[threadId] > 0):

                    # Generate raw per-signature similarities for this window
                    similarity_list_tensor = one_step_model.generate_simularity_lists(idList[threadId], signatureList)

                    # Apply max shifting with carry-over; get filtered [B, n]
                    filtered = similarityMaxShifter.max_shifting_batched(similarity_list_tensor)
                    # Convert to a list of per-signature tensors to match downstream expectations
                    filtered_list = tf.unstack(filtered, axis=0)
                    similarityDataList.append(filtered_list)      

                        
                        
            inferenceTimeTaken = time.time() - InferenceStartTime

            idList          = []
            
            totalNumberOfExecutionsWereInProgress = 0
            
            for executionInProgress in numberOfExecutionsInProgressPerThread:
                totalNumberOfExecutionsWereInProgress += executionInProgress
            
            numberOfExecutionsInProgressPerThread         = [0 for _ in range(numberOfThreads)]
           
            indexOfFirstPredictionTemp = indexOfFirstPrediction
            
            for threadId in range(0,numberOfThreads):
                outputQueue = Workers[threadId]["outputQueue"]

                idInnerList, numberOfSamples = outputQueue.get(timeout=queueTimeout) #nextTokens is a list of a list of tokens
                
                numberOfExecutionsInProgressPerThread[threadId] = numberOfSamples
                            
                idList.append(idInnerList)

                indexOfFirstPredictionTemp += numberOfSamples

            # Get outputs to write to file
            #assert len(signatureList) == numberOfThreads, "Number of signatures in signatureList does not match numberOfThreads"
            outputStartTime   = time.time()
            for threadId in range(0, numberOfThreads):
                if (numberOfExecutionsInProgressPerThread[threadId] > 0):
                    for signatureIndex, simularityData in enumerate(similarityDataList[threadId]):
    
                        #shumanReadableOutputFileList[signatureIndex].write(f"{simularityCalculationIndexes[signatureIndex]}\t{simularityData.numpy()}\n")              
                        machineReadableOutputFileList[signatureIndex].appendNumpy(simularityData.numpy())
                        simularityCalculationIndexes[signatureIndex] += simularityData.shape[0]

                numberOfAnalyzedLines                += numberOfExecutionsInProgressPerThread[threadId]
                numberOfAnalyzedLinesThisCycle       += numberOfExecutionsInProgressPerThread[threadId]
                
                       
            Endtime = time.time()
            outputStartTime = Endtime - outputStartTime

            analysedLinesPerSecond           = numberOfAnalyzedLinesThisCycle / (Endtime - cycleStartTime) 
            remainingTimeInMinutes           = remainingNumberOfLines / (numberOfAnalyzedLines / (time.time() - analyzeStartTime)) / 60
            
            # Report progress (convert to int percentage). Callback expects keyword argument 'progress'.
            if progress_cb:
                try:
                    progress_percent = int(numberOfAnalyzedLines / numberOfLines * 100)
                    progress_cb(progress=progress_percent)
                except Exception as e:
                    print(f"Progress callback failed: {e}")
            print("Annotated lines: " + str(numberOfAnalyzedLines) + "/" + str(numberOfLines) + " ({:2.2f}%)".format((numberOfAnalyzedLines * 100) / numberOfLines) +
                ". Nr of lines/s: {:2.2f}.".format(analysedLinesPerSecond) + 
                " inputTimeTaken: {:3.3f}.".format(inputTimeTaken) +
                " inferenceTimeTaken: {:3.3f}.".format(inferenceTimeTaken) +
                " outputTimeTaken: {:3.3f}.".format(outputStartTime) +
                "  Time: {:3.3f}. ".format(Endtime - cycleStartTime) + "Remaining time = {:2.2f} min".format(remainingTimeInMinutes))
                
            
            indexOfFirstPrediction  += totalNumberOfExecutionsWereInProgress
            
            if (totalNumberOfExecutionsWereInProgress == 0):
                print("No more executions \n")
                break
            else:
                print("Total number of executions in progress: ", totalNumberOfExecutionsWereInProgress, "\n")
           

        Result = "\nTotal number of annotated lines: " + str(numberOfAnalyzedLines) + "/" + str(numberOfLines) +                \
                    " ({:2.2f}%)".format((numberOfAnalyzedLines * 100) / numberOfLines) + "\n"                               
        
        print(Result)
        
        for machineReadableOutputFile in machineReadableOutputFileList:
            machineReadableOutputFile.close()
        
        for humanReadableOutputFile in humanReadableOutputFileList:
            humanReadableOutputFile.close()
        
        
        LogfileToAnalyze.close()
        
        for SimularityFile in machineReadableOutputFileList:
            SimularityGraph.MakeSimularityPlot(SimularityFile, block=False, Save=True, DataTypeForTensors=DataTypeForTensors)

        
        
def AddProbabilitiesPrintfFileParallel(model, 
                                       userId,
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

        FileHumanReadablePooledFirstVectors       = open(FileNamesHumanReadablePooledVectors["first"],     "w")      
        FileHumanReadablePooledLastVectors        = open(FileNamesHumanReadablePooledVectors["last"],      "w")      
        FileHumanReadablePooledPredictedVectors   = open(FileNamesHumanReadablePooledVectors["predicted"], "w")  
        FileHumanReadablePooledAllVectors         = open(FileNamesHumanReadablePooledVectors["all"],       "w")  
        FileHumanReadablePooledAllWeightedVectors = open(FileNamesHumanReadablePooledVectors["all_weighted"], "w")           
        FileHumanReadableReceivedAttention        = open(FileNamesHumanReadablePooledVectors["attention_received"], "w")          
        FileHumanReadableAllReceivedAttention     = open(FileNamesHumanReadablePooledVectors["all_attention_received"], "w")     
        
        
        modelHash = Model.GetHashFromModel(model)
        LogfileAnalyzed.write("Model hash: " + modelHash + "\n")
        LogfileCompared.write("Model hash: " + modelHash + "\n")
        
        FileHumanReadablePooledFirstVectors.write("Model hash: " + modelHash + "\n")
        FileHumanReadablePooledLastVectors.write("Model hash: " + modelHash + "\n")
        FileHumanReadablePooledPredictedVectors.write("Model hash: " + modelHash + "\n")
        FileHumanReadablePooledAllVectors.write("Model hash: " + modelHash + "\n")
        FileHumanReadablePooledAllWeightedVectors.write("Model hash: " + modelHash + "\n")
        FileHumanReadableReceivedAttention.write("Model hash: " + modelHash + "\n")
        FileHumanReadableAllReceivedAttention.write("Model hash: " + modelHash + "\n")       
        
        one_step_model = ModelClass.OneStep(model, DataTypeForTensors, seq_length, itemList, token_from_idList, id_from_tokenList, temperature=temperature)
       
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
                
        hiddenVectorSize = one_step_model.CalculateHiddenVectorSize()
          
        binaryFileHandlerFirstLogits          = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["first"], sequence_length=window_size, data_type=DataTypeForTensors, numbers_per_line=hiddenVectorSize, modelHash=modelHash)  
        binaryFileHandlerLastLogits           = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["last"],  sequence_length=window_size, data_type=DataTypeForTensors, numbers_per_line=hiddenVectorSize, modelHash=modelHash)        
        binaryFileHandlerPredictedLogits      = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["predicted"], sequence_length=window_size, data_type=DataTypeForTensors, numbers_per_line=hiddenVectorSize, modelHash=modelHash)  
        binaryFileHandlerAllLogits            = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["all"],   sequence_length=window_size, data_type=DataTypeForTensors, numbers_per_line=hiddenVectorSize, modelHash=modelHash)   
        binaryFileHandlerReceivedAttention    = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["attention_received"],   sequence_length=window_size, data_type=DataTypeForTensors, numbers_per_line=window_size, modelHash=modelHash) 
        binaryFileHandlerAllReceivedAttention = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["all_attention_received"],   sequence_length=window_size, data_type=DataTypeForTensors, numbers_per_line=window_size, modelHash=modelHash) 
      
        
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
            attention_receivedVectorDataList= []
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
                        attention_receivedVectorDataList.append([])
                        firstLogitsVectorDataList.append([])
                        lastLogitsVectorDataList.append([])
                        predictedLogitsVectorsDataList.append([])
                    else:
                        allLogitsVectorDataList.append([])
                        attention_receivedVectorDataList.append([])
                        firstLogitsVectorDataList.append([])
                        lastLogitsVectorDataList.append([])
                        predictedLogitsVectorsDataList.append([])
                        
                        nextPredictionsList.append([])
                        probabilityOfPredictionsList.append([])
                        probabilityOfActualsList.append([])
                        crossEntropyOfActualsList.append([])
                else:

                    if (onlyLogLogitsVector):
                        _, pooledLogitsVectorData = one_step_model.generate_hidded_states(idList[threadId])
                        
                        attention_receivedList = pooledLogitsVectorData['attention_received'].numpy()
                        allLogitsVectorsList = pooledLogitsVectorData['hidden_state_all_flattened'].numpy()
                        firstLogitsVectorsList = pooledLogitsVectorData['hidden_state_first'].numpy()   
                        lastLogitsVectorsList = pooledLogitsVectorData['hidden_state_last'].numpy()
                        predictedLogitsVectorsList = pooledLogitsVectorData['hidden_state_prediction'].numpy()  
                        
                        attention_receivedVectorDataList.append(attention_receivedList)
                        allLogitsVectorDataList.append(allLogitsVectorsList)
                        firstLogitsVectorDataList.append(firstLogitsVectorsList)
                        lastLogitsVectorDataList.append(lastLogitsVectorsList)
                        predictedLogitsVectorsDataList.append(predictedLogitsVectorsList)
                        
                        
                        nextPredictionsList.append([])
                        probabilityOfPredictionsList.append([])
                        probabilityOfActualsList.append([])
                        crossEntropyOfActualsList.append([])
                        
                    else:
                        predictions, probabilityOfPredictions, probabilityOfActuals, crossEntropyOfActuals, pooledLogitsVectorData = one_step_model.generate_steps_optimized(idList[threadId], nextIdList[threadId], window_size=window_size, DataTypeForTensors=DataTypeForTensors)
                        
                        attention_receivedList = pooledLogitsVectorData['attention_received'].numpy()
                        allLogitsVectorsList = pooledLogitsVectorData['hidden_state_all_flattened'].numpy()
                        firstLogitsVectorsList = pooledLogitsVectorData['hidden_state_first'].numpy()   
                        lastLogitsVectorsList = pooledLogitsVectorData['hidden_state_last'].numpy() 
                        predictedLogitsVectorsList = pooledLogitsVectorData['hidden_state_prediction'].numpy()
                        
                        attention_receivedVectorDataList.append(attention_receivedList)
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
                                                                            None,
                                                                            window_size,
                                                                            numberOfAnalyzedLines,
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
                                                    attention_receivedVectorDataList[threadId],
                                                    lastLogitsVectorDataList[threadId], 
                                                    window_size,
                                                    numberOfAnalyzedLines,
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
                HumanReadableTextOutput, pooledHumanReadableLogitsLastVectorlinesToWrite, pooledHumanReadableLogitsAllVectorlinesToWrite, attention_receivedVectorlinesToWrite, all_attention_receivedVectorlinesToWrite = outputQueue.get(timeout=queueTimeout)
                
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
                
                    
                    try:
                        for _, lineToWrite in enumerate(attention_receivedVectorDataList[threadId]):   # this list is a flattened list of allLogitsVectorDataList. Each element is a timeslice of the sequence. Then the next seqenence and so on.
                            binaryFileHandlerReceivedAttention.append(lineToWrite)   
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
                
                if (attention_receivedVectorlinesToWrite is not None):
                    for lineToWrite in attention_receivedVectorlinesToWrite:
                        FileHumanReadableReceivedAttention.write(lineToWrite)

                if (all_attention_receivedVectorlinesToWrite is not None):
                    for lineToWrite in all_attention_receivedVectorlinesToWrite:
                        FileHumanReadableAllReceivedAttention.write(lineToWrite)
            
            
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
        binaryFileHandlerReceivedAttention.close()
        binaryFileHandlerAllReceivedAttention.close()
        
        FileHumanReadablePooledLastVectors.close()
        FileHumanReadablePooledFirstVectors.close()
        FileHumanReadablePooledPredictedVectors.close()
        FileHumanReadablePooledAllVectors.close()
        
        
        LogfileToAnalyze.close()
        LogfileAnalyzed.close()   

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