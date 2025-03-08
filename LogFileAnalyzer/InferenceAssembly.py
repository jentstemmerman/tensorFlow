import multiprocessing
import numpy as np
import copy
import tensorflow as tf
import time
import StreamingDataSet

with tf.device('/CPU:0'):
    def GetNextTokenWindow(oldWindow, newToken):
        newWindow = oldWindow[1:]  
        newWindowr = np.append(newWindow, newToken)
        #newWindow.append(newToken)
        return newWindow

    def GetNextTokenIdWindow(oldTokenIdWindow, newTokenId, numberOfFeatures):
        
        nextTokenIdWindow = oldTokenIdWindow[1:]
        newTokenId = tf.reshape(newTokenId, [1, numberOfFeatures])  # maak compatibel met concat van nextTokenIdWindow
        nextTokenIdWindow = tf.concat([nextTokenIdWindow, newTokenId], axis=0)

        return nextTokenIdWindow

    def GetFirstSequenceData(tokenList, id_from_tokenList, window_size, DataTypeForTensors, threadId, numberOfParallelExecutions):

        if (tokenList is not None):
            try:
                tokenWindow       = tokenList[0:window_size]
                outputToken       = tokenList[window_size]
            except IndexError:
                print(f"IndexError: threadId: {threadId} numberOfParallelExecutions {numberOfParallelExecutions}")     
                print(f"IndexError: len(tokenList) = {len(tokenList)} ")
                exit()

            tokenIdWindow = copy.deepcopy(tokenWindow)
            outputTokenId = copy.deepcopy(outputToken)

            column = 0
            for id_from_token in id_from_tokenList:
                if (id_from_token is not None):
                    for tokenInWindow in tokenIdWindow:
                        tokenInWindow[column] = id_from_token(tokenInWindow[column])
                                                  
                    outputTokenId[column] = id_from_token(outputToken[column])
                column += 1  

            tokenIdWindow = tf.convert_to_tensor(tokenIdWindow, dtype=DataTypeForTensors)
            outputTokenId = tf.convert_to_tensor(outputTokenId, dtype=DataTypeForTensors)
        else:
            tokenWindow = None 
            outputToken = None 
            tokenIdWindow = None  
            outputTokenId = None 

        return tokenWindow, outputToken, tokenIdWindow, outputTokenId

    def GetNextSequenceData(newToken, id_from_tokenList, prevWindow, prevInputTokens, prevOutputToken, prevOutputTokenId, DataTypeForTensors):
        
        newWindow = GetNextTokenWindow(prevWindow, prevOutputToken)
        newTokenId = copy.deepcopy(newToken)

        column = 0
        for id_from_token in id_from_tokenList:
            if (id_from_token is not None):
                if (type(newToken[column]) == str):
                    newTokenId[column] = id_from_token(newToken[column])
                else:
                    exit()

            column += 1

        newTokenId          = tf.convert_to_tensor(newTokenId, dtype=DataTypeForTensors)
        nextTokenIdWindow   = GetNextTokenIdWindow(prevInputTokens, prevOutputTokenId, column)

        return newWindow, newToken, newTokenId, nextTokenIdWindow

    datasetStreamReader = None
    datasetStreamerInterator = None

    def AssembleInputsThread(workerParameter,
                             chardId,
                             dataSetFile, 
                             itemList, 
                             DataTypeForTensors, 
                             seq_length, 
                             stride, 
                             numberOfShards, 
                             indexOfFirstPrediction, 
                             numberOfParallelExecutions, 
                             origPrintfList, 
                             window_size):
        
        with tf.device('/CPU:0'):
            printfList   = []
            assert (indexOfFirstPrediction >= window_size)
            
            global datasetStreamReader
            global datasetStreamerInterator            
            
            if (datasetStreamerInterator is None):
                datasetStreamReader = StreamingDataSet.DatasetStreamReader(dataSetFile=dataSetFile, itemList=itemList, DataTypeForTensors=DataTypeForTensors, seq_length=seq_length, stride=stride, numberOfShards=numberOfShards)
                datasetStreamerInterator = datasetStreamReader.GetStreamingIterator(shard_index=chardId)  
            else:
                datasetStreamerInterator = SkipInferenceDatasetStreamer(seq_length, datasetStreamerInterator)
            
            datasetStreamerInterator, idInnerList, nextIdsInnerList = GetInferenceDataFromDatasetStreamer(numberOfParallelExecutions, datasetStreamerInterator)
                        
            if (origPrintfList is not None):
                printfList = origPrintfList[indexOfFirstPrediction:indexOfFirstPrediction + numberOfParallelExecutions]
                
                       
            return printfList, idInnerList, nextIdsInnerList
            
        
        
    def SkipInferenceDatasetStreamer(numberOfSamples, datasetStreamerInterator):
        with tf.device('/CPU:0'):
            
            for _ in range(numberOfSamples):
                next(datasetStreamerInterator)
                
            return datasetStreamerInterator

    #@tf.function   
    def GetInferenceDataFromDatasetStreamer(numberOfSamples, datasetStreamerInterator):
        with tf.device('/CPU:0'):
            idList, nextIdsList = [], []

            for _ in range(numberOfSamples):
                ids, nextIds = next(datasetStreamerInterator)
                idList.append(ids)
                nextIdsList.append(nextIds)

            idList = tf.stack(idList, axis=0)
            nextIdsList = tf.stack(nextIdsList, axis=0)

            return datasetStreamerInterator, idList, nextIdsList
            

