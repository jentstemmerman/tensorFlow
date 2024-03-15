import DefaultTrainingSettings
import os
import Model
import tensorflow as tf
import worker
import TokenParser
import Dataset

def AnalyzeFilesInDirectory(model, InputDirectory, outputDirectory, id_from_tokenList, token_from_idList, itemFile, seq_length, DataTypeForTensors, temperature):

    numberOfThreads = 7  # 1 for the main process
    print("Calling CreateWorkers numberOfThreads: ", numberOfThreads)

    itemList                             = TokenParser.GetItemListToSearchFor(itemFile)
    
    id_from_tokenList, token_from_idList = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList) 

    with tf.device('/CPU:0'):
        
        Workers = worker.CreateWorkers(numberOfThreads, id_from_tokenList)
    
        if (Workers == None):
            print("Could not start workers")
        else:
            Model.AnalyzePrintfFiles(   model, 
                                        inputDirectory=InputDirectory, 
                                        outputDirectory=outputDirectory, 
                                        itemFile=itemFile, 
                                        temperature=temperature, 
                                        seq_length=seq_length, 
                                        id_from_tokenList = id_from_tokenList,
                                        token_from_idList=token_from_idList,
                                        numberOfThreads=numberOfThreads, 
                                        Workers=Workers,
                                        DataTypeForTensors=DataTypeForTensors )
            
            worker.KillWorkers(Workers)
    


if __name__ == "__main__":
    import sys
    print("Analyzing files in directory...\n")
    parameters = sys.argv[1:]

    if (len(parameters)==6):
        InputDirectory                     = parameters[0]
        itemFile                           = parameters[1]
        IntermediateDirectory              = parameters[2]
        OriginalTokenFileDirectory         = parameters[3]
        Stride                             = int(parameters[4])
        outputFile                         = parameters[5]
    else:
        print("Did not receive 5 parameters. Parameters received: ", len(parameters))

       
    TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
    
    if (Stride is not None):
        TrainingsParameter["stride_dataset"] = Stride

    CreateTrainingDatasetFromDir(InputDirectory=InputDirectory, IntermediateDirectory=IntermediateDirectory, mergeDatasets=True, itemFile=itemFile, OriginalTokenFileDirectory=OriginalTokenFileDirectory, TrainingsParameter=TrainingsParameter)
  