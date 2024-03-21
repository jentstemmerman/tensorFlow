import DefaultTrainingSettings
import os
import Model
import tensorflow as tf
import worker
import TokenParser
import Dataset

def AnalyzeFilesInDirectory(modelFile, InputDirectory, outputDirectory, OriginalTokenFileDirectory, itemFile, seq_length, DataTypeForTensors, temperature):
    with tf.device('/CPU:0'):
        numberOfThreads = 7  # 1 for the main process
        print("Calling CreateWorkers numberOfThreads: ", numberOfThreads)

        itemList                             = TokenParser.GetItemListToSearchFor(itemFile)
        id_from_tokenList, token_from_idList = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList) 
        modelFeatureList                     = Model.CreateModelFeatureListFromItemList(itemList, id_from_tokenList)

        maskValueList                         = Model.GetMaskValueList(id_from_tokenList)


        modelSource = {"Source": "FromFile", "ModelDirectory": modelFile}
        model       = Model.GetModel(modelSource, maskValueList, modelFeatureList, TrainingsParameter)

            
        Workers = worker.CreateWorkers(numberOfThreads, id_from_tokenList)
    
        if (Workers == None):
            print("Could not start workers. END")
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
        modelFile                          = parameters[0]
        itemFile                           = parameters[1]
        InputDirectory                     = parameters[2]
        outputDirectory                    = parameters[3]
        OriginalTokenFileDirectory         = parameters[4]
        temperature                        = int(parameters[5])
    else:
        print("Did not receive 6 parameters. Parameters received: ", len(parameters))
       
    TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
    
    AnalyzeFilesInDirectory(modelFile, InputDirectory, outputDirectory, OriginalTokenFileDirectory, itemFile, TrainingsParameter["Model"]["seq_length"] , TrainingsParameter["DataTypeForTensors"], temperature)  