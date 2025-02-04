import DefaultTrainingSettings
import Model
import tensorflow as tf
import worker
import TokenParser
import Dataset

def AnalyzeFile(modelDirectory, inputFile, outputDirectory, OriginalTokenFileDirectory, itemFile, temperature):
    
    tf.config.run_functions_eagerly(True)  
    tf.data.experimental.enable_debug_mode() 
      
    with tf.device('/CPU:0'):
   
        success = False
        numberOfThreads = 1  # 1 for the main process
        print("Calling CreateWorkers numberOfThreads: ", numberOfThreads)

        itemList                             = TokenParser.GetItemListToSearchFor(itemFile)
        id_from_tokenList, token_from_idList = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList) 
        modelFeatureList                     = Model.CreateModelFeatureListFromItemList(itemList, id_from_tokenList)
        maskValueList                        = Model.GetMaskValueList(id_from_tokenList)

        TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
        modelSource = {"Source": "FromFile", "ModelDirectory": modelDirectory, "Network": TrainingsParameter["Model"]["Network"]}
        model       = Model.GetModel(modelSource, maskValueList, modelFeatureList, TrainingsParameter)
            
        Workers = worker.CreateWorkers(numberOfThreads, id_from_tokenList)
    
        if (Workers == None):
            print("Could not start workers. END")
        else:
            
            
            Model.AnalyzePrintfFile(    model, 
                                        inputFile=inputFile, 
                                        outputDirectory=outputDirectory, 
                                        itemFile=itemFile, 
                                        temperature=temperature, 
                                        seq_length=TrainingsParameter["Model"]["seq_length"], 
                                        id_from_tokenList = id_from_tokenList,
                                        token_from_idList=token_from_idList,
                                        numberOfThreads=numberOfThreads, 
                                        Workers=Workers,
                                        DataTypeForTensors=TrainingsParameter["DataTypeForTensors"] )

            success = True
            worker.KillWorkers(Workers)
    
    return success

