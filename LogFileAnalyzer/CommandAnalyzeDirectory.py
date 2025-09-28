import DefaultTrainingSettings
from Mask import Mask
import Model
import tensorflow as tf
import worker
import TokenParser
import Dataset
 
def AnalyzeFilesInDirectory(modelFile, InputDirectory, outputDirectory, OriginalTokenFileDirectory, itemFile, TrainingsParameter, temperature):
    with tf.device('/CPU:0'):
        #tf.config.run_functions_eagerly(True)  
        #tf.data.experimental.enable_debug_mode()  
        numberOfThreads = 2  # 1 for the main process

        itemList                             = TokenParser.GetItemListToSearchFor(itemFile)
        id_from_tokenList, token_from_idList = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList) 
        modelFeatureList                     = Model.CreateModelFeatureListFromItemList(itemList, id_from_tokenList)
        maskValueList                        = Mask.GetMaskValueList(id_from_tokenList, itemList)

        modelSource = {"Source": "FromFile", "ModelDirectory": modelFile, "Network": TrainingsParameter["Model"]["Network"]}
        model       = Model.GetModel(modelSource, maskValueList, modelFeatureList, TrainingsParameter)
        model.summary()
            
        Workers = worker.CreateWorkers(numberOfThreads, id_from_tokenList)
    
        if (Workers is None):
            print("Could not start workers. END")
        else:

            Model.AnalyzePrintfFiles(   model=model, 
                                        inputDirectory=InputDirectory, 
                                        outputDirectory=outputDirectory, 
                                        itemFile=itemFile, 
                                        temperature=temperature, 
                                        seq_length=TrainingsParameter["Model"]["seq_length"], 
                                        id_from_tokenList = id_from_tokenList,
                                        token_from_idList=token_from_idList,
                                        numberOfThreads=numberOfThreads, 
                                        Workers=Workers,
                                        DataTypeForTensors=TrainingsParameter["DataTypeForTensors"] )
            
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
        ForTraining                        = parameters[5]
        temperature                        = int(parameters[5])
    else:
        print("Did not receive 6 parameters. Parameters received: ", len(parameters))
       
    TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
        
    #tf.config.run_functions_eagerly(True)  
    #tf.data.experimental.enable_debug_mode()    
   
    AnalyzeFilesInDirectory(modelFile, InputDirectory, outputDirectory, OriginalTokenFileDirectory, itemFile, TrainingsParameter, temperature)  