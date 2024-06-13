import TokenParser
import DefaultTrainingSettings
import Dataset
import os
import Model
import tensorflow as tf

def TrainModelOnDataSet(TrainingDataSetFile, id_from_tokenList, itemList, batch_size, TrainingCheckpointDir, TrainedModelDir, TrainingsParameter):

    modelFeatureList = Model.CreateModelFeatureListFromItemList(itemList, id_from_tokenList)
    maskValueList    = Model.GetMaskValueList(id_from_tokenList)

    modelSource = {"Source": "NewModel", "ModelDirectory": TrainedModelDir, "Network": TrainingsParameter["Model"]["Network"]}

    model = Model.GetModel(modelSource, maskValueList, modelFeatureList, TrainingsParameter)
    assert(model is not None)
    
    datasetForTraining = Dataset.LoadDataset(TrainingDataSetFile.strip())   
    datasetForTraining = Dataset.PrepareDataSetForTraining(datasetForTraining, batch_size, shuffle=False, buffer_size=TrainingsParameter["BUFFER_SIZE"])
    model = Model.TrainModel(model, datasetForTraining, TrainingsParameter, TrainingCheckpointDir)
    
    try:
        model.save(TrainedModelDir)
    except:
        model = None







if __name__ == "__main__":
    import sys

    parameters = sys.argv[1:]

    if (len(parameters)==6):
        TrainingDataSetFile           = parameters[0]
        itemFile                      = parameters[1]
        OriginalTokenFileDirectory    = parameters[2]
        batch_size                    = int(parameters[3])
        TrainingCheckpointDir         = parameters[4]
        TrainedModelDir               = parameters[5]
    else:
        print("Did not receive 6 parameters. Parameters received: " + str(sys.argv-1))
    
 #   tf.config.run_functions_eagerly(True)  
 #   tf.data.experimental.enable_debug_mode()    

    TrainingsParameter   = DefaultTrainingSettings.TrainingsParameter
    itemList             = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, _ = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList, OutputVocabularyDirectory=None)  

    TrainModelOnDataSet(TrainingDataSetFile, id_from_tokenList, itemList, batch_size, TrainingCheckpointDir, TrainedModelDir, TrainingsParameter)
