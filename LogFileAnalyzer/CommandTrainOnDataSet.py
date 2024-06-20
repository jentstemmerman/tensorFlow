import TokenParser
import DefaultTrainingSettings
import Dataset
import os
import Model
import tensorflow as tf

def TrainModelOnDataSet(TrainingDataSetFile, id_from_tokenList, itemList, TrainingCheckpointDir, TrainedModelDir, TrainingsParameter):

    modelFeatureList = Model.CreateModelFeatureListFromItemList(itemList, id_from_tokenList)
    maskValueList    = Model.GetMaskValueList(id_from_tokenList)

    modelSource = {"Source": "NewModel", "ModelDirectory": TrainedModelDir, "Network": TrainingsParameter["Model"]["Network"]}

    model = Model.GetModel(modelSource, maskValueList, modelFeatureList, TrainingsParameter)
    assert(model is not None)
    
    datasetForTraining = Dataset.LoadDataset(TrainingDataSetFile.strip())   
    datasetForTraining = Dataset.PrepareDataSetForTraining(datasetForTraining, TrainingsParameter["BatchSize"], shuffle=TrainingsParameter["shuffleTraining"], buffer_size=TrainingsParameter["BUFFER_SIZE"])
    model = Model.TrainModel(model, datasetForTraining, TrainingsParameter, TrainingCheckpointDir)
    
    model.save(TrainedModelDir)








if __name__ == "__main__":
    import sys

    parameters = sys.argv[1:]

    if (len(parameters)==5):
        TrainingDataSetFile           = parameters[0]
        itemFile                      = parameters[1]
        OriginalTokenFileDirectory    = parameters[2]
        TrainingCheckpointDir         = parameters[3]
        TrainedModelDir               = parameters[4]
    else:
        print("Did not receive 5 parameters. Parameters received: " + str(sys.argv-1))
    
 #   tf.config.run_functions_eagerly(True)  
 #   tf.data.experimental.enable_debug_mode()    

    TrainingsParameter   = DefaultTrainingSettings.TrainingsParameter
    itemList             = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, _ = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList, OutputVocabularyDirectory=None)  

    TrainModelOnDataSet(TrainingDataSetFile, id_from_tokenList, itemList, TrainingCheckpointDir, TrainedModelDir, TrainingsParameter)
