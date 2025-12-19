from Mask import Mask
import TokenParser
import DefaultTrainingSettings
import Dataset
import os
import Model
import tensorflow as tf

def TrainModelOnDataSet(TrainingDataSetFile, VerificationDataSetFile, id_from_tokenList, itemList, TrainingCheckpointDir, TrainedModelDir, TrainingsParameter):

    #tf.config.run_functions_eagerly(True)  
   # tf.data.experimental.enable_debug_mode()    

    modelFeatureList = Model.CreateModelFeatureListFromItemList(itemList, id_from_tokenList)
    maskValueList    = Mask.GetMaskValueList(id_from_tokenList, itemList)

    modelSource = {"Source": "NewModel", "ModelDirectory": TrainedModelDir, "Network": TrainingsParameter["Model"]["Network"]}

    model = Model.GetModel(modelSource, maskValueList, modelFeatureList, TrainingsParameter)
    assert(model is not None)
    
    datasetForTraining = Dataset.LoadDataset(TrainingDataSetFile.strip())   
    datasetForTraining = Dataset.PrepareDataSetForTraining(datasetForTraining, TrainingsParameter["BatchSize"], shuffle=TrainingsParameter["shuffleTraining"], buffer_size=TrainingsParameter["BUFFER_SIZE"])
    

    if (VerificationDataSetFile != ""):
        if not os.path.exists(VerificationDataSetFile.strip()):
            datasetForVerification = None
        else:
            print("Use for verification: ", VerificationDataSetFile)
            datasetForVerification = Dataset.LoadDataset(VerificationDataSetFile.strip())  
            datasetForVerification = Dataset.PrepareDataSetForTraining(datasetForVerification, TrainingsParameter["BatchSize"], shuffle=TrainingsParameter["shuffleTraining"], buffer_size=TrainingsParameter["BUFFER_SIZE"])

    model = Model.TrainModel(model, datasetForTraining, datasetForVerification, TrainingsParameter, TrainingCheckpointDir)
    
    model.save(TrainedModelDir)

    model.summary() 
    
    for var in model.trainable_variables:
        print(var.name, var.shape)

        
    try:
        model.save(TrainedModelDir)
    except:
        model = None


if __name__ == "__main__":
    import sys

    print(tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPUs:", tf.config.list_physical_devices('GPU'))



    parameters = sys.argv[1:]

    if (len(parameters)==6):
        TrainingDataSetFile           = parameters[0]
        VerificationDataSetFile       = parameters[1]
        itemFile                      = parameters[2]
        OriginalTokenFileDirectory    = parameters[3]
        TrainingCheckpointDir         = parameters[4]
        TrainedModelDir               = parameters[5]
    else:
        print("Did not receive 6 parameters. Parameters received: " + str(sys.argv-1))
    
    TrainingsParameter   = DefaultTrainingSettings.TrainingsParameter
    itemList             = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, _ = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList, OutputVocabularyDirectory=None)  

    TrainModelOnDataSet(TrainingDataSetFile, VerificationDataSetFile, id_from_tokenList, itemList, TrainingCheckpointDir, TrainedModelDir, TrainingsParameter)
