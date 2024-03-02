import TokenParser
import DefaultTrainingSettings
import Dataset
import os
import Model

def TrainModelOnDataSet(TrainingDataSetFile, id_from_tokenList, itemList, batch_size, OutputDirectory):
    TrainingCheckpointDir  = OutputDirectory + os.path.sep + "TrainingCheckPoints"
    TrainedModelDir        = OutputDirectory + os.path.sep + "TrainedModel"

    modelFeatureList = Model.CreateModelFeatureListFromItemList(itemList, id_from_tokenList)

    modelSource = {"Source": "NewModel"}
    model = Model.GetModel(modelSource, modelFeatureList, TrainingsParameter)

    datasetForTraining = Dataset.LoadDataset(TrainingDataSetFile.strip())   


    model = Model.TrainModel(model, datasetForTraining, OutputDirectory, TrainingsParameter, TrainingCheckpointDir)
    try:
        model.save(TrainedModelDir)
    except:
        model = None

if __name__ == "__main__":
    import sys

    parameters = sys.argv[1:]

    if (len(parameters)==5):
        InputFile                     = parameters[0]
        itemFile                      = parameters[1]
        OriginalTokenFileDirectory    = parameters[2]
        Stride                        = parameters[3]
        DatasetFile                   = parameters[4]
    else:
        print("Did not receive 5 parameters. Parameters received: " + str(sys.argv-1))

    TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
    
    if (Stride is not None):
        TrainingsParameter["stride_dataset"] = Stride

    datasetFromFile = CreateTrainingDatasetFromFile(InputFile, itemFile, OriginalTokenFileDirectory, TrainingsParameter)

    Dataset.SetMarkerAtDataSetEdge(datasetFromFile, 1, TrainingsParameter["DataTypeForTensors"])

    Dataset.SaveDataset(datasetFromFile, DatasetFile.strip())