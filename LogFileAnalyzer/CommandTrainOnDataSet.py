import TokenParser
import DefaultTrainingSettings
import Dataset
import os
import Model

def TrainModelOnDataSet(TrainingDataSetFile, id_from_tokenList, itemList, batch_size, OutputDirectory, TrainingsParameter=None):
    TrainingCheckpointDir  = OutputDirectory + os.path.sep + "TrainingCheckPoints"
    TrainedModelDir        = OutputDirectory + os.path.sep + "TrainedModel"

    modelFeatureList = Model.CreateModelFeatureListFromItemList(itemList, id_from_tokenList)
    maskValueList    = Model.GetMaskValueList(id_from_tokenList)

    modelSource = {"Source": "NewModel"}

    model = Model.GetModel(modelSource, maskValueList, modelFeatureList, TrainingsParameter)

    datasetForTraining = Dataset.LoadDataset(TrainingDataSetFile.strip())   
    datasetForTraining = Dataset.PrepareDataSetForTraining(datasetForTraining, batch_size, shuffle=False, buffer_size=TrainingsParameter["BUFFER_SIZE"])
    model = Model.TrainModel(model, datasetForTraining, OutputDirectory, TrainingsParameter, TrainingCheckpointDir)
    
    try:
        model.save(TrainedModelDir)
    except:
        model = None

if __name__ == "__main__":
    import sys

    parameters = sys.argv[1:]

    if (len(parameters)==5):
        TrainingDataSetFile           = parameters[0]
        itemFile                      = parameters[1]
        OriginalTokenFileDirectory    = parameters[2]
        batch_size                    = int(parameters[3])
        OutputDirectory               = parameters[4]
    else:
        print("Did not receive 5 parameters. Parameters received: " + str(sys.argv-1))

    TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
    itemList             = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, _ = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList, OutputVocabularyDirectory=None)  

    TrainModelOnDataSet(TrainingDataSetFile, id_from_tokenList, itemList, batch_size, OutputDirectory, TrainingsParameter)
