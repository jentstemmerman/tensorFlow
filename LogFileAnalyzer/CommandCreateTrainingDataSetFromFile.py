import TokenParser
import DefaultTrainingSettings
import Dataset
import os

def CreateTrainingDatasetFromFile(InputFile, itemFile, OriginalTokenFileDirectory, TrainingsParameter=None):

    itemList             = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, _ = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList, OutputVocabularyDirectory=None)  
 

    dataSetOfFile = Dataset.CreateDatasetFromFile(  InputFile, 
                                                    None,
                                                    itemList=itemList,
                                                    id_from_tokenList=id_from_tokenList,
                                                    stride=int(TrainingsParameter["stride_dataset"]), 
                                                    seq_length=TrainingsParameter["Model"]["seq_length"],
                                                    DataTypeForTensors = TrainingsParameter["DataTypeForTensors"], 
                                                    removeDuplicates=TrainingsParameter["removeDuplicates"], 
                                                    num_parallel_calls = TrainingsParameter["num_parallel_calls"],
                                                    verbose = False )
    

    return dataSetOfFile


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