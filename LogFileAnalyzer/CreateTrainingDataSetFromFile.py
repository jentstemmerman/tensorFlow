import TokenParser
import DefaultTrainingSettings
import Dataset
import os

def CreateTrainingDatasetFromFile(InputFile, itemFile, OriginalTokenFileDirectory, TrainingsParameter=None):

    itemList             = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, _ = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList, OutputVocabularyDirectory=None)  
 

    dataSetOfFile = Dataset.CreateDatasetFromFile(  InputFile, 
                                                    itemList=itemList,
                                                    id_from_tokenList=id_from_tokenList,
                                                    batch_size=TrainingsParameter["BatchSize"], 
                                                    stride=TrainingsParameter["stride_dataset"], 
                                                    shuffle=TrainingsParameter["shuffleTraining"], 
                                                    BUFFER_SIZE=TrainingsParameter["BUFFER_SIZE"],
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

    dataset = CreateTrainingDatasetFromFile(InputFile, itemFile, OriginalTokenFileDirectory, TrainingsParameter)

    Dataset.SaveDataset(dataset, DatasetFile.strip())