import TokenParser
import DefaultTrainingSettings
import Dataset
import os

def CreateTrainingDatasetFromDir(InputDirectory, IntermediateDirectory, mergeDatasets, itemFile, OriginalTokenFileDirectory, TrainingsParameter=None):

    OutputVocabularyDirectory = IntermediateDirectory + os.path.sep + "Vocabularies"
    
    itemList             = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, _ = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList, OutputVocabularyDirectory=OutputVocabularyDirectory)  
 
    Dataset.CreateDatasetFromDirectoryNew( InputDirectory=InputDirectory, 
                                           OutputDirectory=IntermediateDirectory, 
                                           itemList=itemList,
                                           id_from_tokenList=id_from_tokenList,
                                           mergeDatasets=mergeDatasets,
                                           stride=TrainingsParameter["stride_dataset"], 
                                           seq_length=TrainingsParameter["Model"]["seq_length"],
                                           DataTypeForTensors = TrainingsParameter["DataTypeForTensors"], 
                                           removeDuplicates=TrainingsParameter["removeDuplicates"], 
                                           num_parallel_calls = TrainingsParameter["num_parallel_calls"],
                                           dual_input = TrainingsParameter["Model"]["Network"] == "Transformer",
                                           verbose = False)   
    


if __name__ == "__main__":
    import sys
    print("Creating Dataset from directory...\n")
    parameters = sys.argv[1:]

    if (len(parameters)==5):
        InputDirectory                     = parameters[0]
        itemFile                           = parameters[1]
        IntermediateDirectory              = parameters[2]
        OriginalTokenFileDirectory         = parameters[3]
        outputFile                         = parameters[4]
    else:
        print("Did not receive 5 parameters. Parameters received: ", len(parameters))

       
    TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
    
    CreateTrainingDatasetFromDir(InputDirectory=InputDirectory, IntermediateDirectory=IntermediateDirectory, mergeDatasets=True, itemFile=itemFile, OriginalTokenFileDirectory=OriginalTokenFileDirectory, TrainingsParameter=TrainingsParameter)
  