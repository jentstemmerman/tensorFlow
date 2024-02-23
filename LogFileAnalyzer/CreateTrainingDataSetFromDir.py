import TokenParser
import DefaultTrainingSettings
import Dataset
import os

def CreateTrainingDatasetFromDir(InputDirectory, IntermediateDirectory, itemFile, OriginalTokenFileDirectory, TrainingsParameter=None):

    OutputVocabularyDirectory = IntermediateDirectory + os.path.sep + "Vocabularies"
    
    itemList         = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, token_from_idList = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList, OutputVocabularyDirectory=OutputVocabularyDirectory)  
 
    dataset, verificationFileList = Dataset.CreateDatasetFromDirectoryNew( InputDirectory=InputDirectory, 
                                                                                        OutputDirectory=IntermediateDirectory, 
                                                                                        itemList=itemList,
                                                                                        id_from_tokenList=id_from_tokenList,
                                                                   
                                                                                        stride=256, #TrainingsParameter["stride_dataset"], 
                                                                                        #shuffle=TrainingsParameter["shuffleTraining"], 
                                                                                        #BUFFER_SIZE=TrainingsParameter["BUFFER_SIZE"],
                                                                                        seq_length=TrainingsParameter["Model"]["seq_length"],
                                                                                        DataTypeForTensors = TrainingsParameter["DataTypeForTensors"], 
                                                                                        removeDuplicates=TrainingsParameter["removeDuplicates"], 
                                                                                        num_parallel_calls = TrainingsParameter["num_parallel_calls"],
                                                                                        verbose = False)   
    
    return dataset



if __name__ == "__main__":
    import sys
    print("Creating Dataset from directory...\n")
    parameters = sys.argv[1:]

    if (len(parameters)==6):
        InputDirectory                     = parameters[0]
        itemFile                           = parameters[1]
        IntermediateDirectory              = parameters[2]
        OriginalTokenFileDirectory         = parameters[3]
        Stride                             = parameters[4]
        outputFile                         = parameters[5]
    else:
        print("Did not receive 5 parameters. Parameters received: ", len(parameters))

       
    TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
    
    if (Stride is not None):
        TrainingsParameter["stride_dataset"] = Stride

    dataset = CreateTrainingDatasetFromDir(InputDirectory, IntermediateDirectory, itemFile, OriginalTokenFileDirectory, TrainingsParameter=TrainingsParameter)
    
    Dataset.SaveDataset(dataset, outputFile.strip())
    print("Dataset saved to " + outputFile)