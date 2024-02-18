import TokenParser
import DefaultTrainingSettings
import Dataset
import os

def CreateTrainingDataset(Name, InputDirectory, IntermediateDirectory, itemFile, OriginalTokenFileDirectory, TrainingsParameter=None):


    OutputVocabularyDirectory = IntermediateDirectory + os.path.sep + "Vocabularies"
    
    itemList         = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, token_from_idList = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, itemList=itemList, OutputVocabularyDirectory=OutputVocabularyDirectory)  
 
    datasetForVerification, verificationFileList = Dataset.CreateDatasetFromDirectory(  InputDirectory=InputDirectory, 
                                                                                        OutputDirectory=IntermediateDirectory, 
                                                                                        itemFile=itemFile,
                                                                                        id_from_tokenList=id_from_tokenList,
                                                                                        batch_size=TrainingsParameter["BatchSize"], 
                                                                                        stride=256, #TrainingsParameter["stride_dataset"], 
                                                                                        shuffle=TrainingsParameter["shuffleTraining"], 
                                                                                        BUFFER_SIZE=TrainingsParameter["BUFFER_SIZE"],
                                                                                        seq_length=TrainingsParameter["Model"]["seq_length"],
                                                                                        DataTypeForTensors = TrainingsParameter["DataTypeForTensors"], 
                                                                                        removeDuplicates=TrainingsParameter["removeDuplicates"], 
                                                                                        num_parallel_calls = TrainingsParameter["num_parallel_calls"],
                                                                                        verbose = False)   
    dataSetFile = IntermediateDirectory + os.path.sep + Name + ".dataset"
    Dataset.SaveDataset(datasetForVerification, dataSetFile.strip())
    print("Dataset saved to " + dataSetFile)


if __name__ == "__main__":
    import sys
    print("Creating Dataset from directory...\n")
    parameters = sys.argv[2:]

    if (len(parameters)==5):
        InputDirectory                     = parameters[0]
        itemFile                           = parameters[1]
        IntermediateDirectory              = parameters[2]
        OriginalTokenFileDirectory         = parameters[3]
        Stride                             = parameters[4]
    else:
        print("Did not receive 5 parameters. Parameters received: ", len(parameters))

       
    TrainingsParameter = DefaultTrainingSettings.TrainingsParameter
    
    if (Stride is None):
        TrainingsParameter["stride_dataset"] = Stride

    CreateTrainingDataset(Name, InputDirectory, IntermediateDirectory, itemFile, OriginalTokenFileDirectory, TrainingsParameter=TrainingsParameter)
