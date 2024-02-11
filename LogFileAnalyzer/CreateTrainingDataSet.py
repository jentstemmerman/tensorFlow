import TokenParser
import DefaultTrainingSettings
import Dataset
import os

def CreateTrainingDataset(Name, InputDirectory, IntermediateDirectory, itemFile, OriginalTokenFileDirectory, TrainingsParameter=None):

    if (TrainingsParameter is None):
        TrainingsParameter = DefaultTrainingSettings.TrainingsParameter

    OutputVocabularyDirectory = IntermediateDirectory + os.path.sep + "Vocabularies"
    
    itemList         = TokenParser.GetItemListToSearchFor(itemFile)
    id_from_tokenList, token_from_idList = Dataset.GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory=OriginalTokenFileDirectory, OutputVocabularyDirectory=OutputVocabularyDirectory, itemList=itemList)  
 
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


    # Get command line arguments
    Name                  = sys.argv[1] if len(sys.argv) > 1 else None
    InputDirectory        = sys.argv[2] if len(sys.argv) > 2 else None
    IntermediateDirectory = sys.argv[3] if len(sys.argv) > 3 else None
    itemFile              = sys.argv[4] if len(sys.argv) > 4 else None
    OriginalTokenFileDirectory = sys.argv[5] if len(sys.argv) > 5 else None

    print("Creating dataset from files in " + InputDirectory + " and saving to " + Name)

    CreateTrainingDataset(Name, InputDirectory, IntermediateDirectory, itemFile, OriginalTokenFileDirectory, TrainingsParameter=None)
