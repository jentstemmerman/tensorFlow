import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import LogFileIterator
import Dataset
import os

def GetDataSetInformation(dataSet):
    # Count the number of elements in the dataset
    dataSetTemp         = dataSet
    num_batches_dataset = dataSetTemp.reduce(0, lambda x, _: x + 1).numpy()

    return num_batches_dataset


def CreateDataSetOfMarker(sample, markervalue, DataTypeForTensors):
    input_shape = sample[0].shape
    
    new_shape = tf.concat([[input_shape[0] + 1], input_shape[1:]], axis=0) # maak een shape die 1 meer is in de eerste dimensie
    mask   = tf.fill(new_shape, markervalue)
    mask   = tf.cast(mask, DataTypeForTensors)
    print(mask)
    datasetfast = tf.data.Dataset.from_tensor_slices(mask)
    datasetfast = datasetfast.window(input_shape[0] + 1, shift=1, drop_remainder=True)
    datasetfast = datasetfast.flat_map(lambda window2: window2)
    datasetfast = datasetfast.batch(input_shape[0] + 1, drop_remainder = True)
    datasetfast = datasetfast.map(TokenParser.split_input_target) 
    new_sample = next(iter(datasetfast))
    print(new_sample)
    return datasetfast

def SetMarkerAtDataSetEdge(dataset, maskvalue, DataTypeForTensors):

    sample1 = next(iter(dataset))
    datasetOfMarker = CreateDataSetOfMarker(sample1, maskvalue, DataTypeForTensors)

    combined_dataset = dataset.concatenate(datasetOfMarker)

    return combined_dataset

def CreateDatasetFromDirectoryNew(InputDirectory, OutputDirectory, itemList, id_from_tokenList, mergeDatasets, stride, seq_length, DataTypeForTensors, ReductionInSamples=None, removeDuplicates=False, num_parallel_calls=10, verbose=False):
    FileIterator = LogFileIterator.LogfileIterator(InputDirectory)
    InputFile   = FileIterator.get_first_tokenlogfile()
    ignoreSampleValue = 1000000

    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=10) as executor:
        InputFile = FileIterator.get_first_tokenlogfile()
        futureList = []
        while True:
            
            if InputFile is None:
                break
            future = executor.submit(process_file, InputFile, OutputDirectory, itemList, id_from_tokenList, stride, seq_length, DataTypeForTensors, ignoreSampleValue)
            futureList.append(future)
            InputFile = FileIterator.get_next_tokenlogfile()
        
        concurrent.futures.wait(futureList, return_when=concurrent.futures.ALL_COMPLETED)
        data_dir = OutputDirectory

        if (mergeDatasets):
            # File pattern to match dataset files
            file_pattern = '*.dataset'  # Adjust the pattern according to your file format
            
            import glob
            # Get a list of dataset files
            file_list = glob.glob(f'{data_dir}/{file_pattern}')
            concatenated_dataset = None

            for fileName in file_list:
                dataset = LoadDataset(fileName)

                if (concatenated_dataset is None):
                    concatenated_dataset = dataset
                else:
                    concatenated_dataset = MergeDatasets(concatenated_dataset, dataset)

            Dataset.SaveDataset(concatenated_dataset, OutputDirectory + os.path.sep + 'DirectoryDataSet' + os.path.sep+ 'DataSetOfDirectory.dataset')

        for completed_future in concurrent.futures.as_completed(futureList):
            # Optional: Retrieve the result of each completed future
            result = completed_future.result()

        
def process_file(InputFile, OutputDirectory, itemList, id_from_tokenList, stride, seq_length, DataTypeForTensors, ignoreSampleValue):
    print("process_file " + InputFile + "\n")
    IntermediateFile = None #OutputDirectory + os.path.sep + os.path.basename(InputFile) + ".intermediate"
    outputDatasetFile = OutputDirectory + os.path.sep + os.path.basename(InputFile) + ".dataset"

    dataset = CreateDatasetFromFile(InputFile, 
                                    IntermediateFile, 
                                    itemList, 
                                    id_from_tokenList, 
                                    stride, 
                                    seq_length, 
                                    DataTypeForTensors, 
                                    ReductionInSamples=None, 
                                    removeDuplicates=False, 
                                    num_parallel_calls=10, 
                                    verbose=False)
    
    dataset = SetMarkerAtDataSetEdge(dataset, ignoreSampleValue, DataTypeForTensors)
    Dataset.SaveDataset(dataset, outputDatasetFile.strip())
    print("Dataset " + os.path.basename(InputFile) + " saved.\n")




# This function will not create an intermediate tokenfile
def CreateDatasetFromFile(InputFile, IntermediateFile, itemList, id_from_tokenList, stride, seq_length, DataTypeForTensors, ReductionInSamples=None, removeDuplicates=False, num_parallel_calls=10, verbose=False):

    print("Creating a dataset from  " + InputFile + ".\n")
    
    if(IntermediateFile is not None):
        print(" Intermediate file: " + IntermediateFile + ".\n")

    tokenListList, _, numberOfLines = TokenParser.process_file_to_tokenFile(itemList, InputFile, IntermediateFile)
    print(" Number Of Lines in the file: " + str(numberOfLines) + "\n" )
    
    TensorOfThisFile = []
    
    for index, id_from_token in enumerate(id_from_tokenList):
        
        column = [inner_list[index] for inner_list in tokenListList]
        column = np.array(column)
        
        if (id_from_token is not None):
            TransformedItemTensor = id_from_token(column)  # next todo
        else:
            TransformedItemTensor = tf.convert_to_tensor(column)
        
        TensorOfThisFile.append(TransformedItemTensor)
    
    TensorOfThisFile = tf.stack(TensorOfThisFile, axis=-1)
    
    datasetfast = tf.data.Dataset.from_tensor_slices(TensorOfThisFile)
    
    try:
        datasetfast = datasetfast.window(seq_length + 1, shift=stride, drop_remainder=True)
    except TypeError as e:
        print("Error: {e}")
        print(InputFile  + "\n")

    except OSError as e:
        print(f"Error: {e}")
        print(InputFile  + "\n")

    datasetfast = datasetfast.flat_map(lambda window2: window2)
    datasetfast = datasetfast.batch(seq_length + 1, drop_remainder = True, num_parallel_calls = num_parallel_calls)
    
    if (removeDuplicates):
        datasetfast = datasetfast.unique()

    if (verbose):
        PrintDataset(datasetfast, 10)

    datasetfast = datasetfast.map(lambda x: tf.cast(x, DataTypeForTensors))
    datasetfast = datasetfast.map(TokenParser.split_input_target) 
       
    return datasetfast

def CreateDatasetFromDirectory(InputDirectory, OutputDirectory, itemFile, id_from_tokenList, batch_size, stride, shuffle, BUFFER_SIZE, seq_length, DataTypeForTensors, ReductionInSamples=None, removeDuplicates=False, num_parallel_calls=10, verbose=False):
    print("Creating Item list:\n")
    itemList = TokenParser.GetItemListToSearchFor(itemFile)
    print("Processing log files to tokens:\n")
    tokenListListList = TokenParser.process_log_files_to_tokens_recursive(InputDirectory, OutputDirectory, itemList)
    
    ################# Create a dataset for training and throw all files that are in the training output dir in it
    TraininglogFileIterator = LogFileIterator.LogfileIterator(OutputDirectory)
    dataset                 = None
    filesUsed               = []
    TrainingLogTensor       = None
    tokenLogfileToProcess   = TraininglogFileIterator.get_first_tokenlogfile()

    for tokenListList in tokenListListList:

        numberOfTokens = len([inner_list[0] for inner_list in tokenListList])
        print("Adding " + tokenLogfileToProcess + " to the dataset. Number Of tokens: " + str(numberOfTokens))
        filesUsed.append(tokenLogfileToProcess)
        
        TensorOfThisFile = []

        for index, id_from_token in enumerate(id_from_tokenList):
           
            column = [inner_list[index] for inner_list in tokenListList]
            column = np.array(column)
            if (id_from_token is not None):
                TransformedItemTensor = id_from_token(column)  # next todo
            else:
                TransformedItemTensor = tf.convert_to_tensor(column)

            TensorOfThisFile.append(TransformedItemTensor)
        
        #print("Number of items from file " + tokenLogfileToProcess+ " added " +  str(index) + "/" + str(numberOfTokens))

        TensorOfThisFile = tf.stack(TensorOfThisFile, axis=-1)

        if (TrainingLogTensor is None):
            TrainingLogTensor = TensorOfThisFile
        else:
            TrainingLogTensor = np.concatenate((TrainingLogTensor, TensorOfThisFile), axis=0)

        tokenLogfileToProcess = TraininglogFileIterator.get_next_tokenlogfile()
 

        
        
    datasetfast = tf.data.Dataset.from_tensor_slices(TrainingLogTensor)
    datasetfast = datasetfast.window(seq_length + 1, shift=stride, drop_remainder=True)
    datasetfast = datasetfast.flat_map(lambda window2: window2)
    datasetfast = datasetfast.batch(seq_length + 1, drop_remainder = True, num_parallel_calls = num_parallel_calls)
    
    if (removeDuplicates):
        datasetfast = datasetfast.unique()

    if (verbose):
        PrintDataset(datasetfast, 10)

    datasetfast = datasetfast.map(lambda x: tf.cast(x, DataTypeForTensors))
    datasetfast = datasetfast.map(TokenParser.split_input_target) 
    
    if (shuffle):
        datasetfast = datasetfast.shuffle(BUFFER_SIZE)

    datasetfast = datasetfast.batch(batch_size, drop_remainder=True)
    datasetfast = datasetfast.prefetch(tf.data.experimental.AUTOTUNE)

    if (verbose):
        PrintDatasetTuple(datasetfast)

        

    dataset = datasetfast

    return dataset, filesUsed


def PrintDataset(dataSet, count = 0):
    if (dataSet is not None):
        i = 0
        print ("PrintDataset \n")
        
        for element in dataSet:
            if (i < count):
                print(element.numpy())
            i += 1

        print ("number of elements ", i, "\n")


def PrintDatasetTuple(dataSet, count = 0):
    if (dataSet is not None):
        numberOfElements = 0
        print ("PrintDatasetTuple \n")

        for element in dataSet:
            if (numberOfElements < count):
                print("element: ", str(numberOfElements))
                print(element[0].numpy(), element[1].numpy())
            numberOfElements += 1

        print ("number of elements ", numberOfElements, "\n")
        return numberOfElements

def CompareDatasets(dataSet1, dataSet2):
    def compare_elements(x, y):
        return tf.math.equal(x, y)

    compared_dataset = tf.data.Dataset.zip((dataSet1, dataSet2)).map(compare_elements)

    i = 0
    
    for element in compared_dataset:
        print(element.numpy())
        i += 1

    print ("number of elements ", i, "\n")


def SaveDataset(dataset, fileName):
    dataset.save(fileName, compression="GZIP")
    
def LoadDataset(fileName):
    dataset = tf.data.Dataset.load(fileName)
    return dataset

def MergeDatasets(dataSet1, dataSet2):
    datasetMerged = dataSet1.concatenate(dataSet2)
    return datasetMerged

def GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory, itemList, OutputVocabularyDirectory=None):

    id_from_tokenList   = []
    token_from_idList   = []
    itemIndex = 0
    
    for itemsInLine in itemList:
        # only create encoder-decoder for items that are cathegorical.
        if (itemsInLine["Categorical"] == "True"):
            if ("VocabularyFile" in itemsInLine):
                OriginalTokenFile = itemsInLine["VocabularyFile"]
                
                if (type(OriginalTokenFile) == str):

                    OriginalTokenFileName = OriginalTokenFileDirectory + os.path.sep + OriginalTokenFile
                    
                    # only do one item process at a time
                    if (OutputVocabularyDirectory is not None):
                        OutputVocabularyFile = OutputVocabularyDirectory + os.path.sep + "VocabularyFileItem_" + str(itemIndex)

                        tokenListList, _, _ = TokenParser.process_file_to_tokenFile([itemsInLine], OriginalTokenFileName, OutputVocabularyFile)
                    else:
                        tokenListList, _, _ = TokenParser.process_file_to_tokenFile([itemsInLine], OriginalTokenFileName, None)


                    VocabularyNumpy = [inner_list[0] for inner_list in tokenListList]
                    VocabularyNumpy = np.array(VocabularyNumpy)
                else:
                    raise ValueError("VocabularyFile " + OutputVocabularyFile + " not found")
            elif ("RangeMin" in itemsInLine) and ("RangeMax" in itemsInLine):
                rangeMin = str(itemsInLine["RangeMin"])
                RangeMax = str(itemsInLine["RangeMax"])
                
                if ("Numerical" in itemsInLine):
                    Type = itemsInLine["Numerical"]
                else:
                    Type = "Dec"

                if (Type == "Hex"):
                    rangeMin = int(rangeMin, 16)
                    RangeMax = int(RangeMax, 16)
                else:
                    rangeMin = int(rangeMin, 10)
                    RangeMax = int(RangeMax, 10)

                VocabularyNumpy = np.arange(rangeMin, RangeMax + 1, 1)
                VocabularyNumpy = np.vectorize(hex)(VocabularyNumpy)
                VocabularyNumpy = VocabularyNumpy.astype(str)
                VocabularyNumpy = np.vectorize(lambda x: x[2:])(VocabularyNumpy)  # haal de 0x weg
            else:
                raise ValueError("VocabularyFile " + OutputVocabularyFile + " or range not found")
            
            VocabularyTensor  = tf.convert_to_tensor(VocabularyNumpy)

            if ("MaskValue" in itemsInLine):
                mask_token = str(itemsInLine["MaskValue"])
            else:
                mask_token = None

            # encoder
            id_from_token = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=mask_token)
            id_from_tokenList.append(id_from_token)
            
            # decoder
            token_from_id = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=mask_token, invert=True)
            token_from_idList.append(token_from_id)
        else:
            id_from_tokenList.append(None)
            token_from_idList.append(None)

        itemIndex += 1    
    return id_from_tokenList, token_from_idList