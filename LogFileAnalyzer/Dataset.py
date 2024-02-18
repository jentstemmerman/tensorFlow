import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import LogFileIterator
import os

def GetDataSetInformation(dataSet):
    # Count the number of elements in the dataset
    dataSetTemp         = dataSet
    num_batches_dataset = dataSetTemp.reduce(0, lambda x, _: x + 1).numpy()

    return num_batches_dataset

def SetMarkerAtDataSetEdge(dataset, markervalue, DataTypeForTensors):
    sample1 = next(iter(dataset))
    print(sample1)
    newSample = []
    
    for column in sample1:
        mask   = tf.fill(tf.shape(column), markervalue)
        mask   = tf.cast(mask, DataTypeForTensors)
        newSample.append(mask)
    
    newSample = tf.stack(newSample, axis=0)
    #newSample2 = [newSample]
    datasetFromNewSample = tf.data.Dataset.from_tensor_slices(newSample)
    sample2 = next(iter(datasetFromNewSample))
    print(sample2)

    combined_dataset = dataset.concatenate(datasetFromNewSample)
    print(newSample)


# This function will not create an intermediate tokenfile
def CreateDatasetFromFile(InputFile, itemList, id_from_tokenList, stride, seq_length, DataTypeForTensors, ReductionInSamples=None, removeDuplicates=False, num_parallel_calls=10, verbose=False):

    fileHandler = open(InputFile, "r", errors='ignore')
                                        
    tokenListList, _, numberOfLines = TokenParser.getLinesAndTokens(itemList, -1, fileHandler, tokenList=None, PrintfList=None)
    
    print("Creating a dataset from  " + InputFile + ". Number Of Lines in the file: " + str(numberOfLines))
    
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
    datasetfast = datasetfast.window(seq_length + 1, shift=stride, drop_remainder=True)
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


def SaveDataset(dataSet, fileName):
    dataSet.save(fileName, compression="GZIP")
    
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

                        tokenListList = TokenParser.process_file_to_tokenFile(OriginalTokenFileName, OutputVocabularyFile, [itemsInLine])
                    else:
                        tokenListList = TokenParser.process_file_to_tokenFile(OriginalTokenFileName, None, [itemsInLine])


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

            # encoder
            id_from_token = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None)
            id_from_tokenList.append(id_from_token)
            
            # decoder
            token_from_id = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None, invert=True)
            token_from_idList.append(token_from_id)
        else:
            id_from_tokenList.append(None)
            token_from_idList.append(None)

        itemIndex += 1    
    return id_from_tokenList, token_from_idList