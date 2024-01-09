import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import LogFileIterator


def GetDataSetInformation(dataSet):
    # Count the number of elements in the dataset
    dataSetTemp         = dataSet
    num_batches_dataset = dataSetTemp.reduce(0, lambda x, _: x + 1).numpy()

    return num_batches_dataset

def CreateDatasetFromDirectory(InputDirectory, OutputDirectory, itemFile, embedding_layer_used, id_from_tokenList, batch_size, stride, shuffle, BUFFER_SIZE, seq_length, DataTypeForTensors, ReductionInSamples=None, removeDuplicates=False, num_parallel_calls=10, verbose=False):
    
    itemList = TokenParser.GetItemListToSearchFor(itemFile)
    tokenListListList = TokenParser.process_log_files_to_tokens_recursive(InputDirectory, OutputDirectory, itemList)
    
    ################# Create a dataset for training and throw all files that are in the training output dir in it
    TraininglogFileIterator = LogFileIterator.LogfileIterator(OutputDirectory)
    dataset                 = None
    filesUsed               = []
    TrainingLogTensor       = None
    tokenLogfileToProcess   = TraininglogFileIterator.get_first_tokenlogfile()

    for tokenListList in tokenListListList:

        print("Adding " + tokenLogfileToProcess + " to the dataset...")
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
    else:
        PrintDatasetTuple(datasetfast, 1)
        

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