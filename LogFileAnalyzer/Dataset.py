import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser

def GetDataSetInformation(dataSet):
    # Count the number of elements in the dataset
    dataSetTemp = dataSet
    num_batches_dataset     = dataSetTemp.reduce(0, lambda x, _: x + 1).numpy()

    return num_batches_dataset

def CreateDatasetFromDirectory(InputDirectory, OutputDirectory, id_from_token, batch_size, stride, shuffle, BUFFER_SIZE, seq_length, ReductionInSamples=None, removeDuplicates=False, verbose=False):
    
    TokenParser.process_log_files_to_tokens_recursive(InputDirectory, OutputDirectory)
    
    ################# Create a dataset for training and throw all files that are in the training output dir in it
    TraininglogFileIterator = TokenParser.LogfileIterator(OutputDirectory)
    tokenLogfileToProcess = TraininglogFileIterator.get_first_tokenlogfile()
    dataset = None
    filesUsed = []

    TrainingLogTensor = None

    while (tokenLogfileToProcess is not None):
        logTokenNumpyList = []
        print("Adding " + tokenLogfileToProcess + " to the dataset")
        filesUsed.append(tokenLogfileToProcess)

        logTokenNumpyList = np.loadtxt(tokenLogfileToProcess, delimiter=",", dtype=str)

        TensorOfThisFile = tf.convert_to_tensor(logTokenNumpyList)

        if (TrainingLogTensor is None):
            TrainingLogTensor = TensorOfThisFile
        else:
            TrainingLogTensor = np.concatenate((TrainingLogTensor, TensorOfThisFile), axis=0)

        tokenLogfileToProcess = TraininglogFileIterator.get_next_tokenlogfile()

    ids_from_logfile = id_from_token(TrainingLogTensor)
    #ids_from_logfile = tf.cast(ids_from_logfile, dtype=tf.float32)
    
    #keras.backend.print_tensor(ids_from_logfile)
    #sliced_ids_from_logfile = tf.data.Dataset.from_tensor_slices(ids_from_logfile)  # sliced_ids_from_logfile kan ik niet tonen in de debugger
    # Slide the window over the tensor

    datasetfast = tf.data.Dataset.from_tensor_slices(ids_from_logfile)
    datasetfast = datasetfast.window(seq_length + 1, shift=stride, drop_remainder=True)
    datasetfast = datasetfast.flat_map(lambda window2: window2)
    datasetfast = datasetfast.batch(seq_length+1, drop_remainder = True, num_parallel_calls = 10)
    
    if (removeDuplicates):
        datasetfast = datasetfast.unique()

    if (verbose):
        PrintDataset(datasetfast, 10)

    datasetfast = datasetfast.map(lambda x: tf.cast(x, tf.float32))
    datasetfast = datasetfast.map(TokenParser.split_input_target) 
    
    if (shuffle):
        datasetfast = datasetfast.shuffle(BUFFER_SIZE)

    datasetfast = datasetfast.batch(batch_size, drop_remainder=True)
    datasetfast = datasetfast.prefetch(tf.data.experimental.AUTOTUNE)

    if (verbose):
        PrintDatasetTuple(datasetfast)
    else:
        PrintDatasetTuple(datasetfast, 2)

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