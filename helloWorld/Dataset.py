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
    datasetList = []

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

    #keras.backend.print_tensor(ids_from_logfile)
    #sliced_ids_from_logfile = tf.data.Dataset.from_tensor_slices(ids_from_logfile)  # sliced_ids_from_logfile kan ik niet tonen in de debugger
    # Slide the window over the tensor
    EXECUTE_SLOW_PATH = False
    SLOW_PATH = False

    datasetSlow = None
    datasetfast = None

    if (EXECUTE_SLOW_PATH):
        windows = []
        windowsAcc = []
        print(ids_from_logfile.shape[0])
        for i in range(ids_from_logfile.shape[0] - (seq_length + 1) + 1):
            window = ids_from_logfile[i:i+seq_length+1]
            windowsAcc.append(window)

        # Stack the windows into a new tensor
        sequences = tf.stack(windowsAcc, axis=0)
        datasetSlow = tf.data.Dataset.from_tensor_slices(sequences)
        datasetSlow = datasetSlow.batch(batch_size, drop_remainder=True)
        datasetfast = datasetfast.prefetch(tf.data.experimental.AUTOTUNE)

    datasetfast = tf.data.Dataset.from_tensor_slices(ids_from_logfile)
    datasetfast = datasetfast.window(seq_length + 1, shift=stride, drop_remainder=True)
    datasetfast = datasetfast.flat_map(lambda window2: window2)
    datasetfast = datasetfast.batch(seq_length+1, drop_remainder = True, num_parallel_calls = 10)
    
    if (removeDuplicates):
        datasetfast = datasetfast
        datasetfast = datasetfast.unique()

    if (verbose):
        PrintDataset(datasetSlow, 10)
        PrintDataset(datasetfast, 10)

    
    if (EXECUTE_SLOW_PATH):
        datasetSlow = datasetfast.map(TokenParser.split_input_target) 

    datasetfast = datasetfast.map(TokenParser.split_input_target) 
    
    if (shuffle):
        datasetfast = datasetfast.shuffle(BUFFER_SIZE)

    datasetfast = datasetfast.batch(batch_size, drop_remainder=True)
    datasetfast = datasetfast.prefetch(tf.data.experimental.AUTOTUNE)

    if (verbose):
        PrintDatasetTuple(datasetSlow)
        PrintDatasetTuple(datasetfast)
    else:
        PrintDatasetTuple(datasetSlow, 5)
        PrintDatasetTuple(datasetfast, 5)

    if (SLOW_PATH) and (EXECUTE_SLOW_PATH):
        dataset = datasetSlow
    else:
        dataset = datasetfast

    if (verbose) and (SLOW_PATH) and (EXECUTE_SLOW_PATH):
        CompareDatasets(datasetSlow, datasetSlow)

        print("Adding " + InputDirectory + " to the dataset has finished \n")

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
        i = 0
        print ("PrintDatasetTuple \n")

        for element in dataSet:
            if (i < count):
                print(element[0].numpy(), element[1].numpy())
            i += 1

        print ("number of elements ", i, "\n")

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