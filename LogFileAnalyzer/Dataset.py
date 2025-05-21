import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import LogFileIterator
import Dataset
import os

Universal_Mask_Token = "MaskedToken"


def GetDataSetInformation(dataSet):
    # Count the number of elements in the dataset
    dataSetTemp         = dataSet
    num_batches_dataset = dataSetTemp.reduce(0, lambda x, _: x + 1).numpy()

    return num_batches_dataset

def GetMaskValue(featureIndex, id_from_tokenList):
    id_from_token = id_from_tokenList[featureIndex]
    
    if ( id_from_token is None):
        maskValue = float('inf')
    else:
        maskValue = id_from_token(Universal_Mask_Token) 
    
    return maskValue.numpy()

def CreateTrainingDataSetOfMarker(sample, id_from_tokenList, DataTypeForTensors):
    input_shape     = sample[-1].shape
    numberOfeatures = sample[-1].shape[1]
    
    new_shape = tf.concat([[input_shape[0] + 1], input_shape[1:]], axis=0) # maak een shape die 1 meer is in de eerste dimensie
    

    assert(numberOfeatures == len(id_from_tokenList))

    sequenceLength  = [sample[-1].shape[0] + 1]
    maskList        = []

    for featureIndex, id_from_token in enumerate(id_from_tokenList):
        maskValue  = GetMaskValue(featureIndex, id_from_tokenList)
        maskColumn = tf.fill(sequenceLength, maskValue)
        maskColumn = tf.cast(maskColumn, DataTypeForTensors)
        maskList.append(maskColumn)

    maskTensor = tf.stack(maskList, axis=-1)

    datasetfast = tf.data.Dataset.from_tensor_slices(maskTensor)
    datasetfast = datasetfast.window(input_shape[0] + 1, shift=1, drop_remainder=True)
    datasetfast = datasetfast.flat_map(lambda window2: window2)
    datasetfast = datasetfast.batch(input_shape[0] + 1, drop_remainder = True)
    datasetfast = datasetfast.map(TokenParser.split_input_target) 
    new_sample = next(iter(datasetfast))

    return datasetfast

def SetMarkerAtTrainingDataSetEdge(dataset, id_from_tokenList, DataTypeForTensors):

    sample1 = next(iter(dataset))
    datasetOfMarker = CreateTrainingDataSetOfMarker(sample1, id_from_tokenList, DataTypeForTensors)

    combined_dataset = dataset.concatenate(datasetOfMarker)

    return combined_dataset


def PrepareDataSetForTraining(dataset, batch_size, shuffle=False, buffer_size=0):

    if (shuffle):
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    options = tf.data.Options()    
    #options.autotune.cpu_budget = 8
    
    dataset = dataset.with_options(options)
    return dataset

def CreateDatasetFromDirectoryNew(InputDirectory, OutputDirectory, itemList, id_from_tokenList, mergeDatasets, stride, stride_dataset_inference, seq_length, DataTypeForTensors, ReductionInSamples=None, removeDuplicates=False, num_parallel_calls=10, dual_input = False, forTraining=True, verbose=False):
    
    fileIterator = LogFileIterator.FileIterator(InputDirectory, ".log")
    InputFile   = fileIterator.get_first_file()

    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=7) as executor:
        InputFile = fileIterator.get_first_file()
        futureList = []
        while True:
            
            if InputFile is None:
                break
            future = executor.submit(process_file, InputFile, OutputDirectory, itemList, id_from_tokenList, stride, stride_dataset_inference, seq_length, dual_input, DataTypeForTensors, forTraining, removeDuplicates)
            futureList.append(future)
            InputFile = fileIterator.get_next_file()
        
        concurrent.futures.wait(futureList, return_when=concurrent.futures.ALL_COMPLETED)
        
        # Ensure all tasks are completed (and re-raise any exceptions) by calling result:
        for future in futureList:
            future.result()
        
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

        
def process_file(InputFile, OutputDirectory, itemList, id_from_tokenList, stride, stride_dataset_inference, seq_length, dual_input, DataTypeForTensors, forTraining, removeDuplicates):
    if (forTraining):
        print("process_file " + InputFile + " into a dataset... \n")
    else:
        print("process_file " + InputFile + " into an inference dataset... \n")
        
    IntermediateFile = None #OutputDirectory + os.path.sep + os.path.basename(InputFile) + ".intermediate"
    outputDatasetFile = OutputDirectory + os.path.sep + os.path.basename(InputFile) + ".dataset"

    if (forTraining):
        dataset = CreateTrainingDatasetFromFile(InputFile, 
                                                IntermediateFile, 
                                                itemList, 
                                                id_from_tokenList, 
                                                stride, 
                                                seq_length, 
                                                dual_input,
                                                DataTypeForTensors, 
                                                removeDuplicates=removeDuplicates, 
                                                num_parallel_calls=10, 
                                                verbose=False)
        
        dataset = SetMarkerAtTrainingDataSetEdge(dataset, id_from_tokenList, DataTypeForTensors)
        print("Saving training dataset " + os.path.basename(InputFile) + "...\n")
    else:
        dataset = CreateTrainingDatasetFromFile(InputFile, 
                                                IntermediateFile, 
                                                itemList, 
                                                id_from_tokenList, 
                                                stride_dataset_inference, 
                                                seq_length, 
                                                dual_input,
                                                DataTypeForTensors, 
                                                removeDuplicates=removeDuplicates, 
                                                num_parallel_calls=10, 
                                                verbose=False)
        
        dataset = SetMarkerAtTrainingDataSetEdge(dataset, id_from_tokenList, DataTypeForTensors)
        print("Saving training dataset " + os.path.basename(InputFile) + "...\n")
        
        
    Dataset.SaveDataset(dataset, outputDatasetFile.strip())
    print("Dataset " + os.path.basename(InputFile) + " saved.\n")


# This function will not create an intermediate tokenfile
def CreateInferenceDatasetFromFile(InputFile, IntermediateFile, itemList, id_from_tokenList, stride, seq_length, DataTypeForTensors):

    print("Creating an inference dataset from  " + InputFile + ".\n")
    
    #if(IntermediateFile is not None):
    #    print(" Intermediate file: " + IntermediateFile + ".\n")

    tokenListList, _, _ = TokenParser.process_file_to_tokenFile(itemList=itemList, input_file=InputFile, human_readable_output_file=IntermediateFile)
   #print(" Number Of Lines in the file: " + str(numberOfLines) + "\n" )
    
    TensorOfThisFile = []
    
    for index, id_from_token in enumerate(id_from_tokenList):
        
        column = [inner_list[index] for inner_list in tokenListList]
        column = np.array(column)
        
        if (id_from_token is not None):
            TransformedItemTensor = id_from_token(column)  # next todo
        else:
            TransformedItemTensor = tf.convert_to_tensor(column, dtype=DataTypeForTensors)
        
        TensorOfThisFile.append(TransformedItemTensor)
    
    TensorOfThisFile = tf.stack(TensorOfThisFile, axis=-1)
    
    datasetfast = tf.data.Dataset.from_tensor_slices(TensorOfThisFile)
    
    try:
        datasetfast2 = datasetfast.window(seq_length, shift=stride, drop_remainder=True)
    except TypeError as e:
        print(f"Error: {e}")
        print(InputFile  + "\n")

    except OSError as e:
        print(f"Error: {e}")
        print(InputFile  + "\n")
        
    
    datasetfast3 = datasetfast2.flat_map(lambda window: window.batch(seq_length))    # datasetfast = datasetfast.batch(seq_length + 1, drop_remainder = True, num_parallel_calls = num_parallel_calls)

    return datasetfast3

# This function will not create an intermediate tokenfile
def CreateTrainingDatasetFromFile(InputFile, IntermediateFile, itemList, id_from_tokenList, stride, seq_length, dual_input, DataTypeForTensors, removeDuplicates=False, num_parallel_calls=10,  verbose=False):

    print("Creating a dataset from  " + InputFile + ".\n")
    
    if(IntermediateFile is not None):
        print(" Intermediate file: " + IntermediateFile + ".\n")

    tokenListList, _, numberOfLines = TokenParser.process_file_to_tokenFile(itemList=itemList, input_file=InputFile, human_readable_output_file=IntermediateFile)
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

    datasetfast = datasetfast.map(lambda x: tf.cast(x, DataTypeForTensors))
    
    if (dual_input):
        datasetfast = datasetfast.map(TokenParser.split_input_target_transformer) 
    else:
        datasetfast = datasetfast.map(TokenParser.split_input_target_transformer_decoder_only)
        
    return datasetfast

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
    dataset = tf.data.Dataset.load(fileName, compression="GZIP")
    return dataset

def MergeDatasets(dataSet1, dataSet2):
    datasetMerged = dataSet1.concatenate(dataSet2)
    return datasetMerged

#def CopyTokenfilesToOutputDirectory(itemList, OutputVo)

def GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory, itemList, OutputVocabularyDirectory=None, TokenFileDirectory = "TokenFiles"):

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
                    
                    if (OutputVocabularyDirectory is not None):
                        import shutil
                        
                        destinationDir = OutputVocabularyDirectory + os.path.sep + TokenFileDirectory
                        os.makedirs(destinationDir, exist_ok=True)
                        destinationFile = destinationDir + os.path.sep + os.path.basename(OriginalTokenFile) 
                                            
                        shutil.copy(OriginalTokenFileName, destinationFile)
                    
                    # only do one item process at a time
                    if (OutputVocabularyDirectory is not None):
                        OutputVocabularyFile = OutputVocabularyDirectory + os.path.sep + "VocabularyFileItem_" + str(itemIndex)

                        tokenListList, _, _ = TokenParser.process_file_to_tokenFile(itemList=[itemsInLine], input_file=OriginalTokenFileName, human_readable_output_file=OutputVocabularyFile)
                    else:
                        tokenListList, _, _ = TokenParser.process_file_to_tokenFile(itemList=[itemsInLine], input_file=OriginalTokenFileName, human_readable_output_file=None)


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
            

            id_from_token = tf.keras.layers.StringLookup(vocabulary=VocabularyNumpy, mask_token=Universal_Mask_Token)
            id_from_tokenList.append(id_from_token)
            
            # decoder
            token_from_id = tf.keras.layers.StringLookup(vocabulary=VocabularyNumpy, mask_token=Universal_Mask_Token, invert=True)
            token_from_idList.append(token_from_id)
        else:
            id_from_tokenList.append(None)
            token_from_idList.append(None)

        itemIndex += 1    
    return id_from_tokenList, token_from_idList