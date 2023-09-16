
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
import os
import re
from pathlib import Path
import sys

# To only run on the cpu:
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""  # or even "-1"

DataTypeForTensors = 'int64'

# this gives a high result
def CalculateLossCrossEntropy(datasetForVerification, NumberOfSamples):

    numberOfSamplesPerCycle = 200
    samplesAccumulated = 0
    batch_mean_losses = [] 
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    for i in range(1, int(NumberOfSamples / numberOfSamplesPerCycle)):
        stacked_Predicted_Output_tensor = None
        stacked_realOutput_tensor       = None

        for input_example_batch, target_example_batch in datasetForVerification.take(numberOfSamplesPerCycle):
            outputTensorExpanded = tf.expand_dims(target_example_batch, 0)
            inputTensorExpanded  = tf.expand_dims(input_example_batch, 0)
           # inputTensor = tf.stack(inputTensor, axis=0)
           # nexttokenIds = tf.stack(nexttokenIds, axis=0)
            batch_predictions    = model(inputTensorExpanded, return_state=False)

            if (stacked_Predicted_Output_tensor == None):
                stacked_Predicted_Output_tensor = batch_predictions
            else:
                stacked_Predicted_Output_tensor = tf.concat([stacked_Predicted_Output_tensor, batch_predictions], axis=0)

            if (stacked_realOutput_tensor == None):
                stacked_realOutput_tensor = outputTensorExpanded
            else:
                stacked_realOutput_tensor = tf.concat([stacked_realOutput_tensor, outputTensorExpanded], axis=0)

            batch_mean_loss = loss(outputTensorExpanded, batch_predictions)
            batch_mean_losses.append(batch_mean_loss)
            #print("batch_mean_losses:        ", batch_mean_loss)

    mean_loss = tf.reduce_mean(batch_mean_losses)

    return mean_loss.numpy()



def GetEncoderDecoderFromTokenFile(OriginalTokenFileName, VocabularyFile):
    TokenParser.process_files(OriginalTokenFileName, VocabularyFile)
    VocabularyNumpyList = np.loadtxt(VocabularyFile, delimiter=",", dtype=str)
    VocabularyTensor    = tf.convert_to_tensor(VocabularyNumpyList)

    # encoder
    id_from_token = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None)
    # decoder
    token_from_id = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None, invert=True)

    return id_from_token, token_from_id

def AnalyzeTokenFiles(directory):

    verificationFileIterator = TokenParser.LogfileIterator(OutputDirectoryVerification)
    tokenLogfileToProcess    = verificationFileIterator.get_first_tokenlogfile()

    while (tokenLogfileToProcess is not None):
        LogfileAnalyzed = tokenLogfileToProcess.replace(".log", ".analyzedlogparallel")
        AddProbabilitiesTokenFileParallel(model, tokenLogfileToProcess, LogfileAnalyzed, token_from_id, id_from_token, PrintfFromToken, temperature=1)

        tokenLogfileToProcess = verificationFileIterator.get_next_tokenlogfile()


def AnalyzePrintfFiles(inputDirectory, outputDirectory, temperature):
    verificationFileIterator = TokenParser.LogfileIterator(inputDirectory)
    printfFileToProcess      = verificationFileIterator.get_first_tokenlogfile()

    while (printfFileToProcess is not None):
        outputFile = Path(printfFileToProcess).name.replace(".log", "_analyzedprintf_" + str(temperature) + ".txt")
        LogfileAnalyzed = os.path.join(outputDirectory, outputFile)

        AddProbabilitiesPrintfFileParallel(model, printfFileToProcess, LogfileAnalyzed, token_from_id, id_from_token, PrintfFromToken, temperature=temperature)

        printfFileToProcess = verificationFileIterator.get_next_tokenlogfile()



def AddProbabilitiesTokenFileParallel(Model, LogfileVerification, FileNameAnalyzed, token_from_id, id_from_token, PrintfFromToken, temperature):
    TestFileNumpyList = np.loadtxt(LogfileVerification, delimiter=",", dtype=str)
    idsTestFile = id_from_token(TestFileNumpyList)

    one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token, temperature=temperature)
    window_size    = seq_length

    
    LogfileToAnalyze = open(LogfileVerification, "r", errors='ignore')
    LogfileAnalyzed  = open(FileNameAnalyzed,    "w")

    token_pattern = re.compile(TokenParser.ReForTokens)
    
    lineNumber = 0

    for lineLogfileToAnalyze in LogfileToAnalyze:
            regexFound = re.search(token_pattern, lineLogfileToAnalyze)
            if  (regexFound):
                tokenString = regexFound.string
                tokenString = tokenString.replace("\n", "")
                lineLogfileToAnalyze = lineLogfileToAnalyze.replace(tokenString, tokenString + " Not Appl")
                LogfileAnalyzed.write(lineLogfileToAnalyze)
            lineNumber = lineNumber + 1
            if (lineNumber == window_size):
                break

    endIndex = TestFileNumpyList.shape[0] - window_size
    i = 0
    maxNumberOfParallelExecutions = 500


    while (i <= endIndex):
        
        numberOfParallelExecutions = min(endIndex - i, maxNumberOfParallelExecutions)
        if (numberOfParallelExecutions == 0):
            break

        inputTensor = []
        nexttokenIds = []
        Windows = []
        y = 0

        startTime = time.time()
        while (y < numberOfParallelExecutions):
            window      = TestFileNumpyList[i:i + window_size]
            nextToken   = TestFileNumpyList[i + window_size]
            IdsInWindow = id_from_token(window) 
            NextId      = id_from_token(nextToken) 

            Windows.append(window)

            tempTensor = tf.convert_to_tensor(IdsInWindow, dtype=DataTypeForTensors)
            inputTensor.append(tempTensor)
            
            nexttokenId = id_from_token(nextToken)
            nexttokenId = tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors)
            nexttokenIds.append(nexttokenId)  # make it a list of tensors

            i += 1
            y += 1
    
        inputTensor = tf.stack(inputTensor, axis=0)
        nexttokenIds = tf.stack(nexttokenIds, axis=0)
        y = 0
        
        nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens = one_step_model.generate_steps(inputTensor, nexttokenIds, states=None)
        
        while (y < numberOfParallelExecutions):
            nextPredictedTokens[y] = nextPredictedTokens[y].decode('utf-8')  
            nextToken = token_from_id(nexttokenIds.numpy()[y]).numpy().decode('utf-8')
            nextPredictedToken = nextPredictedTokens[y]
            if (nextToken == nextPredictedToken):
                nextPredictedToken = "Identical  "

            
            printf = PrintfFromToken[nextToken]
            lineToWrite = nextToken  + " Probability = " + "{:2.5f}".format((probabilityOfNextTokens[y]).numpy()) + ". Predicted: " + nextPredictedToken + ". Probability: {:2.5f}".format(probabilityPredictions[y].numpy()) + ". " + "window = " + Windows[y][window_size - 2] + "  " + printf + "\n"
            LogfileAnalyzed.write(lineToWrite)
            
            y += 1
        
        Endtime = time.time()

        print("annotated line " + str(i+y) + " of " + str(TestFileNumpyList.shape[0]) + ". Number of lines per second: {:2.2f}".format((numberOfParallelExecutions / (Endtime - startTime) )))


    LogfileToAnalyze.close()
    LogfileAnalyzed.close()



def AddProbabilitiesPrintfFileParallel(Model, LogfileVerification, FileNameAnalyzed, token_from_id, id_from_token, PrintfFromToken, temperature):

 #   idsTestFile = id_from_token(TestFileNumpyList)
    one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token, temperature=temperature)
    window_size = seq_length

    
    LogfileToAnalyze = open(LogfileVerification, "r", errors='ignore')
    LogfileAnalyzed  = open(FileNameAnalyzed,    "w")

    LogfileAnalyzed.write("Analyzed file at temperature " + str(temperature) + "\n")

    tokenList, PrintfList = TokenParser.getNextLinesAndTokens(TokenParser.ReForTokens, window_size, LogfileToAnalyze, 0)

    for line in PrintfList:
        LogfileAnalyzed.write(line)


    maxNumberOfParallelExecutions = 500
    numberOfAnalyzedLines = 0
    EndOfFileReached = False 
    accumulatedNextTokenProbability = 0.0

    while (not EndOfFileReached):
        
        numberOfParallelExecutions = maxNumberOfParallelExecutions
        if (numberOfParallelExecutions == 0):
            break

        inputTensor = []
        nexttokenIds = []
        Windows = []
        PrintfList = []

        y = 0
        i = 0
        startTime = time.time()
        while (y < numberOfParallelExecutions):
            tokenList, printfList = TokenParser.getNextLinesAndTokens(TokenParser.ReForTokens, window_size + 1, LogfileToAnalyze, 1)
            
            if (tokenList == None):
                EndOfFileReached = True
                break

            window      = tokenList[0:window_size]
            Windows.append(window)

            PrintfList.append(printfList[-1])

            nextToken   = tokenList[window_size]
            IdsInWindow = id_from_token(window) 
            NextId      = id_from_token(nextToken) 

            tempTensor = tf.convert_to_tensor(IdsInWindow, dtype=DataTypeForTensors)
            inputTensor.append(tempTensor)
            
            nexttokenId = id_from_token(nextToken)
            nexttokenId = tf.convert_to_tensor(nexttokenId, dtype=DataTypeForTensors)
            nexttokenIds.append(nexttokenId)  # make it a list of tensors

            y += 1

        if (tokenList == None):
            EndOfFileReached = True
            break
        


        inputTensor  = tf.stack(inputTensor, axis=0)
        nexttokenIds = tf.stack(nexttokenIds, axis=0)
        y = 0
        
        nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens = one_step_model.generate_steps(inputTensor, nexttokenIds, states=None)
        
        while (y < numberOfParallelExecutions):
            nextPredictedTokens[y] = nextPredictedTokens[y].decode('utf-8')  
            nextToken = token_from_id(nexttokenIds.numpy()[y]).numpy().decode('utf-8')
            nextPredictedToken = nextPredictedTokens[y]
            
            if (nextToken == nextPredictedToken):
                nextPredictedToken = "Identical  "
            nextTokenProbability = (probabilityOfNextTokens[y]).numpy()
            lineToWrite = "Actual: " + nextToken  + "  {:2.3f}".format(nextTokenProbability) + ". Pred: " + nextPredictedToken + " {:2.3f}".format(probabilityPredictions[y].numpy()) + ". " + "window = " + Windows[y][window_size - 2] + "  " +  " :" + PrintfList[y]
            LogfileAnalyzed.write(lineToWrite)
            

            accumulatedNextTokenProbability += nextTokenProbability
            y += 1
            numberOfAnalyzedLines += 1
        
        Endtime = time.time()

        print("  annotated line " + str(numberOfAnalyzedLines) + ". Number of lines per second: {:2.2f} ".format((numberOfParallelExecutions / (Endtime - startTime) )))

    print("\n Total number of annotated lines " + str(y) + ". Average next token probability = " + str(accumulatedNextTokenProbability/numberOfAnalyzedLines)+ "\n")
    
    
    LogfileToAnalyze.close()
    LogfileAnalyzed.close()   


def CalculateAverageProbabilityNextToken(OutputDirectoryVerification):

    calculateAverage = 0


    if (calculateAverage):
        verificationFileIterator = TokenParser.LogfileIterator(OutputDirectoryVerification)
        tokenLogfileToProcess    = verificationFileIterator.get_first_tokenlogfile()

        if (tokenLogfileToProcess == None):
            exit()
    

        TestFileNumpyList = np.loadtxt(tokenLogfileToProcess, delimiter=",", dtype=str)

        idsTestFile = id_from_token(TestFileNumpyList)  # dit vertaalt heel de array naar Id's - dit returned een array van Id's

        one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token, temperature=temperature)
        window_size = 8
        Accumulator = 0
        AccumulatorCount = 0
        for i in range(TestFileNumpyList.shape[0] - (window_size + 1)):
            window    = TestFileNumpyList[i:i + window_size - 1]
            nextToken = TestFileNumpyList[i + window_size]
            idsLogFile = id_from_token(window) 
            inputTensor = tf.convert_to_tensor(idsLogFile, dtype=DataTypeForTensors)
            probability = one_step_model.GetTokenProbability(inputTensor, nextToken)
            AccumulatorCount = AccumulatorCount + 1
            Accumulator = Accumulator + probability

    return Accumulator / AccumulatorCount

def GetDataSetInformation(dataSet):
    # Count the number of elements in the dataset
    dataSetTemp = dataSet
    num_samples_dataset     = dataSetTemp.reduce(0, lambda x, _: x + 1).numpy()

    return num_samples_dataset

def CreateDatasetFromDirectory(InputDirectory, OutputDirectory, ReductionInSamples=None, removeDuplicates=False, verbose=False):

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

    datasetfast = tf.data.Dataset.from_tensor_slices(ids_from_logfile)
    datasetfast = datasetfast.window(seq_length + 1, shift=1, drop_remainder=True)
    #datasetfast = datasetfast.flat_map(lambda window2: window2.batch(seq_length+1))
    datasetfast = datasetfast.flat_map(lambda window2: window2)
    datasetfast = datasetfast.batch(seq_length+1, drop_remainder = True, num_parallel_calls = 10)

    if (verbose):
        PrintDataset(datasetSlow)
        PrintDataset(datasetfast)
    
    if (EXECUTE_SLOW_PATH):
        datasetSlow = datasetfast.map(TokenParser.split_input_target) 

    datasetfast = datasetfast.map(TokenParser.split_input_target) 
    
    if (verbose):
        PrintDatasetTuple(datasetSlow)
        PrintDatasetTuple(datasetfast)

    if (SLOW_PATH) and (EXECUTE_SLOW_PATH):
        dataset = datasetSlow
    else:
        dataset = datasetfast

    if (verbose) and (SLOW_PATH) and (EXECUTE_SLOW_PATH):
        CompareDatasets(datasetSlow, datasetSlow)

    if (removeDuplicates):
        sequences2      = sequencesSliced
        sequencesSliced = sequencesSliced.unique()

        print("Adding " + InputDirectory + " to the dataset has finished \n")

    return dataset, filesUsed

def PrintDataset(dataSet):
    if (dataSet is not None):
        i = 0
        print ("PrintDataset \n")
        for element in dataSet:
            print(element.numpy())
            i += 1

        print ("number of elements ", i, "\n")


def PrintDatasetTuple(dataSet):
    if (dataSet is not None):
        i = 0
        print ("PrintDatasetTuple \n")
        for element in dataSet:
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

# https://www.tensorflow.org/text/tutorials/text_generation

force_train_model = True
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.config.threading.set_inter_op_parallelism_threads(4)
#tf.config.threading.set_intra_op_parallelism_threads(4)

EPOCHS = 5
seq_length = 30
trainingSet = 4
removeDuplicates = False

# The embedding dimension: the output dimension of the embedding layer.
embedding_dim = 256

# Number of RNN units, to be checked out
rnn_units = 2048

start = time.time()

Directory = "TrainingSet" + str(trainingSet)

OriginalTokenFileName       = Directory + os.path.sep + "Tokens.dat"

OutputDirectory             = Directory + os.path.sep + "out"

ScoreFile                   = OutputDirectory + os.path.sep + "ScoreFile.txt"
TrainingLogFile             = OutputDirectory + os.path.sep + "TrainingLog.txt"
SummaryFile                 = OutputDirectory + os.path.sep + "Summary.txt"
VocabularyFile              = OutputDirectory + os.path.sep + "TokenfileOnlyTokens.dat"
ModelDirectory              = OutputDirectory + os.path.sep + "SavedModel"

if (force_train_model):
    model = None
else:
    try:
        model = tf.keras.models.load_model(ModelDirectory, compile=True)
    except:
        model = None
train_model = (model == None)

TrainingDirectory           = Directory + os.path.sep + "Training"
VerificationDirectory       = Directory + os.path.sep + "Verification"
OutputDirectoryTraining     = OutputDirectory + os.path.sep + "Training"
OutputDirectoryVerification = OutputDirectory + os.path.sep + "Verification"

if (train_model):

    old_directory_name = OutputDirectory
    
    if os.path.exists(old_directory_name):
        new_directory_name = old_directory_name  # Start with the old name
        counter = 1  # Start with the number 1

        while os.path.exists(new_directory_name):
            new_directory_name = f"{old_directory_name}_{counter}"
            counter += 1

        try:
            os.rename(old_directory_name, new_directory_name)
            print(f"Directory '{old_directory_name}' renamed to '{new_directory_name}' successfully.")
        except OSError as e:
            print(f"Error: {e}")



PrintfFromToken = TokenParser.GetTokenTranslation(OriginalTokenFileName)

######################################################################################################
id_from_token, token_from_id = GetEncoderDecoderFromTokenFile(OriginalTokenFileName, VocabularyFile)
print("Creating datasets:")
################# Create a dataset for training and throw all files that are in the training output dir in it
if (train_model):
    datasetForTraining, trainingFileList = CreateDatasetFromDirectory(TrainingDirectory, OutputDirectoryTraining, removeDuplicates=removeDuplicates, verbose = False) # , ReductionInSamples=10) #, removeDuplicates=removeDuplicates)        
    datasetForTraining2 = datasetForTraining  # for the losstraining

################# Create a dataset for verification and throw all files that are in the verification output dir in it
datasetForVerification, verificationFileList = CreateDatasetFromDirectory(VerificationDirectory, OutputDirectoryVerification)    
datasetForVerification2 = datasetForVerification # for the losstraining
################## check whether the logfile tokens only contains tokens that are in the tokenfile

CheckCardinality = False
if (CheckCardinality):
    print("Checking datasets:")
    if (train_model):
        num_samples_datasetForTraining     = GetDataSetInformation(datasetForTraining)
        print("num_samples_datasetForTraining: " + str(num_samples_datasetForTraining)) 

    num_samples_datasetForVerification = GetDataSetInformation(datasetForVerification)
    print("num_samples_datasetForTraining: " + str(num_samples_datasetForVerification)) 


CrossReferenceTokens = False

if (CrossReferenceTokens):
    print("CrossReferenceTokens:")
    CrossReferenceResultVerification = TokenParser.CrossReferenceTokens(OutputDirectoryVerification, VocabularyFile)

    if (train_model):
        CrossReferenceResultTraining = TokenParser.CrossReferenceTokens(OutputDirectoryTraining, VocabularyFile)
    else:
        CrossReferenceResultTraining = 0
    
    if (CrossReferenceResultTraining == -1) or (CrossReferenceResultVerification == -1):
        print("Logfile contains tokens that are not in the tokenfile!!")
        exit()


if (train_model):

    # jents: organize them in batches and also shuffle .
    # Batch size
    BATCH_SIZE = 32

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).


    numberOfSamplesToTrainPerEon    = 20000
    numberOfSamplesToValidatePerEon = 20000

    BUFFER_SIZE = numberOfSamplesToTrainPerEon
    
    Network = "GRU"
    model

    if (Network == "LSTM"):
        model = tf.keras.Sequential([
            id_from_token,
            tf.keras.layers.Embedding(
                input_dim=len(id_from_token.get_vocabulary()),
                output_dim=64,
                # Use masking to handle the variable sequence lengths
                mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    if (Network == "GRU"):
        model = ModelClass.GruPrintfLogModel(
            vocab_size=len(id_from_token.get_vocabulary()),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units)

    ############################################################################################# Train the model

    SampleDataSet = True


    datasetForTraining = (
        datasetForTraining
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    datasetForTraining  # python convention to show the variable is ready

    datasetForVerification = (
        datasetForVerification
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))


    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = Directory + '/out/training_checkpoints'

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback    = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=False)

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    
    if (latest is not None):
        model.load_weights(latest)


    sys.stdout = TokenParser.Logger(TrainingLogFile)


    EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    
    Epoch_End_callback = None
    
    if (SampleDataSet):
        # dit werkt niet zo goed
        class EpochEndCallback(tf.keras.callbacks.Callback):
            def __init__(self, numberOfSamplesToTrainPerEpoch, numberOfSamplesToValidatePerEpoch):
                print("EpochEndCallback start newcreating new subsetForBatchTraining")
                self.numberOfSamplesToTrainPerEpoch    = numberOfSamplesToTrainPerEpoch
                self.numberOfSamplesToValidatePerEpoch = numberOfSamplesToValidatePerEpoch
                self.subsetForBatchTraining            = datasetForTraining.take(self.numberOfSamplesToTrainPerEpoch)
                self.subsetForBatchVerification        = None
                print("EpochEndCallback start newcreating new subsetForBatchTraining done")

            def on_epoch_end(self, epoch, logs=None):
                print("on_epoch_end -> Creating new subsetForBatchTraining")
                self.subsetForBatchTraining     = datasetForTraining.shuffle(BUFFER_SIZE).take(self.numberOfSamplesToTrainPerEpoch)
                print("on_epoch_end Creating new subsetForBatchTraining done")
        

        Epoch_End_callback = EpochEndCallback(numberOfSamplesToTrainPerEon, numberOfSamplesToValidatePerEon)
        history = model.fit(Epoch_End_callback.subsetForBatchTraining, validation_data=Epoch_End_callback.subsetForBatchVerification, epochs=EPOCHS, callbacks=[checkpoint_callback, EarlyStopping_callback, Epoch_End_callback])
    else:
        history = model.fit(datasetForTraining, validation_data=datasetForVerification, epochs=EPOCHS, callbacks=[checkpoint_callback, EarlyStopping_callback])

    sys.stdout = sys.__stdout__

    model.save(ModelDirectory)

############################################################################################# Use the model

if (train_model):
    end = time.time()

    print("Files used to train: \n")
    print(*trainingFileList, sep="\n")

    print("\nCalculation of losses: ")
    # Open a file in write mode
    with open(ScoreFile, 'w') as f:
        # Redirect the standard output to the file
        original_stdout = sys.stdout
        sys.stdout = f

        print("\nSettings used: ")
        ModelSummary = "EPOCHS: " + str(EPOCHS) + "\n"
        ModelSummary = ModelSummary + "seq_lenth: " + str(seq_length) + "\n"
        ModelSummary = ModelSummary + "trainingSet: " + str(trainingSet) + "\n"
        ModelSummary = ModelSummary + "embedding_dim: " + str(embedding_dim) + "\n"
        ModelSummary = ModelSummary + "rnn_units: " + str(rnn_units) + "\n"
        ModelSummary = ModelSummary + "numberOfSamplesToTrainPerEon: " + str(numberOfSamplesToTrainPerEon) + "\n"
        ModelSummary = ModelSummary + "numberOfSamplesToValidatePerEon: " + str(numberOfSamplesToTrainPerEon) + "\n"

        print(ModelSummary + "\n")
        print("Training finished in ", (end - start)/60, " minutes")
        
        print("\n")
        print("Files used to train: \n")
        print(*trainingFileList, sep="\n")

        print("\n")
        print("Files used in verification: \n")
        print(*verificationFileList, sep="\n")
        print("\n")

        print("ModelSummary: \n")

        # Print the model summary
        model.summary()

        print("\n")
        NumberOfSamplesForLossCalculation = 2000
        losstraining = str(CalculateLossCrossEntropy(datasetForTraining2, NumberOfSamplesForLossCalculation))
        lossverification = str(CalculateLossCrossEntropy(datasetForVerification2, NumberOfSamplesForLossCalculation))
        scoreString = "loss at training: " + losstraining + ", Loss at verification: " + lossverification
        print(scoreString)
        # Reset the standard output
        sys.stdout = original_stdout
else:
    losstraining = "Not Appl"

start = 1.0
stop = 5.0
step = 0.1

Temperature = start
while Temperature < stop:
    AnalyzePrintfFiles(VerificationDirectory, OutputDirectoryVerification, Temperature)
    Temperature += step
 


end = time.time()

print('\nRun time:', end - start)
print("END")



