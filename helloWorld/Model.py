
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
import os
import re
from pathlib import Path

# To only run on the cpu:
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""  # or even "-1"



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
    print("mean_loss.numpy():        ", mean_loss.numpy())
    return mean_loss.numpy()

def CreateDatasetFromDirectory(InputDirectory, OutputDirectory, ReductionInSamples=None, removeDuplicates=False):
    TokenParser.process_log_files_to_tokens_recursive(InputDirectory, OutputDirectory)
    ################# Create a dataset for training and throw all files that are in the training output dir in it
    TraininglogFileIterator = TokenParser.LogfileIterator(OutputDirectory)
    tokenLogfileToProcess = TraininglogFileIterator.get_first_tokenlogfile()
    dataset = None

    while (tokenLogfileToProcess is not None):
        print("adding " + tokenLogfileToProcess + " to the dataset")

        if ((ReductionInSamples is not None) and (ReductionInSamples != 0)):
            logTokenNumpyListTemp = np.loadtxt(tokenLogfileToProcess, delimiter=",", dtype=str)
            num_lines = len(logTokenNumpyListTemp)
            num_linesToRead = int(num_lines / ReductionInSamples)
            logTokenNumpyList = []

            start_index = np.random.randint(0, num_lines - (seq_length + 1))
            random_lines = logTokenNumpyListTemp[start_index:start_index + (seq_length + 1)]
            logTokenNumpyList = random_lines
            
            for i in range(num_linesToRead):
                start_index = np.random.randint(0, num_lines - (seq_length + 1))
                random_lines = logTokenNumpyListTemp[start_index:start_index + (seq_length + 1)]
                logTokenNumpyList = np.vstack((logTokenNumpyList, random_lines))
        else:
            logTokenNumpyList = np.loadtxt(tokenLogfileToProcess, delimiter=",", dtype=str)

        TrainingLogTensor = tf.convert_to_tensor(logTokenNumpyList)
        ids_from_logfile = id_from_token(TrainingLogTensor)
        sliced_ids_from_logfile = tf.data.Dataset.from_tensor_slices(ids_from_logfile)
     
       
        # each sequence entry contains seq_length + 1 number of elements
        sequences = sliced_ids_from_logfile.batch(seq_length+1, drop_remainder=True)

        if (removeDuplicates):
            sequences2 = sequences
            sequences = sequences.unique()

        # run the function split_input_target accross all entries of sequences
        if (dataset == None):
            dataset = sequences.map(TokenParser.split_input_target)
            
        else:
            dataset = dataset.concatenate(sequences.map(TokenParser.split_input_target))

        tokenLogfileToProcess = TraininglogFileIterator.get_next_tokenlogfile()
        
    return dataset


def GetEncoderDecoderFromTokenFile(OriginalTokenFileName, VocabularyFile):
    TokenParser.process_files(OriginalTokenFileName, VocabularyFile)
    VocabularyNumpyList = np.loadtxt(VocabularyFile, delimiter=",", dtype=str)
    VocabularyTensor    = tf.convert_to_tensor(VocabularyNumpyList)

    # encoder
    id_from_token = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None)
    # decoder
    token_from_id = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None, invert=True)

    return id_from_token, token_from_id

def AddScoresToScoreFile(Scorefile, scoreString):
    Scorefilehandle = open(Scorefile, "w")
    Scorefilehandle.write(scoreString)
    Scorefilehandle.close()



def AddProbabilitiesToLogFile(Model, LogfileVerification, LogfileAnalyzed, token_from_id, id_from_token, PrintfFromToken):
    TestFileNumpyList = np.loadtxt(LogfileVerification, delimiter=",", dtype=str)
    idsTestFile = id_from_token(TestFileNumpyList)

    one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token)
    window_size = seq_length
    
    LogfileToAnalyze = open(LogfileVerification, "r", errors='ignore')
    LogfileAnalyzed  = open(LogfileAnalyzed,    "w")

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

    
    for i in range(TestFileNumpyList.shape[0] - (window_size + 1)):
        window      = TestFileNumpyList[i:i + window_size]
        nextToken   = TestFileNumpyList[i + window_size]
        IdsInWindow = id_from_token(window) 
        
        inputTensor = tf.convert_to_tensor(IdsInWindow, dtype='int64')
        nextPredictedToken, probabilityPrediction, probabilityOfNextToken = one_step_model.generate_one_step(tf.expand_dims(inputTensor, 0), nextToken, states=None)
        nextPredictedToken = nextPredictedToken.decode('utf-8')  
        
        if (nextToken == nextPredictedToken):
            nextPredictedToken = "Identical  "
                       
        printf = PrintfFromToken[nextToken]
        lineToWrite = nextToken + " Probability = " + "{:2.5f}".format(probabilityOfNextToken) + ". Predicted: " + nextPredictedToken + ". Probability: {:2.5f}".format(probabilityPrediction) + ". " + "window = " + window[window_size - 2] + "  " + printf + "\n"
        LogfileAnalyzed.write(lineToWrite)

        if (i % 100 == 0):
            print("annotating line " + str(i) + " of " + str(TestFileNumpyList.shape[0]))

    LogfileToAnalyze.close()
    LogfileAnalyzed.close()


def AnalyzeTokenFiles(directory):

    verificationFileIterator = TokenParser.LogfileIterator(OutputDirectoryVerification)
    tokenLogfileToProcess    = verificationFileIterator.get_first_tokenlogfile()

    while (tokenLogfileToProcess is not None):
        LogfileAnalyzed = tokenLogfileToProcess.replace(".log", ".analyzedlogparallel")
        AddProbabilitiesTokenFileParallel(model, tokenLogfileToProcess, LogfileAnalyzed, token_from_id, id_from_token, PrintfFromToken)

        tokenLogfileToProcess = verificationFileIterator.get_next_tokenlogfile()


def AnalyzePrintfFiles(inputDirectory, outputDirectory):
    verificationFileIterator = TokenParser.LogfileIterator(inputDirectory)
    printfFileToProcess      = verificationFileIterator.get_first_tokenlogfile()

    while (printfFileToProcess is not None):
        outputFile = Path(printfFileToProcess).name.replace(".log", "_analyzedprintf.txt")
        LogfileAnalyzed = os.path.join(outputDirectory, outputFile)

        AddProbabilitiesPrintfFileParallel(model, printfFileToProcess, LogfileAnalyzed, token_from_id, id_from_token, PrintfFromToken)

        printfFileToProcess = verificationFileIterator.get_next_tokenlogfile()



def AddProbabilitiesTokenFileParallel(Model, LogfileVerification, FileNameAnalyzed, token_from_id, id_from_token, PrintfFromToken):
    TestFileNumpyList = np.loadtxt(LogfileVerification, delimiter=",", dtype=str)
    idsTestFile = id_from_token(TestFileNumpyList)

    one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token)
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

            tempTensor = tf.convert_to_tensor(IdsInWindow, dtype='int64')
            inputTensor.append(tempTensor)
            
            nexttokenId = id_from_token(nextToken)
            nexttokenId = tf.convert_to_tensor(nexttokenId, dtype='int64')
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



def AddProbabilitiesPrintfFileParallel(Model, LogfileVerification, FileNameAnalyzed, token_from_id, id_from_token, PrintfFromToken):

 #   idsTestFile = id_from_token(TestFileNumpyList)
    one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token)
    window_size = seq_length

    
    LogfileToAnalyze = open(LogfileVerification, "r", errors='ignore')
    LogfileAnalyzed  = open(FileNameAnalyzed,    "w")

    tokenList, PrintfList = TokenParser.getNextLinesAndTokens(TokenParser.ReForTokens, window_size, LogfileToAnalyze, 0)

    for line in PrintfList:
        LogfileAnalyzed.write(line)


    maxNumberOfParallelExecutions = 500
    numberOfAnalyzedLines = 0
    EndOfFileReached = False 

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

            tempTensor = tf.convert_to_tensor(IdsInWindow, dtype='int64')
            inputTensor.append(tempTensor)
            
            nexttokenId = id_from_token(nextToken)
            nexttokenId = tf.convert_to_tensor(nexttokenId, dtype='int64')
            nexttokenIds.append(nexttokenId)  # make it a list of tensors

            y += 1

        if (tokenList == None):
            EndOfFileReached = True
            break
        
        numberOfAnalyzedLines += y

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

            lineToWrite = "Actual: " + nextToken  + "  {:2.3f}".format((probabilityOfNextTokens[y]).numpy()) + ". Pred: " + nextPredictedToken + " {:2.3f}".format(probabilityPredictions[y].numpy()) + ". " + "window = " + Windows[y][window_size - 2] + "  " +  " :" + PrintfList[y]
            LogfileAnalyzed.write(lineToWrite)
            
            y += 1
        
        Endtime = time.time()

        print("annotated line " + str(numberOfAnalyzedLines) + ". Number of lines per second: {:2.2f}".format((numberOfParallelExecutions / (Endtime - startTime) )))


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

        one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token)
        window_size = 8
        Accumulator = 0
        AccumulatorCount = 0
        for i in range(TestFileNumpyList.shape[0] - (window_size + 1)):
            window    = TestFileNumpyList[i:i + window_size - 1]
            nextToken = TestFileNumpyList[i + window_size]
            idsLogFile = id_from_token(window) 
            inputTensor = tf.convert_to_tensor(idsLogFile, dtype='int64')
            probability = one_step_model.GetTokenProbability(inputTensor, nextToken)
            AccumulatorCount = AccumulatorCount + 1
            Accumulator = Accumulator + probability

    return Accumulator / AccumulatorCount





# https://www.tensorflow.org/text/tutorials/text_generation

force_train_model = True
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.config.threading.set_inter_op_parallelism_threads(4)
#tf.config.threading.set_intra_op_parallelism_threads(4)

EPOCHS = 3
seq_length = 20
trainingSet = 3
removeDuplicates = False

# The embedding dimension: the output dimension of the embedding layer.
embedding_dim = 256

# Number of RNN units, to be checked out
rnn_units = 1024

start = time.time()

Directory = "TrainingSet" + str(trainingSet)

OriginalTokenFileName       = Directory + os.path.sep + "Tokens.dat"

OutputDirectory             = Directory + os.path.sep + "out"
ScoreFile                   = OutputDirectory + os.path.sep + "ScoreFile.txt"
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

PrintfFromToken = TokenParser.GetTokenTranslation(OriginalTokenFileName)
######################################################################################################
id_from_token, token_from_id = GetEncoderDecoderFromTokenFile(OriginalTokenFileName, VocabularyFile)

################# Create a dataset for training and throw all files that are in the training output dir in it
if (train_model):
    datasetForTraining = CreateDatasetFromDirectory(TrainingDirectory, OutputDirectoryTraining, removeDuplicates=removeDuplicates) # , ReductionInSamples=10) #, removeDuplicates=removeDuplicates)        
    datasetForTraining2 = datasetForTraining  # for the losstraining

################# Create a dataset for verification and throw all files that are in the verification output dir in it
datasetForVerification = CreateDatasetFromDirectory(VerificationDirectory, OutputDirectoryVerification)    
datasetForVerification2 = datasetForVerification # for the losstraining
################## check whether the logfile tokens only contains tokens that are in the tokenfile


if (train_model):
    CrossReferenceResultTraining = TokenParser.CrossReferenceTokens(OutputDirectoryTraining, VocabularyFile)
else:
    CrossReferenceResultTraining = 0

CrossReferenceResultVerification = TokenParser.CrossReferenceTokens(OutputDirectoryVerification, VocabularyFile)

if (CrossReferenceResultTraining == -1) or (CrossReferenceResultVerification == -1):
    print("Logfile contains tokens that are not in the tokenfile!!")
    exit()


if (train_model):

    # jents: organize them in batches and also shuffle .
    # Batch size
    BATCH_SIZE = seq_length * 5

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 100000

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
        model = ModelClass.PrintfLogModel(
            vocab_size=len(id_from_token.get_vocabulary()),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units)

    ############################################################################################# Train the model

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = Directory + '/out/training_checkpoints'

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback    = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=False)
    EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    
    if (latest is not None):
        model.load_weights(latest)

    #datasetForVerification = datasetForTraining
    history = model.fit(datasetForTraining, validation_data=datasetForVerification, epochs=EPOCHS, callbacks=[checkpoint_callback, EarlyStopping_callback])
    
    model.save(ModelDirectory)

############################################################################################# Use the model

if (train_model):
    print("Calculation of losses: ")
    NumberOfSamplesForLossCalculation = 20000
    losstraining = str(CalculateLossCrossEntropy(datasetForTraining2, NumberOfSamplesForLossCalculation))
    lossverification = str(CalculateLossCrossEntropy(datasetForVerification2, NumberOfSamplesForLossCalculation))
    scoreString = "loss at training: " + losstraining + ", Loss at verification: " + lossverification
    print(scoreString)
    AddScoresToScoreFile(ScoreFile, scoreString)
else:
    losstraining = "Not Appl"


#model.save_model_summary_to_file(SummaryFile)

AnalyzePrintfFiles(VerificationDirectory, OutputDirectoryVerification)
#AnalyzeTokenFiles(OutputDirectoryVerification)



end = time.time()

print('\nRun time:', end - start)
print("END")



