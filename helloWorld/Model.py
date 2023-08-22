
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
import os
import re

# this gives a high result
def verification(datasetForVerification):
    stacked_Predicted_Output_tensor = None
    stacked_realOutput_tensor       = None

    for input_example_batch, target_example_batch in datasetForVerification.take(200):
        inputTensorExpanded = tf.expand_dims(input_example_batch, 0)
        outputTensorExpanded = tf.expand_dims(target_example_batch, 0)
        example_batch_predictions = model(inputTensorExpanded, return_state=False)
        
        if (stacked_Predicted_Output_tensor == None):
            stacked_Predicted_Output_tensor = example_batch_predictions
        else:
            stacked_Predicted_Output_tensor = tf.concat([stacked_Predicted_Output_tensor, example_batch_predictions], axis=0)

        if (stacked_realOutput_tensor == None):
            stacked_realOutput_tensor = outputTensorExpanded
        else:
            stacked_realOutput_tensor = tf.concat([stacked_realOutput_tensor, outputTensorExpanded], axis=0)


    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    batch_mean_loss = loss(stacked_realOutput_tensor, stacked_Predicted_Output_tensor)
    print("Mean loss:        ", batch_mean_loss)

    tf.exp(batch_mean_loss).numpy()
    print("Mean loss exp:        ", tf.exp(batch_mean_loss).numpy())

    return batch_mean_loss.numpy()


def CreateDatasetFromDirectory(InputDirectory, OutputDirectoryTraining):
    TokenParser.process_log_files_to_tokens_recursive(InputDirectory, OutputDirectoryTraining)
    ################# Create a dataset for training and throw all files that are in the training output dir in it
    TraininglogFileIterator = TokenParser.LogfileIterator(OutputDirectoryTraining)
    tokenLogfileToProcess = TraininglogFileIterator.get_first_tokenlogfile()
    datasetForTraining = None

    while (tokenLogfileToProcess is not None):
        print("adding " + tokenLogfileToProcess + " to the dataset")
        logTokenNumpyList = np.loadtxt(tokenLogfileToProcess, delimiter=",", dtype=str)
        TrainingLogTensor = tf.convert_to_tensor(logTokenNumpyList)
        ids_from_logfile = id_from_token(TrainingLogTensor)
        sliced_ids_from_logfile = tf.data.Dataset.from_tensor_slices(ids_from_logfile)
        # each sequence entry contains seq_length number of elements
        sequences = sliced_ids_from_logfile.batch(seq_length+1, drop_remainder=True)
        # run the function split_input_target accross all entries of sequences
        if (datasetForTraining == None):
            datasetForTraining = sequences.map(TokenParser.split_input_target)
            
        else:
            datasetForTraining = datasetForTraining.concatenate(sequences.map(TokenParser.split_input_target))

        tokenLogfileToProcess = TraininglogFileIterator.get_next_tokenlogfile()
        
    return datasetForTraining


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
    Scorefilehandle  = open(Scorefile, "w")
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



def AddProbabilitiesToParallel(Model, LogfileVerification, FileNameAnalyzed, token_from_id, id_from_token, PrintfFromToken):
    TestFileNumpyList = np.loadtxt(LogfileVerification, delimiter=",", dtype=str)
    idsTestFile = id_from_token(TestFileNumpyList)

    one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token)
    window_size = seq_length

    
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
    maxNumberOfParallelExecutions = 100

    while (i <= endIndex):
        
        numberOfParallelExecutions = min(endIndex - i + 1, maxNumberOfParallelExecutions)
        inputTensor = []
        nexttokenIds = []
        Windows = []
        y = 0

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

        while (y < numberOfParallelExecutions):
            nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens = one_step_model.generate_steps(inputTensor, nexttokenIds, states=None)
            nextPredictedTokens[y] = nextPredictedTokens[y].decode('utf-8')  
            if (nexttokenIds.numpy()[y] == nextPredictedTokens[y]):
                nextPredictedToken = "Identical  "
            
            printf = PrintfFromToken[nextToken]
            lineToWrite = token_from_id(nexttokenIds.numpy()[y]).numpy().decode('utf-8')   + " Probability = " + "{:2.5f}".format((probabilityOfNextTokens[y]).numpy()) + ". Predicted: " + nextPredictedTokens[y] + ". Probability: {:2.5f}".format(probabilityPredictions[y].numpy()) + ". " + "window = " + Windows[y][window_size - 2] + "  " + printf + "\n"
            LogfileAnalyzed.write(lineToWrite)
            print("annotating line " + str(i) + " of " + str(TestFileNumpyList.shape[0]))
            y += 1
       
        i += 1

    LogfileToAnalyze.close()
    LogfileAnalyzed.close()

# https://www.tensorflow.org/text/tutorials/text_generation

force_train_model = False

EPOCHS = 10
seq_length = 10
trainingSet = 3

start = time.time()

if (trainingSet == 1):
    Directory = "TrainingSet1"
elif (trainingSet == 2):
    Directory = "TrainingSet2"
elif (trainingSet == 3):
    Directory = "TrainingSet3"
elif (trainingSet == 4):
    Directory = "TrainingSet4"
elif (trainingSet == 5):
    Directory = "TrainingSet5"

OriginalTokenFileName       = Directory + os.path.sep + "Tokens.dat"

OutputDirectory             = Directory + os.path.sep + "out"
ScoreFile                   = OutputDirectory + os.path.sep + "ScoreFile.txt"

VocabularyFile              = OutputDirectory + os.path.sep + "TokenfileOnlyTokens.dat"

ModelDirectory              = OutputDirectory + os.path.sep + "SavedModel"
TrainingDirectory           = Directory + os.path.sep + "Training"
VerificationDirectory       = Directory + os.path.sep + "Verification"
OutputDirectoryTraining     = OutputDirectory + os.path.sep + "Training"
OutputDirectoryVerification = OutputDirectory + os.path.sep + "Verification"

PrintfFromToken = TokenParser.GetTokenTranslation(OriginalTokenFileName)
######################################################################################################
id_from_token, token_from_id = GetEncoderDecoderFromTokenFile(OriginalTokenFileName, VocabularyFile)

################# Create a dataset for training and throw all files that are in the training output dir in it
datasetForTraining = CreateDatasetFromDirectory(TrainingDirectory, OutputDirectoryTraining)        
datasetForTraining2 = datasetForTraining  # for the losstraining
################# Create a dataset for verification and throw all files that are in the verification output dir in it
datasetForVerification = CreateDatasetFromDirectory(VerificationDirectory, OutputDirectoryVerification)    

################## check whether the logfile tokens only contains tokens that are in the tokenfile
CrossReferenceResultTraining     = TokenParser.CrossReferenceTokens(OutputDirectoryTraining, VocabularyFile)
CrossReferenceResultVerification = TokenParser.CrossReferenceTokens(OutputDirectoryVerification, VocabularyFile)

if (CrossReferenceResultTraining == -1) or (CrossReferenceResultVerification == -1):
    print("Logfile contains tokens that are not in the tokenfile!!")
    exit()

try:
    model = tf.keras.models.load_model(ModelDirectory, compile=True)
except:
    model = None

# jents: organize them in batches and also shuffle .
# Batch size
BATCH_SIZE = 32

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

datasetForTraining = (
    datasetForTraining
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

datasetForTraining  # python convention to show the variable is ready

if ((model is None) or (force_train_model)):
    model = None
    
    # The embedding dimension, to be checked out
    embedding_dim = 256

    # Number of RNN units, to be checked out
    rnn_units = 1024  

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

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=False)
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    
    if (latest is not None):
        model.load_weights(latest)

    
    history = model.fit(datasetForTraining, epochs=EPOCHS, callbacks=[checkpoint_callback])
    
    model.save(ModelDirectory)

############################################################################################# Use the model


losstraining     = verification(datasetForTraining2)
lossverification = verification(datasetForVerification)
scoreString = "loss at training: " + str(losstraining) + ", Loss at verification: " + str(lossverification)
print(scoreString)

AddScoresToScoreFile(ScoreFile, scoreString)


verificationFileIterator = TokenParser.LogfileIterator(OutputDirectoryVerification)
tokenLogfileToProcess    = verificationFileIterator.get_first_tokenlogfile()
LogfileAnalyzed = tokenLogfileToProcess.replace(".log", ".analyzedlogparallel")
#AddProbabilitiesToParallel(model, tokenLogfileToProcess, LogfileAnalyzed, token_from_id, id_from_token, PrintfFromToken)
LogfileAnalyzed = tokenLogfileToProcess.replace(".log", ".analyzedlog")
AddProbabilitiesToLogFile(model, tokenLogfileToProcess, LogfileAnalyzed, token_from_id, id_from_token, PrintfFromToken)




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
    


end = time.time()

print('\nRun time:', end - start)
print("END")



