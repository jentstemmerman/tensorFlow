
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
import os




def verification(datasetForVerification):
    
    for input_example_batch, target_example_batch in datasetForVerification.take(1):
        example_batch_predictions = model(input_example_batch)
        print("input_example_batch, example_batch_predictions.shape ", input_example_batch.shape, example_batch_predictions.shape)

     ############################################################################################# Try without training

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    sampled_indices

    print("Input:\n", token_from_id(input_example_batch[0]).numpy())
    print()
    print("Next Token Predictions:\n", token_from_id(sampled_indices).numpy())

    for token in token_from_id(sampled_indices).numpy():
        translation = TokenParser.GetTokenTranslation(token.decode('utf-8'), OriginalTokenFileName)
        print(token.decode('utf-8'), " ", translation)


    ############################################################################################# Setup to train the model

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", example_batch_mean_loss)

    tf.exp(example_batch_mean_loss).numpy()



#tf.config.threading.set_intra_op_parallelism_threads(1)

# https://www.tensorflow.org/text/tutorials/text_generation

force_train_model = False

EPOCHS = 5
seq_length = 7
trainingSet = 3



if (trainingSet == 1):
    Directory = "TrainingSet1"
elif (trainingSet == 2):
    Directory = "TrainingSet2"
elif (trainingSet == 3):
    Directory = "TrainingSet3"

OriginalTokenFileName       = Directory + os.path.sep + "Tokens.dat"

OutputDirectory             = Directory + os.path.sep + "out"
VocabularyFile              = OutputDirectory + os.path.sep + "TokenfileOnlyTokens.dat"

ModelDirectory              = OutputDirectory + os.path.sep + "SavedModel"
TrainingDirectory           = Directory + os.path.sep + "Training"
VerificationDirectory       = Directory + os.path.sep + "Verification"
OutputDirectoryTraining     = OutputDirectory + os.path.sep + "Training"
OutputDirectoryVerification = OutputDirectory + os.path.sep + "Verification"

TokenParser.process_files(OriginalTokenFileName, VocabularyFile)
TokenParser.process_log_files_to_tokens_recursive(TrainingDirectory, OutputDirectoryTraining)
TokenParser.process_log_files_to_tokens_recursive(VerificationDirectory, OutputDirectoryVerification)

################## check whether the logfile tokens only contains tokens that are in the tokenfile
CrossReferenceResultTraining = TokenParser.CrossReferenceTokens(OutputDirectoryTraining, VocabularyFile)
CrossReferenceResultVerification = TokenParser.CrossReferenceTokens(OutputDirectoryVerification, VocabularyFile)

if (CrossReferenceResultTraining == -1) or (CrossReferenceResultVerification == -1):
    print("Logfile contains tokens that are not in the tokenfile!!")
    exit()

######################################################################################################

VocabularyNumpyList = np.loadtxt(VocabularyFile, delimiter=",", dtype=str)
VocabularyTensor    = tf.convert_to_tensor(VocabularyNumpyList)

# encoder
id_from_token = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None)
# decoder
token_from_id = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None, invert=True)


################# Create a dataset for training and throw all files that are in the training output dir in it
TraininglogFileIterator = TokenParser.LogfileIterator(OutputDirectoryTraining)
tokenLogfileToProcess = TraininglogFileIterator.get_first_tokenlogfile()
datasetForTraining = None

while (tokenLogfileToProcess is not None):
    print("adding " + tokenLogfileToProcess + " to the training dataset")
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


################# Create a dataset for verification and throw all files that are in the verification output dir in it
TraininglogFileIterator = TokenParser.LogfileIterator(OutputDirectoryTraining)
tokenLogfileToProcess = TraininglogFileIterator.get_first_tokenlogfile()
datasetForVerification = None

while (tokenLogfileToProcess is not None):
    print("adding " + tokenLogfileToProcess + " to the training dataset")
    logTokenNumpyList = np.loadtxt(tokenLogfileToProcess, delimiter=",", dtype=str)
    TrainingLogTensor = tf.convert_to_tensor(logTokenNumpyList)
    ids_from_logfile = id_from_token(TrainingLogTensor)
    sliced_ids_from_logfile = tf.data.Dataset.from_tensor_slices(ids_from_logfile)
    # each sequence entry contains seq_length number of elements
    sequences = sliced_ids_from_logfile.batch(seq_length+1, drop_remainder=True)
    # run the function split_input_target accross all entries of sequences
    if (datasetForVerification == None):
        datasetForVerification = sequences.map(TokenParser.split_input_target)
    else:
         datasetForVerification = datasetForVerification.concatenate(sequences.map(TokenParser.split_input_target))

    tokenLogfileToProcess = TraininglogFileIterator.get_next_tokenlogfile()

###################################################################################################

try:
    model = tf.keras.models.load_model(ModelDirectory, compile=True)
except:
    model = None

# jents: organize them in batches and also shuffle .
# Batch size
BATCH_SIZE = 64

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

############################################################################################# Train the model
# Checking the shape of the tensor
#print(tensor.shape)   -> je wil deze gebruiken
#print(tf.shape(tensor))
verificationFileIterator = TokenParser.LogfileIterator(OutputDirectoryVerification)
tokenLogfileToProcess = TraininglogFileIterator.get_first_tokenlogfile()

if (tokenLogfileToProcess == None):
    exit()

start = time.time()
TestFileNumpyList = np.loadtxt(tokenLogfileToProcess, delimiter=",", dtype=str)

idsTestFile = id_from_token(TestFileNumpyList)  # dit vertaalt heel de array naar Id's - dit returned een array van Id's


one_step_model = ModelClass.OneStep(model, token_from_id, id_from_token)

calculateAverage = 0

if (calculateAverage):
    window_size = 8
    Accumulator = 0
    AccumulatorCount = 0
    for i in range(TestFileNumpyList.shape[0] - (window_size + 1)):
        window    = TestFileNumpyList[i:i + window_size - 1]
        nextToken = TestFileNumpyList[i + window_size]
        idsLogFile = id_from_token(window) 
        inputTensor = tf.convert_to_tensor(idsLogFile, dtype='int64')
        probability = one_step_model.GetTokenProbability(inputTensor, nextToken)
        #print(window, "probability of next token ", nextToken, " = ", probability)
        AccumulatorCount = AccumulatorCount + 1
        Accumulator = Accumulator + probability
    print('\nAverage score = ', Accumulator / AccumulatorCount)
    print('\nRun time:', end - start)




#verification(datasetForVerification)
end = time.time()
print("END")



