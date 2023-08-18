
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
import os

force_train_model = False

EPOCHS = 10

#tf.config.threading.set_intra_op_parallelism_threads(1)

# https://www.tensorflow.org/text/tutorials/text_generation
trainingSet = 3

if (trainingSet == 1):
    Directory             = "TrainingSet1"
    OriginalTokenFileName = Directory + "/Tokens.dat"
    OriginalLogFileName   = Directory + "/logfile.log"
    TestFile              = Directory + "/TestData.log"
    VocabularyFile        = Directory + "/out/TokenfileOnlyTokens.dat"
    LogFileTokens         = Directory + "/out/LogfileOnlyTokens.log"
    ############### create files that only contain tokens:
    TokenParser.process_files(OriginalTokenFileName, VocabularyFile)
    TokenParser.process_files(OriginalLogFileName,   LogFileTokens)
elif (trainingSet == 2):
    Directory             = "TrainingSet2"
    OriginalTokenFileName = Directory + "/Tokens.dat"
    OriginalLogFileName   = Directory + "/1340_Neo_KioxiaBiCS6_8ch-4TB_SIE_Regression_Combination_QuickCheck_drive_log.log"
    TestFile              = Directory + "/TestData.log"
    VocabularyFile        = Directory + "/out/TokenfileOnlyTokens.dat"
    LogFileTokens         = Directory+ "/out/LogfileOnlyTokens.log"
    ############### create files that only contain tokens:
    TokenParser.process_files(OriginalTokenFileName, VocabularyFile)
    TokenParser.process_files(OriginalLogFileName,   LogFileTokens)
elif (trainingSet == 3):
    Directory             = "TrainingSet3"
    OriginalTokenFileName = Directory + "/Tokens.dat"
    VocabularyFile        = Directory + "/out/TokenfileOnlyTokens.dat"
    OutputDirectory       = Directory + "/out"
    TokenParser.process_files(OriginalTokenFileName, VocabularyFile)
    TokenParser.process_log_files_to_tokens_recursive(Directory, OutputDirectory)


result = TokenParser.CrossReferenceTokens(Directory + "/out", VocabularyFile)
print("Cross reference passed = ", result)

if (result == -1):
    print("Logfile contains tokens that are not in the tokenfile!!")
    exit()

seq_length = 10


################## check whether the logfile tokens only contains tokens that are in the tokenfile


##################################################################################################
VocabularyNumpyList = np.loadtxt(VocabularyFile, delimiter=",", dtype=str)
VocabularyTensor    = tf.convert_to_tensor(VocabularyNumpyList)

# encoder
ids_from_token   = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None)
# decoder
token_from_id    = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None, invert=True)

logFileIterator = TokenParser.LogfileIterator(OutputDirectory)
tokenLogfileToProcess = logFileIterator.get_first_tokenlogfile()
dataset = None

while (tokenLogfileToProcess is not None):
    print("adding " + tokenLogfileToProcess + " to the training dataset")
    logTokenNumpyList = np.loadtxt(tokenLogfileToProcess, delimiter=",", dtype=str)
    TrainingLogTensor = tf.convert_to_tensor(logTokenNumpyList)
    ids_from_logfile = ids_from_token(TrainingLogTensor)
    sliced_ids_from_logfile = tf.data.Dataset.from_tensor_slices(ids_from_logfile)
    # each sequence entry contains seq_length number of elements
    sequences = sliced_ids_from_logfile.batch(seq_length+1, drop_remainder=True)
    # run the function split_input_target accross all entries of sequences
    if (dataset == None):
        dataset = sequences.map(TokenParser.split_input_target)
    else:
         dataset = dataset.concatenate(sequences.map(TokenParser.split_input_target))

    tokenLogfileToProcess = logFileIterator.get_next_tokenlogfile()

# JENTS: create an array of tensors (zero dimension tensors each containing one token)
saved_model = Directory + '/out/saved_model'

try:
    model = tf.keras.models.load_model(saved_model, compile=True)
except:
    model = None

# jents: organize them in batches and shuffle them.
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

dataset  # python convention to show the variable is ready

if ((model is None) or (force_train_model)):

    
    model = None


    # Length of the vocabulary in StringLookup Layer
    vocab_size = len(ids_from_token.get_vocabulary())
    print("vocab_size: ", vocab_size)


    # The embedding dimension, to be checked out
    embedding_dim = 256

    # Number of RNN units, to be checked out
    rnn_units = 1024  

    model = ModelClass.PrintfLogModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)


    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print("input_example_batch, example_batch_predictions.shape ", input_example_batch.shape, example_batch_predictions.shape)

    model.summary()

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

    ############################################################################################# Train the model


    model.compile(optimizer='adam', loss=loss)


    # Directory where the checkpoints will be saved
    checkpoint_dir = Directory + '/out/training_checkpoints'

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=False)
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    
    if (latest is not None):
        model.load_weights(latest)

    
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    
    model.save(saved_model)

############################################################################################# Train the model
# Checking the shape of the tensor
#print(tensor.shape)   -> je wil deze gebruiken
#print(tf.shape(tensor))

start = time.time()
TestFileNumpyList = np.loadtxt(TestFile, delimiter=",", dtype=str)
print(TestFileNumpyList)


idsTestFile = ids_from_token(TestFileNumpyList)  # dit vertaalt heel de array naar Id's - dit returned een array van Id's
print(idsTestFile)
inputTensor = tf.convert_to_tensor(idsTestFile, dtype='int64')

one_step_model = ModelClass.OneStep(model, token_from_id, ids_from_token)
probability = one_step_model.GetTokenProbability(inputTensor, '0F D6 20 07')
next_token_predicted = one_step_model.generate_one_step(inputTensor, states=None)


end = time.time()


LogFileNumpyList = np.loadtxt(LogFileTokens, delimiter=",", dtype=str)
window_size = 6
for i in range(LogFileNumpyList.shape[0] - (window_size + 1)):
    window    = LogFileNumpyList[i:i + window_size - 1]
    nextToken = LogFileNumpyList[i + window_size]
    idsLogFile = ids_from_token(window) 
    inputTensor = tf.convert_to_tensor(idsLogFile, dtype='int64')
    probability = one_step_model.GetTokenProbability(inputTensor, nextToken)
    print(window, "probability of next token ", nextToken, " = ", probability)
     



print('\nRun time:', end - start)


print("END")