import tensorflow as tf
from tensorflow import keras
import TokenParser
import ModelClass
import numpy as np
import os


force_train_model = False


EPOCHS = 1

#tf.config.threading.set_intra_op_parallelism_threads(1)

# https://www.tensorflow.org/text/tutorials/text_generation
trainingSet = 1

if (trainingSet == 1):
    Directory = "TrainingSet1"
    TokenFileName = Directory + "/Tokens.dat"
    LogFileName   = Directory + "/logfile.log"
else:
    Directory = "TrainingSet2"
    TokenFileName = Directory + "/Tokens.dat"
    LogFileName   = Directory + "/1340_Neo_KioxiaBiCS6_8ch-4TB_SIE_Regression_Combination_QuickCheck_drive_log.log"

TokenFileNameCvs = Directory + "/out/TokenfileOnlyTokens.log"
TokenFileNameAndPrintfCvs = Directory + "/out/TokenfileAndPrintf.log"
LogFileNameCvs   = Directory+ "/out/LogfileOnlyTokens.log"

seq_length = 5

TokenParser.process_files(TokenFileName, TokenFileNameCvs)
TokenParser.process_files(LogFileName, LogFileNameCvs)

result = TokenParser.CrossReferenceTokens(LogFileNameCvs, TokenFileNameCvs)
print("Cross reference passed = ", result)

if (result == -1):
    print("Logfile contains tokens that are not in the tokenfile!!")
    exit()

print("Reading dictionary into memory, not sure I will need it")
# using loadtxt()
tokenNumpyList = np.loadtxt(TokenFileNameCvs, delimiter=",", dtype=str)
tokenList      = tf.convert_to_tensor(tokenNumpyList)
print("Reading dictionary into memory DONE")

print("Reading logfile into memory, might want to improve this later")
logTokenNumpyList = np.loadtxt(LogFileNameCvs, delimiter=",", dtype=str)
logtokenList      = tf.convert_to_tensor(logTokenNumpyList)

print("The Shape of tokenList tensor is",tf.shape(tokenList))
print("The Shape of logtokenList tensor is",tf.shape(logtokenList))
print("Reading logfile into memory DONE")

# encoder
ids_from_token   = tf.keras.layers.StringLookup(vocabulary=tokenList, mask_token=None)
# decoder
token_from_id    = tf.keras.layers.StringLookup(vocabulary=tokenList, mask_token=None, invert=True)

ids_from_logfile = ids_from_token(logtokenList)

sliced_ids_from_logfile = tf.data.Dataset.from_tensor_slices(ids_from_logfile)

# each sequence entry contains seq_length number of elements
sequences = sliced_ids_from_logfile.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(2):
    print(token_from_id(seq))

# run the function split_input_target accross all entries of sequences
dataset = sequences.map(TokenParser.split_input_target)
# JENTS: create an array of tensors (zero dimension tensors each containing one token)
saved_model = Directory + '/out/saved_model'

try:
    model = tf.keras.models.load_model(saved_model, compile=True)
except:
    model = None

if ((model is None) or (force_train_model)):
    for ids in sliced_ids_from_logfile.take(10):
        print(token_from_id(ids).numpy().decode('utf-8'))



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
        print("batch_size, sequence_length, vocab_size : ", example_batch_predictions.shape)

    model.summary()

    ############################################################################################# Try without training

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    sampled_indices

    print("Input:\n", token_from_id(input_example_batch[0]).numpy())
    print()
    print("Next Token Predictions:\n", token_from_id(sampled_indices).numpy())

    for token in token_from_id(sampled_indices).numpy():
        translation = TokenParser.GetTokenTranslation(token.decode('utf-8'), TokenFileName)
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

# dit werkt niet in load model mode!!!

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print("batch_size, sequence_length, vocab_size : ", example_batch_predictions.shape)

#print("type(dataset) = ", type(dataset)) 
#loss, acc = model.evaluate(dataset, verbose=2)
#print("Trained model, accuracy: {:5.2f}%".format(100 * acc))


print("END training")