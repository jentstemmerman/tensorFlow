import tensorflow as tf
from tensorflow import keras
import TokenParser
import ModelClass
import numpy as np
import os

# https://www.tensorflow.org/text/tutorials/text_generation

TokenFileName = "Tokens.dat"
LogFileName   = "logfile.log"

TokenFileNameCvs = "TokenfileOnlyTokens.log"
TokenFileNameAndPrintfCvs = "TokenfileAndPrintf.log"
LogFileNameCvs   = "LogfileOnlyTokens.log"

seq_length = 10

TokenParser.process_files(TokenFileName,  TokenFileNameCvs)
TokenParser.process_files(LogFileName, LogFileNameCvs)

result = TokenParser.CrossReferenceTokens(LogFileNameCvs, TokenFileNameCvs)
print("Cross reference passed = ", result)

if (result):
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
    
    # JENTS: create an array of tensors (zero dimension tensors each containing one token)
    
    for ids in sliced_ids_from_logfile.take(10):
        print(token_from_id(ids).numpy().decode('utf-8'))

    # each sequence entry contains seq_length number of elements
    sequences = sliced_ids_from_logfile.batch(seq_length+1, drop_remainder=True)
    
    

    for seq in sequences.take(2):
        print(token_from_id(seq))

    # run the function split_input_target accross all entries of sequences
    dataset = sequences.map(TokenParser.split_input_target)

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

    dataset  # wtf is the function of this: python convention to show the variable is ready

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

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    sampled_indices

    print("Input:\n", token_from_id(input_example_batch[0]).numpy())
    print()
    print("Next Token Predictions:\n", token_from_id(sampled_indices).numpy())

    for token in token_from_id(sampled_indices).numpy():
        translation = TokenParser.GetTokenTranslation(token.decode('utf-8'), TokenFileName)
        print("token ", token.decode('utf-8'), " ", translation)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", example_batch_mean_loss)

    tf.exp(example_batch_mean_loss).numpy()
    
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
    
    EPOCHS = 20
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    #GetTokenTranslation(token, TokenFile)
    print("END")