
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import ModelClass
import os
testcode = False

if (testcode):


    testArray = tf.convert_to_tensor(np.array([1,2,3,4,5,6,7,8,9,10]))
    print(testArray.shape)
    TestArrayTensorSlices = tf.data.Dataset.from_tensor_slices(testArray) 
    print(TestArrayTensorSlices.shape) 
    batched_dataset = TestArrayTensorSlices.batch(2, drop_remainder=True)
    print(batched_dataset.shape) 




force_train_model = False

EPOCHS = 10

#tf.config.threading.set_intra_op_parallelism_threads(1)

# https://www.tensorflow.org/text/tutorials/text_generation
trainingSet = 2

if (trainingSet == 1):
    Directory             = "TrainingSet1"
    OriginalTokenFileName = Directory + "/Tokens.dat"
    OriginalLogFileName   = Directory + "/logfile.log"
    TestFile              = Directory + "/TestData.log"
else:
    Directory             = "TrainingSet2"
    OriginalTokenFileName = Directory + "/Tokens.dat"
    OriginalLogFileName   = Directory + "/1340_Neo_KioxiaBiCS6_8ch-4TB_SIE_Regression_Combination_QuickCheck_drive_log.log"
    TestFile              = Directory + "/TestData.log"

VocabularyFile            = Directory + "/out/TokenfileOnlyTokens.log"
LogFileTokens             = Directory+ "/out/LogfileOnlyTokens.log"
#TokenFileNameAndPrintfCvs = Directory + "/out/TokenfileAndPrintf.log"

seq_length = 5

############### create files that only contain tokens:
TokenParser.process_files(OriginalTokenFileName, VocabularyFile)
TokenParser.process_files(OriginalLogFileName,   LogFileTokens)

################## check whether the logfile tokens only contains tokens that are in the tokenfile
result = TokenParser.CrossReferenceTokens(LogFileTokens, VocabularyFile)
print("Cross reference passed = ", result)

if (result == -1):
    print("Logfile contains tokens that are not in the tokenfile!!")
    exit()

##################################################################################################

print("Reading dictionary into memory, not sure I will need it")
# using loadtxt()
VocabularyNumpyList = np.loadtxt(VocabularyFile, delimiter=",", dtype=str)
VocabularyTensor    = tf.convert_to_tensor(VocabularyNumpyList)
print("Reading dictionary into memory DONE")

print("Reading logfile into a ndarray, might want to improve this later")
logTokenNumpyList = np.loadtxt(LogFileTokens, delimiter=",", dtype=str)
TrainingLogTensor = tf.convert_to_tensor(logTokenNumpyList)

print("The Shape of tokenList tensor is ", VocabularyTensor.shape)
print("The Shape of logtokenList tensor is ", TrainingLogTensor.shape)
print("Reading logfile into memory DONE")


# encoder
ids_from_token   = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None)
# decoder
token_from_id    = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None, invert=True)

ids_from_logfile = ids_from_token(TrainingLogTensor)

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
    for ids in sliced_ids_from_logfile.take(10):
        print(token_from_id(ids).numpy().decode('utf-8'))
    
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


#for input_example_batch, target_example_batch in dataset.take(1):
#    example_batch_predictions = model(input_example_batch)
#    print("input_example_batch.shape : ", input_example_batch.shape)
#    print("example_batch_predictions.shape : ", example_batch_predictions.shape)

TestFileNumpyList = np.loadtxt(TestFile, delimiter=",", dtype=str)
print(TestFileNumpyList)


idsTestFile = ids_from_token(TestFileNumpyList)  # dit vertaalt heel de array naar Id's - dit returned een array van Id's
print(idsTestFile)

idsTestFile = tf.expand_dims(idsTestFile, 0)

TestLogDataset = tf.data.Dataset.from_tensor_slices(idsTestFile) # dit is niet goed



batched_dataset = TestLogDataset.batch(seq_length, drop_remainder=True)
# the wrong dimension

# Create multiple 1-dimensional tensors
input_shape = (None, 5)
tensor1 = tf.constant([0, 1, 2, 3, 4], dtype='int64')
tensor2 = tf.constant([5, 6, 7, 8, 9], dtype='int64')
batched_tensor = tf.stack([tensor1, tensor2])
#batched_tensor = tf.expand_dims(tensor1, axis=0)
# Convert to a batch of tensors


states = None
predicted_logits, states = model(inputs=batched_tensor)

one_step_model = ModelClass.OneStep(model, token_from_id, ids_from_token)

start = time.time()
states = None
next_char = tf.constant(TestFileNumpyList)
test = ids_from_token(next_char)
predicted_logits, states = model(inputs=test, states=states,
                                          return_state=True)

input_ids = test.to_tensor()

result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)



print("END")