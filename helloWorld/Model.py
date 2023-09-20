
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


def AnalyzePrintfFiles(inputDirectory, outputDirectory, temperature, seq_length):
    verificationFileIterator = TokenParser.LogfileIterator(inputDirectory)
    printfFileToProcess      = verificationFileIterator.get_first_tokenlogfile()

    while (printfFileToProcess is not None):
        outputFile = Path(printfFileToProcess).name.replace(".log", "_analyzedprintf_" + str(temperature) + ".txt")
        LogfileAnalyzed = os.path.join(outputDirectory, outputFile)

        AddProbabilitiesPrintfFileParallel(model, printfFileToProcess, LogfileAnalyzed, token_from_id, id_from_token, PrintfFromToken, temperature, seq_length)

        printfFileToProcess = verificationFileIterator.get_next_tokenlogfile()



def AddProbabilitiesPrintfFileParallel(Model, LogfileVerification, FileNameAnalyzed, token_from_id, id_from_token, PrintfFromToken, temperature, seq_length):

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
    num_batches_dataset     = dataSetTemp.reduce(0, lambda x, _: x + 1).numpy()

    return num_batches_dataset

def CreateDatasetFromDirectory(InputDirectory, OutputDirectory, batch_size, stride, shuffle, BUFFER_SIZE, seq_length, ReductionInSamples=None, removeDuplicates=False, verbose=False):
    
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


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.config.threading.set_inter_op_parallelism_threads(4)
#tf.config.threading.set_intra_op_parallelism_threads(4)

# https://www.tensorflow.org/text/tutorials/text_generation

force_train_model = False

TrainingsParameter = {"Network":"GRU",
                      "EPOCHS":1,
                      "Use_Validation":True,
                      "BatchSize":32 ,
                      "seq_length":8 ,
                      "removeDuplicates":False,
                      "stride_dataset": 5,
                      "embedding_dim":256,
                      "rnn_units":1024,
                      "shuffleTraining":True,
                      "BUFFER_SIZE":10000,
                      "SampleDataSet":False,   # does not really work, using stride is better
                    }


trainingSet = 4


if ( TrainingsParameter["SampleDataSet"]):
    # dit werkt niet goed. 
    numberOfBatchesToTrainPerEpoch    = 10
    numberOfBatchesToValidatePerEpoch = 10
    TrainingsParameter["BUFFER_SIZE"] = numberOfBatchesToTrainPerEpoch


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
    datasetForTraining, trainingFileList = CreateDatasetFromDirectory(InputDirectory=TrainingDirectory, 
                                                                      OutputDirectory=OutputDirectoryTraining, 
                                                                      batch_size=TrainingsParameter["BatchSize"], 
                                                                      stride=TrainingsParameter["stride_dataset"], 
                                                                      shuffle=TrainingsParameter["shuffleTraining"], 
                                                                      BUFFER_SIZE=TrainingsParameter["BUFFER_SIZE"],
                                                                      seq_length=TrainingsParameter["seq_length"], 
                                                                      removeDuplicates=TrainingsParameter["removeDuplicates"], 
                                                                      verbose = False) # , ReductionInSamples=10) #, removeDuplicates=removeDuplicates)        

################# Create a dataset for verification and throw all files that are in the verification output dir in it
datasetForVerification, verificationFileList = CreateDatasetFromDirectory(InputDirectory=VerificationDirectory, 
                                                                          OutputDirectory=OutputDirectoryVerification, 
                                                                          batch_size=TrainingsParameter["BatchSize"], 
                                                                          stride=TrainingsParameter["stride_dataset"], 
                                                                          shuffle=TrainingsParameter["shuffleTraining"], 
                                                                          BUFFER_SIZE=TrainingsParameter["BUFFER_SIZE"],
                                                                          seq_length=TrainingsParameter["seq_length"],
                                                                          removeDuplicates=TrainingsParameter["removeDuplicates"], 
                                                                          verbose = False)    

################## check whether the logfile tokens only contains tokens that are in the tokenfile

CheckCardinality = True
if (CheckCardinality):
    print("Checking datasets:")
    if (train_model):
        num_batches_datasetForTraining = GetDataSetInformation(datasetForTraining)
        print("num_batches_datasetForTraining: " + str(num_batches_datasetForTraining)) 

    num_batches_datasetForVerification = GetDataSetInformation(datasetForVerification)
    print("num_batches_datasetForVerification: " + str(num_batches_datasetForVerification)) 


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

    model

    if (TrainingsParameter["Network"] == "LSTM"):  # does not work yet
        model = ModelClass.LSTMPrintfLogModel(
            vocab_size=len(id_from_token.get_vocabulary()),
            embedding_dim=TrainingsParameter["embedding_dim"],
            rnn_units=TrainingsParameter["rnn_units"])
    if (TrainingsParameter["Network"] == "GRU"):
        model = ModelClass.GruPrintfLogModel(
            vocab_size=len(id_from_token.get_vocabulary()),
            embedding_dim=TrainingsParameter["embedding_dim"],
            rnn_units=TrainingsParameter["rnn_units"])
    
    if (TrainingsParameter["Network"] == "SimpleRnn"):    # supposed to be faster. I don't see it.
        model = ModelClass.SimpleRnnPrintfLogModel(
            vocab_size=len(id_from_token.get_vocabulary()),
            embedding_dim=TrainingsParameter["embedding_dim"],
            rnn_units=TrainingsParameter["rnn_units"])
        
    ############################################################################################# Train the model


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
    print("\nSettings used: ")

    ModelSummary = "EPOCHS: " + str(TrainingsParameter["EPOCHS"]) + "\n"
    ModelSummary = ModelSummary + "BATCH_SIZE:       " + str(TrainingsParameter["BatchSize"]) + "\n"
    ModelSummary = ModelSummary + "removeDuplicates: " + str(TrainingsParameter["removeDuplicates"]) + "\n"
    ModelSummary = ModelSummary + "stride:           " + str(TrainingsParameter["stride_dataset"]) + "\n"
    ModelSummary = ModelSummary + "seq_length:       " + str(TrainingsParameter["seq_length"]) + "\n"
    ModelSummary = ModelSummary + "trainingSet:      " + str(trainingSet) + "\n"
    ModelSummary = ModelSummary + "embedding_dim:    " + str(TrainingsParameter["embedding_dim"]) + "\n"
    ModelSummary = ModelSummary + "rnn_units:        " + str(TrainingsParameter["rnn_units"]) + "\n"


    if ( TrainingsParameter["SampleDataSet"]):
        ModelSummary = ModelSummary + "numberOfBatchesToTrainPerEpoch: " + str(numberOfBatchesToTrainPerEpoch) + "\n"
        ModelSummary = ModelSummary + "numberOfBatchesToValidatePerEpoch: " + str(numberOfBatchesToValidatePerEpoch) + "\n"

    print(ModelSummary + "\n")

    EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    Epoch_End_callback = None
    
    if ( TrainingsParameter["SampleDataSet"]):
        # dit werkt niet zo goed
        class EpochEndCallback(tf.keras.callbacks.Callback):
            def __init__(self, numberOfSamplesToTrainPerEpoch, numberOfSamplesToValidatePerEpoch):
                print("EpochEndCallback start newcreating new subsetForBatchTraining")
                self.numberOfBatchesToTrainPerEpoch    = numberOfSamplesToTrainPerEpoch
                self.numberOfBatchesToValidatePerEpoch = numberOfSamplesToValidatePerEpoch
                self.subsetForBatchTraining            = datasetForTraining.take(self.numberOfBatchesToTrainPerEpoch)
                PrintDatasetTuple(self.subsetForBatchTraining)
                self.subsetForBatchVerification        = None
                print("EpochEndCallback start newcreating new subsetForBatchTraining done")

            def on_epoch_end(self, epoch, logs=None):
                print("on_epoch_end -> Creating new subsetForBatchTraining")
                self.subsetForBatchTraining     = datasetForTraining.shuffle(TrainingsParameter["BUFFER_SIZE"]).take(self.numberOfBatchesToTrainPerEpoch)
                PrintDatasetTuple(self.subsetForBatchTraining)
                print("on_epoch_end Creating new subsetForBatchTraining done")
        

        Epoch_End_callback = EpochEndCallback(numberOfBatchesToTrainPerEpoch, numberOfBatchesToValidatePerEpoch)
        history = model.fit(Epoch_End_callback.subsetForBatchTraining, validation_data=Epoch_End_callback.subsetForBatchVerification, epochs=TrainingsParameter["EPOCHS"], callbacks=[checkpoint_callback, EarlyStopping_callback, Epoch_End_callback])
    else:
        #history = model.fit(datasetForTraining, validation_data=datasetForVerification, epochs=TrainingsParameter["EPOCHS"], steps_per_epoch=num_batches_datasetForTraining, callbacks=[checkpoint_callback, EarlyStopping_callback])
        history = model.fit(datasetForTraining, validation_data=datasetForVerification, epochs=TrainingsParameter["EPOCHS"], callbacks=[checkpoint_callback, EarlyStopping_callback])
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
        #losstraining = str(CalculateLossCrossEntropy(datasetForTraining, NumberOfSamplesForLossCalculation))
        #lossverification = str(CalculateLossCrossEntropy(datasetForVerification, NumberOfSamplesForLossCalculation))
        #scoreString = "loss at training: " + losstraining + ", Loss at verification: " + lossverification
        #print(scoreString)
        # Reset the standard output
        sys.stdout = original_stdout
else:
    losstraining = "Not Appl"


AnalyzePrintfFiles(VerificationDirectory, OutputDirectoryVerification, 0, TrainingsParameter["seq_length"])
 


end = time.time()

print('\nRun time:', end - start)
print("END")



