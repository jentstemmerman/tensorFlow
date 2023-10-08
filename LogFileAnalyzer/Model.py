import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import TokenParser
import Inference
import ModelClass
import Dataset
import os
from pathlib import Path
import sys
import worker

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""  # or even "-1"

numberOfThreads = 8
print("Calling CreateWorkers numberOfThreads: ", numberOfThreads)
#if __name__ == "__main__":
    # Main process uses GPU

with tf.device('/CPU:0'):
    
    Workers = worker.CreateWorkers(numberOfThreads)

    if (Workers == None):
        print("Could not start workers")


    DataTypeForTensors = 'int64'

    def CrossReferenceTokens(OutputDirectoryTraining, OutputDirectoryVerification, VocabularyFile):
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

        mean_loss = tf.reduce_mean(batch_mean_losses)

        return mean_loss.numpy()

    def GetEncoderDecoderFromTokenFile(OriginalTokenFileName, VocabularyFile):
        TokenParser.process_files(OriginalTokenFileName, VocabularyFile)
        VocabularyNumpyList = np.loadtxt(VocabularyFile, delimiter=",", dtype=str)
        id_from_tokenNumpy  = VocabularyNumpyList
        VocabularyTensor    = tf.convert_to_tensor(VocabularyNumpyList)
        
        # encoder
        id_from_token = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None)
        # decoder
        token_from_id = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None, invert=True)

        return id_from_token, token_from_id, id_from_tokenNumpy

    def AnalyzePrintfFiles(model, inputDirectory, outputDirectory, temperature, seq_length, token_from_id, id_from_tokenNumpy, numberOfThreads, Workers):
        #if __name__ == '__main__':
        #    multiprocessing.freeze_support()
        #    multiprocessing.set_start_method('spawn')
        verificationFileIterator = TokenParser.LogfileIterator(inputDirectory)
        printfFileToProcess      = verificationFileIterator.get_first_tokenlogfile()

        while (printfFileToProcess is not None):
            outputFile = Path(printfFileToProcess).name.replace(".log", "_analyzedprintf_" + str(temperature) + ".txt")
            LogfileAnalyzed = os.path.join(outputDirectory, outputFile)

            Inference.AddProbabilitiesPrintfFileParallel(model=model, numberOfThreads=numberOfThreads, Workers=Workers, DataTypeForTensors=DataTypeForTensors, LogfileVerification=printfFileToProcess, FileNameAnalyzed=LogfileAnalyzed, token_from_id=token_from_id, id_from_token=id_from_token, PrintfFromToken=PrintfFromToken, temperature=temperature, seq_length=seq_length, id_from_tokenNumpy=id_from_tokenNumpy)

            printfFileToProcess = verificationFileIterator.get_next_tokenlogfile()


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


    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #tf.config.threading.set_inter_op_parallelism_threads(4)
    #tf.config.threading.set_intra_op_parallelism_threads(4)

    # https://www.tensorflow.org/text/tutorials/text_generation

    force_train_model     = False
    CreateNewOutDirectory = False
    AnalyzeFiles          = True

    TrainingsParameter = {"Model": {"Network":"LSTM",
                                    "returnState":True,
                                    "embedding_dim":256,
                                    "rnn_units":2048,
                                    "seq_length":16 ,
                                    },
                        "EPOCHS":5,
                        "Use_Validation":False,
                        "BatchSize":32 ,
                        "removeDuplicates":True,
                        "stride_dataset": 1,
                        "shuffleTraining":True,
                        "BUFFER_SIZE":100000,
                        "SampleDataSet":False,   # does not really work, using stride is better
                        }

    ValidationsParameter = {"temperature": 1 }


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
    TrainingDataSetFile         = OutputDirectory + os.path.sep + "TrainingDataSetFile.dataset"
    VerificationDataSetFile     = OutputDirectory + os.path.sep + "VerificationDataSetFile.dataset"
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

    if (CreateNewOutDirectory):

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
    id_from_token, token_from_id, id_from_tokenNumpy = GetEncoderDecoderFromTokenFile(OriginalTokenFileName, VocabularyFile)

    print("Creating datasets:")

    if (train_model):

        ################# Create a dataset for training and throw all files that are in the training output dir in it
        if os.path.exists(TrainingDataSetFile.strip()):
            datasetForTraining = Dataset.LoadDataset(TrainingDataSetFile.strip())
            trainingFileList = None
        else:
            datasetForTraining, trainingFileList = Dataset.CreateDatasetFromDirectory(  InputDirectory=TrainingDirectory, 
                                                                                        OutputDirectory=OutputDirectoryTraining, 
                                                                                        id_from_token=id_from_token,
                                                                                        batch_size=TrainingsParameter["BatchSize"], 
                                                                                        stride=TrainingsParameter["stride_dataset"], 
                                                                                        shuffle=TrainingsParameter["shuffleTraining"], 
                                                                                        BUFFER_SIZE=TrainingsParameter["BUFFER_SIZE"],
                                                                                        seq_length=TrainingsParameter["Model"]["seq_length"], 
                                                                                        removeDuplicates=TrainingsParameter["removeDuplicates"], 
                                                                                        verbose = False) # , ReductionInSamples=10) #, removeDuplicates=removeDuplicates)        
            Dataset.SaveDataset(datasetForTraining, TrainingDataSetFile.strip())

    ################# Create a dataset for verification and throw all files that are in the verification output dir in it


    if os.path.exists(VerificationDataSetFile.strip()):
        datasetForVerification = Dataset.LoadDataset(VerificationDataSetFile.strip())
        verificationFileList = None
    else:
        datasetForVerification, verificationFileList = Dataset.CreateDatasetFromDirectory(  InputDirectory=VerificationDirectory, 
                                                                                            OutputDirectory=OutputDirectoryVerification, 
                                                                                            id_from_token=id_from_token,
                                                                                            batch_size=TrainingsParameter["BatchSize"], 
                                                                                            stride=TrainingsParameter["stride_dataset"], 
                                                                                            shuffle=TrainingsParameter["shuffleTraining"], 
                                                                                            BUFFER_SIZE=TrainingsParameter["BUFFER_SIZE"],
                                                                                            seq_length=TrainingsParameter["Model"]["seq_length"],
                                                                                            removeDuplicates=TrainingsParameter["removeDuplicates"], 
                                                                                            verbose = False)    
        Dataset.SaveDataset(datasetForVerification, VerificationDataSetFile.strip())

    ################## check whether the logfile tokens only contains tokens that are in the tokenfile



    CrossReferenceTokens(OutputDirectoryTraining, OutputDirectoryVerification, VocabularyFile)

    if (train_model):
        model

        if (TrainingsParameter["Model"]["Network"] == "LSTM"):  # does not work yet
            model = ModelClass.LSTMPrintfLogModel(
                vocab_size=len(id_from_token.get_vocabulary()),
                embedding_dim=TrainingsParameter["Model"]["embedding_dim"],
                rnn_units=TrainingsParameter["Model"]["rnn_units"])
        if (TrainingsParameter["Model"]["Network"] == "GRU"):
            model = ModelClass.GruPrintfLogModel(
                vocab_size=len(id_from_token.get_vocabulary()),
                embedding_dim=TrainingsParameter["Model"]["embedding_dim"],
                rnn_units=TrainingsParameter["Model"]["rnn_units"],
                return_state=TrainingsParameter["Model"]["returnState"])
        
        if (TrainingsParameter["Model"]["Network"] == "SimpleRnn"):    # supposed to be faster. I don't see it.
            model = ModelClass.SimpleRnnPrintfLogModel(
                vocab_size=len(id_from_token.get_vocabulary()),
                embedding_dim=TrainingsParameter["Model"]["embedding_dim"],
                rnn_units=TrainingsParameter["Model"]["rnn_units"])
            
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
        ModelSummary = ModelSummary + "seq_length:       " + str(TrainingsParameter["Model"]["seq_length"]) + "\n"
        ModelSummary = ModelSummary + "trainingSet:      " + str(trainingSet) + "\n"
        ModelSummary = ModelSummary + "embedding_dim:    " + str(TrainingsParameter["Model"]["embedding_dim"]) + "\n"
        ModelSummary = ModelSummary + "rnn_units:        " + str(TrainingsParameter["Model"]["rnn_units"]) + "\n"


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
                    Dataset.PrintDatasetTuple(self.subsetForBatchTraining)
                    self.subsetForBatchVerification        = None
                    print("EpochEndCallback start newcreating new subsetForBatchTraining done")

                def on_epoch_end(self, epoch, logs=None):
                    print("on_epoch_end -> Creating new subsetForBatchTraining")
                    self.subsetForBatchTraining     = datasetForTraining.shuffle(TrainingsParameter["BUFFER_SIZE"]).take(self.numberOfBatchesToTrainPerEpoch)
                    Dataset.PrintDatasetTuple(self.subsetForBatchTraining)
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
            if (trainingFileList is not None):
                print("Files used to train: \n")
                print(*trainingFileList, sep="\n")

            if (verificationFileList is not None):
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

    if (AnalyzeFiles):
        AnalyzePrintfFiles(model, inputDirectory=VerificationDirectory, outputDirectory=OutputDirectoryVerification, temperature=ValidationsParameter["temperature"], seq_length=TrainingsParameter["Model"]["seq_length"], token_from_id=token_from_id, id_from_tokenNumpy=id_from_tokenNumpy, numberOfThreads=numberOfThreads, Workers=Workers)

    end = time.time()

    print('\nRun time:', end - start)
    print("END")



