import time
import tensorflow as tf
import numpy as np
import TokenParser
import Inference
import ModelClass
import Dataset
import os
from pathlib import Path
import sys
import worker
import Logger
import LogFileIterator
import MultiDimensionalModel
import ModelCallbacks
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

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

def GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory, OutputVocabularyFile, itemList):

    id_from_tokenList   = []
    token_from_idList   = []

    for itemsInLine in itemList:
        # only create encoder-decoder for items that are cathegorical.
        if (itemsInLine["Categorical"] == "True"):
            if ("VocabularyFile" in itemsInLine):
                OriginalTokenFile = itemsInLine["VocabularyFile"]
                
                if (type(OriginalTokenFile) == str):

                    OriginalTokenFileName = OriginalTokenFileDirectory + os.path.sep + OriginalTokenFile
                    
                    # only do one item process at a time
                    tokenListList = TokenParser.process_files(OriginalTokenFileName, OutputVocabularyFile, [itemsInLine])
                    VocabularyNumpy = [inner_list[0] for inner_list in tokenListList]
                    VocabularyNumpy = np.array(VocabularyNumpy)
                else:
                    raise ValueError("VocabularyFile " + OutputVocabularyFile + " not found")
            elif ("RangeMin" in itemsInLine) and ("RangeMax" in itemsInLine):
                rangeMin = itemsInLine["RangeMin"]
                RangeMax = itemsInLine["RangeMax"]
                VocabularyNumpy = np.arange(rangeMin, RangeMax + 1, 1)
                VocabularyNumpy = VocabularyNumpy.astype(str)
            else:
                raise ValueError("VocabularyFile " + OutputVocabularyFile + " or range not found")
            
            VocabularyTensor  = tf.convert_to_tensor(VocabularyNumpy)

            # encoder
            id_from_token = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None)
            id_from_tokenList.append(id_from_token)
            
            # decoder
            token_from_id = tf.keras.layers.StringLookup(vocabulary=VocabularyTensor, mask_token=None, invert=True)
            token_from_idList.append(token_from_id)
        else:
            id_from_tokenList.append(None)
            token_from_idList.append(None)
            
    return id_from_tokenList, token_from_idList

def AnalyzePrintfFiles(model, inputDirectory, outputDirectory, itemFile, temperature, seq_length, id_from_tokenList, token_from_idList, numberOfThreads, Workers):

    itemList = TokenParser.GetItemListToSearchFor(itemFile)
    verificationFileIterator = LogFileIterator.LogfileIterator(inputDirectory)
    printfFileToProcess      = verificationFileIterator.get_first_tokenlogfile()

    while (printfFileToProcess is not None):
        outputFile = Path(printfFileToProcess).name.replace(".log", "_analyzedprintf_" + str(temperature) + ".mdtxt")
        LogfileAnalyzed = os.path.join(outputDirectory, outputFile)

        Inference.AddProbabilitiesPrintfFileParallel(model=model, numberOfThreads=numberOfThreads, Workers=Workers, itemList=itemList, DataTypeForTensors=DataTypeForTensors, LogfileVerification=printfFileToProcess, FileNameAnalyzed=LogfileAnalyzed, id_from_tokenList=id_from_tokenList, token_from_idList=token_from_idList, temperature=temperature, seq_length=seq_length)

        printfFileToProcess = verificationFileIterator.get_next_tokenlogfile()

def CalculateAverageProbabilityNextToken(OutputDirectoryVerification, temperature):

    calculateAverage = 0

    if (calculateAverage):
        verificationFileIterator = TokenParser.LogfileIterator(OutputDirectoryVerification)
        tokenLogfileToProcess    = verificationFileIterator.get_first_tokenlogfile()

        if (tokenLogfileToProcess == None):
            exit()
    
        TestFileNumpyList = np.loadtxt(tokenLogfileToProcess, delimiter=",", dtype=str)

        one_step_model = ModelClass.OneStep(model, token_from_idList, id_from_tokenList, temperature=temperature)
        window_size = 8
        Accumulator = 0
        AccumulatorCount = 0
        for i in range(TestFileNumpyList.shape[0] - (window_size + 1)):
            window    = TestFileNumpyList[i:i + window_size - 1]
            nextToken = TestFileNumpyList[i + window_size]
            idsLogFile = id_from_tokenList(window) 
            inputTensor = tf.convert_to_tensor(idsLogFile, dtype=DataTypeForTensors)
            probability = one_step_model.GetTokenProbability(inputTensor, nextToken)
            AccumulatorCount = AccumulatorCount + 1
            Accumulator = Accumulator + probability

    return Accumulator / AccumulatorCount

def CreateModelFeatureListFromItemList(itemList, id_from_tokenList):
    modelFeatureList = []

    for index, item in enumerate(itemList):
        name = "Layer_" + str(index)
        assert(("Categorical" in item) == True)
        if (item["Categorical"] == "True"):
            name = name + "_Cathegorical_layer"
            assert(id_from_tokenList[index] != None, "id_from_tokenList[" + str(index) + "] is None")
            embedded_dimension = 256  #TODO, adapt
            modelFeatureList.append({"embedded": True, "vocab_size" : len(id_from_tokenList[index].get_vocabulary()), "embedding_dim": embedded_dimension, "layer_name": name})
        else:
            assert(id_from_tokenList[index] == None, "id_from_tokenList[" + str(index) + "] is None")
            maskValue = item["MaskValue"]
            name = name + "_NonCathegorical_layer"
            modelFeatureList.append({"embedded": False, "layer_name": name, "MaskValue": maskValue})
    return modelFeatureList

def GetModel(modelSource, modelFeatureList):
    model = MultiDimensionalModel.LSTMPrintfLogMultiModel(  modelFeatureList, 
                                                            TrainingsParameter["Model"]["rnn_units"],
                                                            TrainingsParameter["Model"]["seq_length"], 
                                                            dense_layer = TrainingsParameter["Model"]["dense_layer"], 
                                                            return_sequences = TrainingsParameter["Model"]["return_states"] , 
                                                            datatype = DataTypeForTensors)
    
    loss_function = MultiDimensionalModel.Custom_loss_function(modelFeatureList, DataTypeForTensors)  

    if (modelSource["Source"] == "NewModel"):
        model.compile(optimizer='adam', loss=loss_function)

    elif (modelSource["Source"]  == "FromCheckpoint"):
        checkpointPath        = modelSource["CheckpointPath"] + os.path.sep + "ckpt_" + str(modelSource["CheckpointNumber"])
        model.compile(optimizer='adam', loss=loss_function)
        model.load_weights(checkpointPath)

    elif (modelSource["Source"]  == "FromFile"):
        model = tf.keras.models.load_model(ModelDirectory, compile=False )
        loss_function = MultiDimensionalModel.Custom_loss_function(modelFeatureList, DataTypeForTensors)  
        model.compile(optimizer='adam', loss=loss_function)

    return model

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.config.threading.set_inter_op_parallelism_threads(4)
#tf.config.threading.set_intra_op_parallelism_threads(4)

# https://www.tensorflow.org/text/tutorials/text_generation

force_train_model     = True
CreateNewOutDirectory = True
AnalyzeFiles          = True
loadCheckPoint        = 2

#tf.config.run_functions_eagerly(True)  
#tf.data.experimental.enable_debug_mode()    


TrainingsParameter = {"Model": {"Network":"LSTMMultiModel",
                                "returnState":True,
                                "embedding_dim":128,   # set to -1 for no embedding layer. but you need to use float for DataTypeForTensors then
                                "rnn_units":1024,
                                "seq_length":32 ,
                                "dense_layer": True,
                                "return_states": True
                                },
                      "EPOCHS":8,
                      "Use_Validation":False,
                      "BatchSize":64,
                      "removeDuplicates":False,
                      "num_parallel_calls":64,
                      "stride_dataset": 512,
                      "shuffleTraining":False,
                      "BUFFER_SIZE":65536,
                      "SampleDataSet":False,   # does not really work, using stride is better
                    }

embedding_layer_used = (TrainingsParameter["Model"]["embedding_dim"] != -1)
DataTypeForTensors   = 'float32'
ValidationsParameter = {"temperature": 1 }

trainingSet = 3                

if ( TrainingsParameter["SampleDataSet"]):
    # dit werkt niet goed. 
    numberOfBatchesToTrainPerEpoch    = 10
    numberOfBatchesToValidatePerEpoch = 10
    TrainingsParameter["BUFFER_SIZE"] = numberOfBatchesToTrainPerEpoch

itemFile  = "Itemfile.txt"  # file containing the regexes that we train on
Directory = "../TrainingSet" + str(trainingSet)

TrainingDirectory           = Directory       + os.path.sep + "Training"
VerificationDirectory       = Directory       + os.path.sep + "Verification"
OriginalTokenFileDirectory  = Directory 
OutputDirectory             = Directory       + os.path.sep + "out"
ScoreFile                   = OutputDirectory + os.path.sep + "ScoreFile.txt"
TrainingLogFile             = OutputDirectory + os.path.sep + "TrainingLog.txt"
SummaryFile                 = OutputDirectory + os.path.sep + "Summary.txt"
VocabularyFile              = OutputDirectory + os.path.sep + "TokenfileOnlyTokens.dat"
TrainingDataSetFile         = OutputDirectory + os.path.sep + "TrainingDataSetFile.dataset"
VerificationDataSetFile     = OutputDirectory + os.path.sep + "VerificationDataSetFile.dataset"
ModelDirectory              = OutputDirectory + os.path.sep + "SavedModel"
TrainingCheckpointDir       = OutputDirectory + os.path.sep + "TrainingCheckPoints"
OutputDirectoryTraining     = OutputDirectory + os.path.sep + "Training"
OutputDirectoryVerification = OutputDirectory + os.path.sep + "Verification"

itemList         = TokenParser.GetItemListToSearchFor(itemFile)
id_from_tokenList, token_from_idList = GetEncodersDecodersFromTokenFiles(OriginalTokenFileDirectory, VocabularyFile, itemList)  
modelFeatureList = CreateModelFeatureListFromItemList(itemList, id_from_tokenList)


if (CreateNewOutDirectory) or (force_train_model):
    train_model = True
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
    modelSource = {"Source": "NewModel"}
else:
    modelSource = {"Source": "FromCheckpoint", "CheckpointNumber": 2, "CheckpointPath": TrainingCheckpointDir}
    train_model = False

model = GetModel(modelSource, modelFeatureList)


################## check whether the logfile tokens only contains tokens that are in the tokenfile
CrossReferenceTokens(OutputDirectoryTraining, OutputDirectoryVerification, VocabularyFile)
################# Create a dataset for verification and throw all files that are in the verification output dir in it
Force_dataset_creation = True

if (Force_dataset_creation and  os.path.exists(VerificationDataSetFile.strip())):
    datasetForVerification = Dataset.LoadDataset(VerificationDataSetFile.strip())
    verificationFileList = None
else:
    datasetForVerification, verificationFileList = Dataset.CreateDatasetFromDirectory(  InputDirectory=VerificationDirectory, 
                                                                                    OutputDirectory=OutputDirectoryVerification, 
                                                                                    itemFile=itemFile,
                                                                                    embedding_layer_used=embedding_layer_used,
                                                                                    id_from_tokenList=id_from_tokenList,
                                                                                    batch_size=TrainingsParameter["BatchSize"], 
                                                                                    stride=TrainingsParameter["stride_dataset"], 
                                                                                    shuffle=TrainingsParameter["shuffleTraining"], 
                                                                                    BUFFER_SIZE=TrainingsParameter["BUFFER_SIZE"],
                                                                                    seq_length=TrainingsParameter["Model"]["seq_length"],
                                                                                    DataTypeForTensors = DataTypeForTensors,
                                                                                    removeDuplicates=TrainingsParameter["removeDuplicates"], 
                                                                                    num_parallel_calls = TrainingsParameter["num_parallel_calls"],
                                                                                    verbose = False)    
    Dataset.SaveDataset(datasetForVerification, VerificationDataSetFile.strip())


if (train_model):
    ################# Create a dataset for training and throw all files that are in the training output dir in it
#    if os.path.exists(TrainingDataSetFile.strip()):
#        datasetForTraining = Dataset.LoadDataset(TrainingDataSetFile.strip())
#        trainingFileList = None
#    else:
    datasetForTraining, trainingFileList = Dataset.CreateDatasetFromDirectory(  InputDirectory=TrainingDirectory, 
                                                                                OutputDirectory=OutputDirectoryTraining, 
                                                                                itemFile=itemFile,
                                                                                embedding_layer_used=embedding_layer_used,
                                                                                id_from_tokenList=id_from_tokenList,
                                                                                batch_size=TrainingsParameter["BatchSize"], 
                                                                                stride=TrainingsParameter["stride_dataset"], 
                                                                                shuffle=TrainingsParameter["shuffleTraining"], 
                                                                                BUFFER_SIZE=TrainingsParameter["BUFFER_SIZE"],
                                                                                seq_length=TrainingsParameter["Model"]["seq_length"], 
                                                                                DataTypeForTensors = DataTypeForTensors,
                                                                                removeDuplicates=TrainingsParameter["removeDuplicates"], 
                                                                                num_parallel_calls = TrainingsParameter["num_parallel_calls"],
                                                                                verbose = False, 
                                                                                ) # , ReductionInSamples=10) #, removeDuplicates=removeDuplicates)        
#    Dataset.SaveDataset(datasetForTraining, TrainingDataSetFile.strip())

        
    ############################################################################################# Train the model
    
    sys.stdout = Logger.Logger(TrainingLogFile)
    
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(TrainingCheckpointDir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=False)
  
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

    TrainBatchEnd_callback = ModelCallbacks.DebugCallback()
    EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    Epoch_End_callback = None

    run = neptune.init_run(project='Log/TensorFlowLogger', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NDRiM2JmMC1kNGI4LTQ4NDEtYjc5NS0wYmY2YTdhZjg1YTUifQ==')

    neptune_callback = NeptuneCallback(run=run) 

    history = model.fit(datasetForTraining, epochs=TrainingsParameter["EPOCHS"], callbacks=[checkpoint_callback, neptune_callback,TrainBatchEnd_callback ])
    

    # not sure this is still needed
    sample_element = next(iter(datasetForTraining))[0]
    temp_predicted_logits = model(inputs=sample_element)
    model.compute_output_shape(sample_element.shape)
    ###
    
    sys.stdout = sys.__stdout__

    try:
        model.save(ModelDirectory)
    except:
        model = None

    ############################################################################################# Use the model

    end = time.time()

    print("Files used to train: \n")
    print(*trainingFileList, sep="\n")

    print("\nCalculation of losses: ")
    # Open a file in write mode
    with open(ScoreFile, 'w') as f:
        # Redirect the standard output to the file
        original_stdout = sys.stdout
        sys.stdout = f
        
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
sample_element = next(iter(datasetForVerification))[0]
temp_predicted_logits = model(inputs=sample_element)
print(sample_element)
#print(temp_predicted_logits[0])
print(temp_predicted_logits[1])

if (AnalyzeFiles):
    
    numberOfThreads = 7  # 1 for the main process
    print("Calling CreateWorkers numberOfThreads: ", numberOfThreads)

    with tf.device('/CPU:0'):
        
        Workers = worker.CreateWorkers(numberOfThreads, id_from_tokenList)
    
        if (Workers == None):
            print("Could not start workers")
        else:
            AnalyzePrintfFiles(model, 
                               inputDirectory=VerificationDirectory, 
                               outputDirectory=OutputDirectoryVerification, 
                               itemFile=itemFile, 
                               temperature=ValidationsParameter["temperature"], 
                               seq_length=TrainingsParameter["Model"]["seq_length"], 
                               id_from_tokenList = id_from_tokenList,
                               token_from_idList=token_from_idList,
                               numberOfThreads=numberOfThreads, 
                               Workers=Workers)
            
            worker.KillWorkers(Workers)

end = time.time()

print('\nRun time:', end - start)
print("END")



