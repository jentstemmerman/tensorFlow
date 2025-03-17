import sys
sys.path.append('../Server')

import Constants  # type: ignore

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
import DefaultTrainingSettings
import math



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

def AnalyzePrintfFile(model, 
                      inputFile, 
                      inferenceDataSetFile, 
                      LogFileAnalyzedPath, 
                      FileNameMachineReadablePooledVectors, 
                      FileNameHumanReadablePooledVectors, 
                      itemFile, 
                      temperature, 
                      seq_length, 
                      id_from_tokenList, 
                      token_from_idList,
                      numberOfThreads, 
                      Workers, 
                      DataTypeForTensors):
    
    itemList = TokenParser.GetItemListToSearchFor(itemFile)

    printfFileToProcess      = inputFile

    LogfileAnalyzed = LogFileAnalyzedPath
    
    Inference.AddProbabilitiesPrintfFileParallel(model=model, 
                                                 numberOfThreads=numberOfThreads, 
                                                 Workers=Workers, 
                                                 itemList=itemList, 
                                                 DataTypeForTensors=DataTypeForTensors, 
                                                 LogfileVerification=printfFileToProcess, 
                                                 inferenceDataSetFile=inferenceDataSetFile, 
                                                 FileNameAnalyzed=LogfileAnalyzed, 
                                                 FileNameHumanReadablePooledVectors=FileNameHumanReadablePooledVectors,
                                                 FileNameMachineReadablePooledVectors=FileNameMachineReadablePooledVectors, 
                                                 id_from_tokenList=id_from_tokenList, 
                                                 token_from_idList=token_from_idList, 
                                                 temperature=temperature, 
                                                 seq_length=seq_length)
    
    print("Done analyzing ", os.path.basename(printfFileToProcess), ".")



def AnalyzePrintfFiles(model, inputDirectory, outputDirectory, itemFile, temperature, seq_length, id_from_tokenList, token_from_idList, numberOfThreads, Workers, DataTypeForTensors):

    itemList = TokenParser.GetItemListToSearchFor(itemFile)
    verificationFileIterator = LogFileIterator.FileIterator(inputDirectory, ".log")
    printfFileToProcess      = verificationFileIterator.get_first_tokenlogfile()

    while (printfFileToProcess is not None):
        
        userId = 0
        temperature = 1
        
        FileNameMachineReadablePooledVectors = Constants.GetLogFileMachineReadableVectorsPath(printfFileToProcess, userId, temperature)
        FileNameHumanReadablePooledVectors = Constants.GetLogFileHumanReadableVectorsPath(printfFileToProcess, userId, temperature)
        FileNameAnalyzed = Constants.GetLogFileAnalyzedPath(printfFileToProcess, userId)
        inferenceDataSetFile = Constants.GetInferenceDataSetPath(printfFileToProcess, userId, temperature)
        
        FileNameMachineReadablePooledVectors = os.path.join(os.path.basename(os.path.dirname(FileNameMachineReadablePooledVectors)), os.path.basename(FileNameMachineReadablePooledVectors))
        FileNameHumanReadablePooledVectors = os.path.join(os.path.basename(os.path.dirname(FileNameHumanReadablePooledVectors)), os.path.basename(FileNameHumanReadablePooledVectors))
        FileNameAnalyzed = os.path.join(os.path.basename(os.path.dirname(FileNameAnalyzed)), os.path.basename(FileNameAnalyzed))
        inferenceDataSetFile = os.path.join(os.path.basename(os.path.dirname(inferenceDataSetFile)), os.path.basename(inferenceDataSetFile))     
        
        FileNameMachineReadablePooledVectors = os.path.join(outputDirectory, FileNameMachineReadablePooledVectors)
        FileNameHumanReadablePooledVectors = os.path.join(outputDirectory, FileNameHumanReadablePooledVectors)
        LogFileAnalyzedPath = os.path.join(outputDirectory, FileNameAnalyzed)
        inferenceDataSetFilePath = os.path.join(outputDirectory, inferenceDataSetFile)
       
        Inference.AddProbabilitiesPrintfFileParallel(model=model, 
                                                     numberOfThreads=numberOfThreads, 
                                                     Workers=Workers, 
                                                     itemList=itemList, 
                                                     DataTypeForTensors=DataTypeForTensors, 
                                                     LogfileVerification=printfFileToProcess, 
                                                     inferenceDataSetFile=inferenceDataSetFilePath, 
                                                     FileNameAnalyzed=LogFileAnalyzedPath, 
                                                     FileNameHumanReadablePooledVectors=FileNameHumanReadablePooledVectors, 
                                                     FileNameMachineReadablePooledVectors=FileNameMachineReadablePooledVectors, 
                                                     id_from_tokenList=id_from_tokenList, 
                                                     token_from_idList=token_from_idList, 
                                                     temperature=temperature, 
                                                     seq_length=seq_length)

        printfFileToProcess = verificationFileIterator.get_next_tokenlogfile()

def CalculateAverageProbabilityNextToken(OutputDirectoryVerification, temperature, DataTypeForTensors):

    calculateAverage = 0

    if (calculateAverage):
        verificationFileIterator = TokenParser.FileIterator(OutputDirectoryVerification)
        tokenLogfileToProcess    = verificationFileIterator.get_first_tokenlogfile()

        if (tokenLogfileToProcess == None):
            exit()
    
        TestFileNumpyList = np.loadtxt(tokenLogfileToProcess, delimiter=",", dtype=str)

        one_step_model = ModelClass.OneStep(model, token_from_idList, temperature=temperature)
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

def GetMaskValueList(id_from_tokenList):   # JENTS BUG
    maskValueList = []

    for id_from_token in id_from_tokenList:
        maskValue = id_from_token(Dataset.Universal_Mask_Token)
        maskValueList.append(maskValue)

    return maskValueList

def CreateModelFeatureListFromItemList(itemList, id_from_tokenList):
    modelFeatureList = []

    for index, item in enumerate(itemList):
        name = "Layer_" + str(index)
        assert(("Categorical" in item) == True)
        if (item["Categorical"] == "True"):
            name = name + "_Cathegorical_layer"
            #assert(id_from_tokenList[index] is not None, "id_from_tokenList[" + str(index) + "] is None")
            VocabularySize = len(id_from_tokenList[index].get_vocabulary())
            embedded_dimension_rule_of_thumb = int(VocabularySize ** (1/2) )   
            embedded_dimension_power_of_two = int(2**math.ceil(math.log2(embedded_dimension_rule_of_thumb)))
            modelFeatureList.append({"embedded": True, "vocab_size" : len(id_from_tokenList[index].get_vocabulary()), "embedding_dim": embedded_dimension_power_of_two, "layer_name": name})
        else:
            # assert(id_from_tokenList[index] is None, "id_from_tokenList[" + str(index) + "] is None")
            name = name + "_NonCathegorical_layer"
            modelFeatureList.append({"embedded": False, "layer_name": name})
    return modelFeatureList

             
 
def GetModel(modelSource, maskValueList, modelFeatureList, TrainingsParameter):

    if (modelSource["Network"] == "LSTMMultiModel"):
        model = MultiDimensionalModel.LSTMPrintfLogMultiModel(  modelFeatureList, 
                                                                TrainingsParameter["Model"]["rnn_units"],
                                                                TrainingsParameter["Model"]["seq_length"], 
                                                                return_sequences = TrainingsParameter["Model"]["return_states"] , 
                                                                datatype = TrainingsParameter["DataTypeForTensors"])
        optimizer = 'adam'
    elif (modelSource["Network"] == "Transformer") or (modelSource["Network"] == "Transformer_DecoderOnly"):    
        num_layers = 16
        num_heads = 32
        dropout_rate = 0.1
        import Transformer
        d_model=Transformer.Transformer.CalculateEmbeddingDimensions(featureList=modelFeatureList)[0]                # to be fixed for multifeature
        dff = d_model * 4
        # Parameters, only one feature supported yet (don't know whether it will ever be possible)
        input_vocab_size = modelFeatureList[0]["vocab_size"]
        target_vocab_size = modelFeatureList[0]["vocab_size"]

        if (modelSource["Network"] == "Transformer"):
            model = Transformer.Transformer(
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                input_vocab_size=input_vocab_size,       
                target_vocab_size=target_vocab_size,    
                featureList=modelFeatureList, 
                seq_length=TrainingsParameter["Model"]["seq_length"],
                dropout_rate=dropout_rate)
        else:
            import Transformer_DecoderOnly
            model = Transformer_DecoderOnly.Transformer(
                num_layers=num_layers,
                featureList=modelFeatureList,
                num_heads=num_heads,
                dff=dff,
                datatype=TrainingsParameter["DataTypeForTensors"],
                input_vocab_size=input_vocab_size,       
                seq_length=TrainingsParameter["Model"]["seq_length"],
                dropout_rate=dropout_rate)
            
        optimizer = 'adam'
        learning_rate = Transformer.CustomSchedule(d_model)

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    loss_function = MultiDimensionalModel.Custom_loss_function(modelFeatureList, maskValueList, TrainingsParameter["DataTypeForTensors"])  


    if (modelSource["Source"] == "NewModel"):
        model.compile(optimizer=optimizer, loss=loss_function)
        hash = None
        
    elif (modelSource["Source"]  == "FromCheckpoint"):
        checkpointPath        = modelSource["CheckpointPath"] + os.path.sep + "ckpt_" + str(modelSource["CheckpointNumber"])
        model.compile(optimizer=optimizer, loss=loss_function)
        model.load_weights(checkpointPath)
        hash = GetHashFromModel(model)
    elif (modelSource["Source"]  == "FromFile"):
        ModelDirectory = modelSource["ModelDirectory"]
        

        custom_objects = None
        model = tf.keras.models.load_model(ModelDirectory, custom_objects=custom_objects, compile=False )
        loss_function = MultiDimensionalModel.Custom_loss_function(modelFeatureList, maskValueList, TrainingsParameter["DataTypeForTensors"])  
        model.compile(optimizer=optimizer, loss=loss_function)
        hash = GetHashFromModel(model)
    
    print("Model hash: ", hash)
    
    return model

def GetHashFromModel(model):
    import hashlib
    model_hash = hashlib.md5(model.to_json().encode('utf-8')).hexdigest()
    return model_hash




def TrainModel(model, datasetForTraining, datasetForVerification, TrainingsParameter, TrainingCheckpointDir, TrainingLogFile=None):

    if (TrainingLogFile is not None): 
        sys.stdout = Logger.Logger(TrainingLogFile)
    
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(TrainingCheckpointDir, "ckpt_{epoch}.keras")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=False)
  
    print("\nSettings used: ")

    ModelSummary = "EPOCHS: " + str(TrainingsParameter["EPOCHS"]) + "\n"
    ModelSummary = ModelSummary + "BATCH_SIZE:       " + str(TrainingsParameter["BatchSize"]) + "\n"
    ModelSummary = ModelSummary + "removeDuplicates: " + str(TrainingsParameter["removeDuplicates"]) + "\n"
    ModelSummary = ModelSummary + "stride:           " + str(TrainingsParameter["stride_dataset"]) + "\n"
    ModelSummary = ModelSummary + "seq_length:       " + str(TrainingsParameter["Model"]["seq_length"]) + "\n"

    ModelSummary = ModelSummary + "rnn_units:        " + str(TrainingsParameter["Model"]["rnn_units"]) + "\n"

    if ( TrainingsParameter["SampleDataSet"]):
        ModelSummary = ModelSummary + "numberOfBatchesToTrainPerEpoch: " + str(numberOfBatchesToTrainPerEpoch) + "\n"
        ModelSummary = ModelSummary + "numberOfBatchesToValidatePerEpoch: " + str(numberOfBatchesToValidatePerEpoch) + "\n"

    print(ModelSummary + "\n")

    TrainBatchEnd_callback = None # ModelCallbacks.DebugCallback()
    EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    Epoch_End_callback = None

    run = neptune.init_run(project='Log/TensorFlowLogger', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NDRiM2JmMC1kNGI4LTQ4NDEtYjc5NS0wYmY2YTdhZjg1YTUifQ==')

    neptune_callback = NeptuneCallback(run=run) 

    #batch_x, batch_y = next(iter(datasetForTraining))
    #input_shape = batch_x.shape[1:]  # Determine the input shape from the batch (exclude batch dimension)
    #model.build(input_shape=(None, *input_shape))  


    #history = model.fit(datasetForTraining, epochs=TrainingsParameter["EPOCHS"], callbacks=[checkpoint_callback,  ])
    history = model.fit(datasetForTraining, validation_data=datasetForVerification, epochs=TrainingsParameter["EPOCHS"], callbacks=[neptune_callback])
    #history = model.fit(datasetForTraining, validation_split=0.2, epochs=TrainingsParameter["EPOCHS"], callbacks=[neptune_callback,TrainBatchEnd_callback])
    sys.stdout = sys.__stdout__

    return model
