import datetime
import sys

from tensorboard import program
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
import Mask



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

def AnalyzePrintfFile(*,
                      model,
                      userId,
                      progress_cb,
                      inputFile,
                      inferenceDataSetFile,
                      LogFileAnalyzedPath,
                      signatureInputDirectory,
                      similarityOutputDirectory,
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
    
    Inference.InferSimilarities(  model=model,
                                  userId=userId,
                                  progress_cb=progress_cb,
                                  numberOfThreads=numberOfThreads,
                                  Workers=Workers,
                                  itemList=itemList,
                                  DataTypeForTensors=DataTypeForTensors,
                                  LogfileVerification=printfFileToProcess,
                                  signatureInputDirectory=signatureInputDirectory,
                                  SimilarityOutputDirectory=similarityOutputDirectory,
                                  inferenceDataSetFile=inferenceDataSetFile,
                                  id_from_tokenList=id_from_tokenList,
                                  token_from_idList=token_from_idList,
                                  temperature=temperature,
                                  seq_length=seq_length)

    
    print("Done analyzing ", os.path.basename(printfFileToProcess), ".")



def AnalyzePrintfFiles(model, inputDirectory, outputDirectory, itemFile, temperature, seq_length, id_from_tokenList, token_from_idList, numberOfThreads, Workers, DataTypeForTensors):

    itemList = TokenParser.GetItemListToSearchFor(itemFile)
    verificationFileIterator = LogFileIterator.FileIterator(inputDirectory, ".log")
    printfFileToProcess      = verificationFileIterator.get_first_file()

    while (printfFileToProcess is not None):
        
        userId = 0
        temperature = 1
        
        FileNamesMachineReadablePooledVectors = Constants.GetLogFileMachineReadableVectorsPath(printfFileToProcess, userId)
        FileNamesHumanReadablePooledVectors = Constants.GetLogFileHumanReadableVectorsPath(printfFileToProcess, userId)
        FileNameAnalyzed = Constants.GetLogFileAnalyzedPath(printfFileToProcess, userId)
        inferenceDataSetFile = Constants.GetInferenceDataSetPath(printfFileToProcess, userId, temperature)
        similarityOutputDirectory = os.path.dirname(Constants.GetMachineReadableSimularitiesFile(printfFileToProcess, userId, ""))
        
        
        FileNamesMachineReadablePooledVectors["first"]     = os.path.join(os.path.basename(os.path.dirname(FileNamesMachineReadablePooledVectors["first"])), os.path.basename(FileNamesMachineReadablePooledVectors["first"]))
        FileNamesMachineReadablePooledVectors["last"]      = os.path.join(os.path.basename(os.path.dirname(FileNamesMachineReadablePooledVectors["last"])), os.path.basename(FileNamesMachineReadablePooledVectors["last"]))
        FileNamesMachineReadablePooledVectors["all"]       = os.path.join(os.path.basename(os.path.dirname(FileNamesMachineReadablePooledVectors["all"])), os.path.basename(FileNamesMachineReadablePooledVectors["all"]))
        FileNamesMachineReadablePooledVectors["predicted"] = os.path.join(os.path.basename(os.path.dirname(FileNamesMachineReadablePooledVectors["predicted"])), os.path.basename(FileNamesMachineReadablePooledVectors["predicted"]))
        FileNamesMachineReadablePooledVectors["all_weighted"] = os.path.join(os.path.basename(os.path.dirname(FileNamesMachineReadablePooledVectors["all_weighted"])), os.path.basename(FileNamesMachineReadablePooledVectors["all_weighted"]))
        FileNamesMachineReadablePooledVectors["all_attention_received"] = os.path.join(os.path.basename(os.path.dirname(FileNamesMachineReadablePooledVectors["all_attention_received"])), os.path.basename(FileNamesMachineReadablePooledVectors["all_attention_received"]))
        FileNamesMachineReadablePooledVectors["attention_received"] = os.path.join(os.path.basename(os.path.dirname(FileNamesMachineReadablePooledVectors["attention_received"])), os.path.basename(FileNamesMachineReadablePooledVectors["attention_received"]))

        
        FileNamesHumanReadablePooledVectors["first"]     = os.path.join(os.path.basename(os.path.dirname(FileNamesHumanReadablePooledVectors["first"])), os.path.basename(FileNamesHumanReadablePooledVectors["first"]))
        FileNamesHumanReadablePooledVectors["last"]      = os.path.join(os.path.basename(os.path.dirname(FileNamesHumanReadablePooledVectors["last"])), os.path.basename(FileNamesHumanReadablePooledVectors["last"]))
        FileNamesHumanReadablePooledVectors["all"]       = os.path.join(os.path.basename(os.path.dirname(FileNamesHumanReadablePooledVectors["all"])), os.path.basename(FileNamesHumanReadablePooledVectors["all"]))
        FileNamesHumanReadablePooledVectors["predicted"] = os.path.join(os.path.basename(os.path.dirname(FileNamesHumanReadablePooledVectors["predicted"])), os.path.basename(FileNamesHumanReadablePooledVectors["predicted"]))
        FileNamesHumanReadablePooledVectors["all_weighted"] = os.path.join(os.path.basename(os.path.dirname(FileNamesHumanReadablePooledVectors["all_weighted"])), os.path.basename(FileNamesHumanReadablePooledVectors["all_weighted"])) 
        FileNamesHumanReadablePooledVectors["all_attention_received"] = os.path.join(os.path.basename(os.path.dirname(FileNamesHumanReadablePooledVectors["all_attention_received"])), os.path.basename(FileNamesHumanReadablePooledVectors["all_attention_received"])) 
        FileNamesHumanReadablePooledVectors["attention_received"] = os.path.join(os.path.basename(os.path.dirname(FileNamesHumanReadablePooledVectors["attention_received"])), os.path.basename(FileNamesHumanReadablePooledVectors["attention_received"]))
                
        FileNameAnalyzed = os.path.join(os.path.basename(os.path.dirname(FileNameAnalyzed)), os.path.basename(FileNameAnalyzed))
        inferenceDataSetFile = os.path.join(os.path.basename(os.path.dirname(inferenceDataSetFile)), os.path.basename(inferenceDataSetFile))     
        
        FileNamesMachineReadablePooledVectors["first"]     = os.path.join(outputDirectory, FileNamesMachineReadablePooledVectors["first"])
        FileNamesMachineReadablePooledVectors["last"]      = os.path.join(outputDirectory, FileNamesMachineReadablePooledVectors["last"])
        FileNamesMachineReadablePooledVectors["predicted"] = os.path.join(outputDirectory, FileNamesMachineReadablePooledVectors["predicted"])
        FileNamesMachineReadablePooledVectors["all"]       = os.path.join(outputDirectory, FileNamesMachineReadablePooledVectors["all"])
        FileNamesMachineReadablePooledVectors["all_weighted"] = os.path.join(outputDirectory, FileNamesMachineReadablePooledVectors["all_weighted"])
        FileNamesMachineReadablePooledVectors["all_attention_received"] = os.path.join(outputDirectory, FileNamesMachineReadablePooledVectors["all_attention_received"])
        FileNamesMachineReadablePooledVectors["attention_received"] = os.path.join(outputDirectory, FileNamesMachineReadablePooledVectors["attention_received"])
        
        FileNamesHumanReadablePooledVectors["first"]     = os.path.join(outputDirectory, FileNamesHumanReadablePooledVectors["first"])
        FileNamesHumanReadablePooledVectors["last"]      = os.path.join(outputDirectory, FileNamesHumanReadablePooledVectors["last"])
        FileNamesHumanReadablePooledVectors["predicted"] = os.path.join(outputDirectory, FileNamesHumanReadablePooledVectors["predicted"])
        FileNamesHumanReadablePooledVectors["all"]       = os.path.join(outputDirectory, FileNamesHumanReadablePooledVectors["all"])
        FileNamesHumanReadablePooledVectors["all_weighted"] = os.path.join(outputDirectory, FileNamesHumanReadablePooledVectors["all_weighted"])
        FileNamesHumanReadablePooledVectors["all_attention_received"] = os.path.join(outputDirectory, FileNamesHumanReadablePooledVectors["all_attention_received"])
        FileNamesHumanReadablePooledVectors["attention_received"] = os.path.join(outputDirectory, FileNamesHumanReadablePooledVectors["attention_received"])
                
        LogFileAnalyzedPath = os.path.join(outputDirectory, FileNameAnalyzed)
        inferenceDataSetFilePath = os.path.join(outputDirectory, inferenceDataSetFile)
        
        signatureInputDirectory = Constants.GetUserSpecificBaseDirectory("SignatureType", userId)
        
        Inference.InferSimilarities(  model=model, 
                                        userId=userId,
                                        numberOfThreads=numberOfThreads, 
                                        Workers=Workers, 
                                        itemList=itemList, 
                                        DataTypeForTensors=DataTypeForTensors, 
                                        LogfileVerification=printfFileToProcess,
                                        signatureInputDirectory=signatureInputDirectory,
                                        SimilarityOutputDirectory=similarityOutputDirectory,
                                        inferenceDataSetFile=inferenceDataSetFilePath, 
                                        id_from_tokenList=id_from_tokenList,
                                        token_from_idList=token_from_idList, 
                                        temperature=temperature, 
                                        seq_length=seq_length)


        printfFileToProcess = verificationFileIterator.get_next_file()



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
        num_layers = 24
        num_heads = 8
        dropout_rate = 0.1

        if (modelSource["Network"] == "Transformer"):
            import Transformer
            d_model = Transformer.Transformer.CalculateEmbeddingDimensions(featureList=modelFeatureList)[0]                # to be fixed for multifeature
            dff = d_model * 4
            
            # Parameters, only one feature supported yet
            input_vocab_size = modelFeatureList[0]["vocab_size"]
            target_vocab_size = modelFeatureList[0]["vocab_size"]
        
            model = Transformer.Transformer(
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                input_vocab_size=input_vocab_size,       
                target_vocab_size=target_vocab_size,    
                featureList=modelFeatureList, 
                seq_length=TrainingsParameter["Model"]["seq_length"],
                dropout_rate=dropout_rate, 
                DataTypeForTensors=TrainingsParameter["DataTypeForTensors"])
        else:
            import Transformer_DecoderOnly
            d_model = Transformer_DecoderOnly.Transformer.CalculateEmbeddingDimensions(featureList=modelFeatureList)[0]                # to be fixed for multifeature
            dff = d_model * 4
            model = Transformer_DecoderOnly.Transformer(
                num_layers=num_layers,
                featureList=modelFeatureList,
                num_heads=num_heads,
                dff=dff,
                DataTypeForTensors=TrainingsParameter["DataTypeForTensors"],

                seq_length=TrainingsParameter["Model"]["seq_length"],
                dropout_rate=dropout_rate)
            
        optimizer = 'adam'
        import CustomSchedule
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=200000,
            alpha=0.1
        )
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

    EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    keras_log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    keras_log_file =  os.path.join(TrainingCheckpointDir, "TensorBoardLogs", keras_log_file)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=keras_log_file, histogram_freq=1, update_freq=1000)

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", keras_log_file, "--host", "0.0.0.0", "--port", "6006"])
    url = tb.launch()
    print(f"TensorBoard running at {url}")

    #batch_x, batch_y = next(iter(datasetForTraining))
    #input_shape = batch_x.shape[1:]  # Determine the input shape from the batch (exclude batch dimension)
    #model.build(input_shape=(None, *input_shape))  


    #history = model.fit(datasetForTraining, epochs=TrainingsParameter["EPOCHS"], callbacks=[checkpoint_callback,  ])
    model.fit(datasetForTraining, validation_data=datasetForVerification, epochs=TrainingsParameter["EPOCHS"], callbacks=[EarlyStopping_callback, tensorboard_callback])
    #history = model.fit(datasetForTraining, validation_split=0.2, epochs=TrainingsParameter["EPOCHS"], callbacks=[neptune_callback,TrainBatchEnd_callback])
    sys.stdout = sys.__stdout__

    return model
