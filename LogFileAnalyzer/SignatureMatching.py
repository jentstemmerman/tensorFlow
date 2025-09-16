import sys
sys.path.append('../LogFileAnalyzer')

from flask import jsonify
import tensorflow as tf
import BinaryFileHandler

import numpy as np
import Constants     # type: ignore
import SignatureMatching
import json
import DefaultTrainingSettings
import os
import time
import LogFileIterator
import math
import SimularityGraph
import InferenceAssembly
import Model


def GetInitialInference(DataTypeForTensors, id_from_tokenList, window_size, one_step_model, sharedTokenList, startIndex):
    result = InferenceAssembly.AssembleInputsThread(id_from_tokenList, 0, DataTypeForTensors, startIndex, 1, sharedTokenList, None, window_size)

    inputTensor = result[1]   
    nexttokenIds = result[2]     
       
    _, _, _, _, pooledLogitsVectorData = one_step_model.generate_steps_optimized(inputTensor, nexttokenIds, states=None)
    numberOfVectors = pooledLogitsVectorData['vector'].shape[1]
    return numberOfVectors 
 
def CalculateSignatureDirectlyFromLogFile(model, logFileName, startLine, numberOfLines, itemList, id_from_tokenList, token_from_idList, temperature, window_size, DataTypeForTensors):
    import Logfile
    import ModelClass
    import Dataset
    
    AssertionError(numberOfLines <= window_size)  # this is not supported yet
    
    signatureLogFile = Logfile.Logfile(logFileName, itemList, DataTypeForTensors) 
    
    #tokenList = signatureLogFile.GetTokens(startLine, startLine + numberOfLines - 1)
    tokenList = signatureLogFile.GetTokens(startLine, startLine + window_size - 1)
    
    tokenIdList = []
    
    requiredNumberOfMaskTokens = window_size - numberOfLines    
       
    mask_token_filler_list = [Dataset.Universal_Mask_Token] * requiredNumberOfMaskTokens
    
    
    for columnIndex, id_from_token in enumerate(id_from_tokenList):
        columnList = [column[columnIndex] for column in tokenList]
        columnList.extend(mask_token_filler_list)  # add the mask tokens to the end of the column
               
        if (id_from_token is None):
            idColumn = tf.convert_to_tensor(columnList, dtype=DataTypeForTensors)
        else:
            idColumn = id_from_token(columnList)
        
        tokenIdList.append(idColumn)
        # After the loop, stack all idColumns along the last dimension to form the input tensor
        tokenIdList = tf.stack(tokenIdList, axis=-1)


    # Add a batch dimension of 1 to tokenIdList
    tokenIdList = tf.expand_dims(tokenIdList, axis=0)
    # Cast tokenIdList to float32
    tokenIdList = tf.cast(tokenIdList, DataTypeForTensors)
    
    one_step_model = ModelClass.OneStep(model, DataTypeForTensors, seq_length=window_size, itemList=itemList, Tokens_from_idList=None, id_from_tokenList=None, temperature=temperature)
    
    _, pooledLogitsVectorData = one_step_model.generate_hidded_states(tokenIdList)


    #receivedAttention = GetReceivedAttention(binaryFileHandlerReceivedAttention, startLine, numberOfLines, sequence_length, batch_size=1, DataTypeForTensors=DataTypeForTensors)
    
    hiddenVectors = pooledLogitsVectorData['hidden_state_all']

    hiddenVectors = hiddenVectors[:, 0 : numberOfLines, :]
    
    signature = SignatureMatching.CalculateSignatureFromHiddenVectors(hiddenVectors, None)  # No received attention in this case

    return signature



def read_signature_from_file(filename, userId=0):
    with open(filename, 'r') as file:
        signature_data = json.load(file)    
        
        signature_text = signature_data['signature_text']
        logFileName = signature_data['logFileName']
        modelDirectory = signature_data['modelDirectory']
        modelHash = signature_data['modelHash']
        startLine = signature_data['startLine']
        endLine = signature_data['endLine']
        DataTypeForTensors = signature_data['DataTypeForTensors']
        savedSignature = signature_data['signature']
        
        numberOfPooledVectors = endLine - startLine + 1
        
        recalculate_signature = True
        if recalculate_signature:
            FileNamesMachineReadablePooledVectors = Constants.GetLogFileMachineReadableVectorsPath(logFileName, userId)
            
            calculatedSignature, modelHash, _ = SignatureMatching.CalculateSignature(FileNamesMachineReadablePooledVectors, startLine, endLine - startLine + 1, DataTypeForTensors)
    

            savedSignature_arr = np.array(savedSignature)
            calculatedSignature_arr = np.array(calculatedSignature)
            
            if savedSignature_arr.shape != calculatedSignature_arr.shape:
                print("Warning: signature shapes are different:", savedSignature_arr.shape, "vs", calculatedSignature_arr.shape)
            elif not np.allclose(savedSignature_arr, calculatedSignature_arr, atol=1e-6):
                print("Warning: signature2 is different from signature1")

            signature_tensor = tf.convert_to_tensor(calculatedSignature, dtype=DataTypeForTensors)
        else:
            signature_tensor = tf.convert_to_tensor(savedSignature, dtype=DataTypeForTensors)
 
        return numberOfPooledVectors, DataTypeForTensors, signature_tensor, modelHash, signature_text, logFileName, modelDirectory, startLine, endLine

# This function is used in the simularity file calculation
#@tf.function
def CalculateSignatureFromHiddenVectors(batchedvectors, receivedAttention):
    usePooling = True
    useAveragePooling = True
    useMaxPooling = False
    
    if usePooling:
        
        if receivedAttention is not None:
            reshaped_attention = tf.expand_dims(receivedAttention, axis=-1)
            
            batchedvectors = batchedvectors * reshaped_attention
                
        batchedvectors = tf.nn.l2_normalize(batchedvectors, axis=-1)
        
        if useAveragePooling:
            pooledAverageVector = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(batchedvectors)
        else:
            pooledAverageVector = None   # this might not work
        
        if useMaxPooling:
            pooledMaxVector = tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last")(batchedvectors)
            
            if pooledAverageVector is not None:
                pooledVector = tf.concat([pooledAverageVector, pooledMaxVector], axis=-1)

        else:
            pooledVector = pooledAverageVector    
            
        # pooledVector = tf.nn.l2_normalize(pooledVector, axis=-1)
    else:        
        pooledVector = batchedvectors[:, -1, :]

    return pooledVector


#  startVectorOffset 0 means the first line in the logfile
def CalculateSignature(FileNamesMachineReadablePooledVectors, startLine, numberOfLines, DataTypeForTensors):
    
    binaryFileHandlerFirstLogits       = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["first"],     data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerLastLogits        = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["last"],      data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)        
    binaryFileHandlerPredictedLogits   = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["predicted"], data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerAllLogits         = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["all"] ,      data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerReceivedAttention = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["attention_received"], data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None) 

    modelHash1 = binaryFileHandlerFirstLogits.get_modelHash()
    modelHash2 = binaryFileHandlerLastLogits.get_modelHash()
    modelHash3 = binaryFileHandlerPredictedLogits.get_modelHash()
    modelHash4 = binaryFileHandlerAllLogits.get_modelHash()
    modelHash5 = binaryFileHandlerReceivedAttention.get_modelHash()
    
    assert startLine >= 1, f"Impossible startline {startLine}, line 0 is never predicted!"
    
    assert modelHash1 == modelHash2, f"Model hashes from first and last logits file do not match. {modelHash1} != {modelHash2}"
    assert modelHash1 == modelHash3, f"Model hashes from first and all logits file do not match. {modelHash1} != {modelHash3}"
    assert modelHash1 == modelHash4, f"Model hashes from first and predicted logits file do not match. {modelHash1} != {modelHash4}"
    assert modelHash1 == modelHash5, f"Model hashes from first and Received Attention file do not match. {modelHash1} != {modelHash5}"
    
    sequence_length = binaryFileHandlerFirstLogits.get_sequenceLength()
    
    receivedAttention = None # GetReceivedAttention(binaryFileHandlerReceivedAttention, startLine, numberOfLines, sequence_length, batch_size=1, DataTypeForTensors=DataTypeForTensors)
    
    hiddenVectors = GetHiddenVectorsOfLines(binaryFileHandlerAllLogits, binaryFileHandlerLastLogits, startLine, numberOfLines, sequence_length, batch_size=1)
    
    signature = CalculateSignatureFromHiddenVectors(hiddenVectors, receivedAttention)
    
    signature = tf.squeeze(signature, axis=0)           # remove the batch dimension from the tensor
    hiddenVectors = tf.squeeze(hiddenVectors, axis=0)   # remove the batch dimension from the tensor
    
    return signature, modelHash1, hiddenVectors

# This function returns a batch of attention received 
def GetReceivedAttention(binaryFileHandlerReceivedAttention, startLine, numberOfLines, sequence_length, batch_size, DataTypeForTensors=tf.float32):

    startVectorOffsetInReceivedAttentionFile = startLine - 1   
    numberOfRecordsToRead = numberOfLines + batch_size - 1
    
    attentionRecords = binaryFileHandlerReceivedAttention.read_records(startVectorOffsetInReceivedAttentionFile, numberOfRecordsToRead)

    receivedAttention = tf.signal.frame(attentionRecords, numberOfLines, 1, axis=0)
    
    contrib_counts = tf.range(numberOfLines, 0, -1, dtype=DataTypeForTensors)    
        
    # Create a mask using the contrib_counts
    # First create a range for sequence positions
    seq_positions = tf.range(sequence_length, dtype=DataTypeForTensors)

    # Reshape for proper broadcasting
    contrib_counts_reshaped = tf.reshape(contrib_counts, [1, numberOfLines, 1])  # [1, numberOfLines, 1]
    seq_positions = tf.reshape(seq_positions, [1, 1, sequence_length])           # [1, 1, sequence_length]

    # Create mask: only pay attention to positions where seq_position < line's contrib count
    attention_mask = tf.cast(seq_positions < contrib_counts_reshaped, DataTypeForTensors)

    # Apply mask to receivedAttention
    # If batch_size > 1, tile the mask across the batch dimension
    if batch_size > 1:
        attention_mask = tf.tile(attention_mask, [batch_size, 1, 1])
    
    masked_attention = receivedAttention * attention_mask
    
    received_attention = tf.reduce_sum(masked_attention, axis=-1)
    
    
    return received_attention    



# This function returns a batch of hidden vectors, starting from startLine, each with a length of numberOfLines
# @tf.function
def GetHiddenVectorsOfLines(binaryFileHandlerAllLogits, binaryFileHandlerLastLogits, startLine, numberOfLines, sequence_length, batch_size):
    startVectorOffsetInAllLogitsFile = (startLine - 1) * sequence_length   #this is the transformation to get the hidden vector of startLine as it is generated in the first timeslice of the sequence
    
    hiddenVectors = binaryFileHandlerAllLogits.read_records(startVectorOffsetInAllLogitsFile, sequence_length * batch_size)
    hiddenVectors = tf.reshape(hiddenVectors, (batch_size, sequence_length, -1))
    
    if (numberOfLines < sequence_length):
        hiddenVectors = hiddenVectors[:, 0 : numberOfLines, :]
    else:
        hiddenVectorsPart_1 = hiddenVectors[:, 0 : sequence_length - 1, :]
        newStartLine = startLine + sequence_length - 1
        startVectorOffsetInLastLogitsFile = newStartLine - sequence_length + 1
        numberOfLinesRemainder = numberOfLines - (sequence_length - 1)
        
        hiddenVectorsPart_2 = binaryFileHandlerLastLogits.read_records(startVectorOffsetInLastLogitsFile, batch_size + numberOfLinesRemainder - 1)  # waarom -1 hier jents
        hiddenVectorsPart_2 = tf.signal.frame(hiddenVectorsPart_2, numberOfLinesRemainder, 1, axis=0)
        hiddenVectorsPart_2 = tf.reshape(hiddenVectorsPart_2, (batch_size, numberOfLinesRemainder, -1))
        
        hiddenVectors = tf.concat([hiddenVectorsPart_1, hiddenVectorsPart_2], axis=1)
  
    return hiddenVectors

# @tf.function                
def CalculateSimularityBatched(signature, tensor, receivedAttention):
    pooledVector = CalculateSignatureFromHiddenVectors(tensor, receivedAttention)
    signature = tf.linalg.l2_normalize(signature, axis=-1)
    pooledVector = tf.linalg.l2_normalize(pooledVector, axis=-1)
    
    similarity = tf.keras.losses.cosine_similarity(signature, pooledVector)
    
    return similarity

def GetSignaturesFromDirectoryAndRecalculateDirectlyFromLogFile(model, userId, itemList, id_from_tokenList, token_from_idList, temperature, window_size):
    signatureDirectory = Constants.GetUserSpecificBaseDirectory("SignatureType", userId)
    signatureIterator = LogFileIterator.FileIterator(signatureDirectory, Constants.GetSignatureFilesExtention())

    signature_file = signatureIterator.get_first_file()
    numberOfSignatureFiles = 0
    
    numberOfPooledVectorsList = []
    signatureNames = []
    signatureTensors = []
    signatureFiles = []
       
    
    while (signature_file is not None):
        numberOfSignatureFiles += 1
        
        signature_name = os.path.splitext(os.path.basename(signature_file))[0]
                
        numberOfPooledVectors, DataTypeForTensors, signature_tensor, modelHash, signature_text, logFileName, modelDirectory, startLine, endLine = read_signature_from_file(signature_file, userId)
        
        logFileDirectory = Constants.GetUserSpecificBaseDirectory("LogType", userId)
        
        logFileName = os.path.join(logFileDirectory, logFileName)
        # Check if the log file referenced by the signature exists
        if not os.path.exists(logFileName):
            print(f"Log file {logFileName} referenced in signature {signature_file} does not exist. Skipping.")
            signature_file = signatureIterator.get_next_file()
            continue
        
        numberOfLines = endLine - startLine + 1
        
        signature_tensor = CalculateSignatureDirectlyFromLogFile(model, logFileName, startLine, numberOfLines, itemList, id_from_tokenList, token_from_idList, temperature, window_size, DataTypeForTensors)
        signature_tensor = tf.squeeze(signature_tensor, axis=0)  # remove the batch dimension from the tensor
        signatureNames.append(signature_name)
        signatureFiles.append(signature_file)
        numberOfPooledVectorsList.append(numberOfPooledVectors)
        signatureTensors.append(signature_tensor)
        
        import Signature
        
        signatureTest = Signature.Signature(
                                                logSnipped=None, 
                                                logFile=logFileName, 
                                                linesIndexList=None, 
                                                startLine=startLine,
                                                endLine=endLine,
                                                itemList=itemList, 
                                                model = model,
                                                id_from_tokenList = id_from_tokenList,
                                                temperature=temperature,
                                                seq_length=window_size,
                                                DataTypeForTensors=DataTypeForTensors )
        
        signature_tensor2 = signatureTest.CalculateSignature()
        
        signature_file = signatureIterator.get_next_file()
        
    return numberOfSignatureFiles, numberOfPooledVectorsList, signatureNames, signatureTensors, signatureFiles
    

def GetSignaturesFromDirectory(userId, itemList, tokenList=None, PrintfList=None):
    import TokenParser
    
    
    signatureDirectory = Constants.GetUserSpecificBaseDirectory("SignatureType", userId)
    signatureIterator = LogFileIterator.FileIterator(signatureDirectory, Constants.GetSignatureFilesExtention())

    signature_file = signatureIterator.get_first_file()
    numberOfSignatureFiles = 0
    
    numberOfPooledVectorsList = []
    signatureNames = []
    signatureTensors = []
    signatureFiles = []
    
    while (signature_file is not None):
        numberOfSignatureFiles += 1
        
        signature_name = os.path.splitext(os.path.basename(signature_file))[0]
                
        numberOfPooledVectors, _, signature_tensor, modelHashFromSignature, _, _, _, _, _ = SignatureMatching.read_signature_from_file(signature_file)
        
        signatureNames.append(signature_name)
        signatureFiles.append(signature_file)
        numberOfPooledVectorsList.append(numberOfPooledVectors)
        signatureTensors.append(signature_tensor)
        
        signature_file = signatureIterator.get_next_file()
        
    return numberOfSignatureFiles, numberOfPooledVectorsList, signatureNames, signatureTensors, signatureFiles
    

# limitation: the output of this file starts at linenumber 0 + max signature length - 1, so a very large signature length will cause a lot of empty lines at the start of the file
#tf.function
def CreateSimularityFilesOfLogFile(LogFileName, signatureDirectory, userId, DataTypeForTensors):
    
    if not os.path.exists(LogFileName):
        print(f"[CreateSimularityFilesOfLogFile] LogFileName {LogFileName} does not exist.")
        return False
    
    
    # first, create list of signature files
    signatureIterator = LogFileIterator.FileIterator(signatureDirectory, Constants.GetSignatureFilesExtention())
    
    numberOfPooledVectorsList = []
    signatureNames = []
    signatureTensors = []
    signatureFiles = []
    machineReadableOutputFiles = []
    humanReadableOutputFiles = []
    humanReadableOutputFileHandlers = []
    outputBinaryFileHandlers = []
    modelHashFromSignatureFileList = []
    
    signature_file = signatureIterator.get_first_file()
    
    if signature_file is not None:
        numberOfPooledVectors, _, _, _, _, _, _, _, _ = SignatureMatching.read_signature_from_file(signature_file)
    else:
        print("[CreateSimularityFilesOfLogFile] No signature files found.")
        return False
    
    temperature = 1
    max_SignatureLength = 0
    numberOfSignatureFiles = 0
    
    while (signature_file is not None):
        
        machineReadableOutputFile = Constants.GetMachineReadableSimularitiesFile(LogFileName, userId, signature_file)
        humanReadableOutputFile = Constants.GetHumanReadableSimularitiesFile(LogFileName, userId, signature_file)
        
        # it is possible that this function already ran, and machineReadableOutputFile and humanReadableOutputFile already exist for the given LogFileName
        # we can check that by checking the date. If the LogFileName is older than the machineReadableOutputFile and humanReadableOutputFile, we can skip this function
       
        log_file_mtime = os.path.getmtime(LogFileName)
        machine_readable_mtime = os.path.getmtime(machineReadableOutputFile) if os.path.exists(machineReadableOutputFile) else 0
        human_readable_mtime = os.path.getmtime(humanReadableOutputFile) if os.path.exists(humanReadableOutputFile) else 0

        #if log_file_mtime < machine_readable_mtime and log_file_mtime < human_readable_mtime:
        #    print("[CreateSimularityFilesOfLogFile] Skipping similarity calculation as the output files exist and are newer than the log file.")
         #   signature_file = signatureIterator.get_next_file()
        #    continue
        
        numberOfSignatureFiles += 1
      
        machineReadableOutputFiles.append(machineReadableOutputFile)
        humanReadableOutputFiles.append(humanReadableOutputFile)
        
        signature_name = os.path.splitext(os.path.basename(signature_file))[0]
        
        signatureNames.append(signature_name)
        signatureFiles.append(signature_file)
        
        numberOfPooledVectors, _, signature_tensor, modelHashFromSignature, _, _, _, _, _ = SignatureMatching.read_signature_from_file(signature_file)
        
        modelHashFromSignatureFileList.append(modelHashFromSignature)
        
        outputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(machineReadableOutputFile, data_type=DataTypeForTensors, numbers_per_line=1, sequence_length=None, modelHash=modelHashFromSignature)
        
        outputBinaryFileHandlers.append(outputBinaryFileHandler)
        
                
        max_SignatureLength = max(max_SignatureLength, numberOfPooledVectors)
        
        numberOfPooledVectorsList.append(numberOfPooledVectors)
        signatureTensors.append(signature_tensor)
        
        humanReadableOutputFileHandler = open(humanReadableOutputFile, 'w')
        humanReadableOutputFileHandlers.append(humanReadableOutputFileHandler)
        
        signature_file = signatureIterator.get_next_file()

    
    if (numberOfSignatureFiles == 0):
        return False    
   
    FileNamesMachineReadablePooledVectors = Constants.GetLogFileMachineReadableVectorsPath(LogFileName, userId)
    
    binaryFileHandlerFirstLogits     = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["first"],     data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerLastLogits      = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["last"],      data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)        
    binaryFileHandlerAllLogits       = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["all"] ,      data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerReceivedAttention = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["attention_received"], data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
      
    sequence_length = binaryFileHandlerFirstLogits.get_sequenceLength()
    numberOfLines = binaryFileHandlerFirstLogits.get_recordCount()
    modelHashFromPooledVectorsFile = binaryFileHandlerFirstLogits.get_modelHash()
    
    maxNumberOfLinesPerRead = 2048
    numberOfSimularityCalculationsPerRead = maxNumberOfLinesPerRead - max_SignatureLength + 1   # we moeten voor het einde van de read stoppen, omdat we anders niet genoeg data hebben om de signature te vergelijken. Stel: read is 6 groot en signaturelength is 5 -> je kan er maar 2 berekenen
    numberOfReads = math.ceil(numberOfLines / numberOfSimularityCalculationsPerRead)
    
    numberOfLinesAtLastRead = numberOfLines % numberOfSimularityCalculationsPerRead 
    
    for signatureIndex in range(numberOfSignatureFiles):
        humanReadableOutputFileHandlers[signatureIndex].write(f"The following simularities were calculated starting from line number {max_SignatureLength}\n")
        humanReadableOutputFileHandlers[signatureIndex].write(f"Model hash from signature file: {modelHashFromSignatureFileList[signatureIndex]}\n")
        humanReadableOutputFileHandlers[signatureIndex].write(f"Model hash from pooled vectors: {modelHashFromPooledVectorsFile}\n")
        humanReadableOutputFileHandlers[signatureIndex].write(f"numberOfPooledVectors:          {numberOfPooledVectorsList[signatureIndex]}. ( This can be different for every signature )\n")
        humanReadableOutputFileHandlers[signatureIndex].write(f"SignatureTensor shape:          {signatureTensors[signatureIndex].shape}\n")
        humanReadableOutputFileHandlers[signatureIndex].write(f"SignatureTensor:                {signatureTensors[signatureIndex]}\n")

        humanReadableOutputFileHandlers[signatureIndex].write("\n")
        
    lineNumber = 1
    start_time = time.time()
    simularityCalculationIndexes = np.ones(numberOfSignatureFiles, dtype=int) # because the output of the model starts at line 1
    
    try:
        for i in range(numberOfReads):
            
            if i == (numberOfReads - 1):
                numberOfLinesToRead = numberOfLinesAtLastRead
                if numberOfLinesToRead == 0:
                    break
            else:
                numberOfLinesToRead = maxNumberOfLinesPerRead
                
            numberOfSimularityCalculationsPerRead = numberOfLinesToRead - max_SignatureLength + 1
             
            for signatureIndex in range(numberOfSignatureFiles):
                signature_tensor = signatureTensors[signatureIndex]
                signatureLength = numberOfPooledVectorsList[signatureIndex]
                
                batched_receivedAttention = None # GetReceivedAttention(binaryFileHandlerReceivedAttention, lineNumber, signatureLength, sequence_length, batch_size=numberOfSimularityCalculationsPerRead, DataTypeForTensors=DataTypeForTensors)
                
                batched_hidden_state_tensor = GetHiddenVectorsOfLines(binaryFileHandlerAllLogits, binaryFileHandlerLastLogits, lineNumber, signatureLength, sequence_length, numberOfSimularityCalculationsPerRead)
                
                batched_signature_tensor_new = tf.stack([signature_tensor] * batched_hidden_state_tensor.shape[0], axis=0)
                
                simularity = CalculateSimularityBatched(batched_signature_tensor_new, batched_hidden_state_tensor, batched_receivedAttention)

                for index in range(numberOfSimularityCalculationsPerRead):
                    humanReadableOutputFileHandlers[signatureIndex].write(f"{simularityCalculationIndexes[signatureIndex]}\t{simularity[index]}\n")
                    outputBinaryFileHandlers[signatureIndex].append(simularity[index])
                    simularityCalculationIndexes[signatureIndex] += 1
                    
                    
                elapsed_time = time.time() - start_time
                
                if (signatureIndex == 0):
                    print(f"Processed {simularityCalculationIndexes[signatureIndex]/numberOfLines:.2%} percent. {numberOfSimularityCalculationsPerRead/elapsed_time:.2f} lines per second")
                
                start_time = time.time()
                
            lineNumber += numberOfSimularityCalculationsPerRead
                
        for signatureIndex in range(numberOfSignatureFiles):   
            
            outputBinaryFileHandlers[signatureIndex].close()
            humanReadableOutputFileHandlers[signatureIndex].close()
    
    except (IndexError) as e:
        print(f"IndexError: {e}")
        return False
    
    for SimularityFile in machineReadableOutputFiles:
        SimularityGraph.MakeSimularityPlot(SimularityFile, block=False, Save=True, DataTypeForTensors=DataTypeForTensors)



    
    

def CreateAnnotationListFromSimularitiesFile(machineReadableSimularityFile, outputFile, data_type, annotationId = 0, threshold = 0.5, comment = ""):
    
    inputFileHandler = BinaryFileHandler.BinaryFileHandler(machineReadableSimularityFile, data_type=data_type, sequence_length=None, numbers_per_line=1)
    modelHashFromSimularityFile = inputFileHandler.get_modelHash()
    annotationHeader = { "annotationId": annotationId, "threshold": threshold, "modelHashFromSimularityFile": modelHashFromSimularityFile, "comment": comment}
    annotationList = []
    
    annotationInformation = { "annotationHeader" :annotationHeader, "annotationList": annotationList}
      
    lineNumber = 0
    annotationStart = 0
    annotationLineCount = 0
    withinAnnotation = False
    simularitySummed = 0
    simularityMax = -1
    
    while True:
        simularity = inputFileHandler.read_next()
        
        if simularity is None:
            break
        else:
            simularity = -(simularity[0])
        
        if simularity >= threshold:
            simularitySummed += simularity
            simularityMax = max(simularityMax, simularity)
            if withinAnnotation:
                annotationLineCount += 1
            else:
                annotationStart = lineNumber
                annotationLineCount = 1
                withinAnnotation = True
        else:
            if withinAnnotation:
                annotation = { "lineNumber": annotationStart,  "lineCount": annotationLineCount, "simularity": simularitySummed / annotationLineCount, "simularityMax": simularityMax, "AnnotationId": annotationId}
                annotationInformation["annotationList"].append(annotation)
                withinAnnotation = False
                annotationLineCount = 0
                simularitySummed = 0
                simularityMax = -1
                
        lineNumber += 1
    
    if outputFile is not None:
        output_dir = os.path.dirname(outputFile)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                
        with open(outputFile, 'w') as f:
            json.dump(annotationInformation, f, indent=4)
                      
    print("CreateAnnotationListFromSimularitiesFile DONE")


