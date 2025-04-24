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
 
 
def RecalculateSignature(signatureFilePath, userId, newModelDirectory):
    
    signature_name = Constants.GetSignatureName(signatureFilePath)
    
    numberOfPooledVectors, DataTypeForTensors, old_signature_tensor, modelHash, signature_text, logFileName, modelDirectory, startLine, endLine = read_signature_from_file(signatureFilePath)
     
    SaveSignatureFile(signature_name, logFileName, signature_text, startLine, endLine, userId, newModelDirectory)
     
 
def read_signature_from_file(filename):
    with open(filename, 'r') as file:
        signature_data = json.load(file)    
        
        signature_text = signature_data['signature_text']
        logFileName = signature_data['logFileName']
        modelDirectory = signature_data['modelDirectory']
        modelHash = signature_data['modelHash']
        startLine = signature_data['startLine']
        endLine = signature_data['endLine']
        DataTypeForTensors = signature_data['DataTypeForTensors']
        signature = signature_data['signature']
        
        numberOfPooledVectors = endLine - startLine + 1
        signature_tensor = tf.convert_to_tensor(signature, dtype=DataTypeForTensors)
 
        return numberOfPooledVectors, DataTypeForTensors, signature_tensor, modelHash, signature_text, logFileName, modelDirectory, startLine, endLine

def SaveSignatureFile(signature_name, logFileName, signature_text, startLine, endLine, userId, modelDirectory):
    try:
        FileNamesMachineReadablePooledVectors = Constants.GetLogFileMachineReadableVectorsPath(logFileName, userId)
        DataTypeForTensors = DefaultTrainingSettings.TrainingsParameter["DataTypeForTensors"]
        modelDirectory = os.path.join(Constants.GetModelsDirectoryPath(), modelDirectory)

        # the modelHash is taken from the header of the machineReadablePooledVectors file
        signature, modelHash, hiddenVectors = SignatureMatching.CalculateSignature(FileNamesMachineReadablePooledVectors, startLine, endLine - startLine + 1, DataTypeForTensors)

        # Pack all these values into a json and save that to the file
        signature_data = {
            "modelDirectory": modelDirectory,
            "signature_name": signature_name,
            "logFileName": logFileName,
            "signature_text": signature_text,
            "startLine": startLine,
            "endLine": endLine,
            "DataTypeForTensors": DataTypeForTensors,
            "modelHash": modelHash,
            "signature": signature.numpy().tolist(),
            "hiddenVectors": [[float(f"{x:8.3f}") for x in row] for row in hiddenVectors.numpy()],
        }

        file_path = Constants.GetSignatureFilePath(signature_name, userId)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(signature_data, file, indent=4, sort_keys=True)

        print("Signature saved. modelHash: ", modelHash) 
        return signature_name
        

    except Exception as e:
        print("Error saving signature:", e)
        return None
        
        


# This function is used in the simularity file calculation
# @tf.function
def CalculatePooledVectorsBatched(batchedvectors, receivedAttention):
    if receivedAttention is not None:
        reshaped_attention = tf.expand_dims(receivedAttention, axis=-1)
        
        batchedvectors = batchedvectors * reshaped_attention
        
    batchedvectors = tf.nn.l2_normalize(batchedvectors, axis=-1)
    pooledAverageVector = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(batchedvectors)
    pooledMaxVector = tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last")(batchedvectors)
    pooledVector = tf.concat([pooledAverageVector, pooledMaxVector], axis=-1)
    
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
    assert modelHash1 == modelHash5, f"Model hashes from first and Received Attention file do not match. {modelHash1} != {modelHash5s}"
    
    sequence_length = binaryFileHandlerFirstLogits.get_sequenceLength()
    
    receivedAttention = GetReceivedAttention(binaryFileHandlerReceivedAttention, startLine, numberOfLines, sequence_length, batch_size=1)
    
    hiddenVectors = GetHiddenVectorsOfLines(binaryFileHandlerAllLogits, binaryFileHandlerLastLogits, startLine, numberOfLines, sequence_length, batch_size=1)
    
    signature = CalculatePooledVectorsBatched(hiddenVectors, receivedAttention)
    
    signature = tf.squeeze(signature, axis=0)           # remove the batch dimension from the tensor
    hiddenVectors = tf.squeeze(hiddenVectors, axis=0)   # remove the batch dimension from the tensor
    
    return signature, modelHash1, hiddenVectors

# This function returns a batch of attention received 
def GetReceivedAttention(binaryFileHandlerReceivedAttention, startLine, numberOfLines, sequence_length, batch_size):

    startVectorOffsetInReceivedAttentionFile = startLine - 1   
    numberOfRecordsToRead = numberOfLines + batch_size - 1
    
    attentionRecords = binaryFileHandlerReceivedAttention.read_records(startVectorOffsetInReceivedAttentionFile, numberOfRecordsToRead)

    receivedAttention = tf.signal.frame(attentionRecords, numberOfLines, 1, axis=0)
    
    contrib_counts = tf.range(numberOfLines, 0, -1, dtype=tf.float32)    
        
    # Create a mask using the contrib_counts
    # First create a range for sequence positions
    seq_positions = tf.range(sequence_length, dtype=tf.float32)

    # Reshape for proper broadcasting
    contrib_counts_reshaped = tf.reshape(contrib_counts, [1, numberOfLines, 1])  # [1, numberOfLines, 1]
    seq_positions = tf.reshape(seq_positions, [1, 1, sequence_length])           # [1, 1, sequence_length]

    # Create mask: only pay attention to positions where seq_position < line's contrib count
    attention_mask = tf.cast(seq_positions < contrib_counts_reshaped, tf.float32)

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
    pooledVector = CalculatePooledVectorsBatched(tensor, receivedAttention)
    signature = tf.linalg.l2_normalize(signature, axis=-1)
    pooledVector = tf.linalg.l2_normalize(pooledVector, axis=-1)
    
    similarity = tf.keras.losses.cosine_similarity(signature, pooledVector)
    
    return similarity

# limitation: the output of this file starts at linenumber 0 + max signature length - 1, so a very large signature length will cause a lot of empty lines at the start of the file
#tf.function
def CreateSimularityFilesOfLogFile(LogFileName, signatureDirectory, userId, dataType):
    
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
        
        numberOfPooledVectors, _, signature_tensor, modelHashFromSignature = SignatureMatching.read_signature_from_file(signature_file)
        modelHashFromSignatureFileList.append(modelHashFromSignature)
        
        outputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(machineReadableOutputFile, data_type=dataType, numbers_per_line=1, sequence_length=None, modelHash=modelHashFromSignature)
        
        outputBinaryFileHandlers.append(outputBinaryFileHandler)
        
                
        max_SignatureLength = max(max_SignatureLength, numberOfPooledVectors)
        
        numberOfPooledVectorsList.append(numberOfPooledVectors)
        signatureTensors.append(signature_tensor)
        
        humanReadableOutputFileHandler = open(humanReadableOutputFile, 'w')
        humanReadableOutputFileHandlers.append(humanReadableOutputFileHandler)
        
        signature_file = signatureIterator.get_next_file()

    
    if (numberOfSignatureFiles == 0):
        return False    
   
    FileNamesMachineReadablePooledVectors = Constants.GetLogFileMachineReadableVectorsPath(LogFileName, userId, temperature)
    
    binaryFileHandlerFirstLogits     = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["first"],     data_type=dataType, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerLastLogits      = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["last"],      data_type=dataType, sequence_length=None, numbers_per_line=None)        
    binaryFileHandlerAllLogits       = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["all"] ,      data_type=dataType, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerReceivedAttention = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["attention_received"], data_type=dataType, sequence_length=None, numbers_per_line=None)  
      
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
                
                batched_receivedAttention = GetReceivedAttention(binaryFileHandlerReceivedAttention, lineNumber, signatureLength, sequence_length, batch_size=numberOfSimularityCalculationsPerRead)
                
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
        SimularityGraph.MakeSimularityPlot(SimularityFile, block=False, Save=True)

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


