import sys
sys.path.append('../LogFileAnalyzer')

import tensorflow as tf
import BinaryFileHandler
import struct
import numpy as np
import Constants     # type: ignore
import SignatureMatching
import json
import AnnotationManager
import os
import time
import LogFileIterator
import math
 
 
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
 
    return numberOfPooledVectors, DataTypeForTensors, signature_tensor, modelHash

def CalculatePooledVectors(vectors):
    batchedvectors = tf.expand_dims(vectors, axis=0) # Add a batch dimension. Will be removed in the squeeze
    pooledAverageVector = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(batchedvectors)
    pooledMaxVector = tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last")(batchedvectors)
    pooledAverageVector = tf.squeeze(pooledAverageVector, axis=0) # remove the batch dimension from the tensor
    pooledMaxVector = tf.squeeze(pooledMaxVector, axis=0)
    pooledVector = tf.concat([pooledAverageVector, pooledMaxVector], axis=0)
    
    return pooledVector

@tf.function
def CalculatePooledVectorsBatched(batchedvectors):
    
    pooledAverageVector = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(batchedvectors)
    pooledMaxVector = tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last")(batchedvectors)
    pooledVector = tf.concat([pooledAverageVector, pooledMaxVector], axis=-1)
    
    return pooledVector

#  startVectorOffset 0 means the first line in the logfile
#  Assume seq_length = 64. 
def CalculateSignatureFromFiles(FileNamesMachineReadablePooledVectors, startLine, numberOfLines, DataTypeForTensors):
    
    binaryFileHandlerFirstLogits     = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["first"],     data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerLastLogits      = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["last"],      data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)        
    binaryFileHandlerPredictedLogits = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["predicted"], data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerAllLogits       = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["all"] ,      data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    
    modelHash1 = binaryFileHandlerFirstLogits.get_modelHash()
    modelHash2 = binaryFileHandlerLastLogits.get_modelHash()
    modelHash3 = binaryFileHandlerPredictedLogits.get_modelHash()
    modelHash4 = binaryFileHandlerAllLogits.get_modelHash()
    
    assert startLine >= 1, f"Impossible startline {startLine}, line 0 is never predicted!"
    
    assert modelHash1 == modelHash2, f"Model hashes from first and last logits file do not match. {modelHash1} != {modelHash2}"
    assert modelHash1 == modelHash3, f"Model hashes from first and all logits file do not match. {modelHash1} != {modelHash3}"
    assert modelHash1 == modelHash4, f"Model hashes from first and predicted logits file do not match. {modelHash1} != {modelHash4}"
    
    sequence_length = binaryFileHandlerFirstLogits.get_sequenceLength()
    
    startVectorOffsetInAllLogitsFile       = (startLine - 1) * sequence_length   #this is the transformation to get the hidden vector of startLine as it is generated in the first timeslice of the sequence
    #startVectorOffsetInPredictedLogitsFile = startLine - sequence_length        # ok
    startVectorOffsetInLastLogitsFile      = startLine - sequence_length + 1     # ok

    if numberOfLines <= sequence_length:
           # this is not a typo. It should be - and not %
        hiddenVectorsPart_1 = binaryFileHandlerAllLogits.read_records(startVectorOffsetInAllLogitsFile, numberOfLines)
        numberOfLinesRemainder = 0
    else:
        hiddenVectorsPart_1 = binaryFileHandlerAllLogits.read_records(startVectorOffsetInAllLogitsFile, sequence_length)
        numberOfLinesRemainder = numberOfLines - sequence_length
        
    if (numberOfLinesRemainder > 0):
        hiddenVectorsPart_2 = binaryFileHandlerLastLogits.read_records(startVectorOffsetInLastLogitsFile + 1, numberOfLinesRemainder)
        hiddenVectors = tf.concat([hiddenVectorsPart_1, hiddenVectorsPart_2], axis=0)
    else:
        hiddenVectors = hiddenVectorsPart_1
        
    signature = CalculatePooledVectors(hiddenVectors)
   
    return signature, modelHash1, hiddenVectors








def CalculateSimularity(signature, tensor, data_type):
    signature_tensor = tf.convert_to_tensor(signature, dtype=tf.float32)
    tensor = tf.convert_to_tensor(tensor, dtype=data_type)
    
    pooledVector = CalculatePooledVectors(tensor)
    
    similarity = tf.keras.losses.cosine_similarity(signature_tensor, pooledVector)
    
    return similarity

@tf.function
def CalculateSimularityBatched(signature, tensor):
   
    pooledVector = CalculatePooledVectorsBatched(tensor)
    
    similarity = tf.keras.losses.cosine_similarity(signature, pooledVector)
    
    return similarity

def CreateSimularitiesFile(LogFileName, signatureFile, machineReadableOutputFile, humanReadableOutputFile, userId, dataType):
    
    if not os.path.exists(LogFileName):
        print(f"[CreateSimularitiesFile] LogFileName {LogFileName} does not exist.")
        return False
    
    # it is possible that this function already ran, and machineReadableOutputFile and humanReadableOutputFile already exist for the given LogFileName
    # we can check that by checking the date. If the LogFileName is older than the machineReadableOutputFile and humanReadableOutputFile, we can skip this function
    log_file_mtime = os.path.getmtime(LogFileName)
    machine_readable_mtime = os.path.getmtime(machineReadableOutputFile) if os.path.exists(machineReadableOutputFile) else 0
    human_readable_mtime = os.path.getmtime(humanReadableOutputFile) if os.path.exists(humanReadableOutputFile) else 0

    if log_file_mtime < machine_readable_mtime and log_file_mtime < human_readable_mtime:
        print("[CreateSimularitiesFile] Skipping similarity calculation as the output files exist and are newer than the log file.")
        return False
    
    if os.path.exists(machineReadableOutputFile):
        os.remove(machineReadableOutputFile)
    if os.path.exists(humanReadableOutputFile):
        os.remove(humanReadableOutputFile)
    
    temperature = 1
    hiddenVectorsFile = Constants.GetLogFileMachineReadableVectorsPath(LogFileName, userId, temperature)
    
    numberOfPooledVectors, DataType, signature_tensor, modelHashFromSignature = SignatureMatching.read_signature_from_file(signatureFile)
    num_vectors_to_pool = numberOfPooledVectors
    
    humanReadableOutputFileHandler = open(humanReadableOutputFile, 'w')
    humanReadableOutputFileHandler.write(f"The following simularities were calculated starting from line number {num_vectors_to_pool}\n")
    humanReadableOutputFileHandler.write(f"Model hash used to create signature: {modelHashFromSignature}\n")
    
    #binaryFileHandlerFirstLogits     = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["first"],     data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    #binaryFileHandlerLastLogits      = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["last"],      data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)        
    #binaryFileHandlerPredictedLogits = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["predicted"], data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    #binaryFileHandlerAllLogits       = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["all"] ,      data_type=DataTypeForTensors, sequence_length=None, numbers_per_line=None)  
    
    
    binaryFileHandlerPredictedLogits = BinaryFileHandler.BinaryFileHandler(machineReadableOutputFile["predicted"], data_type=dataType, sequence_length=None, numbers_per_line=None)  
    
    modelHashFromHiddenVectorsFile = binaryFileHandlerPredictedLogits.get_modelHash()    
    humanReadableOutputFileHandler.write(f"Model hash from pooled vector file: {modelHashFromHiddenVectorsFile}\n")
    
    if (modelHashFromSignature != modelHashFromHiddenVectorsFile):
        print(f"WARNING! Model hashes from signature file {modelHashFromSignature} and hidden vectors file {modelHashFromHiddenVectorsFile} do not match.")
        
        
    outputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(machineReadableOutputFile, data_type=dataType, numbers_per_line=1, sequence_length=None, modelHash=modelHashFromSignature)
    
    index = 0
    start_time = time.time()
    
    while True:
        
        try:
            inputTensor = binaryFileHandlerPredictedLogits.read_records(index, numberOfPooledVectors)
            inputTensor2 = binaryFileHandlerPredictedLogits.read_records_optimised(index, numberOfPooledVectors)   # JENTS
            
            if not tf.reduce_all(tf.equal(inputTensor, inputTensor2)):
                print(f"Tensors are not equal at index {index}")
            
            assert tf.reduce_all(tf.equal(inputTensor, inputTensor2)), f"Tensors are not equal at index {index}"
                
        except (IndexError) as e:
            print(f"IndexError: {e}")
            break
            
        if inputTensor is None:
            break
        
        simularity = CalculateSimularity(signature_tensor, inputTensor, dataType).numpy()
        
        humanReadableOutputFileHandler.write(f"{index}\t{simularity}\n")
        outputBinaryFileHandler.append(simularity)
        index += 1
        
        if index % 1000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed 1000 vectors in {elapsed_time:.2f} seconds")
            start_time = time.time()
        
        
    outputBinaryFileHandler.close()
    humanReadableOutputFileHandler.close()
    
# limitation: the output of this file starts at linenumber 0 + max signature length - 1, so a very large signature length will cause a lot of empty lines at the start of the file
#tf.function
def CreateSimularitiesFilesOptimized(LogFileName, signatureDirectory, userId, dataType):
    
    if not os.path.exists(LogFileName):
        print(f"[CreateSimularitiesFile] LogFileName {LogFileName} does not exist.")
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
    
    signature_file = signatureIterator.get_first_tokenlogfile()
    
    if signature_file is not None:
        numberOfPooledVectors, _, _, _ = SignatureMatching.read_signature_from_file(signature_file)
    else:
        print("[CreateSimularitiesFile] No signature files found.")
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
        #    print("[CreateSimularitiesFile] Skipping similarity calculation as the output files exist and are newer than the log file.")
         #   signature_file = signatureIterator.get_next_tokenlogfile()
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
        
        signature_file = signatureIterator.get_next_tokenlogfile()
        
        
    if (numberOfSignatureFiles == 0):
        return False    
   
    FileNamesMachineReadablePooledVectors = Constants.GetLogFileMachineReadableVectorsPath(LogFileName, userId, temperature)
    
    binaryFileHandlerFirstLogits     = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["first"],     data_type=dataType, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerLastLogits      = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["last"],      data_type=dataType, sequence_length=None, numbers_per_line=None)        
    binaryFileHandlerPredictedLogits = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["predicted"], data_type=dataType, sequence_length=None, numbers_per_line=None)  
    binaryFileHandlerAllLogits       = BinaryFileHandler.BinaryFileHandler(FileNamesMachineReadablePooledVectors["all"] ,      data_type=dataType, sequence_length=None, numbers_per_line=None)  
      
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
    simularityCalculationIndexes = np.zeros(numberOfSignatureFiles, dtype=int)
    
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
                
                batched_signature_tensor = GetHiddenVectorsOfLines(binaryFileHandlerAllLogits, binaryFileHandlerLastLogits, lineNumber, signatureLength, sequence_length, numberOfSimularityCalculationsPerRead)
                
                batched_signature_tensor_new = tf.stack([signature_tensor] * batched_signature_tensor.shape[0], axis=0)
                
                simularity = CalculateSimularityBatched(batched_signature_tensor_new, batched_signature_tensor)

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

# This function returns a batch of hidden vectors, starting from startLine, each with a length of numberOfLines
# @tf.function
def GetHiddenVectorsOfLines(binaryFileHandlerAllLogits, binaryFileHandlerLastLogits, startLine, numberOfLines, sequence_length, batch_size):
    startVectorOffsetInAllLogitsFile       = (startLine - 1) * sequence_length   #this is the transformation to get the hidden vector of startLine as it is generated in the first timeslice of the sequence
    #startVectorOffsetInPredictedLogitsFile = startLine - sequence_length         # ok
         # ok
    hiddenVectors = binaryFileHandlerAllLogits.read_records(startVectorOffsetInAllLogitsFile, sequence_length * batch_size)
    hiddenVectors = tf.reshape(hiddenVectors, (batch_size, sequence_length, -1))
    hiddenVectors = hiddenVectors[:, 0 : numberOfLines, :]    
    
    if (numberOfLines >= sequence_length):
        hiddenVectorsPart_1 = hiddenVectors
        newStartLine = startLine + sequence_length - 1
        startVectorOffsetInLastLogitsFile = newStartLine - sequence_length + 1
        numberOfLinesRemainder = numberOfLines - sequence_length
        
        hiddenVectorsPart_2 = binaryFileHandlerLastLogits.read_records(startVectorOffsetInLastLogitsFile, numberOfLinesRemainder * batch_size)
        hiddenVectorsPart_2 = tf.signal.frame(hiddenVectorsPart_2, numberOfLinesRemainder, 1, axis=0)
        
        hiddenVectorsPart_2 = tf.reshape(hiddenVectorsPart_2, (batch_size, sequence_length, -1))
        hiddenVectorsPart_2 = hiddenVectorsPart_1[:, 0 : numberOfLinesRemainder, :]
        
        hiddenVectors = tf.concat([hiddenVectorsPart_1, hiddenVectorsPart_2], axis=0)
        assert False, "Not validated yet. Check this code"

  
    return hiddenVectors


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
    
    while True:
        simularity = inputFileHandler.read_next()
        
        if simularity is None:
            break
        else:
            simularity = -(simularity[0])
        
        if simularity >= threshold:
            if withinAnnotation:
                annotationLineCount += 1
            else:
                annotationStart = lineNumber
                annotationLineCount = 1
                withinAnnotation = True
        else:
            if withinAnnotation:
                annotation = { "lineNumber": annotationStart,  "lineCount": annotationLineCount, "simularity": simularity, "AnnotationId": annotationId}
                annotationInformation["annotationList"].append(annotation)
                withinAnnotation = False
                annotationLineCount = 0

        lineNumber += 1
    
    if outputFile is not None:
        output_dir = os.path.dirname(outputFile)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                
        with open(outputFile, 'w') as f:
            json.dump(annotationInformation, f, indent=4)
                      
    print("CreateAnnotationListFromSimularitiesFile DONE")


