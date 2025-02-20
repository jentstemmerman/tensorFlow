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

def read_signature_from_file(filename):
    with open(filename, 'r') as file:
        signature_data = json.load(file)    
    
    signature_text = signature_data['signature_text']
    logFileName = signature_data['logFileName']
    modelDirectory = signature_data['modelDirectory']
    # modelHash = signature_data['modelHash']
    startLine = signature_data['startLine']
    endLine = signature_data['endLine']
    DataTypeForTensors = signature_data['DataTypeForTensors']
    signature = signature_data['signature']
    
    numberOfHiddenVectors = endLine - startLine + 1
    signature_tensor = tf.convert_to_tensor(signature, dtype=DataTypeForTensors)
 
    return numberOfHiddenVectors, DataTypeForTensors, signature_tensor

def CalculatePooledVectors(vectors):
    batchedvectors = tf.expand_dims(vectors, axis=0) # Add a bactch dimension. Will be removed in the squeeze
    pooledAverageVector = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(batchedvectors)
    pooledMaxVector = tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last")(batchedvectors)
    pooledAverageVector = tf.squeeze(pooledAverageVector, axis=0) # remove the batch dimension from the tensor
    pooledMaxVector = tf.squeeze(pooledMaxVector, axis=0)
    pooledVector = tf.concat([pooledAverageVector, pooledMaxVector], axis=0)
    
    return pooledVector

def CalculatePooledVectorsBatched(batchedvectors):
    pooledAverageVector = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(batchedvectors)
    pooledMaxVector = tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last")(batchedvectors)
    pooledAverageVector = tf.squeeze(pooledAverageVector, axis=0) # remove the batch dimension from the tensor
    pooledVector = tf.concat([pooledAverageVector, pooledMaxVector], axis=-1)
    
    return pooledVector

def CalculateSignature(FileNameMachineReadablePooledVectors, startVectorOffset, numberOfHiddenVectors, DataTypeForTensors):
    binaryFileHandler = BinaryFileHandler.BinaryFileHandler(FileNameMachineReadablePooledVectors, data_type=DataTypeForTensors, numbers_per_line=None)
    
    modelHash = binaryFileHandler.get_modelHash()
    vectors = binaryFileHandler.read_records(startVectorOffset, numberOfHiddenVectors)
        
    signature = CalculatePooledVectors(vectors)
   
    return signature, modelHash



"""
Reads hidden state vectors, computes a rolling mean-pooled vector using TensorFlow's 
GlobalAveragePooling1D, calculates cosine similarity with the signature vector, 
and writes results to a file.

Args:
    signature_vector (tf.Tensor): The signature vector (1D TensorFlow tensor).
    num_vectors_to_pool (int): Number of vectors to pool before computing similarity.
    output_file (str): Path to the file where similarity scores will be saved.
"""
def search_signature(hiddenVectorFileName, output_file, signature_vector, num_vectors_to_pool, data_type):
    binary_file = BinaryFileHandler(hiddenVectorFileName, data_type=data_type, numbers_per_line=num_vectors_to_pool)
    window = []  # Rolling window to store the last `num_vectors_to_pool` vectors
    signature_vector = tf.convert_to_tensor(signature_vector, dtype=tf.float32)  # Ensure TensorFlow format

    # Create a GlobalAveragePooling1D layer
    pooling_layer = tf.keras.layers.GlobalAveragePooling1D()

    similarities = np.array([])

    with open(output_file, 'w') as f:
        while True:
            vector = binary_file.read_next()  

            if vector is None:  # Stop if there are no more vectors
                break

            vector = tf.convert_to_tensor(vector, dtype=tf.float32)  # Ensure TensorFlow tensor
            window.append(vector)

            if len(window) > num_vectors_to_pool:
                window.pop(0)  # Maintain window size

            if len(window) == num_vectors_to_pool:
                # Stack the vectors in the window and apply GlobalAveragePooling1D
                pooled_vector = pooling_layer(tf.stack(window, axis=0)[None, ...])  # Add batch dimension

                # Compute cosine similarity
                similarity = tf.keras.losses.cosine_similarity(pooled_vector[0], signature_vector)
                
                similarity_value = -similarity.numpy()  # Convert to positive similarity (cosine similarity loss returns negative values)
                similarities = np.append(similarities, similarity_value)
                f.write(f"{similarity_value}\n")  # Save result

    # Calculate statistics
    max_similarity = np.max(similarities)
    min_similarity = np.min(similarities)
    stddev_similarity = np.std(similarities)
    num_pooled_vectors = len(similarities)

    # Write header
    with open(output_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"Max Similarity: {max_similarity}\n")
        f.write(f"Min Similarity: {min_similarity}\n")
        f.write(f"Stddev Similarity: {stddev_similarity}\n")
        f.write(f"Number of Pooled Vectors: {num_pooled_vectors}\n")
        f.write(content)

def CalculateSimularity(signature, tensor, data_type):
    signature_tensor = tf.convert_to_tensor(signature, dtype=tf.float32)
    tensor = tf.convert_to_tensor(tensor, dtype=data_type)
    
    pooledVector = CalculatePooledVectors(tensor)
    
    similarity = tf.keras.losses.cosine_similarity(signature_tensor, pooledVector)
    
    return similarity

def CalculateSimularityBatched(signature, tensor, data_type):
    signature_tensor = tf.convert_to_tensor(signature, dtype=tf.float32)
    tensor = tf.convert_to_tensor(tensor, dtype=data_type)
    
    pooledVector = CalculatePooledVectorsBatched(tensor)
    
    similarity = tf.keras.losses.cosine_similarity(signature_tensor, pooledVector)
    
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
    
    numberOfHiddenVectors, DataType, signature_tensor = SignatureMatching.read_signature_from_file(signatureFile)
    num_vectors_to_pool = numberOfHiddenVectors
    
    humanReadableOutputFileHandler = open(humanReadableOutputFile, 'w')
    
    inputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(hiddenVectorsFile, data_type=dataType, numbers_per_line=num_vectors_to_pool)
    
    #modelHash = inputBinaryFileHandler.get_modelHash()
    
    outputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(machineReadableOutputFile, data_type=dataType, numbers_per_line=1) # , modelHash=modelHash)
    
    index = 0
    start_time = time.time()
    
    while True:
        
        try:
            inputTensor = inputBinaryFileHandler.read_records(index, numberOfHiddenVectors)
            inputTensor2 = inputBinaryFileHandler.read_records_optimised(index, numberOfHiddenVectors)
            
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
    

def CreateSimularitiesFilesOptimized(LogFileName, signatureDirectory, userId, dataType):
    
    if not os.path.exists(LogFileName):
        print(f"[CreateSimularitiesFile] LogFileName {LogFileName} does not exist.")
        return False
    
    # first, create list of signature files
    signatureIterator = LogFileIterator.FileIterator(signatureDirectory, Constants.GetSignatureFilesExtention())
    
    signatureLengths = []
    signatureNames = []
    signatureTensors = []
    signatureFiles = []
    machineReadableOutputFiles = []
    humanReadableOutputFiles = []
    humanReadableOutputFileHandlers = []
    outputBinaryFileHandlers = []
    signature_file = signatureIterator.get_first_tokenlogfile()
    
    if signature_file is not None:
        hiddenVectorSize, _, _ = SignatureMatching.read_signature_from_file(signature_file)
    
    
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

        if log_file_mtime < machine_readable_mtime and log_file_mtime < human_readable_mtime:
            print("[CreateSimularitiesFile] Skipping similarity calculation as the output files exist and are newer than the log file.")
            signature_file = signatureIterator.get_next_tokenlogfile()
            continue
        
        numberOfSignatureFiles += 1
        
        outputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(machineReadableOutputFile, data_type=dataType, numbers_per_line=1) # , modelHash=modelHash)
        outputBinaryFileHandlers.append(outputBinaryFileHandler)
        
        machineReadableOutputFiles.append(machineReadableOutputFile)
        humanReadableOutputFiles.append(humanReadableOutputFile)
        
        signature_name = os.path.splitext(os.path.basename(signature_file))[0]
        
        signatureNames.append(signature_name)
        signatureFiles.append(signature_file)
        
        numberOfHiddenVectors, _, signature_tensor = SignatureMatching.read_signature_from_file(signature_file)
        max_SignatureLength = max(max_SignatureLength, numberOfHiddenVectors)
        
        signatureLengths.append(numberOfHiddenVectors)
        signatureTensors.append(signature_tensor)
        
        humanReadableOutputFileHandler = open(humanReadableOutputFile, 'w')
        humanReadableOutputFileHandlers.append(humanReadableOutputFileHandler)
        
        signature_file = signatureIterator.get_next_tokenlogfile()
        
        
    if (numberOfSignatureFiles == 0):
        return False    
    
    hiddenVectorsFile = Constants.GetLogFileMachineReadableVectorsPath(LogFileName, userId, temperature)
    
    inputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(hiddenVectorsFile, data_type=dataType, numbers_per_line=hiddenVectorSize)
    numberOfLines = inputBinaryFileHandler.get_recordCount()
    
    numberOfLinesPerRead = 128
    numberOfSimularityCalculationsPerRead = numberOfLinesPerRead - max_SignatureLength + 1
    numberOfReads = numberOfLines // numberOfSimularityCalculationsPerRead 
    numberOfLinesAtLastRead = numberOfLines % numberOfSimularityCalculationsPerRead 
    
       
    recordReadStart = 0
    start_time = time.time()
    simularityCalculationIndex = np.zeros(numberOfSignatureFiles, dtype=int)
    
    try:
        for i in range(numberOfReads):
            
            if i == (numberOfReads - 1):
                numberOfLinesToRead = numberOfLinesAtLastRead
                if numberOfLinesToRead == 0:
                    break
            else:
                numberOfLinesToRead = numberOfLinesPerRead
            
            inputTensor = inputBinaryFileHandler.read_records(recordReadStart, numberOfLinesToRead)

            for signatureIndex in range(numberOfSignatureFiles):
                signature_tensor = signatureTensors[signatureIndex]
                signatureLength = signatureLengths[signatureIndex]
                
                for index in range(numberOfSimularityCalculationsPerRead):
                    testTensor = inputTensor[index : index + signatureLength, :]
                    
                    simularity = CalculateSimularity(signature_tensor, testTensor, dataType).numpy()
                    humanReadableOutputFileHandlers[signatureIndex].write(f"{simularityCalculationIndex[signatureIndex]}\t{simularity}\n")
                    outputBinaryFileHandlers[signatureIndex].append(simularity)
                    print(f"index: {simularityCalculationIndex[signatureIndex]}   simularity: {simularity}")
                    
                    simularityCalculationIndex[signatureIndex] += 1
                    
                elapsed_time = time.time() - start_time
                print(f"Processed {numberOfSimularityCalculationsPerRead}/{numberOfLines} vectors in {elapsed_time:.2f} seconds")
                start_time = time.time()
                
            recordReadStart += numberOfSimularityCalculationsPerRead
                
        for signatureIndex in range(numberOfSignatureFiles):   
            outputBinaryFileHandlers[signatureIndex].close()
            humanReadableOutputFileHandlers[signatureIndex].close()
    
    except (IndexError) as e:
        print(f"IndexError: {e}")
        return False


def CreateAnnotationListFromSimularitiesFile(machineReadableSimularityFile, outputFile, data_type, annotationId = 0, threshold = 0.5, comment = ""):
    
    inputFileHandler = BinaryFileHandler.BinaryFileHandler(machineReadableSimularityFile, data_type=data_type, numbers_per_line=1)
    
    annotationList = []
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
                annotation = { "lineNumber": annotationStart,  "lineCount": annotationLineCount, "simularity": simularity, "AnnotationId": annotationId,  "Comment": comment}
                annotationList.append(annotation)
                withinAnnotation = False
                annotationLineCount = 0

        lineNumber += 1
    
    if outputFile is not None:
        output_dir = os.path.dirname(outputFile)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                
        with open(outputFile, 'w') as f:
            json.dump(annotationList, f, indent=4)
                      
    print("CreateAnnotationListFromSimularitiesFile DONE")

    
