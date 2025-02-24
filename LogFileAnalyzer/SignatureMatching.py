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
    pooledVector = tf.concat([pooledAverageVector, pooledMaxVector], axis=-1)
    
    return pooledVector

def CalculateSignature(FileNameMachineReadablePooledVectors, startVectorOffset, numberOfHiddenVectors, DataTypeForTensors):
    binaryFileHandler = BinaryFileHandler.BinaryFileHandler(FileNameMachineReadablePooledVectors, data_type=DataTypeForTensors, numbers_per_line=None)
    
    modelHash = binaryFileHandler.get_modelHash()
    print(f"[CalculateSignature] ModelHash from pooled vector file: {modelHash}")
    
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
    
    inputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(hiddenVectorsFile, data_type=dataType, numbers_per_line=num_vectors_to_pool)
    modelHashFromHiddenVectorsFile = inputBinaryFileHandler.get_modelHash()    
    humanReadableOutputFileHandler.write(f"Model hash from pooled vector file: {modelHashFromHiddenVectorsFile}\n")
    
    if (modelHashFromSignature != modelHashFromHiddenVectorsFile):
        print(f"WARNING! Model hashes from signature file {modelHashFromSignature} and hidden vectors file {modelHashFromHiddenVectorsFile} do not match.")
        
        
    outputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(machineReadableOutputFile, data_type=dataType, numbers_per_line=1, modelHash=modelHashFromSignature)
    
    index = 0
    start_time = time.time()
    
    while True:
        
        try:
            inputTensor = inputBinaryFileHandler.read_records(index, numberOfPooledVectors)
            inputTensor2 = inputBinaryFileHandler.read_records_optimised(index, numberOfPooledVectors)
            
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
    signature_file = signatureIterator.get_first_tokenlogfile()
    
    if signature_file is not None:
        numberOfPooledVectors, _, _, modelHashOfFirstSignatureFile = SignatureMatching.read_signature_from_file(signature_file)
    
    
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
        
        numberOfPooledVectors, _, signature_tensor, modelHash = SignatureMatching.read_signature_from_file(signature_file)
        max_SignatureLength = max(max_SignatureLength, numberOfPooledVectors)
        
        numberOfPooledVectorsList.append(numberOfPooledVectors)
        signatureTensors.append(signature_tensor)
        
        humanReadableOutputFileHandler = open(humanReadableOutputFile, 'w')
        humanReadableOutputFileHandlers.append(humanReadableOutputFileHandler)
        
        signature_file = signatureIterator.get_next_tokenlogfile()
        
        
    if (numberOfSignatureFiles == 0):
        return False    
   
    hiddenVectorsFile = Constants.GetLogFileMachineReadableVectorsPath(LogFileName, userId, temperature)
    
    inputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(hiddenVectorsFile, data_type=dataType, numbers_per_line=numberOfPooledVectors)
    numberOfLines = inputBinaryFileHandler.get_recordCount()
    
    maxNumberOfLinesPerRead = 2048
    numberOfSimularityCalculationsPerRead = maxNumberOfLinesPerRead - max_SignatureLength + 1   # we moeten voor het einde van de read stoppen, omdat we anders niet genoeg data hebben om de signature te vergelijken. Stel: read is 6 groot en signaturelength is 5 -> je kan er maar 2 berekenen
    numberOfReads = math.ceil(numberOfLines / numberOfSimularityCalculationsPerRead)
    
    numberOfLinesAtLastRead = numberOfLines % numberOfSimularityCalculationsPerRead 
    
    for signatureIndex in range(numberOfSignatureFiles):
        humanReadableOutputFileHandlers[signatureIndex].write(f"The following simularities were calculated starting from line number {max_SignatureLength}\n")
        humanReadableOutputFileHandlers[signatureIndex].write(f"  numberOfPooledVectors {numberOfPooledVectorsList[signatureIndex]}\n")
        humanReadableOutputFileHandlers[signatureIndex].write(f"  SignatureTensor {signatureTensors[signatureIndex]}\n")
        humanReadableOutputFileHandlers[signatureIndex].write(f"  SignatureTensor shape {signatureTensors[signatureIndex].shape}\n")
        humanReadableOutputFileHandlers[signatureIndex].write("\n")
        
    recordReadStart = 0
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
            
            inputTensor = inputBinaryFileHandler.read_records(recordReadStart, numberOfLinesToRead)

            for signatureIndex in range(numberOfSignatureFiles):
                signature_tensor = signatureTensors[signatureIndex]
                signatureLength = numberOfPooledVectorsList[signatureIndex]
                
                TestTensorList = []
                
                for index in range(numberOfSimularityCalculationsPerRead):
                    testTensor = inputTensor[index : index + signatureLength, :]
                    TestTensorList.append(testTensor)
                
                batchedTestTensor = tf.stack(TestTensorList, axis=0)
                batched_signature_tensor = tf.stack([signature_tensor] * batchedTestTensor.shape[0], axis=0)

                simularity = CalculateSimularityBatched(batched_signature_tensor, batchedTestTensor)
                
                for index in range(numberOfSimularityCalculationsPerRead):
                    humanReadableOutputFileHandlers[signatureIndex].write(f"{simularityCalculationIndexes[signatureIndex]}\t{simularity[index]}\n")
                    outputBinaryFileHandlers[signatureIndex].append(simularity[index])
                    simularityCalculationIndexes[signatureIndex] += 1
                    
                    
                elapsed_time = time.time() - start_time
                
                if (signatureIndex == 0):
                    print(f"Processed {simularityCalculationIndexes[signatureIndex]/numberOfLines:.2%} percent. {numberOfSimularityCalculationsPerRead/elapsed_time:.2f} lines per second")
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

    
