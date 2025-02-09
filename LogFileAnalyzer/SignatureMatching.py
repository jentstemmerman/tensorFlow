import sys
sys.path.append('../LogFileAnalyzer')

import tensorflow as tf
import BinaryFileHandler
import struct
import numpy as np
import Constants     # type: ignore
import SignatureMatching
import json
import os






def read_signature_from_file(filename):
    with open(filename, 'r') as file:
        signature_data = json.load(file)    
    
    signature_text = signature_data['signature_text']
    logFileName = signature_data['logFileName']
    modelDirectory = signature_data['modelDirectory']
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
    # pooledMaxVector = tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last")(batchedvectors)
    pooledAverageVector = tf.squeeze(pooledAverageVector, axis=0) # remove the batch dimension from the tensor
    # pooledMaxVector = tf.squeeze(pooledMaxVector, axis=0)
    # pooledVector = tf.concat([pooledAverageVector, pooledMaxVector], axis=0)
    pooledVector = pooledAverageVector
    
    return pooledVector

def CalculateSignature(FileNameMachineReadablePooledVectors, startVectorOffset, numberOfHiddenVectors, DataTypeForTensors):
    binaryFileHandler = BinaryFileHandler.BinaryFileHandler(FileNameMachineReadablePooledVectors, data_type="float32", numbers_per_line=None)
    
    vectors = binaryFileHandler.read_records(startVectorOffset, numberOfHiddenVectors)
        
    return CalculatePooledVectors(vectors)

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

def CreateSimularitiesFile(LogFileName, signatureFile, machineReadableOutputFile, humanReadableOutputFile, userId, similaritythreshold, dataType):
    temperature = 1
    hiddenVectorsFile = Constants.GetLogFileMachineReadableVectorsPath(LogFileName, userId, temperature)
    
    numberOfHiddenVectors, DataType, signature_tensor = SignatureMatching.read_signature_from_file(signatureFile)
    num_vectors_to_pool = numberOfHiddenVectors
    
    humanReadableOutputFileHandler = open(humanReadableOutputFile, 'w')
    
    outputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(machineReadableOutputFile, data_type=dataType, numbers_per_line=1)
    inputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(hiddenVectorsFile, data_type=dataType, numbers_per_line=num_vectors_to_pool)

    index = 0
    similarityTresholdCrossedCounter = 0
    
    while True:
        inputTensor = inputBinaryFileHandler.read_records(index, numberOfHiddenVectors)

        if inputTensor is None:
            break
        
        simularity = CalculateSimularity(signature_tensor, inputTensor, dataType).numpy()
        
        if ((similaritythreshold is not None) and (simularity > similaritythreshold)):
            similarityTresholdCrossedCounter += 1
        
        humanReadableOutputFileHandler.write(f"{index}\t{simularity}\n")
        outputBinaryFileHandler.append(simularity)
        index += 1

    humanReadableOutputFileHandler.close
    return similarityTresholdCrossedCounter

    