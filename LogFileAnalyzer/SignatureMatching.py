import tensorflow as tf
import BinaryFileHandler
import struct
import numpy as np






def save_signature_to_file(signatureFileName, FileNameMachineReadablePooledVectors, startVectorOffset, numberOfHiddenVectors, DataTypeForTensors):
    # Assuming CalculateSignature is a function that returns the signature
    signature = CalculateSignature(FileNameMachineReadablePooledVectors, startVectorOffset, numberOfHiddenVectors, DataTypeForTensors)
    
    with open(signatureFileName, 'wb') as file:
        # Save the number of hidden vectors
        file.write(struct.pack('i', numberOfHiddenVectors))
        # Save the data type for tensors
        file.write(DataTypeForTensors.encode('utf-8'))
        # Save the signature
        file.write(signature.numpy().tobytes())

def read_signature_from_file(filename):
    with open(filename, 'rb') as file:
        # Read the number of hidden vectors
        numberOfHiddenVectors = struct.unpack('i', file.read(4))[0]
        # Read the data type for tensors
        DataTypeForTensors = file.read(4).decode('utf-8')
        # Read the signature
        signature = file.read()
    
    signature_tensor = tf.convert_to_tensor(np.frombuffer(signature, dtype=DataTypeForTensors))
    return numberOfHiddenVectors, DataTypeForTensors, signature_tensor

def CalculateSignature(FileNameMachineReadablePooledVectors, startVectorOffset, numberOfHiddenVectors, DataTypeForTensors):
    binaryFileHandler = BinaryFileHandler.BinaryFileHandler(FileNameMachineReadablePooledVectors, data_type="float32", numbers_per_line=numberOfHiddenVectors)
    pooledVectors = binaryFileHandler.read_records(startVectorOffset, numberOfHiddenVectors)
    
    pooledAverageVector = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(pooledVectors)
    pooledMaxVector = tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last")(pooledVectors)
    pooledVectors = tf.concat([pooledAverageVector, pooledMaxVector], axis=1)
    
    return pooledAverageVector

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
