import tensorflow as tf
from TensorShift import TensorShifter
import sys
import numpy as np


class SimilarityMaxShifter:
    def __init__(self, number_of_simularities, max_similarity_length, shift, signatureLengthList, DataTypeForTensors=tf.float32):
        self.max_similarity_length = max_similarity_length
        self.shift = shift
        
        self.signatureLengthList  = signatureLengthList
        self.number_of_simularities = number_of_simularities
        self.max_list = tf.zeros([number_of_simularities, max_similarity_length], dtype=DataTypeForTensors)
        self.indices = tf.tile(
            tf.range(max_similarity_length, dtype=DataTypeForTensors)[tf.newaxis, :],
            [number_of_simularities, 1]
        )
        self.mask = tf.cast(self.indices < tf.constant(signatureLengthList, dtype=DataTypeForTensors)[:, tf.newaxis], DataTypeForTensors)
        self.shifter = TensorShifter(shape=[number_of_simularities, max_similarity_length], axis=1, shift=shift, datatype=DataTypeForTensors)
    
    def GetShiftedMaxSimilarity(self, simularities):
        
        assert simularities.shape[0] == self.number_of_simularities, f"simularities.shape[0]: {simularities.shape[0]} != self.number_of_simularities: {self.number_of_simularities}"

        for simularityIndex in range(self.number_of_simularities):
            simularity_expanded = self.mask[simularityIndex] * simularities[simularityIndex][0]
            padding = self.max_similarity_length - self.signatureLengthList[number] 
            
            
            #zeros = tf.zeros(reduced_shape_zeros, dtype=datatype)

        self.shifter.shift_tensor(simularities)
        return self.previous_similarities
    
def max_shifting(batch, simularityLengths):

    n = tf.shape(batch)[-1]

    number_of_columns = n + simularityLengths - 1

    cols_base = tf.tile(tf.range(simularityLengths, dtype=tf.int32), multiples=[n])                  # 0..L-1 repeated n times
    row_offsets = tf.repeat(tf.range(n, dtype=tf.int32), repeats=simularityLengths)                  # 0 repeated L, 1 repeated L, ...
    id_column = cols_base + row_offsets   
    
    id_row    = tf.repeat(tf.range(n), repeats=simularityLengths)


    id_array = tf.stack([id_row, id_column], axis=1) 

    updates = tf.repeat(batch, repeats=simularityLengths)

    temp__matrix = tf.scatter_nd(id_array, updates, [n, number_of_columns])

    return temp__matrix

def init_remainder_for_batch(batch, simularityLengths, dtype=None):
    """
    Initialize the carry-over remainder for the first call to max_shifting_batched.

    Args:
        batch: Tensor [B, n] similarities for the current chunk.
        simularityLengths: Tensor/List [B] lengths per similarity; determines Lmax.
        dtype: Optional tf.DType; defaults to batch.dtype.

    Returns:
        Tensor zeros of shape [B, Lmax - 1], the initial remainder to pass to max_shifting_batched.
    """
    batch = tf.convert_to_tensor(batch)
    if dtype is None:
        dtype = batch.dtype
    simularityLengths = tf.convert_to_tensor(simularityLengths, dtype=tf.int32)
    B = tf.shape(batch)[0]
    Lmax = tf.reduce_max(simularityLengths)
    tail_size = tf.maximum(Lmax - 1, 0)
    return tf.zeros([B, tail_size], dtype=dtype)

def max_shifting_batched(batch, simularityLengths, previous_remainder=None):
    # batch: [B, n], simularityLengths: [B]
    batch = tf.convert_to_tensor(batch, dtype=tf.float32)
    simularityLengths = tf.convert_to_tensor(simularityLengths, dtype=tf.int32)

    batchSize = tf.shape(batch)[0]    # het aantal simularities
    n = tf.shape(batch)[1]
    Lmax = tf.reduce_max(simularityLengths)
    C = n + Lmax - 1

    b = tf.range(batchSize, dtype=tf.int32)[:, tf.newaxis, tf.newaxis]   # [B,1,1]
    k = tf.range(n, dtype=tf.int32)[tf.newaxis, :, tf.newaxis]   # [1,n,1]
    o = tf.range(Lmax, dtype=tf.int32)[tf.newaxis, tf.newaxis, :]# [1,1,Lmax]

    b_b = tf.broadcast_to(b, [batchSize, n, Lmax])                       # [B,n,Lmax]
    k_b = tf.broadcast_to(k, [batchSize, n, Lmax])                       # [B,n,Lmax]
    o_b = tf.broadcast_to(o, [batchSize, n, Lmax])                       # [B,n,Lmax]

    cols = k_b + o_b                                             # [B,n,Lmax]
    mask = o_b < simularityLengths[:, tf.newaxis, tf.newaxis]    # [B,n,Lmax]

    indices = tf.stack([b_b, k_b, cols], axis=-1)                # [B,n,Lmax,3]
    indices = tf.reshape(indices, [-1, 3])

    updates = tf.where(
        mask,
        tf.broadcast_to(batch[:, :, tf.newaxis], [batchSize, n, Lmax]),
        tf.zeros([batchSize, n, Lmax], dtype=batch.dtype),
    )
    updates = tf.reshape(updates, [-1])

    scattered = tf.scatter_nd(indices, updates, [batchSize, n, C])       # [B,n,C]

    # Incorporate previous remainder into the max calculation for continuity across calls.
    # previous_remainder is expected shape [B, R_prev] representing the tail from the previous step.
    # We align it to the first columns and pad/clip to length C, then take max with current step's maxima.
    reduced_scattered = tf.reduce_max(scattered, axis=1)                  # [B,C]

    if previous_remainder is None:
        prev_row = tf.zeros([batchSize, C], dtype=batch.dtype)
    else:
        prev = tf.convert_to_tensor(previous_remainder, dtype=batch.dtype)
        # Validate batch dimension matches
        tf.debugging.assert_equal(
            tf.shape(prev)[0], batchSize,
            message="previous_remainder batch dimension must match current batch size",
        )
        prev_len = tf.shape(prev)[1]
        def pad_prev():
            return tf.pad(prev, paddings=[[0, 0], [0, C - prev_len]])
        def clip_prev():
            return prev[:, :C]
        prev_row = tf.cond(prev_len < C, pad_prev, clip_prev)            # [B,C]

    reduced = tf.maximum(reduced_scattered, prev_row)                     # [B,C]

    # Split off the last tail_size elements along the last dimension for carry-over
    tail_size = C - n                                                     # equals Lmax - 1
    remainder = reduced[:, -tail_size:]
    filtered = reduced[:, :-tail_size]
    
    return filtered, remainder
    
if __name__ == "__main__":
    
    # Example
    x1 = tf.constant([[4., 2., 5., 1., 7., 3., 4., 5., 6., 5], 
                      [4., 2., 5., 1., 7., 3., 4., 5., 1., 5]])
    
    x2 = tf.constant([[0., 7., 0., 0., 0., 0., 0., 0., 0., 0], 
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0]])
    
    simularityLengths = tf.constant([3, 6 ], dtype=tf.int32)

    # First step: initialize remainder for this batch
    remainder = init_remainder_for_batch(x1, simularityLengths)
    filtered1, remainder = max_shifting_batched(x1, simularityLengths, previous_remainder=remainder)
    remainder1=remainder
        
    filtered2, remainder = max_shifting_batched(x2, simularityLengths, previous_remainder=remainder)
    remainder2=remainder
    
    
    np.set_printoptions(threshold=sys.maxsize, linewidth=10**6)
    print("Filtered 1:", filtered1.numpy())
    print("Remainder 1:", remainder1.numpy())

    print("Filtered 2:", filtered2.numpy())
    print("Remainder 2:", remainder2.numpy())