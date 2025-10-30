import tensorflow as tf
from TensorShift import TensorShifter


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

def max_shifting_batched(batch, simularityLengths):
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
    
    reduced = tf.reduce_max(scattered, axis=1)                            # [B,C]

    # Split off the last 12 elements along the last dimension
    # head: [B, C-12], tail12: [B, 12]
    tail_size = C - n
    remainder = reduced[:, -tail_size:]
    filtered = reduced[:, :-tail_size]
    
    return filtered, remainder
    
if __name__ == "__main__":
    
    # Example
    x1 = tf.constant([[4., 2., 5., 1., 7., 3., 4., 5., 1., 6], [4., 2., 5., 1., 7., 3., 4., 5., 1., 6]])
    x2 = tf.constant([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    simularityLengths = tf.constant([3, 1 ], dtype=tf.int32)
    
    filtered, remainder = max_shifting_batched(x1, simularityLengths)

    tf.print("result-head:", filtered)

    
