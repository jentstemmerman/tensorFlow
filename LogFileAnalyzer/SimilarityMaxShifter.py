import tensorflow as tf
from TensorShift import TensorShifter
import sys
import numpy as np


class SimilarityMaxShifter:
    def __init__(self, number_of_simularities, max_similarity_length, shift, signatureLengthList, DataTypeForTensors=tf.float32, mode="min"):
        self.max_similarity_length = max_similarity_length
        self.shift = shift
        # mode: "max" to keep highest values, "min" to keep lowest values (e.g., cosine similarity where -1 is best)
        assert mode in ("max", "min"), "mode must be 'max' or 'min'"
        self.mode = mode
        
        # Store signature lengths as int32 tensor
        self.signatureLengthList  = tf.convert_to_tensor(signatureLengthList, dtype=tf.int32)
        self.number_of_simularities = number_of_simularities
        self.max_list = tf.zeros([number_of_simularities, max_similarity_length], dtype=DataTypeForTensors)
        self.indices = tf.tile(
            tf.range(max_similarity_length, dtype=tf.int32)[tf.newaxis, :],
            [number_of_simularities, 1]
        )
        self.mask = tf.cast(self.indices < self.signatureLengthList[:, tf.newaxis], DataTypeForTensors)
        self.shifter = TensorShifter(shape=[number_of_simularities, max_similarity_length], axis=1, shift=shift, datatype=DataTypeForTensors)
        self.previous_remainder = self.init_remainder_for_batch(self.signatureLengthList, number_of_simularities, max_similarity_length, dtype=DataTypeForTensors)
        
    def init_remainder_for_batch(self, simularityLengths, number_of_simularities, max_similarity_length, dtype=None):
        """
        Initialize the carry-over remainder for the first call to max_shifting_batched.

        Args:
            batch: Tensor [B, n] similarities for the current chunk.
            simularityLengths: Tensor/List [B] lengths per similarity; determines Lmax.
            dtype: Optional tf.DType; defaults to batch.dtype.

        Returns:
            Tensor zeros of shape [B, Lmax - 1], the initial remainder to pass to max_shifting_batched.
        """
        simularityLengths = tf.convert_to_tensor(simularityLengths, dtype=tf.int32)
        B = number_of_simularities
        Lmax = tf.reduce_max(simularityLengths)
        tail_size = tf.maximum(Lmax - 1, 0)  # int32
        # Choose sentinel in ORIGINAL domain so that after any transform it becomes the neutral element
        if self.mode == "min":
            # We operate on -values internally; store +inf so that -prev -> -inf inside computation
            sentinel = tf.constant(float('inf'), dtype=dtype or tf.float32)
        else:
            # Max mode: neutral is -inf
            sentinel = tf.constant(-float('inf'), dtype=dtype or tf.float32)
        return tf.fill([B, tail_size], sentinel)
       
            
    # I had copilot make a batched version of max_shifting, which I progremmed myself before. 
    # I did verify that it gives the same results as my original max_shifting in a few test cases.
    # TODO: verify correctness and efficiency, and understand it properly
    def max_shifting_batched(self, batch):

        simularityLengths = self.signatureLengthList
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
        o_b = tf.broadcast_to(o, [batchSize, n, Lmax])                       # [B,n,Lmax]

        cols = k + o                                                         # [1,n,Lmax] via broadcasting
        cols = tf.broadcast_to(cols, [batchSize, n, Lmax])                   # [B,n,Lmax]
        mask = o_b < simularityLengths[:, tf.newaxis, tf.newaxis]            # [B,n,Lmax]

        # Build 2D indices [B, C] and scatter maxima directly to avoid [B,n,C] intermediates
        indices_2d = tf.stack([b_b, cols], axis=-1)                          # [B,n,Lmax,2]
        indices_2d = tf.reshape(indices_2d, [-1, 2])                         # [B*n*Lmax,2]

        neg_inf = tf.constant(-float('inf'), dtype=batch.dtype)
        # If mode == 'min', operate in negated domain so we can still use max
        if self.mode == "min":
            batch_eff = -batch
        else:
            batch_eff = batch

        updates = tf.where(
            mask,
            tf.broadcast_to(batch_eff[:, :, tf.newaxis], [batchSize, n, Lmax]),
            tf.fill([batchSize, n, Lmax], neg_inf),
        )
        updates = tf.reshape(updates, [-1])                                   # [B*n*Lmax]

        base = tf.fill([batchSize, C], neg_inf)
        reduced_scattered = tf.tensor_scatter_nd_max(base, indices_2d, updates)  # [B,C]

        if self.previous_remainder is None:
            prev_row = tf.fill([batchSize, C], neg_inf)
        else:
            prev = tf.convert_to_tensor(self.previous_remainder, dtype=batch.dtype)
            # Convert stored ORIGINAL-domain remainder to effective domain when in min mode
            if self.mode == "min":
                prev = -prev
            # Validate batch dimension matches
            tf.debugging.assert_equal(
                tf.shape(prev)[0], batchSize,
                message="previous_remainder batch dimension must match current batch size",
            )
            prev_len = tf.shape(prev)[1]
            def pad_prev():
                return tf.pad(prev, paddings=[[0, 0], [0, C - prev_len]], constant_values=neg_inf)
            def clip_prev():
                return prev[:, :C]
            prev_row = tf.cond(prev_len < C, pad_prev, clip_prev)            # [B,C]

        reduced = tf.maximum(reduced_scattered, prev_row)                     # [B,C]

        # Split off the last tail_size elements along the last dimension for carry-over
        tail_size = C - n                                                     # equals Lmax - 1
        prev_eff = reduced[:, -tail_size:]
        # Store remainder in ORIGINAL domain for visibility; convert from effective domain if needed
        if self.mode == "min":
            self.previous_remainder = -prev_eff
        else:
            self.previous_remainder = prev_eff
        filtered = reduced[:, :-tail_size]

        # If we operated in negated domain (min mode), invert back for the output only
        if self.mode == "min":
            filtered = -filtered
        
        return filtered
        
        
    def GetShiftedMaxSimilarity(self, simularities):
        # Compute max-shifted similarities and update internal remainder.
        simularities = tf.convert_to_tensor(simularities, dtype=tf.float32)
        tf.debugging.assert_equal(
            tf.shape(simularities)[0], self.number_of_simularities,
            message="simularities batch size must equal number_of_simularities",
        )
        return self.max_shifting_batched(simularities)
    
    def max_shifting(self, batch):
        # Wrapper that updates internal remainder and returns filtered
        return self.max_shifting_batched(batch)

    
if __name__ == "__main__":
    
    # Example
    x1 = tf.constant([[-4., -2., -5., -1., -7., -3., -4., -5., -6., -5.], 
                      [-4., -2., -5., -1., -7., -3., -4., -5., -1., -5.]])
    
    x2 = tf.constant([[-0., -7., -0., -0., -0., -0., -0., -0., -0., -0.], 
                      [-0., -0., -0., -0., -0., -0., -0., -0., -0., -0.]])
    
    simularityLengths = tf.constant([3, 6], dtype=tf.int32)
    max_similarity_length = int(tf.reduce_max(simularityLengths).numpy())

    shifter = SimilarityMaxShifter(
        number_of_simularities=2,
        max_similarity_length=max_similarity_length,
        shift=1,
        signatureLengthList=simularityLengths,
        DataTypeForTensors=tf.float32,
        mode="min",  # use "max" to mirror the standalone function behavior
    )

    filtered1 = shifter.GetShiftedMaxSimilarity(x1)
    remainder1 = shifter.previous_remainder

    filtered2 = shifter.GetShiftedMaxSimilarity(x2)
    remainder2 = shifter.previous_remainder

    np.set_printoptions(
        threshold=sys.maxsize,
        linewidth=200,
        formatter={'float_kind': lambda x: f"{x: 3.0f}"}
    )
    print("Filtered 1: \n", filtered1.numpy())
    print("Remainder 1: \n", remainder1.numpy())

    print("Filtered 2: \n", filtered2.numpy())
    print("Remainder 2: \n", remainder2.numpy())