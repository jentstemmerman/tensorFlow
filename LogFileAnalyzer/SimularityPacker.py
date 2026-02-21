import numpy as np
import tensorflow as tf
import BinaryFileHandler


class SimilarityPacker:
    def __init__(self, similarity_threshold, DataTypeForTensors):
        self.similarity_threshold = similarity_threshold
        self.DataTypeForTensors = DataTypeForTensors

    def pack(self, input_file_path, output_file_path):
        """
        Reads similarity values from a binary file and packs intervals
        where values exceed the threshold into (lineNumber, numLines) records.
        Output is a flat binary file of int32 pairs for efficient transfer.
        """
        assert isinstance(input_file_path, str), "input_file_path must be a string"

        inputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(input_file_path, data_type=self.DataTypeForTensors, numbers_per_line=1, sequence_length=None)

        recordCount = inputBinaryFileHandler.get_recordCount()
        
        simularity_data = inputBinaryFileHandler.read_records(0, recordCount)
        similarity_data = tf.squeeze(simularity_data)
        similarity_data = np.array(similarity_data, dtype=np.float32)
        
        # Boolean mask of values above threshold
        above = similarity_data < self.similarity_threshold

        # Detect transitions by diffing the boolean mask
        # Pad with False at both ends to catch intervals at boundaries
        padded = np.concatenate(([False], above, [False]))
        diff = np.diff(padded.astype(np.int8))

        # Rising edges (0->1) = interval starts, falling edges (1->0) = interval ends
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # Build (lineNumber, numLines) pairs directly as numpy array
        packed_data = np.column_stack((starts, ends - starts)).astype(np.uint64) if len(starts) > 0 else np.empty((0, 2), dtype=np.uint64)

        if (output_file_path is not None) and (len(packed_data) > 0):
            assert isinstance(output_file_path, str), "output_file_path must be a string"
            
            with open(output_file_path, 'wb') as f:
                f.write(packed_data.tobytes())



        return len(starts)

