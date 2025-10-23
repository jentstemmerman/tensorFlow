import tensorflow as tf
from TensorShift import TensorShifter


class SimilarityMaxShifter:
    def __init__(self, number_of_simularities, max_similarity_length, shift, signatureLengthList):
        self.max_similarity_length = max_similarity_length
        self.shift = shift
        self.signatureLengthList  = signatureLengthList
        self.number_of_simularities = number_of_simularities
        self.shifter = TensorShifter(shape=[number_of_simularities, max_similarity_length], axis=1, shift=-1, datatype=tf.float32)
    
    def GetShiftedMaxSimilarity(self, simularities):
        
        self.shifter.shift_tensor(simularities)
        return self.previous_similarities
    
    
