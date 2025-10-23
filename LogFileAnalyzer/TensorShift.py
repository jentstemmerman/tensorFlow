import tensorflow as tf

class TensorShifter:
    def __init__(self, shape, axis, shift, datatype=tf.float32):

        self.shape = tf.constant(shape, dtype=datatype)
        self.axis = axis
        self.shift = shift
        
        shift_abs = abs(shift)
        
        reduced_shape_zeros = self.set_shape_on_axis(shape, axis, shift_abs)
        reduced_shape_ones  = self.set_shape_on_axis(shape, axis, shape[axis] -  shift_abs)
        zeros = tf.zeros(reduced_shape_zeros, dtype=datatype)
        ones  = tf.ones(reduced_shape_ones,   dtype=datatype)
               
        if shift > 0:
            # in this case, the mask should be zero at the beginning, and ones at the end
            self.mask = tf.concat([zeros, ones], axis=self.axis)    
        else:
            # in this case, the mask should be ones at the beginning, and zeros at the end
            self.mask = tf.concat([ones, zeros], axis=self.axis)
               
    def reduce_shape(self, shape, axis, shift):    
        reduced_shape = tf.TensorShape(shape[:axis] + [shape[axis] + shift] + shape[axis+1:])
        return reduced_shape
    
    
    def set_shape_on_axis(self, shape, axis, value):    
        reduced_shape = tf.TensorShape(shape[:axis] + [value] + shape[axis+1:])
        return reduced_shape
  
    def shift_tensor(self, tensor):

        rolled = tf.roll(tensor, shift=self.shift, axis=self.axis)

        return rolled * self.mask
    
    
    
if __name__ == "__main__":
    
    # small test case
    
    simularities = tf.constant([[1,2,3,4,5], [1,3,5,7,6], [1,3,5,7,7]], dtype=tf.float32)
    print("Original similarities:")
    print(simularities.numpy())
    
    ahifter = TensorShifter(shape=simularities.shape, axis=0, shift=1, datatype=tf.float32)
    shifted = ahifter.shift_tensor(simularities)
    print("Shifted similarities:")
    
    print(shifted.numpy())