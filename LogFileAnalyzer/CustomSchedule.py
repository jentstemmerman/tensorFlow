import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """
  Custom learning rate schedule that implements the learning rate function
  described in the original Transformer paper. The learning rate is increased
  linearly for the first `warmup_steps` training steps, and then decreased
  proportionally to the inverse square root of the step number.

  Args:
    d_model (int): the dimensionality of the model.
    warmup_steps (int): the number of steps taken to increase the learning rate
      linearly. Default is 4000.

  Attributes:
    d_model (float): the dimensionality of the model as a float.
    warmup_steps (int): the number of steps taken to increase the learning rate
      linearly.

  Methods:
    __call__(step): returns the learning rate at the given step.

  Returns:
    The learning rate at the given step.
  """
  def __init__(self, d_model, DataTypeForTensors, warmup_steps=3000):
    super().__init__()
    self.DataTypeForTensors = DataTypeForTensors
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, self.DataTypeForTensors)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    """
    Returns the learning rate at the given step.

    Args:
      step (int): the current training step.

    Returns:
      The learning rate at the given step as a float32 tensor.
    """
    step = tf.cast(step, dtype=self.DataTypeForTensors)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
  def get_config(self):
      config =  {
            "d_model": self.d_model.numpy(),
            "warmup_steps": self.warmup_steps,}

      return config  
  
  @classmethod
  def from_config(cls, config):
      return cls(**config)