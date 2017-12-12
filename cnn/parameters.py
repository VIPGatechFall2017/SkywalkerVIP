import tensorflow as tf

# Create some wrappers for simplicity
def weight_variable(shape):
    # weight placeholder wrapper
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def bias_variable(shape):
    # bias placeholder wrapper
    return tf.Variable(tf.constant(0.01, shape=shape))