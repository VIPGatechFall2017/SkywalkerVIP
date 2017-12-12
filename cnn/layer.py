import tensorflow as tf

def conv2d_layer(x, W, b, act, strides1=1, strides2=1, name='conv', padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    with tf.name_scope('conv2d'):
        conv = tf.nn.conv2d(x, W, strides=[1, strides1, strides2, 1], padding=padding)
        conv = act(conv + b)
    return conv

def maxpool2d_layer(x, strides1=2, strides2=2, padding='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, strides1, strides2, 1], strides=[1, strides1, strides2, 1],
                          padding=padding)

def dropout_layer(x, name='dropout', keep_prob=0.5):
    # Dropout wrapper
    return tf.nn.dropout(x, keep_prob)

def flatten_layer(x):
    # Flatten wrapper
    layer_shape = x.get_shape()
    num_features = layer_shape[1:4].num_elements()
    flat = tf.reshape(x, [-1, num_features])
    return flat

def fullyconnect_layer(x, W, b, act=None):
    fc = tf.matmul(x, W) + b
    if act != None:
        fc = act(fc)
    return fc