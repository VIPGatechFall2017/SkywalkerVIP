from layer import *
from parameters import *

def conv_net(x, weights, biases, img_size, num_channels, keep_prob=0.5):
    x = tf.reshape(x, [-1, img_size, img_size, num_channels])

    conv1 = conv2d_layer(x, weights['wc1'], biases['bc1'], tf.nn.relu)
    conv1 = maxpool2d_layer(conv1, strides1=2, strides2=2)
    conv1 = dropout_layer(conv1, name='dropout1', keep_prob=0.5)
    
    conv2 = conv2d_layer(conv1, weights['wc2'], biases['bc2'], tf.nn.relu)
    conv2 = maxpool2d_layer(conv2, strides1=2, strides2=2)
    conv2 = dropout_layer(conv2, name='dropout2', keep_prob=0.5)
    
    flat = flatten_layer(conv2)
    fc1 = fullyconnect_layer(flat, weights['wfc1'], biases['bfc1'], tf.nn.relu)
    logit = fullyconnect_layer(fc1, weights['wfc2'], biases['bfc2'])
    return logit