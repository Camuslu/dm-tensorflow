from nn_layers import *
import tensorflow as tf
import numpy as np

#test to see if conv layer output size is as expected
dim = 5
input = tf.placeholder(tf.int32, shape = [4, 8])
embedding_tensor = tf.placeholder(tf.float32, shape = [10, dim])
widths = [3, 4, 5]
out_channels = 3
pad = 0

conv_output = convolution(input, embedding_tensor, dim, widths = widths, out_channels= out_channels, pad = pad, name = "conv_test")
np.random.seed(5)
input_ = np.array([[3,1,5,5,1,2,3,7],
                   [6,6,6,6,6,0,0,0],
                   [1,2,3,4,5,0,0,0],
                   [3,0,0,0,0,0,0,0]])
embed_ = np.random.normal(size = (10, dim))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    conv_output_val = sess.run(conv_output, feed_dict = {input: input_, embedding_tensor: embed_})

print(conv_output_val) #shape should be [batchsize, output_channel x len(width)] = [4, 2*3 ] = [4,6] in the case above



#test to see if conv layer output size is as expected
dim = 5
input = tf.placeholder(tf.int32, shape = [4, 8])
embedding_tensor = tf.placeholder(tf.float32, shape = [10, dim])
widths = [3, 4, 5]
out_channels = 3
pad = 0

conv_output = convolution(input, embedding_tensor, dim, widths = widths, out_channels= out_channels, pad = pad, name = "conv_test")
np.random.seed(5)
input_ = np.array([[3,1,5,5,1,2,3,7],
                   [6,6,6,6,6,0,0,0],
                   [1,2,3,4,5,0,0,0],
                   [3,0,0,0,0,0,0,0]])
embed_ = np.random.normal(size = (10, dim))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    conv_output_val = sess.run(conv_output, feed_dict = {input: input_, embedding_tensor: embed_})

print(conv_output_val) #shape should be [batchsize, output_channel x len(width)] = [4, 2*3 ] = [4,6] in the case above


