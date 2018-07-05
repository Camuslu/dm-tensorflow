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



seq_len = tf.shape(input)[1]
embedded_pad_masked = lookup_and_mask(input, embedding_tensor, pad=pad, name= 'blah') # [batchsize, seq_len, embed_dim]. zero-padded
width = 5
w = tf.get_variable(name="weights-%s-%s" % ('blah', width), shape=[width, dim, out_channels],
                    initializer=tf.contrib.layers.xavier_initializer(seed = 42), trainable=True)
b = tf.Variable(tf.zeros(shape=[out_channels]) + 0.1)
conv_width = seq_len - width + 1
conv = tf.nn.conv1d(embedded_pad_masked, w, 1, 'VALID') # tensor of [batchsize, out_width, out_channels]
conv_biased = tf.nn.bias_add(conv, b)
condition = tf.equal(conv, 0.0)
epsilon = -10.0
epsilon_mask = tf.where(condition, tf.fill([tf.shape(input)[0], conv_width, out_channels], epsilon),
                                   tf.fill([tf.shape(input)[0], conv_width, out_channels], 0.0))
conv_relu_out = tf.nn.relu(conv_biased)
epsilon_adjusted = tf.add(epsilon_mask, conv_relu_out)
pool = tf.reduce_max(tf.expand_dims(epsilon_adjusted, 2), axis=1, keep_dims=True)
