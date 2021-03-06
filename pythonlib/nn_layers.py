from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq
# from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
import math
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper


tf.logging.set_verbosity(tf.logging.INFO)


def fully_connected(input, input_size, output_size, const=0.0, name=""):
    with tf.name_scope("weights_biases"):
        W = tf.get_variable(name="weights-%s" % name, shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer(seed = 42), trainable=True)
        tf.summary.histogram(name + "W", W)

        b = tf.get_variable(name="biases-%s" % name, shape=[output_size],
                            initializer=init_ops.constant_initializer(const, dtype=tf.float32),
                            trainable=True)
        tf.summary.histogram(name + "B", W)

    with tf.name_scope("layer"):
        layer = tf.nn.xw_plus_b(input, W, b, name = name)
    return layer


def embedding(vocab_size, embedding_size, name="", pretrained=None, init = "normal"):
    embeddings = None
    with tf.name_scope("embeddings"):
        if pretrained is not None:
            embeddings = tf.get_variable(name="Ww-%s" % name, shape=pretrained.shape,
                                         initializer=tf.constant_initializer(pretrained),
                                         trainable=True, )
        elif init == "normal":
            embeddings = tf.get_variable(name="Ww-%s" % name,
                                         trainable=True,
                                         dtype=tf.float32,
                                         initializer=init_ops.VarianceScaling(mode='fan_out', seed = 42),
                                         shape=[vocab_size, embedding_size])
        elif init == "uniform":
            embeddings = tf.get_variable(name="Ww-%s" % name,
                                         trainable=True,
                                         dtype=tf.float32,
                                         initializer=init_ops.random_uniform_initializer(minval = -1/vocab_size, maxval = 1/vocab_size, seed = 42),
                                         shape=[vocab_size, embedding_size])
        elif init == "xavier":
            embeddings = tf.get_variable(name="Ww-%s" % name,
                                         trainable=True,
                                         dtype=tf.float32,
                                         initializer=init_ops.glorot_uniform_initializer(seed = 42),
                                         shape=[vocab_size, embedding_size])
        else:
            raise Exception("embedding initialize: %s is either pretrained, or initialized from {normal, uniform, xavier}" %(name))

    return embeddings


def lookup_and_mask(input, embedding, pad = 0, name = "", dim = 2):
    """
    :param input:  Tensor of any shape, usually 2D [batch_size, length]
    :param embedding: 2D tensor embedding
    :param pad: the pad that will be embedded to [0.0,...0.0]. If input is padded as [a,b,c,x,x,x], set pad = x
    :param name:
    :param dim:
    :return:
    """
    embeddings = tf.nn.embedding_lookup(embedding, input)
    pad_mask = tf.expand_dims(tf.cast(tf.not_equal(input, pad), dtype=tf.float32), dim = dim)
    with tf.name_scope ("masked_embedding"):
        masked_embedding = tf.multiply(embeddings, pad_mask, name = name)
    return masked_embedding


def avg_w2v(input, embedded, pad = 0, name = ""):
    sum_embedding = tf.reduce_sum(embedded, 1)
    embedding_length = tf.reduce_sum(tf.cast(tf.not_equal(input, pad), dtype=tf.float32), axis=1,
                                     keep_dims=True)

    embedding_length_smoothed = tf.where(tf.greater(embedding_length, 0.0), embedding_length, tf.ones(tf.shape(embedding_length))) # if length = 0, change it to 1 to avoid nan!
    avg_embedding = sum_embedding / embedding_length_smoothed
    return avg_embedding


def convolution(input, embedding, dim, widths, out_channels, pad, name=''):
    """
    https://arxiv.org/pdf/1510.03820.pdf has a nice visualization

    Assume input is [batchsize, seq_len]
    Embedd

    :param input: tf.placeholder,
    :param embedding: tf embedding matrix
    :param dim: embed dim
    :param widths: the width of filter
    :param out_channels: number of output channels after convolution
    :param name:
    :return: tensor of shape [batchsize, out_channels x len(widths) ]
    """
    output = []
    seq_len = tf.shape(input)[1]
    embedded_pad_masked = lookup_and_mask(input, embedding, pad=pad, name=name) # [batchsize, seq_len, embed_dim]. zero-padded
    for width in widths:
        w = tf.get_variable(name="weights-%s-%s" % (name, width), shape=[width, dim, out_channels],
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
        output.append(pool)
    cnn_out = tf.squeeze(tf.concat(output, 3), [1, 2])
    return cnn_out


def recurrent_operation(input, rnn_dim, embedding, cell_type, pad, input_dropout = 0.5, output_dropout = 0.5, state_dropout = 0.5, name = ''):
        masked_embedded = lookup_and_mask(input, embedding, pad, name= name) # [batch_size, seq_len, embed_dim]
        sequence_lengths = tf.reduce_sum(tf.cast(tf.not_equal(input, pad), tf.int32), axis=1)
        cell_with_dropout = get_cell_type(rnn_dim, cell_type, input_dropout, output_dropout, state_dropout)

        rnn_output, rnn_state = tf.nn.dynamic_rnn(cell=cell_with_dropout, inputs=masked_embedded, sequence_length=sequence_lengths, dtype=tf.float32)
        return rnn_output, rnn_state


def get_cell_type(rnn_dim, cell_type, input_dropout, output_dropout, state_dropout):
    if cell_type == 'LSTM':
        cell =  BasicLSTMCell(num_units=rnn_dim, name="lstmCell")
    elif cell_type == 'GRU':
        cell =  GRUCell(num_units=rnn_dim, name="gruCell")
    else:
        print("specify encoder cell type in {LSTM, GRU}")
    cell_with_dropout = DropoutWrapper(cell, input_keep_prob =  1 - input_dropout, output_keep_prob= 1- output_dropout, state_keep_prob= 1- state_dropout )
    return cell_with_dropout



