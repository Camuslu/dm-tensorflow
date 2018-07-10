from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append('../dm-tensorflow/')

import numpy as np
import tensorflow as tf
import nn_layers

class Encoder(object):
    """This object encodes the input into a vector representation"""

    def __init__(self, embedding_sizes, vocab_sizes, embeddings, pads):
        """
        :param embedding_sizes: {"feat name": embed_size, ...}
        :param vocab_sizes: {"feat name": vocab_size, ...}
        :param pads: {"feat name": pad (0/V/..)}
        :param embeddings: {"feat name": tf.nn.embedding operation}

        """
        # tf.set_random_seed(54321)
        self.embedding_sizes = embedding_sizes
        self.vocab_sizes = vocab_sizes
        self.embeddings = embeddings
        self.pads = pads

    def encode(self, embed_inputs, dropouts):
        """
        an operation that concatenates all the embedded vectors after dropout

        :param embed_inputs: {"feat name":tf.placeholder(x1)], should be of type tf.int32
        :param dropouts: {"feat name": tf.placeholder(float), ..}
        :return: tf node
        """
        raise NotImplementedError("Abstract method")


class AvgWord2Vec(Encoder):
    def __init__(self, embedding_sizes, vocab_sizes, embeddings, pads):
        """
        :param embedding_sizes: {"feat name": embed_size, ...}
        :param vocab_sizes: {"feat name": vocab_size, ...}
        :param pads: {"feat name": pad (0/V/..)}
        :param embeddings: {"feat name": tf.nn.embedding operation}

        """
        # tf.set_random_seed(54321)
        super().__init__(embedding_sizes, vocab_sizes, embeddings, pads)

    def encode(self, embed_inputs, dropouts = None):
        
        """
        an operation that concatenates all the embedded vectors after dropout

        :param embed_inputs: {"feat name":tf.placeholder(x1)], should be of type tf.int32
        :param dropouts: {"feat name": tf.placeholder(float), ..}
        :return: tf node
        """
        embedded_before_concat = []
        for feature in self.embeddings.keys():
            masked_embedded = nn_layers.lookup_and_mask(embed_inputs[feature], self.embeddings[feature], self.pads[feature], feature)
            embedded = nn_layers.avg_w2v(input = embed_inputs[feature], embedded = masked_embedded, pad = self.pads[feature])
            if dropouts != None:
                with tf.name_scope("embedded_dropout"):
                    embedded = tf.nn.dropout(embedded, dropouts.get(feature,1.0), name = feature)
            embedded_before_concat.append(embedded)
        concat_after_embed = tf.concat(embedded_before_concat, axis=1, name="concat_after_embed")
        # tf.summary.histogram('avg_embedding', avg_embedding)
        return concat_after_embed

class CNNEncoder(Encoder):
    def __init__(self, embedding_sizes, vocab_sizes, embeddings, pads, widths, out_channels = 25):
        """
        :param embedding_sizes: {"feat name": embed_size, ...}
        :param vocab_sizes: {"feat name": vocab_size, ...}
        :param pads: {"feat name": pad (0/V/..)}
        :param embeddings: {"feat name": tf.nn.embedding operation}
        :param widths: a list of int for the width of filters, e.g. {"feat name": [3,4,5]}
        :param out_channels:
        """
        # tf.set_random_seed(54321)
        self.widths = widths
        self.out_channels = out_channels
        super().__init__(embedding_sizes, vocab_sizes, embeddings, pads)

    def encode(self, embed_inputs, dropouts):
        pooled_before_concat = []
        for feature in self.embeddings.keys():
            conv_x = nn_layers.convolution(embed_inputs[feature], self.embeddings[feature], self.embedding_sizes[feature], self.widths[feature], self.out_channels, self.pads[feature], name = feature)
            conv_x_dropout = tf.nn.dropout(conv_x, dropouts[feature])
            pooled_before_concat.append(conv_x_dropout)
        concat_after_conv = tf.concat(pooled_before_concat, 1, name = "concat_after_conv")
        return concat_after_conv

