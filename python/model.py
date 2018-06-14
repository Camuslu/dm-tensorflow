import sys, os
sys.path.append(sys.path[0] + '/../../dm-tensorflow/python/')
import numpy as np
import tensorflow as tf
import encoder


class Model(object):
    """FastSentModel model"""

    def __init__(self, feat_length, input_encoder: encoder.Encoder, logdir):
        self.sess = tf.Session()
        # self.merged = tf.summary.merge_all()
        self.feat_length = feat_length
        self.logdir = logdir
        self.train_writer = tf.summary.FileWriter(logdir, self.sess.graph)
        self.input_encoder = input_encoder


    def init_variables(self):
        self.sess.run(tf.global_variables_initializer())

    def export(self, model_path, output_node_names):
        """
        Freezes the session and serializes it at the given path.
        :param model_path: The path at which to save the model.
        """
        graph_as_constants = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph.as_graph_def(),
                                                                          output_node_names=output_node_names)
        tf.train.write_graph(graph_as_constants, model_path, as_text=False, name="model_graph")

    def train(self, feed_dict, step):
        raise NotImplementedError("Abstract method")

    def test(self, feed_dict, step):
        raise NotImplementedError("Abstract method")

    def close(self):
        self.sess.close()