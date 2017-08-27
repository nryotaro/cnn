# -*- coding: utf-8 -*-
import tensorflow as tf


class Cnn(object):
    """A CNN for text classification.

    Uses an embedding layer,
    followed by a convolutional, max-pooling and softmax layer.

    TODO:
        summary
        checkpoint

    Properties:
        loss
        accuracy
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        Args:
            sequence_length: the length of our sentences
        """

        # Placeholders for input, output and dropout
        # 
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # Convolution Layer
            # filter_shape shuld be
            # [filter_height, filter_width, in_channels, out_channels]
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            # the dimension of the first argment should be
            # [None, sequence_length, embedding_size, 1]
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            pass
