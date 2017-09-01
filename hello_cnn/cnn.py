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
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        Args:
            sequence_length: the length of our sentences
        """

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(
            tf.float32,
            [None, sequence_length, embedding_size],
            name="input_x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # filter_shape shuld be
                # [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # the dimension of the first argment should be
                # [None, sequence_length, embedding_size, 1]
                b = tf.Variable(
                    tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                """
                Apply nonlinearity

                the demension of h is ? x 52 x 1 x 128
                """
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                """
                Maxpooling over the outputs
                the dimension pooled is ? x 1 x 1 x 128
                """
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        """
        tf.concat([ [[[[1]], [[2]]]], [[[[3]], [[4]]]]], 3)
        -> [[[[1, 3]], [[2, 4]]]]

        h_pool: ? x 1 x 1 x 384
        """
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            """
            tf.nn.l2_loss
            Computes half the L2 norm of a tensor without the sqrt:
                output = sum(t ** 2) / 2
            """
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            """
            Computes the mean of elements across dimensions of a tensor.
            'x' is [[1., 1.]
                    [2., 2.]]
            tf.reduce_mean(x) ==> 1.5
            tf.reduce_mean(x, 0) ==> [1.5, 1.5]
            tf.reduce_mean(x, 1) ==> [1.,  2.]
            """
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
