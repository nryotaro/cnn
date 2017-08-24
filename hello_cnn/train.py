# -*- coding: utf-8 -*-
import tensorflow as tf


class TextClassifyLayer(object):
    """
    TODO:
        summary
        checkpoint

    """
    def __init__(self):

        with tf.Graph().as_default():
            sess = tf.InteractiveSession()
            sess.run(tf.initialize_all_variables())


def train():

    TextClassifyLayer()
