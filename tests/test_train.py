# -*- coding: utf-8 -*-
from hello_cnn.train import binarize
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf


def test_binarize():
    labels = ('a', 'b', 'c', 'd', 'a')
    unique_labels = set(labels)
    bin = LabelBinarizer().fit(sorted(unique_labels))
    res = binarize(bin, labels)

    assert (res == [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0]]).all()

    num_classes = len(unique_labels)
    input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")

    tf.Session().run(input_y, {input_y: res})
