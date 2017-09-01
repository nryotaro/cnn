# -*- coding: utf-8 -*-
from hello_cnn.train import binarize, read_test_data
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from io import StringIO
from unittest.mock import MagicMock
import numpy as np


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


def test_read_test_data():
    txt = StringIO((
        'id,label,desc\n'
        '1,a,aa bb\n'
        '2,b,cc dd\n'
        '3,c,ee ff'))
    binarizer = LabelBinarizer()
    binarizer.fit(['a', 'b', 'c'])
    m = MagicMock()
    m.vectorize = lambda txt: np.array(
        [[0.1, 0.2, 0.4, 5], [0.3, 0.4, 0.2, 0.1]])
    x, y = read_test_data(txt, binarizer, m)
    assert x.shape == (3, 2, 4), \
        ('shape of x must be ('
         '   size of data, length of each text, embedding_size'
         ')')
