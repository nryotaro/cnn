# -*- coding: utf-8 -*-
from hello_cnn.train import binarize
from sklearn.preprocessing import LabelBinarizer


def test_binarize():
    labels = ('a', 'b', 'c')
    bin = LabelBinarizer().fit(labels)
    res = binarize(bin, labels)
    assert (res == [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]).all()
