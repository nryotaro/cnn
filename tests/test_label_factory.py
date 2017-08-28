# -*- coding: utf-8 -*-
from io import StringIO
from hello_cnn.label_factory import create_label_binarizer
import numpy as np


def test_create_label_binarizer():

    txt = StringIO('id,label,text\n1,a,aa\n2,b,bb')

    binarizer = create_label_binarizer(txt, 1)
    assert (binarizer.transform(
        ['b', 'a', 'c']) == np.array([[0, 1], [1, 0], [0, 0]])).all(),\
        'Returns a pre-fitted LabelBinarizer'
