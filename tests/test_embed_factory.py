# -*- coding: utf-8 -*-
from hello_cnn.embed_factory import create_batch_gen, _count_txt_file_lines
from io import StringIO
import os
from unittest.mock import patch 


txt = StringIO(
    ('test_data\n'
     '0\n'
     '1\n'
     '2\n'
     '3\n'
     '4\n'
     '5\n'
     '6\n'
     '7\n'
     '8\n'
     '9\n'))


def test__count_txt_file_lines():
    assert _count_txt_file_lines(txt) == 11


@patch('numpy.random.permutation')
def test__create_batch_gen(m):
    m.side_effect = lambda *x: [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    p = os.path.join(os.path.dirname(__file__), 'test_data.csv')
    list(create_batch_gen(p, 3))
    assert False
