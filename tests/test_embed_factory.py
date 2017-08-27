# -*- coding: utf-8 -*-
from hello_cnn.embed_factory import create_batch_gen, _count_txt_file_lines
from io import StringIO


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


def test__create_batch_gen():
    a = create_batch_gen(txt, 2)
