# -*- coding: utf-8 -*-
from hello_cnn.embed_factory import EmbedFactory
import tensorflow as tf
from io import StringIO
import os
import numpy as np
from unittest.mock import MagicMock
import pprint as pp


test_data_path = os.path.join(
    os.path.dirname(__file__), 'test_data.csv')


class TestEmbedFactory(object):

    def test__count_txt_file_lines(self):
        txt = StringIO(
            ('test_data\n'
             '0\n'
             '1\n'
             '2\n'))
        f = EmbedFactory(None)
        assert f._count_txt_file_lines(txt) == 4

    def test__create_batch_gen(self):
        expected = []

        f = EmbedFactory(None)

        for a in f._create_batch_gen(test_data_path, 3):
            for k, r in a.iterrows():
                assert len(r) == 3
                expected.append((r[0], r[1], r[2]))

        sorted(expected, key=lambda r: r[0]) == [
            (0, 'a', 'aa'),
            (1, 'b', 'bb'),
            (2, 'c', 'cc'),
            (3, 'd', 'dd'),
            (4, 'e', 'ee'),
            (5, 'f', 'ff'),
            (6, 'g', 'gg'),
            (7, 'h', 'hh'),
            (8, 'i', 'ii'),
            (9, 'j', 'jj')]

    def test_create_betch_gen(self):
        m = MagicMock()
        m.vectorize = lambda txt: np.array([[0, 1]])
        f = EmbedFactory(m)

        res = list(f.create_batch_gen(test_data_path, 2))
        pp.pprint(res)
        assert len(res) == 5
        assert [len(e) for e in res] == [2, 2, 2, 2, 2]
        assert res[0][1][0] \
            in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j')
        assert (res[0][0][0] == np.array([[0, 1]])).all()

    def test_create_epoch_batch_gen(self):
        m = MagicMock()
        m.vectorize = lambda txt: np.array([[0, 1]])
        f = EmbedFactory(m)

        res = list(f.create_epoch_batch_gen(test_data_path, 2, 4))

        assert len(res) == 20, \
            ('The size of data is 10, batch size is 2, epoch size is 4, '
             'so it yields 20 elements.')

        assert len(res[0]) == 2, 'The size of each element is batch size'

        embedding_size, text_length = 2, 1
        input_x = tf.placeholder(tf.int32,
                                 [None, text_length, embedding_size],
                                 name='input_x')
        sess = tf.Session()
        for x, y in f.create_epoch_batch_gen(test_data_path, 2, 4):
            x  # e.g., (array([[0, 1]]), array([[0, 1]]))
            y  # e.g., ('a', 'b')
            sess.run(input_x, {input_x: x})
