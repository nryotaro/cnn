# -*- coding: utf-8 -*-
from hello_cnn.embed_factory import EmbedFactory
from io import StringIO
import os
import pandas as pd
from unittest.mock import MagicMock


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
        p = os.path.join(os.path.dirname(__file__), 'test_data.csv')
        f = EmbedFactory(None)

        for a in f._create_batch_gen(p, 3):
            for k, r in a.iterrows():
                assert len(r) == 3
                expected.append((r[0], r[1], r[2]))

        sorted(expected, key=lambda r: r[0]) == [
            (0, 'a', 10),
            (1, 'b', 11),
            (2, 'c', 22),
            (3, 'd', 33),
            (4, 'e', 44),
            (5, 'f', 55),
            (6, 'g', 66),
            (7, 'h', 77),
            (8, 'i', 88),
            (9, 'j', 99)]

    def test_create_betch_gen(self):
        m, _create_batch_gen = MagicMock(), MagicMock(
            return_value=[pd.DataFrame([['001', 'aaa', 'a'],
                                        ['002', 'bbb', 'b']])])
        m.vectorize = lambda txt: [[0, 1]] if txt == 'a' else [[1, 0]]
        f = EmbedFactory(m)

        f._create_batch_gen = _create_batch_gen

        res = list(f.create_batch_gen(None, None))[0]

        print(res)
        assert res['id'] == '001'
        assert res['label'] == 'aaa'
        assert (res['text'] == pd.DataFrame([[0, 1], [1, 0]])).all().all()
