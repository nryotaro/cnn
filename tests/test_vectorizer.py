# -*- coding: utf-8 -*-
from hello_cnn.vectorizer import Vectorizer, build_vectorizer
from unittest.mock import MagicMock, patch
import numpy as np


class TestVectorizer(object):

    def test__to_alphabet_word_list(self):
        vec = Vectorizer(MagicMock())
        words = vec._to_alphabet_word_list(
            'The quick brown fox jumps over the lazy dog ! あ')

        assert words == ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']

    def test__to_word_matrix(self):
        m = {'the': [1, 2, 3], 'quick': [4, 5, 6], 'fox': [7, 8, 9]}
        vec = Vectorizer(m, length=2)
        a = vec._to_word_matrix(['the', 'quick', 'fox'])
        assert a == [[1, 2, 3], [4, 5, 6]]

    def test__to_word_matrix_without_cut(self):
        m = {'the': [1, 2, 3], 'quick': [4, 5, 6], 'fox': [7, 8, 9]}
        vec = Vectorizer(m)
        a = vec._to_word_matrix(['the', 'quick', 'fox'])
        assert a == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test__padding(self):
        m = MagicMock()
        m.vector_size = 3
        vec = Vectorizer(m, length=5)

        a = vec._padding(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        assert a.shape == (5, 3)

    def test__padding_filled(self):
        m = MagicMock()
        m.vector_size = 3
        vec = Vectorizer(m, length=3)

        a = vec._padding(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        assert (a == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).all()


@patch('gensim.models.Word2Vec.load_word2vec_format')
def test_build_vectorizer(m):
    build_vectorizer('model_path')
    m.assert_called_once()
