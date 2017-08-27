# -*- coding: utf-8 -*-
import re
import numpy as np
from nltk.tokenize import TweetTokenizer
from hello_cnn.stop_words import stop_words


class Vectorizer(object):

    def __init__(self, model, length=200, tokenizer=TweetTokenizer(),
                 alphabet_pattern=re.compile('^[a-zA-Z]+$')):
        self.model = model
        self.tokenizer = tokenizer
        self.alphabet_pattern = alphabet_pattern
        self.length = length

    def _to_alphabet_word_list(self, txt):
        return list((word.lower() for word in self.tokenizer.tokenize(txt)
                     if self.alphabet_pattern.match(word)
                     and word.lower() not in stop_words))

    def _to_word_matrix(self, words):
        vectorizable_words = list(filter(
            lambda word: word in self.model, words))[0:self.length]

        return [self.model[word] for word in vectorizable_words]

    def _padding(self, matrix):
        padding_length = self.length - matrix.shape[0]

        if padding_length <= 0:
            return matrix

        mu = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)

        pad = np.random.normal(
            mu, std, [padding_length, self.model.vector_size])

        return np.concatenate((matrix, pad), axis=0)

    def vectorize(self, txt):
        ary = np.array(self._to_word_matrix(self._to_alphabet_word_list(txt)))
        return self._padding(ary)
