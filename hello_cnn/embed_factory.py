# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def _count_txt_file_lines(src):

    def count(f):
        return sum(1 for line in f)

    if isinstance(src, str):
        with open(src, 'r') as f:
            return count(f)

    return count(src)


def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_batch_gen(src, batch_size):

    data_size = _count_txt_file_lines(src)

    # 1 in range(1, _): ignore the header line
    indices = np.random.permutation(range(1, data_size))

    for chunk in _chunks(indices, batch_size):
        print(chunk)
        res = pd.read_csv(src, skiprows=lambda n: n not in chunk,
                          header=None)
        print(res)
        yield res
