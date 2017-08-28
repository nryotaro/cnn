# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def create_label_binarizer(src, label_index=1) -> LabelBinarizer:
    df = pd.read_csv(src)
    lb = LabelBinarizer()
    lb.fit(df.iloc[:, label_index].unique())
    return lb
