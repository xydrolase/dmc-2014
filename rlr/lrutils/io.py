#!/usr/bin/env python

import pandas as pd

def load_pickle(data_file, y_column):
    _d = pd.read_pickle(data_file)
    return _d.ix[:, _d.columns != y_column], _d[y_column] 
