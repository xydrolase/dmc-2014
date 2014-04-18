#!/usr/bin/env python

from math import log

import numpy as np
import pandas as pd

# TODO: to write a broadcast decorator 

def trans_group_count(x):
    # IMPORTANT: Preserving original index
    return pd.Series(np.repeat(len(x), len(x)), index=x.index)

def trans_llr(x, c1, c2):
    R = float(np.sum(x['return']))
    NR = len(x) - R
    
    return pd.Series(np.repeat(log((R + c1) / (NR + c2)), len(x)), 
                     index=x.index)
