#!/usr/bin/env python

from __future__ import print_function, division

__doc__ = """
Feature extraction script for DMC 2014.

Author: Xin Yin <xinyin at iastate dot edu>
"""

from collections import defaultdict, Counter
from itertools import combinations, izip 
from argparse import ArgumentParser 

from dmc.datamanip import trans_group_count, trans_llr, batch_summarize

import os
import logging

import pandas as pd
import numpy as np

### Some options, later will be moved into ArgumentParser
GLOBAL_FEATURES = ['iid']
#GLOBAL_FEATURES = ['iid', 'size', 'color', ('iid', 'size'), 
#                   ('iid', 'color')]

BATCH_FEATURES = ['iid', 'size', 'color', 'mid']

WITHIN_FEATURES = {
    'iid': ['size', 'color'],
    #'size': ['iid', 'mid'],
    #'color': ['iid', 'mid']
}

def parse_options():
    parser = ArgumentParser(
        description="DMC2014 feature matrix extraction script.")

    parser.add_argument("--c1c2", type=float, default=[1.0, 1.0], nargs=2,
                        metavar='NUM',
                        help="Constants c1 c2 used in computing LLR")
    parser.add_argument("data_matrix", 
                        help="Path to the (cleaned) input data matrix")

    args = parser.parse_args()
    return args

def main():
    args = parse_options()

    c1, c2 = args.c1c2

    # 1) data loading
    assert os.path.exists(args.data_matrix)
    train = pd.read_csv(args.data_matrix, sep=',')

    # 2) split rows/orders into batches by identifying consecutive customer IDs
    # (cid).
    cid_edge = np.concatenate(
        [[1], np.array(train['cid'][1:]) - np.array(train['cid'][:-1])])
    cid_edge[cid_edge != 0] = 1
    train['batch'] = np.cumsum(cid_edge)

    train_test = train.iloc[:100, :]

    # 3) whole dataset aggregation
    for feat in GLOBAL_FEATURES:
        break
        feat_join = feat if type(feat) is str else "_".join(feat)

        name_S = "S_{0}".format(feat_join)
        name_LLR = "LLR_{0}".format(feat_join)

        _group = train.groupby(feat)

        p_S = _group.apply(trans_group_count)
        p_LLR = _group.apply(trans_llr, c1=c1, c2=c2)

        train[name_S] = p_S
        train[name_LLR] = p_LLR

    # 4) per batch summary statistics and ...
    # 5) features within substructures of batch

    batches = train.groupby('batch')
    bat_feats = batches.apply(
        batch_summarize, u_feats=BATCH_FEATURES, wi_feats=WITHIN_FEATURES)

    print(bat_feats.head(n=50))

if __name__ == "__main__":
    main()
