#!/usr/bin/env python

from __future__ import print_function, division

__doc__ = """
Feature extraction script for DMC 2014.

Author: Xin Yin <xinyin at iastate dot edu>
"""

from collections import defaultdict, Counter
from itertools import combinations
from argparse import ArgumentParser 

from dmc.datamanip import * 

import os
import logging

import pandas as pd
import numpy as np

### Some options, later will be moved into ArgumentParser
#GLOBAL_FEATURES = ['iid']
GLOBAL_FEATURES = ['mid', 'iid', 'size', 'color', ('iid', 'size'), 
                   ('iid', 'color')]

BATCH_FEATURES = ['iid', 'size', 'color', 'mid']

WITHIN_FEATURES = {
    'iid': ['size', 'color'],
    'size': ['iid', 'mid'],
    'color': ['iid', 'mid']
}

#HISTORICAL_FEATURES = ['iid']
HISTORICAL_FEATURES = ['iid', 'size', 'color', 
                       ('iid', 'size'), ('iid', 'color')]

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

    # for test only
    #train = train.iloc[:1000, :]

    # 3) whole dataset aggregation
    for feat in GLOBAL_FEATURES:
        feat_join = feat if type(feat) is str else "_".join(feat)

        name_S = "all.count.{0}".format(feature_name(feat))
        name_LLR = "all.llr.{0}".format(feature_name(feat))

        _group = train.groupby(feat, sort=False)

        p_S = _group.apply(trans_group_count)
        p_LLR = _group.apply(trans_llr, c1=c1, c2=c2)

        train[name_S] = p_S
        train[name_LLR] = p_LLR

    # 4) per batch summary statistics and ...
    # 5) features within substructures of batch
    batches = train.groupby('batch', sort=False)
    bat_feats = batches.apply(
        batch_summarize, u_feats=BATCH_FEATURES, wi_feats=WITHIN_FEATURES)

    # 6) historical features
    #    number of times an item had been purchased by customer
    #    number of batches an item had been purchased by customer
    #    number of times returned
    #    had been purchased?
    #    had been returned?
    hist_feat_list = []
    for feat in HISTORICAL_FEATURES:
        hist_row_cnts = defaultdict(int)
        hist_batch_cnts = defaultdict(int)
        hist_batch_returns = defaultdict(int)

        cols = ['cid']
        if type(feat) in (tuple, list):
            cols.extend(feat)
        elif type(feat) is str:
            cols.append(feat)

        #hist_feats = train[cols].apply(
        #    sequential_counter, counter=hist_row_cnts, axis=1)

        batch_hist_feats = batches.apply(
            batch_counter, order_cnts=hist_batch_cnts, 
            rt_cnts=hist_batch_returns, columns=cols)

        batch_hist_feats.columns = [
            'bhist.purcnt.{0}'.format(feature_name(feat)),
            'bhist.retcnt.{0}'.format(feature_name(feat))]

        hist_feat_list.append(batch_hist_feats)

    # concatenate features and dump
    feat_mat = pd.concat([train, bat_feats] + hist_feat_list, axis=1)
    feat_mat.to_csv("out.csv", index=False)

if __name__ == "__main__":
    main()
