#!/usr/bin/env python
# -*- coding: utf-8

from argparse import ArgumentParser
from itertools import combinations
from scipy import sparse

import re
import multiprocessing 

import numpy as np
import pandas as pd

import lrutils.io
import lrutils.transform

def main():
    X_test, y_test = lrutils.io.load_pickle(args.feature_matrix, 'return')

    candidates = ('cid', 'iid', re.compile('bwi_'), re.compile('bat\.'),
                  'deal', 'isdisc', 'zsize', 'ztype', 'mid', 'state',
                  'month', 'season', 'dow', 'lowdisc', 'pricer')

    # find all matching columns
    X_cand = lrutils.transform.select(X_test, candidates)

    # transform all columns to factors (label encoded integers)
    X_factor = lrutils.transform.as_factor(X_cand)

    # all interactions
    X_factor.span_interactions(slice(None))

    # perform one hot encoding
    X_ohe = lrutils.transform.one_hot_encode(X_factor)

    # Lasso logistic regression ??
    
