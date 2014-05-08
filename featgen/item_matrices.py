#!/usr/bin/env python
# -*- coding: utf-8

"""
Implementing a network model that advanced by Guillermo, Zoe and Cory,
depending on discussions with Fan and Cory.
"""

from scipy import sparse
from sklearn import preprocessing
from itertools import combinations
from collections import Counter

import sys

import pandas as pd
import numpy as np

def main():
    train = pd.read_csv(sys.argv[1])
    vs = pd.read_csv(sys.argv[2])

    validset = vs['L3']

    # remove all rows with deldate == NA
    train = train.drop(train.index[np.isnan(train['deldate'])])
    # reset index
    train.index = range(len(train))

    # subsetting columns that we need
    cols_subset = ['cid', 'iid', 'mid'] 

    # split into training set and learning set
    tr = train[~validset] 
    va = train[validset] 

    # leakage protection
    va.ix[:, 'return'] = np.nan
    
    encoders = {}
    # relabel iid, cid, mid
    for col in cols_subset:
        lblenc = preprocessing.LabelEncoder()
        # fitting using the union of all possible values
        # that can exist both in training and learning sets.
        encoders[col] = lblenc.fit(train[col])

        tr.ix[:, col] = lblenc.transform(tr[col])
        va.ix[:, col] = lblenc.transform(va[col])

    nitems = len(encoders['iid'].classes_)
    ncusts = len(encoders['cid'].classes_)

    # deprecated: using sklearn.preprocessing.LabelEncoder() instead.
    # nitems = sorted(set(train['iid']))
    # iidmap = dict(zip(nitems, range(len(nitems))))

    count_matrices = [
        np.zeros((nitems, nitems), dtype=int), # kept/kept
        np.zeros((nitems, nitems), dtype=int), # kept/return
        np.zeros((nitems, nitems), dtype=int), # return/kept
        np.zeros((nitems, nitems), dtype=int), # return/return
    ]

    # sparse matrices, since customer purchase history will be 
    # VERY VERY sparse in reality
    #Cret = sparse.lil_matrix((ncusts, nitems))
    #Ckep = sparse.lil_matrix((ncusts, nitems))
    Cret = np.zeros((ncusts, nitems))
    Ckep = np.zeros((ncusts, nitems))

    # this will sort cid in the transformed order
    customers = tr.groupby('cid', sort=True)

    for cid, gidx in customers.groups.items():
        if cid % 1000 == 0:
            print(" :: {0}/{1}".format(cid, ncusts))

        sub_df = tr.ix[gidx, :]

        items = list(sub_df[['iid', 'return']].itertuples(index=False))

        # if the exact same pair occurs multiple times for a given customer,
        # only count as once.
        all_pairs = set(combinations(
            sub_df[['iid', 'return']].itertuples(index=False), 2))

        # keepers
        _keeps = [iid for iid, ret in items if ret == 0]
        if _keeps:
            indx, summand = zip(*tuple(Counter(_keeps).items()))

            _chist = np.zeros(nitems)
            _chist[list(indx)] += summand

            Ckep[cid, :] = _chist

        # returns
        _rets = [iid for iid, ret in items if ret == 1]
        if _rets:
            indx, summand = zip(*tuple(Counter(_rets).items()))

            _chist = np.zeros(nitems)
            _chist[list(indx)] += summand

            Cret[cid, :] = _chist

        for (iid1, ret1), (iid2, ret2) in all_pairs:
            matidx = ret1 * 2 + ret2

            count_matrices[matidx][iid1, iid2] += 1
            if matidx == 1 or matidx == 2:
                # symmetric keep/return - return/keep
                count_matrices[3-matidx][iid2, iid1] += 1
            else:
                count_matrices[matidx][iid2, iid1] += 1

    print("Converting to sparse matrices...")
    Ckep = sparse.csr_matrix(Ckep)
    Cret = sparse.csr_matrix(Cret)
    for i in range(4):
        count_matrices[i] = sparse.csc_matrix(count_matrices[i])

    print("Matrix multiplications...")

    # for validation set, compute the features
    Nkept_kept = (Ckep * count_matrices[0]).toarray()
    Nkept_ret = (Ckep * count_matrices[1]).toarray()

    Nret_kept = (Cret * count_matrices[2]).toarray()
    Nret_ret = (Cret * count_matrices[3]).toarray()


    del Ckep
    del Cret

    print("Computing log-likelihood ratios...")

    Mllr_kept = np.log((Nkept_ret + 0.5) / (Nkept_kept + 0.5))
    Mllr_ret = np.log((Nret_ret + 0.5) / (Nret_kept + 0.5))

    del Nkept_ret
    del Nkept_kept
    del Nret_kept
    del Nret_ret

    print("Generating features...")

    llr_kept = [Mllr_kept[cid, iid]
                for cid, iid in va[['cid', 'iid']].itertuples(index=False)]
    llr_ret = [Mllr_ret[cid, iid]
                for cid, iid in va[['cid', 'iid']].itertuples(index=False)]

    llrs = pd.DataFrame({'nm.llr.kept': llr_kept, 'nm.llr.ret': llr_ret})
    llrs.to_csv("llr.csv")

if __name__ == "__main__":
    main()
