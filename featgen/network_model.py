#!/usr/bin/env python
# -*- coding: utf-8

from __future__ import print_function

"""
Implementing a network model that advanced by Guillermo, Zoe and Cory,
depending on discussions with Fan and Cory.
"""

from scipy import sparse
from sklearn import preprocessing
from itertools import permutations, repeat
from collections import Counter
from argparse import ArgumentParser

import sys

import pandas as pd
import numpy as np

def elems_to_coo_matrix(coords, shape):
    """Converts a list of 3-tuples, where each 3-tuple encodes:
        (data, rowidx, colidx),
       into a COOrdinate sparse matrix."""

    data, i, j = zip(*coords)
    return sparse.coo_matrix((data, (i, j)), shape=shape)

def main():
    parser = ArgumentParser(
        description="Generate network model features.")
    parser.add_argument("-d", dest="data_matrix", required=True,
                        help="Original data matrix, in CSV format.")
    parser.add_argument("-v", dest="validset", required=True,
                        help="Validation/classification set indicators, in CSV"
                        " format")

    parser.add_argument("set_id", choices=["L1", "L2", "L3", "C1", "C2", "C3",
                                        "CM", "t1", "t2", "s1", "s2", "s3",
                                        "s4", "s5"])

    parser.add_argument("features", nargs="+",
                        help="Features to be used to build the network model.")

    args = parser.parse_args()

    train = pd.read_csv(args.data_matrix)
    vs = pd.read_csv(args.validset)

    validset = vs[args.set_id]

    # remove all rows with deldate == NA
    train = train.drop(train.index[np.isnan(train['deldate'])])
    # reset index
    train.index = range(len(train))

    nm_feat = args.features[0]
    print(args.features)
    if len(args.features) > 1:
        nm_feat = "_".join(args.features)
        # convert tuples to strings, otherwise can't be transformed.
        train[nm_feat] = ['/'.join(map(str, x))
            for x in train[args.features].itertuples(index=False)]
        args.features = [nm_feat]

    # subsetting columns that we need
    cols_subset = ['cid'] + args.features

    # split into training set and learning set
    tr = train[~validset] 
    va = train[validset] 

    # leakage protection
    va.ix[:, 'return'] = np.nan
    
    print("Encoding features into indices...")
    encoders = {}
    # relabel iid, cid, mid
    for col in cols_subset:
        lblenc = preprocessing.LabelEncoder()
        # fitting using the union of all possible values
        # that can exist both in training and learning sets.  
        encoders[col] = lblenc.fit(train[col])

        tr.ix[:, col] = lblenc.transform(tr[col])
        va.ix[:, col] = lblenc.transform(va[col])

    nflvls = len(encoders[nm_feat].classes_)
    ncusts = len(encoders['cid'].classes_)

    # use COOrdinated sparse matrices to build these counters fast
    # kept/kept, kept/return, return/kept, return/return
    # .append((value, rowidx, colidx))

    count_matrices = [[], [], [], []]
    #count_matrices = [
    #    np.zeros((nflvls, nflvls), dtype=int), # kept/kept
    #    np.zeros((nflvls, nflvls), dtype=int), # kept/return
    #    np.zeros((nflvls, nflvls), dtype=int), # return/kept
    #    np.zeros((nflvls, nflvls), dtype=int), # return/return
    #]

    # sparse matrices, since customer purchase history will be 
    # VERY VERY sparse in reality
    #Cret = sparse.lil_matrix((ncusts, nflvls))
    #Ckep = sparse.lil_matrix((ncusts, nflvls))
    #Cret = np.zeros((ncusts, nflvls))
    #Ckep = np.zeros((ncusts, nflvls))

    # Again, use COOrdinated sparse matrix
    Cret = []
    Ckep = []

    # this will sort cid in the transformed order
    customers = tr.groupby('cid', sort=True)

    for cid, gidx in customers.groups.items():
        if cid % 1000 == 0:
            print(" :: {0}/{1}".format(cid, ncusts))

        sub_df = tr.ix[gidx, :]

        items = list(sub_df[[nm_feat, 'return']].itertuples(index=False))

        # if the exact same pair occurs multiple times for a given customer,
        # only count as once.
        all_pairs = set(permutations(
            sub_df[[nm_feat, 'return']].itertuples(index=False), 2))

        # keepers
        _keeps = [f for f, ret in items if ret == 0]
        if _keeps:
            findx, summand = zip(*tuple(Counter(_keeps).items()))

            # row: cid, col: findx
            Ckep.extend(zip(summand, repeat(cid), findx))

        # returns
        _rets = [f for f, ret in items if ret == 1]
        if _rets:
            findx, summand = zip(*tuple(Counter(_rets).items()))

            # row: cid, col: findx
            Cret.extend(zip(summand, repeat(cid), findx))

        for (f1, ret1), (f2, ret2) in all_pairs:
            matidx = ret1 * 2 + ret2

            count_matrices[matidx].append((1, f1, f2))

            #count_matrices[matidx][iid1, iid2] += 1
            #if matidx == 1 or matidx == 2:
            #    # symmetric keep/return - return/keep
            #    count_matrices[3-matidx][iid2, iid1] += 1
            #else:
            #    count_matrices[matidx][iid2, iid1] += 1


    print("Converting to sparse matrices...")
    # convert the list of matrix entries into COO matrix,
    # then CSR matrix.
    Ckep = elems_to_coo_matrix(
        Ckep, (ncusts, nflvls)).tocsr()
    Cret = elems_to_coo_matrix(
        Cret, (ncusts, nflvls)).tocsr()

    for i in range(4):
        count_matrices[i] = elems_to_coo_matrix(
                count_matrices[i], (nflvls, nflvls)).tocsc()

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

    llr_kept = [Mllr_kept[cid, f]
                for cid, f in va[cols_subset].itertuples(index=False)]
    llr_ret = [Mllr_ret[cid, f]
                for cid, f in va[cols_subset].itertuples(index=False)]

    llrs = pd.DataFrame({'nm.llr.kept': llr_kept, 'nm.llr.ret': llr_ret})
    llrs.to_csv("{0}_nm_llr_{1}.csv".format(args.set_id, nm_feat), index=False)

if __name__ == "__main__":
    main()
