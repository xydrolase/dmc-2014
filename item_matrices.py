#!/usr/bin/env python

import sys

import pandas as pd
import numpy as np

from itertools import combinations

def main():
    train = pd.read_csv(sys.argv[1], sep=',')
    cids = train.groupby('cid')

    nitems = sorted(set(train['iid']))
    iidmap = dict(zip(nitems, range(len(nitems))))

    count_matrices = [
        np.zeros((3007, 3007), dtype=int),
        np.zeros((3007, 3007), dtype=int)]

    for cid, gidx in cids.groups.iteritems():
        sub_df = train.iloc[gidx, :]

        all_pairs = set(combinations(
            sub_df[['iid', 'return']].itertuples(index=False), 2))

        for (iid1, ret1), (iid2, ret2) in all_pairs:
            if ret1 == 0:
                count_matrices[ret2][iidmap[iid1], iidmap[iid2]] += 1

    np.savetxt("keptkept.csv", count_matrices[0], fmt='%d', delimiter=",")
    np.savetxt("keptret.csv", count_matrices[1], fmt='%d', delimiter=",")

    #pcnts = cids.apply(purchase_count, iidmap=iidmap)

if __name__ == "__main__":
    main()
