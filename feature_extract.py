#!/usr/bin/env python
# -*- coding: utf-8

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

def create_logger():
    logger = logging.getLogger('feat_extract')
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger

def parse_options():
    parser = ArgumentParser(
        description="DMC2014 feature matrix extraction script.")

    parser.add_argument("-o", dest="output_file", default="out.csv",
                        help="File to which results will be dumped.")
    parser.add_argument("--feat-matrix-only", default=False, 
                        action="store_true", 
                        help="Output only the feature matrix "\
                        "(sans original columns)")
    parser.add_argument("--c1c2", type=float, default=[1.0, 1.0], nargs=2,
                        metavar='NUM',
                        help="Constants c1 c2 used in computing LLR")
    parser.add_argument("--skip-cid-features", default=False,
                        action="store_true",
                        help="Do not generate all customer related features.")
    parser.add_argument("--skip-intervals", default=False,
                        action="store_true", 
                        help="Do not compute purchase interval features.")
    parser.add_argument("--global-features", 
                        default="state,state:mid,state:iid,mid,iid,size,"\
                        "color,iid:size,iid:color,mid:size,mid:color,"\
                        "zsize,ztype,mid:zsize,iid:zsize,cid,cid:zsize,"\
                        "cid:ztype,cid:zsize:ztype",
                        help="Features to be extracted from the entire " \
                        "dataset. Separate different features by comma.")
    parser.add_argument("--batch-features", 
                        default="iid,size,color,mid,zsize,ztype",
                        help="Features extracted at per batch level.")
    parser.add_argument("--within-features",
                        default="iid|size,iid|color,size|iid,size|mid," \
                        "color|iid,color|mid",
                        help="Features to be extracted within substructures "\
                        "within batches. Use 'within_feat|feat' to specify "\
                        "nested structure.")
    parser.add_argument("data_matrix", 
                        help="Path to the (cleaned) input data matrix")
    parser.add_argument("class_ind",
                        help="Learning/validation set indicators.")

    args = parser.parse_args()

    # process features
    if not args.global_features:
        args.global_features = []
    else :
        args.global_features = [
            feat if feat.find(":") == -1 else feat.split(":")
            for feat in args.global_features.split(',')]

    if not args.batch_features:
        args.batch_features = []
    else:
        args.batch_features = [
            feat if feat.find(":") == -1 else feat.split(":")
            for feat in args.batch_features.split(',')]

    if not args.within_features:
        args.within_features = {}
    else:
        _within_feats = defaultdict(list)
        for wifeat, feat in [feat.split('|')
                             for feat in args.within_features.split(',')]:
            _within_feats[wifeat].append(feat)

        args.within_features = _within_feats
    
    return args

def main():
    args = parse_options()
    logger = create_logger()

    c1, c2 = args.c1c2

    # 1) data loading
    assert os.path.exists(args.data_matrix)
    assert os.path.exists(args.class_ind)

    logger.info("Loading datasets...")
    cind = pd.read_csv(args.class_ind, sep=',')
    train = pd.read_csv(args.data_matrix, sep=',')

    train = pd.concat([train, cind], axis=1)

    logger.info("Data cleaning...")
    # remove all data with deldate == NA

    train = train.drop(train.index[np.isnan(train['deldate'])])
    #train = train[~np.isnan(train['deldate'])]

    # make new index consecutive
    #train.reset_index()
    train.index = range(len(train))

    logger.info("Splitting purchases into batches...")
    # 2) split rows/orders into batches by identifying consecutive customer IDs
    # (cid).
    cid_edge = np.concatenate(
        [[1], np.array(train['cid'][1:]) - np.array(train['cid'][:-1])])
    cid_edge[cid_edge != 0] = 1
    train['batch'] = np.cumsum(cid_edge)

    # leaking protection
    ret_valid = list(train.ix[train['valid'] == 1, 'return'])
    train.ix[train['valid'] == 1, 'return'] = np.NAN

    # rows on which features are created.
    learn_idx = train['valid'] == 0
    tr_cr = train[learn_idx]
    tr_va = train[~learn_idx]

    # 3) per batch summary statistics and ...
    # 4) features within substructures of batch
    # NOTE: these features are independent of training/validation, thus 
    # can be aggregated first.

    batches = train.groupby('batch', sort=False)
    bat_feats = None
    if args.batch_features or args.within_features:
        logger.info("Generating per batch and within batch features...")
        bat_feats = batches.apply(
            batch_summarize, 
            u_feats=args.batch_features,
            wi_feats=args.within_features)
    else:
        logger.info(":: Skipping per batch and within batch features...")

    logger.info("Generating global features...")
    glob_feat_list = []
    # 5) whole dataset aggregation
    for feat in args.global_features:
        logger.info("  >> Columns: {0}".format(feature_name(feat)))

        # counts in the learning set
        learn_counts = tr_cr.groupby(feat, sort=False).size()
        learn_returns = tr_cr[tr_cr['return'] == 1].groupby(
            feat, sort=False)['return'].agg(np.sum)

        x = batches.apply(
            extract_global_features, 
            feat=feat, counts=learn_counts, returns=learn_returns,
            c1=c1, c2=c2)

        x.columns = ["all.count.{0}".format(feature_name(feat)),

                     "all.llr.{0}".format(feature_name(feat))]

        glob_feat_list.append(x)

    # 6) average batch size for each cid
    cid_feats = None
    if not args.skip_cid_features:
        # compute historical customer batch features for the learning set 
        logger.info("Summarizing batch related features for customers...")
        cid_ln_feats = tr_cr.groupby('cid', sort=False).apply(
            cid_batch_summarize)
        logger.info("Indexing customer/batch features...")
        cid_agg = pd.concat([pd.DataFrame(tr_cr['cid']), cid_ln_feats], 
                            axis=1).groupby('cid').apply(
                                lambda x: x.iloc[0, :])
        # convert cid_agg into a dictionary for lookups
        cid_bat_dict = dict([(r[0], r) 
                             for r in cid_agg.itertuples(index=False)])

        # releasing memory
        del cid_agg

        # now for validation set
        logger.info("Batch/customer related features for validation set...")
        cid_va_feats = tr_va['cid'].apply(
            lambda cid: pd.Series(cid_bat_dict.get(
                cid, # if it is a new customer, we don't have hist data
                (cid, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        ))

        # release memory
        del cid_bat_dict

        # combine the two parts
        cid_feats = pd.DataFrame(
            np.zeros((train.shape[0], cid_ln_feats.shape[1])),
            columns = cid_ln_feats.columns
        )

        cid_feats[learn_idx] = cid_ln_feats
        cid_feats[~learn_idx] = cid_va_feats.iloc[:, 1:]

        # deleting intermediate data frames
        del cid_ln_feats
        del cid_va_feats
    else:
        logger.info(":: Skipping customer related batch features...")

    # 7) interval between purchases
    if not args.skip_intervals:
        logger.info("Computing intervals between customer purchases...")
        batch_interval.first_invoked = True
        hist_pur_dict = {'iid': {}, 'mid': {}, 'size': {}, 'color': {}}
        intvl_feats = batches.apply(batch_interval, hist_dict=hist_pur_dict)

        del hist_pur_dict
    else:
        logger.info(":: Skipping purchase interval features...")

    # 8) other customer historical features
    #    number of times an item had been purchased by customer
    #    number of batches an item had been purchased by customer
    #    number of times returned
    #    had been purchased?
    #    had been returned?
    hist_feat_list = []
    #for feat in HISTORICAL_FEATURES:
    #    hist_row_cnts = defaultdict(int)
    #    hist_batch_cnts = defaultdict(int)
    #    hist_batch_returns = defaultdict(int)

    #    cols = ['cid']
    #    if type(feat) in (tuple, list):
    #        cols.extend(feat)
    #    elif type(feat) is str:
    #        cols.append(feat)

    #    #hist_feats = train[cols].apply(
    #    #    sequential_counter, counter=hist_row_cnts, axis=1)

    #    batch_counter.skip_test_drive = True
    #    batch_hist_feats = batches.apply(
    #        batch_counter, order_cnts=hist_batch_cnts, 
    #        rt_cnts=hist_batch_returns, columns=cols)

    #    batch_hist_feats.columns = [
    #        'bhist.retcnt.{0}'.format(feature_name(feat)),
    #        'bhist.purcnt.{0}'.format(feature_name(feat)),
    #        'bhist.llr.{0}'.format(feature_name(feat))]

    #    hist_feat_list.append(batch_hist_feats)

    # restore the return values for validation set
    train.ix[train['valid'] == 1, 'return'] = ret_valid

    # concatenate features and dump
    blob_to_dump = [train] if not arg.feat_matrix_only else []

    if args.batch_features:
        blob_to_dump.append(bat_feats)

    if not args.skip_cid_features:
        blob_to_dump.append(cid_feats)

    if not args.skip_intervals:
        blob_to_dump.append(intvl_feats)

    blob_to_dump += glob_feat_list
    blob_to_dump += hist_feat_list

    feat_mat = pd.concat(blob_to_dump, axis=1)
    feat_mat.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
