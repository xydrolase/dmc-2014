#!/usr/bin/env python

from itertools import chain
from collections import Counter
from math import log

import numpy as np
import pandas as pd

def index_mapping(func):
    """A decorator that wraps an iterable return from a function
    into a DataFrame object, while mirroring the original index."""

    def cast_to_dataframe(*args, **kwargs):
        _iterable = func(*args, **kwargs)
        # index mirroring
        return pd.DataFrame(_iterable, index=args[0].index)

    return cast_to_dataframe

def feature_name(feats, sep='_'):
    return feats if type(feats) is str else sep.join(feats)

def batch_counter(df, order_cnts, rt_cnts, columns, c1, c2):
    groups = df.groupby(columns, sort=False)
    feat_return = groups['return'].agg(sum)
    feat_order = groups.size()

    ordered_times = np.array([(rt_cnts[x], order_cnts[x]) 
                     for x in df[columns].itertuples(index=False)])
    llr = np.log((ordered_times[:, 0] + c1) / 
                 (ordered_times[:, 1] - ordered_times[:, 0] + c2))

    # pandas will apply this function onto the first group to test whether the 
    # returned value is mutated, which tampers with the counts. 
    # hence we use a "static" variable to skip counting when this function is
    # called for the first time.
    if not batch_counter.skip_test_drive:
        for feat, crt, cord in zip(
                groups.groups.keys(), feat_return, feat_order):
            order_cnts[feat] += cord
            rt_cnts[feat] += crt

    batch_counter.skip_test_drive = False

    return pd.DataFrame(np.hstack(ordered_times, llr), index=df.index)

batch_counter.skip_test_drive = True

def sequential_counter(row, counter):
    ntuple = tuple(row)
    n = counter[ntuple]
    counter[ntuple] += 1

    return (n > 0, n)

@index_mapping
def trans_unique_count(x):
    return np.repeat(len(set(x)), len(x))

@index_mapping
def trans_group_count(x):
    # IMPORTANT: Preserving original index
    return np.repeat(len(x), len(x))

@index_mapping
def trans_llr(x, c1, c2):
    R = float(np.sum(x['return']))
    NR = len(x) - R
    
    return np.repeat(log((R + c1) / (NR + c2)), len(x))

def extract_global_features(bat, feat, counts, returns, c1, c2):
    gbf = bat.groupby(feat, sort=False)

    mul = 1
    if np.all(bat['valid']):
        # for validation set, use only the counts/LLRs from the learning set.
        mul = 0
    
    # counts
    col_cnts = np.concatenate([
        np.repeat(counts.get(k, 0) - v * mul, v)
        for k, v in bcnts.items()])
    col_rets = np.concatenate([
        np.repeat(returns.get(k, 0) - v * mul, bcnts[k])
        for k, v in brets.items()])

    # LLRs
    col_llrs = np.log((col_rets + c1) / (col_cnts - col_rets + c2))

    return pd.DataFrame({'x': col_cnts, 'y': col_llrs}, index=bat.index)

def cid_batch_summarize(df):
    """Summarizing batch related features for each customer."""

    batches = df.groupby('batch', sort=False)
    bsize = batches.size()
    breturn = batches['return'].agg(sum)
    bkept = bsize - breturn

    avg_bsize = np.mean(bsize)
    sum_kept_rate = np.sum(bkept / bsize)
    sum_ret_rate = np.sum(breturn / bsize)

    n = len(df)
    nbatch = len(bsize)

    # average batch size / avg & sum kept rate / 
    # / avg & sum returned rate / rate ratio
    ret_df = pd.DataFrame({
        'cid.avg.bsize': np.repeat(avg_bsize, n),
        'cid.sum.krate': np.repeat(sum_kept_rate, n),
        'cid.avg.krate': np.repeat(sum_kept_rate / nbatch, n),
        'cid.sum.rrate': np.repeat(sum_ret_rate, n),
        'cid.avg.rrate': np.repeat(sum_ret_rate / nbatch, n), 
        'cid.llr.rk': np.repeat(
            log((sum_ret_rate + 0.5) / (sum_kept_rate + 0.5)), n),
    }, 
        index=df.index)

    return ret_df

def batch_summarize(bat_df, u_feats, wi_feats):
    # a) per batch features: U.iid, U.size ...
    u_names = map(lambda n: 'bat.uniq.{0}'.format(n),
        [x if type(x) is str else '_'.join(x) for x in u_feats])

    # aggregation of all per batch features
    u_values = [len(set(bat_df[feat])) for feat in u_feats]

    data = zip(u_names, u_values)

    # b) within feature features: WIiid.size ...
    for wi_feat, agg_feats in wi_feats.items():
        break
        _group = bat_df.groupby(wi_feat, sort=False)

        for _feat in agg_feats:
            wi_name = "bwi_{0}.uniq.{1}".format(wi_feat, feature_name(_feat))
            wi_data = _group[_feat].apply(trans_unique_count)

            data.append((wi_name, wi_data))

    # IMPORATANT: mirroring index!
    df = pd.DataFrame(dict(data), index=bat_df.index)

    # c) other simple features 
    df['priceorder'] = np.argsort(bat_df['price'])
    df['bat.n'] = len(bat_df)

    return df
