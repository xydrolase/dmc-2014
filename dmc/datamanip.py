#!/usr/bin/env python

from collections import Counter
from math import log

import numpy as np
import pandas as pd

# TODO: to write a broadcast decorator 

def feature_name(feats, sep='_'):
    return feats if type(feats) is str else sep.join(feats)

def batch_counter(df, order_cnts, rt_cnts, columns):
    groups = df.groupby(columns)
    feat_return = groups['return'].agg(sum)
    feat_order = groups.size()

    ordered_times = pd.DataFrame([(rt_cnts[x], order_cnts[x]) 
                     for x in df[columns].itertuples()], index=df.index)

    for feat, crt, cord in zip(groups.groups.keys(), feat_return, feat_order):
        order_cnts[feat] += cord
        rt_cnts[feat] += crt

    return ordered_times

def sequential_counter(row, counter):
    ntuple = tuple(row)
    n = counter[ntuple]
    counter[ntuple] += 1

    return (n > 0, n)

def trans_unique_count(x):
    return pd.Series(np.repeat(len(set(x)), len(x)), index=x.index)

def trans_group_count(x):
    # IMPORTANT: Preserving original index
    return pd.Series(np.repeat(len(x), len(x)), index=x.index)

def trans_llr(x, c1, c2):
    R = float(np.sum(x['return']))
    NR = len(x) - R
    
    return pd.Series(np.repeat(log((R + c1) / (NR + c2)), len(x)), 
                     index=x.index)

def batch_summarize(bat_df, u_feats, wi_feats):
    # a) per batch features: U.iid, U.size ...
    u_names = map(lambda n: 'bat.uniq.{0}'.format(n),
        [x if type(x) is str else '_'.join(x) for x in u_feats])

    # aggregation of all per batch features
    u_values = [len(set(bat_df[feat])) for feat in u_feats]

    data = zip(u_names, u_values)

    # b) within feature features: WIiid.size ...
    for wi_feat, agg_feats in wi_feats.items():
        _group = bat_df.groupby(wi_feat, sort=False)

        for _feat in agg_feats:
            wi_name = "bwi_{0}.uniq.{1}".format(wi_feat, feature_name(_feat))
            wi_data = _group[_feat].apply(trans_unique_count)

            data.append((wi_name, wi_data))

    # IMPORATANT: mirroring index!
    df = pd.DataFrame(dict(data), index=bat_df.index)

    # c) other simple features 
    df['bat.n'] = len(bat_df)
    df['bat.sum.return'] = sum(bat_df['return'])

    return df
