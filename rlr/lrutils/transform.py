#!/usr/bin/env python

from itertools import combinations
from sklearn import preprocessing

import pandas as pd

class FactorDataFrame(pd.core.frame.DataFrame):
    """A derived pandas DataFrame type that caches the LabelEncoder
    from sklearn.preprocessing module, allowing convenient inverse
    transformation from encoded factors to the original levels."""

    def __init__(self, dframe):
        if type(dframe) is not pd.core.frame.DataFrame:
            raise TypeError

        pd.core.frame.DataFrame.__init__(
            self, dframe, copy=True)

        self.encoders = {}
        for col in self.columns:
            le = preprocessing.LabelEncoder()
            self.encoders[col] = le.fit(self.ix[:, col])
            self.ix[:, col] = le.transform(self.ix[:, col])

    def span_interactions(cols, nway=2, sep="_", hash=hash):
        """Span N-way interactions for selected columns. New columns
        will be automatically named by concatenating main factor names 
        with the given separator."""

        inter_cols = combinations(self.columns[col], nway)
        for icol in inter_cols:
            icol_name = "_".join(icol)
            _levels = [
                hash(row) 
                for row in self[icol].itertuples(index=False)
            ]
            le = preprocessing.LabelEncoder()
            self.encoders[icol_name] = le.fit(_levels)

            self.ix[:, icol] = le.transform(_levels)

    def inverse_transform(self):
        for col in self.columns:
            self.ix[:, col] = self.encoders[col].inverse_transform(
                self.ix[:, col])

def select(dframe, sel):
    """Select columns using selector sel."""

    _scolumns = set(dframe.columns)
    _cols = []
    for cs in sel:
        if type(cs) in (unicode, str):
            if cs in _scolumns:
                _cols.append(cs)

        elif isinstance(cs, re._pattern_type):
            _cols.extend([col for col in dframe.columns if cs.search(cs)])
    
    return dframe[_cols]

def as_factor(dframe):
    """Return a transformed FactorDataFrame."""
    if type(dframe) is FactorDataFrame:
        return dframe

    return FactorDataFrame(dframe)

