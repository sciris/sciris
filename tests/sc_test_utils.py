'''
Helper functions for test scripts
'''

import inspect
import datetime as dt
import dateutil as du
import numpy as np
import pandas as pd
import sciris as sc


def check_signatures(func1, func2, extras=None, missing=None, die=True):
    '''
    Check that two functions have the same call signature. Useful for validating
    that a class init and a function call are the same. "extras" are known arguments
    in func2 that are not in func1, and "missing" are known arguments in func1
    that are not in func2.
    '''
    extras = set(sc.mergelists(extras))
    missing = set(sc.mergelists(missing))
    sigs = sc.objdict()
    for k,func in zip(['f1', 'f2'], [func1, func2]):
        sigs[k] = set(list(inspect.signature(func).parameters.keys()))
    if extras:
        sigs.f2 = sigs.f2 - extras
    if missing:
        sigs.f1 = sigs.f1 - missing
    
    eq = sigs.f1 == sigs.f2
    if die:
        assert eq, 'Signatures do not match'
        
    return  eq


def create_complex_data(alt=False, nan=True, mixed=True, pandas=True):
    ''' Create complex pickle data -- adapted from pandas/tests/io/generate_legacy_storage_files.py '''
    data = sc.objdict(
        a = [0.0, 1.0, 2.0, 3.0 + alt, [-1, np.nan][nan]],
        b = [0, 1, 0, 1, 0 + alt],
        c = ["foo1", "foo2", "foo3", "foo4", "foo5" + alt*'alt'],
    )
    if pandas:
        data.d = pd.date_range("1/1/2009", periods=5)
    if mixed:
        data.e = [0.0, 1, pd.Timestamp("20100101"), "foo", 2.0+alt],
    if alt:
        data.f = ['Using', 'alternate', 'data', 'creation', 'method']

    index = sc.objdict(
        ind_int   = pd.Index(np.arange(10+alt)),
        ind_date  = pd.date_range("20130101", periods=10+alt),
        ind_float = pd.Index(np.arange(10+alt, dtype=np.float64)),
        ind_range = pd.RangeIndex(10+alt),
    )

    frame = pd.DataFrame(data)
    
    if pandas:
        out = sc.objdict(data=data, frame=frame, index=index)
    else:
        out = data

    return out


class MyClass(sc.prettyobj):
    ''' Store common data types for compatibility checks'''
    
    def __init__(self, date='2023-08-11', alt=False, nan=True, mixed=True, pandas=True):
        self.date = date
        self.strings = ['a', 'b', 'c', 'd', 'e']
        self.nparray = np.arange(5)
        self.datetime = dt.datetime(2022, 4, 4, tzinfo=du.tz.tzutc())
        self.pandas = create_complex_data(alt=alt, nan=nan, mixed=mixed, pandas=pandas)

    def sum(self):
        return self.nparray.sum()