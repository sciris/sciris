'''
Script to generate test data files for checking load compatibility.

Instructions:

    1. Create a new conda environment, install pypi-timemachine, and start a server.
    For example, for 2022:

        conda create -n tm python=3.9 -y 
        conda activate tm
        pip install pypi-timemachine
        pypi-timemachine 2022-01-01 # shows port being used

    2. In a separate terminal, install Sciris and run this script:

        conda activate tm
        pip install --index-url http://localhost:<PORT> sciris
        python make_archive.py 2022-01-01

This has been run with the following arguments:
    # For the archive
    python make_archive.py 2023-04-19 1
    
    # For the pickles
    python make_archive.py 2021-01-01 0
    python make_archive.py 2022-01-01 0

Version history:
    2023-04-19: Original version
    2023-08-06: Updated to save pickles and additional pandas data
'''

import sys
import datetime as dt
import dateutil as du
import numpy as np
import pandas as pd
import sciris as sc

# If using pypi-timemachine, set the corresponding date here
date = '2023-04-19'
as_archive = True

if len(sys.argv) > 1:
    date = sys.argv[1]
if len(sys.argv) > 2:
    as_archive = int(sys.argv[2])
    
    
def create_pandas_data():
    """ Create the pandas pickle data -- adapted from pandas/tests/io/generate_legacy_storage_files.py """
    data = sc.objdict(
        a = [0.0, 1.0, 2.0, 3.0, np.nan],
        b = [0, 1, 0, 1, 0],
        c = ["foo1", "foo2", "foo3", "foo4", "foo5"],
        d = pd.date_range("1/1/2009", periods=5),
        e = [0.0, 1, pd.Timestamp("20100101"), "foo", 2.0],
    )

    index = sc.objdict(
        ind_int   = pd.Index(np.arange(10)),
        ind_date  = pd.date_range("20130101", periods=10),
        ind_float = pd.Index(np.arange(10, dtype=np.float64)),
        ind_range = pd.RangeIndex(10),
    )

    frame = pd.DataFrame(data)
    
    out = sc.objdict(data=data, frame=frame, index=index)

    return out


class MyClass:
    ''' Store common data types for compatibility checks'''
    
    def __init__(self):
        self.date = date
        self.strings = ['a', 'b', 'c', 'd', 'e']
        self.nparray = np.arange(5)
        self.datetime = dt.datetime.now(du.tz.tzutc())
        self.pandas = create_pandas_data()

    def sum(self):
        return self.nparray.sum()

myclass = MyClass()

if as_archive:
    sc.savearchive('archive_{date}.zip', myclass)
else:
    sc.saveobj(f'pickle_{date}.obj', myclass.__dict__)