'''
Script to generate test data files for checking load compatibility.

Instructions:

    1. Create a new conda environment, install pypi-timemachine, and start a server:

        conda create -n tm2021 python=3.9 -y 
        conda activate tm2021
        pip install pypi-timemachine
        pypi-timemachine 2021-01-01 # shows port being used

    2. In a separate terminal, install Sciris and run this script:

        conda activate tm2021
        pip install --index-url http://localhost:<PORT> sciris
        python make_archive.py 2021-01-01

Version: 2023-08-06
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
    as_archive = bool(sys.argv[2])

class MyClass:
    ''' Store common data types for compatibility checks'''
    
    def __init__(self):
        self.date = date
        self.strings = ['a', 'b', 'c', 'd', 'e']
        self.nparray = np.arange(5)
        self.datetime = dt.datetime.now(du.tz.tzutc())
        self.dataframe = pd.DataFrame(dict(labels=self.strings, vals=self.nparray))

    def sum(self):
        return self.nparray.sum()

myclass = MyClass()

if as_archive:
    sc.savearchive('archive.zip', myclass)
else:
    sc.saveobj(f'pickle_{date}.obj', myclass