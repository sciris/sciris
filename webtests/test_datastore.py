"""
test_datastore.py -- test module for sc_datastore.py
"""

# Imports
import sciris as sc
import scirisweb as sw

torun = [
'datastore',
]

if 'datastore' in torun:
    testkey = 'testkey'
    ds = sw.DataStore(verbose=True)
    testdata = sc.odict({'foo':[1,2,3], 'bar':[4,5,6]})
    ds.saveblob(data=testdata, key=testkey)
    dataout = ds.loadblob(testkey)
    ds.delete(testkey)
    assert testdata == dataout