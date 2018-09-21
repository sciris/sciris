"""
test_datastore.py -- test module for sc_datastore.py
"""

# Imports
import sciris as sc
import scirisweb as sw

torun = [
'datastore',
]

T = sc.tic()

if 'datastore' in torun:
    key_in   = 'testkey'
    ds       = sw.DataStore(verbose=True)
    data_in  = sc.odict({'foo':[1,2,3], 'bar':[4,5,6]})
    key_out  = ds.saveblob(obj=data_in, key=key_in)
    data_out = ds.loadblob(key_in)
    success  = ds.delete(key_in)
    failure  = ds.delete(key_in+'foo')
    assert key_in  == key_out
    assert data_in == data_out
    assert success == 1
    assert failure == 0
    


sc.toc(T)
print('Done running %s tests.' % len(torun))