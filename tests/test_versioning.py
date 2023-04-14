'''
Test Sciris miscellaneous utility/helper functions.
'''

import sciris as sc
import numpy as np
import pytest

filedir = sc.path('files')


def test_functions():
    sc.heading('Test versioning functions')
    o = sc.objdict()
    
    print('Testing freeze')
    o.freeze = sc.freeze()
    assert 'numpy' in o.freeze
    
    print('Testing require; will print warning messages')
    sc.require('numpy')
    sc.require(numpy='')
    sc.require(reqs={'numpy':'1.19.1', 'matplotlib':'2.2.2'})
    sc.require('numpy>=1.19.1', 'matplotlib==2.2.2', die=False)
    data, _ = sc.require(numpy='1.19.1', matplotlib='==44.2.2', die=False, detailed=True)
    with pytest.raises(ModuleNotFoundError): sc.require('matplotlib==99.23')
    with pytest.raises(ModuleNotFoundError): sc.require('not_a_valid_module')
    print('↑↑↑ Should be warning messages above')
    
    print('Testing gitinfo')
    o.gitinfo = sc.gitinfo() # Try getting gitinfo; will likely fail though
    assert 'branch' in o.gitinfo
    
    print('Testing compareversions')
    assert sc.compareversions(np, '>1.0')
    v1 = '1.2.3'
    assert sc.compareversions(v1, '=1.2.3')
    assert sc.compareversions(v1, '==1.2.3')
    assert sc.compareversions(v1, '>=1.2.3')
    assert sc.compareversions(v1, '<=1.2.3')
    assert sc.compareversions(v1, '>1.2.2')
    assert sc.compareversions(v1, '>=1.2.2')
    assert sc.compareversions(v1, '<1.2.4')
    assert sc.compareversions(v1, '<=1.2.4')
    assert sc.compareversions(v1, '!1.2.9')
    assert sc.compareversions(v1, '~=1.2.9')
    assert sc.compareversions(v1, '!=1.2.9')
    with pytest.raises(ValueError):
        assert sc.compareversions(v1, '~1.2.9')
    
    print('Testing getcaller')
    o.caller = sc.getcaller(frame=1) # Frame = 1 is the current file
    assert 'test_versioning.py' in o.caller
    
    return o


def test_load_save():
    sc.heading('Test load/save with versioning')
    o = sc.objdict()
    
    f = sc.objdict()
    f.md = 'md.json'
    f.obj = 'md_obj.zip'
    f.wmd = filedir / 'withmetadata.zip'
    
    print('Testing savewithmethadata')
    obj = sc.prettyobj(label='foo', a=np.random.rand(5), b='label')
    sc.savewithmetadata(f.obj, obj)
    
    print('Testing loadmetadata')
    o.md = sc.metadata(outfile=f.md)
    assert sc.compareversions(o.md.versions.sciris, '>=2.2.0')
    
    md2 = sc.loadmetadata(f.md)
    md3 = sc.loadmetadata(f.obj)
    assert o.md.system.platform == md2.system.platform == md3.system.platform
    
    print('Testing loadwithmetadata')
    o.obj = sc.loadwithmetadata(f.wmd)
    data = sc.loadwithmetadata(f.wmd, loadmetadata=True) # To test a different conditional
    assert o.obj.sum() == 10
    assert data['obj'].sum() == 10
    
    for file in [f.md, f.obj]:
        sc.rmpath(file)
    
    return o


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    o = test_functions()
    test_load_save()
    

    sc.toc()
    print('Done.')