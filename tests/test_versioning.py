'''
Test Sciris miscellaneous utility/helper functions.
'''

import sciris as sc
import numpy as np
import pytest

filedir = sc.thispath() / 'files'


def test_functions():
    sc.heading('Test versioning functions')
    o = sc.objdict()
    
    print('Testing freeze')
    o.freeze = sc.freeze()
    assert 'numpy' in o.freeze, 'NumPy not found, but should be'
    v1 = o.freeze['numpy']
    v2 = np.__version__
    assert v1 == v2, f'Versions do not match ({v1} != {v2})'
    
    print('Testing require')
    sc.require('numpy')
    sc.require(numpy='')
    sc.require(reqs={'numpy':'1.19.1', 'matplotlib':'2.2.2'})
    with pytest.warns(UserWarning):           sc.require('numpy>=1.19.1', 'matplotlib==2.2.2', die=False)
    with pytest.warns(UserWarning): data, _ = sc.require(numpy='1.19.1', matplotlib='==44.2.2', die=False, detailed=True)
    with pytest.raises(ModuleNotFoundError): sc.require('matplotlib==99.23')
    with pytest.raises(ModuleNotFoundError): sc.require('not_a_valid_module')
    with sc.capture() as txt:
        sc.require('numpy>=1.19.1', 'matplotlib==2.2.2', die=False, warn=False, message='<MISSING> is gone!')
    assert 'gone' in txt
    
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


def test_metadata():
    sc.heading('Test metadata')
    o = sc.objdict()
    
    f = sc.objdict()
    f.md = 'md.json'
    f.obj = 'md_obj.zip'
    f.wmd = filedir / 'archive.zip'
    
    print('Testing savewithmethadata')
    obj = sc.prettyobj(label='foo', a=np.random.rand(5), b='label')
    sc.savearchive(f.obj, obj)
    
    print('Testing loadmetadata')
    o.md = sc.metadata(outfile=f.md)
    assert sc.compareversions(o.md.versions.sciris, '>=2.2.0')
    
    md2 = sc.loadmetadata(f.md)
    md3 = sc.loadmetadata(f.obj)
    assert o.md.system.platform == md2.system.platform == md3.system.platform
    
    for file in [f.md, f.obj]:
        sc.rmpath(file)
    
    return o


def test_regressions():
    sc.heading('Test load/save with versioning')
    o = sc.objdict()
    md = sc.metadata() # Get current metadata information, since not all tests are expected to pass for all versions
    
    print('Testing loadarchive')
    archives = sc.getfilelist(folder=filedir, pattern='archive*.zip')
    for i,fn in enumerate(archives):
        print(f'  Testing {fn}...')
        key = f'archive{i}'
        o[key] = sc.loadarchive(fn)
        data = sc.loadarchive(fn, loadmetadata=True) # To test a different conditional
        if sc.compareversions(md.versions.python, '<3.11'): # Due to new opcodes, old pickled methods can't be loaded
            assert o[key].sum() == 10
            assert data['obj'].sum() == 10
    
    print('Testing loadobj')
    pickles = sc.getfilelist(folder=filedir, pattern='pickle*.obj')
    for i,fn in enumerate(pickles):
        print(f'  Testing {fn} ({i})...')
        if i==0 or sc.compareversions(md.versions.pandas, '<2.1.0'): # Pandas broke backwards compatibility here
            key = f'pickle{i}'
            o[key] = sc.load(fn)
            assert o[key]['nparray'].sum() == np.arange(5).sum()
            assert o[key]['pandas'].frame.a.values[:4].sum() == np.arange(4).sum()
        else:
            print(f'    Warning, skipping {fn} since known failure with pandas versions >=2.1.0')
    
    return o


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    o1 = test_functions()
    o2 = test_metadata()
    o3 = test_regressions()
    

    sc.toc()
    print('Done.')