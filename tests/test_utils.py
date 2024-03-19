'''
Test Sciris utility/helper functions.
'''

import numpy as np
import sciris as sc
import pytest


#%% Adaptations from other libraries

url1 = 'wikipedia.org'
url2 = 'http://google.com/'

def test_adaptations():
    sc.heading('Test function adaptations')
    o = sc.objdict()

    print('\nTesting sha')
    o.sha = sc.sha({'a':np.random.rand(5)})

    print('\nTesting cp and dcp')
    o.sha2 = sc.dcp(o.sha.hexdigest())
    with pytest.raises(ValueError):
        o.sha3 = sc.cp(o.sha)
    with pytest.raises(ValueError):
        o.sha3 = sc.dcp(o.sha)

    print('Testing asciify')
    o.ascii = sc.asciify('föö→λ ∈ ℝ')
    assert o.ascii == 'foo  R'

    print('\nTesting traceback')
    o.traceback = sc.traceback()
    
    print('\nTesting platforms')
    sc.getplatform()
    assert sc.iswindows() + sc.ismac() + sc.islinux() == 1
    assert not sc.isjupyter() # Assume this won't be called in Jupyter!
    shell = sc.isjupyter(detailed=True)
    print(shell)
    assert isinstance(shell, str)

    return o


def test_download():
    print('\nTesting download')
    o = sc.objdict()
    o.download = sc.download(url1, url2, save=False, parallel=True) # Set parallel=False to debug

    print('\nTesting htmlify')
    o.html = sc.htmlify('foo&\nbar')
    assert o.html == b'foo&amp;<br>bar'
    o.nothtml = sc.htmlify(o.download[0], reverse=True)
    return o


def test_download_save(): # Split up to take advantage of parallelization
    print('\nTesting download and saving')
    fn = 'temp.html'
    sc.download({fn:url1})
    sc.rmpath(fn)
    return fn


def test_uuid():
    sc.heading('Test UID generation')
    import uuid

    # Create them
    u = sc.objdict()
    u.u0 = uuid.uuid4()
    u.u1 = sc.uuid()
    u.u2 = sc.uuid()
    u.u3 = sc.uuid(length=4)
    u.u4 = sc.uuid(which='ascii', length=16)
    u.u5 = sc.uuid(n=3)
    u.u6 = sc.uuid(which='hex', length=20)
    u.u7 = sc.uuid(which='numeric', length=10, n=5)
    u.u8 = sc.uuid(sc.uuid())
    u.u9 = sc.uuid(238)

    # Tests
    assert u.u1 != u.u2
    assert isinstance(u.u1, type(u.u0))
    assert isinstance(u.u3, str)
    with pytest.raises(ValueError):
        sc.uuid(length=400) # UUID is only 16 characters long
    with pytest.raises(ValueError):
        sc.uuid(which='numeric', length=2, n=10) # Not enough unique choices

    print('NOTE: This is supposed to print warnings and then raise a (caught) exception\n')
    with pytest.raises(ValueError):
        sc.uuid(which='numeric', length=2, n=99, safety=1, verbose=True) # Not enough unique choices

    # Print results
    print('UIDs:')
    for key,val in u.items():
        print(f'{key}: {val}')


    print('\nTesting fast_uuid')
    u.uuids = sc.fast_uuid(n=100) # Generate 100 UUIDs

    print('\nTesting uuid')
    u.uuid = sc.uuid()

    return u


def test_traceback():
    sc.heading('Test printing traceback text')

    dct = {'a':3}
    try:
        dct['b'] # This will cause a KeyError
    except:
        text = sc.traceback()

    print('NOTE: This is an example traceback, not an actual error!\n')
    print(f'Example traceback text:\n{text}')

    return text


def test_tryexcept():    
    sc.heading('Testing tryexcept')
    
    print('NOTE: This will print some exception text; this is expected\n')
    
    values = [0,1]
    with sc.tryexcept(): # Equivalent to contextlib.suppress(Exception)
        values[2]
        
    # Raise only certain errors
    with pytest.raises(IndexError):
        with sc.tryexcept(die=IndexError): # Catch everything except IndexError
            values[2]

    # Catch (do not raise) only certain errors
    with sc.tryexcept(catch=IndexError): # Raise everything except IndexError
        values[2]
        
    # Storing the history of multiple exceptions
    te = None
    repeats = 5
    for i in range(repeats):
        with sc.tryexcept(history=te) as te:
            values[i]
    assert len(te.exceptions) == repeats - len(values)
    assert te.died
    te.disp()
    te.traceback()
    
    return te



#%% Type functions

def test_tolist():
    sc.heading('test_promotetolist()')
    ex0 = 1
    ex1 = 'a'
    ex2 = {'a', 'b'}
    ex3 = np.array([0,1,2])
    ex4 = [1,2,3]
    res0 = sc.promotetolist(ex0, int)
    res1 = sc.promotetolist(ex1)
    res2a = sc.promotetolist(ex2)
    res2b = sc.promotetolist(ex2, objtype='str')
    res3a = sc.promotetolist(ex3)
    res3b = sc.promotetolist(ex3, objtype='number')
    res4a = sc.tolist('foo')
    res4b = sc.tolist('foo', coerce=str)
    res4c = sc.tolist(('foo', 'bar'), coerce=tuple)
    res5 = sc.tolist(range(3))
    with pytest.raises(TypeError):
        sc.promotetolist(ex0, str)
    with pytest.raises(TypeError):
        sc.promotetolist(ex1, int)
    with pytest.raises(TypeError):
        sc.promotetolist(ex3, objtype='str')
    with pytest.raises(TypeError):
        sc.promotetolist(ex4, objtype='str')
    assert res0 == [1]
    assert res1 == ['a']
    assert res2a == [{'a', 'b'}]
    assert sorted(res2b) == ['a', 'b'] # Sets randomize the order...
    assert repr(res3a) == repr([np.array([0,1,2])]) # Direct quality comparison fails due to the array
    assert res3b == [0,1,2]
    assert len(res4a) == 1
    assert len(res4b) == 3
    assert len(res4c) == 2
    assert res5[2] == 2
    print(res1)
    print(res2a)
    print(res2b)
    print(res3a)
    print(res3b)
    print(res4a)
    print(res4b)
    print(res5)

    # Check that type checking works
    sc.tolist(ex2, objtype=str)
    
    return res3b


def test_transpose_swap():
    print('\nTesting sc.transposelist() and sc.swapdict()')
    o = sc.odict(a=1, b=4, c=9, d=16)
    itemlist = o.enumitems()
    inds, keys, vals = sc.transposelist(itemlist)
    assert keys[2] == 'c'
    assert inds[3] == 3
    assert vals[1] == 4
    
    listoflists = [
        ['a', 1, 3],
        ['b', 4, 5],
        ['c', 7, 8, 9, 10]
    ]
    trans = sc.transposelist(listoflists, fix_uneven=True)
    assert all([len(l) == 5 for l in sc.transposelist(trans)])
    
    d1 = {'a':'foo', 'b':'bar'} 
    d2 = sc.swapdict(d1)
    assert d2 == {'foo':'a', 'bar':'b'} 
    with pytest.raises(TypeError):
        sc.swapdict(1) # Not a dict
    with pytest.raises(TypeError):
        listdict = dict(a=[1,2,3])
        sc.swapdict(listdict) # Unhashable type
    
    return d1, d2


def test_merge():
    sc.heading('Testing merge functions')

    print('\nTesting mergelists')
    assert sc.mergelists(None, copy=True)                   == []
    assert sc.mergelists([1,2,3], [4,5,6])                  == [1, 2, 3, 4, 5, 6]
    assert sc.mergelists([1,2,3], 4, 5, 6)                  == [1, 2, 3, 4, 5, 6]
    assert sc.mergelists([(1,2), (3,4)], (5,6))             == [(1, 2), (3, 4), (5, 6)]
    assert sc.mergelists((1,2), (3,4), (5,6))               == [(1, 2), (3, 4), (5, 6)]
    assert sc.mergelists((1,2), (3,4), (5,6), coerce=tuple) == [1, 2, 3, 4, 5, 6]
    
    print('\nTesting mergedicts')
    big = sc.mergedicts(sc.odict({'b':3, 'c':4}), {'a':1, 'b':2})
    assert big == sc.odict({'b':2, 'c':4, 'a':1})
    assert sc.mergedicts(None) == {}
    assert sc.mergedicts({'a':1}, {'b':2}, _copy=True) == {'a':1, 'b':2}
    assert sc.mergedicts({'a':1, 'b':2}, {'b':3, 'c':4}, None) == {'a':1, 'b':3, 'c':4}
    with pytest.raises(KeyError):
        assert sc.mergedicts({'b':3, 'c':4}, {'a':1, 'b':2}, _overwrite=False)
        
    print('\nTesting ifelse')
    
    # Basic usage
    a = None
    b = 3
    assert sc.ifelse(a, b) == a if a is not None else b
    
    # Boolean usage
    args = ['', False, {}, 'ok']
    assert sc.ifelse(*args, check=bool) == next((arg for arg in args if arg), None)
    
    # Custom function
    args = [1, 3, 5, 7]
    assert sc.ifelse(*args, check=lambda x: x>5) == 7
    
    # Default value
    assert sc.ifelse(default=[4]) == [4]
    
    return big


def test_types():
    sc.heading('Test type functions')
    o = sc.objdict()

    print('\nTesting isarray')
    assert sc.isarray(np.array([1,2,3]))
    assert not sc.isarray([1,2,3])
    assert not sc.isarray(np.array([1,2,3]), dtype=float)

    print('\nTesting flexstr')
    o.flexstr = sc.flexstr(b'bytestring')
    
    print('\nTesting sanitizestr')
    assert sc.sanitizestr('This Is a String', lower=True) == 'this is a string'
    assert sc.sanitizestr('Lukáš wanted €500‽', asciify=True, nospaces=True, symchar='*') == 'Lukas_wanted_*500*'
    assert sc.sanitizestr('"Ψ scattering", María said, "at ≤5 μm?"', asciify=True, alphanumeric=True, nospaces=True, spacechar='') == '??scattering??Mariasaid??at?5?m??'
    assert sc.sanitizestr('4 path/names/to variable!', validvariable=True, spacechar='') == '_4pathnamestovariable'

    print('\nTesting promotetoarray')
    assert not len(sc.promotetoarray(None, keepnone=False))
    assert sc.promotetoarray(np.array(3))[0] == 3
    with pytest.raises(ValueError):
        sc.toarray('not convertible', dtype=float)

    return o


#%% Misc. functions

def test_suggest():
    sc.heading('test_suggest()')
    string = 'foo'
    ex1 = ['Foo','Bar']
    ex2 = ['FOO','Foo']
    ex3 = ['Foo','boo']
    ex4 = ['asldfkj', 'aosidufasodiu']
    ex5 = ['foo', 'fou', 'fol', 'fal', 'fil']
    res1 = sc.suggest(string, ex1)
    res2 = sc.suggest(string, ex2)
    res3 = sc.suggest(string, ex3)
    res4 = sc.suggest(string, ex4, threshold=4)
    with pytest.raises(Exception):
        sc.suggest(string, ex1, threshold=4, die=True)
    res5a = sc.suggest(string, ex5, n=3)
    res5b = sc.suggest(string, ex5, fulloutput=True)
    assert res1 == 'Foo'
    assert res2 == 'Foo'
    assert res3 == 'Foo'
    assert res4 == None
    assert res5a == ['foo', 'fou', 'fol']
    assert res5b == {'foo': 0.0, 'fou': 1.0, 'fol': 1.0, 'fal': 2.0, 'fil': 2.0}
    print(res1)
    print(res2)
    print(res3)
    print(res4)
    print(res5a)
    print(res5b)
    return res5b


def test_misc():
    sc.heading('Testing miscellaneous functions')
    o = sc.objdict()

    print('\nTesting runcommand')
    sc.runcommand('command_probably_not_found', printinput=True, printoutput=True)
    sc.runcommand('ls', wait=False)

    print('\nTesting gitinfo functions')
    o.gitinfo = sc.gitinfo()

    print('\nTesting compareversions')
    assert sc.compareversions('1.2.3', '2.3.4') == -1
    assert sc.compareversions(2, '2') == 0
    assert sc.compareversions('3.1', '2.99') == 1

    print('\nTesting uniquename')
    namelist = ['file', 'file (1)', 'file (2)']
    o.unique = sc.uniquename(name='file', namelist=namelist)
    assert o.unique not in namelist

    print('\nTesting importbyname and importbypath')
    global lazynp
    sc.importbyname(lazynp='numpy', lazy=True, namespace=globals())
    print(lazynp)
    assert isinstance(lazynp, sc.LazyModule)
    lazynp.array(0)
    assert not isinstance(lazynp, sc.LazyModule)
    module_path = sc.thispath() / 'test_settings.py'
    test_set = sc.importbyname(path=module_path, variable='test_set')
    assert 'test_options' in dir(test_set)
    test_set2 = sc.importbypath(path=module_path)
    assert 'test_options' in dir(test_set2)
    with pytest.raises(FileNotFoundError):
        sc.importbypath(path='/not/a/valid/path')

    print('\nTesting get_caller()')
    o.caller = sc.getcaller(includeline=True)
    print(o.caller)

    print('\nTesting nestedloop')
    o.nested = list(sc.nestedloop([['a','b'],[1,2]],[0,1]))

    print('\nTesting strsplit')
    target = ['a', 'b', 'c']
    s1 = sc.strsplit('a b c') # Returns ['a', 'b', 'c']
    s2 = sc.strsplit('a,b,c') # Returns ['a', 'b', 'c']
    s3 = sc.strsplit('a, b, c') # Returns ['a', 'b', 'c']
    s4 = sc.strsplit('  foo_bar  ', sep='_') # Returns ['foo', 'bar']
    assert s1 == s2 == s3 == target
    assert s4 == ['foo', 'bar']

    print('\nTesting autolist')
    ls = sc.autolist('test')
    ls += 'a'
    ls += [3, 'b']
    assert ls ==  ['test', 'a', 3, 'b']
    ls2 = sc.autolist('x', 'y')
    ls3 = ls + ls2
    assert ls3 == ['test', 'a', 3, 'b', 'x', 'y']

    return o



#%% Classes

def test_links():
    sc.heading('Testing links')
    o = sc.objdict()

    with pytest.raises(KeyError):
        raise sc.KeyNotFoundError('Example')

    obj = sc.objdict()
    obj.uid  = sc.uuid()
    obj.data = np.random.rand(5)
    o.obj    = obj
    o.link   = sc.Link(obj)
    o.o_copy = sc.dcp(o)

    assert np.all(o.link()['data'] == o.obj['data'])

    with pytest.raises(sc.LinkException):
        o.o_copy.link()

    return o



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    # Adaptations
    adapt     = test_adaptations()
    download  = test_download()
    filename  = test_download_save()
    uid       = test_uuid()
    traceback = test_traceback()
    tryexc    = test_tryexcept()

    # Type
    plist = test_tolist()
    d1d2  = test_transpose_swap()
    mdict = test_merge()
    types = test_types()

    # Miscellaneous
    dists = test_suggest()
    misc  = test_misc()

    # Classes
    links = test_links()

    sc.toc()
    print('Done.')