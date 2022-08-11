'''
Test Sciris utility/helper functions.
'''

import numpy as np
import sciris as sc
import pytest


#%% Adaptations from other libraries

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

    print('\nTesting wget')
    o.wget = sc.wget('http://wikipedia.org/')

    print('\nTesting htmlify')
    o.html = sc.htmlify('foo&\nbar') # Returns b'foo&amp;<br>bar'
    o.nothtml = sc.htmlify(o.wget, reverse=True)

    print('\nTesting traceback')
    o.traceback = sc.traceback()

    return o


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


def test_versions():
    sc.heading('Testing freeze, compareversions, and require')

    # Test freeze
    assert 'numpy' in sc.freeze()

    # Test compareversions
    assert sc.compareversions(np, '>1.0')

    # Test require
    sc.require('numpy')
    sc.require(numpy='')
    sc.require(reqs={'numpy':'1.19.1', 'matplotlib':'3.2.2'})
    sc.require('numpy>=1.19.1', 'matplotlib==3.2.2', die=False)
    data, _ = sc.require(numpy='1.19.1', matplotlib='==4.2.2', die=False, detailed=True)
    with pytest.raises(ModuleNotFoundError): sc.require('matplotlib==99.23')
    with pytest.raises(ModuleNotFoundError): sc.require('not_a_valid_module')
    return data


#%% Type functions

def test_promotetolist():
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

    print('\nTesting transposelist')
    o = sc.odict(a=1, b=4, c=9, d=16)
    itemlist = o.enumitems()
    inds, keys, vals = sc.transposelist(itemlist)
    assert keys[2] == 'c'
    assert inds[3] == 3
    assert vals[1] == 4

    print('\nTesting mergelists')
    assert sc.mergelists(None, copy=True)                   == []
    assert sc.mergelists([1,2,3], [4,5,6])                  == [1, 2, 3, 4, 5, 6]
    assert sc.mergelists([1,2,3], 4, 5, 6)                  == [1, 2, 3, 4, 5, 6]
    assert sc.mergelists([(1,2), (3,4)], (5,6))             == [(1, 2), (3, 4), (5, 6)]
    assert sc.mergelists((1,2), (3,4), (5,6))               == [(1, 2), (3, 4), (5, 6)]
    assert sc.mergelists((1,2), (3,4), (5,6), coerce=tuple) == [1, 2, 3, 4, 5, 6]

    return res3b


def test_types():
    sc.heading('Test type functions')
    o = sc.objdict()

    print('\nTesting isarray')
    assert sc.isarray(np.array([1,2,3]))
    assert not sc.isarray([1,2,3])
    assert not sc.isarray(np.array([1,2,3]), dtype=float)

    print('\nTesting flexstr')
    o.flexstr = sc.flexstr(b'bytestring')

    print('\nTesting promotetoarray')
    assert not len(sc.promotetoarray(None, keepnone=False))
    assert sc.promotetoarray(np.array(3))[0] == 3
    with pytest.raises(ValueError):
        sc.toarray('not convertible', dtype=float)

    return o


#%% Misc. functions

def test_profile():
    sc.heading('Test profiling functions')

    def slow_fn():
        n = 10000
        int_list = []
        int_dict = {}
        for i in range(n):
            int_list.append(i)
            int_dict[i] = i
        return

    def big_fn():
        n = 1000
        int_list = []
        int_dict = {}
        for i in range(n):
            int_list.append([i]*n)
            int_dict[i] = [i]*n
        return

    class Foo:
        def __init__(self):
            self.a = 0
            return

        def outer(self):
            for i in range(100):
                self.inner()
            return

        def inner(self):
            for i in range(1000):
                self.a += 1
            return

    foo = Foo()
    try:
        sc.mprofile(big_fn) # NB, cannot re-profile the same function at the same time
    except TypeError as E: # This happens when re-running this script
        print(f'Unable to re-profile memory function; this is usually not cause for concern ({E})')
    sc.profile(run=foo.outer, follow=[foo.outer, foo.inner])
    lp = sc.profile(slow_fn)

    return lp


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

    print('\nTesting checkmem')
    sc.checkmem(['spiffy',np.random.rand(243,589)], descend=True)

    print('\nTesting checkram')
    o.ram = sc.checkram()
    print(o.ram)

    print('\nTesting runcommand')
    sc.runcommand('command_probably_not_found')

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

    print('\nTesting importbyname')
    sc.importbyname('numpy')

    print('\nTesting get_caller()')
    o.caller = sc.getcaller()
    print(o.caller)

    print('\nTesting nestedloop')
    o.nested = list(sc.nestedloop([['a','b'],[1,2]],[0,1]))

    return o



#%% Classes


def test_links():
    sc.heading('Testing links')
    o = sc.objdict()

    with pytest.raises(KeyError):
        raise sc.KeyNotFoundError('Example')

    obj = sc.objdict()
    obj.uid = sc.uuid()
    obj.data = np.random.rand(5)
    o.obj = obj
    o.link = sc.Link(obj)
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
    uid       = test_uuid()
    traceback = test_traceback()
    versions  = test_versions()

    # Type
    plist     = test_promotetolist()
    types     = test_types()

    # Miscellaneous
    lp        = test_profile()
    dists     = test_suggest()
    misc      = test_misc()

    # Classes
    links   = test_links()

    sc.toc()
    print('Done.')