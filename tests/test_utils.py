'''
Test Sciris utility/helper functions.
'''

import pytest
import numpy as np
import sciris as sc


def test_colorize():
    sc.heading('Test text colorization')
    sc.colorize(showhelp=True)
    print('Simple example:')
    sc.colorize('green', 'hi')
    print('More complicated example:')
    sc.colorize(['yellow', 'bgblack'])
    print('Hello world')
    print('Goodbye world')
    sc.colorize('reset') # Colorize all output in between
    bluearray = sc.colorize(color='blue', string=str(range(5)), output=True)
    print("This should be blue: " + bluearray)
    return


def test_printing():
    sc.heading('Test printing functions')
    example = sc.prettyobj()
    example.data = sc.vectocolor(10)
    print('sc.pr():')
    sc.pr(example)
    print('sc.pp():')
    sc.pp(example.data)
    string = sc.pp(example.data, doprint=False)
    print('sc.printdata():')
    sc.printdata(example.data)
    return string


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
    except TypeError: # This happens when re-running this script
        pass
    sc.profile(run=foo.outer, follow=[foo.outer, foo.inner])
    sc.profile(slow_fn)

    return foo


def test_prepr():
    sc.heading('Test pretty representation of an object')
    n_attrs = 500
    myobj = sc.prettyobj()
    for i in range(n_attrs):
        key = f'attr{i:03d}'
        setattr(myobj, key, i**2)
    print(myobj)
    return myobj


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
    print(f'UIDs:')
    for key,val in u.items():
        print(f'{key}: {val}')

    return u


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
    print(res1)
    print(res2a)
    print(res2b)
    print(res3a)
    print(res3b)
    return res3b


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


def test_thisdir():
    sc.heading('Test getting the current file directory')
    import os

    thisdir = sc.thisdir(__file__)
    assert os.path.split(thisdir)[-1] == 'tests'
    print(f'Current folder: {thisdir}')

    return thisdir


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


def test_readdate():
    sc.heading('Test string-to-date conversion')

    string1 = '2020-Mar-21'
    string2 = '2020-03-21'
    string3 = 'Sat Mar 21 23:13:56 2020'
    dateobj1 = sc.readdate(string1)
    dateobj2 = sc.readdate(string2)
    sc.readdate(string3)
    assert dateobj1 == dateobj2
    with pytest.raises(ValueError):
        sc.readdate('Not a date')

    # Automated tests
    formats_to_try = sc.readdate(return_defaults=True)
    for key,fmt in formats_to_try.items():
        datestr = sc.getdate(dateformat=fmt)
        dateobj = sc.readdate(datestr, dateformat=fmt)
        print(f'{key:15s} {fmt:22s}: {dateobj}')

    return dateobj1


def test_flattendict():
    # Simple integration test to make sure the function runs without raising an error
    sc.flattendict({'a': {'b': 1, 'c': {'d': 2, 'e': 3}}})
    flat = sc.flattendict({'a': {'b': 1, 'c': {'d': 2, 'e': 3}}}, sep='_')
    return flat


def test_mergedicts():
    sc.heading('Test merging dictionaries')

    md = sc.mergedicts({'a':1}, {'b':2}) # Returns {'a':1, 'b':2}
    sc.mergedicts({'a':1, 'b':2}, {'b':3, 'c':4}) # Returns {'a':1, 'b':3, 'c':4}
    sc.mergedicts({'b':3, 'c':4}, {'a':1, 'b':2}) # Returns {'a':1, 'b':2, 'c':4}
    with pytest.raises(KeyError):
        sc.mergedicts({'b':3, 'c':4}, {'a':1, 'b':2}, overwrite=False) # Raises exception
    with pytest.raises(TypeError):
        sc.mergedicts({'b':3, 'c':4}, None, strict=True) # Raises exception

    return md


def test_nested_dicts():
    sc.heading('Testing nested dictionaries')
    dict1 = {'key1':{'a':'A'},  'key2':{'b':'B'}}
    dict2 = {'key1':{'a':'A*'}, 'key2':{'b+':'B+'}, 'key3':{'c':'C'}}
    dict3 = sc.mergenested(dict1, dict2, verbose=True)
    print(f'Dict1: {dict1}')
    print(f'Dict2: {dict2}')
    print(f'Dict3: {dict3}')
    assert dict3 == {'key1': {'a': 'A*'}, 'key2': {'b': 'B', 'b+': 'B+'}, 'key3': {'c': 'C'}}
    return dict3


def test_search():
    sc.heading('Testing search')
    nested = {'a':{'foo':1, 'bar':2}, 'b':{'bar':3, 'cat':4}}
    matches = sc.search(nested, 'bar') # Returns ['["a"]["bar"]', '["b"]["bar"]']
    print(matches)
    return matches


def test_progress_bar():
    sc.heading('Progress bar')
    n = 50
    for i in range(n):
        sc.progressbar(i+1, n)
        sc.timedsleep(1.0/n, verbose=False)
    return i


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    bluearray = test_colorize()
    string    = test_printing()
    foo       = test_profile()
    myobj     = test_prepr()
    uid       = test_uuid()
    plist     = test_promotetolist()
    dists     = test_suggest()
    thisdir   = test_thisdir()
    traceback = test_traceback()
    dateobj   = test_readdate()
    flat      = test_flattendict()
    md        = test_mergedicts()
    nested    = test_nested_dicts()
    matches   = test_search()
    ind       = test_progress_bar()

    sc.toc()
    print('Done.')