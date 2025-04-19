'''
Test nested dict functions
'''

import numpy as np
import pandas as pd
import sciris as sc
import pytest
ut = sc.importbypath(sc.thispath() / 'sc_test_utils.py')


def test_nested():
    sc.heading('Testing nested dicts')
    o = sc.objdict()

    foo = {}
    sc.makenested(foo, ['a','b'])
    foo['a']['b'] = 3
    print(sc.getnested(foo, ['a','b']))    # 3
    sc.setnested(foo, ['a','b'], 7)
    print(sc.getnested(foo, ['a','b']))    # 7
    sc.makenested(foo, ['bar','cat'])
    sc.setnested(foo, ['bar','cat'], 'in the hat')
    print(foo['bar'])  # {'cat': 'in the hat'}
    sc.setnested(foo, 'soliton', 999) # Works with a single key
    key_tuple = ('a', 'b')
    sc.setnested(foo, key_tuple, 45)
    assert sc.getnested(foo, key_tuple) == 45, "Did not correctly set/get from dict"
    o.foo1 = foo

    foo = {}
    sc.makenested(foo, ['a','x'])
    sc.makenested(foo, ['a','y'])
    sc.makenested(foo, ['a','z'])
    sc.makenested(foo, ['b','a','x'])
    sc.makenested(foo, ['b','a','y'])
    count = 0
    for twig in sc.iternested(foo):
        count += 1
        sc.setnested(foo, twig, count) # Yields {'a': {'y': 1, 'x': 2, 'z': 3}, 'b': {'a': {'y': 4, 'x': 5}}}
    assert sc.getnested(foo, ['b','a','y']) == foo['b']['a']['y'] == 5, 'Incorrect value/keys'
    o.foo2 = foo

    return o


def test_dicts():
    sc.heading('Testing dicts')
    o = sc.objdict()

    print('\nTesting flattendict')
    sc.flattendict({'a': {'b': 1, 'c': {'d': 2, 'e': 3}}})
    o.flat = sc.flattendict({'a': {'b': 1, 'c': {'d': 2, 'e': 3}}}, sep='_')

    print('Testing merging dictionaries')
    o.md1 = sc.mergedicts({'a':1}, {'b':2}) # Returns {'a':1, 'b':2}
    o.md2 = sc.mergedicts({'a':1, 'b':2}, {'b':3, 'c':4}) # Returns {'a':1, 'b':3, 'c':4}
    o.md3 = sc.mergedicts({'b':3, 'c':4}, a=1, b=2) # Returns {'a':1, 'b':2, 'c':4}
    assert o.md3 == {'a':1, 'b':2, 'c':4}
    with pytest.raises(KeyError):
        sc.mergedicts({'b':3, 'c':4}, {'a':1, 'b':2}, _overwrite=False) # Raises exception
    with pytest.raises(TypeError):
        sc.mergedicts({'b':3, 'c':4}, None, _strict=True) # Raises exception

    print('\nTesting nested dictionaries')
    dict1 = {'key1':{'a':'A'},  'key2':{'b':'B'}}
    dict2 = {'key1':{'a':'A*'}, 'key2':{'b+':'B+'}, 'key3':{'c':'C'}}
    dict3 = sc.mergenested(dict1, dict2, verbose=True)
    print('↑ Should print warning above')
    print(f'Dict1: {dict1}')
    print(f'Dict2: {dict2}')
    print(f'Dict3: {dict3}')
    assert dict3 == {'key1': {'a': 'A*'}, 'key2': {'b': 'B', 'b+': 'B+'}, 'key3': {'c': 'C'}}
    o.dict3 = dict3

    return o


def test_search():
    sc.heading('Testing search')
    o = sc.objdict()

    print('Docstring tests')
    # Create a nested dictionary
    nested = {'a':{'foo':1, 'bar':['moat', 'goat']}, 'b':{'car':'far', 'cat':[1,2,4,8]}}

    # Find keys
    keymatches = sc.search(nested, 'bar').keys()
    assert 'bar' in keymatches[0]

    # Find values
    val = 4
    valmatches = sc.search(nested, value=val).keys() # Returns  [('b', 'cat', 2)]
    assert sc.getnested(nested, valmatches[0]) == val # Get from the original nested object

    # Find values with a function
    def find(v):
        return True if isinstance(v, int) and v >= 3 else False

    func_found = sc.search(nested, value=find).keys()
    assert ('b', 'cat', 3) in func_found

    # Find partial or regex matches
    partial = sc.search(nested, value='oat', method='partial') # Search keys only
    d = sc.search(nested, '^.ar', method='regex', verbose=True)
    keys,vals = d.keys(), d.values()

    assert ('a', 'bar', 1) in partial
    assert ('a', 'bar') in keys
    assert 'far' in vals


    # Define the nested object
    nested = {
        'a': {
             'foo':1,
             'bar':2,
        },
        'b':{
            'bar':3,
            'cat':sc.prettyobj(hat=[1,2,4,8]),
        }
    }

    print('\nTesting search by key')
    key = 'bar'
    keymatches = sc.search(nested, key, flatten=True).keys() # Returns ['a_bar', 'b_bar']
    o.keymatches = keymatches
    print(keymatches)
    assert len(keymatches) == 2
    for keymatch in keymatches:
        assert key in keymatch

    print('\nTesting search by value')
    val = 8
    valmatches = sc.search(nested, value=val, flatten=False).keys() # Returns [('b', 'cat', 'hat', 3)]
    o.valmatches = valmatches
    print(valmatches)
    assert len(valmatches) == 1
    assert sc.getnested(nested, valmatches[0]) == val # Get from the original nested object

    return o


def test_iterobj():
    sc.heading('Testing iterobj')

    o = sc.prettyobj()
    o.a = sc.prettyobj()
    o.b = sc.prettyobj()
    o.a.i1 = [1,2,3]
    o.b.i2 = dict(cat=[4,5,6])
    data = dict(
        a = dict(
            x = [1,2,3],
            y = [4,5,6]),
        b = dict(
            foo = 'string',
            bar = 'other_string'),
        c = o,
    )

    # Search through an object
    def check_type(obj, which):
        return isinstance(obj, which)

    out1 = sc.iterobj(data, check_type, which=int)
    out2 = sc.iterobj(data, check_type, which=int, depthfirst=False)
    assert out1.keys().index(('a', 'x')) < out1.keys().index(('b',))
    assert out2.keys().index(('a', 'x')) > out2.keys().index(('b',))
    print('Depth first:')
    print(out1)
    print('Breadth first:')
    print(out2)

    # Modify in place -- collapse mutliple short lines into one
    def collapse(obj, maxlen):
        string = str(obj)
        if len(string) < maxlen:
            return string
        else:
            return obj

    print('Before collapse:')
    sc.printjson(data)
    sc.iterobj(data, collapse, inplace=True, maxlen=10) # Note passing of keyword argument to function
    # sc.iterobj(data, collapse, inplace=True, maxlen=10) # Note passing of keyword argument to function
    print('After collapse:')
    sc.printjson(data)
    assert data['a']['x'] == '[1, 2, 3]'

    return out1


def test_iterobj_class():
    sc.heading('Testing iterobj class')

    # Create a simple class for storing data
    class DataObj(sc.prettyobj):
        def __init__(self, **kwargs):
            self.keys   = tuple(kwargs.keys())
            self.values = tuple(kwargs.values())

    # Create the data
    obj1 = DataObj(a=[0,1,2], b=[3,4,5])
    obj2 = DataObj(c=[6,7,8], d=[9])
    obj = DataObj(obj1=obj1, obj2=obj2)

    # Define custom methods for iterating over tuples and the DataObj
    def custom_iter(obj):
        if isinstance(obj, tuple):
            return enumerate(obj)
        if isinstance(obj, DataObj):
            return [(k,v) for k,v in zip(obj.keys, obj.values)]

    # Define custom method for getting data from each
    def custom_get(obj, key):
        if isinstance(obj, tuple):
            return obj[key]
        elif isinstance(obj, DataObj):
            return obj.values[obj.keys.index(key)]

    # Gather all data into one list
    all_data = []
    def gather_data(obj, all_data=all_data):
        if isinstance(obj, list):
            all_data += obj

    # Run the iteration
    io = sc.IterObj(obj, func=gather_data, custom_type=(tuple, DataObj), custom_iter=custom_iter, custom_get=custom_get)
    print(all_data)
    assert all_data == list(range(10))

    return io


def test_equal():
    sc.heading('Testing equal')
    out = sc.objdict()

    print('Validating signatures')
    ut.check_signatures(sc.equal, sc.Equal.__init__, extras=['self', 'compare'], die=True)


    print('Testing docstring examples')
    o1 = dict(
        a = [1,2,3],
        b = np.array([4,5,6]),
        c = dict(
            df = sc.dataframe(q=[sc.date('2022-02-02'), sc.date('2023-02-02')])
        ),
        d = pd.Series([1,2,np.nan]),
    )

    # Identical object
    o2 = sc.dcp(o1)

    # Non-identical object
    o3 = sc.dcp(o1)
    o3['b'][2] = 8

    # Different subtype
    o4 = sc.dcp(o1)
    o4['a'] = dict(a=3, b=5)

    # Extra and missing keys
    o5 =  dict(
        b = np.array([4,5,6]),
        d = pd.Series([1,2,np.nan]),
        e = dict(this='is', a='test')
    )

    out.e1 = sc.equal(o1, o2) # Returns True
    out.e2 = sc.equal(o1, o3) # Returns False
    out.e3 = sc.Equal(o1, o2, o3, detailed=True, equal_nan=True) # Create an object
    out.e4 = sc.Equal(o1, o4, verbose=True)
    out.e5 = sc.Equal(o1, o5, detailed=2)

    # Do tests
    assert out.e1
    assert not out.e2
    assert not out.e3.eq
    assert not out.e4.eq

    print('Testing other features')
    for method in ['pickle', 'eq', 'json', 'str']:
        assert sc.equal(o1, o2, method=method, equal_nan=True) or method == 'json' # Known failure for JSON
        assert not sc.equal(o1, o2, method=method, equal_nan=False) or method in ['pickle', 'str'] # Known failure for pickle and string
        assert not sc.equal(o1, o3, method=method)

    out.e6 = sc.Equal(o1, o3, detailed=True, verbose=True)
    print('↑↑↑ Should print some handled exceptions')

    # Test totally different objects
    assert not sc.equal(1, 'a')
    assert not sc.equal(dict(a=dict(x=1,y=2), b=3), dict(a=dict(x=1,y=2)), detailed=True).all(axis=None) # Returns a dataframe

    return out


#%% Run as a script
if __name__ == '__main__':
    T = sc.timer()

    # Nested
    nested  = test_nested()
    dicts   = test_dicts()
    search  = test_search()
    iterobj = test_iterobj()
    io_obj  = test_iterobj_class()
    equal   = test_equal()

    T.toc('Done.')