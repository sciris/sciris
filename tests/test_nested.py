'''
Test nested dict functions
'''

import sciris as sc
import pytest


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
    keymatches = sc.search(nested, key) # Returns ['["a"]["bar"]', '["b"]["bar"]'] as lists
    o.keymatches = keymatches
    print(keymatches)
    assert len(keymatches) == 2
    for keymatch in keymatches:
        assert key in keymatch
        
    print('\nTesting search by value')
    val = 8
    valmatches = sc.search(nested, value=val, aslist=True) # Returns ['["a"]["bar"]', '["b"]["bar"]']
    o.valmatches = valmatches
    print(valmatches)
    assert len(valmatches) == 1
    assert sc.getnested(nested, valmatches[0]) == val # Get from the original nested object
    
    # Test aslist=False
    valstrs = sc.search(nested, value=val, aslist=False)
    assert isinstance(valstrs[0], str)
    
    return o


def test_iterobj():
    sc.heading('Testing iterobj')
    data = dict(a=dict(x=[1,2,3], y=[4,5,6]), b=dict(foo='string', bar='other_string'))
    
    # Search through an object
    def check_type(obj, which):
        return isinstance(obj, which)

    out = sc.iterobj(data, check_type, which=int)
    print(out)
    
    # Modify in place -- collapse mutliple short lines into one
    def collapse(obj):
        string = str(obj)
        if len(string) < 10:
            return string
        else:
            return obj

    sc.printjson(data)
    sc.iterobj(data, collapse, inplace=True)
    sc.printjson(data)
    
    return out


#%% Run as a script
if __name__ == '__main__':
    T = sc.timer()

    # Nested
    nested  = test_nested()
    dicts   = test_dicts()
    search  = test_search()
    iterobj = test_iterobj()

    T.toc('Done.')