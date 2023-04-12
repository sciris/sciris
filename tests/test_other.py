'''
Test Sciris miscellaneous utility/helper functions.
'''

import os
import sciris as sc
import pytest


#%% Test options

def test_options():
    sc.heading('Test options')

    print('Testing options')
    sc.options.help(detailed=True)
    sc.options.disp()
    print(sc.options)
    with sc.options.context(aspath=True):
        pass
    sc.options(dpi=150)
    sc.options('default')

    fn = 'options.json'
    sc.options.save(fn)
    sc.options.load(fn)
    sc.rmpath(fn)

    print('Testing sc.parse_env()')
    os.environ['TMP_STR'] = 'test'
    os.environ['TMP_INT'] = '4'
    os.environ['TMP_FLOAT'] = '2.3'
    os.environ['TMP_BOOL'] = 'False'
    assert sc.parse_env('TMP_STR',   which='str')   == 'test'
    assert sc.parse_env('TMP_INT',   which='int')   == 4
    assert sc.parse_env('TMP_FLOAT', which='float') == 2.3
    assert sc.parse_env('TMP_BOOL',  which='bool')  == False

    print('Testing help')
    sc.help()
    sc.help('smooth')
    sc.help('JSON', ignorecase=False, context=True)
    sc.help('pickle', source=True, context=True)

    return


#%% Nested dictionary functions

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
    
    # Define th enested object
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
    keymatches = sc.search(nested, key) # Returns ['["a"]["bar"]', '["b"]["bar"]']
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

    return o


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    # Options
    test_options()

    # Nested
    nested    = test_nested()
    dicts     = test_dicts()
    search    = test_search()

    sc.toc()
    print('Done.')