'''
Nested dictionary functions.
'''

import itertools
from functools import reduce
from . import sc_utils as scu


__all__ = ['getnested', 'setnested', 'makenested', 'iternested', 'mergenested',
           'flattendict', 'search', 'nestedloop']


def makenested(nesteddict, keylist=None, value=None, overwrite=False, generator=None):
    '''
    Little functions to get and set data from nested dictionaries.

    The first two were adapted from: http://stackoverflow.com/questions/14692690/access-python-nested-dictionary-items-via-a-list-of-keys

    "getnested" will get the value for the given list of keys:

    >>> sc.getnested(foo, ['a','b'])

    "setnested" will set the value for the given list of keys:

    >>> sc.setnested(foo, ['a','b'], 3)

    "makenested" will recursively update a dictionary with the given list of keys:

    >>> sc.makenested(foo, ['a','b'])

    "iternested" will return a list of all the twigs in the current dictionary:

    >>> twigs = sc.iternested(foo)

    **Example 1**::

        foo = {}
        sc.makenested(foo, ['a','b'])
        foo['a']['b'] = 3
        print(sc.getnested(foo, ['a','b']))    # 3
        sc.setnested(foo, ['a','b'], 7)
        print(sc.getnested(foo, ['a','b']))    # 7
        sc.makenested(foo, ['bar','cat'])
        sc.setnested(foo, ['bar','cat'], 'in the hat')
        print(foo['bar'])  # {'cat': 'in the hat'}

    **Example 2**::

        foo = {}
        sc.makenested(foo, ['a','x'])
        sc.makenested(foo, ['a','y'])
        sc.makenested(foo, ['a','z'])
        sc.makenested(foo, ['b','a','x'])
        sc.makenested(foo, ['b','a','y'])
        count = 0
        for twig in sc.iternested(foo):
            count += 1
            sc.setnested(foo, twig, count)   # {'a': {'y': 1, 'x': 2, 'z': 3}, 'b': {'a': {'y': 4, 'x': 5}}}

    Version: 2014nov29
    '''
    if generator is None:
        generator = nesteddict.__class__ # By default, generate new dicts of the same class as the original one
    currentlevel = nesteddict
    for i,key in enumerate(keylist[:-1]):
        if not(key in currentlevel):
            currentlevel[key] = generator() # Create a new dictionary
        currentlevel = currentlevel[key]
    lastkey = keylist[-1]
    if isinstance(currentlevel, dict):
        if overwrite or lastkey not in currentlevel:
            currentlevel[lastkey] = value
        elif not overwrite and value is not None:
            errormsg = f'Not overwriting entry {keylist} since overwrite=False'
            raise ValueError(errormsg)
    elif value is not None:
        errormsg = f'Cannot set value {value} since entry {keylist} is a {type(currentlevel)}, not a dict'
        raise TypeError(errormsg)
    return


def getnested(nesteddict, keylist, safe=False):
    '''
    Get the value for the given list of keys

    >>> sc.getnested(foo, ['a','b'])

    See sc.makenested() for full documentation.
    '''
    output = reduce(lambda d, k: d.get(k) if d else None if safe else d[k], keylist, nesteddict)
    return output


def setnested(nesteddict, keylist, value, force=True):
    '''
    Set the value for the given list of keys

    >>> sc.setnested(foo, ['a','b'], 3)

    See sc.makenested() for full documentation.
    '''
    if force:
        makenested(nesteddict, keylist, overwrite=False)
    currentlevel = getnested(nesteddict, keylist[:-1])
    if not isinstance(currentlevel, dict):
        errormsg = f'Cannot set {keylist} since parent is a {type(currentlevel)}, not a dict'
        raise TypeError(errormsg)
    else:
        currentlevel[keylist[-1]] = value
    return # Modify nesteddict in place


def iternested(nesteddict, previous=None):
    '''
    Return a list of all the twigs in the current dictionary

    >>> twigs = sc.iternested(foo)

    See sc.makenested() for full documentation.
    '''
    if previous is None:
        previous = []
    output = []
    for k in nesteddict.items():
        if isinstance(k[1],dict):
            output += iternested(k[1], previous+[k[0]]) # Need to add these at the first level
        else:
            output.append(previous+[k[0]])
    return output


def mergenested(dict1, dict2, die=False, verbose=False, _path=None):
    '''
    Merge different nested dictionaries

    See sc.makenested() for full documentation.

    Adapted from https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge
    '''
    if _path is None: _path = []
    if _path:
        a = dict1 # If we're being recursive, work in place
    else:
        a = scu.dcp(dict1) # Otherwise, make a copy
    b = dict2 # Don't need to make a copy

    for key in b:
        keypath = ".".join(_path + [str(key)])
        if verbose:
            print(f'Working on {keypath}')
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                mergenested(dict1=a[key], dict2=b[key], _path=_path+[str(key)], die=die, verbose=verbose)
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                errormsg = f'Warning! Conflict at {keypath}: {a[key]} vs. {b[key]}'
                if die:
                    raise ValueError(errormsg)
                else:
                    a[key] = b[key]
                    if verbose:
                        print(errormsg)
        else:
            a[key] = b[key]
    return a


def flattendict(nesteddict, sep=None, _prefix=None):
    """
    Flatten nested dictionary

    **Example**::

        >>> sc.flattendict({'a':{'b':1,'c':{'d':2,'e':3}}})
        {('a', 'b'): 1, ('a', 'c', 'd'): 2, ('a', 'c', 'e'): 3}
        >>> sc.flattendict({'a':{'b':1,'c':{'d':2,'e':3}}}, sep='_')
        {'a_b': 1, 'a_c_d': 2, 'a_c_e': 3}

    Args:
        d: Input dictionary potentially containing dicts as values
        sep: Concatenate keys using string separator. If ``None`` the returned dictionary will have tuples as keys
        _prefix: Internal argument for recursively accumulating the nested keys

    Returns:
        A flat dictionary where no values are dicts
    """
    output_dict = {}
    for k, v in nesteddict.items():
        if sep is None:
            if _prefix is None:
                k2 = (k,)
            else:
                k2 = _prefix + (k,)
        else:
            if _prefix is None:
                k2 = k
            else:
                k2 = _prefix + sep + k

        if isinstance(v, dict):
            output_dict.update(flattendict(nesteddict[k], sep=sep, _prefix=k2))
        else:
            output_dict[k2] = v

    return output_dict


def search(obj, attribute, _trace=''):
    """
    Find a key or attribute within a dictionary or object.

    This function facilitates finding nested key(s) or attributes within an object,
    by searching recursively through keys or attributes.


    Args:
        obj: A dict or class with __dict__ attribute
        attribute: The substring to search for
        _trace: Not for user input - internal variable used for recursion

    Returns:
        A list of matching attributes. The items in the list are the Python
        strings used to access the attribute (via attribute or dict indexing)

    **Example**::

        nested = {'a':{'foo':1, 'bar':2}, 'b':{'bar':3, 'cat':4}}
        matches = sc.search(nested, 'bar') # Returns ['["a"]["bar"]', '["b"]["bar"]']
    """

    matches = []

    if isinstance(obj, dict):
        d = obj
    elif hasattr(obj, '__dict__'):
        d = obj.__dict__
    else:
        return matches

    for attr in d:

        if isinstance(obj, dict):
            s = _trace + f'["{attr}"]'
        else:
            s = _trace + f'.{attr}'

        if attribute in attr:
            matches.append(s)

        matches += search(d[attr], attribute, s)

    return matches


def nestedloop(inputs, loop_order):
    """
    Zip list of lists in order

    This function takes in a list of lists to iterate over, and their nesting order.
    It then yields tuples of items in the given order. Only tested for two levels
    but in theory supports an arbitrary number of items.

    Args:
        inputs (list): List of lists. All lists should have the same length
        loop_order (list): Nesting order for the lists

    Returns:
        Generator yielding tuples of items, one for each list

    Example usage:

    >>> list(sc.nestedloop([['a','b'],[1,2]],[0,1]))
    [['a', 1], ['a', 2], ['b', 1], ['b', 2]]

    Notice how the first two items have the same value for the first list
    while the items from the second list vary. If the `loop_order` is
    reversed, then:

    >>> list(sc.nestedloop([['a','b'],[1,2]],[1,0]))
    [['a', 1], ['b', 1], ['a', 2], ['b', 2]]

    Notice now how now the first two items have different values from the
    first list but the same items from the second list.

    From Atomica by Romesh Abeysuriya.

    New in version 1.0.0.
    """
    loop_order = list(loop_order)  # Convert to list, in case loop order was passed in as a generator e.g. from map()
    inputs = [inputs[i] for i in loop_order]
    iterator = itertools.product(*inputs)  # This is in the loop order
    for item in iterator:
        out = [None] * len(loop_order)
        for i in range(len(item)):
            out[loop_order[i]] = item[i]
        yield out