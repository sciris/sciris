'''
Nested dictionary functions.
'''

import itertools
from functools import reduce, partial
from . import sc_utils as scu


__all__ = ['getnested', 'setnested', 'makenested', 'iternested', 'iterobj',
           'mergenested', 'flattendict', 'search', 'nestedloop']


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
        elif not overwrite and value is not None: # pragma: no cover
            errormsg = f'Not overwriting entry {keylist} since overwrite=False'
            raise ValueError(errormsg)
    elif value is not None: # pragma: no cover
        errormsg = f'Cannot set value {value} since entry {keylist} is a {type(currentlevel)}, not a dict'
        raise TypeError(errormsg)
    return


def check_iter_type(obj):
    ''' Helper function to determine if an object is a dict, list, or neither '''
    if isinstance(obj, dict):
        out = 'dict'
    elif isinstance(obj, list):
        out = 'list'
    elif hasattr(obj, '__dict__'):
        out = 'object'
    else:
        out = '' # Evaluates to false
    return out
    

def get_from_obj(ndict, key, safe=False):
    ''' Get an item from a dict, list, or object by key '''
    itertype = check_iter_type(ndict)
    if itertype == 'dict':
        if safe:
            out = ndict.get(key)
        else:
            out = ndict[key]
    elif itertype == 'list':
        out = ndict[key]
    elif itertype == 'object':
        out = getattr(ndict, key)
    else:
        out = None
    return out


def getnested(nesteddict, keylist, safe=False):
    '''
    Get the value for the given list of keys

    >>> sc.getnested(foo, ['a','b'])

    See sc.makenested() for full documentation.
    '''
    get = partial(get_from_obj, safe=safe)
    output = reduce(get, keylist, nesteddict)
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
    if not isinstance(currentlevel, dict): # pragma: no cover
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


def iterobj(obj, func, inplace=False, twigs_only=False, verbose=False, _trace=None, _output=None, *args, **kwargs):
    '''
    Iterate over an object and apply a function to each twig.
    
    Can modify an object in-place, or return a value. See also :func:`sc.search() <search>`
    for a function to search through complex objects.
    
    Args:
        obj (any): the object to iterate over
        func (function): the function to apply
        inplace (bool): whether to modify the object in place (else, collate the output of the functions)
        twigs_only (bool): whether to apply the function only to twigs of the object
        verbose (bool): whether to print progress.
        _trace (list): used internally for recursion
        _output (list): used internally for recursion
        *args (list): passed to func()
        **kwargs (dict): passed to func()
    
    **Examples**::
        
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
        
    *New in version 3.0.0.*
    '''
    # Set the trace and output if needed
    if _trace is None:
        _trace = []
    if _output is None:
        _output = {}
        if not inplace and not twigs_only:
            _output['root'] = func(obj, *args, **kwargs)
    
    itertype = check_iter_type(obj)
    
    def iteritems(obj):
        ''' Return an iterator over items in this object '''
        if itertype == 'dict':
            return obj.items()
        elif itertype == 'list':
            return enumerate(obj)
        elif itertype == 'object':
            return obj.__dict__.items()
    
    def getitem(obj, key):
        ''' Get the value for the item '''
        if itertype in ['dict', 'list']:
            return obj[key]
        elif itertype == 'object':
            return obj.__dict__[key]
    
    def setitem(obj, key, value):
        ''' Set the value for the item '''
        if itertype in ['dict', 'list']:
            obj[key] = value
        elif itertype == 'object':
            obj.__dict__[key] = value
        return
        
    # Next, check if we need to iterate
    if itertype:
        for key,subobj in iteritems(obj):
            trace = _trace + [key]
            newobj = func(subobj, *args, **kwargs)
            if inplace:
                setitem(obj, key, newobj)
            else:
                _output[tuple(trace)] = newobj
            iterobj(getitem(obj, key), func, inplace=inplace, twigs_only=twigs_only,  # Run recursively
                    verbose=verbose, _trace=trace, _output=_output, *args, **kwargs)
        
    if inplace:
        return
    else:
        return _output


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
        nesteddict (dict): Input dictionary potentially containing dicts as values
        sep        (str): Concatenate keys using string separator. If ``None`` the returned dictionary will have tuples as keys
        _prefix: Internal argument for recursively accumulating the nested keys

    Returns:
        A flat dictionary where no values are dicts

    *New in version 2.0.0:* handle non-string keys.
    """
    output_dict = {}
    for k, v in nesteddict.items():
        if sep is None: # Create tuples
            if _prefix is None:
                k2 = (k,)
            else:
                k2 = _prefix + (k,)
        else: # Create strings
            if _prefix is None:
                k2 = k
            else:
                k2 = str(_prefix) + str(sep) + str(k)

        if isinstance(v, dict):
            output_dict.update(flattendict(nesteddict[k], sep=sep, _prefix=k2))
        else:
            output_dict[k2] = v

    return output_dict


# Define a custom "None" value to allow searching for actual None values
_None = 'sc.search() placeholder'
def search(obj, key=_None, value=_None, aslist=True, _trace=None):
    """
    Find a key/attribute or value within a list, dictionary or object.

    This function facilitates finding nested key(s) or attributes within an object,
    by searching recursively through keys or attributes.

    Args:
        obj (any): A dict, list, or object
        key (any): The key to search for (or a function)
        value (any): The value to search for
        aslist (bool): return entries as a list (else, return as a string)
        _trace: Not for user input - internal variable used for recursion

    Returns:
        A list of matching attributes. The items in the list are the Python
        strings (or lists) used to access the attribute (via attribute, dict, 
        or listindexing).

    **Examples**::

        # Create a nested dictionary
        nested = {'a':{'foo':1, 'bar':2}, 'b':{'car':3, 'cat':[1,2,4,8]}}
        
        # Find keys
        keymatches = sc.search(nested, 'bar', aslist=False) # Returns ['["a"]["bar"]', '["b"]["bar"]']
        
        # Find values
        val = 4
        valmatches = sc.search(nested, value=val, aslist=True) # Returns  [['b', 'cat', 2]]
        assert sc.getnested(nested, valmatches[0]) == val # Get from the original nested object
        
        # Find values with a function
        def find(v):
            return True if isinstance(v, int) and v >= 3 else False
            
        found = sc.search(nested, value=find) # Returns  [['b', 'cat', 2]]
    
    *New in version 3.0.0:* ability to search for values as well as keys/attributes; "aslist" argument
    """
    
    def check_match(source, target):
        ''' Check if there is a match between the "source" and "target" '''
        if target != _None: # See above for definition
            if callable(target):
                match = target(source)
            else:
                match = target == source
        else:
            match = False
        return match
            

    matches = []

    itertype = check_iter_type(obj)
    if itertype == 'dict':
        o = obj
    elif itertype == 'list':
        o = {k:v for k,v in zip(list(range(len(obj))), obj)}
    elif itertype == 'object':
        o = obj.__dict__
    else:
        return matches
    
    if _trace is None:
        if aslist:
            _trace = []
        else:
            _trace = ''
        
    for k,v in o.items():

        if aslist:
            trace = _trace + [k]
        else:
            if itertype == 'dict':
                trace = _trace + f'["{k}"]'
            elif itertype == 'list':
                trace = _trace + f'[{k}]'
            else:
                trace = _trace + f'.{k}'

        for source,target in [[k, key], [v, value]]:
            if check_match(source, target):
                matches.append(trace)

        matches += search(o[k], key, value, aslist=aslist, _trace=trace)

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

    *New in version 1.0.0.*
    """
    loop_order = list(loop_order)  # Convert to list, in case loop order was passed in as a generator e.g. from map()
    inputs = [inputs[i] for i in loop_order]
    iterator = itertools.product(*inputs)  # This is in the loop order
    for item in iterator:
        out = [None] * len(loop_order)
        for i in range(len(item)):
            out[loop_order[i]] = item[i]
        yield out
