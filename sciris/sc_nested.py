"""
Functions for working on nested (multi-level) dictionaries and objects.

Highlights:
    - :func:`sc.getnested() <getnested>`: get a value from a highly nested dictionary
    - :func:`sc.search() <search>`: find a value in a nested object
    - :func:`sc.equal() <equal>`: check complex objects for equality
"""

import re
import itertools
import pickle as pkl
from functools import reduce, partial
import numpy as np
import pandas as pd
from . import sc_utils as scu

# Define objects for which it doesn't make sense to descend further -- used here and sc.equal()
_atomic_classes = (np.ndarray, pd.Series, pd.DataFrame, pd.core.indexes.base.Index) 


##############################################################################
#%% Nested dict and object functions
##############################################################################


__all__ = ['getnested', 'setnested', 'makenested', 'iternested', 'IterObj', 'iterobj',
           'mergenested', 'flattendict', 'nestedloop']


def makenested(nesteddict, keylist=None, value=None, overwrite=False, generator=None):
    """
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
    """
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


def check_iter_type(obj, check_array=False, known=None, known_to_none=True, custom=None):
    """ Helper function to determine if an object is a dict, list, or neither -- not for the user """
    out = None
    if custom is not None: # Handle custom first, to allow overrides
        if custom and not callable(custom): # Ensure custom_type is callable
            custom_func = (lambda obj: 'custom' if isinstance(obj, custom) else None)
        else:
            custom_func = custom
        out = custom_func(obj)
    if out is None:
        if known is not None and isinstance(obj, known):
            out = '' if known_to_none else 'known' # Choose how known objects are handled
        elif isinstance(obj, dict):
            out = 'dict'
        elif isinstance(obj, list):
            out = 'list'
        elif hasattr(obj, '__dict__'):
            out = 'object'
        elif check_array and isinstance(obj, np.ndarray):
            out = 'array'
        else:
            out = '' # Evaluates to false
    return out
    

def get_from_obj(ndict, key, safe=False, **kwargs):
    """
    Get an item from a dict, list, or object by key
    
    Args:
        ndict (dict/list/obj): the object to get from
        key (any): the key to get
        safe (bool): whether to return None if the key is not found (default False)
        kwargs (dict): passed to ``check_iter_type()``
    """
    itertype = check_iter_type(ndict, **kwargs)
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


def getnested(nested, keylist, safe=False):
    """
    Get the value for the given list of keys
    
    Args:
        nested (any): the nested object (dict, list, or object) to get from
        keylist (list): the list of keys
        safe (bool): whether to return None if the key is not found
    
    **Example**::

        sc.getnested(foo, ['a','b']) # Gets foo['a']['b']

    See :func:`sc.makenested() <makenested>` for full documentation.
    """
    get = partial(get_from_obj, safe=safe)
    nested = reduce(get, keylist, nested)
    return nested


def setnested(nested, keylist, value, force=True):
    """
    Set the value for the given list of keys
    
    Args:
        nested (any): the nested object (dict, list, or object) to modify
        keylist (list): the list of keys to use
        value (any): the value to set
        force (bool): whether to create the keys if they don't exist (NB: only works for dictionaries)
    
    **Example**::

        sc.setnested(foo, ['a','b'], 3) # Sets foo['a']['b'] = 3

    See :func:`sc.makenested() <makenested>` for full documentation.
    """
    if force:
        makenested(nested, keylist, overwrite=False)
    currentlevel = getnested(nested, keylist[:-1])
    if not isinstance(currentlevel, dict): # pragma: no cover
        errormsg = f'Cannot set {keylist} since parent is a {type(currentlevel)}, not a dict'
        raise TypeError(errormsg)
    else:
        currentlevel[keylist[-1]] = value
    return nested # Return object, but note that it's modified in place


def iternested(nesteddict, _previous=None):
    """
    Return a list of all the twigs in the current dictionary
    
    Args:
        nesteddict (dict): the dictionary
    
    **Example**::

        twigs = sc.iternested(foo)

    See :func:`sc.makenested() <makenested>` for full documentation.
    """
    if _previous is None:
        _previous = []
    output = []
    for k in nesteddict.items():
        if isinstance(k[1], dict):
            output += iternested(k[1], _previous+[k[0]]) # Need to add these at the first level
        else:
            output.append(_previous+[k[0]])
    return output


class IterObj(object):
    """
    Object iteration manager
    
    For arguments and usage documentation, see :func:`sc.iterobj() <iterobj>`.
    Use this class only if you want more control over how the object is iterated over.
    
    Class-specific args:
        custom_type (func): a custom function for returning a string for a specific object type (should return '' by default)
        custom_iter (func): a custom function for iterating (returning a list of keys) over an object
        custom_get  (func): a custom function for getting an item from an object
        custom_set  (func): a custom function for setting an item in an object
    
    **Example**::
        
        import sciris as sc

        # Create a simple class for storing data
        class DataObj(sc.prettyobj):
            def __init__(self, **kwargs):
                self.keys   = tuple(kwargs.keys())
                self.values = tuple(kwargs.values())

        # Create the data
        obj1 = DataObj(a=[1,2,3], b=[4,5,6])
        obj2 = DataObj(c=[7,8,9], d=[10])
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
        io.iterate()
        print(all_data)
        
    | *New in version 3.1.2.*
    """    
    def __init__(self, obj, func=None, inplace=False, copy=False, leaf=False, atomic='default', verbose=False, 
                _trace=None, _output=None, custom_type=None, custom_iter=None, custom_get=None, custom_set=None,
                *args, **kwargs):
        from . import sc_odict as sco # To avoid circular import
        
        # Default argument
        self.obj       = obj
        self.func      = func
        self.inplace   = inplace
        self.copy      = copy
        self.leaf      = leaf
        self.atomic    = atomic
        self.verbose   = verbose
        self._trace    = _trace
        self._output   = _output
        self.func_args = args
        self.func_kw   = kwargs
        
        # Custom arguments
        self.custom_type = custom_type
        self.custom_iter = custom_iter
        self.custom_get  = custom_get
        self.custom_set  = custom_set
        
        # Handle inputs
        if self.func is None: # Define the default function
            self.func = lambda obj: obj 
        if self.atomic == 'default': # Handle objects to not descend into
            self.atomic = _atomic_classes 
        if self._trace is None:
            self._trace = [] # Handle where we are in the object
            if inplace and copy: # Only need to copy once
                self.obj = scu.dcp(obj)
        if self._output is None: # Handle the output at the root level
            self._output = sco.objdict()
            if not inplace:
                self._output['root'] = self.func(self.obj, *args, **kwargs)
                
        # Check what type of object we have
        self.itertype = self.check_iter_type(self.obj)
        
        return
    
    def indent(self, string='', space='  '):
        """ Print, with output indented successively """
        if self.verbose:
            print(space*len(self._trace) + string)
        return
        
    def iteritems(self):
        """ Return an iterator over items in this object """
        self.indent(f'Iterating with type "{self.itertype}"')
        out = None
        if self.custom_iter:
            out = self.custom_iter(self.obj)
        if out is None:
            if self.itertype == 'dict':
                out = self.obj.items()
            elif self.itertype == 'list':
                out = enumerate(self.obj)
            elif self.itertype == 'object':
                out = self.obj.__dict__.items()
            else:
                out = {}.items() # Return nothing if not recognized
        return out
    
    def getitem(self, key):
        """ Get the value for the item """
        self.indent(f'Getting key "{key}"')
        if self.itertype in ['dict', 'list']:
            return self.obj[key]
        elif self.itertype == 'object':
            return self.obj.__dict__[key]
        elif self.custom_get:
            return self.custom_get(self.obj, key)
        else:
            return None
    
    def setitem(self, key, value):
        """ Set the value for the item """
        self.indent(f'Setting key "{key}"')
        if self.itertype in ['dict', 'list']:
            self.obj[key] = value
        elif self.itertype == 'object':
            self.obj.__dict__[key] = value
        elif self.custom_set:
            self.custom_set(self.obj, key, value)
        return
    
    def check_iter_type(self, obj):
        """ Shortcut to check_iter_type() """
        return check_iter_type(obj, known=self.atomic, custom=self.custom_type)
    
    def iterate(self):
        """ Actually perform the iteration over the object """
        
        # Iterate over the object
        for key,subobj in self.iteritems():
            trace = self._trace + [key]
            newobj = subobj
            subitertype = self.check_iter_type(subobj)
            self.indent(f'Working on {trace}, leaf={self.leaf}, type={str(subitertype)}')
            if not (self.leaf and subitertype):
                newobj = self.func(subobj, *self.func_args, **self.func_kw)
                if self.inplace:
                    self.setitem(key, newobj)
                else:
                    self._output[tuple(trace)] = newobj
            io = IterObj(self.getitem(key), self.func, inplace=self.inplace, leaf=self.leaf,  # Create a new instance
                    atomic=self.atomic, verbose=self.verbose, _trace=trace, _output=self._output, 
                    custom_type=self.custom_type, custom_iter=self.custom_iter, custom_get=self.custom_get, custom_set=self.custom_set,
                    *self.func_args, **self.func_kw)
            io.iterate() # Run recursively
            
        if self.inplace:
            newobj = self.func(self.obj, *self.func_args, **self.func_kw) # Set at the root level
            return newobj
        else:
            if (not self._trace) and (len(self._output)>1) and self.leaf: # We're at the top level, we have multiple entries, and only leaves are requested
                self._output.pop('root') # Remove "root" with leaf=True if it's not the only node
            return self._output
        

def iterobj(obj, func=None, inplace=False, copy=False, leaf=False, atomic='default', verbose=False, 
            _trace=None, _output=None, *args, **kwargs):
    """
    Iterate over an object and apply a function to each node (item with or without children).
    
    Can modify an object in-place, or return a value. See also :func:`sc.search() <search>`
    for a function to search through complex objects.
    
    By default, lists, dictionaries, and objects are iterated over. For custom iteration
    options, see :class:`sc.IterObj() <IterObj>`.
    
    Note: there are three different output possibilities, depending on the keywords:
        
        - ``inplace=False``, ``copy=False`` (default): collate the output of the function into a flat dictionary, with keys corresponding to each node of the project
        - ``inplace=True``, ``copy=False``: modify the actual object in-place, such that the original object is modified
        - ``inplace=True``, ``copy=True``: make a deep copy of the object, modify that object, and return it (the original is unchanged)
    
    Args:
        obj (any): the object to iterate over
        func (function): the function to apply; if None, return a dictionary of all leaf nodes in the object
        inplace (bool): whether to modify the object in place (else, collate the output of the functions)
        copy (bool): if modifying an object in place, whether to make a copy first
        leaf (bool): whether to apply the function only to leaf nodes of the object
        atomic (list): a list of known classes to treat as atomic (do not descend into); if 'default', use defaults (e.g. ``np.array``, ``pd.DataFrame``)
        verbose (bool): whether to print progress.
        _trace (list): used internally for recursion
        _output (list): used internally for recursion
        *args (list): passed to func()
        **kwargs (dict): passed to func()
    
    **Examples**::
        
        data = dict(a=dict(x=[1,2,3], y=[4,5,6]), b=dict(foo='string', bar='other_string'))
        
        # Search through an object
        def check_int(obj):
            return isinstance(obj, int)
    
        out = sc.iterobj(data, check_type)
        print(out)
        
        
        # Modify in place -- collapse mutliple short lines into one
        def collapse(obj, maxlen):
            string = str(obj)
            if len(string) < maxlen:
                return string
            else:
                return obj

        sc.printjson(data)
        sc.iterobj(data, collapse, inplace=True, maxlen=10) # Note passing of keyword argument to function
        sc.printjson(data)
        
    | *New in version 3.0.0.*
    | *New in version 3.1.0:* default ``func``, renamed "twigs_only" to "leaf", "atomic" keyword
    | *New in version 3.1.2:* ``copy`` defaults to ``False``; refactored into class
    """
    io = IterObj(obj=obj, func=func, inplace=inplace, copy=copy, leaf=leaf, atomic=atomic, verbose=verbose, 
            _trace=_trace, _output=_output, *args, **kwargs) # Create the object
    out = io.iterate() # Iterate
    return out


def mergenested(dict1, dict2, die=False, verbose=False, _path=None):
    """
    Merge different nested dictionaries

    See sc.makenested() for full documentation.

    Adapted from https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge
    """
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
                pass # same leaf value # pragma: no cover
            else:
                errormsg = f'Warning! Conflict at {keypath}: {a[key]} vs. {b[key]}'
                if die: # pragma: no cover
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
    
    Args:
        nesteddict (dict): the dictionary to flatten
        sep (str): the separator used to separate keys

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



##############################################################################
#%% Search and equality operators
##############################################################################


__all__ += ['search', 'Equal', 'equal']

# Define a custom "None" value to allow searching for actual None values
_None = '<sc_nested_custom_None>' # This should not be equal to any other value the user could supply
def search(obj, query=_None, key=_None, value=_None, aslist=True, method='exact', 
           return_values=False, verbose=False, _trace=None, **kwargs):
    """
    Find a key/attribute or value within a list, dictionary or object.

    This function facilitates finding nested key(s) or attributes within an object,
    by searching recursively through keys or attributes.

    Args:
        obj (any): A dict, list, or object
        query (any): The key or value to search for (or a function); equivalent to setting both ``key`` and ``value``
        key (any): The key to search for
        value (any): The value to search for
        aslist (bool): return entries as a list (else, return as a string)
        method (str): choose how to check for matches: 'exact' (test equality), 'partial' (partial/lowercase string match), or 'regex' (treat as a regex expression)
        return_values (bool): return matching values as well as keys
        verbose (bool): whether to print details of the search
        _trace: Not for user input - internal variable used for recursion
        kwargs (dict): passed to :func:`sc.iterobj() <iterobj>`

    Returns:
        A list of matching attributes. The items in the list are the Python
        strings (or lists) used to access the attribute (via attribute, dict, 
        or listindexing).

    **Examples**::

        # Create a nested dictionary
        nested = {'a':{'foo':1, 'bar':['moat', 'goat']}, 'b':{'car':3, 'cat':[1,2,4,8]}}
        
        # Find keys
        keymatches = sc.search(nested, 'bar', aslist=False)
        
        # Find values
        val = 4
        valmatches = sc.search(nested, value=val, aslist=True) # Returns  [['b', 'cat', 2]]
        assert sc.getnested(nested, valmatches[0]) == val # Get from the original nested object
        
        # Find values with a function
        def find(v):
            return True if isinstance(v, int) and v >= 3 else False
            
        found = sc.search(nested, value=find)
        
        # Find partial or regex matches
        found = sc.search(nested, key='oat', method='partial') # Search keys only
        keys,vals = sc.search(nested, '^.ar', method='regex', return_values=True, verbose=True)
    
    | *New in version 3.0.0:* ability to search for values as well as keys/attributes; "aslist" argument
    | *New in version 3.1.0:* "query", "method", and "verbose" keywords; improved searching for lists
    """
    
    # Collect keywords that won't change, for later use in the recursion
    kw = dict(method=method, verbose=verbose, aslist=aslist)
    
    def check_match(source, target, method):
        """ Check if there is a match between the "source" and "target" """
        if source != _None and target != _None: # See above for definition; a source and target were supplied
            if callable(target):
                match = target(source)
            elif method == 'exact':
                match = target == source
            elif method in [str, 'string', 'partial']:
                match = str(target).lower() in str(source).lower()
            elif method == 'regex':
                match = bool(re.match(str(target), str(source)))
            else: # pragma: no cover
                errormsg = f'Could not understand method "{method}": must be "exact", "string", or "regex"'
                raise ValueError(errormsg)
        else: # No target was supplied, return no match
            match = False
        
        return match
    
    # Handle query
    if query != _None:
        if key != _None or value != _None: # pragma: no cover
            errormsg = '"query" cannot be used with "key" or "value"; it is a shortcut to set both'
            raise ValueError(errormsg)
        key = query
        value = query

    # Look for matches
    matches = []

    # Determine object type
    itertype = check_iter_type(obj, **kwargs)
    if itertype == 'dict':
        o = obj
    elif itertype == 'list':
        o = {k:v for k,v in zip(list(range(len(obj))), obj)}
    elif itertype == 'object':
        o = obj.__dict__
    else:
        if verbose > 1: print(f'  For trace="{_trace}", cannot descend into "{type(obj)}"')
        return matches
    
    # Initialize the trace if it doesn't exist
    if _trace is None:
        if aslist:
            _trace = []
        else:
            _trace = ''
        
    # Iterate over the items
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
        
        # Sanitize object value
        if method in ['partial', 'regex'] and check_iter_type(v, **kwargs): # We want to exclude values that can be descended into if we're doing string matching
            objvalue = _None
        else:
            objvalue = v
        
        # Sanitize object key
        if key != _None and itertype == 'list': # "Keys" don't make sense for lists, so just duplicate values
            objkey = objvalue
        else:
            objkey = k

        # Actually check for matches
        for source,target in [[objkey, key], [objvalue, value]]:
            if check_match(source, target, method=method):
                if trace not in matches:
                    matches.append(trace)
        
        if verbose:
            nmatches = len(matches)
            msg = f'Checking trace="{trace}" for key="{key}", value="{value}" using "{method}": '
            if nmatches:
                msg += f'found {len(matches)} matches'
            else:
                msg += 'no matches'
            print(msg)

        # Continue searching recursively, and avoid duplication
        newmatches = search(o[k], key=key, value=value, _trace=trace, **kw)
        for newmatch in newmatches:
            if newmatch not in matches:
                matches.append(newmatch)
    
    # Tidy up
    if return_values:
        values = []
        for match in matches:
            value = getnested(obj, match)
            values.append(value)
        return matches, values
    else:
        return matches


class Equal(scu.prettyobj):
    
    # Define known special cases for equality checking
    special_cases = (float,) + _atomic_classes
    valid_methods = [None, 'eq', 'pickle', 'json', 'str']
    
    
    def __init__(self, obj, obj2, *args, method=None, detailed=False, equal_nan=True, 
                 leaf=False, verbose=None, compare=True, die=False, **kwargs):
        """
        Compare equality between two arbitrary objects -- see :func:`sc.equal() <equal>` for full documentation.

        Args:
            obj, obj2, etc: see :func:`sc.equal() <equal>`
            compare (bool): whether to perform the comparison on object creation

        *New in version 3.1.0.*
        """
        from . import sc_odict as sco # To avoid circular import
        
        # Set properties
        self.objs = [obj, obj2] + list(args) # All objects for comparison
        self.method = method
        self.detailed = detailed
        self.equal_nan = equal_nan
        self.verbose = verbose
        self.die = die
        self.kwargs = scu.mergedicts(kwargs, dict(leaf=leaf))
        self.check_method() # Check that the method is valid
        
        # Derived results
        self.walked = False # Whether the objects have already been walked
        self.compared = False # Whether the objects have already been compared
        self.dicts = [] # Object dictionaries
        self.results = sco.objdict() # Detailed output, 1D dict
        self.fullresults = sco.objdict() # Detailed output, 2D dict
        self.exceptions = sco.objdict() # Store any exceptions encountered
        self.eq = None # Final value to be populated
        
        # Run the comparison if requested
        if compare:
            self.walk()
            self.compare()
            self.to_df()
        return
    
    
    @property
    def n(self):
        """ Find out how many objects are being compared """
        return len(self.objs)

    @property
    def base(self):
        """ Get the base object """
        return self.objs[0]

    @property
    def others(self):
        """ Get the other objects """
        return self.objs[1:]

    @property
    def bdict(self):
        """ Get the base dictionary """
        return self.dicts[0] if len(self.dicts) else None
    
    @property
    def odicts(self):
        """ Get the other dictionaries """
        return self.dicts[1:]
    
    
    def check_method(self):
        """ Check that a valid method is supplied """
        if self.method is None:
            self.method = ['eq', 'pickle'] # Define the default method sequence to try
        self.method = scu.tolist(self.method)
        assert len(self.method), 'No methods supplied'
        for method in self.method:
            if method not in self.valid_methods and not callable(method): # pragma: no cover
                errormsg = f'Method "{method}" not recognized: must be one of {scu.strjoin(self.valid_methods)}'
                raise ValueError(errormsg)
    
    
    def get_method(self, method=None):
        """ Use the method if supplied, else use the default one """
        if method is None:
            method = self.method[0] # Use default method if none provided
        return method
    
    
    def walk(self):
        """ Use :func:`sc.iterobj() <iterobj>` to convert the objects into dictionaries """
        
        for obj in self.objs:
            self.dicts.append(iterobj(obj, **self.kwargs))
        self.walked = True
        if self.verbose:
            nkeystr = scu.strjoin([len(d) for d in self.dicts])
            print(f'Walked {self.n} objects with {nkeystr} keys respectively')
        return
    
    
    def convert(self, obj, method=None):
        """ Convert an object to the right type prior to comparing """
        
        method = self.get_method(method)
        
        # Do the conversion
        if method == 'eq':
            out = obj
        elif method == 'pickle':
            out = pkl.dumps(obj)
        elif method == 'json':
            from . import sc_fileio as scf # To avoid circular import
            out = scf.jsonpickle(obj)
        elif method == 'str':
            out = str(obj)
        elif callable(method):
            out = method(obj)
        else: # pragma: no cover
            errormsg = f'Method {method} not recognized'
            raise ValueError(errormsg)
        
        return out
    

    def compare_special(self, obj, obj2):
        """ Do special comparisons for known objects where == doesn't work """
        from . import sc_math as scm # To avoid circular import
        
        # For floats, check for NaN equality
        if isinstance(obj, float):
            if not np.isnan(obj) or not np.isnan(obj2) or not self.equal_nan: # Either they're not NaNs or we're not counting NaNs as equal
                eq = obj == obj2 # Do normal comparison
            else: # They are both NaNs and equal_nan is True
                eq = True
        
        # For numpy arrays, must use something to handle NaNs
        elif isinstance(obj, (np.ndarray, pd.Series, pd.core.indexes.base.Index)):
            eq = scm.nanequal(obj, obj2, scalar=True, equal_nan=self.equal_nan)
    
        # For dataframes, use Sciris
        elif isinstance(obj, pd.DataFrame):
            from . import sc_dataframe as scd # To avoid circular import
            eq = scd.dataframe.equal(obj, obj2, equal_nan=self.equal_nan)
        
        else: # pragma: no cover
            errormsg = f'Not able to handle object of {type(obj)}'
            raise TypeError(errormsg)

        return eq
    
    
    @staticmethod
    def keytostr(k, ind='', sep='.'):
        """ Helper method to convert a key to a "trace" for printing """
        out = f'<obj{str(ind)}>{sep}{scu.strjoin(k, sep=sep)}'
        return out
    
    
    @staticmethod
    def is_subkey(ckey, key):
        if len(key) <= len(ckey):
            return False
        else:
            return key[:len(ckey)] == ckey
        
    
    def compare(self):
        """ Perform the comparison """
        
        # Walk the objects if not already walked
        if not self.walked: # pragma: no cover
            self.walk()
        
        btree = self.bdict.enumitems() # Get the full object tree for the base object
        bkeys = set(self.bdict.keys()) # Get the base keys (object structure)
        while btree:
            i,key,baseobj = btree.pop(0) # Get the index, key, and base object
            eqs = [] # Store equality across all objects
            for j,otree in enumerate(self.odicts): # Iterate over other object trees
                
                # Check if the keys don't match, in which case objects differ
                eq = True
                if key == 'root':
                    okeys = set(otree.keys())
                    eq = bkeys == okeys
                    if eq is False and self.verbose: # pragma: no cover
                        print(f'Objects have different structures: {bkeys ^ okeys}') # Use XOR operator
                
                # If key not present, false by default
                if key not in otree:
                    eq = False
                
                # If keys match, proceed
                if eq:
                    methods = scu.dcp(self.method) # Copy the methods to try one by one
                    compared = False # Check if comparison succeeded
                    otherobj = otree[key] # Get the other object
                    
                    # Convert the objects
                    while len(methods) and not compared:
                        method = methods.pop(0)
                        bconv = self.convert(baseobj, method)
                        oconv = self.convert(otherobj, method)
                        
                        # Actually check equality -- can be True, False, or None
                        if type(bconv) != type(oconv):
                            eq = False # Unlike types are always not equal
                            compared = True
                        elif isinstance(bconv, self.special_cases):
                            eq = self.compare_special(bconv, oconv) # Compare known exceptions
                            compared = True
                        else:
                            try:
                                eq = (bconv == oconv) # Main use case: do the comparison!
                                eq = bool(eq) # Ensure it's true or false
                                compared = True # Comparison succeeded, break the loop
                            except Exception as E: # Store exceptions if encountered
                                eq = None
                                self.exceptions[key] = E
                                if self.verbose:
                                    print(f'Exception encountered on "{self.keytostr(key, j+1)}" ({type(bconv)}) with method "{method}": {E}')
                    
                    # All methods failed, check that the equality isn't defined
                    if not compared:
                        assert eq is None
                
                # Append the result
                eqs.append(eq)
                if self.verbose:
                    print(f'Item {i+1}/{len(self.odicts)} ({j+2}/{self.n}) "{self.keytostr(key, j+1)}": {eq}')
                
            # Store the results, and break if any equalities are found unless we're doing detailed
            has_none = None in eqs
            has_false = False in eqs
            result = None if has_none else all(eqs)
            self.fullresults[key] = eqs
            self.results[key] = result
            if not self.detailed and has_false: # Don't keep going unless needed
                if self.verbose: # pragma: no cover
                    print('Objects are not equal and detailed=False, breaking')
                break
        
            # Check for matches and avoid recursing further
            if not self.detailed and result is True:
                origlen = len(btree)
                btree = [ikv for ikv in btree if not self.is_subkey(key, ikv[1])]
                skipped = origlen - len(btree)
                if self.verbose:
                    print(f'Object {self.keytostr(key)} are equal, skipping {skipped} sub-objects')
            
        # Tidy up
        self.eq = all([v for v in self.results.values() if v is not None])
        if self.verbose:
            self.check_exceptions() # Check if any exceptions were encountered
        self.compared = True
            
        return self
    
    
    def check_exceptions(self):
        """ Check if any exceptions were encountered during comparison """
        if len(self.exceptions):
            string = 'The following exceptions were encountered:\n'
            for i,k,exc in self.exceptions.enumitems():
                string += f'{i}. {self.keytostr(k)}: {str(exc)}\n'
            print(string)
        return
    
    
    def to_df(self):
        """ Convert the detailed results dictionary to a dataframe """
        from . import sc_dataframe as scd # To avoid circular import
        
        # Ensure they've been compared
        if not self.compared: # pragma: no cover
            self.compare()
            
        # Make dataframe
        columns = [f'obj0_obj{i+1}' for i in range(self.n-1)]
        df = scd.dataframe.from_dict(scu.dcp(self.fullresults), orient='index', columns=columns)
        df['equal'] = df.all(axis=1)
        self.df = df
        return df
        
    

def equal(obj, obj2, *args, method=None, detailed=False, equal_nan=True, leaf=False, verbose=None, die=False, **kwargs):
    """
    Compare equality between two arbitrary objects
    
    There is no universal way to check equality between objects in Python. Some
    objects define their own equals method which may not evaluate to true/false
    (e.g., Numpy arrays and pandas dataframes). For others it may be undefined.
    For this reasons, different ways of checking equality may give different results
    in edge cases. The available methods are:
        
        - ``'eq'``: uses the objects' built-in ``__eq__()`` methods (most accurate, but most likely to fail)
        - ``'pickle'``: converts the object to a binary pickle (most robust)
        - ``'json'``: converts the object to a JSON via ``jsonpickle`` (gives most detailed object structure, but can be lossy)
        - ``'str'``: converts the object to its string representation (least amount of detail)
    
    By default, 'eq' is tried first, and if that raises an exception, 'pickle' is tried.
    
    Args:
        obj (any): the first object to compare
        obj2 (any): the second object to compare
        args (list): additional objects to compare
        method (str): see above
        detailed (bool): whether to compute a detailed comparison of the objects, and return a dataframe of the results
        equal_nan (bool): whether matching ``np.nan`` should compare as true (default True; NB, False not guaranteed to work with ``method='pickle'`` or ``'str'``, which includes the default; True not guaranteed to work with ``method='json'``)
        leaf (bool): if True, only compare the object's leaf nodes (those with no children); otherwise, compare everything
        verbose (bool): level of detail to print
        die (bool): whether to raise an exception if an error is encountered (else return False)
        kwargs (dict): passed to :func:`sc.iterobj() <iterobj>`
        
    **Examples**::
        
        o1 = dict(
            a = [1,2,3],
            b = np.array([4,5,6]),
            c = dict(
                df = sc.dataframe(q=[sc.date('2022-02-02'), sc.date('2023-02-02')])
            )
        )
        
        # Identical object
        o2 = sc.dcp(o1)
        
        # Non-identical object
        o3 = sc.dcp(o1)
        o3['b'][2] = 8
        
        sc.equal(o1, o2) # Returns True
        sc.equal(o1, o3) # Returns False
        e = sc.Equal(o1, o2, o3, detailed=True) # Create an object
        e.df.disp() # Show results as a dataframe
        
    *New in version 3.1.0.*
    """
    e = Equal(obj, obj2, *args, method=method, detailed=detailed, equal_nan=equal_nan, leaf=leaf, verbose=verbose, die=die, **kwargs)
    if detailed:
        return e.df
    else:
        return e.eq
