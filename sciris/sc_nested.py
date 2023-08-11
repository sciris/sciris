'''
Functions for working on nested (multi-level) dictionaries and objects.

Highlights:
    - :func:`sc.getnested() <getnested>`: get a value from a highly nested dictionary
    - :func:`sc.search() <search>`: find a value in a nested object
    - :func:`sc.equal() <equal>`: check complex objects for equality
'''

import re
import itertools
import pickle as pkl
from functools import reduce, partial
import numpy as np
import pandas as pd
from . import sc_utils as scu


##############################################################################
#%% Nested dict and object functions
##############################################################################


__all__ = ['getnested', 'setnested', 'makenested', 'iternested', 'iterobj',
           'mergenested', 'flattendict', 'nestedloop']


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


def check_iter_type(obj, check_array=False):
    ''' Helper function to determine if an object is a dict, list, or neither '''
    if isinstance(obj, dict):
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
    

def get_from_obj(ndict, key, safe=False):
    '''
    Get an item from a dict, list, or object by key
    
    Args:
        ndict (dict/list/obj): the object to get from
        key (any): the key to get
        safe (bool): whether to return None if the key is not found (default False)
    '''
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


def getnested(nested, keylist, safe=False):
    '''
    Get the value for the given list of keys
    
    Args:
        nested (any): the nested object (dict, list, or object) to get from
        keylist (list): the list of keys
        safe (bool): whether to return None if the key is not found
    
    **Example**::

        sc.getnested(foo, ['a','b']) # Gets foo['a']['b']

    See :func:`sc.makenested() <makenested>` for full documentation.
    '''
    get = partial(get_from_obj, safe=safe)
    nested = reduce(get, keylist, nested)
    return nested


def setnested(nested, keylist, value, force=True):
    '''
    Set the value for the given list of keys
    
    Args:
        nested (any): the nested object (dict, list, or object) to modify
        keylist (list): the list of keys to use
        value (any): the value to set
        force (bool): whether to create the keys if they don't exist (NB: only works for dictionaries)
    
    **Example**::

        sc.setnested(foo, ['a','b'], 3) # Sets foo['a']['b'] = 3

    See :func:`sc.makenested() <makenested>` for full documentation.
    '''
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
    '''
    Return a list of all the twigs in the current dictionary
    
    Args:
        nesteddict (dict): the dictionary
    
    **Example**::

        twigs = sc.iternested(foo)

    See :func:`sc.makenested() <makenested>` for full documentation.
    '''
    if _previous is None:
        _previous = []
    output = []
    for k in nesteddict.items():
        if isinstance(k[1], dict):
            output += iternested(k[1], _previous+[k[0]]) # Need to add these at the first level
        else:
            output.append(_previous+[k[0]])
    return output


def iteritems(obj, itertype):
    ''' Return an iterator over items in this object -- for internal use '''
    if itertype == 'dict':
        return obj.items()
    elif itertype == 'list':
        return enumerate(obj)
    elif itertype == 'object':
        return obj.__dict__.items()

def getitem(obj, key, itertype):
    ''' Get the value for the item -- for internal use '''
    if itertype in ['dict', 'list']:
        return obj[key]
    elif itertype == 'object':
        return obj.__dict__[key]

def setitem(obj, key, value, itertype):
    ''' Set the value for the item -- for internal use '''
    if itertype in ['dict', 'list']:
        obj[key] = value
    elif itertype == 'object':
        obj.__dict__[key] = value
    return


def iterobj(obj, func=None, inplace=False, copy=True, twigs_only=False, verbose=False, 
            _trace=None, _output=None, *args, **kwargs):
    '''
    Iterate over an object and apply a function to each twig.
    
    Can modify an object in-place, or return a value. See also :func:`sc.search() <search>`
    for a function to search through complex objects.
    
    Args:
        obj (any): the object to iterate over
        func (function): the function to apply; if None, return a dictionary of all twigs in the object
        inplace (bool): whether to modify the object in place (else, collate the output of the functions)
        copy (bool): if modifying an object in place, whether to make a copy first
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
        data = sc.iterobj(data, collapse, inplace=True)
        sc.printjson(data)
        
    | *New in version 3.0.0.*
    | *New in version 3.1.0:* default ``func``, improved ``twigs_only``
    '''
    from . import sc_odict as sco # To avoid circular import
    
    if func is None:
        func = lambda obj: obj
        
    # Set the trace and output if needed
    if _trace is None:
        _trace = []
        if inplace and copy: # Only need to copy once
            obj = scu.dcp(obj)
    if _output is None:
        _output = sco.objdict()
        if not inplace and not twigs_only:
            _output['root'] = func(obj, *args, **kwargs)
    
    itertype = check_iter_type(obj)
    
    # Next, check if we need to iterate
    if itertype:
        for key,subobj in iteritems(obj, itertype):
            trace = _trace + [key]
            newobj = subobj
            subitertype = check_iter_type(subobj)
            if verbose:
                print(f'Working on {trace}, {twigs_only}, {subitertype}')
            if not (twigs_only and subitertype):
                newobj = func(subobj, *args, **kwargs)
                if inplace:
                    setitem(obj, key, newobj, itertype)
                else:
                    _output[tuple(trace)] = newobj
            iterobj(getitem(obj, key, itertype), func, inplace=inplace, twigs_only=twigs_only,  # Run recursively
                    verbose=verbose, _trace=trace, _output=_output, *args, **kwargs)
        
    if inplace:
        newobj = func(obj, *args, **kwargs) # Set at the root level
        return newobj
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


__all__ += ['search', 'Equality', 'equal']

# Define a custom "None" value to allow searching for actual None values
_None = '<sc_nested_custom_None>' # This should not be equal to any other value the user could supply
def search(obj, query=_None, key=_None, value=_None, aslist=True, method='exact', 
           return_values=False, verbose=False, _trace=None):
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
    
    *New in version 3.0.0:* ability to search for values as well as keys/attributes; "aslist" argument
    *New in version 3.1.0:* "query", "method", and "verbose" keywords; improved searching for lists
    """
    
    # Collect keywords that won't change, for later use in the recursion
    kw = dict(method=method, verbose=verbose, aslist=aslist)
    
    def check_match(source, target, method):
        ''' Check if there is a match between the "source" and "target" '''
        if source != _None and target != _None: # See above for definition; a source and target were supplied
            if callable(target):
                match = target(source)
            elif method == 'exact':
                match = target == source
            elif method in [str, 'string', 'partial']:
                match = str(target).lower() in str(source).lower()
            elif method == 'regex':
                match = bool(re.match(str(target), str(source)))
            else:
                errormsg = f'Could not understand method "{method}": must be "exact", "string", or "regex"'
                raise ValueError(errormsg)
        else: # No target was supplied, return no match
            match = False
        
        return match
    
    # Handle query
    if query != _None:
        if key != _None or value != _None:
            errormsg = '"query" cannot be used with "key" or "value"; it is a shortcut to set both'
            raise ValueError(errormsg)
        key = query
        value = query

    # Look for matches
    matches = []

    # Determine object type
    itertype = check_iter_type(obj)
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
        if method in ['partial', 'regex'] and check_iter_type(v): # We want to exclude values that can be descended into if we're doing string matching
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


class Equality(scu.prettyobj):
    
    # Define known special cases for equality checking
    special_cases = (np.ndarray, pd.Series, pd.DataFrame)
    
    def __init__(self, obj, obj2, *args, method='json', detailed=False, 
                 equal_nan=True, verbose=None, die=False):
        '''
        Compare equality between two arbitrary objects -- see :func:`sc.equal() <equal>` for full documentation.

        *New in version 3.1.0.*
        '''
        from . import sc_odict as sco # To avoid circular import
        
        # Validate
        if method not in ['json', 'pickle', 'eq']:
            errormsg = f'Method "{self.method}" not recognized: must be json, pickle, or eq'
            raise ValueError(errormsg)
        
        # Set properties
        self.orig_base = obj
        self.orig_others = [obj2] + list(args)
        self.method = method
        self.detailed = detailed
        self.equal_nan = equal_nan
        self.verbose = verbose
        self.die = die
        
        # Derived results
        self.conv_base = None # Base object after conversion
        self.conv_others = [] # Other objects after conversion
        self.dict_base = None # Base dictionary
        self.dict_others = [] # Other dictionaries
        self.checked = [] # Keys that have been checked already
        self.eq = None # Final value to be populated
        self.results = sco.objdict() # Detailed output, 1D dict
        self.fullresults = sco.objdict() # Detailed output, 2D dict
        self.exceptions = sco.objdict() # Store any exceptions encountered
        self.converted = False # Whether the objects have already been converted
        return
    
    @property
    def n(self):
        ''' Find out how many objects are being compared '''
        return len(self.orig_others)
    
    
    def convert(self):
        ''' Convert the objects into the right format '''
        from . import sc_fileio as scf # To avoid circular import
        
        # Define the mapping
        mapping = dict(
            json   = scf.jsonpickle,
            pickle = pkl.dumps,
            eq     = lambda obj: obj,
        )
        conv_func = mapping[self.method]
        
        # Do the mapping
        self.conv_base = conv_func(self.orig_base)
        self.dict_base = iterobj(self.conv_base)
        for other in self.orig_others:
            conv_other = conv_func(other)
            self.conv_others.append(conv_other)
            self.dict_others.append(iterobj(conv_other))
        
        self.converted = True
                
        return
    

    def compare_special(self, obj, obj2):
        ''' Do special comparisons for known objects where == doesn't work '''
        from . import sc_dataframe as scd 
        
        # For numpy arrays, must use np.array_equals()
        if isinstance(obj, np.ndarray):
            eq = np.array_equal(obj, obj2, equal_nan=self.equal_nan)
    
        # For series and dataframes, use equals()
        elif isinstance(obj, pd.Series):
            eq = obj.equals(obj2)
        elif isinstance(obj, pd.DataFrame):
            eq = scd.dataframe(obj).equals(scd.dataframe(obj2))
        
        else:
            errormsg = f'Not able to handle object of {type(obj)}'
            raise TypeError(errormsg)
        
        return eq
    
    
    @staticmethod
    def keytostr(k, ind='', sep='.'):
        ''' Helper method to convert a key to a "trace" for printing '''
        out = f'<obj{str(ind)}>{sep}{scu.strjoin(k, sep=sep)}'
        return out
    
    
    @staticmethod
    def is_subkey(ckey, key):
        if len(key) <= len(ckey):
            return False
        else:
            return key[:len(ckey)] == ckey
        
    
    def compare(self):
        ''' Perform the comparison '''
        
        if not self.converted:
            self.convert()
        
        to_check = self.dict_base.enumitems()
        while to_check:
            i,k,v = to_check.pop(0)
            eqs = []
            for j,od in enumerate(self.dict_others):
                
                # If keys don't match, objects differ
                eq = True
                if k == 'root':
                    bkeys = set(self.dict_base.keys())
                    okeys = set(od.keys())
                    eq = bkeys == okeys
                    if eq is False and self.verbose:
                        print(f'Objects have different structures: {bkeys ^ okeys}') # Use XOR operator
                
                # It's not root or the keys match, proceed
                if eq:
                
                    # If not present, false by default
                    if k not in od:
                        eq = False
                    else:
                        ov = od[k]
                    
                        # Actually check equality -- can be True, False, or None
                        with scu.tryexcept(die=self.die, verbose=False) as te:
                            if type(v) != type(ov):
                                eq = False # Unlike types are always not equal
                            elif isinstance(v, self.special_cases):
                                eq == self.compare_special(v, ov) # Compare known exceptions
                            else:
                                eq = (v == ov) # Main use case: do the comparison!
                            eq = bool(eq) # Ensure it's true or false
                        if te.died: # Store exceptions if encountered
                            eq = None
                            exc = te.exception
                            self.exceptions[k] = exc
                            if self.verbose:
                                print(f'Exception encountered on "{self.keytostr(k, j+1)}" ({type(v)}): {exc}')
                
                # Append the result
                eqs.append(eq)
                if self.verbose:
                    print(f'Item {i+1}/{len(self.dict_base)} ({j+1}/{self.n}) "{self.keytostr(k, j+1)}": {eq}')
                
            # Store the results, and break if any equalities are found unless we're doing detailed
            has_none = None in eqs
            has_false = False in eqs
            all_true = None if has_none else all(eqs)
            self.fullresults[k] = eqs
            self.results[k] = all_true
            if not self.detailed and has_false: # Don't keep going unless needed
                if self.verbose:
                    print('Objects are not equal and detailed=False, breaking')
                break
            
            # Check for matches and avoid recursing further
            if all_true:
                self.checked.append(k)
                to_check = [ikv for ikv in to_check if not self.is_subkey(k, ikv[1])]
            
        # Tidy up
        self.eq = all([v for v in self.results.values() if v is not None])
        if self.verbose is not False:
            self.check_exceptions() # Check if any exceptions were encountered
            
        return self.eq
    
    
    def check_exceptions(self):
        ''' Check if any exceptions were encountered during comparison '''
        if len(self.exceptions):
            string = 'The following exceptions were encountered:\n'
            for i,k,exc in self.exceptions.enumitems():
                string += f'{i}. {self.keytostr(k)}: {str(exc)}\n'
            print(string)
        return
    
    
    def to_df(self):
        ''' Convert the detailed results dictionary to a dataframe '''
        from . import sc_dataframe as scd # To avoid circular import
        columns = [f'obj1_obj{i+2}' for i in range(self.n)]
        df = scd.dataframe.from_dict(scu.dcp(self.fullresults), orient='index', columns=columns)
        df['equal'] = df.all(axis=1)
        self.df = df
        return df
        
    

def equal(obj, obj2, *args, method='json', detailed=False, equal_nan=True, verbose=None, die=False):
    '''
    Compare equality between two arbitrary objects
    
    Args:
        obj (any): the first object to compare
        obj2 (any): the second object to compare
        args (list): additional objects to compare
        method (str): choose between 'json' (default; convert to JSON and then compare), 'pickle' (convert to binary pickle), or 'eq' (use ==)
        detailed (bool): whether to return a detailec comparison of the objects (else just true/false)
        equal_nan (bool): whether matching ``np.nan`` should compare as true (default yes)
        verbose (bool): level of detail to print
        die (bool): whether to raise an exception if an error is encountered (else return False)
        
    **Example**:
        
        o1 = dict(
            a = [1,2,3],
            b = np.array([4,5,6]),
            c = dict(
                df = pd.DataFrame(q=[sc.date('2022-02-02'), sc.date('2023-02-02')])
            )
        )
        
        o2 = sc.dcp(o1)
        
        o3 = sc.dcp(o1)
        o3['b'][2] = 8
        
        sc.equal(o1, o2) # Returns True
        sc.equal(o1, o3) # Returns False
        e = sc.Equality(o1, o2, o3, detailed=True) # Create an object
        e.to_df() # Convert to a dataframe
        
    *New in version 3.1.0.*
    '''
    equality = Equality(obj, obj2, *args, method=method, detailed=detailed, equal_nan=equal_nan, verbose=verbose, die=die)
    return equality.compare()
