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
import functools as ft
import collections as co
import numpy as np
import pandas as pd
import sciris as sc

# Define objects for which it doesn't make sense to descend further -- used here and sc.equal()
atomic_classes = [np.ndarray, pd.Series, pd.DataFrame, pd.core.indexes.base.Index]
atomic_with_tuple = atomic_classes + [tuple]

# Define a custom "None" value to allow searching for actual None values
_None = '<sc_nested_custom_None>' # This should not be equal to any other value the user could supply

def not_none(obj):
    """ Check if an object does not match "_None" (the special None value to allow None input) """
    return not isinstance(obj, str) or obj != _None


##############################################################################
#%% Nested dict and object functions
##############################################################################


__all__ = ['getnested', 'setnested', 'makenested', 'iternested', 'IterObj', 'iterobj',
           'mergenested', 'flattendict', 'nestedloop']


def makenested(obj, keylist=None, value=None, overwrite=True, generator=None, copy=False):
    """
    Make a nested object (such as a dictionary).

    Args:
        obj (any): the object to make the nested list in
        keylist (list): a list of keys (strings) of the path to make
        value (any): the value to set at the final key
        overwrite (bool): if True, overwrite a value even if it exists
        generator (class/func): the function used to create new levels of nesting (default: same as original object)
        copy (bool): if True, copy the object before modifying it

    Functions to get and set data from nested dictionaries (including objects).

    ``sc.getnested()`` will get the value for the given list of keys:

    >>> sc.getnested(foo, ['a','b'])

    ``sc.setnested`` will set the value for the given list of keys:

    >>> sc.setnested(foo, ['a','b'], 3)

    ``sc.makenested`` will recursively update a dictionary with the given list of keys:

    >>> sc.makenested(foo, ['a','b'])

    ``sc.iternested`` will return a list of all the twigs in the current dictionary:

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

    **Example 3**::

        foo = sc.makenested(sc.prettyobj(), ['level1', 'level2', 'level3'], 'done')
        assert foo.level1.level2.level3 == 'done'


    | *New in version 2014nov29.*
    | *New in version 3.2.0:* operate on arbitrary objects; "overwrite" defaults to True; returns object
    """
    if copy:
        obj = sc.dcp(obj)
    currentlevel = obj
    for i,key in enumerate(keylist[:-1]):
        if not check_in_obj(currentlevel, key):
            if generator is not None:
                gen_func = generator
            else:
                gen_func = currentlevel.__class__ # By default, generate new dicts of the same class as the most recent level
            new = gen_func() # Create a new dictionary
            set_in_obj(currentlevel, key, new)
        currentlevel = get_from_obj(currentlevel, key)

    # Set the value
    lastkey = keylist[-1]
    if overwrite or lastkey not in currentlevel:
        set_in_obj(currentlevel, lastkey, value)
    elif not overwrite and value is not None: # pragma: no cover
        errormsg = f'Not overwriting entry {keylist} since overwrite=False'
        raise ValueError(errormsg)
    return obj


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
        elif isinstance(obj, tuple):
            out = 'tuple'
        elif hasattr(obj, '__dict__'):
            out = 'object'
        elif check_array and isinstance(obj, np.ndarray):
            out = 'array'
        else:
            out = '' # Evaluates to false
    return out


def check_in_obj(parent, key):
    """
    Check to see if a given key is present in an object
    """
    itertype = check_iter_type(parent)
    if itertype == 'dict':
        out = key in parent.keys()
    elif itertype in ['list', 'tuple']:
        out = isinstance(key, int) and 0 <= key < len(parent)
    elif itertype == 'object':
        out = key in parent.__dict__.keys()
    else:
        errormsg = f'Cannot check for type "{type(parent)}", itertype "{itertype}"'
        raise Exception(errormsg)
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
    elif itertype in ['list', 'tuple']:
        out = ndict[key]
    elif itertype == 'object':
        out = getattr(ndict, key)
    else:
        out = None
    return out


def set_in_obj(parent, key, value):
    """ Set the value for the item """
    itertype = check_iter_type(parent)
    if itertype in ['dict', 'list']:
        parent[key] = value
    elif itertype == 'object':
        parent.__dict__[key] = value
    else:
        errormsg = f'Cannot set value for type "{type(parent)}", itertype "{itertype}"'
        raise Exception(errormsg)
    return


def flatten_traces(tupledict, sep='_'):
    """ Convert trace tuples to strings for easier reading """
    strdict = type(tupledict)() # Create new dictionary of the same type
    for key,val in tupledict.items():
        if isinstance(key, tuple):
            key = sep.join([str(k) for k in key])
        strdict[key] = val
    return strdict


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
    get = ft.partial(get_from_obj, safe=safe)
    nested = ft.reduce(get, keylist, nested)
    return nested


def setnested(nested, keylist, value, force=True):
    """
    Set the value for the given list of keys

    Args:
        nested (any): the nested object (dict, list, or object) to modify
        keylist (list): the list of keys to use
        value (any): the value to set
        force (bool): whether to create the keys if they don't exist

    **Example**::

        sc.setnested(foo, ['a','b'], 3) # Sets foo['a']['b'] = 3

    See :func:`sc.makenested() <makenested>` for full documentation.
    """
    if not isinstance(keylist, (list, tuple)): # Be careful not to wrap tuples in lists
        keylist = sc.tolist(keylist)
    parentkeys = keylist[:-1]
    if force and parentkeys:
        makenested(nested, parentkeys, overwrite=False)
    currentlevel = getnested(nested, parentkeys)
    set_in_obj(currentlevel, keylist[-1], value)
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


class IterObj:
    """
    Object iteration manager

    For arguments and usage documentation, see :func:`sc.iterobj() <iterobj>`.
    Use this class only if you want more control over how the object is iterated over.

    Class-specific args:
        iterate (bool): whether to do iteration upon object creation
        custom_type (func): a custom function for returning a string for a specific object type (should return None by default)
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
        print(all_data)

    | *New in version 3.1.2.*
    | *New in version 3.1.5:* "norecurse" argument; better handling of atomic classes
    | *New in version 3.1.6:* "depthfirst" argument; replace recursion with a queue; "to_df()" method
    | *New in version 3.2.1:* improved recursion handling; "disp()" method
    """
    def __init__(self, obj, func=None, inplace=False, copy=False, leaf=False, recursion=0, depthfirst=True,
                 atomic='default', skip=None, rootkey='root', verbose=False, iterate=True,
                 custom_type=None, custom_iter=None, custom_get=None, custom_set=None, *args, **kwargs):

        # Default arguments
        self.obj          = obj
        self.func         = func
        self.inplace      = inplace
        self.copy         = copy
        self.leaf         = leaf
        self.recursion    = recursion
        self.depthfirst   = depthfirst
        self.atomic       = atomic
        self.skip         = skip
        self.rootkey      = rootkey
        self.verbose      = verbose
        self.func_args    = args
        self.func_kw      = kwargs

        # Custom arguments
        self.custom_type = custom_type
        self.custom_iter = custom_iter
        self.custom_get  = custom_get
        self.custom_set  = custom_set

        # Attributes with initialization required
        self._trace     = []
        self._memo      = co.defaultdict(int)
        self.output     = sc.objdict()
        if self.func is None: # If no function provided, define a function that just returns the contents of the current node
            self.func = lambda obj: obj

        # Handle atomic classes
        base_atomic = []
        atomic_list = sc.tolist(self.atomic)
        if 'default' in atomic_list: # Handle objects to not descend into
            atomic_list.remove('default')
            base_atomic = atomic_with_tuple
        if 'default-tuple' in atomic_list:
            atomic_list.remove('default-tuple')
            base_atomic = atomic_classes
        self.atomic = tuple(base_atomic + atomic_list)

        # Handle objects to skip
        if isinstance(skip, dict):
            skip = sc.dcp(skip)
            skip_keys       = sc.tolist(skip.pop('keys', None))
            skip_ids        = sc.tolist(skip.pop('ids', None))
            skip_subclasses = sc.tolist(skip.pop('subclasses', None))
            skip_instances  = sc.tolist(skip.pop('instances', None))
            if len(skip):
                errormsg = f'Unrecognized skip keys {skip.keys()}: must be "keys", "ids", "subclasses", and/or "instances"'
                raise KeyError(errormsg)
        else:
            skip = sc.tolist(self.skip)
            skip_keys       = []
            skip_ids        = []
            skip_subclasses = []
            skip_instances  = [] # This isn't populated in list form
            for entry in skip:
                if isinstance(entry, int):
                    skip_ids.append(entry)
                elif isinstance(entry, type):
                    skip_subclasses.append(entry)
                elif isinstance(entry, str):
                    skip_keys.append(entry)
                else:
                    errormsg = f'Expecting skip entries to be keys, classes or object IDs, not {entry}'
                    raise TypeError(errormsg)

        self._skip_keys       = tuple(skip_keys)
        self._skip_ids        = tuple(skip_ids)
        self._skip_subclasses = tuple(skip_subclasses)
        self._skip_instances  = tuple(skip_instances)

        # Copy the object if needed
        if inplace and copy:
            self.obj = sc.dcp(obj)

        # Actually do the iteration
        if iterate:
            self.iterate()

        return

    def __repr__(self):
        """ Show the object """
        objstr = f'{type(self.obj)}'
        lenstr = f'{len(self)}' if len(self) else '<not parsed>'
        string = f'{self.__class__.__name__}(obj={objstr}, len={lenstr})'
        return string

    def __len__(self):
        """ Define the length as the length of the output dictionary """
        try:    return len(self.output)
        except: return None

    def disp(self):
        """ Display the full object """
        return sc.pr(self)

    def indent(self, string='', space='  '):
        """ Print, with output indented successively """
        if self.verbose:
            print(space*len(self._trace) + string)
        return

    def iteritems(self, parent, trace):
        """ Return an iterator over items in this object """
        itertype = self.check_iter_type(parent)
        out = None
        if self.custom_iter:
            out = self.custom_iter(parent)
        if out is None:
            if itertype == 'dict':
                out = parent.items()
            elif itertype in ['list', 'tuple']:
                out = enumerate(parent)
            elif itertype == 'object':
                out = parent.__dict__.items()
            else:
                out = {}.items() # Return nothing if not recognized
        if trace is not _None:
            out = list(out)
            for i in range(len(out)):
                out[i] = [parent, trace, *list(out[i])] # Prepend parent and trace to the arguments
        return out

    def getitem(self, key, parent):
        """ Get the value for the item """
        self.indent(f'Getting key "{key}"')
        itertype = self.check_iter_type(parent)
        if itertype in ['dict', 'list', 'tuple']:
            return parent[key]
        elif itertype == 'object':
            return parent.__dict__[key]
        elif self.custom_get:
            return self.custom_get(parent, key)
        else:
            return None

    def setitem(self, key, value, parent):
        """ Set the value for the item """
        itertype = self.check_iter_type(parent)
        self.indent(f'Setting key "{key}"')
        if itertype in ['dict', 'list']:
            parent[key] = value
        elif itertype == 'object':
            parent.__dict__[key] = value
        elif self.custom_set:
            self.custom_set(parent, key, value)
        elif itertype == 'tuple':
            errormsg = f'Trying to set key={key} to {value} in a tuple; not possible since tuples are immutable'
            raise TypeError(errormsg)
        return

    def check_iter_type(self, obj):
        """ Shortcut to check_iter_type() """
        return check_iter_type(obj, known=self.atomic, custom=self.custom_type)

    def check_proceed(self, key, subobj, newid):
        """ Check if we should continue or not """

        # If we've already parsed this object, don't parse it again if it's iterable
        memo_skip = False
        in_memo = (newid in self._memo) and (self._memo[newid] > self.recursion)
        if in_memo: # We only skip processing if we've both seen an object before and it's iterable
            if check_iter_type(subobj, known=self.atomic, custom=self.custom_type):
                memo_skip = True

        # Skip this object if we've been asked to
        key_skip = key in self._skip_keys
        id_skip = (newid in self._skip_ids)
        subclass_skip = issubclass(type(subobj), self._skip_subclasses)
        instance_skip = isinstance(subobj, self._skip_instances)

        # Finalize
        skips = [memo_skip, key_skip, id_skip, subclass_skip, instance_skip]
        proceed = not any(skips)

        if not proceed and self.verbose: # Just for debugging
            labels = ['memo', 'key', 'id', 'subclass', 'instance']
            pairs = [f'{label}_skip=True' for label,skip in zip(labels, skips) if skip]
            self.indent(f'Skipping "{key}" because {sc.strjoin(pairs)}')

        return proceed

    def process_obj(self, parent, trace, key, subobj, newid):
        """ Process a single object """
        self._memo[newid] += 1
        trace = trace + [key]
        subitertype = self.check_iter_type(subobj)
        self.indent(f'{len(self)} Trace {trace} | Type "{str(subitertype)}" | {type(subobj)}')
        if not (self.leaf and subitertype):
            newobj = self.func(subobj, *self.func_args, **self.func_kw)
            if self.inplace:
                self.setitem(key, newobj, parent=parent)
            else:
                self.output[tuple(trace)] = newobj
        return trace

    def iterate(self):
        """ Actually perform the iteration over the object """

        # Initialize the output for the root node
        if not self.inplace:
            self.output[self.rootkey] = self.func(self.obj, *self.func_args, **self.func_kw)

        # Initialize the memo with the current object
        self._memo[id(self.obj)] = 1

        # Iterate
        queue = co.deque(self.iteritems(self.obj, self._trace))
        while queue:
            parent,trace,key,subobj = queue.popleft()
            newid = id(subobj)
            proceed = self.check_proceed(key, subobj, newid)
            if proceed: # Actually descend into the object
                newtrace = self.process_obj(parent, trace, key, subobj, newid) # Process the object
                newitems = self.iteritems(subobj, newtrace)
                if self.depthfirst:
                    queue.extendleft(reversed(newitems)) # extendleft() swaps order, so swap back
                else:
                    queue.extend(newitems)

        # Finish up
        if self.inplace:
            newobj = self.func(self.obj, *self.func_args, **self.func_kw) # Set at the root level
            return newobj
        else:
            if (not self._trace) and (len(self.output)>1) and self.leaf: # We're at the top level, we have multiple entries, and only leaves are requested
                self.output.pop('root') # Remove "root" with leaf=True if it's not the only node
            return self.output

    def flatten_traces(self, sep='_', inplace=True):
        """ Flatten the traces """
        output = flatten_traces(self.output, sep=sep)
        if inplace:
            self.output = output
        return output

    def to_df(self, skip_root=True):
        """
        Convert the output dictionary to a dataframe.

        Args:
            skip_root (bool): if True (default), only include the object's subcomponents
        """
        if not len(self):
            errormsg = 'No output to convert to a dataframe: length is zero'
            raise ValueError(errormsg)
        trace = self.output.keys()
        depth = [(0 if tr==self.rootkey else len(tr)) for tr in trace] # The depth is the length of the tuple, except the special case of the root key
        value = self.output.values()
        if skip_root:
            trace = trace[1:]
            depth = depth[1:]
            value = value[1:]
        self.df = sc.dataframe(trace=trace, depth=depth, value=value)
        return self.df


def iterobj(obj, func=None, inplace=False, copy=False, leaf=False, recursion=0, depthfirst=True, atomic='default',
            skip=None, rootkey='root', verbose=False, flatten=False, to_df=False, *args, **kwargs):
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
        func (function): the function to apply; if None, return a flat dictionary of all nodes in the object
        inplace (bool): whether to modify the object in place (else, collate the output of the functions)
        copy (bool): if modifying an object in place, whether to make a copy first
        leaf (bool): whether to apply the function only to leaf nodes of the object
        recursion (int): number of recursive steps to allow, i.e. parsing the same objects multiple times (default 0)
        depthfirst (bool): whether to parse the object depth-first (default) or breadth-first
        atomic (list): a list of known classes to treat as atomic (do not descend into); if 'default', use defaults (e.g. tuple, ``np.array``, ``pd.DataFrame``); if 'default-tuple', use defaults except for tuples
        skip (list/dict): a list of objects to skip over entirely; can also be a dict with "keys", "ids", "subclasses", and/or "instances", which skip each of those
        rootkey (str): the key to list as the root of the object (default ``'root'``)
        verbose (bool): whether to print progress
        flatten (bool): whether to use flattened traces (single strings) rather than tuples
        to_df (bool): whether to return a dataframe of the output instead of a dictionary (not valid with inplace=True)
        *args (list): passed to func()
        **kwargs (dict): passed to func()

    **Examples**::

        data = dict(a=dict(x=[1,2,3], y=[4,5,6]), b=dict(foo='string', bar='other_string'))

        # Search through an object
        def check_int(obj):
            return isinstance(obj, int)

        out = sc.iterobj(data, check_int)
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
    | *New in version 3.1.0:* default ``func``, renamed "twigs_only" to "leaf", "atomic" argument
    | *New in version 3.1.2:* ``copy`` defaults to ``False``; refactored into class
    | *New in version 3.1.3:* "rootkey" argument
    | *New in version 3.1.5:* "recursion" argument; better handling of atomic classes
    | *New in version 3.1.6:* "skip", "depthfirst", "to_df", and "flatten" arguments
    """
    # Create the object
    io = IterObj(obj=obj, func=func, inplace=inplace, copy=copy, leaf=leaf, recursion=recursion, depthfirst=depthfirst,
                 atomic=atomic, skip=skip, rootkey=rootkey, verbose=verbose, iterate=False, *args, **kwargs)
    out = io.iterate() # Iterate

    if flatten:
        out = io.flatten_traces()
    if to_df:
        out = io.to_df()
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
        a = sc.dcp(dict1) # Otherwise, make a copy
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


def search(obj, query=_None, key=_None, value=_None, type=_None, method='exact', **kwargs):
    """
    Find a key/attribute or value within a list, dictionary or object.

    This function facilitates finding nested key(s) or attributes within an object,
    by searching recursively through keys or attributes. See :func:`sc.iterobj() <iterobj>`
    for more detail.

    Args:
        obj (any): A dict, list, or object
        query (any): The key or value to search for (or a function or a type); equivalent to setting both ``key`` and ``value``
        key (any): The key to search for
        value (any): The value to search for
        type (type): The type (or list of types) to match against (for values only)
        method (str): if the query is a string, choose how to check for matches: 'exact' (test equality), 'partial' (partial/lowercase string match), or 'regex' (treat as a regex expression)
        kwargs (dict): passed to :func:`sc.iterobj() <iterobj>`

    Returns:
        A dictionary of matching attributes; like :func:`sc.iterobj() <iterobj>`,
        but filtered to only include matches.

    **Examples**::

        # Create a nested dictionary
        nested = {'a':{'foo':1, 'bar':['moat', 'goat']}, 'b':{'car':3, 'cat':[1,2,4,8]}}

        # Find keys
        keymatches = sc.search(nested, 'bar', flatten=True)

        # Find values
        val = 4
        valmatches = sc.search(nested, value=val).keys()[0] # Returns  ('b', 'cat', 2)
        assert sc.getnested(nested, valmatches) == val # Get from the original nested object

        # Find values with a function
        def find(v):
            return True if isinstance(v, int) and v >= 3 else False

        found = sc.search(nested, value=find)

        # Find partial or regex matches
        found = sc.search(nested, value='oat', method='partial', leaf=True) # Search keys only
        keys,vals = sc.search(nested, '^.ar', method='regex', verbose=True)

    | *New in version 3.0.0:* ability to search for values as well as keys/attributes; "aslist" argument
    | *New in version 3.1.0:* "query", "method", and "verbose" keywords; improved searching for lists
    | *New in version 3.2.0:* allow type matching; removed "return_values"; renamed "aslist" to "flatten" (reversed)
    """

    def check_match(source, target):
        """ Check if there is a match between the "source" and "target" """
        if not_none(source) and not_none(target): # See above for definition of _None; a source and target were supplied
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
    if not_none(query):
        if not_none(key) or not_none(value): # pragma: no cover
            errormsg = '"query" cannot be used with "key" or "value"; it is a shortcut to set both'
            raise ValueError(errormsg)
        key = query
        value = query

    # Handle type
    if not_none(type):
        if not_none(key) or not_none(value): # pragma: no cover
            errormsg = '"type" cannot be used with "key" or "value"; replaces "value"'
            raise ValueError(errormsg)
        typetuple = tuple(sc.tolist(type))
        value = lambda source: isinstance(source, typetuple) # Define a lambda function for the matching


    # Parse the object tree
    flatten = kwargs.pop('flatten', False) # Don't flatten because that will disrupt the matching
    tree = iterobj(obj, **kwargs) # For key matching

    # Do the matching
    matches = []

    # Match keys
    if not_none(key):
        for k in tree.keys():
            if check_match(k[-1], key): # Only want the last key of the trace
                matches.append(k)

    # Match values (including types)
    if not_none(value):
        for k,v in tree.items():
            if check_match(v, value):
                matches.append(k)

    # Reassemble dict to maintain order
    out = sc.objdict({k:v for k,v in tree.items() if k in matches})

    if flatten:
        out = flatten_traces(out)

    return out



class Equal(sc.prettyobj):

    # Define known special cases for equality checking
    special_cases = tuple([float] + atomic_classes)
    valid_methods = [None, 'eq', 'pickle', 'json', 'str']


    def __init__(self, obj, obj2, *args, method=None, detailed=False, equal_nan=True,
                 leaf=False, union=True, verbose=None, compare=True, die=False, **kwargs):
        """
        Compare equality between two arbitrary objects -- see :func:`sc.equal() <equal>` for full documentation.

        Args:
            obj, obj2, etc: see :func:`sc.equal() <equal>`
            compare (bool): whether to perform the comparison on object creation

        *New in version 3.1.0.*
        """

        # Set properties
        self.objs = [obj, obj2] + list(args) # All objects for comparison
        self.method = method
        self.detailed = detailed
        self.missingstr = '<MISSING>'
        self.equal_nan = equal_nan
        self.union = union
        self.verbose = verbose
        self.die = die
        self.kwargs = sc.mergedicts(kwargs, dict(leaf=leaf))
        self.check_method() # Check that the method is valid

        # Derived results
        self.walked = False # Whether the objects have already been walked
        self.compared = False # Whether the objects have already been compared
        self.dicts = [] # Object dictionaries
        self.treekeys = None # The object keys to walk over
        self.results = sc.objdict() # Detailed output, 1D dict
        self.fullresults = sc.objdict() # Detailed output, 2D dict
        self.exceptions = sc.objdict() # Store any exceptions encountered
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
        self.method = sc.tolist(self.method)
        assert len(self.method), 'No methods supplied'
        for method in self.method:
            if method not in self.valid_methods and not callable(method): # pragma: no cover
                errormsg = f'Method "{method}" not recognized: must be one of {sc.strjoin(self.valid_methods)}'
                raise ValueError(errormsg)


    def get_method(self, method=None):
        """ Use the method if supplied, else use the default one """
        if method is None:
            method = self.method[0] # Use default method if none provided
        return method


    def walk(self):
        """ Use :func:`sc.iterobj() <iterobj>` to convert the objects into dictionaries """

        # Walk the objects
        for obj in self.objs:
            self.dicts.append(iterobj(obj, **self.kwargs))
        self.walked = True
        if self.verbose:
            nkeystr = sc.strjoin([len(d) for d in self.dicts])
            print(f'Walked {self.n} objects with {nkeystr} keys respectively')

        self.make_tree()
        return


    def make_tree(self):
        """ Determine the keys to iterate over """
        treekeys = list(self.bdict.keys()) # Start with the base keys

        if self.union:
            fullset = set()
            for odict in self.odicts:
                fullset = fullset.union(odict.keys())
            extras = fullset - set(treekeys)
            pos = 0
            if len(extras): # Shortcut if all the keys are the same
                for odict in self.odicts:
                    for key in odict.keys():
                        try:
                            pos = treekeys.index(key)
                        except ValueError:
                            treekeys.insert(pos+1, key)

        self.treekeys = treekeys

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
            out = sc.jsonpickle(obj)
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

        # For floats, check for NaN equality
        if isinstance(obj, float):
            if not np.isnan(obj) or not np.isnan(obj2) or not self.equal_nan: # Either they're not NaNs or we're not counting NaNs as equal
                eq = obj == obj2 # Do normal comparison
            else: # They are both NaNs and equal_nan is True
                eq = True

        # For numpy arrays, must use something to handle NaNs
        elif isinstance(obj, (np.ndarray, pd.Series, pd.core.indexes.base.Index)):
            eq = sc.nanequal(obj, obj2, scalar=True, equal_nan=self.equal_nan)

        # For dataframes, use Sciris
        elif isinstance(obj, pd.DataFrame):
            eq = sc.dataframe.equal(obj, obj2, equal_nan=self.equal_nan)

        else: # pragma: no cover
            errormsg = f'Not able to handle object of {type(obj)}'
            raise TypeError(errormsg)

        return eq


    @staticmethod
    def keytostr(k, ind='', sep='.'):
        """ Helper method to convert a key to a "trace" for printing """
        out = f'<obj{str(ind)}>{sep}{sc.strjoin(k, sep=sep)}'
        return out


    @staticmethod
    def is_subkey(ckey, key):
        if len(key) <= len(ckey):
            return False
        else:
            return key[:len(ckey)] == ckey


    def compare(self):
        """ Perform the comparison """

        def appendval(vals, obj):
            """ Append a value to the list of values for printing """
            if self.detailed > 1:
                try:    string = str(obj) # Convert to string since some objects can't be printed in a dataframe (e.g. another dataframe)
                except: string = f'Error showing {type(obj)}'
                vals += [string]
            return

        # Walk the objects if not already walked
        if not self.walked: # pragma: no cover
            self.walk()

        bkeys = set(self.bdict.keys()) # Get the base keys (object structure)
        for i,key in enumerate(self.treekeys):
            baseobj = self.bdict.get(key, self.missingstr)
            eqs = [] # Store equality across all objects
            vals = [] # Store values of each object
            appendval(vals, baseobj)
            for j,otree in enumerate(self.odicts): # Iterate over other object trees

                # Check if the keys don't match, in which case objects differ
                eq = True
                if key == 'root':
                    appendval(vals, otree['root'])
                    okeys = set(otree.keys())
                    eq = bkeys == okeys
                    if eq is False and self.verbose: # pragma: no cover
                        print(f'Objects have different structures: {bkeys ^ okeys}') # Use XOR operator

                # If key not present, false by default
                if key not in otree:
                    eq = False
                    appendval(vals, self.missingstr)

                # If keys match, proceed
                if eq:
                    methods = sc.dcp(self.method) # Copy the methods to try one by one
                    compared = False # Check if comparison succeeded
                    otherobj = otree[key] # Get the other object
                    if key != 'root': appendval(vals, otherobj)

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
                    print(f'Item {i+1}/{len(self.treekeys)} of object {j+1}/{self.n-1}: "{self.keytostr(key, j+1)}": {eq}')

            # Store the results, and break if any equalities are found unless we're doing detailed
            has_none = None in eqs
            has_false = False in eqs
            result = None if has_none else all(eqs)
            self.fullresults[key] = eqs + vals
            self.results[key] = result
            if not self.detailed and has_false: # Don't keep going unless needed
                if self.verbose: # pragma: no cover
                    print('Objects are not equal and detailed=False, breaking')
                break

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
        # Ensure they've been compared
        if not self.compared: # pragma: no cover
            self.compare()

        # Make dataframe
        columns = [f'obj0==obj{i+1}' for i in range(self.n-1)]
        if self.detailed>1: columns = columns + [f'val{i}' for i in range(self.n)]
        df = sc.dataframe.from_dict(sc.dcp(self.fullresults), orient='index', columns=columns)
        equal = df.iloc[:, :(self.n-1)].all(axis=1)
        df.insert(0, 'equal', equal)

        self.df = df
        return df



def equal(obj, obj2, *args, method=None, detailed=False, equal_nan=True, leaf=False,
          union=True, verbose=None, die=False, **kwargs):
    """
    Compare equality between two arbitrary objects

    This method parses two (or more) objects of any type (lists, dictionaries,
    custom classes, etc.) and determines whether or not they are equal. By default
    it returns true/false for whether or not the objects match, but it can also
    return a detailed comparison of exactly which attributes (or keys, etc) match
    or don't match between the two objects. It works by first parsing the entire
    object into "leaves" via :func:`sc.iterobj() <iterobj>`, and then comparing each
    "leaf" via one of the methods described below.

    There is no universal way to check equality between objects in Python. Some
    objects define their own equals method which may not evaluate to true/false
    (e.g., Numpy arrays and pandas dataframes). For others it may be undefined.
    For this reasons, different ways of checking equality may give different results
    in edge cases. The available methods are:

        - ``'eq'``: uses the objects' built-in ``__eq__()`` methods (most accurate, but most likely to fail)
        - ``'pickle'``: converts the object to a binary pickle (most robust)
        - ``'json'``: converts the object to a JSON via ``jsonpickle`` (gives most detailed object structure, but can be lossy)
        - ``'str'``: converts the object to its string representation (least amount of detail)
        - In addition, any custom function can be provided

    By default, 'eq' is tried first, and if that raises an exception, 'pickle' is tried (equivalent to ``method=['eq', 'pickle']``).

    Args:
        obj (any): the first object to compare
        obj2 (any): the second object to compare
        args (list): additional objects to compare
        method (str): see above
        detailed (int): whether to compute a detailed comparison of the objects, and return a dataframe of the results (if detailed=2, return the value of each object as well)
        equal_nan (bool): whether matching ``np.nan`` should compare as true (default True; NB, False not guaranteed to work with ``method='pickle'`` or ``'str'``, which includes the default; True not guaranteed to work with ``method='json'``)
        leaf (bool): if True, only compare the object's leaf nodes (those with no children); otherwise, compare everything
        union (bool): if True, construct the comparison tree as the union of the trees of each object (i.e., an extra attribute in one object will show up as an additional row in the comparison; otherwise rows correspond to the attributes of the first object)
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

    | *New in version 3.1.0.*
    | *New in version 3.1.3:* "union" argument; more detailed output
    """
    e = Equal(obj, obj2, *args, method=method, detailed=detailed, equal_nan=equal_nan, leaf=leaf, verbose=verbose, die=die, **kwargs)
    if detailed:
        return e.df
    else:
        return e.eq
