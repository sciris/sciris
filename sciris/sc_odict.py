"""
Dictionaries such as `sc.obj`, combining features from dicts, objects, lists, and arrays.

Highlights:
    - `sc.obj()`: fast and flexible container representing the best-of-all-worlds across objects, dicts, lists, and arrays
    - `sc.objdict()`: a slower `sc.obj()` with more features, such as setting items by position
    - `sc.odict()`: like `sc.objdict()`, but with dict syntax only, e.g. `foo['bar']` instead of `foo.bar`
"""
import re
import sys
import json
import copyreg
import types
import warnings
import numpy as np
import collections as co
import sciris as sc
from .sc_utils import _not_given # Imported directly since it's needed at definition time

# Restrict imports to user-facing modules
__all__ = ['ddict', 'counter', 'odict', 'objdict', 'obj', 'dictobj', 'asobj', 'argparse']

ddict = co.defaultdict # Define alias

# Critical attributes that should be resolved immediately, not via any special handling
_fundamental_attrs = frozenset(['__class__', '__reduce__', '__reduce_ex__', '__getstate__', '__setstate__', '__copy__','__deepcopy__'])

# Attribute names of each sc.obj class, cached so keys can be checked against them quickly
_class_attrs = {}


class counter(co.Counter):
    """
    Like `collections.Counter`, but with additional supported mathematical operations.

    `sc.counter` has an "array" property, which converts the values to a NumPy array;
    methods not available to a `Counter` object are performed on the array instead.

    **Examples**:

    ```python
    vals = [1,1,12,3,4,2,4,2,53,5,5,6,2,3,5]
    counts = sc.counter(vals)
    counts.max() # returns 3
    ```

    - *New in version 3.2.3.*
    """
    @property
    def array(self):
        """ NumPy array of values """
        return np.array(list(self.values()))

    def disp(self):
        return sc.pr(self)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.array[key]
        else:
            return super().__getitem__(key)

    def __getattr__(self, attr):
        """ For all other operations, try to perform them on the array """
        if attr in _fundamental_attrs:
            return self.__getattribute__(attr)
        else:
            try:
                return getattr(self.array, attr)
            except Exception as e:
                errormsg = f'"{attr}" is not a recognized method of Counter or array objects'
                raise AttributeError(errormsg) from e

    def _rewrap(self, out):
        """ Convert the output of a Counter operation back to a counter """
        return self.__class__(out) if isinstance(out, co.Counter) else out

    # Arithmetic returns a plain Counter by default
    def __add__(self, other): return self._rewrap(super().__add__(other))
    def __sub__(self, other): return self._rewrap(super().__sub__(other))
    def __or__( self, other): return self._rewrap(super().__or__(other))
    def __and__(self, other): return self._rewrap(super().__and__(other))


##############################################################################
#%% odict class
##############################################################################


class odict(dict):
    """
    Ordered dictionary with integer indexing

    An ordered dictionary, like the OrderedDict class, but supports list methods like integer
    indexing, key slicing, and item inserting. It can also replicate defaultdict behavior
    via the `defaultdict` argument.

    Args:
        args (dict): convert an existing dict, e.g. `sc.odict({'a':1})`
        defaultdict (class): if provided, create as a defaultdict as well
        kwargs (dict): additional keyword arguments, e.g. `sc.odict(a=1)`

    **Examples**:

    ```python
    # Simple example
    mydict = sc.odict(foo=[1,2,3], bar=[4,5,6]) # Assignment is the same as ordinary dictionaries
    mydict['foo'] == mydict[0] # Access by key or by index
    mydict[:].sum() == 21 # Slices are returned as numpy arrays by default
    for i,key,value in mydict.enumitems(): # Additional methods for iteration
        print(f'Item {i} is named {key} and has value {value}')

    # Detailed example
    foo = sc.odict({'ant':3,'bear':4, 'clam':6, 'donkey': 8}) # Create odict
    bar = foo.sorted() # Sort the dict
    assert bar['bear'] == 4 # Show get item by value
    assert bar[1] == 4 # Show get item by index
    assert (bar[0:2] == [3,4]).all() # Show get item by slice
    assert (bar['clam':'donkey'] == [6,8]).all() # Show alternate slice notation
    assert (bar[np.array([2,1])] == [6,4]).all() # Show get item by list
    assert (bar[:] == [3,4,6,8]).all() # Show slice with everything
    assert (bar[2:] == [6,8]).all() # Show slice without end
    bar[3] = [3,4,5] # Show assignment by item
    bar[0:2] = ['the', 'power'] # Show assignment by slice
    bar[[0,3]] = ['hill', 'trip'] # Show assignment by list
    bar.rename('clam','oyster') # Show rename
    print(bar) # Print results

    # Defaultdict examples
    dd = sc.odict(a=[1,2,3], defaultdict=list)
    dd['c'].append(4)

    nested = sc.odict(a=0, defaultdict='nested') # Create a infinitely nested dictionary (NB: may behave strangely on IPython)
    nested['b']['c']['d'] = 2
    ```
    Note: by default, integers are used as an alias to string keys, so cannot be used
    as keys directly. However, you can force regular-dict behavior using `setitem()`,
    and you can convert a dictionary with integer keys to an odict using `sc.odict.makefrom()`.
    If an odict has integer keys and the keys do not match the key positions, then the
    key itself will take precedence (e.g., `od[3]` is equivalent to `dict(od)[3]`,
    not `dict(od)[od.keys()[3]]`). This usage is discouraged.

    - *New in version 1.1.0:* "defaultdict" argument
    - *New in version 1.3.1:* allow integer keys via `makefrom()`; removed `to_OD`; performance improvements
    - *New in version 2.0.1:* allow deletion by index
    - *New in version 3.0.0:* allow numeric indices; inherit from dict rather than OrderedDict
    - *New in version 3.1.1:* allow steps in slices; `copy()` now behaves as a standard dict
    """

    _stale = True # The keys are only cached (as a list, for integer lookup) when first needed

    def __init__(self, *args, defaultdict=None, **kwargs):
        """ Combine the init properties of both OrderedDict and defaultdict """

        # Standard init
        dict.__init__(self, *args, **kwargs)

        # Handle defaultdict
        if defaultdict is not None:
            if defaultdict != 'nested' and not callable(defaultdict): # pragma: no cover
                errormsg = f'The defaultdict argument must be either "nested" or callable, not {type(defaultdict)}'
                raise TypeError(errormsg)
            self._setattr('_defaultdict', defaultdict) # Use object.__setattr__() since setattr() is overridden by sc.obj()

        return


    @classmethod
    def _new(cls, *args, **kwargs):
        """ Constructor for a new instance -- used in place of self.__class__() """
        return cls(*args, **kwargs)


    def __getstate__(self):
        """ Don't pickle or copy the cached keys """
        state = self.__dict__.copy()
        state.pop('_cached_keys', None)
        state.pop('_stale', None)
        return state


    def _cache_keys(self):
        """ Store a copy of the keys as a list so integer lookup doesn't have to regenerate it each time """
        self._setattr('_cached_keys', self.keys())
        self._setattr('_stale', False)
        return


    def _setattr(self, key, value):
        """ Shortcut to object method """
        return object.__setattr__(self, key, value)


    def _setitem(self, key, value):
        """ Shortcut to dict method """
        return dict.__setitem__(self, key, value)


    def __getitem__(self, key, allow_default=True):
        """ Allows getitem to support strings, integers, slices, lists, or arrays """

        # First, try just getting the item
        try:
            output = dict.__getitem__(self, key)
            return output
        except Exception as E:

            if isinstance(key, sc._stringtypes) or isinstance(key, tuple): # Normal use case: just use a string key
                if isinstance(E, KeyError): # We already encountered an exception, usually a KeyError
                    try: # Handle defaultdict behavior by first checking if it exists
                        _defaultdict = object.__getattribute__(self, '_defaultdict')
                    except:
                        _defaultdict = None
                    if _defaultdict is not None and allow_default: # If it does, use it, then get the key again
                        if _defaultdict == 'nested':
                            _defaultdict = lambda: self._new(defaultdict=_defaultdict) # Make recursive
                        dd = _defaultdict() # Create the new object
                        self._setitem(key, dd) # Add it to the dictionary
                        self._setattr('_stale', True) # Flag to refresh the cached keys
                        return dd # Return
                    else:
                        keys = self.keys()
                        if len(keys): errormsg = f'odict key "{key}" not found; available keys are:\n{sc.newlinejoin(keys)}'
                        else:         errormsg = f'Key {key} not found since odict is empty'
                        raise sc.KeyNotFoundError(errormsg)
                else: # Exception raised wasn't a key error -- just raise it again
                    raise E

            elif isinstance(key, sc._numtype): # Convert automatically from float
                thiskey = self._ikey(key)
                return dict.__getitem__(self, thiskey) # Note that defaultdict behavior isn't supported for non-string lookup

            elif isinstance(key, slice): # Handle a slice -- complicated
                try:
                    slicekeys = self._slicetokeys(key)
                    slicevals = [dict.__getitem__(self, k) for k in slicekeys]
                    output = self._sanitize_items(slicevals)
                    return output
                except Exception as E: # pragma: no cover
                    errormsg = 'Invalid odict slice'
                    raise ValueError(errormsg) from E

            elif self._is_odict_iterable(key): # Iterate over items
                listvals = [self.__getitem__(item) for item in key]
                if isinstance(key, list): # If the user supplied the keys as a list, assume they want the output as a list
                    output = listvals
                else:
                    output = self._sanitize_items(listvals) # Otherwise, assume as an array
                return output

            else: # pragma: no cover # Handle everything else (rare)
                return dict.__getitem__(self, key)


    def __setitem__(self, key, value):
        """ Allows setitem to support strings, integers, slices, lists, or arrays """

        object.__setattr__(self, '_stale', True) # Flag to refresh the cached keys

        if isinstance(key, (str,tuple)):
            dict.__setitem__(self, key, value)

        elif isinstance(key, sc._numtype): # Convert automatically from float...dangerous?
            thiskey = key if dict.__contains__(self, key) else self._ikey(key) # As with getitem, an existing key takes precedence
            self._setitem(thiskey, value)

        elif isinstance(key, slice):
            slicekeys = self._slicetokeys(key)
            if hasattr(value, '__len__'):
                valuelen = len(value)
                slicelen = len(slicekeys)
                if valuelen == slicelen:
                    for valind,k in enumerate(slicekeys):
                        dict.__setitem__(self, k, value[valind])  # e.g. odict[:] = arr[:]
                else: # pragma: no cover
                    errormsg = f'Slice "{key}" and values "{value}" have different lengths! ({slicelen}, {len(value)})'
                    raise ValueError(errormsg)
            else:
                for k in slicekeys:
                    dict.__setitem__(self, k, value) # e.g. odict[:] = 4

        elif self._is_odict_iterable(key) and hasattr(value, '__len__'): # Iterate over items
            if len(key)==len(value):
                for valind,thiskey in enumerate(key):
                    self.__setitem__(thiskey, value[valind])
            else: # pragma: no cover
                errormsg = f'Keys "{key}" and values "{value}" have different lengths! ({len(key)}, {len(value)})'
                raise ValueError(errormsg)

        else: # pragma: no cover
            self._setitem(key, value)

        return


    def setitem(self, key, value):
        """ Use regular dictionary `setitem`, rather than odict's """
        self._setattr('_stale', True) # Flag to refresh the cached keys
        self._setitem(key, value)
        return


    def update(self, *args, **kwargs):
        """ Update dict contents; set _stale so _cached_keys is refreshed on next use """
        self._setattr('_stale', True) # Flag to refresh the cached keys
        super().update(*args, **kwargs)
        return


    def setdefault(self, key, default=None):
        """ Default setdefault, except set stale to true """
        self._setattr('_stale', True)
        return dict.setdefault(self, key, default)


    def popitem(self):
        """ Default popitem, except set stale to true """
        self._setattr('_stale', True)
        return dict.popitem(self)


    def clear(self):
        """ Default clear, except set stale to true """
        self._setattr('_stale', True)
        return dict.clear(self)


    def __ior__(self, other):
        """ Default in-place merge (`|=`), except set stale to true """
        self._setattr('_stale', True)
        return dict.__ior__(self, other)


    def __repr__(self, maxlen=None, showmultilines=True, divider=False, dividerthresh=10,
                 numindents=0, recursionlevel=0, sigfigs=None, numformat=None, maxitems=200,
                 classname='odict', quote='', numleft='#', numsep=':', keysep=':', dividerchar='—'):
        """ Print a meaningful representation of the odict """

        # Set non-customizable primitives for display
        toolong      = ' [...]'    # String to display at end of line when maximum value character length is overrun.
        dividerstr   = dividerchar*40+'\n' # String to use as an inter-item divider.
        indentstr    = '    '      # Create string to use to indent.
        maxrecursion = 5           # It's a repr, no one could want more than that

        # Only if we are in the root call, indent by the number of indents.
        if recursionlevel == 0:
            theprefix = indentstr * numindents
        else: # Otherwise (if we are in a recursive call), indent only 1 indent.
            theprefix = indentstr

        # If the odict is empty, make the string just indicate it's an odict.
        allkeys = self.keys()
        nkeys = len(allkeys)
        halfmax = int(maxitems/2)
        extraitems = 0
        if nkeys == 0:
            output = classname + '()'
        else:
            output = ''
            keystrs = [] # Start with an empty list which we'll save key strings in.
            valstrs = [] # Start with an empty list which we'll save value strings in.
            vallinecounts = [] # Start with an empty list which we'll save line counts in.
            extraitems = nkeys - maxitems
            if extraitems <= 0:
                keylist = allkeys
            else:
                keylist = allkeys[:halfmax] + allkeys[-halfmax:]
            for thiskey in keylist: # Loop over the dictionary values
                thiskeystr = repr(thiskey) # Grab a str representation of the current key.
                thisval = self.__getitem__(thiskey) # Grab the current value.

                try: # It's rare, but sometimes repr fails
                    # If it's another odict, make a call increasing the recursionlevel and passing the same parameters we received.
                    if isinstance(thisval, odict):
                        if recursionlevel <= maxrecursion:
                            thisvalstr = sc.flexstr(thisval.__repr__(maxlen=maxlen, showmultilines=showmultilines, divider=divider, dividerthresh=dividerthresh, numindents=numindents, recursionlevel=recursionlevel+1, sigfigs=sigfigs, numformat=numformat))
                        else:
                            thisvalstr = f'{classname} [maximum recursion reached]'
                    elif sc.isnumber(thisval): # Flexibly print out numbers, since they're largely why we're here
                        if numformat is not None:
                            thisvalstr = numformat % thisval
                        elif sigfigs is not None:
                            thisvalstr = sc.sigfig(thisval, sigfigs=sigfigs)
                        else:
                            thisvalstr = sc.flexstr(thisval) # To avoid numpy's stupid 0.4999999999945
                    else: # Otherwise, do the normal repr() read.
                        thisvalstr = repr(thisval)
                except Exception as E: # pragma: no cover
                    thisvalstr = f'{sc.objectid(thisval)} read failed: {str(E)}'

                # Add information to the lists to retrace afterwards.
                keystrs.append(thiskeystr)
                valstrs.append(thisvalstr)
                vallinecounts.append(thisvalstr.count('\n') + 1) # Count the number of lines in the value.
            maxvallinecounts = max(vallinecounts)   # Grab the maximum count of lines in the dict values.

            maxkeylen = max([len(keystr) for keystr in keystrs])
            lenkeystrs = len(keystrs) # TODO: remove duplication with above
            if extraitems <= 0:
                indlist = range(lenkeystrs)
            else:
                indlist = list(range(halfmax)) + list(range(nkeys-halfmax, nkeys))
            for i,ind in enumerate(indlist): # Loop over the lists
                keystr = keystrs[i]
                valstr = valstrs[i]
                vallinecount = vallinecounts[i]

                if (divider or (maxvallinecounts>dividerthresh)) and \
                    showmultilines and recursionlevel==0 and i!=0: # Add a divider line if we should.
                    newoutput = sc.indent(prefix=theprefix, text=dividerstr, width=80)
                    if newoutput[-1] == '\n':
                        newoutput = newoutput[:-1]
                    output += newoutput

                # Trim the length of the entry if we need to.
                if not showmultilines:
                    valstr = valstr.replace('\n','\\n') # Replace line breaks with characters

                # Trim long entries
                if maxlen and len(valstr) > maxlen:
                    valstr = valstr[:maxlen-len(toolong)] + toolong

                # Create the the text to add, apply the indent, and add to the output
                spacer = ' '*(maxkeylen-len(keystr))
                if vallinecount == 1 or not showmultilines:
                    rawoutput = f'{numleft}{ind:d}{numsep} {quote}{keystr}{quote}{keysep}{spacer} {valstr}\n'
                else:
                    rawoutput = f'{numleft}{ind:d}{numsep} {quote}{keystr}{quote}{keysep}{spacer} \n{valstr}\n'

                # Perform the indentation
                newoutput = sc.indent(prefix=theprefix, text=rawoutput, width=80)

                # Strip out any terminal newline
                if newoutput[-1] == '\n':
                    newoutput = newoutput[:-1]

                # Add the new output to the full output
                if extraitems>0 and i == halfmax:
                    output += f'\n[{extraitems} additional odict items not shown]\n\n'
                output += newoutput

        # Trim off any terminal '\n'.
        if output[-1] == '\n':
            output = output[:-1]

        if nkeys > maxitems:
            output += '\n\n' + f'Note: only {maxitems} of {nkeys} entries shown.'

        # Return the formatted string.
        return output


    def _repr_pretty_(self, p, cycle, *args, **kwargs): # pragma: no cover
        """ Function to fix __repr__ in IPython"""
        p.text(self.__repr__(*args, **kwargs))
        return


    def __or__(self, other):
        """ Merge with `|`, returning the same type as this dict """
        if not isinstance(other, dict):
            return NotImplemented
        new = self.copy()
        new.update(other)
        return new


    def __ror__(self, other):
        """ Merge with `|` when this dict is on the right, e.g. `{'a':1} | sc.odict(b=2)` """
        if not isinstance(other, dict):
            return NotImplemented
        new = self._new(other)
        new.update(self)
        return new


    def __add__(self, dict2):
        """
        Allow two dictionaries to be added (merged).

        **Example**:

        ```python
        dict1 = sc.odict(a=3, b=4)
        dict2 = sc.odict(c=5, d=7)
        dict3 = dict1 + dict2
        ```
        """
        return sc.mergedicts(self, dict2)


    def __radd__(self, dict2):
        """ Allows sum() to work correctly """
        if not dict2: return self # Skips if start=0, as default with sum()
        else:         return sc.mergedicts(self._new(), dict2, self) # The right operand takes precedence, as with __add__()


    def __delitem__(self, key):
        """ Default delitem, except set stale to true and allow numeric values; slices etc are not supported """
        try:
            return dict.__delitem__(self, key)
        except Exception as E:
            if isinstance(key, sc._numtype): # If it's a number, use that
                thiskey = self._ikey(key)
                return dict.__delitem__(self, thiskey) # Note that defaultdict behavior isn't supported for non-string lookup
            else:
                raise E
        finally:
            self._setattr('_stale', True) # Flag to refresh the cached keys, after the deletion since _ikey() refreshes them


    def disp(self, maxlen=None, showmultilines=True, divider=False, dividerthresh=10, numindents=0, sigfigs=5, numformat=None, maxitems=20, **kwargs):
        """
        Print out flexible representation, short by default.

        **Example**:

        ```python
        z = sc.odict().make(keys=['a','b','c'], vals=[4.293487,3,6])
        z.disp(sigfigs=3)
        z.disp(numformat='%0.6f')
        ```
        """
        kwargs = sc.mergedicts(dict(maxlen=maxlen, showmultilines=showmultilines, divider=divider, dividerthresh=dividerthresh, numindents=numindents, recursionlevel=0, sigfigs=sigfigs, numformat=numformat, maxitems=maxitems), kwargs)
        print(self.__repr__(**kwargs))
        return


    def _ikey(self, key):
        """ Handle an integer key """
        if self._stale:
            self._cache_keys()
        try:
            return self._cached_keys[key]
        except IndexError: # pragma: no cover
            errormsg = f'index {key} out of range for dict of length {len(self)}'
            raise IndexError(errormsg) from None # Don't show the traceback


    def _slicetokeys(self, raw):
        """ Validate a key supplied as a slice object """

        startstop = dict(start=raw.start, stop=raw.stop) # start, stop, step

        # Handle possible string values for start and stop
        for key,val in startstop.items():
            if isinstance(val, str):
                startstop[key] = self.index(val)
                if key == 'stop':
                    startstop[key] += 1  # +1 on stop since otherwise confusing with names

        # Turn from a slice into keys
        sli = slice(startstop['start'], startstop['stop'], raw.step)
        inds = range(*sli.indices(len(self)))
        keys = [self._ikey(i) for i in inds]

        return keys


    @staticmethod
    def _matchkey(key, pattern, method):
        """ Helper function for findkeys """
        match = False
        if isinstance(key, tuple):
            for item in key:
                if odict._matchkey(item, pattern, method):
                    return True
        else: # For everything except a tuple, treat it as a string
            if not sc.isstring(key): # pragma: no cover
                try:
                    key = str(key) # Try to cast it to a string
                except Exception as E: # pragma: no cover
                    errormsg = f'Could not convert odict key of type {type(key)} to a string: {str(E)}'
                    raise TypeError(E)
            if   method == 're':         match = bool(re.search(pattern, key))
            elif method == 'in':         match = pattern in key
            elif method == 'startswith': match = key.startswith(pattern)
            elif method == 'endswith':   match = key.endswith(pattern)
            else: # pragma: no cover
                errormsg = f'Method "{method}" not found; must be "re", "in", "startswith", or "endswith"'
                raise ValueError(errormsg)
        return match


    def _is_odict_iterable(self, key):
        """ Check to see whether the "key" is actually an iterable: a list or array """
        output = isinstance(key, (list, np.ndarray))
        return output


    def _sanitize_items(self, items):
        """ Try to convert the output of a slice to an array, but give up easily and return a list """
        try:
            output = np.array(items) # Try standard Numpy array...
            if 'S' in str(output.dtype) or 'U' in str(output.dtype): # ...but instead of converting to string, convert to object array for Python 2 or 3 -- WARNING, fragile!
                output = np.array(items, dtype=object)
        except: # pragma: no cover
            output = items # If that fails, just give up and return the list
        return output


    @staticmethod
    def _transpose(iterator, n):
        """ Transpose a list of tuples into a tuple of n lists, including if the list is empty """
        return tuple(sc.transposelist(iterator)) if len(iterator) else tuple([] for _ in range(n))


    def export(self, doprint=True, classname='odict'):
        """ Export the odict in a form that is valid Python code """
        start = classname + '(['
        end = '])'
        output = start

        for key in self.keys():
            output += '('+repr(key)
            output += ', '
            child = self.get(key)
            if isinstance(child, odict): output += child.export(doprint=False) # Handle nested odicts -- WARNING, can't doesn't work for e.g. lists of odicts!
            else:                        output += repr(child)
            output += '), '

        output += end
        if doprint:
            print(output)
            return
        else:
            return output


    def pop(self, key, *args, **kwargs):
        """ Allows pop to support strings, integers, slices, lists, or arrays """
        try:
            if isinstance(key, sc._stringtypes):
                return dict.pop(self, key, *args, **kwargs)
            elif isinstance(key, sc._numtype): # Convert automatically from float...dangerous?
                thiskey = key if dict.__contains__(self, key) else self._ikey(key) # As with getitem, an existing key takes precedence
                return dict.pop(self, thiskey, *args, **kwargs)
            elif isinstance(key, slice): # Handle a slice -- complicated
                try:
                    slicekeys = self._slicetokeys(key)
                    slicevals = [dict.pop(self, k, *args, **kwargs) for k in slicekeys]
                    output = self._sanitize_items(slicevals)
                    return output
                except Exception as E: # pragma: no cover
                    errormsg = 'Invalid odict slice'
                    raise ValueError(errormsg) from E
            elif self._is_odict_iterable(key): # Iterate over items
                keys = self.keys()
                poplist = [keys[int(item)] if isinstance(item, sc._numtype) else item for item in key] # Convert to text keys, because indices change
                listvals = [self.pop(item, *args, **kwargs) for item in poplist]
                try:
                    return np.array(listvals)
                except: # pragma: no cover
                    return listvals
            else: # pragma: no cover # Handle string but also everything else
                try:
                    return dict.pop(self, key, *args, **kwargs)
                except: # Duplicated from __getitem__
                    keys = self.keys()
                    if len(keys): errormsg = f'odict key "{key}" not found; available keys are:\n{sc.newlinejoin(keys)}'
                    else:         errormsg = f'Key {key} not found since odict is empty'
                    raise sc.KeyNotFoundError(errormsg)
        finally:
            self._setattr('_stale', True) # Flag to refresh the cached keys, after popping since _ikey() refreshes them


    def remove(self, key, *args, **kwargs):
        """ Remove an item by key and do not return it """
        self.pop(key, *args, **kwargs)
        return


    def index(self, value):
        """ Return the index of a given key """
        return self.keys().index(value)


    def valind(self, value):
        """ Return the index of a given value """
        return self.values().index(value)


    def findkeys(self, pattern=None, method=None, first=None):
        """
        Find all keys that match a given pattern. By default uses regex, but other options
        are 'find', 'startswith', 'endswith'. Can also specify whether or not to only return
        the first result (default false). If the key is a tuple instead of a string, it will
        search each element of the tuple.
        """
        if pattern is None: pattern = ''
        if method  is None: method  = 're'
        if first   is None: first   = False
        keys = []
        for key in self.keys():
            if self._matchkey(key, pattern, method):
                if first: return key
                else:     keys.append(key)
        return keys


    def findbykey(self, pattern=None, method=None, first=True):
        """ Same as findkeys, but returns values instead """
        keys = self.findkeys(pattern=pattern, method=method, first=first)
        if not first and len(keys) == 1: keys = keys[0] # If it's a list of one element, turn it into that element instead
        return self.__getitem__(keys)


    def findbyval(self, value, first=True, strict=False):
        """
        Returns the key(s) that match a given value -- reverse of findbykey, except here
        uses exact matches to the value or values provided.

        **Example**:

        ```python
        z = sc.odict({'dog':[2,3], 'cat':[4,6], 'mongoose':[4,6]})
        z.findbyval([4,6]) # returns 'cat'
        z.findbyval([4,6], first=False) # returns ['cat', 'mongoose']
        ```
        """
        keys = []
        for key,val in self.items():
            try:
                match = bool(val == value) # Exact match
                match = match or (not strict and isinstance(value, list) and val in value) # "value" is a list and it's contained
            except ValueError: # Comparisons involving arrays give arrays
                match = np.array_equal(val, value)
            if match:
                if first: return key
                else:     keys.append(key)
        return keys


    def filter(self, keys=None, pattern=None, method=None, exclude=False):
        """
        Find matching keys in the odict, and return a new odict

        Filter the odict keys and return a new odict which is a subset. If keys is a list,
        then uses that for matching. If the first argument is a string, then treats as a pattern
        for matching using `findkeys() <odict.findkeys`.

        Args:
            keys (list): the list of keys to keep (or exclude)
            pattern (str): the pattern by which to match keys; see `findkeys() <odict.findkeys` for details
            method (str): the method by which to match keys; see `findkeys() <odict.findkeys` for details
            exclude (bool): if exclude=True, then exclude rather than include matches

        See also `sort()`, which includes filtering by position.
        """
        if sc.isstring(keys) and pattern is None: # Assume first argument, transfer
            pattern = keys
            keys = None
        filtered = self._new() # Create a new instance of the same class
        if keys is None:
            keys = self.findkeys(pattern=pattern, method=method, first=False)
        if not exclude:
            for key in keys:
                filtered[key] = self.__getitem__(key)
        else:
            for key in self.keys():
                if key not in keys:
                    filtered[key] = self.__getitem__(key)
        return filtered


    def filtervals(self, value):
        """ Like filter, but filters by value rather than key """
        keys = self.findbyval(value)
        return self.filter(keys=keys)


    def append(self, key=_not_given, value=_not_given):
        """ Support an append method, like a list """
        if value is _not_given: # Assume called with a single argument
            key, value = None, (None if key is _not_given else key)
        if key is None or key is _not_given:
            key = 'key'+sc.flexstr(len(self))  # Define the key just to be the current index
        self.__setitem__(key, value)
        return


    def insert(self, pos=None, key=_not_given, value=_not_given):
        """
        Function to do insert a key -- note, computationally inefficient.

        **Example**:

        ```python
        z = sc.odict()
        z['foo'] = 1492
        z.insert(1604)
        z.insert(0, 'ganges', 1444)
        z.insert(2, 'mekong', 1234)
        ```
        """

        # Handle inputs
        realpos, realkey, realvalue = pos, key, value
        if value is _not_given:
            if key is _not_given: # Assume it's called like odict.insert(66)
                realpos, realvalue = 0, pos
            else: # Assume it's called like odict.insert('route', 66)
                realpos, realkey, realvalue = 0, pos, key
        if realkey is _not_given:
            realkey = 'key'+sc.flexstr(len(self))
        if realpos is None:
            realpos = 0
        if realkey in self: # Remove an existing key first, so it's moved and overwritten
            dict.pop(self, realkey)
            self._setattr('_stale', True)
        if realpos>len(self): # pragma: no cover
            errormsg = f'Cannot insert {key} at position {pos} since length of odict is {len(self)}'
            raise ValueError(errormsg)

        # Create a temporary dictionary to hold all of the items after the insertion point
        tmpdict = self._new()
        origkeys = self.keys()
        originds = range(len(origkeys))
        if not len(originds) or realpos==len(originds): # It's empty or in the final position, just append
            self.__setitem__(realkey, realvalue) # pragma: no cover
        else: # Main usage case, it's not empty
            try:
                insertind = originds.index(realpos) # Figure out which index we're inseting at
            except Exception as E: # pragma: no cover
                errormsg = f'Could not insert item at position {realpos} in odict with {len(originds)} items'
                raise ValueError(errormsg) from E
            keystopop = origkeys[insertind:] # Pop these keys until we get far enough back
            for keytopop in keystopop:
                tmpdict.__setitem__(keytopop, self.pop(keytopop))
            self.__setitem__(realkey, realvalue) # Insert the new item at the right location
            for keytopop in keystopop: # Insert popped items back in
                self.__setitem__(keytopop, tmpdict.pop(keytopop))

        return


    def copy(self, deep=False):
        """
        Make a (shallow) copy of the dict.

        Args:
            deep (bool): if True, do a deep rather than shallow copy

        **Examples**:

        ```python
        d1 = sc.odict(a=[1,2,3], b='foo')
        d2 = d1.copy()
        d3 = d1.copy(deep=True)

        d1.pop('b') # affects d1 but not d2 or d3
        d1[0].append(4) # affects d1 and d2 but not d3
        ```
        """
        if not deep:
            new = self._new(super().copy())
            defaultdict = self.__dict__.get('_defaultdict')
            if defaultdict is not None:
                new._setattr('_defaultdict', defaultdict)
        else:
            new = sc.dcp(self)
        return new


    def dcp(self):
        """
        Shortcut to odict.copy(deep=True).

        - *New in version 3.2.3.*
        """
        return self.copy(deep=True)


    def rename(self, oldkey, newkey):
        """ Change a key name (note: not optimized for speed) """
        nkeys = len(self)
        if isinstance(oldkey, sc._numtype):
            index = oldkey if oldkey >= 0 else oldkey + nkeys
            keystr = self.keys()[index]
        else: # Forge ahead for strings and anything else!
            index = self.keys().index(oldkey)
            keystr = oldkey
        self.__setitem__(newkey, self.pop(keystr))
        if index<nkeys-1:
            for i in range(index+1, nkeys):
                key = self.keys()[index]
                value = self.pop(key)
                self.__setitem__(key, value)
        return


    def sort(self, sortby=None, reverse=False, copy=False):
        """
        Create a sorted version of the odict.

        By default, this method sorts alphabetically by key, but many other options
        are possible:

            - 'keys' sorts alphabetically by key
            - 'values' sorts in ascending order by value
            - if a list of keys is provided, sort by that order (any keys not provided will be omitted from the sorted dict!)
            - if a list of numbers is provided, treat these as indices and sort by that order
            - if a list of boolean values is provided, then omit False entries

        Args:
            sortby (str or list): what to sort by; see above for options
            reverse (bool): whether to return results in reverse order
            copy (bool): whether to return a copy (same as `sorted()`)

        For filtering by string matching on keys, see `filter()`.

        - *New in version 3.0.0:* removed "verbose" argument
        """
        origkeys = self.keys()
        if (sortby is None) or (isinstance(sortby, str) and sortby == 'keys'):
            allkeys = sorted(origkeys)
        else:
            if isinstance(sortby, str) and sortby == 'values':
                origvals = self.values()
                sortby = sorted(range(len(origvals)), key=origvals.__getitem__) # Reset sortby based on https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
            if not sc.isiterable(sortby): # pragma: no cover
                raise Exception('Please provide a list to determine the sort order.')

            if all([isinstance(x, sc._stringtypes + (tuple,)) for x in sortby]): # Going to sort by keys
                allkeys = list(sortby) # Assume the user knows what s/he is doing
            elif all([isinstance(x, sc._booltypes) for x in sortby]): # Using Boolean values to filter
                allkeys = [origkeys[i] for i,tf in enumerate(sortby) if tf]
            elif all([isinstance(x, sc._numtype) for x in sortby]): # Going to sort by numbers
                allkeys = [origkeys[ind] for ind in sortby]
            else: # pragma: no cover
                errormsg = f'Cannot figure out how to sort by "{sortby}"'
                raise TypeError(errormsg)

        if reverse:
            allkeys.reverse() # If requested, reverse order
        items = [(key, self[key]) for key in allkeys]
        if copy:
            out = self._new()
        else:
            out = self
            dict.clear(out) # Remove all keys, including any not in allkeys
        dict.update(out, items) # Use the dict method, since integer keys would be treated as indices
        out._setattr('_stale', True)
        return out if copy else None


    def sorted(self, sortby=None, reverse=False):
        """ Shortcut for making a copy of the sorted odict -- see `sort()` for options """
        return self.sort(sortby=sortby, copy=True, reverse=reverse)


    def reverse(self, copy=False):
        """ Reverse the order of an odict """
        reversedinds = list(range(len(self)))[::-1] # Use indices rather than keys, which may be of any type
        output = self.sort(sortby=reversedinds, copy=copy)
        return output


    def reversed(self):
        """ Shortcut for making a copy of the sorted odict """
        return self.reverse(copy=True)


    def make(self, keys=None, vals=None, keys2=None, keys3=None, coerce='full'):
        """
        An alternate way of making or adding to an odict.

        Args:
            keys (list/int): the list of keys to use
            vals (list/arr): the list of values to use
            keys2 (list/int): for a second level of nesting
            keys3 (list/int): for a third level of nesting
            coerce (str): what types to coerce into being separate dict entries

        **Examples**:

        ```python
        a = sc.odict().make(5) # Make an odict of length 5, populated with empty lists and default key names
        b = sc.odict().make('foo',34) # Make an odict with a single key 'foo' of value 34
        c = sc.odict().make(['a','b']) # Make an odict with keys 'a' and 'b', initialized to empty lists
        d = sc.odict().make(['a','b'], 0) # Make an odict with keys 'a' and 'b', initialized to 0
        e = sc.odict().make(keys=['a','b'], vals=[1,2]) # Make an odict with 'a':1 and 'b':2
        f = sc.odict().make(keys=['a','b'], vals=np.array([1,2])) # As above, since arrays are coerced into lists
        g = sc.odict({'a':34, 'b':58}).make(['c','d'],[99,45]) # Add extra keys to an exising odict
        h = sc.odict().make(keys=['a','b','c'], keys2=['A','B','C'], keys3=['x','y','z'], vals=0) # Make a triply nested odict
        ```
        *New in version 1.2.2:* "coerce" argument
        """
        # Handle keys
        keylist = []
        if keys is None and vals is None: # pragma: no cover
            return self # Nothing to do if nothing supplied
        if keys is None and vals is not None: # pragma: no cover
            keys = len(sc.tolist(vals)) # Values are supplied but keys aren't: use default keys
        if isinstance(keys, sc._numtype): # It's a single number: pre-generate
            keylist = ['%i'%i for i in range(keys)] # Generate keylist
        elif isinstance(keys, sc._stringtypes): # It's a single string
            keylist = [sc.flexstr(keys)]
        elif isinstance(keys, list): # It's a list: use directly
            keylist = keys
        else: # pragma: no cover
            errormsg = f'Could not understand keys "{keys}": must be number, string, or list'
            raise TypeError(errormsg)
        nkeys = len(keylist)

        # Handle values
        vals = sc.tolist(vals, coerce=coerce)
        nvals = len(vals)
        if nvals==0: # Special case: it's an empty list
            vallist = [sc.dcp(vals) for _ in range(nkeys)]
        elif nvals==1: # Only a single value: duplicate it
            vallist = [sc.dcp(vals[0]) for _ in range(nkeys)]
        elif nvals==nkeys: # Lengths match, can use directly
            vallist = vals
        else: # pragma: no cover
            errormsg = f'Must supply either a single value or a list of same length as the keys ({nkeys} keys, {nvals} values supplied)'
            raise ValueError(errormsg)

        # Handle nested keys -- warning, would be better to not hard-code this, but does the brain in as it is!
        if keys2 is not None and keys3 is not None: # Doubly nested
            self.make(keys=keys, vals=self._new().make(keys=keys2, vals=self._new().make(keys=keys3, vals=vals)))
        elif keys2 is not None: # Singly nested # pragma: no cover
            self.make(keys=keys, vals=self._new().make(keys=keys2, vals=vals))
        else: # Not nested -- normal case of making an odict
            for key,val in zip(keylist,vallist): # Update odict
                self.__setitem__(key, val)

        return self # Return the odict


    @classmethod
    def makefrom(cls, source=None, include=None, keynames=None, force=True, *args, **kwargs):
        """
        Create an odict from entries in another dictionary. If keys is None, then
        use all keys from the current dictionary.

        Args:
            source (dict/list/etc): the item(s) to convert to an odict
            include (list): list of keys to include from the source dict in the odict (default: all)
            keynames (list): names of keys if source is not a dict
            force (bool): whether to force conversion to an odict even if e.g. the source has numeric keys

        **Examples**:

        ```python
        a = 'cat'
        b = 'dog'
        o = sc.odict.makefrom(source=locals(), include=['a','b']) # Make use of fact that variables are stored in a dictionary

        d = {'a':'cat', 'b':'dog'}
        o = sc.odict.makefrom(d) # Same as sc.odict(d)
        l = ['cat', 'monkey', 'dog']
        o = sc.odict.makefrom(source=l, include=[0,2], keynames=['a','b'])

        d = {12:'monkeys', 3:'musketeers'}
        o = sc.odict.makefrom(d)
        ```
        """
        # Initialize new odict
        out = cls()

        # Make sure it's iterable
        if source is not None: # Don't do anything if there's nothing there # pragma: no cover
            if not(sc.isiterable(source)): # Make sure it's iterable
                source = sc.tolist(source)
            elif isinstance(source, sc._stringtypes):
                source = [source] # Special case -- strings are iterable, but we don't want to

            if len(source)==0:
                return out # Nothing to do here
            else:
                # Handle cases where keys or keynames are not supplied
                if include is None:
                    if   isinstance(source, (list, tuple)):   include = range(len(source))
                    elif isinstance(source, dict):            include = list(source.keys())
                    else:                                     raise TypeError(f'Unable to guess keys for object of type {type(source)}')
                include = sc.tolist(include) # Make sure it's a list -- note, does not convert other iterables to a list!
                if keynames is None: keynames = include # Use key names

                # Loop over supplied keys -- source keys and output keys
                for skey,okey in zip(include,keynames):
                    try:
                        v = source[skey]
                        if force:
                            out.setitem(okey, v)
                        else:
                            okey = str(okey)
                            out.__setitem__(okey, v)
                    except Exception as E: # pragma: no cover
                        errormsg = f'Key "{skey}" not found: {repr(E)}'
                        raise sc.KeyNotFoundError(errormsg) from E

        return out # As with make()


    def map(self, func=None):
        """
        Apply a function to each element of the odict, returning
        a new odict with the same keys.

        **Example**:

        ```python
        cat = sc.odict({'a':[1,2], 'b':[3,4]})
        def myfunc(mylist): return [i**2 for i in mylist]
        dog = cat.map(myfunc) # Returns odict({'a':[1,4], 'b':[9,16]})
        ```
        """
        output = self._new()
        for key in self.keys():
            output[key] = func(self.__getitem__(key))
        return output


    def fromeach(self, ind=None, asdict=True):
        """
        Take a "slice" across all the keys of an odict, picking the same index
        (or indices) from each entry. To apply a function to each entry instead,
        use `map()`.

        Args:
            ind (int/list/slice): the index or indices to pick from each entry
            asdict (bool): whether to return an odict (default) or just the picked values

        **Examples**:

        ```python
        z = sc.odict({'a':np.array([1,2,3,4]), 'b':np.array([5,6,7,8])})
        z.fromeach(2) # Returns odict({'a':3, 'b':7})
        z.fromeach(2, asdict=False) # Returns array([3,7])
        z.fromeach([1,3]) # Returns odict({'a':array([2,4]), 'b':array([6,8])})

        z = sc.odict({'a':[1,2,3,4], 'b':[5,6,7,8]})
        z.fromeach([1,3]) # Returns odict({'a':[2,4], 'b':[6,8]})
        ```
        """
        output = self._new()
        for key,val in self.items():
            if isinstance(ind, list) and not isinstance(val, np.ndarray): # Lists etc. can't be indexed by a list, so pick each index
                output[key] = [val[i] for i in ind]
            else:
                output[key] = val[ind]
        if asdict: return output # Output as a slimmed-down odict
        else:      return output[:] # Output as just the entries


    def toeach(self, ind=None, val=None):
        """
        The inverse of fromeach: partially reset elements within
        each odict key.

        **Example**:

        ```python
        z = sc.odict({'a':[1,2,3,4], 'b':[5,6,7,8]})
        z.toeach(2, [10,20])    # z is now odict({'a':[1,2,10,4], 'b':[5,6,20,8]})
        z.toeach(ind=3,val=666) #  z is now odict({'a':[1,2,10,666], 'b':[5,6,20,666]})
        ```
        """
        nkeys = len(self.keys())
        if not(sc.isiterable(val)): # Assume it's meant to be populated in each
            val = [val]*nkeys # Duplicated
        if len(val)!=nkeys: # pragma: no cover
            errormsg = f'To map values onto each key, they must be the same length ({len(val)} vs. {nkeys})'
            raise ValueError(errormsg)
        for k,key in self.enumkeys():
            self.__getitem__(key)[ind] = val[k]
        return


    def enumkeys(self, transpose=False):
        """
        Shortcut for enumerate(odict.keys()).

        If transpose=True, return a tuple of lists rather than a list of tuples.
        """
        iterator = list(enumerate(self.keys()))
        if transpose: iterator = self._transpose(iterator, 2)
        return iterator


    def enumvals(self, transpose=False):
        """
        Shortcut for enumerate(odict.values())

        If transpose=True, return a tuple of lists rather than a list of tuples.
        """
        iterator = list(enumerate(self.values()))
        if transpose: iterator = self._transpose(iterator, 2)
        return iterator


    def enumvalues(self, transpose=False): # pragma: no cover
        """ Alias for enumvals(). *New in version 1.2.0.* """
        return self.enumvals(transpose=transpose)


    def enumitems(self, transpose=False):
        """
        Returns tuple of 3 things: index, key, value.

        If transpose=True, return a tuple of lists rather than a list of tuples.
        """
        iterator = []
        for ind,item in enumerate(self.items()):
            thistuple = (ind,)+item # Combine into one tuple
            iterator.append(thistuple)
        if transpose: iterator = self._transpose(iterator, 3)
        return iterator


    @classmethod
    def promote(cls, obj=None):
        """
        Like promotetolist, but for odicts.

        **Example**:

        ```python
        od = sc.odict.promote(['There','are',4,'keys'])
        ```
        Note, in most cases sc.odict(obj) or sc.odict().make(obj) can be used instead.
        """
        if isinstance(obj, odict):
            return obj # Don't need to do anything
        elif isinstance(obj, dict):
            return cls._new(obj)
        elif isinstance(obj, list):
            newobj = cls._new()
            for i,val in enumerate(obj):
                newobj['Key %i'%i] = val
            return newobj
        else:
            return cls._new({'Key':obj})

    def keys(self):
        """ Return a list of keys (as in Python 2), not a dict_keys object. """
        return list(dict.keys(self))

    def values(self):
        """ Return a list of values (as in Python 2). """
        return list(dict.values(self))

    def items(self, transpose=False):
        """ Return a list of items (as in Python 2). """
        iterator = list(dict.items(self))
        if transpose: iterator = self._transpose(iterator, 2)
        return iterator

    def dict_keys(self):
        """ Return an iterator (not a list) over keys (as in Python 2). """
        return dict.keys(self)

    def dict_values(self):
        """ Return an iterator (not a list) over values (as in Python 2). """
        return dict.values(self)

    def dict_items(self):
        """ Return an iterator (not a list) over items (as in Python 2). """
        return dict.items(self)

    def iteritems(self, transpose=False):
        """ Alias to items() """
        return self.items(transpose=transpose)

    def makenested(self, *args, **kwargs):
        """ Alias to sc.makenested(odict); see sc.makenested() for full documentation. *New in version 1.2.0.* """
        return sc.makenested(self, *args, **kwargs)

    def getnested(self, *args, **kwargs):
        """ Alias to sc.getnested(odict); see sc.makenested() for full documentation. *New in version 1.2.0.* """
        return sc.getnested(self, *args, **kwargs)

    def setnested(self, *args, **kwargs):
        """ Alias to sc.setnested(odict); see sc.makenested() for full documentation. *New in version 1.2.0.* """
        return sc.setnested(self, *args, **kwargs)

    def iternested(self, *args, **kwargs):
        """ Alias to sc.iternested(odict); see sc.makenested() for full documentation. *New in version 1.2.0.* """
        return sc.iternested(self, *args, **kwargs)


##############################################################################
#%% Other classes
##############################################################################

class objdict(odict):
    """
    An `odict` that acts like an object -- allow keys to be set/retrieved by object
    notation.

    In general, operations that would normally act on attributes (e.g. `obj.x = 3`)
    instead act on dict keys (e.g. `obj['x'] = 3`). If you want to actually get/set
    an attribute, use `obj.getattribute()`/`obj.setattribute()`.

    For a much faster alternative with fewer features, see `sc.obj()`.

    **Examples**:

    ```python
    import sciris as sc

    obj = sc.objdict(foo=3, bar=2)
    obj.foo + obj.bar # Gives 5
    for key in obj.keys(): # It's still a dict
        obj[key] = 10

    od = sc.objdict({'height':1.65, 'mass':59})
    od.bmi = od.mass/od.height**2
    od['bmi'] = od['mass']/od['height']**2 # Vanilla syntax still works
    od.keys = 3 # This raises an exception (you can't overwrite the keys() method)
    ```
    Nested logic based in part on addict: https://github.com/mewwts/addict

    - *New in version 3.5.0:* much faster attribute access
    - *New in version 3.6.0:* faster creation, integer indexing, and failed attribute lookups
    - *New in version 3.7.0:* `sc.obj` is a separate, faster class, rather than an alias
    """

    def __init__(self, *args, **kwargs):
        odict.__init__(self, *args, **kwargs)
        if kwargs: # Warn if a keyword argument, e.g. sc.obj(index=3), can't be accessed as an attribute
            attrs = _class_attrs.get(type(self))
            if attrs is None:
                attrs = _class_attrs[type(self)] = frozenset(dir(type(self)))
            for key in attrs.intersection(kwargs):
                warnmsg = f'Key "{key}" has the same name as an attribute of {type(self).__name__}, so obj.{key} will not return it; use obj["{key}"] instead'
                warnings.warn(warnmsg, category=UserWarning, stacklevel=2)
        return


    def __repr__(self, *args, **kwargs):
        """
        Show e.g. `sc.obj(a=1, b=2)`, or one key per line if that's too long

        If arguments are supplied (e.g. `sigfigs`), use the `sc.odict()` format instead.
        """
        if args or kwargs:
            return odict.__repr__(self, *args, quote='', numsep='.', classname=type(self).__name__, **kwargs)
        return self._repr()


    def _repr(self, width=80, maxitems=200, _seen=None):
        """ Create the repr, recursing into child objs """
        name = f'sc.{type(self).__name__}' if type(self) in (objdict, obj) else type(self).__name__
        if not len(self):
            return f'{name}()'
        if _seen is None:
            _seen = set()
        if id(self) in _seen: # Handle an obj that contains itself
            return f'{name}(...)'
        _seen = _seen | {id(self)}

        # Get the keys and values to show
        keys = self.keys()
        nkeys = len(keys)
        if nkeys > maxitems:
            half = maxitems//2
            keys = keys[:half] + keys[-half:]
        kwstyle = all(isinstance(k, str) and k.isidentifier() for k in keys) # Use keyword style, e.g. a=1, if possible
        keystrs = keys if kwstyle else [repr(k) for k in keys]
        keylen = max(len(k) for k in keystrs)
        valwidth = width - keylen - 7 # Room left on the line after the indent and key
        valstrs = []
        for key in keys:
            val = dict.__getitem__(self, key)
            try:
                if isinstance(val, (objdict, obj)):
                    valstr = val._repr(width=valwidth, maxitems=maxitems, _seen=_seen)
                elif sc.isnumber(val):
                    valstr = str(val) # Avoid e.g. "np.float64(0.5)"
                else:
                    valstr = repr(val)
            except Exception as E: # pragma: no cover
                valstr = f'<{type(val).__name__} repr failed: {E}>'
            valstrs.append(valstr)

        # Try the short form, e.g. sc.obj(a=1, b=2)
        if kwstyle:
            short = f'{name}(' + ', '.join(f'{k}={v}' for k,v in zip(keystrs, valstrs)) + ')'
        else:
            short = f'{name}({{' + ', '.join(f'{k}: {v}' for k,v in zip(keystrs, valstrs)) + '})'
        if nkeys <= maxitems and len(short) <= width and '\n' not in short:
            return short

        # Otherwise, show one key per line
        lines = [f'{name}(' if kwstyle else f'{name}({{']
        for i,(keystr,valstr) in enumerate(zip(keystrs, valstrs)):
            if nkeys > maxitems and i == half:
                lines.append(f'    # [{nkeys - maxitems} additional items not shown]')
            prefix = f'    {keystr:<{keylen}} = ' if kwstyle else f'    {keystr+":":<{keylen+1}} '
            if isinstance(dict.__getitem__(self, keys[i]), (objdict, obj)):
                valstr = valstr.replace('\n', '\n    ') # Indent child objs by one level
            else:
                valstr = valstr.replace('\n', '\n' + ' '*len(prefix)) # Align e.g. arrays with the start of the value
            lines.append(f'{prefix}{valstr},')
        lines.append(')' if kwstyle else '})')
        return '\n'.join(lines)


    def __dir__(self):
        """ Include keys, for tab completion """
        return object.__dir__(self) + [k for k in dict.keys(self) if isinstance(k, str) and k.isidentifier()]


    def __getattr__(self, attr):
        """ Only called if not found as an attribute, so get it as a dict key """
        if attr not in _fundamental_attrs: # Pickle/copy protocol: must resolve from class, not as dict keys
            try: # The usual case: it's a key
                return dict.__getitem__(self, attr)
            except KeyError:
                pass
            if self.__dict__.get('_defaultdict') is not None:
                return self.__getitem__(attr)
        errormsg = f"'{type(self).__name__}' object has no attribute '{attr}'"
        raise AttributeError(errormsg, name=attr, obj=self)


    def __setattr__(self, name, value):
        """ Set key in dict, not attribute! """
        if name[:2] == '__': # If it starts with a double underscore, it's almost certainly an attribute, not a key
            return object.__setattr__(self, name, value)
        elif name in self.__dict__ or hasattr(type(self), name):
            errormsg = f'"{name}" exists as an attribute, so cannot be set as key; use setattribute() instead'
            raise KeyError(errormsg)
        else:
            return self.__setitem__(name, value)


    def __getitem__(self, attr):
        """ Handle dict keys, including nested defaultdict logic """
        try: # Try retrieving normally
            return dict.__getitem__(self, attr)
        except Exception:
            pass
        if isinstance(attr, sc._numtype): # Integer index
            return dict.__getitem__(self, self._ikey(attr))
        _defaultdict = self.__dict__.get('_defaultdict') # Handle defaultdict behavior by first checking if it exists
        if _defaultdict is None or not isinstance(attr, (str, tuple)):
            return odict.__getitem__(self, attr, allow_default=False) # Slices, lists, etc., or raise the KeyError
        elif _defaultdict == 'nested': # Create a recursive object with links back to the ultimate parent
            dd = self._new(defaultdict='nested')
            object.__setattr__(dd, '_nested_parent', self)
            object.__setattr__(dd, '_nested_attr', attr)
            return dd
        else:
            dd = _defaultdict() # Create the new object
            self[attr] = dd # Since we don't know what this object is, we can't be clever about recursion
            return dd


    def __setitem__(self, name, value):
        """ Set as a dict item """
        if isinstance(name, str): # Fast path for the usual case
            if not dict.__contains__(self, name):
                object.__setattr__(self, '_stale', True) # Flag to refresh the cached keys
            dict.__setitem__(self, name, value)
        else:
            odict.__setitem__(self, name, value)
        attrs = self.__dict__
        if '_nested_parent' in attrs:
            attrs['_nested_parent'][attrs['_nested_attr']] = self
            object.__delattr__(self, '_nested_parent')
            object.__delattr__(self, '_nested_attr')
        return


    def __delattr__(self, name):
        """ Delete dict key before deleting actual attribute """
        try:
            del self[name]
        except: # pragma: no cover
            object.__delattr__(self, name)
        return


    def getattribute(self, name):
        """ Get attribute if truly desired """
        return object.__getattribute__(self, name)


    def setattribute(self, name, value, force=False):
        """ Set attribute if truly desired """
        if hasattr(self.__class__, name) and not force:
            errormsg = f'{type(self).__name__} attribute "{name}" is read-only'
            raise AttributeError(errormsg)
        return object.__setattr__(self, name, value)


    def delattribute(self, name):
        """ Delete attribute if truly desired """
        return object.__delattr__(self, name)


class _objclassmethod:
    """ A read-only classmethod or staticmethod of `sc.obj` (see `_protect_methods()`) """

    def __init__(self, method, fset):
        self.method = method
        self.fset = fset

    def __get__(self, obj, objtype=None):
        return self.method.__get__(obj, objtype)

    def __set__(self, obj, value):
        self.fset(obj, value)


def _protect_methods(cls, attrs):
    """
    Make the methods of a `sc.obj` class read-only, so a key with the same name can't
    replace them. Instance attributes are keys, and the only class attributes that
    take priority over these are ones that define `__set__()`. Using `property()` and
    `__get__()` keeps it all in C, and adds nothing to the lookup of other attributes.
    """
    def readonly(name):
        def fset(self, value):
            errormsg = f'"{name}" is a method of {type(self).__name__}, so cannot be set as an attribute; use obj["{name}"] instead'
            raise AttributeError(errormsg)
        return fset

    for name,attr in attrs.items():
        if name[:2] != '__':
            if isinstance(attr, (classmethod, staticmethod, types.ClassMethodDescriptorType)):
                setattr(cls, name, _objclassmethod(attr, readonly(name)))
            elif callable(attr) and hasattr(attr, '__get__') and not isinstance(attr, type):
                setattr(cls, name, property(attr.__get__, readonly(name), doc=attr.__doc__))
    return


class obj(odict):
    """
    An object that is also a dict, as fast as a plain object and a plain dict.

    The keys and the attributes are the same thing: `obj.x` is `obj['x']`. It is an
    `sc.odict()`, so it has the same methods (e.g. `obj.sort()`, `obj.enumitems()`,
    `obj.filter()`) and can read values by position (e.g. `obj[0]`).

    **Examples**:

    ```python
    import sciris as sc

    obj = sc.obj(foo=3, bar=2)
    obj.foo + obj.bar # Gives 5
    obj.baz = 7
    obj[0] # Gives 3
    for key in obj.keys(): # It's still a dict
        obj[key] = 10
    ```

    As with `sc.objdict()`, methods are read-only: `obj.keys = 3` raises an exception,
    and a key with the name of a method can only be used as an item, e.g. `obj['keys']`.

    To be this fast, it has no code that runs when an attribute or item is set, so
    it differs from `sc.objdict()` as follows:

    - There are no attributes other than keys, unless a subclass declares them with `__slots__` (see below).
    - Position only works for reading: `obj[0] = 3` makes a new key `0`, and `del obj[0]` raises a `KeyError`.
    - Lists of keys (e.g. `obj[['a','b']]`) aren't supported, and nor are slices (e.g. `obj[:2]`) before Python 3.12.
    - There is no `defaultdict` argument.

    For these, use `sc.objdict()`, which is 2 to 13 times slower per operation.

    Subclasses are as fast as `sc.obj()`. A subclass can define methods, which are
    read-only too, and can keep attributes that aren't keys by declaring them in
    `__slots__`. If it overrides `__setitem__()`, `obj.x = 3` calls it, at the cost
    of that subclass's attributes being slower to set:

    ```python
    class Locked(sc.obj):
        __slots__ = ('locked',) # An attribute, not a key

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.locked = True

        def __setitem__(self, key, value):
            if self.locked:
                raise KeyError(key)
            super().__setitem__(key, value)
    ```

    - *New in version 3.5.0:* `sc.obj()` alias of `sc.objdict()`
    - *New in version 3.7.0:* a separate class, as fast as `sc.dictobj()`
    """

    # Use the dict methods directly, since there are no cached keys to update
    __getitem__ = dict.__getitem__
    __setitem__ = dict.__setitem__
    __delitem__ = dict.__delitem__
    update = dict.update
    setdefault = dict.setdefault
    popitem = dict.popitem
    clear = dict.clear
    __repr__ = objdict.__repr__
    _repr = objdict._repr
    _stale = property(lambda self: True, lambda self, value: None) # Always stale; ignored when odict methods set it

    def __init__(self, *args, **kwargs):
        """ Create the obj, with the same arguments as `dict()`; replaced by `dict.__init__` below """
        dict.__init__(self, *args, **kwargs)

    def __new__(cls, *args, **kwargs):
        if 'defaultdict' in kwargs:
            errormsg = 'sc.obj() does not support "defaultdict"; use sc.objdict() instead'
            raise TypeError(errormsg)
        self = dict.__new__(cls)
        self.__dict__ = self # Use the dict itself to store the attributes
        return self

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _protect_methods(cls, dict(vars(cls)))
        if '__setitem__' in vars(cls): # Only these classes need code to run on setting an attribute
            cls.__setattr__ = cls._setattr_item
        return

    def _setattr_item(self, name, value):
        """ Replaces `__setattr__()` in subclasses that override `__setitem__()` """
        if hasattr(type(self), name): # A slot, or raises an exception for a method
            return object.__setattr__(self, name, value)
        self[name] = value
        return

    def __missing__(self, key):
        """ Handle what isn't a key: integer indices, and slices as of Python 3.12 (before that they're unhashable) """
        if isinstance(key, sc._numtype):
            return dict.__getitem__(self, self._ikey(key))
        elif isinstance(key, slice):
            return self._sanitize_items([dict.__getitem__(self, k) for k in self._slicetokeys(key)])
        keys = list(dict.keys(self))
        if len(keys): errormsg = f'obj key "{key}" not found; available keys are:\n{sc.newlinejoin(keys)}'
        else:         errormsg = f'Key {key} not found since obj is empty'
        raise sc.KeyNotFoundError(errormsg)

    def __reduce__(self):
        """ Pickle and copy the contents once, rather than as both items and attributes """
        slots = {k:getattr(self, k) for k in copyreg._slotnames(type(self)) if hasattr(self, k)}
        return (copyreg.__newobj__, (self.__class__,), (dict(self), slots))

    def __setstate__(self, state):
        """ Restore from a pickle or copy, including pickles from before version 3.7.0 """
        items, slots = state if isinstance(state, tuple) else (state, {})
        dict.update(self, items)
        for k,v in slots.items():
            object.__setattr__(self, k, v)
        return

    def setattribute(self, name, value):
        """ Set an attribute declared in `__slots__` """
        if not hasattr(type(self), name):
            errormsg = f'To set "{name}" as an attribute of {type(self).__name__}, add it to __slots__'
            raise AttributeError(errormsg)
        return object.__setattr__(self, name, value)

    def getattribute(self, name):
        """ Get an attribute, not a key """
        return object.__getattribute__(self, name)

    def __dir__(self):
        """ Include keys, for tab completion """
        return [k for k in object.__dir__(self) if isinstance(k, str)]

    def _ikey(self, key):
        """ Handle an integer key; the keys can't be cached since nothing runs when one is added """
        try:
            return list(dict.keys(self))[key]
        except IndexError: # pragma: no cover
            errormsg = f'index {key} out of range for dict of length {len(self)}'
            raise IndexError(errormsg) from None # Don't show the traceback


obj.__init__ = dict.__init__ # Set here since quartodoc can't read __init__ from an assignment in the class
_protect_methods(obj, {**vars(dict), **vars(odict), **vars(obj)})


class dictobj(dict):
    """
    Lightweight class to create an object that can also act like a dictionary.

    **Example**:

    ```python
    obj = sc.dictobj()
    obj.a = 5
    obj['b'] = 10
    print(obj.items())
    ```
    For the same speed with more features, see `sc.obj()`.

    The keys and the attributes are the same thing (the dict is its own `__dict__`),
    so it is as fast as a plain object for attributes and nearly as fast as a plain
    dict for items. This also means that keys can overwrite methods (e.g. `obj.items = 5`).

    - *New in version 1.3.0.*
    - *New in version 1.3.1:* inherit from dict
    - *New in version 2.0.0:* allow positional arguments
    - *New in version 3.0.0:* "fromkeys" now a class method; `to_json()` method
    - *New in version 3.1.6:* "copy" returns another dictobj
    - *New in version 3.5.0:* keys are stored in the dict itself, so e.g. `json.dumps()` works; faster
    """

    def __new__(cls, *args, **kwargs):
        self = dict.__new__(cls)
        self.__dict__ = self # Use the dict itself to store the attributes
        return self

    def __init__(self, *args, **kwargs):
        for arg in args:
            if arg is not None:
                self.update(arg)
        self.update(kwargs)
        return

    def __repr__(self):
        output = 'dictobj(' + dict.__repr__(self) + ')'
        return output

    def __reduce__(self):
        """ Pickle and copy the contents once, rather than as both items and attributes """
        return (copyreg.__newobj__, (self.__class__,), dict(self))

    def __setstate__(self, state):
        """ Restore from a pickle or copy, including pickles from before version 3.5.0 """
        self.update(state)
        return

    def to_json(self):
        """ Export the dictobj to JSON """
        return json.dumps(self)

    def copy(self):
        """ Create a shallow copy """
        new = self.__class__()
        new.update(self)
        return new

    def __or__(self, other):
        new = self.copy()
        new.update(other)
        return new

    def __ror__(self, other):
        new = self.__class__(other)
        new.update(self)
        return new


def asobj(obj, strict=True):
    """
    Convert any object for which you would normally do `a['b']` to one where you
    can do `a.b`.

    Note: this may lead to unexpected behavior in some cases. Use at your own risk.
    At minimum, objects created using this function have an extremely odd type -- namely
    `sciris.sc_odict.asobj.<locals>.objobj`.

    Args:
        obj (anything): the object you want to convert
        strict (bool): whether to raise an exception if an attribute is being set (instead of a key)

    **Example**:

    ```python
    d = dict(foo=1, bar=2)
    d_obj = sc.asobj(d)
    d_obj.foo = 10
    ```
    *New in version 1.0.0.*
    """

    objtype = type(obj)

    class objobj(objtype):
        """ Define the "objobj" "type": convert any object to, well, an object. """

        def __getattribute__(self, attr):
            try: # First, try to get the attribute as an attribute
                output = objtype.__getattribute__(self, attr)
                return output
            except Exception as E: # pragma: no cover # If that fails, try to get it as a dict item
                try:
                    output = objtype.__getitem__(self, attr)
                    return output
                except: # If that fails, raise the original exception
                    raise E

        def __setattr__(self, name, value):
            """ Set key in dict, not attribute! """
            try:
                objtype.__getattribute__(self, name) # Try retrieving this as an attribute, expect AttributeError...
            except AttributeError:
                return objtype.__setitem__(self, name, value) # If so, simply return

            if not strict: # Let the attribute be set anyway # pragma: no cover
                return objtype.__setattr__(self, name, value)
            else: # pragma: no cover # Otherwise, raise an exception
                errormsg = f'"{name}" exists as an attribute, so cannot be set as key; use setattribute() instead'
                raise ValueError(errormsg)

        def getattribute(self, name):
            """ Get attribute if truly desired """
            return objtype.__getattribute__(self, name)

        def setattribute(self, name, value):
            """ Set attribute if truly desired """
            return objtype.__setattr__(self, name, value)


    return objobj(obj)


class argparse(objdict):
    """ Ultra-simple argument parser

    Accepts positional or keyword arguments, and converts them to the correct type.
    While Python's built in argparse has more features (such as help for each argument),
    this allows single-line parsing of arguments.

    Args:
        parse (bool): whether to parse the arguments immediately (default True)
        **kwargs (dict): keyword arguments to add to the parser

    Returns:
        args (sc.obj): a dictionary-like object with the arguments

    **Examples**:

    ```python
    # Option 1: Supply arguments directly
    args = sc.argparse(iterations=10, output_file='results.csv')

    # Option 2: Add arguments one by one
    args = sc.argparse()
    args.add(iterations=10)
    args.add(output_file='results.csv')
    args.parse()

    # Command-line usage
    python argparse_example.py 100 'data.csv'
    python argparse_example.py 100 output_file='data.csv'
    python argparse_example.py iterations=100 --output_file='data.csv'
    python argparse_example.py --iterations 100 --output-file data.csv

    # Result
    args.iterations == 100
    args.output_file == 'data.csv'

    # Boolean flags
    args = sc.argparse(full=False, verbose=False)
    python argparse_example.py --verbose # Sets verbose=True
    ```
    - *New in version 3.2.6.*
    - *New in version 3.4.0:* bare flags (e.g. `--verbose`) set boolean arguments to True, and `--key value` is supported
    """
    def __init__(self, parse=True, **kwargs):
        self.update(kwargs)
        self.setattribute('_parsed', False)
        if parse and len(kwargs): # If supplied, parse immediately
            self.parse()
        return

    def add(self, **kwargs):
        """ Add an argument """
        if self._parsed:
            errormsg = 'Cannot add an argument to an already parsed object'
            raise ValueError(errormsg)
        self.update(kwargs)
        return

    def parse(self):
        """ Parse the arguments into the dictionary """
        args = []
        kw = {}

        def getkey(key):
            key = key.lstrip('-').strip()
            if key not in self and key.replace('-', '_') in self: # Allow e.g. --output-file for output_file
                key = key.replace('-', '_')
            return key

        items = sys.argv[1:]
        i = 0
        while i < len(items):
            item = items[i]
            if '=' in item: # E.g. key=value or --key=value
                key, val = item.split('=', 1)
                kw[getkey(key)] = val.strip()
            elif item.startswith('--'): # E.g. --flag or --key value
                key = getkey(item)
                if key not in self:
                    errormsg = f'Unrecognized argument {item}; valid arguments are:\n{sc.strjoin(self.keys())}'
                    raise sc.KeyNotFoundError(errormsg)
                if isinstance(self[key], bool): # A bare flag sets a boolean argument to True
                    kw[key] = 'True'
                elif i+1 < len(items): # Otherwise, use the next item as the value
                    i += 1
                    kw[key] = items[i]
                else:
                    errormsg = f'No value supplied for argument {item}'
                    raise ValueError(errormsg)
            else:
                if kw:
                    errormsg = 'Cannot supply a positional argument after a keyword argument'
                    raise ValueError(errormsg)
                else:
                    args.append(item)
            i += 1

        def convert(default, arg):
            if isinstance(default, bool): # bool('False') is True, so handle explicitly
                if   arg.lower() in ['1', 'true', 't', 'yes', 'y']:  return True
                elif arg.lower() in ['0', 'false', 'f', 'no', 'n']: return False
                else: raise ValueError(f'"{arg}" is not a recognized boolean value')
            elif isinstance(default, (list, tuple)): # Split on commas, e.g. "a,b" -> ['a','b']
                eldefault = default[0] if len(default) else ''
                return default.__class__(convert(eldefault, a.strip()) for a in arg.split(','))
            else:
                return default.__class__(arg)

        def keep_type(key, arg):
            out = arg
            if self[key] is not None:
                default_type = self[key].__class__
                try:
                    out = convert(self[key], arg)
                except Exception as e:
                    errormsg = f'Could not convert "{arg}" to {default_type}, leaving as string.\n{e}'
                    print(errormsg) # TODO: convert to warning
                    out = arg
            return out

        keys = self.keys()
        if len(args) > len(keys):
            errormsg = f'{len(args)} positional arguments were supplied, but only {len(keys)} are valid:\n{sc.strjoin(keys)}'
            raise ValueError(errormsg)
        for i,arg in enumerate(args):
            key = keys[i]
            self[key] = keep_type(key, arg)

        for key,val in kw.items():
            if key not in self:
                errormsg = f'Unrecognized key {key}; valid arguments are:\n{sc.strjoin(self.keys())}'
                raise sc.KeyNotFoundError(errormsg)
            self[key] = keep_type(key, val)
        self.setattribute('_parsed', True)

        return self
