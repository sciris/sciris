'''
The 'odict' class, combining features from an OrderedDict and a list/array.

Highlights:
    - ``sc.odict()``: flexible container representing the best-of-all-worlds across lists, dicts, and arrays
    - ``sc.objdict()``: like an odict, but allows get/set via e.g. ``foo.bar`` instead of ``foo['bar']``
'''

##############################################################################
### ODICT CLASS
##############################################################################

import re
import numpy as np
from collections import OrderedDict as OD, defaultdict as ddict
from . import sc_utils as scu
from . import sc_printing as scp
from . import sc_nested as scn

# Restrict imports to user-facing modules
__all__ = ['ddict', 'odict', 'objdict', 'asobj']


class odict(OD):
    '''
    Ordered dictionary with integer indexing


    An ordered dictionary, like the OrderedDict class, but supports list methods like integer
    indexing, key slicing, and item inserting. It can also replicate defaultdict behavior
    via the ``defaultdict`` argument.

    **Examples**::

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

    Note: by default, integers are used as an alias to string keys, so cannot be used
    as keys directly. However, you can force regular-dict behavior using ``setitem()``,
    and you can convert a dictionary with integer keys to an odict using ``sc.odict.makefrom()``.
    If an odict has integer keys and the keys do not match the key positions, then the
    key itself will take precedence (e.g., ``od[3]`` is equivalent to ``dict(od)[3]``,
    not ``dict(od)[od.keys()[3]]``). This usage is discouraged.

    | New in version 1.1.0: "defaultdict" argument
    | New in version 1.3.1: allow integer keys via ``makefrom()``; removed ``to_OD``; performance improvements
    '''

    def __init__(self, *args, defaultdict=None, **kwargs):
        ''' Combine the init properties of both OrderedDict and defaultdict '''
        if len(args)==1 and args[0] is None: args = [] # Remove a None argument
        OD.__init__(self, *args, **kwargs) # Standard init
        if defaultdict is not None:
            if defaultdict != 'nested' and not callable(defaultdict): # pragma: no cover
                errormsg = f'The defaultdict argument must be either "nested" or callable, not {type(defaultdict)}'
                raise TypeError(errormsg)
            self._setattr('_defaultdict', defaultdict) # Use OD.__setattr__() since setattr() is overridden by sc.objdict()
        self._cache_keys()
        return


    def _cache_keys(self):
        ''' Store a copy of the keys as a list so integer lookup doesn't have to regenerate it each time '''
        self._setattr('_cached_keys', self.keys())
        self._setattr('_stale', False)
        return


    def _setattr(self, key, value):
        ''' Shortcut to OrderedDict method '''
        return OD.__setattr__(self, key, value)


    def _setitem(self, key, value):
        ''' Shortcut to OrderedDict method '''
        return OD.__setitem__(self, key, value)


    def __getitem__(self, key, allow_default=True):
        ''' Allows getitem to support strings, integers, slices, lists, or arrays '''

        # First, try just getting the item
        try:
            output = OD.__getitem__(self, key)
            return output
        except Exception as E:

            if isinstance(key, scu._stringtypes) or isinstance(key, tuple): # Normal use case: just use a string key
                if isinstance(E, KeyError): # We already encountered an exception, usually a KeyError
                    try: # Handle defaultdict behavior by first checking if it exists
                        _defaultdict = OD.__getattribute__(self, '_defaultdict')
                    except:
                        _defaultdict = None
                    if _defaultdict is not None and allow_default: # If it does, use it, then get the key again
                        if _defaultdict == 'nested':
                            _defaultdict = lambda: self.__class__(defaultdict=_defaultdict) # Make recursive
                        dd = _defaultdict() # Create the new object
                        self._setitem(key, dd) # Add it to the dictionary
                        self._setattr('_stale', True) # Flag to refresh the cached keys
                        return dd # Return
                    else:
                        keys = self.keys()
                        if len(keys): errormsg = f'odict key "{key}" not found; available keys are:\n{scu.newlinejoin(keys)}'
                        else:         errormsg = f'Key {key} not found since odict is empty'
                        raise scu.KeyNotFoundError(errormsg)
                else: # Exception raised wasn't a key error -- just raise it again
                    raise E

            elif isinstance(key, scu._numtype): # Convert automatically from float
                thiskey = self._ikey(key)
                return OD.__getitem__(self, thiskey) # Note that defaultdict behavior isn't supported for non-string lookup

            elif type(key)==slice: # Handle a slice -- complicated
                try:
                    startind = self._slicekey(key.start, 'start')
                    stopind = self._slicekey(key.stop, 'stop')
                    if stopind<startind: # pragma: no cover
                        errormsg = f'Stop index must be >= start index (start={startind}, stop={stopind})'
                        raise ValueError(errormsg)
                    slicevals = [self.__getitem__(i) for i in range(startind,stopind)]
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
                return OD.__getitem__(self,key)


    def __setitem__(self, key, value):
        ''' Allows setitem to support strings, integers, slices, lists, or arrays '''

        self._setattr('_stale', True) # Flag to refresh the cached keys

        if isinstance(key, (str,tuple)):
            self._setitem(key, value)

        elif isinstance(key, scu._numtype): # Convert automatically from float...dangerous?
            thiskey = self._ikey(key)
            self._setitem(thiskey, value)

        elif type(key)==slice:
            startind = self._slicekey(key.start, 'start')
            stopind = self._slicekey(key.stop, 'stop')
            if stopind<startind: # pragma: no cover
                errormsg = f'Stop index must be >= start index (start={startind}, stop={stopind})'
                raise ValueError(errormsg)
            slicerange = range(startind,stopind)
            enumerator = enumerate(slicerange)
            slicelen = len(slicerange)
            if hasattr(value, '__len__'):
                if len(value)==slicelen:
                    for valind,index in enumerator:
                        self.__setitem__(index, value[valind])  # e.g. odict[:] = arr[:]
                else: # pragma: no cover
                    errormsg = f'Slice "{slicerange}" and values "{value}" have different lengths! ({slicelen}, {len(value)})'
                    raise ValueError(errormsg)
            else:
                for valind,index in enumerator:
                    self.__setitem__(index, value) # e.g. odict[:] = 4

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
        ''' Use regular dictionary ``setitem``, rather than odict's '''
        self._setattr('_stale', True) # Flag to refresh the cached keys
        self._setitem(key, value)
        return


    def __repr__(self, maxlen=None, showmultilines=True, divider=False, dividerthresh=10, numindents=0, recursionlevel=0, sigfigs=None, numformat=None, maxitems=200, classname='odict()', quote='', numleft='#', numsep=':', keysep=':', dividerchar='—'):
        ''' Print a meaningful representation of the odict '''

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
            output = classname
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
                            thisvalstr = scu.flexstr(thisval.__repr__(maxlen=maxlen, showmultilines=showmultilines, divider=divider, dividerthresh=dividerthresh, numindents=numindents, recursionlevel=recursionlevel+1, sigfigs=sigfigs, numformat=numformat))
                        else:
                            thisvalstr = f'{classname} [maximum recursion reached]'
                    elif scu.isnumber(thisval): # Flexibly print out numbers, since they're largely why we're here
                        if numformat is not None:
                            thisvalstr = numformat % thisval
                        elif sigfigs is not None:
                            thisvalstr = scp.sigfig(thisval, sigfigs=sigfigs)
                        else:
                            thisvalstr = scu.flexstr(thisval) # To avoid numpy's stupid 0.4999999999945
                    else: # Otherwise, do the normal repr() read.
                        thisvalstr = repr(thisval)
                except Exception as E: # pragma: no cover
                    thisvalstr = f'{scp.objectid(thisval)} read failed: {str(E)}'

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
                    newoutput = scp.indent(prefix=theprefix, text=dividerstr, width=80)
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

                # Perform the indentation.
                newoutput = scp.indent(prefix=theprefix, text=rawoutput, width=80)

                # Strip ot any terminal newline.
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
        ''' Function to fix __repr__ in IPython'''
        print(self.__repr__(*args, **kwargs))
        return


    def __add__(self, dict2):
        '''
        Allow two dictionaries to be added (merged).

        **Example**::

            dict1 = sc.odict(a=3, b=4)
            dict2 = sc.odict(c=5, d=7)
            dict3 = dict1 + dict2
        '''
        return scu.mergedicts(self, dict2)


    def __radd__(self, dict2):
        ''' Allows sum() to work correctly '''
        if not dict2: return self
        else:         return self.__add__(dict2)


    def __delitem__(self, *args, **kwargs):
        ''' Default delitem, except set stale to true '''
        self._setattr('_stale', True) # Flag to refresh the cached keys
        return OD.__delitem__(self, *args, **kwargs)


    def disp(self, maxlen=None, showmultilines=True, divider=False, dividerthresh=10, numindents=0, sigfigs=5, numformat=None, maxitems=20, **kwargs):
        '''
        Print out flexible representation, short by default.

        **Example**::

            z = sc.odict().make(keys=['a','b','c'], vals=[4.293487,3,6])
            z.disp(sigfigs=3)
            z.disp(numformat='%0.6f')
        '''
        kwargs = scu.mergedicts(dict(maxlen=maxlen, showmultilines=showmultilines, divider=divider, dividerthresh=dividerthresh, numindents=numindents, recursionlevel=0, sigfigs=sigfigs, numformat=None, maxitems=maxitems), kwargs)
        print(self.__repr__(**kwargs))
        return


    def _ikey(self, key):
        ''' Handle an integer key '''
        if self._stale:
            self._cache_keys()
        try:
            return self._cached_keys[key]
        except IndexError:
            errormsg = f'index {key} out of range for dict of length {len(self)}'
            raise IndexError(errormsg) from None # Don't show the traceback


    def _slicekey(self, key, slice_end):
        ''' Validate a key supplied as a slice object '''
        if slice_end == 'stop':
            shift = 1
            default = len(self)
        else:
            shift = 0
            default = 0

        if isinstance(key, scu._numtype):
            if key < 0:
                key = len(self) + key
            output = key
        elif isinstance(key, str):
            output = self.index(key) + shift # +1 since otherwise confusing with names (CK)
        elif key is None:
            output = default
        else: # pragma: no cover
            errormsg = f'To use a slice, {slice_end} must be either int or str ({key})'
            raise TypeError(errormsg)

        return output


    @staticmethod
    def _matchkey(key, pattern, method):
        ''' Helper function for findkeys '''
        match = False
        if isinstance(key, tuple):
            for item in key:
                if odict._matchkey(item, pattern, method):
                    return True
        else: # For everything except a tuple, treat it as a string
            if not scu.isstring(key):
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
        ''' Check to see whether the "key" is actually an iterable '''
        output = isinstance(key, (list, np.ndarray))
        return output


    def _sanitize_items(self, items):
        ''' Try to convert the output of a slice to an array, but give up easily and return a list '''
        try:
            output = np.array(items) # Try standard Numpy array...
            if 'S' in str(output.dtype) or 'U' in str(output.dtype): # ...but instead of converting to string, convert to object array for Python 2 or 3 -- WARNING, fragile!
                output = np.array(items, dtype=object)
        except: # pragma: no cover
            output = items # If that fails, just give up and return the list
        return output


    def export(self, doprint=True):
        ''' Export the odict in a form that is valid Python code '''
        start = 'odict(['
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
        ''' Allows pop to support strings, integers, slices, lists, or arrays '''
        self._setattr('_stale', True) # Flag to refresh the cached keys
        if isinstance(key, scu._stringtypes):
            return OD.pop(self, key, *args, **kwargs)
        elif isinstance(key, scu._numtype): # Convert automatically from float...dangerous?
            thiskey = self._ikey(key)
            return OD.pop(self, thiskey, *args, **kwargs)
        elif type(key)==slice: # Handle a slice -- complicated
            try:
                startind = self._slicekey(key.start, 'start')
                stopind = self._slicekey(key.stop, 'stop')
                if stopind<startind: # pragma: no cover
                    errormsg = f'Stop index must be >= start index (start={startind}, stop={stopind})'
                    raise ValueError(errormsg)
                slicevals = [self.pop(i, *args, **kwargs) for i in range(startind,stopind)] # WARNING, not tested
                try:
                    return np.array(slicevals) # Try to convert to an array
                except: # pragma: no cover
                    return slicevals
            except Exception as E: # pragma: no cover
                errormsg = 'Invalid odict slice'
                raise ValueError(errormsg) from E
        elif self._is_odict_iterable(key): # Iterate over items
            keys = self.keys()
            poplist = [keys[int(item)] if isinstance(item, scu._numtype) else item for item in key] # Convert to text keys, because indices change
            listvals = [self.pop(item, *args, **kwargs) for item in poplist]
            try:
                return np.array(listvals)
            except: # pragma: no cover
                return listvals
        else: # pragma: no cover # Handle string but also everything else
            try:
                return OD.pop(self, key, *args, **kwargs)
            except: # Duplicated from __getitem__
                keys = self.keys()
                if len(keys): errormsg = f'odict key "{key}" not found; available keys are:\n{scu.newlinejoin(keys)}'
                else:         errormsg = f'Key {key} not found since odict is empty'
                raise scu.KeyNotFoundError(errormsg)


    def remove(self, key, *args, **kwargs):
        ''' Remove an item by key and do not return it '''
        self.pop(key, *args, **kwargs)
        return


    def clear(self):
        ''' Reset to an empty odict '''
        for key in self.keys():
            self.remove(key)
        return


    def index(self, value):
        ''' Return the index of a given key '''
        return self.keys().index(value)


    def valind(self, value):
        ''' Return the index of a given value '''
        return self.values().index(value)


    def findkeys(self, pattern=None, method=None, first=None):
        '''
        Find all keys that match a given pattern. By default uses regex, but other options
        are 'find', 'startswith', 'endswith'. Can also specify whether or not to only return
        the first result (default false). If the key is a tuple instead of a string, it will
        search each element of the tuple.
        '''
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
        ''' Same as findkeys, but returns values instead '''
        keys = self.findkeys(pattern=pattern, method=method, first=first)
        if not first and len(keys) == 1: keys = keys[0] # If it's a list of one element, turn it into that element instead
        return self.__getitem__(keys)


    def findbyval(self, value, first=True, strict=False):
        '''
        Returns the key(s) that match a given value -- reverse of findbykey, except here
        uses exact matches to the value or values provided.

        **Example**::

            z = odict({'dog':[2,3], 'cat':[4,6], 'mongoose':[4,6]})
            z.findvals([4,6]) # returns 'cat'
            z.findvals([4,6], first=False) # returns ['cat', 'mongoose']
        '''
        keys = []
        for key,val in self.items():
            if val == value:  # Exact match, return a match
                match = True
            elif not strict and isinstance(value, list) and val in value: # "value" is a list and it's contained
                match = True
            else:
                match = False
            if match:
                if first: return key
                else:     keys.append(key)
        return keys


    def filter(self, keys=None, pattern=None, method=None, exclude=False):
        '''
        Filter the odict keys and return a new odict which is a subset. If keys is a list,
        then uses that for matching. If the first argument is a string, then treats as a pattern
        for matching using findkeys(). If exclude=True, then will exclude rather than include matches.
        '''
        if scu.isstring(keys) and pattern is None: # Assume first argument, transfer
            pattern = keys
            keys = None
        filtered = odict()
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
        ''' Like filter, but filters by value rather than key '''
        keys = self.findbyval(value)
        return self.filter(keys=keys)


    def append(self, key=None, value=None):
        ''' Support an append method, like a list '''
        needkey = False
        if value is None: # Assume called with a single argument
            value = key
            needkey = True
        if key is None or needkey:
            keyname = 'key'+scu.flexstr(len(self))  # Define the key just to be the current index
        else:
            keyname = key
        self.__setitem__(keyname, value)
        return


    def insert(self, pos=None, key=None, value=None):
        '''
        Function to do insert a key -- note, computationally inefficient.

        **Example**::

            z = odict()
            z['foo'] = 1492
            z.insert(1604)
            z.insert(0, 'ganges', 1444)
            z.insert(2, 'meikang', 1234)
        '''

        # Handle inputs
        realpos, realkey, realvalue = pos, key, value
        if key is None and value is None: # Assume it's called like odict.insert(666)
            realvalue = pos
            realkey = 'key'+scu.flexstr(len(self))
            realpos = 0
        elif value is None: # Assume it's called like odict.insert('devil', 666)
            realvalue = key
            realkey = pos
            realpos = 0
        if pos is None:
            realpos = 0
        if realpos>len(self): # pragma: no cover
            errormsg = f'Cannot insert {key} at position {pos} since length of odict is {len(self)}'
            raise ValueError(errormsg)

        # Create a temporary dictionary to hold all of the items after the insertion point
        tmpdict = odict()
        origkeys = self.keys()
        originds = range(len(origkeys))
        if not len(originds) or realpos==len(originds): # It's empty or in the final position, just append
            self.__setitem__(realkey, realvalue)
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


    def copy(self, oldkey, newkey):
        ''' Make a copy of an item '''
        newval = scu.dcp(self.__getitem__(oldkey))
        self.__setitem__(newkey, newval)
        return


    def rename(self, oldkey, newkey):
        ''' Change a key name -- WARNING, very inefficient! '''
        nkeys = len(self)
        if isinstance(oldkey, scu._numtype):
            index = oldkey
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


    def sort(self, sortby=None, reverse=False, copy=False, verbose=True):
        '''
        Create a sorted version of the odict. Sorts by order of sortby, if provided, otherwise alphabetical.
        If copy is True, then returns a copy (like sorted()).

        Note that you can also use this to do filtering.
        '''
        origkeys = self.keys()
        if sortby is None or sortby == 'keys':
            allkeys = sorted(origkeys)
        else:
            if sortby == 'values':
                origvals = self.values()
                sortby = sorted(range(len(origvals)), key=origvals.__getitem__) # Reset sortby based on https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
            if not scu.isiterable(sortby): # pragma: no cover
                raise Exception('Please provide a list to determine the sort order.')
            if all([isinstance(x, scu._stringtypes) for x in sortby]): # Going to sort by keys
                allkeys = sortby # Assume the user knows what s/he is doing
            elif all([isinstance(x,bool) for x in sortby]): # Using Boolean values to filter
                allkeys = []
                for i,x in enumerate(sortby):
                     if x: allkeys.append(origkeys[i])
            elif all([isinstance(x, scu._numtype) for x in sortby]): # Going to sort by numbers
                if not set(sortby)==set(range(len(self))): # pragma: no cover
                    warningmsg = f'Warning: list to sort by "{sortby}" has different length than odict "{len(self)}"'
                    if verbose: print(warningmsg)
                allkeys = [origkeys[ind] for ind in sortby]
            else: # pragma: no cover
                errormsg = f'Cannot figure out how to sort by "{sortby}"'
                raise TypeError(errormsg)
        tmpdict = odict()
        if reverse:
            allkeys.reverse() # If requested, reverse order
        if copy:
            for key in allkeys: tmpdict[key] = self[key]
            return tmpdict
        else:
            for key in allkeys: tmpdict.__setitem__(key, self.pop(key))
            for key in allkeys: self.__setitem__(key, tmpdict.pop(key))
            return


    def sorted(self, sortby=None, reverse=False):
        ''' Shortcut for making a copy of the sorted odict -- see sort() for options '''
        return self.sort(sortby=sortby, copy=True, reverse=reverse)


    def reverse(self, copy=False):
        ''' Reverse the order of an odict '''
        reversedkeys = self.keys()
        reversedkeys.reverse()
        output = self.sort(sortby=reversedkeys, copy=copy)
        return output


    def reversed(self):
        ''' Shortcut for making a copy of the sorted odict '''
        return self.reverse(copy=True)


    def make(self, keys=None, vals=None, keys2=None, keys3=None, coerce='full'):
        '''
        An alternate way of making or adding to an odict.

        Args:
            keys (list/int): the list of keys to use
            vals (list/arr): the list of values to use
            keys2 (list/int): for a second level of nesting
            keys3 (list/int): for a third level of nesting
            coerce (str): what types to coerce into being separate dict entries

        **Examples**::

            a = sc.odict().make(5) # Make an odict of length 5, populated with Nones and default key names
            b = sc.odict().make('foo',34) # Make an odict with a single key 'foo' of value 34
            c = sc.odict().make(['a','b']) # Make an odict with keys 'a' and 'b'
            d = sc.odict().make(['a','b'], 0) # Make an odict with keys 'a' and 'b', initialized to 0
            e = sc.odict().make(keys=['a','b'], vals=[1,2]) # Make an odict with 'a':1 and 'b':2
            f = sc.odict().make(keys=['a','b'], vals=np.array([1,2])) # As above, since arrays are coerced into lists
            g = sc.odict({'a':34, 'b':58}).make(['c','d'],[99,45]) # Add extra keys to an exising odict
            h = sc.odict().make(keys=['a','b','c'], keys2=['A','B','C'], keys3=['x','y','z'], vals=0) # Make a triply nested odict

        New in version 1.2.2: "coerce" argument
        '''
        # Handle keys
        keylist = []
        if keys is None and vals is None:
            return self # Nothing to do if nothing supplied
        if keys is None and vals is not None:
            keys = len(scu.promotetolist(vals)) # Values are supplied but keys aren't: use default keys
        if isinstance(keys, scu._numtype): # It's a single number: pre-generate
            keylist = ['%i'%i for i in range(keys)] # Generate keylist
        elif isinstance(keys, scu._stringtypes): # It's a single string
            keylist = [scu.flexstr(keys)]
        elif isinstance(keys, list): # It's a list: use directly
            keylist = keys
        else: # pragma: no cover
            errormsg = f'Could not understand keys "{keys}": must be number, string, or list'
            raise TypeError(errormsg)
        nkeys = len(keylist)

        # Handle values
        vals = scu.promotetolist(vals, coerce=coerce)
        nvals = len(vals)
        if nvals==0: # Special case: it's an empty list
            vallist = [scu.dcp(vals) for _ in range(nkeys)]
        elif nvals==1: # Only a single value: duplicate it
            vallist = [scu.dcp(vals[0]) for _ in range(nkeys)]
        elif nvals==nkeys: # Lengths match, can use directly
            vallist = vals
        else: # pragma: no cover
            errormsg = f'Must supply either a single value or a list of same length as the keys ({nkeys} keys, {nvals} values supplied)'
            raise ValueError(errormsg)

        # Handle nested keys -- warning, would be better to not hard-code this, but does the brain in as it is!
        if keys2 is not None and keys3 is not None: # Doubly nested
            self.make(keys=keys, vals=odict().make(keys=keys2, vals=odict().make(keys=keys3, vals=vals)))
        elif keys2 is not None: # Singly nested
            self.make(keys=keys, vals=odict().make(keys=keys2, vals=vals))
        else: # Not nested -- normal case of making an odict
            for key,val in zip(keylist,vallist): # Update odict
                self.__setitem__(key, val)

        return self # Return the odict


    @staticmethod
    def makefrom(source=None, include=None, keynames=None, force=True, *args, **kwargs):
        '''
        Create an odict from entries in another dictionary. If keys is None, then
        use all keys from the current dictionary.

        Args:
            source (dict/list/etc): the item(s) to convert to an odict
            include (list): list of keys to include from the source dict in the odict (default: all)
            keynames (list): names of keys if source is not a dict
            force (bool): whether to force conversion to an odict even if e.g. the source has numeric keys

        **Examples**::

            a = 'cat'
            b = 'dog'
            o = sc.odict.makefrom(source=locals(), include=['a','b']) # Make use of fact that variables are stored in a dictionary

            d = {'a':'cat', 'b':'dog'}
            o = sc.odict.makefrom(d) # Same as odict(d)
            l = ['cat', 'monkey', 'dog']
            o = sc.odict.makefrom(source=l, include=[0,2], keynames=['a','b'])

            d = {12:'monkeys', 3:'musketeers'}
            o = sc.odict.makefrom(d)
        '''
        # Initialize new odict
        out = odict()

        # Make sure it's iterable
        if source is not None: # Don't do anything if there's nothing there
            if not(scu.isiterable(source)): # Make sure it's iterable
                source = scu.promotetolist(source)
            elif isinstance(source, scu._stringtypes):
                source = [source] # Special case -- strings are iterable, but we don't want to

            if len(source)==0:
                return out # Nothing to do here
            else:
                # Handle cases where keys or keynames are not supplied
                if include is None:
                    if   isinstance(source, (list, tuple)):   include = range(len(source))
                    elif isinstance(source, dict):            include = list(source.keys())
                    else:                                     raise TypeError(f'Unable to guess keys for object of type {type(source)}')
                include = scu.promotetolist(include) # Make sure it's a list -- note, does not convert other iterables to a list!
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
                        raise scu.KeyNotFoundError(errormsg) from E

        return out # As with make()


    def map(self, func=None):
        '''
        Apply a function to each element of the odict, returning
        a new odict with the same keys.

        **Example**::

            cat = odict({'a':[1,2], 'b':[3,4]})
            def myfunc(mylist): return [i**2 for i in mylist]
            dog = cat.map(myfunc) # Returns odict({'a':[1,4], 'b':[9,16]})
        '''
        output = odict()
        for key in self.keys():
            output[key] = func(self.__getitem__(key))
        return output


    def fromeach(self, ind=None, asdict=True):
        '''
        Take a "slice" across all the keys of an odict, applying the same
        operation to entry. The simplest usage is just to pick an index.
        However, you can also use it to apply a function to each key.

        **Example**::

            z = odict({'a':array([1,2,3,4]), 'b':array([5,6,7,8])})
            z.fromeach(2) # Returns array([3,7])
            z.fromeach(ind=[1,3], asdict=True) # Returns odict({'a':array([2,4]), 'b':array([6,8])})
        '''
        output = odict()
        for key in self.keys():
            output[key] = self.__getitem__(key)[ind]
        if asdict: return output # Output as a slimmed-down odict
        else:      return output[:] # Output as just the entries


    def toeach(self, ind=None, val=None):
        '''
        The inverse of fromeach: partially reset elements within
        each odict key.

        **Example**::

            z = odict({'a':[1,2,3,4], 'b':[5,6,7,8]})
            z.toeach(2, [10,20])    # z is now odict({'a':[1,2,10,4], 'b':[5,6,20,8]})
            z.toeach(ind=3,val=666) #  z is now odict({'a':[1,2,10,666], 'b':[5,6,20,666]})
        '''
        nkeys = len(self.keys())
        if not(scu.isiterable(val)): # Assume it's meant to be populated in each
            val = [val]*nkeys # Duplicated
        if len(val)!=nkeys: # pragma: no cover
            errormsg = f'To map values onto each key, they must be the same length ({len(val)} vs. {nkeys})'
            raise ValueError(errormsg)
        for k,key in self.enumkeys():
            self.__getitem__(key)[ind] = val[k]
        return


    def enumkeys(self, transpose=False):
        '''
        Shortcut for enumerate(odict.keys()).

        If transpose=True, return a tuple of lists rather than a list of tuples.
        '''
        iterator = list(enumerate(self.keys()))
        if transpose: iterator = tuple(scu.transposelist(iterator))
        return iterator


    def enumvals(self, transpose=False):
        '''
        Shortcut for enumerate(odict.values())

        If transpose=True, return a tuple of lists rather than a list of tuples.
        '''
        iterator = list(enumerate(self.values()))
        if transpose: iterator = tuple(scu.transposelist(iterator))
        return iterator


    def enumvalues(self, transpose=False):
        ''' Alias for enumvals(). New in version 1.2.0. '''
        return self.enumvals(transpose=transpose)


    def enumitems(self, transpose=False):
        '''
        Returns tuple of 3 things: index, key, value.

        If transpose=True, return a tuple of lists rather than a list of tuples.
        '''
        iterator = []
        for ind,item in enumerate(self.items()):
            thistuple = (ind,)+item # Combine into one tuple
            iterator.append(thistuple)
        if transpose: iterator = tuple(scu.transposelist(iterator))
        return iterator

    @staticmethod
    def promote(obj=None):
        '''
        Like promotetolist, but for odicts.

        **Example**::

            od = sc.odict.promote(['There','are',4,'keys'])

        Note, in most cases sc.odict(obj) or sc.odict().make(obj) can be used instead.
        '''
        if isinstance(obj, odict):
            return obj # Don't need to do anything
        elif isinstance(obj, dict):
            return odict(obj)
        elif isinstance(obj, list):
            newobj = odict()
            for i,val in enumerate(obj):
                newobj['Key %i'%i] = val
            return newobj
        else:
            return odict({'Key':obj})

    def keys(self):
        """ Return a list of keys (as in Python 2), not a dict_keys object. """
        return list(OD.keys(self))

    def values(self):
        """ Return a list of values (as in Python 2). """
        return list(OD.values(self))

    def items(self, transpose=False):
        """ Return a list of items (as in Python 2). """
        iterator = list(OD.items(self))
        if transpose: iterator = tuple(scu.transposelist(iterator))
        return iterator

    def iteritems(self, transpose=False):
        """ Alias to items() """
        return self.items(transpose=transpose)

    def makenested(self, *args, **kwargs):
        ''' Alias to sc.makenested(odict); see sc.makenested() for full documentation. New in version 1.2.0. '''
        return scn.makenested(self, *args, **kwargs)

    def getnested(self, *args, **kwargs):
        ''' Alias to sc.getnested(odict); see sc.makenested() for full documentation. New in version 1.2.0. '''
        return scn.getnested(self, *args, **kwargs)

    def setnested(self, *args, **kwargs):
        ''' Alias to sc.setnested(odict); see sc.makenested() for full documentation. New in version 1.2.0. '''
        return scn.setnested(self, *args, **kwargs)

    def iternested(self, *args, **kwargs):
        ''' Alias to sc.iternested(odict); see sc.makenested() for full documentation. New in version 1.2.0. '''
        return scn.iternested(self, *args, **kwargs)


class objdict(odict):
    '''
    An ``odict`` that acts like an object -- allow keys to be set/retrieved by object
    notation.

    Example
    -------
    >>> import sciris as sc
    >>> od = sc.objdict({'height':1.65, 'mass':59})
    >>> od.bmi = od.mass/od.height**2
    >>> od['bmi'] = od['mass']/od['height']**2 # Vanilla syntax still works
    >>> od.keys = 3 # This raises an exception (you can't overwrite the keys() method)

    Nested logic based in part on addict: https://github.com/mewwts/addict

    For a lighter-weight equivalent (based on ``dict`` instead of ``odict``), see
    ``sc.dictobj()``.
    '''

    def __init__(self, *args, **kwargs):
        nested_parent = kwargs.pop('_nested_parent', None)
        nested_attr  = kwargs.pop('_nested_attr', None)
        if nested_parent is not None:
            odict.__setattr__(self, '_nested_parent', nested_parent)
            odict.__setattr__(self, '_nested_attr', nested_attr)
        odict.__init__(self, *args, **kwargs) # Standard init
        return


    def __repr__(self, *args, **kwargs):
        ''' Use odict repr, but with a custom class name and no quotes '''
        return odict.__repr__(self, quote='', numsep='.', classname='objdict()', *args, **kwargs)


    def __getattribute__(self, attr):
        ''' Handle as attributes first, then as dict keys '''
        try: # First, try to get the attribute as an attribute
            return odict.__getattribute__(self, attr)
        except Exception as E: # If that fails, try to get it as a dict item, but pass along the original exception
            return self.__getitem__(attr, exception=E)


    def __setattr__(self, name, value):
        ''' Set key in dict, not attribute! '''
        try:
            odict.__getattribute__(self, name) # Try retrieving this as an attribute, expect AttributeError...
            if name[:2] == '__': # If it starts with a double underscore, it's almost certainly an attribute, not a key
                odict.__setattr__(self, name, value)
            else:
                errormsg = f'"{name}" exists as an attribute, so cannot be set as key; use setattribute() instead'
                raise ValueError(errormsg)
        except AttributeError:
            return self.__setitem__(name, value) # If so, simply return


    def __getitem__(self, attr, exception=None):
        ''' Handle dict keys, including nested defaultdict logic '''
        try: # Try retrieving normally
            return odict.__getitem__(self, attr, allow_default=False) # Do not allow odict to handle default
        except Exception as E2: # If that fails, raise the original exception
            try: # Handle defaultdict behavior by first checking if it exists
                _defaultdict = odict.__getattribute__(self, '_defaultdict')
            except:
                _defaultdict = None
            if _defaultdict is not None: # If it does, use it, then get the key again
                if _defaultdict == 'nested':
                    return self.__class__(defaultdict='nested', _nested_parent=self, _nested_attr=attr) # Create a recursive object with links back to the ultimate parent
                else:
                    dd = _defaultdict() # Create the new object
                    self[attr] = dd # Since we don't know what this object is, we can't be clever about recursion
                    return dd
            else:
                if exception:
                    raise exception
                else:
                    raise E2


    def __setitem__(self, name, value):
        ''' Set as a dict item '''
        odict.__setitem__(self, name, value)
        try:
            p = object.__getattribute__(self, '_nested_parent')
            key = object.__getattribute__(self, '_nested_attr')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '_nested_parent')
            object.__delattr__(self, '_nested_attr')
        return


    def __delattr__(self, name):
        ''' Delete dict key before deleting actual attribute '''
        try:
            del self[name]
        except:
            odict.__delattr__(self, name)
        return


    def getattribute(self, name):
        ''' Get attribute if truly desired '''
        return odict.__getattribute__(self, name)


    def setattribute(self, name, value, force=False):
        ''' Set attribute if truly desired '''
        if hasattr(self.__class__, name) and not force:
            errormsg = f'objdict attribute "{name}" is read-only'
            raise AttributeError(errormsg)
        return odict.__setattr__(self, name, value)


    def delattribute(self, name):
        ''' Delete attribute if truly desired '''
        try:
            del self[name]
        except:
            odict.__delattr__(self, name)
        return


def asobj(obj, strict=True):
    '''
    Convert any object for which you would normally do a['b'] to one where you
    can do a.b.

    Note: this may lead to unexpected behavior in some cases. Use at your own risk.
    At minimum, objects created using this function have an extremely odd type -- namely
    "sciris.sc_odict.asobj.<locals>.objobj".

    Args:
        obj (anything): the object you want to convert
        strict (bool): whether to raise an exception if an attribute is being set (instead of a key)

    **Example**::

        d = dict(foo=1, bar=2)
        d_obj = sc.asobj(d)
        d_obj.foo = 10

    New in version 1.0.0.
    '''

    objtype = type(obj)

    class objobj(objtype):

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
            ''' Set key in dict, not attribute! '''
            try:
                objtype.__getattribute__(self, name) # Try retrieving this as an attribute, expect AttributeError...
            except AttributeError:
                return objtype.__setitem__(self, name, value) # If so, simply return

            if not strict: # Let the attribute be set anyway
                objtype.__setattr__(self, name, value)
            else: # pragma: no cover # Otherwise, raise an exception
                errormsg = f'"{name}" exists as an attribute, so cannot be set as key; use setattribute() instead'
                raise ValueError(errormsg)

            return

        def getattribute(self, name):
            ''' Get attribute if truly desired '''
            return objtype.__getattribute__(self, name)

        def setattribute(self, name, value):
            ''' Set attribute if truly desired '''
            return objtype.__setattr__(self, name, value)

    return objobj(obj)
