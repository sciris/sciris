##############################################################################
### ODICT CLASS
##############################################################################

from collections import OrderedDict as OD
import re
import numpy as np
from . import sc_utils as ut

# Restrict imports to user-facing modules
__all__ = ['odict', 'objdict']

class odict(OD):
    '''
    An ordered dictionary, like the OrderedDict class, but supporting list methods like integer
    referencing, slicing, and appending.

    Examples:
        foo = odict({'ah':3,'boo':4, 'cough':6, 'dill': 8}) # Create odict
        bar = foo.sorted() # Sort the list
        assert(bar['boo'] == 4) # Show get item by value
        assert(bar[1] == 4) # Show get item by index
        assert((bar[0:2] == [3,4]).all()) # Show get item by slice
        assert((bar['cough':'dill'] == [6,8]).all()) # Show alternate slice notation
        assert((bar[[2,1]] == [6,4]).all()) # Show get item by list
        assert((bar[:] == [3,4,6,8]).all()) # Show slice with everything
        assert((bar[2:] == [6,8]).all()) # Show slice without end
        bar[3] = [3,4,5] # Show assignment by item
        bar[0:2] = ['the', 'power'] # Show assignment by slice -- NOTE, inclusive slice!!
        bar[[0,2]] = ['cat', 'trip'] # Show assignment by list
        bar.rename('cough','chill') # Show rename
        print(bar) # Print results

    Version: 2018jun25
    '''

    def __init__(self, *args, **kwargs):
        ''' See collections.py '''
        # Standard OrderedDictionary initialization
        if len(args)==1 and args[0] is None: args = [] # Remove a None argument
        OD.__init__(self, *args, **kwargs) # Standard init
        return None

    def __slicekey(self, key, slice_end):
        shift = int(slice_end=='stop')
        if isinstance(key, ut._numtype): return key
        elif type(key) is str: return self.index(key)+shift # +1 since otherwise confusing with names (CK)
        elif key is None: return (len(self) if shift else 0)
        else: raise Exception('To use a slice, %s must be either int or str (%s)' % (slice_end, key))
        return None


    def __isODict_iterable(self, key):
        ''' Check to see whether the "key" is actually an iterable '''
        output = type(key)==list or type(key)==type(np.array([])) # Do *not* include dict, since that would be recursive
        return output


    def __sanitize_items(self, items):
        ''' Try to convert the output of a slice to an array, but give up easily and return a list '''
        try:
            output = np.array(items) # Try standard Numpy array...
            if 'S' in str(output.dtype) or 'U' in str(output.dtype): # ...but instead of converting to string, convert to object array for Python 2 or 3 -- WARNING, fragile!
                output = np.array(items, dtype=object)
        except:
            output = items # If that fails, just give up and return the list
        return output



    def __getitem__(self, key):
        ''' Allows getitem to support strings, integers, slices, lists, or arrays '''
        if isinstance(key, ut._stringtypes) or isinstance(key, tuple):
            try:
                output = OD.__getitem__(self, key)
                return output
            except Exception as E: # WARNING, should be KeyError, but this can't print newlines!!!
                if len(self.keys()): errormsg = '%s\nodict key "%s" not found; available keys are:\n%s' % (repr(E), ut.flexstr(key), '\n'.join([ut.flexstr(k) for k in self.keys()]))
                else:                errormsg = 'Key "%s" not found since odict is empty'% key
                raise Exception(errormsg)
        elif isinstance(key, ut._numtype): # Convert automatically from float...dangerous?
            thiskey = self.keys()[int(key)]
            return OD.__getitem__(self,thiskey)
        elif type(key)==slice: # Handle a slice -- complicated
            try:
                startind = self.__slicekey(key.start, 'start')
                stopind = self.__slicekey(key.stop, 'stop')
                if stopind<startind:
                    print('Stop index must be >= start index (start=%i, stop=%i)' % (startind, stopind))
                    raise Exception
                slicevals = [self.__getitem__(i) for i in range(startind,stopind)]
                output = self.__sanitize_items(slicevals)
                return output
            except:
                print('Invalid odict slice... returning empty list...')
                return []
        elif self.__isODict_iterable(key): # Iterate over items
            listvals = [self.__getitem__(item) for item in key]
            if isinstance(key, list): # If the user supplied the keys as a list, assume they want the output as a list
                output = listvals
            else:
                output = self.__sanitize_items(listvals) # Otherwise, assume as an array
            return output
        else: # Handle everything else
            return OD.__getitem__(self,key)


    def __setitem__(self, key, value):
        ''' Allows setitem to support strings, integers, slices, lists, or arrays '''
        if isinstance(key, (str,tuple)):
            OD.__setitem__(self, key, value)
        elif isinstance(key, ut._numtype): # Convert automatically from float...dangerous?
            thiskey = self.keys()[int(key)]
            OD.__setitem__(self, thiskey, value)
        elif type(key)==slice:
            startind = self.__slicekey(key.start, 'start')
            stopind = self.__slicekey(key.stop, 'stop')
            if stopind<startind:
                errormsg = 'Stop index must be >= start index (start=%i, stop=%i)' % (startind, stopind)
                raise Exception(errormsg)
            slicerange = range(startind,stopind)
            enumerator = enumerate(slicerange)
            slicelen = len(slicerange)
            if hasattr(value, '__len__'):
                if len(value)==slicelen:
                    for valind,index in enumerator:
                        self.__setitem__(index, value[valind])  # e.g. odict[:] = arr[:]
                else:
                    errormsg = 'Slice "%s" and values "%s" have different lengths! (%i, %i)' % (slicerange, value, slicelen, len(value))
                    raise Exception(errormsg)
            else:
                for valind,index in enumerator:
                    self.__setitem__(index, value) # e.g. odict[:] = 4
        elif self.__isODict_iterable(key) and hasattr(value, '__len__'): # Iterate over items
            if len(key)==len(value):
                for valind,thiskey in enumerate(key):
                    self.__setitem__(thiskey, value[valind])
            else:
                errormsg = 'Keys "%s" and values "%s" have different lengths! (%i, %i)' % (key, value, len(key), len(value))
                raise Exception(errormsg)
        else:
            OD.__setitem__(self, key, value)
        return None


    def __repr__(self, maxlen=None, showmultilines=True, divider=False, dividerthresh=10, numindents=0, recursionlevel=0, sigfigs=None, numformat=None, maxitems=200):
        ''' Print a meaningful representation of the odict '''

        # Set primitives for display.
        toolong      = ' [...]'    # String to display at end of line when maximum value character length is overrun.
        dividerstr   = '*'*40+'\n' # String to use as an inter-item divider.
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
            output = 'odict()'
        else:
            output = '' # Initialize the output to nothing.
            keystrs = [] # Start with an empty list which we'll save key strings in.
            valstrs = [] # Start with an empty list which we'll save value strings in.
            vallinecounts = [] # Start with an empty list which we'll save line counts in.
            extraitems = nkeys - maxitems
            if extraitems <= 0:
                keylist = allkeys
            else:
                keylist = allkeys[:halfmax] + allkeys[-halfmax:]
            for thiskey in keylist: # Loop over the dictionary values
                thiskeystr = ut.flexstr(thiskey) # Grab a str representation of the current key.
                thisval = self.__getitem__(thiskey) # Grab the current value.

                try: # It's rare, but sometimes repr fails
                    # If it's another odict, make a call increasing the recursionlevel and passing the same parameters we received.
                    if isinstance(thisval, odict):
                        if recursionlevel <= maxrecursion:
                            thisvalstr = ut.flexstr(thisval.__repr__(maxlen=maxlen, showmultilines=showmultilines, divider=divider, dividerthresh=dividerthresh, numindents=numindents, recursionlevel=recursionlevel+1, sigfigs=sigfigs, numformat=numformat))
                        else:
                            thisvalstr = 'odict() [maximum recursion reached]'
                    elif ut.isnumber(thisval): # Flexibly print out numbers, since they're largely why we're here
                        if numformat is not None:
                            thisvalstr = numformat % thisval
                        elif sigfigs is not None:
                            thisvalstr = ut.sigfig(thisval, sigfigs=sigfigs)
                        else:
                            thisvalstr = ut.flexstr(thisval) # To avoid numpy's stupid 0.4999999999945
                    else: # Otherwise, do the normal repr() read.
                        thisvalstr = repr(thisval)
                except Exception as E:
                    thisvalstr = '%s read failed: %s' % (ut.objectid(thisval), str(E))

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
                    newoutput = ut.indent(prefix=theprefix, text=dividerstr, width=80)
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
                    rawoutput = '#%i: "%s":%s %s\n' % (ind, keystr, spacer, valstr)
                else:
                    rawoutput = '#%i: "%s":%s \n%s\n' % (ind, keystr, spacer, valstr)

                # Perform the indentation.
                newoutput = ut.indent(prefix=theprefix, text=rawoutput, width=80)

                # Strip ot any terminal newline.
                if newoutput[-1] == '\n':
                    newoutput = newoutput[:-1]

                # Add the new output to the full output.
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


    def _repr_pretty_(self, p, cycle):
        ''' Function to fix __repr__ in IPython'''
        print(self.__repr__())


    def disp(self, maxlen=None, showmultilines=True, divider=False, dividerthresh=10, numindents=0, sigfigs=5, numformat=None):
        '''
        Print out flexible representation, short by default.

        Example:
            import pylab as pl
            z = odict().make(keys=['a','b','c'], vals=(10*pl.rand(3)).tolist())
            z.disp(sigfigs=3)
            z.disp(numformat='%0.6f')
        '''
        print(self.__repr__(maxlen=maxlen, showmultilines=showmultilines, divider=divider, dividerthresh=dividerthresh, numindents=numindents, recursionlevel=0, sigfigs=sigfigs, numformat=None))
        return None


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
            return None
        else:
            return output


    def to_OD(self):
        ''' Export the odict to an OrderedDict '''
        return OD(self)


    def pop(self, key, *args, **kwargs):
        ''' Allows pop to support strings, integers, slices, lists, or arrays '''
        if isinstance(key, ut._stringtypes):
            return OD.pop(self, key, *args, **kwargs)
        elif isinstance(key, ut._numtype): # Convert automatically from float...dangerous?
            thiskey = self.keys()[int(key)]
            return OD.pop(self, thiskey, *args, **kwargs)
        elif type(key)==slice: # Handle a slice -- complicated
            try:
                startind = self.__slicekey(key.start, 'start')
                stopind = self.__slicekey(key.stop, 'stop')
                if stopind<startind:
                    print('Stop index must be >= start index (start=%i, stop=%i)' % (startind, stopind))
                    raise Exception
                slicevals = [self.pop(i, *args, **kwargs) for i in range(startind,stopind)] # WARNING, not tested
                try: return np.array(slicevals) # Try to convert to an array
                except: return slicevals
            except:
                print('Invalid odict slice... returning empty list...')
                return []
        elif self.__isODict_iterable(key): # Iterate over items
            listvals = [self.pop(item, *args, **kwargs) for item in key]
            try: return np.array(listvals)
            except: return listvals
        else: # Handle string but also everything else
            try:
                return OD.pop(self, key, *args, **kwargs)
            except: # WARNING, should be KeyError, but this can't print newlines!!!
                if len(self.keys()):
                    errormsg = 'odict key "%s" not found; available keys are:\n%s' % (ut.flexstr(key),
                        '\n'.join([ut.flexstr(k) for k in self.keys()]))
                else: errormsg = 'Key "%s" not found since odict is empty'% key
                raise Exception(errormsg)


    def remove(self, key, *args, **kwargs):
        ''' Remove an item by key and do not return it '''
        self.pop(key, *args, **kwargs)
        return None


    def clear(self):
        ''' Reset to an empty odict '''
        for key in self.keys():
            self.remove(key)
        return None


    def index(self, value):
        ''' Return the index of a given key '''
        return self.keys().index(value)


    def valind(self, value):
        ''' Return the index of a given value '''
        return self.values().index(value)


    @staticmethod
    def _matchkey(key, pattern, method):
        ''' Helper function for findkeys '''
        match = False
        if isinstance(key, tuple):
            for item in key:
                if odict._matchkey(item, pattern, method):
                    return True
        else: # For everything except a tuple, treat it as a string
            if not ut.isstring(key):
                try:
                    key = str(key) # Try to cast it to a string
                except Exception as E:
                    errormsg = 'Could not convert odict key of type %s to a string: %s' % (type(key), str(E))
                    raise Exception(E)
            if   method == 're':         match = bool(re.search(pattern, key))
            elif method == 'in':         match = pattern in key
            elif method == 'startswith': match = key.startswith(pattern)
            elif method == 'endswith':   match = key.endswith(pattern)
            else:
                errormsg = 'Method "%s" not found; must be "re", "in", "startswith", or "endswith"' % method
                raise Exception(errormsg)
        return match


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

        Example:
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
        if ut.isstring(keys) and pattern is None: # Assume first argument, transfer
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
            keyname = 'key'+ut.flexstr(len(self))  # Define the key just to be the current index
        else:
            keyname = key
        self.__setitem__(keyname, value)
        return None


    def insert(self, pos=None, key=None, value=None):
        '''
        Stupid, slow function to do insert -- WARNING, should be able to use approach more like rename...

        Usage:
            z = odict()
            z['foo'] = 1492
            z.insert(1604)
            z.insert(0, 'ganges', 1444)
            z.insert(2, 'midway', 1234)
        '''

        # Handle inputs
        realpos, realkey, realvalue = pos, key, value
        if key is None and value is None: # Assume it's called like odict.insert(666)
            realvalue = pos
            realkey = 'key'+ut.flexstr(len(self))
            realpos = 0
        elif value is None: # Assume it's called like odict.insert('devil', 666)
            realvalue = key
            realkey = pos
            realpos = 0
        if pos is None:
            realpos = 0
        if realpos>len(self):
            errormsg = 'Cannot insert %s at position %i since length of odict is %i ' % (key, pos, len(self))
            raise Exception(errormsg)

        # Create a temporary dictionary to hold all of the items after the insertion point
        tmpdict = odict()
        origkeys = self.keys()
        originds = range(len(origkeys))
        if not len(originds) or realpos==len(originds): # It's empty or in the final position, just append
            self.__setitem__(realkey, realvalue)
        else: # Main usage case, it's not empty
            try: insertind = originds.index(realpos) # Figure out which index we're inseting at
            except:
                errormsg = 'Could not insert item at position %i in odict with %i items' % (realpos, len(originds))
                raise Exception(errormsg)
            keystopop = origkeys[insertind:] # Pop these keys until we get far enough back
            for keytopop in keystopop:
                tmpdict.__setitem__(keytopop, self.pop(keytopop))
            self.__setitem__(realkey, realvalue) # Insert the new item at the right location
            for keytopop in keystopop: # Insert popped items back in
                self.__setitem__(keytopop, tmpdict.pop(keytopop))

        return None


    def copy(self, oldkey, newkey):
        ''' Make a copy of an item '''
        newval = ut.dcp(self.__getitem__(oldkey))
        self.__setitem__(newkey, newval)
        return None


    def rename(self, oldkey, newkey):
        ''' Change a key name -- WARNING, very inefficient! '''
        nkeys = len(self)
        if isinstance(oldkey, ut._numtype):
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
        return None


    def sort(self, sortby=None, reverse=False, copy=False, verbose=True):
        '''
        Create a sorted version of the odict. Sorts by order of sortby, if provided, otherwise alphabetical.
        If copy is True, then returns a copy (like sorted()).

        Note that you can also use this to do filtering.

        Note: slow, do not use for time-limited computations!!
        '''
        origkeys = self.keys()
        if sortby is None or sortby == 'keys':
            allkeys = sorted(origkeys)
        else:
            if sortby == 'values':
                origvals = self.values()
                sortby = sorted(range(len(origvals)), key=origvals.__getitem__) # Reset sortby based on https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
            if not ut.isiterable(sortby):
                raise Exception('Please provide a list to determine the sort order.')
            if all([isinstance(x, ut._stringtypes) for x in sortby]): # Going to sort by keys
                allkeys = sortby # Assume the user knows what s/he is doing
            elif all([isinstance(x,bool) for x in sortby]): # Using Boolean values to filter
                allkeys = []
                for i,x in enumerate(sortby):
                     if x: allkeys.append(origkeys[i])
            elif all([isinstance(x, ut._numtype) for x in sortby]): # Going to sort by numbers
                if not set(sortby)==set(range(len(self))):
                    warningmsg = 'Warning: list to sort by "%s" has different length than odict "%i"' % (sortby, len(self))
                    if verbose: print(warningmsg)
                allkeys = [origkeys[ind] for ind in sortby]
                print(allkeys)
            else:
                raise Exception('Cannot figure out how to sort by "%s"' % sortby)
        tmpdict = odict()
        if reverse:
            allkeys.reverse() # If requested, reverse order
        if copy:
            for key in allkeys: tmpdict[key] = self[key]
            return tmpdict
        else:
            for key in allkeys: tmpdict.__setitem__(key, self.pop(key))
            for key in allkeys: self.__setitem__(key, tmpdict.pop(key))
            return None


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


    def make(self, keys=None, vals=None, keys2=None, keys3=None):
        '''
        An alternate way of making or adding to an odict. Examples:
            a = odict().make(5) # Make an odict of length 5, populated with Nones and default key names
            b = odict().make('foo',34) # Make an odict with a single key 'foo' of value 34
            c = odict().make(['a','b']) # Make an odict with keys 'a' and 'b'
            d = odict().make(['a','b'],0) # Make an odict with keys 'a' and 'b', initialized to 0
            e = odict().make(keys=['a','b'], vals=[1,2]) # Make an odict with 'a':1 and 'b':2
            f = odict({'a':34, 'b':58}).make(['c','d'],[99,45]) # Add extra keys to an exising odict
            g = odict().make(keys=['a','b','c'], keys2=['A','B','C'], keys3=['x','y','z'], vals=0) # Make a triply nested odict
        '''
        # Handle keys
        keylist = []
        if keys is None and vals is None:
            return None # Nothing to do if nothing supplied
        if keys is None and vals is not None:
            keys = len(ut.promotetolist(vals)) # Values are supplied but keys aren't: use default keys
        if isinstance(keys, ut._numtype): # It's a single number: pre-generate
            keylist = ['%i'%i for i in range(keys)] # Generate keylist
        elif isinstance(keys, ut._stringtypes): # It's a single string
            keylist = [ut.flexstr(keys)]
        elif isinstance(keys, list): # It's a list: use directly
            keylist = keys
        else:
            errormsg = 'Could not understand keys "%s": must be number, string, or list' % keys
            raise Exception(errormsg)
        nkeys = len(keylist)

        # Handle values
        vals = ut.promotetolist(vals)
        nvals = len(vals)
        if nvals==0: # Special case: it's an empty list
            vallist = [ut.dcp(vals) for _ in range(nkeys)]
        elif nvals==1: # Only a single value: duplicate it
            vallist = [ut.dcp(vals[0]) for _ in range(nkeys)]
        elif nvals==nkeys: # Lengths match, can use directly
            vallist = vals
        else:
            errormsg = 'Must supply either a single value or a list of same length as the keys (%i keys, %i values supplied)' % (nkeys, nvals)
            raise Exception(errormsg)

        # Handle nested keys -- warning, would be better to not hard-code this, but does the brain in as it is!
        if keys2 is not None and keys3 is not None: # Doubly nested
            self.make(keys=keys, vals=odict().make(keys=keys2, vals=odict().make(keys=keys3, vals=vals)))
        elif keys2 is not None: # Singly nested
            self.make(keys=keys, vals=odict().make(keys=keys2, vals=vals))
        else: # Not nested -- normal case of making an odict
            for key,val in zip(keylist,vallist): # Update odict
                self.__setitem__(key, val)

        return self # A bit weird, but usually would use this return an odict


    def makefrom(self, source=None, keys=None, keynames=None, *args, **kwargs):
        '''
        Create an odict from entries in another dictionary. If keys is None, then
        use all keys from the current dictionary.

        Examples:
            a = 'cat'; b = 'dog'; o = odict().makefrom(source=locals(), keys=['a','b']) # Make use of fact that variables are stored in a dictionary
            d = {'a':'cat', 'b':'dog'}; o = odict().makefrom(d) # Same as odict(d)
            l = ['cat', 'monkey', 'dog']; o = odict().makefrom(source=l, keys=[0,2], keynames=['a','b'])
        '''

        # Make sure it's iterable
        if source is not None: # Don't do anything if there's nothing there
            if not(ut.isiterable(source)): # Make sure it's iterable
                source = ut.promotetolist(source)
            elif isinstance(source, ut._stringtypes):
                source = [source] # Special case -- strings are iterable, but we don't want to

            if len(source)==0:
                return self # Nothing to do here
            else:
                # Handle cases where keys or keynames are not supplied
                if keys is None:
                    if   isinstance(source, (list, tuple)):   keys = range(len(source))
                    elif isinstance(source, dict):            keys = list(source.keys())
                    else:                                     raise Exception('Unable to guess keys for object of type %s' % type(source))
                keys = ut.promotetolist(keys) # Make sure it's a list -- note, does not convert other iterables to a list!
                if keynames is None: keynames = keys # Use key names

                # Loop over supplied keys
                for key,keyname in zip(keys,keynames):
                    try:
                        self.__setitem__(str(keyname), source[key])
                    except Exception as E:
                        raise Exception('Key "%s" not found: %s' % (key, repr(E)))

        return self # As with make()


    def map(self, func=None):
        '''
        Apply a function to each element of the odict, returning
        a new odict with the same keys.

        Example:
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

        Example:
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

        Example:
            z = odict({'a':[1,2,3,4], 'b':[5,6,7,8]})
            z.toeach(2, [10,20])    # z is now odict({'a':[1,2,10,4], 'b':[5,6,20,8]})
            z.toeach(ind=3,val=666) #  z is now odict({'a':[1,2,10,666], 'b':[5,6,20,666]})
        '''
        nkeys = len(self.keys())
        if not(ut.isiterable(val)): # Assume it's meant to be populated in each
            val = [val]*nkeys # Duplicated
        if len(val)!=nkeys:
            errormsg = 'To map values onto each key, they must be the same length (%i vs. %i)' % (len(val), nkeys)
            raise Exception(errormsg)
        for k,key in self.enumkeys():
            self.__getitem__(key)[ind] = val[k]
        return None


    def enumkeys(self):
        ''' Shortcut for enumerate(odict.keys()) '''
        iterator = list(enumerate(self.keys()))
        return iterator


    def enumvals(self):
        ''' Shortcut for enumerate(odict.values()) '''
        iterator = list(enumerate(self.values()))
        return iterator


    def enumitems(self):
        ''' Returns tuple of 3 things: index, key, value '''
        iterator = [] # Would be better to not pre-allocate but what can you do...
        for ind,item in enumerate(self.items()):
            thistuple = (ind,)+item # Combine into one tuple
            iterator.append(thistuple)
        return iterator

    @staticmethod
    def promote(obj=None):
        '''
        Like promotetolist, but for odicts. Example:
            od = sc.odict.promote(['There','are',4,'keys'])

        Note, in most cases odict(obj) or odict().make(obj) can be used instead.
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

    # Python 3 compatibility
    def keys(self):
        """ Method to get a list of keys as in Python 2. """
        return list(OD.keys(self))

    def values(self):
        """ Method to get a list of values as in Python 2. """
        return list(OD.values(self))

    def items(self):
        """ Method to generate an item iterator as in Python 2. """
        return list(OD.items(self))

    def iteritems(self):
        """ Method to generate an item iterator as in Python 2. """
        return list(OD.items(self))


class objdict(odict):
    '''
    Exactly the same as an odict, but allows keys to be set/retrieved by object
    notiation.

    Example
    -------
    >>> import sciris as sc
    >>> od = sc.objdict({'height':1.65, 'mass':59})
    >>> od.bmi = od.mass/od.height**2
    >>> od.keys = 3 # This will return an exception since od.keys already exists
    '''


    def __getattribute__(self, attr):
        try: # First, try to get the attribute as an attribute
            output = odict.__getattribute__(self, attr)
            return output
        except Exception as E: # If that fails, try to get it as a dict item
            try:
                output = odict.__getitem__(self, attr)
                return output
            except: # If that fails, raise the original exception
                raise E

    def __setattr__(self, name, value):
        ''' Set key in dict, not attribute! '''

        try:
            odict.__getattribute__(self, name) # Try retrieving this as an attribute, expect AttributeError...
        except AttributeError:
            return odict.__setitem__(self, name, value) # If so, simply return

        # Otherwise, raise an exception
        errormsg = '"%s" exists as an attribute, so cannot be set as key; use setattribute() instead' % name
        raise Exception(errormsg)

        return None

    def setattribute(self, name, value):
        ''' Set attribute if truly desired '''
        return odict.__setattr__(self, name, value)
