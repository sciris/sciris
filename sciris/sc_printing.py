"""
Printing/notification functions.

Highlights:
    - :func:`sc.heading() <heading>`: print text as a 'large' heading
    - :func:`sc.colorize() <colorize>`: print text in a certain color
    - :func:`sc.pr() <pr>`: print full representation of an object, including methods and each attribute
    - :func:`sc.sigfig() <sigfig>`: truncate a number to a certain number of significant figures
    - :func:`sc.progressbar() <progressbar>`: show a (text-based) progress bar
    - :func:`sc.capture() <capture>`: capture text output (e.g., stdout) as a variable
"""

import io
import os
import sys
import types
import time
import tqdm
import pprint
import inspect
import warnings
import numpy as np
import collections as co
from textwrap import fill
from contextlib import redirect_stdout
import sciris as sc
import sciris.sc_utils as scu
from ._extras import ansicolors as ac

# Add Windows support for colors (do this at the module level so that colorama.init() only gets called once)
if sc.iswindows(): # pragma: no cover # NB: can't use startswith() because of 'cygwin'
    try:
        import colorama
        colorama.init()
        ansi_support = True
    except:
        ansi_support = False
else:
    ansi_support = True


#%% Object display functions

__all__ = ['createcollist', 'objectid', 'classatt', 'objatt', 'objmeth', 'objprop', 
           'objrepr', 'prepr', 'pr', 'prettyobj', 'quickobj']

_ncol = 3 # Default number of columns to display
_strlen = 22 # Default string length for a 3-column display
_dividerlen = 72 # Length of the divider between printed sections

def createcollist(items, title=None, strlen=_strlen, ncol=_ncol):
    """ Creates a string for a nice columnated list (e.g. to use in __repr__ method) """
    if len(items):
        nrow = int(np.ceil(float(len(items))/ncol))
        newkeys = []
        for x in range(nrow):
            newkeys += items[x::nrow]
    
        string = title + ':' if title else ''
        c = 0
        for x in newkeys:
            if c%ncol == 0: string += '\n  '
            if len(x) > strlen: x = x[:strlen-3] + '...'
            string += '%-*s  ' % (strlen,x)
            c += 1
        string += '\n'
    else:
        string = ''
    return string


def objectid(obj, showclasses=False):
    """
    Return the object ID as per the default Python ``__repr__`` method
    
    *New in version 3.1.0:* "showclasses" argument
    """
    c = obj.__class__
    output = f'<{c.__module__}.{c.__name__} at {hex(id(obj))}>\n'
    if showclasses:
        mro = c.mro()
        if len(mro) > 1: # Remove 'object' unless it's the only class
            mro = mro[:-1]
        output += f'{mro}\n'
    return output


def _get_obj_keys(obj, private=False, sort=True, use_dir=False):
    """ Helper method to get the keys of an object """
    
    # Get the list of keys
    if use_dir: # This forces the use of dir(), as opposed to checking for dict or slots
        keys = obj.__dir__() # This is the unsorted version of dir()
    else:
        if   hasattr(obj, '__dict__'):  keys = obj.__dict__.keys()
        elif hasattr(obj, '__slots__'): keys = obj.__slots__
        else:                           keys = [] # pragma: no cover
    
    if isinstance(private, str):
        private = [private]
        
    # Sort by private keys
    if not private:
        keys = [k for k in keys if not k.startswith('__')]
    elif isinstance(private, list):
        keys = [k for k in keys if (not k.startswith('__') or k in private)]
        
    # Optionally sort
    if sort:
        keys = sorted(keys)
    return list(keys) # Sometimes dict_keys


def _is_meth(obj, attr, die=False):
    """
    Helper function to check if an attribute is a method; do not distinguish between bound and unbound
    
    | *New in version 3.2.0:* use method and function types instead of callable()    
    """
    try:
        obj = getattr(obj, attr, None)
        return isinstance(obj, (types.MethodType, types.FunctionType)) # Equivalent to sc.isfunc()
    except Exception as E:
        if die:
            raise E
        else:
            return False


def _is_prop(obj, attr, die=False):
    """ Helper function to check if an attribute is a property """
    try:
        obj = getattr(type(obj), attr, None)
        return isinstance(obj, property)
    except Exception as E:
        if die:
            raise E
        else:
            return False


def objatt(obj, strlen=_strlen, ncol=_ncol, private=False, sort=True, _keys=None, return_keys=False):
    """ Return a sorted string of object attributes for the Python __repr__ method; see :func:`sc.prepr() <prepr>` for options """
    keys = _get_obj_keys(obj, private=private, sort=sort) if _keys is None else _keys
    if return_keys:
        return keys
    else:
        output = createcollist(keys, 'Attributes', strlen=strlen, ncol=ncol)
        return output


def classatt(obj, strlen=_strlen, ncol=_ncol, private=False, sort=True, _objkeys=None, _dirkeys=None, return_keys=False):
    """ Return a sorted string of class attributes for the Python __repr__ method; see :func:`sc.prepr() <prepr>` for options """
    objkeys = _get_obj_keys(obj, private=private, sort=sort, use_dir=False) if _objkeys is None else _objkeys
    dirkeys = _get_obj_keys(obj, private=private, sort=sort, use_dir=True)  if _dirkeys is None else _dirkeys
    keys = set(dirkeys) - set(objkeys) # Find attributes in dir() that are not in __dict__
    keys = list(filter(keys.__contains__, dirkeys)) # Maintain original ordering
    keys = [k for k in keys if not (_is_meth(obj, k) or _is_prop(obj, k))]
    if return_keys:
        return keys
    else:
        output = createcollist(keys, 'Class attributes', strlen=strlen, ncol=ncol)
        return output


def objmeth(obj, strlen=_strlen, ncol=_ncol, private=False, sort=True, _keys=None, return_keys=False):
    """ Return a sorted string of object methods for the Python __repr__ method; see :func:`sc.prepr() <prepr>` for options """
    try: # In very rare cases this fails, so put it in a try-except loop
        _keys = _get_obj_keys(obj, private=private, sort=sort, use_dir=True) if _keys is None else _keys
        keys = sorted([meth + '()' for meth in _keys if _is_meth(obj, meth)])
        if return_keys:
            return _keys
    except Exception as E: # pragma: no cover
        keys = [f'Methods N/A ({E})']
    output = createcollist(keys, 'Methods', strlen=strlen, ncol=ncol)
    return output


def objprop(obj, strlen=_strlen, ncol=_ncol, private=False, sort=True, _keys=None, return_keys=False):
    """ Return a sorted string of object properties for the Python __repr__ method; see :func:`sc.prepr() <prepr>` for options """
    try: # In very rare cases this fails, so put it in a try-except loop
        _keys = _get_obj_keys(obj, private=private, sort=sort, use_dir=True) if _keys is None else _keys
        keys = [prop for prop in _keys if _is_prop(obj, prop)]
        if return_keys:
            return keys
    except Exception as E: # pragma: no cover
        keys = [f'Properties N/A ({E})']
    output = createcollist(keys, 'Properties', strlen=strlen, ncol=ncol)
    return output


def objrepr(obj, showid=True, showmeth=True, showprop=True, showatt=True, showclassatt=True, 
            private=False, sort=True, dividerchar='—', dividerlen=_dividerlen, strlen=_strlen, ncol=_ncol, 
            _objkeys=None, _dirkeys=None):
    """
    Print out a detailed representation of an object: methods, properties, attributes, etc.
    
    Similar to ``sc.prepr(obj, vals=False)``.
    
    See :func:`sc.prepr() <prepr>` for an explanation of arguments.
    """
    
    # Call the object twice to get the keys
    objkeys = _get_obj_keys(obj, private=private, sort=sort, use_dir=False) if _objkeys is None else _objkeys
    dirkeys = _get_obj_keys(obj, private=private, sort=sort, use_dir=True)  if _dirkeys is None else _dirkeys
    
    divider = dividerchar*dividerlen + '\n'
    output = ''
    
    def assemble(show, string):
        """ Helper function to construct the string """
        return string + divider if (show and string) else ''
    
    # Assemble the output string
    output += assemble(showid,       objectid(obj, showclasses=True))
    output += assemble(showmeth,     objmeth(obj,  strlen=strlen, ncol=ncol, _keys=dirkeys))
    output += assemble(showprop,     objprop(obj,  strlen=strlen, ncol=ncol, _keys=dirkeys))
    output += assemble(showatt,      objatt(obj,   strlen=strlen, ncol=ncol, _keys=objkeys))
    output += assemble(showclassatt, classatt(obj, strlen=strlen, ncol=ncol, _objkeys=objkeys, _dirkeys=dirkeys))
    
    return output


def prepr(obj, vals=True, maxlen=None, maxitems=None, skip=None, dividerchar='—', dividerlen=_dividerlen, use_repr=True, 
          private=False, sort=True, strlen=_strlen, ncol=_ncol, maxtime=3, maxrecurse=5, die=False, debug=False):
    """
    Pretty-print a detailed representation of an object.
    
    This function returns a pretty (and pretty detailed) representation of an object --
    all attributes (except any that are skipped), plus methods and ID.
    
    This function is usually used via the interactive :func:`sc.pr() <pr>` (which prints), 
    rather than this function (which returns a string).

    Args:
        obj (anything): the object to be represented
        vals (bool): whether to show attribute values (else, just list attributes; similar to :func:`sc.objrepr() <objrepr>`)
        maxlen (int): maximum number of characters to show for each attribute
        maxitems (int): maximum number of attribute to show in the object
        skip (list): any attributes to skip
        dividerchar (str): divider for methods, attributes, etc.
        dividerlen (int): number of divider characters
        use_repr (bool): whether to use repr() or str() to parse the object
        private (bool): whether to include private methods/attributes (those starting with "__")
        maxtime (float): maximum amount of time (in seconds) to spend on trying to print the object
        maxrecurse (int): maximum number of levels to descend in the object (set to 0 to turn off the check)
        die (bool): whether to raise an exception if an error is encountered
        debug (bool): print out detail during string construction
    
    | *New in version 3.0.0:* "debug" argument
    | *New in version 3.1.4:* more robust handling of invalid object properties
    | *New in version 3.1.5:* "vals" argument to turn off printing attribute values
    | *New in version 3.1.6:* "maxrecurse" argument, and checking for recursion
    
    **Examples**::

        # Default options
        df = sc.dataframe(a=[1,2,3], b=[4,5,6])
        print(df) # See just the data
        sc.pr(df) # See all the methods too
        sc.pr(df, vals=False) # Only see methods, not the values
        
        # Demonstrate options
        obj = sc.prettyobj({k:k for k in [l + str(n) for n in range(10) for l in 'abcde']}) # Big object
        sc.pr(obj, maxitems=20, sort=False, dividerchar='•', dividerlen=43, private=True)
    """
    # Check recursion before we proceed -- adds about 2-5 ms per call, but prevents infinite recursion
    if maxrecurse:
        frames = inspect.stack()[:maxrecurse*10] # Only get the most recent frames
        matches = sum([frame.function in ('prepr', '__repr__', '__str__') for frame in frames])
        if matches > maxrecurse:
            string = f'[sc.prepr(): not shown, recursion exceeded ({matches}>{maxrecurse})]' # This is unlikely to ever be seen since it's deep within the object
            return string

    # Decide how to handle representation function -- repr is dangerous since can lead to recursion
    repr_fn = repr if use_repr else str
    T = time.time() # Start the timer
    time_exceeded = False

    # Handle input arguments
    divider = dividerchar*dividerlen + '\n'
    if maxlen   is None: maxlen   = 80
    if maxitems is None: maxitems = 100
    if skip     is None: skip = []
    else:                skip = sc.tolist(skip)

    # Initialize things to print out
    labels = []
    values = []
    
    # Call the object twice to get the keys
    objkeys = _get_obj_keys(obj, private=private, sort=sort, use_dir=False)
    dirkeys = _get_obj_keys(obj, private=private, sort=sort, use_dir=True)

    # Wrap entire process for getting attribute strings in a try-except in case it fails
    E1, E2, E3 = None, None, None
    kw = dict(private=private, sort=sort, dividerchar=dividerchar, dividerlen=dividerlen, strlen=strlen, ncol=ncol)
    try:
        if not (hasattr(obj, '__dict__') or hasattr(obj, '__slots__')): # pragma: no cover
            # It's a plain object
            labels = [f'{type(obj)}']
            values = [repr_fn(obj)]
        else:
            labels = objkeys
            labels += classatt(obj, private=private, sort=sort, return_keys=True, _objkeys=objkeys, _dirkeys=dirkeys)
            if skip is not None:
                diff = set(labels) - set(skip)
                labels = list(filter(diff.__contains__, labels))
            if debug: # pragma: no cover
                print(f'Working on {len(labels)} entries...')

            if not len(labels):
                extraitems = 0
            else: # Usual case
                extraitems = max(0, len(labels) - maxitems)
                if extraitems > 0:
                    labels = labels[:maxitems]
                
                # Get the values of the attributes
                values = []
                if vals:
                    for a,attr in enumerate(labels):
                        if debug: # pragma: no cover
                            tdiff = time.time() - T
                            print(f'  Working on attribute {a}: {attr}... ({tdiff:0.3f})')
                        if (time.time() - T) < maxtime:
                            try: # Be especially robust in getting individual attributes
                                value = getattr(obj, attr)
                            except Exception as E:
                                value = 'N/A' # pragma: no cover
                                if die:
                                    raise E
                            try: # Separately, be robust about getting their attributes
                                value = repr_fn(value)
                            except Exception as E:
                                value = object.__repr__(value) # pragma: no cover
                                if die:
                                    raise E          
                            values.append(value)
                        else:
                            labels = labels[:a]
                            labels.append('etc. (time exceeded)')
                            values.append(f'{len(labels)-a} entries not shown')
                            time_exceeded = True
                            break
                
            if extraitems > 0:
                labels.append('etc. (too many items)')
                values.append(f'{extraitems} entries not shown')

        # Decide how to print them
        maxkeylen = 0
        if len(labels):
            maxkeylen = max([len(label) for label in labels]) # Find the maximum length of the attribute keys
        if maxkeylen<maxlen:
            maxlen = maxlen - maxkeylen # Shorten the amount of data shown if the keys are long
        formatstr = f'%{maxkeylen}s' # Assemble the format string for the keys, e.g. '%21s'
        
        # Actually get the methods
        output  = objrepr(obj, showatt=False, showclassatt=False, _objkeys=objkeys, _dirkeys=dirkeys, **kw) 
        if not len(labels):
            output += 'No attributes\n'
        else:
            if vals:
                for label,value in zip(labels,values): # Loop over each attribute
                    if len(value)>maxlen: value = value[:maxlen] + ' [...]' # Shorten it
                    prefix = formatstr%label + ': ' # The format key
                    output += indent(prefix, value)
            else:
                output += createcollist(labels, 'Attributes', strlen=strlen, ncol=ncol)
        output += divider
        if time_exceeded:
            timestr = f'\nNote: the object did not finish printing within maxtime={maxtime} s.\n'
            timestr += 'To see the full object, call prepr() with increased maxtime.'
            output += timestr

    # If that failed, try progressively simpler approaches
    except Exception as E: # pragma: no cover
        E1 = E
        if die:
            errormsg = 'Failed to create pretty representation of object'
            raise RuntimeError(errormsg) from E
        else:
            try: # Next try the objrepr, which is the same except doesn't print attribute values
                output = objrepr(obj, **kw)
                output += f'\nWarning: showing simplified output since full repr failed {str(E)}'
            except Exception as E: # If that fails, try the most basic object representation
                E2 = E
                output = object.__repr__(obj)
    
    if any([E is not None for E in [E1, E2, E3]]):
        warnmsg = f'Exception(s) encountered displaying object {objectid(obj)}:\n'
        if E1 is not None: warnmsg += f'{E1}\n'
        if E2 is not None: warnmsg += f'{E2}\n'
        if E3 is not None: warnmsg += f'{E3}\n'
        warnings.warn(warnmsg, category=RuntimeWarning, stacklevel=2)

    return output


def pr(obj, *args, **kwargs):
    """
    Pretty-print a detailed representation of an object ("pr" is short for "print repr").
    
    See :func:`sc.prepr() <prepr>` for arguments and examples.

    Note: :func:`sc.prepr() <prepr>` creates a string, while ``sc.pr()`` prints
    the output, i.e. ``sc.pr(obj)`` is an alias to ``print(sc.prepr(obj))``.
    """
    print(prepr(obj, *args, **kwargs))
    return


class prettyobj:
    """
    Use pretty repr for objects, instead of just showing the type and memory pointer
    (the Python default for objects). Can also be used as the base class for custom
    classes.
    
    See :class:`sc.quickobj() <quickobj>` for a similar class that does not print
    attribute values (better for large objects that take a while to display).
    
    Args:
        args (dict): dictionaries which are used to assign attributes
        kwargs (any): can also be used to assign attributes

    **Example 1**::

        myobj = sc.prettyobj(a=3)
        print(myobj)
        
        # <sciris.sc_printing.prettyobj at 0x7fbba4a97f40>
        # ————————————————————————————————————————————————————————————
        # Methods:
        #   Methods N/A         
        # ————————————————————————————————————————————————————————————
        # a: 3
        # ————————————————————————————————————————————————————————————

    **Example 2**::

        myobj = sc.prettyobj(a=3)
        myobj.b = {'a':6}
        print(myobj)
        
        # <sciris.sc_printing.prettyobj at 0x7ffa1e243910>
        # ————————————————————————————————————————————————————————————
        # Methods:
        #   Methods N/A         
        # ————————————————————————————————————————————————————————————
        # a: 3
        # b: {'a': 6}
        # ————————————————————————————————————————————————————————————


    **Example 3**::
        
        class MyObj(sc.prettyobj):
       
            def __init__(self, a, b):
                self.a = a
                self.b = b
       
            def mult(self):
                return self.a * self.b
       
        myobj = MyObj(a=4, b=6)
        print(myobj)
        
        # <__main__.MyObj at 0x7fd9acd96c10>
        # ————————————————————————————————————————————————————————————
        # Methods:
        #   mult()
        # ————————————————————————————————————————————————————————————
        # a: 4
        # b: 6
        # ————————————————————————————————————————————————————————————

    | *New in version 2.0.0:* allow positional arguments
    | *New in version 3.1.4:* moved from sc_utils to sc_printing
    | *New in version 3.1.6:* linked back to sc_utils to prevent unpickling errors
    """
    def __init__(self, *args, **kwargs):
        """ Simple initialization """
        kwargs = sc.mergedicts(*args, kwargs)
        for k,v in kwargs.items():
            self.__dict__[k] = v
        return

    def __repr__(self):
        """ The point of this class: use a more detailed repr by default """
        return prepr(self)

# Handle deprecation so loading old pickles still works
scu.prettyobj = prettyobj


class quickobj(prettyobj):
    """
    Like :class:`sc.prettyobj() <prettyobj>`, but do not print attribute values.
    
    This class is better for large objects that take a while to display. It is
    somewhat similar to calling dir() on an object.
    
    This class also defines a ``disp()`` method, which calls :func:`sc.pr() <pr>` on the object.
    
    **Example**::
        
        import numpy as np
        myobj = sc.quickobj(big1=np.random.rand(100,100), big2=sc.dataframe(a=np.arange(1000)))
        print(myobj)
    
    | *New in version 3.1.5.*
    """
    def __repr__(self):
        """ Use a more detailed repr than default, but less detailed that prettyobj """
        return prepr(self, vals=False)
    
    def disp(self, output=False, *args, **kwargs):
        """ Return full display of the object """
        string = prepr(self, *args, **kwargs)
        if output:
            return string
        else:
            print(string)
            return 


#%% Spacing functions

__all__ += ['blank', 'indent']


def blank(n=3):
    """ Tiny function to print n blank lines, 3 by default """
    print('\n'*n)


def indent(prefix=None, text=None, suffix='\n', n=0, pretty=False, width=70, **kwargs):
    """
    Small wrapper to make textwrap more user friendly.

    Args:
        prefix (str): text to begin with (optional)
        text (str): text to wrap
        suffix (str): what to put on the end (by default, a newline)
        n (int): if prefix is not specified, the size of the indent
        pretty (bool): whether to use pprint to format the text
        width (int): maximum width before wrapping (if None, don't wrap)
        kwargs (dict): passed to :func:`textwrap.fill()`

    **Examples**::

        prefix = 'and then they said: '
        text = 'blah '*100
        print(sc.indent(prefix, text))

        print('my fave is: ' + sc.indent(text=rand(100), n=12))

    *New in version 1.3.1:* more flexibility in arguments
    """
    # If "prefix" is given but text isn't, swap them
    if text is None and prefix is not None: # pragma: no cover
        text, prefix = prefix, text

    # Handle prefix and width
    if prefix is None: prefix = ' '*n
    if width  is None: width = 999_999

    # Get text in the right format -- i.e. a string
    if pretty: text = pprint.pformat(text)
    else:      text = sc.flexstr(text)

    # If there is no newline in the text, process the output normally.
    if text.find('\n') == -1:
        output = fill(text, initial_indent=prefix, subsequent_indent=' '*len(prefix), width=width, **kwargs)+suffix
    # Otherwise, handle each line separately and splice together the output.
    else:
        textlines = text.split('\n')
        output = ''
        for i, textline in enumerate(textlines):
            if i == 0:
                theprefix = prefix
            else:
                theprefix = ' '*len(prefix)
            output += fill(textline, initial_indent=theprefix, subsequent_indent=' '*len(prefix), width=width, **kwargs)+suffix

    if n: output = output[n:] # Need to remove the fake prefix
    return output



#%% Data representation functions

__all__ += ['sigfig', 'sigfigs', 'sigfiground', 'arraymean', 'arraymedian', 'printmean', 'printmedian', 
            'humanize_bytes', 'printarr', 'printdata', 'printvars']


def sigfig(x, sigfigs=4, SI=False, sep=False, keepints=False):
    """
    Return a string representation of variable x with sigfigs number of significant figures

    Note: :func:`sc.sigfig() <sigfig>` and :func:`sc.sigfigs() <sigfigs>` are aliases.

    Args:
        x (int/float/list/arr): the number(s) to round
        sigfigs (int): number of significant figures to round to
        SI (bool): whether to use SI notation (only for numbers >1)
        sep (bool/str): if provided, use as thousands separator
        keepints (bool): never round ints

    **Examples**::

        x = 3432.3842
        sc.sigfig(x, SI=True) # Returns '3.432k'
        sc.sigfig(x, sep=True) # Returns '3,432'
        
        vals = np.random.rand(5)
        sc.sigfig(vals, sigfigs=3)
    
    *New in version 3.0.0:* changed default number of significant figures from 5 to 4; return list rather than tuple; changed SI suffixes to uppercase
    """
    output = []

    islist = sc.isiterable(x)
    istuple = isinstance(x, tuple)
    xlist = x if islist else sc.tolist(x)
    for x in xlist:
        suffix = ''
        formats = [(1e18,'e18'), (1e15,'e15'), (1e12,'T'), (1e9,'B'), (1e6,'M'), (1e3,'K')]
        if SI:
            for val,suff in formats:
                if abs(x) >= val:
                    x = x/val
                    suffix = suff
                    break # Find at most one match

        try:
            if x == 0:
                output.append('0')
            elif sigfigs is None:
                output.append(sc.flexstr(x)+suffix)
            elif x > (10**sigfigs) and not SI and keepints: # e.g. x = 23432.23, sigfigs=3, output is 23432
                roundnumber = int(round(x))
                if sep: string = format(roundnumber, ',')
                else:   string = f'{x:0.0f}'
                output.append(string)
            else:
                magnitude = np.floor(np.log10(abs(x)))
                factor = 10**(sigfigs-magnitude-1)
                x = round(x*factor)/float(factor)
                digits = int(abs(magnitude) + max(0, sigfigs - max(0,magnitude) - 1) + 1 + (x<0) + (abs(x)<1)) # one because, one for decimal, one for minus
                decimals = int(max(0,-magnitude+sigfigs-1))
                strformat = '%' + f'{digits}.{decimals}' + 'f'
                string = strformat % x
                if sep: # To insert separators in the right place, have to convert back to a number
                    if decimals>0:  roundnumber = float(string)
                    else:           roundnumber = int(string)
                    string = format(roundnumber, ',') # Allow comma separator
                string += suffix
                output.append(string)
        except: # pragma: no cover
            output.append(sc.flexstr(x))
    if islist:
        if istuple:
            output = tuple(output)
        return output
    else:
        return output[0]

# Alias to avoid confusion
sigfigs = sigfig


def sigfiground(x, sigfigs=4):
    """
    Round number(s) to the specified number of significant figures.

    Args:
        x (int/float/list/arr): the number(s) to round
        sigfigs (int): number of significant figures to round to

    **Examples**::
        
        sc.sigfiground(3.28343) # Returns 3.283
        sc.sigfiground(834_875, 5)  # Returns 834880
        sc.sigfiground([3.28343, 834_874, 0, -83_742], 2) # Returns [3.3, 830000, 0, -84000]
    
    | *New in version 3.2.0.*
    """
    
    def round_arr(arr, sigfigs):
        """ Function for rounding a single number """
        out = arr.copy() # Preallocate output
        inds = np.nonzero(arr) # Skip zero indices
        nzarr = arr[inds] # Pull out those values
        magnitude = np.floor(np.log10(np.abs(nzarr))) # Determine the order of magnitude of the number
        exponent = sigfigs - magnitude - 1 # Convert to an exponent
        factor = 10**exponent # Calculate the factor to scale the number
        rndarr = np.round(nzarr*factor)/factor # Round the scaled number and scale it back
        out[inds] = rndarr # Put the rounded numbers back in the array
        if np.all(exponent <= 0): # They're all integers
            out = out.astype(np.int64)
        return out
    
    # Process inputs
    arr = sc.toarray(x)
    out = round_arr(arr, sigfigs) # Do the rounding
    if sc.isnumber(x): out = out[0]
    elif isinstance(x, list):
        out = out.tolist()
        out = [int(x) if x.is_integer() else x for x in out] # Allow mix of ints and floats
    return out
        

def arraymean(data, stds=2, axis=None, mean_sf=None, err_sf=None, tostring=True, doprint=False, **kwargs):
    """
    Quickly calculate the mean and standard deviation of an array.
    
    By default, will calculate the correct number of significant figures based on
    the deviation. The default is to multiply the standard deviation by 2, as an
    approximation of the 95% confidence level (z=1.96).
    
    Args:
        data (array): the data to summarize
        stds (int): the number of multiples of the standard deviation to show (default 2)
        axis (int): the axis of the data to operate on (default None)
        mean_sf (int): if provided, use this number of significant figures for the mean rather than the auto-calculated
        err_sf (int): ditto, but for the error (standard deviation)
        doprint (bool): whether to print (else, return the string)
        kwargs (dict): passed to :func:`sc.sigfig() <sigfig>`
    
    **Example**::
        
        data = [1210, 1072, 1722, 1229, 1902]
        sc.arraymean(data) # Returns 1430 ± 320
    
    | *New in version 3.0.0.*
    | *New in version 3.2.0:* "axis" argument, "tostring" argument (allow numerical output)
    """
    vsf = mean_sf # vsf = "value significant figures"
    esf = err_sf if err_sf is not None else 2
    data = sc.toarray(data)
    val = data.mean(axis=axis)
    err = data.std(axis=axis)*stds
    
    relsize = np.floor(np.log10(abs(val))) - np.floor(np.log10(abs(err)))
    if vsf is None:
        vsf = esf + relsize
    elif vsf is not None and err_sf is None:
        esf = min(vsf, vsf - relsize)
    
    if tostring:
        valstr = sigfig(val, vsf, **kwargs)
        errstr = sigfig(err, esf, **kwargs)
        string = f'{valstr} ± {errstr}'
        
        if doprint:
            print(string)
        else:
            return string
    else:
        valnum = sigfiground(val, vsf)
        errnum = sigfiground(err, esf)
        return valnum, errnum


def arraymedian(data, ci=95, sf=3, doprint=False, **kwargs):
    """
    Quickly calculate the median and confidence interval of an array.
    
    The confidence interval defaults to 95%. If an integer is supplied, this is
    treated as a percentile (e.g. 95=95% CI). If a float is supplied, it's treated
    as a quantile (e.g. 0.95=95% CI). If a pair of ints or floats is provided, these are
    treated as upper and lower percentiles/quantiles. If 'iqr' is provided, then
    print the interquartile range (equivalent to 50% CI). If 'range' is provided
    then print the full range (equivalent to 100% CI).
    
    Args:
        data (array): the data to summarize
        ci (int/float/list/str): the confidence interval to use to use (see above for details)
        sf (int): number of significant figures to use
        doprint (bool): whether to print (else, return the string)
        kwargs (dict): passed to :func:`sc.sigfig() <sigfig>`
    
    **Examples**::
        
        data = [1210, 1072, 1722, 1229, 1902]
        sc.printmedian(data, 80) # Returns '1230 (80.0% CI: 1130, 1830)'
    
    *New in version 3.0.0.*
    """
    # Handle quantiles
    if ci is None: # pragma: no cover
        ci = 95
    elif str(ci).lower() == 'iqr':
        ci = 50
    elif str(ci).lower() in ['range', 'minmax']:
        ci = 100
        
    if sc.isnumber(ci):
        if isinstance(ci, int):
            x = ci/100/2
        elif isinstance(ci, float):
            x = ci/2
        quantiles = [0.5-x, 0.5+x]
        if x == 0.25:
            cistr = 'IQR'
        elif x == 0.5:
            cistr = 'min, max'
        else:
            cistr = f'{x*100*2:n}% CI'
    elif sc.isiterable(ci):
        if len(ci) != 2: # pragma: no cover
            errormsg = f'If providing a list of quantiles, must provide 2, not {len(ci)}'
            raise ValueError(errormsg)
        quantiles = ci
        for i,q in enumerate(quantiles):
            if isinstance(q, int):
                quantiles[i] = q/100
        cistr = f'{quantiles[0]*100:n}%, {quantiles[1]*100:n}%'
    else: # pragma: no cover
        errormsg = f'Could not understand confidence interval "{ci}"'
        raise ValueError(errormsg)
    
    # Do calculations
    data    = sc.toarray(data)
    median  = np.quantile(data, 0.5)
    bounds  = np.quantile(data, quantiles)
    relsize = np.floor(np.log10(abs(median))) - np.floor(np.log10(np.abs(bounds)))
        
    # Assemble string
    valstr  = sigfig(median, sf, **kwargs)
    lowstr  = sigfig(bounds[0], sf-relsize[0], **kwargs)
    highstr = sigfig(bounds[1], sf-relsize[1], **kwargs)
    string = f'{valstr} ({cistr}: {lowstr}, {highstr})'
    
    if doprint:
        print(string)
    else:
        return string


def printmean(*args, doprint=True, **kwargs):
    """ Alias to :func:`sc.arraymean() <arraymean>` with doprint=True """
    return arraymean(*args, doprint=doprint, **kwargs)


def printmedian(*args, doprint=True, **kwargs):
    """ Alias to :func:`sc.arraymedian() <arraymedian>` with doprint=True """
    return arraymedian(*args, doprint=doprint, **kwargs)


def humanize_bytes(bytesize, decimals=3):
    """
    Convert a number of bytes into a human-readable total.
    
    Args:
        bytesize (int): the number of bytes
        decimals (int): the number of decimal places to show
    
    **Example**::
        
        sc.humansize(2.3423887e6, decimals=2) # Returns '2.34 MB'
        
    See the humansize library for more flexibility.
    
    *New in version 3.0.0.*
    """
    # Convert to string
    factor = 1
    label = 'B'
    labels = ['KB','MB','GB']
    for i,f in enumerate([3,6,9]):
        if bytesize >= 10**f:
            factor = 10**f
            label = labels[i]
    if factor == 1:
        decimals = 0 # Do not show decimals for bytes
    humansize = float(bytesize/float(factor))
    string = f'{humansize:0.{decimals}f} {label}'
    return string



def printarr(arr, fmt=None, colsep='  ', vsep='—', decimals=2, doprint=True, dtype=None):
    """
    Print a numpy array nicely.
    
    Args:
        arr (array): the array to print
        fmt (str): the formatting string to use
        colsep (str): the separator between columns of values
        vsep (str): the vertical separator between 2D slices
        decimals (int): number of decimal places to print
        doprint (bool): whether to print (else, return the string)

    **Examples**::

        numeric = np.random.randn(3,7,4)**10
        mixed = np.array([['cat', 'nudibranch'], [23, 2423482]], dtype=object)
        sc.printarr(numeric)
        sc.printarr(mixed)

    *New in version 2.0.3:* "fmt", "colsep", "vsep", "decimals", and "dtype" arguments
    *New in version 3.0.0:* "doprint" argument
    """
    from . import sc_math as scm # To avoid circular import
    
    string = ''
    arr = sc.toarray(arr, dtype=dtype)
    if fmt is None:
        if arr.dtype == object: # pragma: no cover
            maxdigits = max([len(str(v)) for v in arr.flatten()])
            fmt = f'%{maxdigits}s'
        else:
            maxdigits = sc.numdigits(arr.max())
            if arr.dtype == float:
                fmt = f'%{maxdigits+decimals+1}.{decimals}f'
            else: # pragma: no cover
                fmt = f'%{maxdigits}.0f'
    if np.ndim(arr)==1:
        for i in range(len(arr)):
            string += fmt % arr[i] + colsep
        string += '\n'
    elif np.ndim(arr)==2:
        for i in range(len(arr)):
            string += printarr(arr[i], fmt, colsep, doprint=False) + '\n'
    elif np.ndim(arr)==3:
        for i in range(len(arr)):
            ncols  = len(arr[i][0])
            vlen   = len(fmt % arr.flatten()[0])
            seplen = len(colsep)
            n = ncols*(vlen + seplen) - seplen
            string += vsep*n + '\n'
            for j in range(len(arr[i])):
                string += printarr(arr[i][j], fmt, colsep, doprint=False)
    else: # pragma: no cover
        print('Dimensions higher than 3 are not supported')
        string = str(arr) # Give up
    
    if doprint:
        print(string)
    else:
        return string



def printdata(data, name='Variable', depth=1, maxlen=40, indent='', level=0, showcontents=False): # pragma: no cover
    """
    Nicely print a complicated data structure, a la Matlab.

    Note: this function is deprecated.

    Args:
      data: the data to display
      name: the name of the variable (automatically read except for first one)
      depth: how many levels of recursion to follow
      maxlen: number of characters of data to display (if 0, don't show data)
      indent: where to start the indent (used internally)

    Version: 2015aug21
    """
    datatype = type(data)
    def printentry(data):
        if   datatype==dict:              string = (f'dict with {len(data.keys())} keys')
        elif datatype==list:              string = (f'list of length {len(data)}')
        elif datatype==tuple:             string = (f'tuple of length {len(data)}')
        elif datatype==np.ndarray:        string = (f'array of shape {np.shape(data)}')
        elif datatype.__name__=='module': string = (f'module with {len(dir(data))} components')
        elif datatype.__name__=='class':  string = (f'class with {len(dir(data))} components')
        else: string = datatype.__name__
        if showcontents and maxlen>0:
            datastring = ' | '+sc.flexstr(data)
            if len(datastring)>maxlen: datastring = datastring[:maxlen] + ' <etc> ' + datastring[-maxlen:]
        else: datastring=''
        return string+datastring

    string = printentry(data).replace('\n',' ') # Remove newlines
    print(level*'..' + indent + name + ' | ' + string)

    if depth>0:
        level += 1
        if type(data)==dict:
            keys = data.keys()
            maxkeylen = max([len(key) for key in keys])
            for key in keys:
                thisindent = ' '*(maxkeylen-len(key))
                printdata(data[key], name=key, depth=depth-1, indent=indent+thisindent, level=level)
        elif type(data) in [list, tuple]:
            for i in range(len(data)):
                printdata(data[i], name='[%i]'%i, depth=depth-1, indent=indent, level=level)
        elif type(data).__name__ in ['module', 'class']:
            keys = dir(data)
            maxkeylen = max([len(key) for key in keys])
            for key in keys:
                if key[0]!='_': # Skip these
                    thisindent = ' '*(maxkeylen-len(key))
                    printdata(getattr(data,key), name=key, depth=depth-1, indent=indent+thisindent, level=level)
        print('\n')
    return


def printvars(localvars=None, varlist=None, label=None, divider=True, spaces=1, color=None):
    """
    Print out a list of variables. Note that the first argument must be locals().

    Args:
        localvars: function must be called with locals() as first argument
        varlist: the list of variables to print out
        label: optional label to print out, so you know where the variables came from
        divider: whether or not to offset the printout with a spacer (i.e. ------)
        spaces: how many spaces to use between variables
        color: optionally label the variable names in color so they're easier to see

    **Example**::

    >>> a = range(5)
    >>> b = 'example'
    >>> sc.printvars(locals(), ['a','b'], color='green')

    Another useful usage case is to print out the kwargs for a function:

    >>> sc.printvars(locals(), kwargs.keys())

    Version: 2017oct28
    """

    varlist = sc.tolist(varlist) # Make sure it's actually a list
    dividerstr = '-'*40

    if label:  print(f'Variables for {label}:')
    if divider: print(dividerstr)
    for varnum,varname in enumerate(varlist):
        controlstr = f'{varnum}. "{varname}": ' # Basis for the control string -- variable number and name
        if color: controlstr = colorize(color, output=True) + controlstr + colorize('reset', output=True) # Optionally add color
        if spaces>1: controlstr += '\n' # Add a newline if the variables are going to be on different lines
        try:    controlstr += f'{localvars[varname]}' # The variable to be printed
        except: controlstr += 'Warning, could not be printed' # In case something goes wrong # pragma: no cover
        controlstr += '\n' * spaces # The number of spaces to add between variables
        print(controlstr), # Print it out
    if divider: print(dividerstr) # If necessary, print the divider again
    return



#%% Color functions

__all__ += ['colorize', 'heading', 'printred', 'printyellow', 'printgreen',
            'printcyan', 'printblue', 'printmagenta']


def colorize(color=None, string=None, doprint=None, output=False, enable=True, showhelp=False, fg=None, bg=None, style=None):
    """
    Colorize output text.

    Args:
        color (str): the color you want (use 'bg' with background colors, e.g. 'bgblue'); alternatively, use fg, bg, and style
        string (str): the text to be colored
        doprint (bool): whether to print the string (default true unless output)
        output (bool): whether to return the modified version of the string (default false)
        enable (bool): switch to allow :func:`sc.colorize() <colorize>` to be easily turned off without converting to a :func:`print()` statement
        showhelp (bool): show help rather than changing colors
        fg (str): foreground colour
        bg (str): background colour
        style (str): font style (eg, italic, underline, bold)

    **Examples**::

        sc.colorize('green', 'hi') # Simple example
        sc.colorize(['yellow', 'bgblack']); print('Hello world'); print('Goodbye world'); colorize() # Colorize all output in between
        bluearray = sc.colorize(color='blue', string=str(range(5)), output=True); print("c'est bleu: " + bluearray)
        sc.colorize('magenta') # Now type in magenta for a while
        sc.colorize() # Stop typing in magenta
        sc.colorize('cat in the hat', fg='#ffa044', bg='blue', style='italic+underline') # Alternate usage example

    To get available colors, type :func:`sc.colorize(showhelp=True) <colorize>`.

    | *New in version 1.3.1:* "doprint" argument; ansicolors shortcut
    """

    # Handle short-circuit case
    if not enable: # pragma: no cover
        if output:
            return string
        else:
            print(string)
            return

    # Decide which path we'll be taking
    ansistring = ''
    alt_usage = (fg is not None) or (bg is not None) or (style is not None)

    # Handle alternate usage pattern -- string as first argument rather than color, so swap
    if alt_usage:
        if string is None and color is not None:
            string, color = color, string
        if color is not None: # pragma: no cover
            errormsg = 'You can supply either color or fg, but not both'
            raise ValueError(errormsg)

        if ansi_support: ansistring = ac.color(s=string, fg=fg, bg=bg, style=style) # Actually apply color
        else:            ansistring = str(string) # Otherwise, just return the string # pragma: no cover

    # Original use case
    else:

        # Define ANSI colors
        ansicolors = co.OrderedDict([
            ('black', '30'),
            ('red', '31'),
            ('green', '32'),
            ('yellow', '33'),
            ('blue', '34'),
            ('magenta', '35'),
            ('cyan', '36'),
            ('gray', '37'),
            ('bgblack', '40'),
            ('bgred', '41'),
            ('bggreen', '42'),
            ('bgyellow', '43'),
            ('bgblue', '44'),
            ('bgmagenta', '45'),
            ('bgcyan', '46'),
            ('bggray', '47'),
            ('reset', '0'),
        ])
        for key, val in ansicolors.items():
            ansicolors[key] = '\033[' + val + 'm'

        # Determine what color to use
        colorlist = sc.tolist(color)  # Make sure it's a list
        for color in colorlist:
            if color not in ansicolors.keys(): # pragma: no cover
                print(f'Color "{color}" is not available, use colorize(showhelp=True) to show options.')
                return  # Don't proceed if the color isn't found
        ansicolor = ''
        for color in colorlist:
            ansicolor += ansicolors[color]

        # Modify string, if supplied
        if string is None: ansistring = ansicolor # Just return the color
        else:              ansistring = ansicolor + str(string) + ansicolors['reset'] # Add to start and end of the string
        if not ansi_support: # pragma: no cover
            ansistring = str(string) # To avoid garbling output on unsupported systems

    if showhelp:
        print('Available colors are:')
        for key in ansicolors.keys():
            if key[:2] == 'bg':
                darks = ['bgblack', 'bgred', 'bgblue', 'bgmagenta']
                if key in darks: foreground = 'gray'
                else:            foreground = 'black'
                helpcolor = [foreground, key]
            else:
                helpcolor = key
            colorize(helpcolor, '  ' + key)

    return sc.sc_utils._printout(string=ansistring, doprint=doprint, output=output)


# Alias certain colors functions -- not including white and black since poor practice on light/dark terminals
def printred(s, **kwargs):
    """ Alias to print(colors.red(s)) """
    return print(ac.red(s, **kwargs))

def printgreen(s, **kwargs):
    """ Alias to print(colors.green(s)) """
    return print(ac.green(s, **kwargs))

def printblue(s, **kwargs):
    """ Alias to print(colors.blue(s)) """
    return print(ac.blue(s, **kwargs))

def printcyan(s, **kwargs):
    """ Alias to print(colors.cyan(s)) """
    return print(ac.cyan(s, **kwargs))

def printyellow(s, **kwargs):
    """ Alias to print(colors.yellow(s)) """
    return print(ac.yellow(s, **kwargs))

def printmagenta(s, **kwargs):
    """ Alias to print(colors.magenta(s)) """
    return print(ac.magenta(s, **kwargs))


def heading(string='', *args, color='cyan', divider='—', spaces=2, spacesafter=1, 
            minlength=10, maxlength=200, sep=' ', doprint=None, output=False, **kwargs):
    """
    Create a colorful heading. If just supplied with a string (or list of inputs like print()),
    create blue text with horizontal lines above and below and 3 spaces above. You
    can customize the color, the divider character, how many spaces appear before
    the heading, and the minimum length of the divider (otherwise will expand to
    match the length of the string, up to a maximum length).

    Args:
        string      (str):  the string to print as the heading (or object to convert to a string)
        args        (list): additional strings to print
        color       (str):  color to use for the heading (default cyan)
        divider     (str):  symbol to use for the divider (default '—')
        spaces      (int):  number of spaces to put before the heading (default 2)
        spacesafter (int):  number of spaces to put after the heading (default 1)
        minlength   (int):  minimum length of the divider (default 10)
        maxlength   (int):  maximum length of the divider (default 200)
        sep         (str):  if multiple arguments are supplied, use this separator to join them
        doprint     (bool): whether to print the string (default true if no output)
        output      (bool): whether to return the string as output (else, print)
        kwargs      (dict): passed to :func:`sc.colorize() <colorize>`

    Returns:
        Formatted string if ``output=True``

    **Examples**::
        sc.heading('This is a heading')
        sc.heading(string='This is also a heading', color='red', divider='*', spaces=0, minlength=50)

    | *New in version 1.3.1.*: "spacesafter"
    """

    # Convert to single string
    args = sc.mergelists(string, list(args))
    string = sep.join([str(item) for item in args])

    # Add header and footer
    length      = int(np.median([minlength, len(string), maxlength]))
    space       = '\n'*spaces
    spaceafter  = '\n'*spacesafter
    fulldivider = divider*length
    if fulldivider:
        string = sc.newlinejoin(fulldivider, string, fulldivider)
    fullstring = space + string + spaceafter

    # Create output
    return colorize(color=color, string=fullstring, doprint=doprint, output=output, **kwargs)



#%% Other

__all__ += ['printv', 'slacknotification', 'printtologfile', 'percentcomplete', 
            'progressbar', 'progressbars', 'capture']


def printv(string, thisverbose=1, verbose=2, indent=2, **kwargs):
    """
    Optionally print a message and automatically indent. The idea is that
    a global or shared "verbose" variable is defined, which is passed to
    subfunctions, determining how much detail to print out.

    The general idea is that verbose is an integer from 0-4 as follows:

    * 0 = no printout whatsoever
    * 1 = only essential warnings, e.g. suppressed exceptions
    * 2 = standard printout
    * 3 = extra debugging detail (e.g., printout on each iteration)
    * 4 = everything possible (e.g., printout on each timestep)

    Thus a very important statement might be e.g.

    >>> sc.printv('WARNING, everything is wrong', 1, verbose)

    whereas a much less important message might be

    >>> sc.printv(f'This is timestep {i}', 4, verbose)
    
    Args:
        string (str): string to print
        thisverbose (int): level of verbosity at which to print this message
        verbose (int): global verbose variable
        indent (int): amount by which to indent based on verbosity level
        kwargs (dict): passed to :func:`print()`
        
    *New in version 3.0.0:* "kwargs" argument; removed "newline" argument
    """
    if verbose >= thisverbose: # Only print if sufficiently verbose
        indents = ' '*thisverbose*indent # Create automatic indenting
        print(indents+sc.flexstr(string), **kwargs) # Actually print
    return


def slacknotification(message=None, webhook=None, to=None, fromuser=None, verbose=2, die=False):  # pragma: no cover
    """
    Send a Slack notification when something is finished.

    The webhook is either a string containing the webhook itself, or a plain text file containing
    a single line which is the Slack webhook. By default it will look for the file
    ".slackurl" in the user's home folder. The webhook needs to look something like
    "https://hooks.slack.com/services/af7d8w7f/sfd7df9sb/lkcpfj6kf93ds3gj". Webhooks are
    effectively passwords and must be kept secure! Alternatively, you can specify the webhook
    in the environment variable SLACKURL.

    Args:
        message (str): The message to be posted.
        webhook (str): See above
        to (str): The Slack channel or user to post to. Channels begin with #, while users begin with @ (note: ignored by new-style webhooks)
        fromuser (str): The pseudo-user the message will appear from (note: ignored by new-style webhooks)
        verbose (bool): How much detail to display.
        die (bool): If false, prints warnings. If true, raises exceptions.

    **Example**::

        sc.slacknotification('Long process is finished')
        sc.slacknotification(webhook='/.slackurl', channel='@username', message='Hi, how are you going?')

    What's the point? Add this to the end of a very long-running script to notify
    your loved ones that the script has finished.

    Version: 2018sep25
    """
    try:
        from requests import post # Simple way of posting data to a URL
        from json import dumps # For sanitizing the message
    except Exception as E:
        errormsg = f'Cannot use Slack notification since imports failed: {str(E)}'
        if die: raise ImportError(errormsg)
        else:   print(errormsg)

    # Validate input arguments
    printv('Sending Slack message', 1, verbose)
    if not webhook:  webhook    = os.path.expanduser('~/.slackurl')
    if not to:       to       = '#general'
    if not fromuser: fromuser = 'sciris-bot'
    if not message:  message  = 'This is an automated notification: your notifier is notifying you.'
    printv(f'Channel: {to} | User: {fromuser} | Message: {message}', 3, verbose) # Print details of what's being sent

    # Try opening webhook as a file
    if webhook.find('hooks.slack.com')>=0: # It seems to be a URL, let's proceed
        slackurl = webhook
    elif os.path.exists(os.path.expanduser(webhook)): # If not, look for it as a file
        with open(os.path.expanduser(webhook)) as f: slackurl = f.read()
    elif os.getenv('SLACKURL'): # See if it's set in the user's environment variables
        slackurl = os.getenv('SLACKURL')
    else:
        slackurl = webhook # It doesn't seemt to be a URL but let's try anyway
        errormsg = f'"{webhook}" does not seem to be a valid webhook string or file'
        if die: raise ValueError(errormsg)
        else:   print(errormsg)

    # Package and post payload
    try:
        payload = '{"text": %s, "channel": %s, "username": %s}' % (dumps(message), dumps(to), dumps(fromuser))
        printv(f'Full payload: {payload}', 4, verbose)
        response = post(url=slackurl, data=payload)
        printv(response, 3, verbose) # Optionally print response
        printv('Message sent.', 2, verbose) # We're done
    except Exception as E:
        errormsg = f'Sending of Slack message failed: {repr(E)}'
        if die: raise RuntimeError(errormsg)
        else:   print(errormsg)
    return


def printtologfile(message=None, filename=None):
    """
    Append a message string to a file specified by a filename name/path.

    Note: in almost all cases, you are better off using Python's built-in logging
    system rather than this function.
    """

    # Set defaults
    if message is None: # pragma: no cover
        return # Return immediately if nothing to append
    if filename is None:
        import tempfile
        tempdir = tempfile.gettempdir()
        filename = os.path.join(tempdir, 'logfile') # Some generic filename that should work on *nix systems

    # Try writing to file
    try:
        with open(filename, 'a') as f:
            f.write('\n'+message+'\n') # Add a newline to the message.
    except Exception as E: # pragma: no cover # Fail gracefully
        print(f'Warning, could not write to logfile {filename}: {str(E)}')

    return


def percentcomplete(step=None, maxsteps=None, stepsize=1, prefix=None):
    """
    Display progress as a percentage.

    **Examples**::

        maxiters = 500
        
        # Will print on every 5th iteration
        for i in range(maxiters):
            sc.percentcomplete(i, maxiters) 
        
        # Will print on every 50th iteration
        for i in range(maxiters):
            sc.percentcomplete(i, maxiters, stepsize=10)
        
        # Will print e.g. 'Completeness: 1%'
        for i in range(maxiters):
            sc.percentcomplete(i, maxiters, prefix='Completeness: ') 
    
    See also :func:`sc.progressbar() <progressbar>` for a progress bar.
    """
    if prefix is None:
        prefix = ' '
    elif sc.isnumber(prefix): # pragma: no cover
        prefix = ' '*prefix
    onepercent = max(stepsize,round(maxsteps/100*stepsize)) # Calculate how big a single step is -- not smaller than 1
    if not step%onepercent: # Does this value lie on a percent
        thispercent = round(step/maxsteps*100) # Calculate what percent it is
        print(prefix + '%i%%'% thispercent) # Display the output
    return


def progressbar(i=None, maxiters=None, label='', every=1, length=30, empty='—', full='•', 
                newline=False, flush=False, **kwargs):
    """
    Show a progress bar for a for loop.
    
    It can be called manually inside each iteration of the loop, or it can be used 
    to wrap the object being iterated. In the latter case, it acts as an alias 
    for the ``tqdm.tqdm()`` progress bar.

    Args:
        i        (int/iterable): current iteration (for text output), or iterable object (for tqdm)
        maxiters (int): maximum number of iterations (can also use an object with length)
        label    (str): initial label to print
        every    (int/float): if int, print every "every"th iteration (if 1, print all); if float and <1, print every maxiters*every iteration
        length   (int): length of progress bar
        empty    (str): character for not-yet-completed steps
        full     (str): character for completed steps
        newline  (bool): whether to print each iteration on a new line
        flush    (bool): whether to force-flush the buffer
        kwargs   (dict): passed to ``tqdm.tqdm()``; see its documentation for full options

    **Examples**::

        # Direct usage inside a loop
        for i in range(20):
            sc.progressbar(i+1, 20)
            sc.timedsleep(0.05)
        
        # Direct usage inside a loop with custom formatting
        for i in range(1000):
            sc.progressbar(i+1, 1000, every=100, length=10, empty=' ', full='✓', newline=True)
            sc.timedsleep(0.001)

        # Used to wrap an iterable, using tqdm
        x = np.arange(100)
        for i in sc.progressbar(x):
            plt.pause(0.01)

    Adapted from example by Greenstick (https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console)

    | *New in version 1.3.3:* "every" argument
    | *New in version 3.0.0:* wrapper for tqdm
    """
    if i is None or sc.isiterable(i):
        desc = kwargs.pop('desc', label)
        return tqdm.tqdm(i, desc=desc, **kwargs)
    
    # Handle inputs
    if hasattr(maxiters, '__len__'):
        maxiters = len(maxiters)
    ending = None if newline else '\r'
    if every < 1: # pragma: no cover
        every = max(1, int(every*maxiters)) # Don't let it go below 1

    # Calculate percent and handle zero case
    if maxiters > 0:
        pct = i/maxiters*100
    else: # pragma: no cover
        i = 1
        maxiters = 1
        pct = 100
    percent = f'{pct:0.0f}%'

    # Assemble string
    filled = int(length*i//maxiters)
    bar = full*filled + empty*(length-filled)

    # Print
    lastiter = (i == maxiters)
    if not(i%every) or lastiter:
        print(f'\r{label} {bar} {percent}', end=ending, flush=flush)
        if lastiter:
            print() # Newline at the end

    return


class tqdm_pickle(tqdm.tqdm):
    """
    Simple subclass of tqdm that allows pickling
    
    Usually not used directly by the user; used via :func:`sc.progressbars() <progressbars>` instead.
    Pickling is required for passing ``tqdm`` instances between processes.
    
    Based on ``tqdm`` 4.65.0; may become deprecated in future ``tqdm`` releases.
    
    *New in version 3.0.0.*
    """
    
    def __getstate__(self):
        """ Overwrite default __getstate__ """
        d = {k:v for k,v in self.__dict__.items() if k not in ['sp', 'fp']}
        return d
     
    def __setstate__(self, d): # pragma: no cover
        """ Overwrite default __getstate__ """
        self.__dict__ = d
        self.__dict__['fp'] = tqdm.utils.DisableOnWriteError(sys.stderr, tqdm_instance=self)
        self.__dict__['sp'] = self.status_printer(self.fp)
        return


class progressbars(prettyobj):
    """
    Create multiple progress bars
    
    Useful for tracking the progress of multiple long-running tasks. Unlike regular
    ``tqdm`` instances, this uses a pickable version so it can be used directly
    in multiprocessing instances.
    
    Args:
        n (int): number of progress bars to create
        total (float): length of the progress bars
        label (str/list): an optional prefix for the progress bar, or list of all labels
        leave (bool): whether to remove the progress bars when they're done
        kwargs (dict): passed to ``tqdm.tqdm()``
    
    **Note**: bars are supposed to update in-place, but may appear on separate lines
    instead if not run in the terminal (e.g. if run in IPython environments like 
    Spyder or Jupyter).
    
    **Example**::
        
        import sciris as sc
        import random

        def run_sim(index, ndays, pbs):
            for i in range(ndays):
                val = random.random()
                sc.timedsleep(val*5/ndays)
                pbs.update(index) # Update this progress bar based on the index
            return

        nsims = 5
        ndays = 365

        # Create progress bars
        pbs = sc.progressbars(nsims, total=ndays, label='Sim')

        # Run tasks
        sc.parallelize(run_sim, iterarg=range(nsims), ndays=ndays, pbs=pbs)
        
        # Produces output like:
        # Sim 0:  39%|███████████████████████████▊           | 143/365 [00:01<00:01, 137.17it/s]
        # Sim 1:  42%|████████████████████████████▉          | 154/365 [00:01<00:01, 148.70it/s]
        # Sim 2:  45%|████████████████████████████████       | 165/365 [00:01<00:01, 144.19it/s]
        # Sim 3:  44%|███████████████████████████████        | 160/365 [00:01<00:01, 151.22it/s]
        # Sim 4:  42%|████████████████████████████▏          | 145/365 [00:01<00:01, 136.75it/s]

    *New in version 3.0.0.*
    """
    
    def __init__(self, n=1, total=1, label=None, leave=False, **kwargs):
        self.n      = n
        self.total  = total
        self.leave  = leave
        self.desc   = kwargs.pop('desc', label)
        self.kwargs = kwargs
        self.bars   = []
        self.make()
        return

    def make(self):
        for i in range(self.n):
            total = self.total
            if sc.isiterable(total):
                total = total[i]
            desc = self.desc
            if sc.isiterable(desc, exclude=str):
                desc = desc[i]
            else:
                if desc is None:
                    desc = f'{i}'
                else:
                    desc += f' {i}'
            bar = tqdm_pickle(total=total, position=i, desc=desc, leave=self.leave, **self.kwargs)
            self.bars.append(bar)
        return
    
    def __getitem__(self, key): # pragma: no cover
        return self.bars[key]
    
    def __setitem__(self, key, value): # pragma: no cover
        self.bars[key] = value
        return
    
    def update(self, index=0, amount=1): # pragma: no cover
        self.bars[index].update(amount)
        return


class capture(co.UserString, str, redirect_stdout):
    """
    Captures stdout (e.g., from :func:`print()`) as a variable.

    Based on :obj:`contextlib.redirect_stdout`, but saves the user the trouble of
    defining and reading from an IO stream. Useful for testing the output of functions
    that are supposed to print certain output.

    **Examples**::

        # Using with...as
        with sc.capture() as txt1:
            print('Assign these lines')
            print('to a variable')

        # Using start()...stop()
        txt2 = sc.capture().start()
        print('This works')
        print('the same way')
        txt2.stop()

        print('txt1:')
        print(txt1)
        print('txt2:')
        print(txt2)

    *New in version 1.3.3.*
    """

    def __init__(self, seq='', *args, **kwargs):
        self._io = io.StringIO()
        self.stdout = sys.stdout
        co.UserString.__init__(self, seq=seq, *args, **kwargs)
        redirect_stdout.__init__(self, self._io)
        return

    def __enter__(self, *args, **kwargs):
        redirect_stdout.__enter__(self, *args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        self.data += self._io.getvalue()
        redirect_stdout.__exit__(self, *args, **kwargs)
        return

    def start(self):
        self.__enter__()
        return self

    def stop(self):
        self.__exit__(None, None, None)
        return