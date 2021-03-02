'''
Miscellaneous utilities for type checking, printing, dates and times, etc.

Note: there are a lot! The design philosophy has been that it's easier to
ignore a function that you don't need than write one from scratch that you
do need.

Highlights:
    - ``sc.dcp()``: shortcut to ``copy.deepcopy()``
    - ``sc.pp()``: shortcut to ``pprint.pprint()``
    - ``sc.pr()``: print full representation of an object, including methods and each attribute
    - ``sc.heading()``: print text as a 'large' heading
    - ``sc.colorize()``: print text in a certain color
    - ``sc.sigfigs()``: truncate a number to a certain number of significant figures
    - ``sc.isnumber()``: checks if something is any number type
    - ``sc.promotetolist()``: converts any object to a list, for easy iteration
    - ``sc.promotetoarray()``: tries to convert any object to an array, for easy use with numpy
    - ``sc.readdate()``: convert strings to dates using common formats
    - ``sc.tic()/sc.toc()``: simple method for timing durations
    - ``sc.runcommand()``: simple way of executing a shell command
'''

##############################################################################
### Imports
##############################################################################

import os
import re
import sys
import copy
import time
import json
import zlib
import psutil
import pprint
import hashlib
import dateutil
import subprocess
import itertools
import numbers
import string
import numpy as np
import random as rnd
import datetime as dt
import uuid as py_uuid
import tempfile
import traceback as py_traceback
from textwrap import fill
from functools import reduce
from collections import OrderedDict as OD
from distutils.version import LooseVersion

# Handle types and legacy Python 2 compatibility
_stringtypes = (str, bytes)
_numtype    = numbers.Number

# Add Windows support for colors (do this at the module level so that colorama.init() only gets called once)
if 'win' in sys.platform:
    try:
        import colorama
        colorama.init()
        ansi_support = True
    except:
        ansi_support = False  # print('Warning: you have called colorize() on Windows but do not have either the colorama or tendo modules.')
else:
    ansi_support = True



##############################################################################
### Adaptations from other libraries
##############################################################################

# Define the modules being loaded
__all__ = ['fast_uuid', 'uuid', 'dcp', 'cp', 'pp', 'sha', 'wget', 'htmlify', 'traceback']


def fast_uuid(which=None, length=None, n=1, secure=False, forcelist=False, safety=1000, recursion=0, recursion_limit=10, verbose=True):
    '''
    Create a fast UID or set of UIDs.

    Args:
        which (str): the set of characters to choose from (default ascii)
        length (int): length of UID (default 6)
        n (int): number of UIDs to generate
        forcelist (bool): whether or not to return a list even for a single UID (used for recursive calls)
        safety (float): ensure that the space of possible UIDs is at least this much larger than the number requested
        recursion (int): the recursion level of the call (since the function calls itself if not all UIDs are unique)
        recursion_limit (int): # Maximum number of times to try regeneraring keys

    Returns:
        uid (str or list): a string UID, or a list of string UIDs

    **Example**::

        uuids = sc.fast_uuid(n=100) # Generate 100 UUIDs

    Inspired by https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits/30038250#30038250
    '''

    # Set defaults
    if which  is None: which  = 'ascii'
    if length is None: length = 6
    length = int(length)
    n = int(n)

    choices = {
        'lowercase':    string.ascii_lowercase,
        'letters':      string.ascii_letters,
        'numeric':      string.digits,
        'digits':       string.digits,
        'hex':          string.hexdigits.lower(),
        'hexdigits':    string.hexdigits.lower(),
        'alphanumeric': string.ascii_lowercase + string.digits,
        'ascii':        string.ascii_letters + string.digits,
        }

    if which not in choices:
        choicekeys = ', '.join(list(choices.keys()))
        errormsg = f'Choice {which} not found; choices are: {choicekeys}'
        raise KeyError(errormsg)
    else:
        charlist = choices[which]

    # Check that there are enough options
    if n > 1:
        n_possibilities = len(charlist)**length
        allowed = n_possibilities//safety
        if n > allowed:
            errormsg = f'With a UID of type "{which}" and length {length}, there are {n_possibilities} possible UIDs, and you requested {n}, which exceeds the maximum allowed ({allowed})'
            raise ValueError(errormsg)

    # Secure uses system random which is secure, but >10x slower
    if secure:
        choices_func = rnd.SystemRandom().choices
    else:
        choices_func = rnd.choices

    # Generate the UUID(s) string as one big block
    uid_str = ''.join(choices_func(charlist, k=length*n))

    # Parse if n==1
    if n == 1:
        if forcelist:
            output = [uid_str]
        else:
            output = uid_str

    # Otherwise, we're generating multiple, so do additional checking to ensure they're actually unique
    else:
        # Split from one long string into multiple and check length
        output = [uid_str[chunk*length:(chunk+1)*length] for chunk in range(len(uid_str)//length)]
        n_unique_keys = len(dict.fromkeys(output))

        # Check that length is correct, i.e. no duplicates!
        while n_unique_keys != n:

            # Set recursion and do error checking
            recursion += 1
            if recursion > recursion_limit:
                errormsg = f'Could only generate {n_unique_keys}/{n} unique UIDs after {recursion_limit} tries: please increase UID length or character set size to ensure more unique options'
                raise ValueError(errormsg)
            if verbose:
                print(f'Warning: duplicates found in UID list ({n_unique_keys}/{n} unique); regenerating...')

            # Extend the list of UIDs
            new_n = n - n_unique_keys
            new_uuids = fast_uuid(which=which, length=length, n=new_n, secure=secure, safety=safety, recursion=recursion, recursion_limit=recursion_limit, verbose=verbose, forcelist=True)
            output.extend(new_uuids)
            n_unique_keys = len(dict.fromkeys(output)) # Recalculate the number of keys

    return output


def uuid(uid=None, which=None, die=False, tostring=False, length=None, n=1, **kwargs):
    '''
    Shortcut for creating a UUID; default is to create a UUID4. Can also convert a UUID.

    Args:
        uid (str or uuid): if a string, convert to an actual UUID; otherwise, return unchanged
        which (int or str): if int, choose a Python UUID function; otherwise, generate a random alphanumeric string (default 4)
        die (bool): whether to fail for converting a supplied uuid (default False)
        tostring (bool): whether or not to return a string instead of a UUID object (default False)
        length (int): number of characters to trim to, if returning a string
        n (int): number of UUIDs to generate; if n>1, return a list

    Returns:
        uid (UUID or str): the UID object

    **Examples**::

        sc.uuid() # Alias to uuid.uuid4()
        sc.uuid(which='hex') # Creates a length-6 hex string
        sc.uuid(which='ascii', length=10, n=50) # Creates 50 UUIDs of length 10 each using the full ASCII character set
    '''

    # Set default UUID type
    if which is None:
        which = 4
    n = int(n)

    # Choose the different functions
    if   which==1: uuid_func = py_uuid.uuid1
    elif which==3: uuid_func = py_uuid.uuid3
    elif which==4: uuid_func = py_uuid.uuid4
    elif which==5: uuid_func = py_uuid.uuid5
    else:
        return fast_uuid(which=which, length=length, n=n, **kwargs) # ...or just go to fast_uuid()

    # If a UUID was supplied, try to parse it
    if uid is not None:
        try:
            if isinstance(uid, py_uuid.UUID):
                output = uid # Use directly
            else: # Convert
                output = py_uuid.UUID(uid)
        except Exception as E:
            errormsg = 'Could not convert "%s" to a UID (%s)' % (uid, repr(E))
            if die:
                raise Exception(errormsg)
            else:
                print(errormsg)
                uid = None # Just create a new one

    # If not, make a new one
    if uid is None:
        uuid_list = []
        for i in range(n): # Loop over
            uid = uuid_func(**kwargs)  # If not supplied, create a new UUID

            # Convert to a string, and optionally trim
            if tostring or length:
                uid = str(uid)
            if length:
                if length<len(uid):
                    uid = uid[:length]
                else:
                    errormsg = f'Cannot choose first {length} chars since UID has length {len(uid)}'
                    raise ValueError(errormsg)
            uuid_list.append(uid)

        # Process the output: string if 1, list if more
        if len(uuid_list) == 1:
            output = uuid_list[0]
        else:
            output = uuid_list

    return output


def dcp(obj, verbose=True, die=False):
    '''
    Shortcut to perform a deep copy operation

    Almost identical to ``copy.deepcopy()``
    '''
    try:
        output = copy.deepcopy(obj)
    except Exception as E:
        output = cp(obj)
        errormsg = 'Warning: could not perform deep copy, performing shallow instead: %s' % str(E)
        if die: raise Exception(errormsg)
        else:   print(errormsg)
    return output


def cp(obj, verbose=True, die=True):
    '''
    Shortcut to perform a shallow copy operation

    Almost identical to ``copy.copy()``
    '''
    try:
        output = copy.copy(obj)
    except Exception as E:
        output = obj
        errormsg = 'Could not perform shallow copy, returning original object'
        if die: raise ValueError(errormsg) from E
        else:   print(errormsg)
    return output


def pp(obj, jsonify=True, verbose=False, doprint=True, *args, **kwargs):
    '''
    Shortcut for pretty-printing the object

    Almost identical to ``pprint.pprint()``
    '''
    # Get object
    if jsonify:
        try:
            toprint = json.loads(json.dumps(obj)) # This is to handle things like OrderedDicts
        except Exception as E:
            if verbose: print('Could not jsonify object ("%s"), printing default...' % str(E))
            toprint = obj # If problems are encountered, just return the object
    else:
        toprint = obj

    # Decide what to do with object
    if doprint:
        pprint.pprint(toprint, *args, **kwargs)
        return None
    else:
        output = pprint.pformat(toprint, *args, **kwargs)
        return output


def sha(obj, encoding='utf-8', digest=False):
    '''
    Shortcut for the standard hashing (SHA) method

    Equivalent to ``hashlib.sha224()``.

    Args:
        obj (any): the object to be hashed; if not a string, converted to one
        encoding (str): the encoding to use
        digest (bool): whether to return the hex digest instead of the hash objet

    **Example**::

        sha1 = sc.sha(dict(foo=1, bar=2), digest=True)
        sha2 = sc.sha(dict(foo=1, bar=2), digest=True)
        sha3 = sc.sha(dict(foo=1, bar=3), digest=True)
        assert sha1 == sha2
        assert sha2 != sha3
    '''
    if not isstring(obj): # Ensure it's actually a string
        string = repr(obj)
    else:
        string = obj
    needsencoding = isinstance(string, str)
    if needsencoding: # If it's unicode, encode it to bytes first
        string = string.encode(encoding)
    output = hashlib.sha224(string)
    if digest:
        output = output.hexdigest()
    return output


def wget(url, convert=True):
    '''
    Download a URL

    Alias to urllib.request.urlopen(url).read()

    **Example**::

        html = sc.wget('http://sciris.org')
    '''
    from urllib import request # Bizarrely, urllib.request sometimes fails
    output = request.urlopen(url).read()
    if convert:
        output = output.decode()
    return output


def htmlify(string, reverse=False, tostring=False):
    '''
    Convert a string to its HTML representation by converting unicode characters,
    characters that need to be escaped, and newlines. If reverse=True, will convert
    HTML to string. If tostring=True, will convert the bytestring back to Unicode.

    **Examples**::

        output = sc.htmlify('foo&\\nbar') # Returns b'foo&amp;<br>bar'
        output = sc.htmlify('föö&\\nbar', tostring=True) # Returns 'f&#246;&#246;&amp;&nbsp;&nbsp;&nbsp;&nbsp;bar'
        output = sc.htmlify('foo&amp;<br>bar', reverse=True) # Returns 'foo&\\nbar'
    '''
    import html
    if not reverse: # Convert to HTML
        output = html.escape(string).encode('ascii', 'xmlcharrefreplace') # Replace non-ASCII characters
        output = output.replace(b'\n', b'<br>') # Replace newlines with <br>
        output = output.replace(b'\t', b'&nbsp;&nbsp;&nbsp;&nbsp;') # Replace tabs with 4 spaces
        if tostring: # Convert from bytestring to unicode
            output = output.decode()
    else: # Convert from HTML
        output = html.unescape(string)
        output = output.replace('<br>','\n').replace('<br />','\n').replace('<BR>','\n')
    return output


def traceback(*args, **kwargs):
    '''
    Shortcut for accessing the traceback

    Alias for ``traceback.format_exc()``.
    '''
    return py_traceback.format_exc(*args, **kwargs)



##############################################################################
### Printing/notification functions
##############################################################################

__all__ += ['printv', 'blank', 'createcollist', 'objectid', 'objatt', 'objmeth', 'objprop', 'objrepr',
            'prepr', 'pr', 'indent', 'sigfig', 'printarr', 'printdata', 'printvars',
            'slacknotification', 'printtologfile', 'colorize', 'heading', 'percentcomplete', 'progressbar']

def printv(string, thisverbose=1, verbose=2, newline=True, indent=True):
    '''
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

    Version: 2016jan30
    '''
    if thisverbose>4 or verbose>4: print('Warning, verbosity should be from 0-4 (this message: %i; current: %i)' % (thisverbose, verbose))
    if verbose>=thisverbose: # Only print if sufficiently verbose
        indents = '  '*thisverbose*bool(indent) # Create automatic indenting
        if newline: print(indents+flexstr(string)) # Actually print
        else: print(indents+flexstr(string)), # Actually print
    return None


def blank(n=3):
    ''' Tiny function to print n blank lines, 3 by default '''
    print('\n'*n)


def createcollist(oldkeys, title, strlen = 18, ncol = 3):
    ''' Creates a string for a nice columnated list (e.g. to use in __repr__ method) '''
    nrow = int(np.ceil(float(len(oldkeys))/ncol))
    newkeys = []
    for x in range(nrow):
        newkeys += oldkeys[x::nrow]

    attstring = title + ':'
    c = 0
    for x in newkeys:
        if c%ncol == 0: attstring += '\n  '
        if len(x) > strlen: x = x[:strlen-3] + '...'
        attstring += '%-*s  ' % (strlen,x)
        c += 1
    attstring += '\n'
    return attstring


def objectid(obj):
    ''' Return the object ID as per the default Python __repr__ method '''
    output = '<%s.%s at %s>\n' % (obj.__class__.__module__, obj.__class__.__name__, hex(id(obj)))
    return output


def objatt(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object attributes for the Python __repr__ method '''
    oldkeys = sorted(obj.__dict__.keys())
    if len(oldkeys): output = createcollist(oldkeys, 'Attributes', strlen = 18, ncol = 3)
    else:            output = ''
    return output


def objmeth(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object methods for the Python __repr__ method '''
    try:
        oldkeys = sorted([method + '()' for method in dir(obj) if callable(getattr(obj, method)) and not method.startswith('__')])
    except:
        oldkeys = ['Methods N/A']
    if len(oldkeys): output = createcollist(oldkeys, 'Methods', strlen=strlen, ncol=ncol)
    else:            output = ''
    return output


def objprop(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object properties for the Python __repr__ method '''
    try:
        oldkeys = sorted([prop for prop in dir(obj) if isinstance(getattr(type(obj), prop, None), property) and not prop.startswith('__')])
    except:
        oldkeys = ['Properties N/A']
    if len(oldkeys): output = createcollist(oldkeys, 'Properties', strlen=strlen, ncol=ncol)
    else:            output = ''
    return output


def objrepr(obj, showid=True, showmeth=True, showprop=True, showatt=True, dividerchar='—', dividerlen=60):
    ''' Return useful printout for the Python __repr__ method '''
    divider = dividerchar*dividerlen + '\n'
    output = ''
    if showid:
        output += objectid(obj)
        output += divider
    if showmeth:
        meths = objmeth(obj)
        if meths:
            output += objmeth(obj)
            output += divider
    if showprop:
        props = objprop(obj)
        if props:
            output += props
            output += divider
    if showatt:
        attrs = objatt(obj)
        if attrs:
            output += attrs
            output += divider
    return output


def prepr(obj, maxlen=None, maxitems=None, skip=None, dividerchar='—', dividerlen=60, use_repr=True, maxtime=3, die=False):
    '''
    Akin to "pretty print", returns a pretty representation of an object --
    all attributes (except any that are skipped), plust methods and ID. Usually
    used via the interactive sc.pr() (which prints), rather than this (which returns
    a string).

    Args:
        obj (anything): the object to be represented
        maxlen (int): maximum number of characters to show for each item
        maxitems (int): maximum number of items to show in the object
        skip (list): any properties to skip
        dividerchar (str): divider for methods, attributes, etc.
        divierlen (int): number of divider characters
        use_repr (bool): whether to use repr() or str() to parse the object
        maxtime (float): maximum amount of time to spend on trying to print the object
        die (bool): whether to raise an exception if an error is encountered
    '''

    # Decide how to handle representation function -- repr is dangerous since can lead to recursion
    repr_fn = repr if use_repr else str
    T = time.time() # Start the timer
    time_exceeded = False

    # Handle input arguments
    divider = dividerchar*dividerlen + '\n'
    if maxlen   is None: maxlen   = 80
    if maxitems is None: maxitems = 100
    if skip     is None: skip = []
    else:                skip = promotetolist(skip)

    # Initialize things to print out
    labels = []
    values = []

    # Wrap entire process in a try-except in case it fails
    try:
        if not (hasattr(obj, '__dict__') or hasattr(obj, '__slots__')):
            # It's a plain object
            labels = ['%s' % type(obj)]
            values = [repr_fn(obj)]
        else:
            if hasattr(obj, '__dict__'):
                labels = sorted(set(obj.__dict__.keys()) - set(skip))  # Get the dict attribute keys
            else:
                labels = sorted(set(obj.__slots__) - set(skip))  # Get the slots attribute keys

            if len(labels):
                extraitems = len(labels) - maxitems
                if extraitems>0:
                    labels = labels[:maxitems]
                values = []
                for a,attr in enumerate(labels):
                    if (time.time() - T) < maxtime:
                        try: value = repr_fn(getattr(obj, attr))
                        except: value = 'N/A'
                        values.append(value)
                    else:
                        labels = labels[:a]
                        labels.append('etc. (time exceeded)')
                        values.append(f'{len(labels)-a} entries not shown')
                        time_exceeded = True
                        break
            else:
                items = dir(obj)
                extraitems = len(items) - maxitems
                if extraitems > 0:
                    items = items[:maxitems]
                for a,attr in enumerate(items):
                    if not attr.startswith('__'):
                        if (time.time() - T) < maxtime:
                            try:    value = repr_fn(getattr(obj, attr))
                            except: value = 'N/A'
                            labels.append(attr)
                            values.append(value)
                        else:
                            labels.append('etc. (time exceeded)')
                            values.append(f'{len(labels)-a} entries not shown')
                            time_exceeded = True
            if extraitems > 0:
                labels.append('etc. (too many items)')
                values.append(f'{extraitems} entries not shown')

        # Decide how to print them
        maxkeylen = 0
        if len(labels):
            maxkeylen = max([len(label) for label in labels]) # Find the maximum length of the attribute keys
        if maxkeylen<maxlen:
            maxlen = maxlen - maxkeylen # Shorten the amount of data shown if the keys are long
        formatstr = '%'+ '%i'%maxkeylen + 's' # Assemble the format string for the keys, e.g. '%21s'
        output  = objrepr(obj, showatt=False, dividerchar=dividerchar, dividerlen=dividerlen) # Get the methods
        for label,value in zip(labels,values): # Loop over each attribute
            if len(value)>maxlen: value = value[:maxlen] + ' [...]' # Shorten it
            prefix = formatstr%label + ': ' # The format key
            output += indent(prefix, value)
        output += divider
        if time_exceeded:
            timestr = f'\nNote: the object did not finish printing within maxtime={maxtime} s.\n'
            timestr += 'To see the full object, call prepr() with increased maxtime.'
            output += timestr

    # If that failed, try progressively simpler approaches
    except Exception as E:
        if die:
            errormsg = 'Failed to create pretty representation of object'
            raise RuntimeError(errormsg) from E
        else:
            try: # Next try the objrepr, which is the same except doesn't print attribute values
                output = objrepr(obj, dividerchar=dividerchar, dividerlen=dividerlen)
                output += f'\nWarning: showing simplified output since full repr failed {str(E)}'
            except: # If that fails, try just the string representation
                output = str(obj)

    return output


def pr(obj, *args, **kwargs):
    '''
    Shortcut for printing the pretty repr for an object -- similar to prettyprint

    **Example**::

        import pandas as pd
        df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
        print(df) # See just the data
        sc.pr(df) # See all the methods too
    '''
    print(prepr(obj, *args, **kwargs))
    return None


def indent(prefix=None, text=None, suffix='\n', n=0, pretty=False, simple=True, width=70, **kwargs):
    '''
    Small wrapper to make textwrap more user friendly.

    Args:
        prefix: text to begin with (optional)
        text: text to wrap
        suffix: what to put on the end (by default, a newline)
        n: if prefix is not specified, the size of the indent
        prettify: whether to use pprint to format the text
        kwargs: anything to pass to textwrap.fill() (e.g., linewidth)

    **Examples**::

        prefix = 'and then they said:'
        text = 'blah '*100
        print(indent(prefix, text))

        print('my fave is: ' + indent(text=rand(100), n=14))

    Version: 2017feb20
    '''
    # Handle no prefix
    if prefix is None: prefix = ' '*n

    # Get text in the right format -- i.e. a string
    if pretty: text = pprint.pformat(text)
    else:      text = flexstr(text)

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



def sigfig(x, sigfigs=5, SI=False, sep=False, keepints=False):
    '''
    Return a string representation of variable x with sigfigs number of significant figures

    Args:
        x (int/float/arr): the number(s) to round
        sigfigs (int): number of significant figures to round to
        SI (bool): whether to use SI notation
        sep (bool/str): if provided, use as thousands separator
        keepints (bool): never round ints

    **Examples**::

        x = 32433.3842
        sc.sigfig(x, SI=True) # Returns 32.433k
        sc.sigfig(x, sep=True) # Returns 32,433
    '''
    output = []

    try:
        n=len(x)
        X = x
        islist = True
    except:
        X = [x]
        n = 1
        islist = False
    for i in range(n):
        x = X[i]

        suffix = ''
        formats = [(1e18,'e18'), (1e15,'e15'), (1e12,'t'), (1e9,'b'), (1e6,'m'), (1e3,'k')]
        if SI:
            for val,suff in formats:
                if abs(x)>=val:
                    x = x/val
                    suffix = suff
                    break # Find at most one match

        try:
            if x==0:
                output.append('0')
            elif sigfigs is None:
                output.append(flexstr(x)+suffix)
            elif x>(10**sigfigs) and not SI and keepints: # e.g. x = 23432.23, sigfigs=3, output is 23432
                roundnumber = int(round(x))
                if sep: string = format(roundnumber, ',')
                else:   string = '%0.0f' % x
                output.append(string)
            else:
                magnitude = np.floor(np.log10(abs(x)))
                factor = 10**(sigfigs-magnitude-1)
                x = round(x*factor)/float(factor)
                digits = int(abs(magnitude) + max(0, sigfigs - max(0,magnitude) - 1) + 1 + (x<0) + (abs(x)<1)) # one because, one for decimal, one for minus
                decimals = int(max(0,-magnitude+sigfigs-1))
                strformat = '%' + '%i.%i' % (digits, decimals)  + 'f'
                string = strformat % x
                if sep: # To insert separators in the right place, have to convert back to a number
                    if decimals>0:  roundnumber = float(string)
                    else:           roundnumber = int(string)
                    string = format(roundnumber, ',') # Allow comma separator
                string += suffix
                output.append(string)
        except:
            output.append(flexstr(x))
    if islist:
        return tuple(output)
    else:
        return output[0]



def printarr(arr, arrformat='%0.2f  '):
    '''
    Print a numpy array nicely.

    **Example**::

        sc.printarr(pl.rand(3,7,4))

    Version: 2014dec01
    '''
    if np.ndim(arr)==1:
        string = ''
        for i in range(len(arr)):
            string += arrformat % arr[i]
        print(string)
    elif np.ndim(arr)==2:
        for i in range(len(arr)):
            printarr(arr[i], arrformat)
    elif np.ndim(arr)==3:
        for i in range(len(arr)):
            print('='*len(arr[i][0])*len(arrformat % 1))
            for j in range(len(arr[i])):
                printarr(arr[i][j], arrformat)
    else:
        print(arr) # Give up
    return None



def printdata(data, name='Variable', depth=1, maxlen=40, indent='', level=0, showcontents=False):
    '''
    Nicely print a complicated data structure, a la Matlab.

    Args:
      data: the data to display
      name: the name of the variable (automatically read except for first one)
      depth: how many levels of recursion to follow
      maxlen: number of characters of data to display (if 0, don't show data)
      indent: where to start the indent (used internally)

    Version: 2015aug21
    '''
    datatype = type(data)
    def printentry(data):
        if   datatype==dict:              string = ('dict with %i keys' % len(data.keys()))
        elif datatype==list:              string = ('list of length %i' % len(data))
        elif datatype==tuple:             string = ('tuple of length %i' % len(data))
        elif datatype==np.ndarray:        string = ('array of shape %s' % flexstr(np.shape(data)))
        elif datatype.__name__=='module': string = ('module with %i components' % len(dir(data)))
        elif datatype.__name__=='class':  string = ('class with %i components' % len(dir(data)))
        else: string = datatype.__name__
        if showcontents and maxlen>0:
            datastring = ' | '+flexstr(data)
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
    return None


def printvars(localvars=None, varlist=None, label=None, divider=True, spaces=1, color=None):
    '''
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
    '''

    varlist = promotetolist(varlist) # Make sure it's actually a list
    dividerstr = '-'*40

    if label:  print('Variables for %s:' % label)
    if divider: print(dividerstr)
    for varnum,varname in enumerate(varlist):
        controlstr = '%i. "%s": ' % (varnum, varname) # Basis for the control string -- variable number and name
        if color: controlstr = colorize(color, output=True) + controlstr + colorize('reset', output=True) # Optionally add color
        if spaces>1: controlstr += '\n' # Add a newline if the variables are going to be on different lines
        try:    controlstr += '%s' % localvars[varname] # The variable to be printed
        except: controlstr += 'WARNING, could not be printed' # In case something goes wrong
        controlstr += '\n' * spaces # The number of spaces to add between variables
        print(controlstr), # Print it out
    if divider: print(dividerstr) # If necessary, print the divider again
    return None



def slacknotification(message=None, webhook=None, to=None, fromuser=None, verbose=2, die=False):
    '''
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
    '''
    try:
        from requests import post # Simple way of posting data to a URL
        from json import dumps # For sanitizing the message
    except Exception as E:
        errormsg = 'Cannot use Slack notification since imports failed: %s' % str(E)
        if die: raise Exception(errormsg)
        else:   print(errormsg)

    # Validate input arguments
    printv('Sending Slack message', 1, verbose)
    if not webhook:  webhook    = os.path.expanduser('~/.slackurl')
    if not to:       to       = '#general'
    if not fromuser: fromuser = 'sciris-bot'
    if not message:  message  = 'This is an automated notification: your notifier is notifying you.'
    printv('Channel: %s | User: %s | Message: %s' % (to, fromuser, message), 3, verbose) # Print details of what's being sent

    # Try opening webhook as a file
    if webhook.find('hooks.slack.com')>=0: # It seems to be a URL, let's proceed
        slackurl = webhook
    elif os.path.exists(os.path.expanduser(webhook)): # If not, look for it sa a file
        with open(os.path.expanduser(webhook)) as f: slackurl = f.read()
    elif os.getenv('SLACKURL'): # See if it's set in the user's environment variables
        slackurl = os.getenv('SLACKURL')
    else:
        slackurl = webhook # It doesn't seemt to be a URL but let's try anyway
        errormsg = '"%s" does not seem to be a valid webhook string or file' % webhook
        if die: raise Exception(errormsg)
        else:   print(errormsg)

    # Package and post payload
    try:
        payload = '{"text": %s, "channel": %s, "username": %s}' % (dumps(message), dumps(to), dumps(fromuser))
        printv('Full payload: %s' % payload, 4, verbose)
        response = post(url=slackurl, data=payload)
        printv(response, 3, verbose) # Optionally print response
        printv('Message sent.', 2, verbose) # We're done
    except Exception as E:
        errormsg = 'Sending of Slack message failed: %s' % repr(E)
        if die: raise Exception(errormsg)
        else:   print(errormsg)
    return None


def printtologfile(message=None, filename=None):
    '''
    Append a message string to a file specified by a filename name/path.  This
    is especially useful for capturing information from spawned processes not
    so handily captured through print statements.

    Warning: If you pass a file in, existing or not, it will try to append
    text to it!
    '''

    # Set defaults
    if message is None:
        return None # Return immediately if nothing to append
    if filename is None:
        filename = '/tmp/logfile' # Some generic filename that should work on *nix systems

    # Try writing to file
    try:
        with open(filename, 'a') as f:
            f.write('\n'+message+'\n') # Add a newline to the message.
    except: # Fail gracefully
        print('WARNING, could not write to logfile %s' % filename)

    return None


def colorize(color=None, string=None, output=False, showhelp=False, enable=True):
    '''
    Colorize output text.

    Args:
        color: the color you want (use 'bg' with background colors, e.g. 'bgblue')
        string: the text to be colored
        output: whether to return the modified version of the string
        enable: switch to allow colorize() to be easily turned off

    **Examples**::

        sc.colorize('green', 'hi') # Simple example
        sc.colorize(['yellow', 'bgblack']); print('Hello world'); print('Goodbye world'); colorize() # Colorize all output in between
        bluearray = sc.colorize(color='blue', string=str(range(5)), output=True); print("c'est bleu: " + bluearray)
        sc.colorize('magenta') # Now type in magenta for a while
        sc.colorize() # Stop typing in magenta

    To get available colors, type ``sc.colorize(showhelp=True)``.

    Version: 2018sep09
    '''

    # Handle short-circuit case
    if not enable:
        if output:
            return string
        else:
            print(string)
            return None

    # Define ANSI colors
    ansicolors = OD([
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
    for key, val in ansicolors.items(): ansicolors[key] = '\033[' + val + 'm'

    # Determine what color to use
    colorlist = promotetolist(color)  # Make sure it's a list
    for color in colorlist:
        if color not in ansicolors.keys():
            print('Color "%s" is not available, use colorize(showhelp=True) to show options.' % color)
            return None  # Don't proceed if the color isn't found
    ansicolor = ''
    for color in colorlist:
        ansicolor += ansicolors[color]

    # Modify string, if supplied
    if string is None: ansistring = ansicolor # Just return the color
    else:              ansistring = ansicolor + str(string) + ansicolors['reset'] # Add to start and end of the string
    if not ansi_support: ansistring = str(string) # To avoid garbling output on unsupported systems

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
    elif output:
        return ansistring  # Return the modified string
    else:
        try:    print(ansistring) # Content, so print with newline
        except: print(string) # If that fails, just go with plain version
        return None


def heading(string=None, color=None, divider=None, spaces=None, minlength=None, maxlength=None, **kwargs):
    '''
    Shortcut to sc.colorize() to create a heading. If just supplied with a string,
    create blue text with horizontal lines above and below and 3 spaces above. You
    can customize the color, the divider character, how many spaces appear before
    the heading, and the minimum length of the divider (otherwise will expand to
    match the length of the string, up to a maximum length).

    Args:
        string (str): The string to print as the heading
        color (str): The color to use for the heading (default blue)
        divider (str): The symbol to use for the divider (default em dash)
        spaces (int): The number of spaces to put before the heading
        minlength (int): The minimum length of the divider
        maxlength (int): The maximum length of the divider
        kwargs (dict): Arguments to pass to sc.colorize()

    Returns:
        None, unless specified to produce the string as output using output=True.


    Examples
    --------
    >>> import sciris as sc
    >>> sc.heading('This is a heading')
    >>> sc.heading(string='This is also a heading', color='red', divider='*', spaces=0, minlength=50)
    '''
    if string    is None: string    = ''
    if color     is None: color     = 'cyan' # Reasonable defualt for light and dark consoles
    if divider   is None: divider   = '—' # Em dash for a continuous line
    if spaces    is None: spaces    = 2
    if minlength is None: minlength = 30
    if maxlength is None: maxlength = 120

    length = int(np.median([minlength, len(string), maxlength]))
    space = '\n'*spaces
    if divider and length: fulldivider = '\n'+divider*length+'\n'
    else:                  fulldivider = ''
    fullstring = space + fulldivider + string + fulldivider
    output = colorize(color=color, string=fullstring, **kwargs)
    return output


def percentcomplete(step=None, maxsteps=None, stepsize=1, prefix=None):
    '''
    Display progress.

    **Example**::

        maxiters = 500
        for i in range(maxiters):
            sc.percentcomplete(i, maxiters) # will print on every 5th iteration
            sc.percentcomplete(i, maxiters, stepsize=10) # will print on every 50th iteration
            sc.percentcomplete(i, maxiters, prefix='Completeness: ') # will print e.g. 'Completeness: 1%'
    '''
    if prefix is None:
        prefix = ' '
    elif isnumber(prefix):
        prefix = ' '*prefix
    onepercent = max(stepsize,round(maxsteps/100*stepsize)); # Calculate how big a single step is -- not smaller than 1
    if not step%onepercent: # Does this value lie on a percent
        thispercent = round(step/maxsteps*100) # Calculate what percent it is
        print(prefix + '%i%%'% thispercent) # Display the output
    return None


def progressbar(i, maxiters, label='', length=30, empty='—', full='•', newline=False):
    '''
    Call in a loop to create terminal progress bar.

    Args:
        i (int): current iteration
        maxiters (int): maximum number of iterations
        label (str): initial label to print
        length (int): length of progress bar
        empty (str): character for empty steps
        full (str): character for empty steps

    **Example**::

        import pylab as pl
        for i in range(100):
            progressbar(i+1, 100)
            pl.pause(0.05)

    Adapted from example by Greenstick (https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console)
    '''
    ending = None if newline else '\r'
    pct = i/maxiters*100
    percent = f'{pct:0.0f}%'
    filled = int(length*i//maxiters)
    bar = full*filled + empty*(length-filled)
    print(f'\r{label} {bar} {percent}', end=ending)
    if i == maxiters: print()
    return



##############################################################################
### Type functions
##############################################################################

__all__ += ['flexstr', 'isiterable', 'checktype', 'isnumber', 'isstring', 'isarray', 'promotetoarray', 'promotetolist', 'mergedicts']

def flexstr(arg, force=True):
    ''' Try converting to a "regular" string (i.e. "str" in both Python 2 or 3), but proceed if it fails '''
    if isstring(arg): # It's a string
        if isinstance(arg, bytes):
            try:
                output = arg.decode() # If it's bytes, decode to unicode
            except:
                if force: output = repr(arg) # If that fails, just print its representation
                else:     output = arg
        else:
            output = arg # Otherwise, return as-is
    else:
        if force: output = repr(arg)
        else:     output = arg # Optionally don't do anything for non-strings
    return output


def isiterable(obj):
    '''
    Simply determine whether or not the input is iterable.

    Works by trying to iterate via iter(), and if that raises an exception, it's
    not iterable.

    From http://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    '''
    try:
        iter(obj)
        return True
    except:
        return False


def checktype(obj=None, objtype=None, subtype=None, die=False):
    '''
    A convenience function for checking instances. If objtype is a type,
    then this function works exactly like isinstance(). But, it can also
    be a string, e.g. 'array'.

    If subtype is not None, then checktype will iterate over obj and check
    recursively that each element matches the subtype.

    Special types are "listlike", which will check for lists, tuples, and
    arrays; and "arraylike", which is the same as "listlike" but will also
    check that elements are numeric.

    Args:
        obj:     the object to check the type of
        objtype: the type to confirm the object belongs to
        subtype: optionally check the subtype if the object is iterable
        die:     whether or not to raise an exception if the object is the wrong type

    **Examples**::

        sc.checktype(rand(10), 'array', 'number') # Returns True
        sc.checktype(['a','b','c'], 'listlike') # Returns True
        sc.checktype(['a','b','c'], 'arraylike') # Returns False
        sc.checktype([{'a':3}], list, dict) # Returns True
    '''

    # Handle "objtype" input
    if   objtype in ['str','string']:          objinstance = _stringtypes
    elif objtype in ['num', 'number']:         objinstance = _numtype
    elif objtype in ['arr', 'array']:          objinstance = np.ndarray
    elif objtype in ['listlike', 'arraylike']: objinstance = (list, tuple, np.ndarray) # Anything suitable as a numerical array
    elif type(objtype) == type:                objinstance = objtype  # Don't need to do anything
    elif objtype is None:                      return None # If not supplied, exit
    else:
        errormsg = 'Could not understand what type you want to check: should be either a string or a type, not "%s"' % objtype
        raise Exception(errormsg)

    # Do first-round checking
    result = isinstance(obj, objinstance)

    # Do second round checking
    if result and objtype in ['listlike', 'arraylike']: # Special case for handling arrays which may be multi-dimensional
        obj = promotetoarray(obj).flatten() # Flatten all elements
        if objtype == 'arraylike' and subtype is None: subtype = 'number'
    if isiterable(obj) and subtype is not None:
        for item in obj:
            result = result and checktype(item, subtype)

    # Decide what to do with the information thus gleaned
    if die: # Either raise an exception or do nothing if die is True
        if not result: # It's not an instance
            errormsg = 'Incorrect type: object is %s, but %s is required' % (type(obj), objtype)
            raise TypeError(errormsg)
        else:
            return None # It's fine, do nothing
    else: # Return the result of the comparison
        return result


def isnumber(obj, isnan=None):
    '''
    Determine whether or not the input is a number.

    Almost identical to isinstance(obj, numbers.Number).
    '''
    output = checktype(obj, 'number')
    if output and isnan is not None: # It is a number, so can check for nan
        output = (np.isnan(obj) == isnan) # See if they match
    return output


def isstring(obj):
    '''
    Determine whether or not the input is a string (i.e., str or bytes).

    Equivalent to isinstance(obj, (str, bytes))
    '''
    return checktype(obj, 'string')


def isarray(obj, dtype=None):
    '''
    Check whether something is a Numpy array, and optionally check the dtype.

    Almost the same as ``isinstance(obj, np.ndarray)``.

    **Example**::

        sc.isarray(np.array([1,2,3]), dtype=float) # False, dtype is int

    New in version 1.0.0.
    '''
    if isinstance(obj, np.ndarray):
        if dtype is None:
            return True
        else:
            if obj.dtype == dtype:
                return True
            else:
                return False


def promotetoarray(x, skipnone=False):
    '''
    Small function to ensure consistent format for things that should be arrays

    **Examples**::

        sc.promotetoarray(5) # Returns array([5])
        sc.promotetoarray([3,5]) # Returns array([3,5])
        sc.promotetoarray(None, skipnone=True) # Returns array([])
    '''
    if x is None and skipnone:
        return np.array([])
    elif isnumber(x):
        return np.array([x]) # e.g. 3
    elif isinstance(x, (list, tuple)):
        return np.array(x) # e.g. [3]
    elif isinstance(x, np.ndarray):
        if np.shape(x):
            return x # e.g. array([3])
        else:
            return np.array([x]) # e.g. array(3)
    else: # e.g. 'foo'
        errormsg = f'Expecting a number/list/tuple/ndarray; got type {type(x)}: "{x}"'
        raise TypeError(errormsg)
    return # Should be unreachable


def promotetolist(obj=None, objtype=None, keepnone=False):
    '''
    Make sure object is always a list.

    Used so functions can handle inputs like ``'a'``  or ``['a', 'b']``. In other
    words, if an argument can either be a single thing (e.g., a single dict key)
    or a list (e.g., a list of dict keys), this function can be used to do the
    conversion, so it's always safe to iterate over the output.

    Args:
        obj (anything): object to ensure is a list
        objtype (anything): optional type to check for each element
        keepnone (bool): if ``keepnone`` is false, then ``None`` is converted to ``[]``; else, it's converted to ``[None]``

    **Examples**::

        sc.promotetolist(5) # Returns [5]
        sc.promotetolist(np.array([3,5])) # Returns [np.array([3,5])] -- not [3,5]!
        sc.promotetoarray(None) # Returns []

        def myfunc(data, keys):
            keys = sc.promotetolist(keys)
            for key in keys:
                print(data[key])

        data = {'a':[1,2,3], 'b':[4,5,6]}
        myfunc(data, keys=['a', 'b']) # Works
        myfunc(data, keys='a') # Still works, equivalent to needing to supply keys=['a'] without promotetolist()

    Version: 2019nov10
    '''
    if objtype is None: # Don't do type checking
        if isinstance(obj, list):
            output = obj # If it's already a list and we're not doing type checking, just return
        elif obj is None:
            if keepnone:
                output = [None] # Wrap in a list
            else:
                output = [] # Return an empty list, the "none" equivalent for a list
        else:
            output = [obj] # Main usage case -- listify it
    else: # Do type checking
        if checktype(obj=obj, objtype=objtype, die=False):
            output = [obj] # If the object is already of the right type, wrap it in a list
        else:
            try:
                if not isiterable(obj): # Ensure it's iterable -- a mini promote-to-list
                    iterable_obj = [obj]
                else:
                    iterable_obj = obj
                for item in iterable_obj:
                    checktype(obj=item, objtype=objtype, die=True)
                output = list(iterable_obj) # If all type checking passes, cast to list instead of wrapping
            except TypeError as E:
                errormsg = 'promotetolist() type mismatch: %s' % str(E)
                raise TypeError(errormsg).with_traceback(E.__traceback__)
    return output


def mergedicts(*args, strict=False, overwrite=True):
    '''
    Small function to merge multiple dicts together. By default, skips things
    that are not, dicts (e.g., None), and allows keys to be set multiple times.
    Similar to dict.update(), except returns a value. The first dictionary supplied
    will be used for the output type (e.g. if the first dictionary is an odict,
    an odict will be returned).

    Useful for cases, e.g. function arguments, where the default option is ``None``
    but you will need a dict later on.

    Args:
        strict (bool): if True, raise an exception if an argument isn't a dict
        overwrite (bool): if False, raise an exception if multiple keys are found
        *args (dict): the sequence of dicts to be merged

    **Examples**::

        d0 = sc.mergedicts(user_args) # Useful if user_args might be None, but d0 is always a dict
        d1 = sc.mergedicts({'a':1}, {'b':2}) # Returns {'a':1, 'b':2}
        d2 = sc.mergedicts({'a':1, 'b':2}, {'b':3, 'c':4}) # Returns {'a':1, 'b':3, 'c':4}
        d3 = sc.mergedicts(sc.odict({'b':3, 'c':4}), {'a':1, 'b':2}) # Returns sc.odict({'b':2, 'c':4, 'a':1})
        d4 = sc.mergedicts({'b':3, 'c':4}, {'a':1, 'b':2}, overwrite=False) # Raises exception
    '''
    # Try to get the output type from the first argument, but revert to a standard dict if that fails
    try:
        assert isinstance(args[0], dict)
        outputdict = args[0].__class__() # This creates a new instance of the class
    except:
        outputdict = {}

    # Merge over the dictionaries in order
    for arg in args:
        is_dict = isinstance(arg, dict)
        if strict and not is_dict:
            errormsg = f'Argument of "{type(arg)}" found; must be dict since strict=True'
            raise TypeError(errormsg)
        if is_dict:
            if not overwrite:
                intersection = set(outputdict.keys()).intersection(arg.keys())
                if len(intersection):
                    keys = ', '.join(intersection)
                    errormsg = f'Could not merge dicts since keys "{keys}" overlap and overwrite=False'
                    raise KeyError(errormsg)
            outputdict.update(arg)
    return outputdict



##############################################################################
### Time/date functions
##############################################################################

__all__ += ['now', 'getdate', 'readdate', 'date', 'day', 'daydiff', 'daterange', 'datetoyear',
            'elapsedtimestr', 'tic', 'toc', 'toctic', 'timedsleep']

def now(timezone=None, utc=False, die=False, astype='dateobj', tostring=False, dateformat=None):
    '''
    Get the current time, optionally in UTC time.

    **Examples**::

        sc.now() # Return current local time, e.g. 2019-03-14 15:09:26
        sc.now('US/Pacific') # Return the time now in a specific timezone
        sc.now(utc=True) # Return the time in UTC
        sc.now(astype='str') # Return the current time as a string instead of a date object
        sc.now(tostring=True) # Backwards-compatible alias for astype='str'
        sc.now(dateformat='%Y-%b-%d') # Return a different date format
    '''
    if isinstance(utc, str): timezone = utc # Assume it's a timezone
    if timezone is not None: tzinfo = dateutil.tz.gettz(timezone) # Timezone is a string
    elif utc:                tzinfo = dateutil.tz.tzutc() # UTC has been specified
    else:                    tzinfo = None # Otherwise, do nothing
    if tostring: astype = 'str'
    timenow = dt.datetime.now(tzinfo)
    output = getdate(timenow, astype=astype, dateformat=dateformat)
    return output



def getdate(obj=None, astype='str', dateformat=None):
        '''
        Alias for converting a date object to a formatted string.

        **Examples**::

            sc.getdate() # Returns a string for the current date
            sc.getdate(sc.now(), astype='int') # Convert today's time to an integer
        '''
        if obj is None:
            obj = now()

        if dateformat is None:
            dateformat = '%Y-%b-%d %H:%M:%S'
        else:
            astype = 'str' # If dateformat is specified, assume type is a string

        try:
            if isstring(obj): return obj # Return directly if it's a string
            obj.timetuple() # Try something that will only work if it's a date object
            dateobj = obj # Test passed: it's a date object
        except Exception as E: # It's not a date object
            raise Exception('Getting date failed; date must be a string or a date object: %s' % repr(E))

        if astype=='str':
            output = dateobj.strftime(dateformat)
            return output
        elif astype=='int':
            output = time.mktime(dateobj.timetuple()) # So ugly!! But it works -- return integer representation of time
            return output
        elif astype=='dateobj':
            return dateobj
        else:
            errormsg = '"astype=%s" not understood; must be "str" or "int"' % astype
            raise Exception(errormsg)
        return None # Should not be possible to get to this point


def readdate(datestr=None, *args, dateformat=None, return_defaults=False):
    '''
    Convenience function for loading a date from a string. If dateformat is None,
    this function tries a list of standard date types.

    Args:
        datestr (int, float, str or list): the string containing the date, or the timestamp (in seconds), or a list of either
        args (list): additional dates to convert
        dateformat (str or list): the format for the date, if known; can be a list of options
        return_defaults (bool): don't convert the date, just return the defaults

    Returns:
        dateobj (date): a datetime object

    **Examples**::

        dateobj = sc.readdate('2020-03-03') # Standard format, so works
        dateobj = sc.readdate(1611661666) # Can read timestamps as well
        dateobjs = sc.readdate(['2020-06', '2020-07'], dateformat='%Y-%m') # Can read custom date formats
        dateobjs = sc.readdate('20200321', 1611661666) # Can mix and match formats
    '''

    formats_to_try = {
        'date':           '%Y-%m-%d', # 2020-03-21
        'date-alpha':     '%Y-%b-%d', # 2020-Mar-21
        'date-numeric':   '%Y%m%d',   # 20200321
        'datetime':       '%Y-%m-%d %H:%M:%S',    # 2020-03-21 14:35:21
        'datetime-alpha': '%Y-%b-%d %H:%M:%S',    # 2020-Mar-21 14:35:21
        'default':        '%Y-%m-%d %H:%M:%S.%f', # 2020-03-21 14:35:21.23483
        'ctime':          '%a %b %d %H:%M:%S %Y', # Sat Mar 21 23:09:29 2020
        }

    # To get the available formats
    if return_defaults:
        return formats_to_try

    # Handle date formats
    if isstring(dateformat):
        format_list = promotetolist(dateformat)
        formats_to_try = {}
        for f,fmt in enumerate(format_list):
            formats_to_try[f'User supplied {f}'] = fmt

    # Actually process the dates
    datestrs = promotetolist(datestr)
    datestrs.extend(args)
    dateobjs = []
    for datestr in datestrs: # Iterate over them
        dateobj = None
        exceptions = {}
        if isinstance(datestr, dt.datetime):
            dateobj = datestr # Nothing to do
        elif isnumber(datestr):
            dateobj = dt.datetime.fromtimestamp(datestr)
        else:
            for key,fmt in formats_to_try.items():
                try:
                    dateobj = dt.datetime.strptime(datestr, fmt)
                    break # If we find one that works, we can stop
                except Exception as E:
                    exceptions[key] = str(E)
            if dateobj is None:
                formatstr = '\n'.join([f'Format "{item[1]}" raised exception: {exceptions[item[0]]}' for item in formats_to_try.items()])
                errormsg = f'Was unable to convert "{datestr}" to a date using the formats:\n{formatstr}'
                raise ValueError(errormsg)
        dateobjs.append(dateobj)

    # If only a single date was supplied, return just that; else return the list
    if not isinstance(datestr, list) and not len(args):
        return dateobjs[0]
    else:
        return dateobjs


def date(obj, *args, start_date=None, dateformat=None, as_date=True):
    '''
    Convert a string or a datetime object to a date object. To convert to an integer
    from the start day, it is recommended you supply a start date, or use sc.date()
    instead; otherwise, it will calculate the date counting days from 2020-01-01.
    This means that the output of sc.date() will not necessarily match the output
    of sc.date() for an integer input.

    Args:
        obj (str, int, float, date, datetime, list, array): the object to convert
        args (str, int, float, date, datetime): additional objects to convert
        start_date (str, date, datetime): the starting date, if an integer is supplied
        dateformat (str): the format to return the date in
        as_date (bool): whether to return as a datetime date instead of a string

    Returns:
        dates (date or list): either a single date object, or a list of them

    **Examples**::

        sc.date('2020-04-05') # Returns datetime.date(2020, 4, 5)
        sc.date('2020-04-14', start_date='2020-04-04', as_date=False) # Returns 10
        sc.date([35,36,37], as_date=False) # Returns ['2020-02-05', '2020-02-06', '2020-02-07']

    New in version 1.0.0.
    '''

    if obj is None:
        return None

    # Convert to list and handle other inputs
    if isinstance(obj, np.ndarray):
        obj = obj.tolist() # If it's an array, convert to a list
    obj = promotetolist(obj) # Ensure it's iterable
    obj.extend(args)
    if dateformat is None:
        dateformat = '%Y-%m-%d'
    if start_date is None:
        start_date = '2020-01-01'

    dates = []
    for d in obj:
        if d is None:
            dates.append(d)
            continue
        try:
            if type(d) == dt.date: # Do not use isinstance, since must be the exact type
                pass
            elif isstring(d) or isnumber(d):
                d = readdate(d).date()
            elif isinstance(d, dt.datetime):
                d = d.date()
            elif isnumber(d):
                if start_date is None:
                    errormsg = f'To convert the number {d} to a date, you must supply start_date'
                    raise ValueError(errormsg)
                d = date(start_date) + dt.timedelta(days=int(d))
            else:
                errormsg = f'Cannot interpret {type(d)} as a date, must be date, datetime, or string'
                raise TypeError(errormsg)
            if as_date:
                dates.append(d)
            else:
                dates.append(d.strftime(dateformat))
        except Exception as E:
            errormsg = f'Conversion of "{d}" to a date failed: {str(E)}'
            raise ValueError(errormsg)

    # Return an integer rather than a list if only one provided
    if len(dates)==1:
        dates = dates[0]

    return dates


def day(obj, *args, start_day=None):
    '''
    Convert a string, date/datetime object, or int to a day (int), the number of
    days since the start day. See also sc.date() and sc.daydiff().

    Args:
        obj (str, date, int, or list): convert any of these objects to a day relative to the start day
        args (list): additional days
        start_day (str or date): the start day; if none is supplied, return days since (supplied year)-01-01.

    Returns:
        days (int or str): the day(s) in simulation time

    **Example**::

        sc.day(sc.now()) # Returns how many days into the year we are

    New in version 1.0.0.
    '''

    # Do not process a day if it's not supplied
    if obj is None:
        return None

    # Convert to list
    if isstring(obj) or isnumber(obj) or isinstance(obj, (dt.date, dt.datetime)):
        obj = promotetolist(obj) # Ensure it's iterable
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist() # Convert to list if it's an array
    obj.extend(args)

    days = []
    for d in obj:
        if d is None:
            days.append(d)
        elif isnumber(d):
            days.append(int(d)) # Just convert to an integer
        else:
            try:
                if isstring(d):
                    d = readdate(d).date()
                elif isinstance(d, dt.datetime):
                    d = d.date()
                if start_day:
                    start_date = date(start_day)
                else:
                    start_date = date(f'{d.year}-01-01')
                d_day = (d - start_date).days # Heavy lifting -- actually compute the day
                days.append(d_day)
            except Exception as E:
                errormsg = f'Could not interpret "{d}" as a date: {str(E)}'
                raise ValueError(errormsg)

    # Return an integer rather than a list if only one provided
    if len(days)==1:
        days = days[0]

    return days


def daydiff(*args):
    '''
    Convenience function to find the difference between two or more days. With
    only one argument, calculate days since 2020-01-01.

    **Examples**::

        diff  = sc.daydiff('2020-03-20', '2020-04-05') # Returns 16
        diffs = sc.daydiff('2020-03-20', '2020-04-05', '2020-05-01') # Returns [16, 26]

    New in version 1.0.0.
    '''
    days = [date(day) for day in args]
    if len(days) == 1:
        days.insert(0, date(f'{now().year}-01-01')) # With one date, return days since Jan. 1st

    output = []
    for i in range(len(days)-1):
        diff = (days[i+1] - days[i]).days
        output.append(diff)

    if len(output) == 1:
        output = output[0]

    return output


def daterange(start_date, end_date, inclusive=True, as_date=False, dateformat=None):
    '''
    Return a list of dates from the start date to the end date. To convert a list
    of days (as integers) to dates, use sc.date() instead.

    Args:
        start_date (int/str/date): the starting date, in any format
        end_date (int/str/date): the end date, in any format
        inclusive (bool): if True (default), return to end_date inclusive; otherwise, stop the day before
        as_date (bool): if True, return a list of datetime.date objects instead of strings
        dateformat (str): passed to date()

    **Example**::

        dates = sc.daterange('2020-03-01', '2020-04-04')

    New in version 1.0.0.
    '''
    start_day = day(start_date)
    end_day = day(end_date)
    if inclusive:
        end_day += 1
    days = np.arange(start_day, end_day)
    dates = date(days, as_date=as_date, dateformat=dateformat)
    return dates


def datetoyear(dateobj, dateformat=None):
    """
    Convert a DateTime instance to decimal year.

    Args:
        dateobj (date, str):  The datetime instance to convert
        dateformat (str): If dateobj is a string, the optional date conversion format to use

    Returns:
        Equivalent decimal year

    **Example**::

        sc.datetoyear('2010-07-01') # Returns approximately 2010.5

    By Luke Davis from https://stackoverflow.com/a/42424261, adapted by Romesh Abeysuriya.

    New in version 1.0.0.
    """
    if isstring(dateobj):
        dateobj = readdate(dateobj, dateformat=dateformat)
    year_part = dateobj - dt.datetime(year=dateobj.year, month=1, day=1)
    year_length = dt.datetime(year=dateobj.year + 1, month=1, day=1) - dt.datetime(year=dateobj.year, month=1, day=1)
    return dateobj.year + year_part / year_length


def elapsedtimestr(pasttime, maxdays=5, shortmonths=True):
    """
    Accepts a datetime object or a string in ISO 8601 format and returns a
    human-readable string explaining when this time was.

    The rules are as follows:

    * If a time is within the last hour, return 'XX minutes'
    * If a time is within the last 24 hours, return 'XX hours'
    * If within the last 5 days, return 'XX days'
    * If in the same year, print the date without the year
    * If in a different year, print the date with the whole year

    These can be configured as options.
    """

    # Elapsed time function by Alex Chan
    # https://gist.github.com/alexwlchan/73933442112f5ae431cc
    def print_date(date, includeyear=True, shortmonths=True):
        """Prints a datetime object as a full date, stripping off any leading
        zeroes from the day (strftime() gives the day of the month as a zero-padded
        decimal number).
        """
        # %b/%B are the tokens for abbreviated/full names of months to strftime()
        if shortmonths:
            month_token = '%b'
        else:
            month_token = '%B'

        # Get a string from strftime()
        if includeyear:
            date_str = date.strftime('%d ' + month_token + ' %Y')
        else:
            date_str = date.strftime('%d ' + month_token)

        # There will only ever be at most one leading zero, so check for this and
        # remove if necessary
        if date_str[0] == '0':
            date_str = date_str[1:]

        return date_str
    now_time = dt.datetime.now()

    # If the user passes in a string, try to turn it into a datetime object before continuing
    if isinstance(pasttime, str):
        try:
            pasttime = dt.datetime.strptime(pasttime, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            raise ValueError("User supplied string %s is not in ISO 8601 "
                             "format." % pasttime)
    elif isinstance(pasttime, dt.datetime):
        pass
    else:
        raise ValueError("User-supplied value %s is neither a datetime object "
                         "nor an ISO 8601 string." % pasttime)

    # It doesn't make sense to measure time elapsed between now and a future date, so we'll just print the date
    if pasttime > now_time:
        includeyear = (pasttime.year != now_time.year)
        time_str = print_date(pasttime, includeyear=includeyear, shortmonths=shortmonths)

    # Otherwise, start by getting the elapsed time as a datetime object
    else:
        elapsed_time = now_time - pasttime

        # Check if the time is within the last minute
        if elapsed_time < dt.timedelta(seconds=60):
            if elapsed_time.seconds <= 10:
                time_str = "just now"
            else:
                time_str = "%d secs ago" % elapsed_time.seconds

        # Check if the time is within the last hour
        elif elapsed_time < dt.timedelta(seconds=60 * 60):

            # We know that seconds > 60, so we can safely round down
            minutes = int(elapsed_time.seconds / 60)
            if minutes == 1:
                time_str = "a minute ago"
            else:
                time_str = "%d mins ago" % minutes

        # Check if the time is within the last day
        elif elapsed_time < dt.timedelta(seconds=60 * 60 * 24 - 1):

            # We know that it's at least an hour, so we can safely round down
            hours = int(elapsed_time.seconds / (60 * 60))
            if hours == 1:
                time_str = "1 hour ago"
            else:
                time_str = "%d hours ago" % hours

        # Check if it's within the last N days, where N is a user-supplied argument
        elif elapsed_time < dt.timedelta(days=maxdays):
            if elapsed_time.days == 1:
                time_str = "yesterday"
            else:
                time_str = "%d days ago" % elapsed_time.days

        # If it's not within the last N days, then we're just going to print the date
        else:
            includeyear = (pasttime.year != now_time.year)
            time_str = print_date(pasttime, includeyear=includeyear, shortmonths=shortmonths)

    return time_str



def tic():
    '''
    With toc(), a little pair of functions to calculate a time difference:

    **Examples**::

        sc.tic()
        slow_func()
        sc.toc()

        T = sc.tic()
        slow_func2()
        sc.toc(T, label='slow_func2')
    '''
    global _tictime  # The saved time is stored in this global
    _tictime = time.time()  # Store the present time in the global
    return _tictime    # Return the same stored number



def toc(start=None, output=False, label=None, sigfigs=None, filename=None, reset=False):
    '''
    With tic(), a little pair of functions to calculate a time difference.

    Args:
        start (float): the starting time, as returned by e.g. sc.tic()
        output (bool): whether to return the output (otherwise print)
        label (str): optional label to add
        sigfigs (int): number of significant figures for time estimate
        filename (str): log file to write results to
        reset (bool): reset the time; like calling sc.toctic() or sc.tic() again

    **Examples**::

        sc.tic()
        slow_func()
        sc.toc()

        T = sc.tic()
        slow_func2()
        sc.toc(T, label='slow_func2')
    '''
    global _tictime  # The saved time is stored in this global

    # Set defaults
    if label   is None: label = ''
    if sigfigs is None: sigfigs = 3

    # If no start value is passed in, try to grab the global _tictime.
    if start is None:
        try:    start = _tictime
        except: start = 0 # This doesn't exist, so just leave start at 0.

    # Get the elapsed time in seconds.
    elapsed = time.time() - start

    # Create the message giving the elapsed time.
    if label=='': base = 'Elapsed time: '
    else:         base = 'Elapsed time for %s: ' % label
    logmessage = base + '%s s' % sigfig(elapsed, sigfigs=sigfigs)

    # Optionally reset the counter
    if reset:
        _tictime = time.time()  # Store the present time in the global

    if output:
        return elapsed
    else:
        if filename is not None: printtologfile(logmessage, filename) # If we passed in a filename, append the message to that file.
        else: print(logmessage) # Otherwise, print the message.
        return None


def toctic(returntic=False, returntoc=False, *args, **kwargs):
    '''
    A convenience function for multiple timings. Can return the default output of
    either tic() or toc() (default neither). Arguments are passed to toc(). Equivalent
    to sc.toc(reset=True).

    **Example**::

        sc.tic()
        slow_operation_1()
        sc.toctic()
        slow_operation_2()
        sc.toc()

    New in version 1.0.0.
    '''
    tocout = toc(*args, **kwargs)
    ticout = tic()
    if   returntic: return ticout
    elif returntoc: return tocout
    else:           return None


def timedsleep(delay=None, verbose=True):
    '''
    Delay for a certain amount of time, to ensure accurate timing.

    **Example**::

        for i in range(10):
            sc.timedsleep('start') # Initialize
            for j in range(int(1e6)):
                tmp = pl.rand()
            sc.timedsleep(1) # Wait for one second including computation time
    '''
    global _delaytime
    if delay is None or delay=='start':
        _delaytime = time.time()  # Store the present time in the global.
        return _delaytime         # Return the same stored number.
    else:
        try:    start = _delaytime
        except: start = time.time()
        elapsed = time.time() - start
        remaining = delay-elapsed
        if remaining>0:
            if verbose:
                print('Pausing for %0.1f s' % remaining)
            time.sleep(remaining)
        else:
            if verbose:
                print('Warning, delay less than elapsed time (%0.1f vs. %0.1f)' % (delay, elapsed))
    return None



##############################################################################
### Misc. functions
##############################################################################

__all__ += ['checkmem', 'checkram', 'runcommand', 'gitinfo', 'compareversions',
            'uniquename', 'importbyname', 'suggest', 'profile', 'mprofile', 'getcaller']


def checkmem(var, descend=None, alphabetical=False, plot=False, verbose=False):
    '''
    Checks how much memory the variable or variables in question use by dumping
    them to file. See also checkram().

    Args:
        var (any): the variable being checked
        descend (bool): whether or not to descend one level into the object
        alphabetical (bool): if descending into a dict or object, whether to list items by name rather than size
        plot (bool): if descending, show the results as a pie chart
        verbose (bool or int): detail to print, if >1, print repr of objects along the way

    **Example**::

        import sciris as sc
        sc.checkmem(['spiffy',rand(2483,589)], descend=True)
    '''
    from .sc_fileio import saveobj # Here to avoid recursion

    def check_one_object(variable):
        ''' Check the size of one variable '''

        if verbose>1:
            print(f'  Checking size of {variable}...')

        # Create a temporary file, save the object, check the size, remove it
        filename = tempfile.mktemp()
        saveobj(filename, variable)
        filesize = os.path.getsize(filename)
        os.remove(filename)

        # Convert to string
        factor = 1
        label = 'B'
        labels = ['KB','MB','GB']
        for i,f in enumerate([3,6,9]):
            if filesize>10**f:
                factor = 10**f
                label = labels[i]
        humansize = float(filesize/float(factor))
        sizestr = f'{humansize:0.3f} {label}'
        return filesize, sizestr

    # Initialize
    varnames  = []
    variables = []
    sizes     = []
    sizestrs  = []

    # Create the object(s) to check the size(s) of
    varnames = [''] # Set defaults
    variables = [var]
    if descend or descend is None:
        if hasattr(var, '__dict__'): # It's an object
            if verbose>1: print('Iterating over object')
            varnames = sorted(list(var.__dict__.keys()))
            variables = [getattr(var, attr) for attr in varnames]
        elif np.iterable(var): # Handle dicts and lists
            if isinstance(var, dict): # Handle dicts
                if verbose>1: print('Iterating over dict')
                varnames = list(var.keys())
                variables = var.values()
            else: # Handle lists and other things
                if verbose>1: print('Iterating over list')
                varnames = [f'item {i}' for i in range(len(var))]
                variables = var
        else:
            if descend: # Could also be None
                print('Object is not iterable: cannot descend') # Print warning and use default

    # Compute the sizes
    for v,variable in enumerate(variables):
        if verbose:
            print(f'Processing variable {v} of {len(variables)}')
        filesize, sizestr = check_one_object(variable)
        sizes.append(filesize)
        sizestrs.append(sizestr)

    if alphabetical:
        inds = np.argsort(varnames)
    else:
        inds = np.argsort(sizes)[::-1]

    for i in inds:
        varstr = f'Variable "{varnames[i]}"' if varnames[i] else 'Variable'
        print(f'{varstr} is {sizestrs[i]}')

    if plot==True:
        import pylab as pl # Optional import
        pl.axes(aspect=1)
        pl.pie(pl.array(sizes)[inds], labels=pl.array(varnames)[inds], autopct='%0.2f')

    return None


def checkram(unit='mb', fmt='0.2f', start=0, to_string=True):
    '''
    Unlike checkmem(), checkram() looks at actual memory usage, typically at different
    points throughout execution.

    **Example**::

        import sciris as sc
        import numpy as np
        start = sc.checkram(to_string=False)
        a = np.random.random((1_000, 10_000))
        print(sc.checkram(start=start))

    New in version 1.0.0.
    '''
    process = psutil.Process(os.getpid())
    mapping = {'b':1, 'kb':1e3, 'mb':1e6, 'gb':1e9}
    try:
        factor = mapping[unit.lower()]
    except KeyError:
        raise KeyNotFoundError(f'Unit {unit} not found')
    mem_use = process.memory_info().rss/factor - start
    if to_string:
        output = f'{mem_use:{fmt}} {unit.upper()}'
    else:
        output = mem_use
    return output


def runcommand(command, printinput=False, printoutput=False, wait=True):
    '''
    Make it easier to run shell commands.

    Alias to ``subprocess.Popen()``.

    **Examples**::

        myfiles = sc.runcommand('ls').split('\\n') # Get a list of files in the current folder
        sc.runcommand('sshpass -f %s scp myfile.txt me@myserver:myfile.txt' % 'pa55w0rd', printinput=True, printoutput=True) # Copy a file remotely
        sc.runcommand('sleep 600; mkdir foo', wait=False) # Waits 10 min, then creates the folder "foo", but the function returns immediately

    Date: 2019sep04
    '''
    if printinput:
        print(command)
    try:
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if wait: # Whether to run in the background
            stderr = p.stdout.read().decode("utf-8") # Somewhat confusingly, send stderr to stdout
            stdout = p.communicate()[0].decode("utf-8") # ...and then stdout to the pipe
            output = stdout + '\n' + stderr if stderr else stdout # Only include the error if it was non-empty
        else:
            output = ''
    except Exception as E:
        output = 'runcommand(): shell command failed: %s' % str(E) # This is for a Python error, not a shell error -- those get passed to output
    if printoutput:
        print(output)
    return output



def gitinfo(path=None, hashlen=7, die=False, verbose=True):
    """
    Retrieve git info

    This function reads git branch and commit information from a .git directory.
    Given a path, it will check for a ``.git`` directory. If the path doesn't contain
    that directory, it will search parent directories for ``.git`` until it finds one.
    Then, the current information will be parsed.

    Args:
        path (str): A folder either containing a .git directory, or with a parent that contains a .git directory
        hashlen (int): Length of hash to return (default: 7)
        die (bool): whether to raise an exception if git information can't be retrieved (default: no)
        verbose (bool): if not dying, whether to print information about the exception

    Returns:
        Dictionary containing the branch, hash, and commit date

    **Examples**::

        info = sc.gitinfo() # Get git info for current script repository
        info = sc.gitinfo(my_package.__file__) # Get git info for a particular Python package
    """

    if path is None:
        path = os.getcwd()

    gitbranch = "Branch N/A"
    githash   = "Hash N/A"
    gitdate   = "Date N/A"

    try:
        # First, get the .git directory
        curpath = os.path.dirname(os.path.abspath(path))
        while curpath:
            if os.path.exists(os.path.join(curpath, ".git")):
                gitdir = os.path.join(curpath, ".git")
                break
            else:
                parent, _ = os.path.split(curpath)
                if parent == curpath:
                    curpath = None
                else:
                    curpath = parent
        else:
            raise Exception("Could not find .git directory")

        # Then, get the branch and commit
        with open(os.path.join(gitdir, "HEAD"), "r") as f1:
            ref = f1.read()
            if ref.startswith("ref:"):
                refdir = ref.split(" ")[1].strip()  # The path to the file with the commit
                gitbranch = refdir.replace("refs/heads/", "")  # / is always used (not os.sep)
                with open(os.path.join(gitdir, refdir), "r") as f2:
                    githash = f2.read().strip()  # The hash of the commit
            else:
                gitbranch = "Detached head (no branch)"
                githash = ref.strip()

        # Now read the time from the commit
        compressed_contents = open(os.path.join(gitdir, "objects", githash[0:2], githash[2:]), "rb").read()
        decompressed_contents = zlib.decompress(compressed_contents).decode()
        for line in decompressed_contents.split("\n"):
            if line.startswith("author"):
                _re_actor_epoch = re.compile(r"^.+? (.*) (\d+) ([+-]\d+).*$")
                m = _re_actor_epoch.search(line)
                actor, epoch, offset = m.groups()
                t = time.gmtime(int(epoch))
                gitdate = time.strftime("%Y-%m-%d %H:%M:%S UTC", t)

    except Exception as E:
        errormsg = 'Could not extract git info; please check paths'
        if die:
            raise Exception(errormsg) from E
        elif verbose:
            print(errormsg + f'\nError: {str(E)}')

    # Trim the hash, but not if loading failed
    if len(githash)>hashlen and 'N/A' not in githash:
        githash = githash[:hashlen]

    # Assemble output
    output = {"branch": gitbranch, "hash": githash, "date": gitdate}

    return output


def compareversions(version1, version2):
    '''
    Function to compare versions, expecting both arguments to be a string of the
    format 1.2.3, but numeric works too.

    **Examples**::

        sc.compareversions('1.2.3', '2.3.4') # returns -1
        sc.compareversions(2, '2') # returns 0
        sc.compareversions('3.1', '2.99') # returns 1
    '''
    if LooseVersion(str(version1)) > LooseVersion(str(version2)):
        return 1
    elif LooseVersion(str(version1)) == LooseVersion(str(version2)):
        return 0
    else:
        return -1


def uniquename(name=None, namelist=None, style=None):
    """
    Given a name and a list of other names, find a replacement to the name
    that doesn't conflict with the other names, and pass it back.

    **Example**::

        name = sc.uniquename(name='file', namelist=['file', 'file (1)', 'file (2)'])
    """
    if style is None: style = ' (%d)'
    namelist = promotetolist(namelist)
    unique_name = str(name) # Start with the passed in name.
    i = 0 # Reset the counter
    while unique_name in namelist: # Try adding an index (i) to the name until we find one that's unique
        i += 1
        unique_name = str(name) + style%i
    return unique_name # Return the found name.


def importbyname(name=None, output=False, die=True):
    '''
    A little function to try loading optional imports.

    **Example**::

        np = sc.importbyname('numpy')
    '''
    import importlib
    try:
        module = importlib.import_module(name)
        globals()[name] = module
    except Exception as E:
        errormsg = f'Cannot use "{name}" since {name} is not installed.\nPlease install {name} and try again.'
        print(errormsg)
        if die: raise E
        else:   return False
    if output: return module
    else:      return True


def suggest(user_input, valid_inputs, n=1, threshold=4, fulloutput=False, die=False, which='damerau'):
    """
    Return suggested item

    Returns item with lowest Levenshtein distance, where case substitution and stripping
    whitespace are not included in the distance. If there are ties, then the additional operations
    will be included.

    Args:
        user_input (str): User's input
        valid_inputs (list): List/collection of valid strings
        n (int): Maximum number of suggestions to return
        threshold (int): Maximum number of edits required for an option to be suggested
        die (bool): If True, an informative error will be raised (to avoid having to implement this in the calling code)
        which (str): Distance calculation method used; options are "damerau" (default), "levenshtein", or "jaro"

    Returns:
        suggestions (str or list): Suggested string. Returns None if no suggestions with edit distance less than threshold were found. This helps to make
             suggestions more relevant.

    **Examples**::

        >>> suggest('foo',['Foo','Bar'])
        'Foo'
        >>> suggest('foo',['FOO','Foo'])
        'Foo'
        >>> suggest('foo',['Foo ','boo'])
        'Foo '
    """
    try:
        import jellyfish # To allow as an optional import
    except ModuleNotFoundError as e:
        raise Exception('The "jellyfish" Python package is not available; please install via "pip install jellyfish"') from e

    valid_inputs = promotetolist(valid_inputs, objtype='string')

    mapping = {
        'damerau':     jellyfish.damerau_levenshtein_distance,
        'levenshtein': jellyfish.levenshtein_distance,
        'jaro':        jellyfish.jaro_distance,
        }

    keys = list(mapping.keys())
    if which not in keys:
        available = ', '.join(keys)
        errormsg = f'Method {which} not available; options are {available}'
        raise NotImplementedError(errormsg)

    dist_func = mapping[which]

    distance = np.zeros(len(valid_inputs))
    cs_distance = np.zeros(len(valid_inputs))
    # We will switch inputs to lowercase because we want to consider case substitution a 'free' operation
    # Similarly, stripping whitespace is a free operation. This ensures that something like
    # 'foo ' will match 'Foo' ahead of 'boo '
    for i, s in enumerate(valid_inputs):
        distance[i]    = dist_func(user_input, s.strip().lower())
        cs_distance[i] = dist_func(user_input, s.strip())

    # If there is a tie for the minimum distance, use the case sensitive comparison
    if sum(distance==min(distance)) > 1:
        distance = cs_distance

    # Order by distance, then pull out the right inputs, then turn them into a list
    order = np.argsort(distance)
    suggestions = [valid_inputs[i] for i in order]
    suggestionstr = ', '.join(['"' + sugg + '"' for sugg in suggestions[:n]])

    if min(distance) > threshold:
        if die:
            raise Exception('"%s" not found' % user_input)
        else:
            return None
    elif die:
        raise Exception('"%s" not found - did you mean %s?' % (user_input, suggestionstr))
    else:
        if fulloutput:
            output = dict(zip(suggestions, distance[order]))
            return output
        else:
            if n==1:
                return suggestions[0]
            else:
                return suggestions[:n]


def profile(run, follow=None, print_stats=True, *args, **kwargs):
    '''
    Profile the line-by-line time required by a function.

    Args:
        run (function): The function to be run
        follow (function): The function or list of functions to be followed in the profiler; if None, defaults to the run function
        print_stats (bool): whether to print the statistics of the profile to stdout
        args, kwargs: Passed to the function to be run

    Returns:
        LineProfiler (by default, the profile output is also printed to stdout)

    **Example**::

        def slow_fn():
            n = 10000
            int_list = []
            int_dict = {}
            for i in range(n):
                int_list.append(i)
                int_dict[i] = i
            return

        class Foo:
            def __init__(self):
                self.a = 0
                return

            def outer(self):
                for i in range(100):
                    self.inner()
                return

            def inner(self):
                for i in range(1000):
                    self.a += 1
                return

        foo = Foo()
        sc.profile(run=foo.outer, follow=[foo.outer, foo.inner])
        sc.profile(slow_fn)

        # Profile the constructor for Foo
        f = lambda: Foo()
        sc.profile(run=f, follow=[foo.__init__])
    '''

    try:
        from line_profiler import LineProfiler
    except ModuleNotFoundError as e:
        raise Exception('The "line_profiler" Python package is required to perform profiling') from e

    if follow is None:
        follow = run
    orig_func = run

    lp = LineProfiler()
    follow = promotetolist(follow)
    for f in follow:
        lp.add_function(f)
    lp.enable_by_count()
    wrapper = lp(run)

    if print_stats:
        print('Profiling...')
    wrapper(*args, **kwargs)
    run = orig_func
    if print_stats:
        lp.print_stats()
        print('Done.')
    return lp


def mprofile(run, follow=None, show_results=True, *args, **kwargs):
    '''
    Profile the line-by-line memory required by a function. See profile() for a
    usage example.

    Args:
        run (function): The function to be run
        follow (function): The function or list of functions to be followed in the profiler; if None, defaults to the run function
        show_results (bool): whether to print the statistics of the profile to stdout
        args, kwargs: Passed to the function to be run

    Returns:
        LineProfiler (by default, the profile output is also printed to stdout)
    '''

    try:
        import memory_profiler as mp
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('The "memory_profiler" Python package is required to perform profiling') from e

    if follow is None:
        follow = run

    lp = mp.LineProfiler()
    follow = promotetolist(follow)
    for f in follow:
        lp.add_function(f)
    lp.enable_by_count()
    try:
        wrapper = lp(run)
    except TypeError as e:
        raise TypeError('Function wrapping failed; are you profiling an already-profiled function?') from e

    if show_results:
        print('Profiling...')
    wrapper(*args, **kwargs)
    if show_results:
        mp.show_results(lp)
        print('Done.')
    return lp


def getcaller(frame=2, tostring=True):
    '''
    Try to get information on the calling function, but fail gracefully.

    Frame 1 is the current file (this one), so not very useful. Frame 2 is
    the default assuming it is being called directly. Frame 3 is used if
    another function is calling this function internally.

    Args:
        frame (int): how many frames to descend (e.g. the caller of the caller of the...)
        tostring (bool): whether to return a string instead of a dict

    Returns:
        output (str/dict): the filename and line number of the calling function, either as a string or dict

    New in version 1.0.0.
    '''
    try:
        import inspect
        result = inspect.getouterframes(inspect.currentframe(), 2)
        fname = str(result[frame][1])
        lineno = str(result[frame][2])
        if tostring:
            output = f'{fname}, line {lineno}'
        else:
            output = {'filename':fname, 'lineno':lineno}
    except Exception as E:
        if tostring:
            output = f'Calling function information not available ({str(E)})'
        else:
            output = {'filename':'N/A', 'lineno':'N/A'}
    return output



##############################################################################
### Nested dictionary functions
##############################################################################

__all__ += ['getnested', 'setnested', 'makenested', 'iternested', 'mergenested',
            'flattendict', 'search', 'nestedloop']


def makenested(nesteddict, keylist,item=None):
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
    currentlevel = nesteddict
    for i,key in enumerate(keylist[:-1]):
        if not(key in currentlevel):
            currentlevel[key] = {}
        currentlevel = currentlevel[key]
    currentlevel[keylist[-1]] = item
    return


def getnested(nesteddict, keylist, safe=False):
    '''
    Get the value for the given list of keys

    >>> sc.getnested(foo, ['a','b'])

    See makenested() for full documentation.
    '''
    output = reduce(lambda d, k: d.get(k) if d else None if safe else d[k], keylist, nesteddict)
    return output


def setnested(nesteddict, keylist, value):
    '''
    Set the value for the given list of keys

    >>> sc.setnested(foo, ['a','b'], 3)

    See makenested() for full documentation.
    '''
    getnested(nesteddict, keylist[:-1])[keylist[-1]] = value
    return # Modify nesteddict in place


def iternested(nesteddict,previous = []):
    '''
    Return a list of all the twigs in the current dictionary

    >>> twigs = sc.iternested(foo)

    See makenested() for full documentation.
    '''
    output = []
    for k in nesteddict.items():
        if isinstance(k[1],dict):
            output += iternested(k[1],previous+[k[0]]) # Need to add these at the first level
        else:
            output.append(previous+[k[0]])
    return output


def mergenested(dict1, dict2, die=False, verbose=False, _path=None):
    '''
    Merge different nested dictionaries

    See makenested() for full documentation.

    Adapted from https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge
    '''
    if _path is None: _path = []
    if _path:
        a = dict1 # If we're being recursive, work in place
    else:
        a = dcp(dict1) # Otherwise, make a copy
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
                    raise Exception(errormsg)
                else:
                    a[key] = b[key]
                    if verbose:
                        print(errormsg)
        else:
            a[key] = b[key]
    return a


def flattendict(input_dict: dict, sep: str = None, _prefix=None) -> dict:
    """
    Flatten nested dictionary

    **Example**::

        >>> flattendict({'a':{'b':1,'c':{'d':2,'e':3}}})
        {('a', 'b'): 1, ('a', 'c', 'd'): 2, ('a', 'c', 'e'): 3}
        >>> flattendict({'a':{'b':1,'c':{'d':2,'e':3}}}, sep='_')
        {'a_b': 1, 'a_c_d': 2, 'a_c_e': 3}

    Args:
        d: Input dictionary potentially containing dicts as values
        sep: Concatenate keys using string separator. If ``None`` the returned dictionary will have tuples as keys
        _prefix: Internal argument for recursively accumulating the nested keys

    Returns:
        A flat dictionary where no values are dicts
    """
    output_dict = {}
    for k, v in input_dict.items():
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
            output_dict.update(flattendict(input_dict[k], sep=sep, _prefix=k2))
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



##############################################################################
### Classes
##############################################################################

__all__ += ['KeyNotFoundError', 'LinkException', 'prettyobj', 'Link', 'Timer']


class KeyNotFoundError(KeyError):
    '''
    A tiny class to fix repr for KeyErrors. KeyError prints the repr of the error
    message, rather than the actual message, so e.g. newline characters print as
    the character rather than the actual newline.

    **Example**::

        raise sc.KeyNotFoundError('The key "foo" is not available, but these are: "bar", "cat"')
    '''

    def __str__(self):
        return Exception.__str__(self)


class LinkException(Exception):
    '''
    An exception to raise when links are broken, for exclusive use with the Link
    class.
    '''

    def __init(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class prettyobj(object):
    '''
    Use pretty repr for objects, instead of just showing the type and memory pointer
    (the Python default for objects). Can also be used as the base class for custom
    classes.

    **Examples**

        >>> myobj = sc.prettyobj()
        >>> myobj.a = 3
        >>> myobj.b = {'a':6}
        >>> print(myobj)
        <sciris.sc_utils.prettyobj at 0x7ffa1e243910>
        ————————————————————————————————————————————————————————————
        a: 3
        b: {'a': 6}
        ————————————————————————————————————————————————————————————

        >>> class MyObj(sc.prettyobj):
        >>>
        >>>     def __init__(self, a, b):
        >>>         self.a = a
        >>>         self.b = b
        >>>
        >>>     def mult(self):
        >>>         return self.a * self.b
        >>>
        >>> myobj = MyObj(a=4, b=6)
        >>> print(myobj)
        <__main__.MyObj at 0x7fd9acd96c10>
        ————————————————————————————————————————————————————————————
        Methods:
          mult()
        ————————————————————————————————————————————————————————————
        a: 4
        b: 6
        ————————————————————————————————————————————————————————————


    '''

    def __repr__(self):
        output  = prepr(self)
        return output


class Link(object):
    '''
    A class to differentiate between an object and a link to an object. Not very
    useful at the moment, but the idea eventually is that this object would be
    parsed differently from other objects -- most notably, a recursive method
    (such as a pickle) would skip over Link objects, and then would fix them up
    after the other objects had been reinstated.

    Version: 2017jan31
    '''

    def __init__(self, obj=None):
        self.obj = obj # Store the object -- or rather a reference to it, if it's mutable
        try:    self.uid = obj.uid # If the object has a UID, store it separately
        except: self.uid = None # If not, just use None


    def __repr__(self):
        ''' Just use default '''
        output  = prepr(self)
        return output

    def __call__(self, obj=None):
        ''' If called with no argument, return the stored object; if called with argument, update object '''
        if obj is None:
            if type(self.obj)==LinkException: # If the link is broken, raise it now
                raise self.obj
            return self.obj
        else:
            self.__init__(obj)
            return None

    def __copy__(self, *args, **kwargs):
        ''' Do NOT automatically copy link objects!! '''
        return Link(LinkException('Link object copied but not yet repaired'))

    def __deepcopy__(self, *args, **kwargs):
        ''' Same as copy '''
        return self.__copy__(*args, **kwargs)


class Timer(object):
    '''
    Simple timer class

    This wraps `tic` and `toc` with the formatting arguments and
    the start time (at construction)
    Use this in a ``with...as``` block to automatically print
    elapsed time when the block finishes.

    Implementation based on https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/

    Example making repeated calls to the same Timer:

    >>> timer = Timer()
    >>> timer.toc()
    Elapsed time: 2.63 s
    >>> timer.toc()
    Elapsed time: 5.00 s

    Example wrapping code using with-as:

    >>> with Timer(label='mylabel') as t:
    >>>     foo()

    '''

    def __init__(self,**kwargs):
        self.tic()
        self.kwargs = kwargs #: Store kwargs to pass to :func:`toc` at the end of the block

    def __enter__(self):
        '''
        Reset start time when entering with-as block
        '''

        self.tic()
        return self

    def __exit__(self, *args):
        '''
        Print elapsed time when leaving a with-as block
        '''

        self.toc()

    def tic(self):
        '''
        Set start time
        '''

        self.start = tic()

    def toc(self):
        '''
        Print elapsed time
        '''

        toc(self.start,**self.kwargs)
