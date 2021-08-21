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
    - ``sc.mergedicts()``: merges any set of inputs into a dictionary
    - ``sc.readdate()``: convert strings to dates using common formats
    - ``sc.daterange()``: create a list of dates
    - ``sc.datedelta()``: perform calculations on date strings
    - ``sc.tic()/sc.toc()``: simple method for timing durations
    - ``sc.runcommand()``: simple way of executing a shell command
'''

##############################################################################
#%% Imports
##############################################################################

import os
import re
import sys
import copy
import time
import json
import zlib
import types
import psutil
import pprint
import hashlib
import dateutil
import subprocess
import itertools
import numbers
import string
import tempfile
import warnings
import numpy as np
import pylab as pl
import random as rnd
import datetime as dt
import uuid as py_uuid
import pkg_resources as pkgr
import traceback as py_traceback
from textwrap import fill
from functools import reduce
from collections import OrderedDict as OD
from distutils.version import LooseVersion

# Handle types
_stringtypes = (str, bytes)
_numtype    = numbers.Number

# Add Windows support for colors (do this at the module level so that colorama.init() only gets called once)
if 'win' in sys.platform and sys.platform != 'darwin': # pragma: no cover # NB: can't use startswith() because of 'cygwin'
    try:
        import colorama
        colorama.init()
        ansi_support = True
    except:
        ansi_support = False  # print('Warning: you have called colorize() on Windows but do not have either the colorama or tendo modules.')
else:
    ansi_support = True



##############################################################################
#%% Adaptations from other libraries
##############################################################################

# Define the modules being loaded
__all__ = ['fast_uuid', 'uuid', 'dcp', 'cp', 'pp', 'sha', 'wget', 'htmlify', 'freeze', 'require',
           'traceback', 'getplatform', 'iswindows', 'islinux', 'ismac']


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

    if which not in choices: # pragma: no cover
        errormsg = f'Choice {which} not found; choices are: {strjoin(choices.keys())}'
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
        except Exception as E: # pragma: no cover
            errormsg = f'Could not convert "{uid}" to a UID ({repr(E)})'
            if die:
                raise TypeError(errormsg)
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
    except Exception as E: # pragma: no cover
        output = cp(obj)
        errormsg = f'Warning: could not perform deep copy, performing shallow instead: {str(E)}'
        if die: raise RuntimeError(errormsg)
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
            if verbose: print(f'Could not jsonify object ("{str(E)}"), printing default...')
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


def freeze(lower=False):
    '''
    Alias for pip freeze.

    Args:
        lower (bool): convert all keys to lowercase

    **Example**::

        assert 'numpy' in sc.freeze() # One way to check for versions

    New in version 1.2.2.
    '''
    raw = dict(tuple(str(ws).split()) for ws in pkgr.working_set)
    keys = sorted(raw.keys())
    if lower:
        labels = {k:k.lower() for k in keys}
    else:
        labels = {k:k for k in keys}
    data = {labels[k]:raw[k] for k in keys} # Sort alphabetically
    return data


def require(reqs=None, *args, exact=False, detailed=False, die=True, verbose=True, **kwargs):
    '''
    Check whether environment requirements are met. Alias to pkg_resources.require().

    Args:
        reqs (list/dict): a list of strings, or a dict of package names and versions
        args (list): additional requirements
        kwargs (dict): additional requirements
        exact (bool): use '==' instead of '>=' as the default comparison operator if not specified
        detailed (bool): return a dict of which requirements are/aren't met
        die (bool): whether to raise an exception if requirements aren't met
        verbose (bool): print out the exception if it's not being raised

    **Examples**::

        sc.require('numpy')
        sc.require(numpy='')
        sc.require(reqs={'numpy':'1.19.1', 'matplotlib':'3.2.2'})
        sc.require('numpy>=1.19.1', 'matplotlib==3.2.2', die=False)
        sc.require(numpy='1.19.1', matplotlib='==4.2.2', die=False, detailed=True)

    New in version 1.2.2.
    '''

    # Handle inputs
    reqlist = list(args)
    reqdict = kwargs
    if isinstance(reqs, dict):
        reqdict.update(reqs)
    else:
        reqlist = mergelists(reqs, reqlist)

    # Turn into a list of strings
    comparechars = '<>=!~'
    for k,v in reqdict.items():
        if not v:
            entry = k # If no version is provided, entry is just the module name
        else:
            compare = '' if v.startswith(tuple(comparechars)) else ('==' if exact else '>=')
            entry = k + compare + v
        reqlist.append(entry)

    # Check the requirements
    data = dict()
    errs = dict()
    for entry in reqlist:
        try:
            pkgr.require(entry)
            data[entry] = True
        except Exception as E:
            data[entry] = False
            errs[entry] = E

    # Figure out output
    met = all([e==True for e in data.values()])

    # Handle exceptions
    if not met:
        errormsg = 'The following requirements were not met:'
        for k,v in data.items():
            if not v:
                errormsg += f'\n  {k}: {str(errs[k])}'
        if die:
            raise ModuleNotFoundError(errormsg) from errs[k] # Use the last one
        elif verbose:
            print(errormsg)

    # Handle output
    if detailed:
        return data, errs
    else:
        return met


def traceback(*args, **kwargs):
    '''
    Shortcut for accessing the traceback

    Alias for ``traceback.format_exc()``.
    '''
    return py_traceback.format_exc(*args, **kwargs)


def getplatform(expected=None, die=False):
    '''
    Return the name of the current platform: 'linux', 'windows', 'mac', or 'other'.
    Alias (kind of) to sys.platform.

    Args:
        expected (str): if not None, check if the current platform is this
        die (bool): if True and expected is defined, raise an exception

    **Example**::d

        sc.getplatform() # Get current name of platform
        sc.getplatform('windows', die=True) # Raise an exception if not on Windows
    '''
    # Define different aliases for each operating system
    mapping = dict(
        linux   = ['linux', 'posix'],
        windows = ['windows', 'win', 'win32', 'cygwin', 'nt'],
        mac     = ['mac', 'macos', 'darwin', 'osx']
    )

    # Check to see what system it is
    sys_plat = sys.platform
    plat = 'other'
    for key,aliases in mapping.items():
        if sys_plat.lower() in aliases:
            plat = key
            break

    # Handle output
    if expected is not None:
        output = (expected.lower() in mapping[plat]) # Check if it's as expecte
        if not output and die:
            errormsg = f'System is "{plat}", not "{expected}"'
            raise EnvironmentError(errormsg)
    else:
        output = plat
    return output


def iswindows(die=False):
    ''' Alias to sc.getplatform('windows') '''
    return getplatform('windows', die=die)

def islinux(die=False):
    ''' Alias to sc.getplatform('linux') '''
    return getplatform('linux', die=die)

def ismac(die=False):
    ''' Alias to sc.getplatform('mac') '''
    return getplatform('mac', die=die)


##############################################################################
#%% Printing/notification functions
##############################################################################

__all__ += ['printv', 'blank', 'strjoin', 'newlinejoin', 'createcollist', 'objectid', 'objatt', 'objmeth', 'objprop', 'objrepr',
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
    if thisverbose>4 or verbose>4: print(f'Warning, verbosity should be from 0-4 (this message: {thisverbose}; current: {verbose})')
    if verbose>=thisverbose: # Only print if sufficiently verbose
        indents = '  '*thisverbose*bool(indent) # Create automatic indenting
        if newline: print(indents+flexstr(string)) # Actually print
        else: print(indents+flexstr(string)), # Actually print
    return None


def blank(n=3):
    ''' Tiny function to print n blank lines, 3 by default '''
    print('\n'*n)


def strjoin(*args, sep=', '):
    '''
    Like string ``join()``, but handles more flexible inputs, converts items to
    strings. By default, join with commas.

    Args:
        args (list): the list of items to join
        sep (str): the separator string

    **Example**::

        sc.strjoin([1,2,3], 4, 'five')

    New in version 1.1.0.
    '''
    obj = []
    for arg in args:
        if isstring(arg):
            obj.append(arg)
        elif isiterable(arg):
            obj.extend([str(item) for item in arg])
        else:
            obj.append(str(arg))
    output = sep.join(obj)
    return output


def newlinejoin(*args):
    '''
    Alias to ``strjoin(*args, sep='\\n')``.

    **Example**::

        sc.newlinejoin([1,2,3], 4, 'five')

    New in version 1.1.0.
    '''
    return strjoin(*args, sep='\n')


def createcollist(items, title=None, strlen=18, ncol=3):
    ''' Creates a string for a nice columnated list (e.g. to use in __repr__ method) '''
    nrow = int(np.ceil(float(len(items))/ncol))
    newkeys = []
    for x in range(nrow):
        newkeys += items[x::nrow]

    attstring = title + ':' if title else ''
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
    c = obj.__class__
    output = f'<{c.__module__}.{c.__name__} at {hex(id(obj))}>\n'
    return output


def objatt(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object attributes for the Python __repr__ method '''
    if   hasattr(obj, '__dict__'):  oldkeys = sorted(obj.__dict__.keys())
    elif hasattr(obj, '__slots__'): oldkeys = sorted(obj.__slots__)
    else:                           oldkeys = []
    if len(oldkeys): output = createcollist(oldkeys, 'Attributes', strlen = 18, ncol = 3)
    else:            output = ''
    return output


def objmeth(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object methods for the Python __repr__ method '''
    try:
        oldkeys = sorted([method + '()' for method in dir(obj) if callable(getattr(obj, method)) and not method.startswith('__')])
    except: # pragma: no cover
        oldkeys = ['Methods N/A']
    if len(oldkeys): output = createcollist(oldkeys, 'Methods', strlen=strlen, ncol=ncol)
    else:            output = ''
    return output


def objprop(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object properties for the Python __repr__ method '''
    try:
        oldkeys = sorted([prop for prop in dir(obj) if isinstance(getattr(type(obj), prop, None), property) and not prop.startswith('__')])
    except: # pragma: no cover
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
            labels = [f'{type(obj)}']
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
    except Exception as E: # pragma: no cover
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
    else: # pragma: no cover
        print(arr) # Give up
    return None



def printdata(data, name='Variable', depth=1, maxlen=40, indent='', level=0, showcontents=False): # pragma: no cover
    '''
    Nicely print a complicated data structure, a la Matlab.

    Note: this function is deprecated.

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
        if   datatype==dict:              string = (f'dict with {len(data.keys())} keys')
        elif datatype==list:              string = (f'list of length {len(data)}')
        elif datatype==tuple:             string = (f'tuple of length {len(data)}')
        elif datatype==np.ndarray:        string = (f'array of shape {np.shape(data)}')
        elif datatype.__name__=='module': string = (f'module with {len(dir(data))} components')
        elif datatype.__name__=='class':  string = (f'class with {len(dir(data))} components')
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

    if label:  print(f'Variables for {label}:')
    if divider: print(dividerstr)
    for varnum,varname in enumerate(varlist):
        controlstr = f'{varnum}. "{varname}": ' # Basis for the control string -- variable number and name
        if color: controlstr = colorize(color, output=True) + controlstr + colorize('reset', output=True) # Optionally add color
        if spaces>1: controlstr += '\n' # Add a newline if the variables are going to be on different lines
        try:    controlstr += f'{localvars[varname]}' # The variable to be printed
        except: controlstr += 'WARNING, could not be printed' # In case something goes wrong
        controlstr += '\n' * spaces # The number of spaces to add between variables
        print(controlstr), # Print it out
    if divider: print(dividerstr) # If necessary, print the divider again
    return None



def slacknotification(message=None, webhook=None, to=None, fromuser=None, verbose=2, die=False):  # pragma: no cover
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
    elif os.path.exists(os.path.expanduser(webhook)): # If not, look for it sa a file
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
        import tempfile
        tempdir = tempfile.gettempdir()
        filename = os.path.join(tempdir, 'logfile') # Some generic filename that should work on *nix systems

    # Try writing to file
    try:
        with open(filename, 'a') as f:
            f.write('\n'+message+'\n') # Add a newline to the message.
    except Exception as E: # pragma: no cover # Fail gracefully
        print(f'Warning, could not write to logfile {filename}: {str(E)}')

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
    if not enable: # pragma: no cover
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
        if color not in ansicolors.keys(): # pragma: no cover
            print(f'Color "{color}" is not available, use colorize(showhelp=True) to show options.')
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


def heading(string=None, *args, color=None, divider=None, spaces=None, minlength=None, maxlength=None, sep=' ', output=True, **kwargs):
    '''
    Create a colorful heading. If just supplied with a string (or list of inputs like print()),
    create blue text with horizontal lines above and below and 3 spaces above. You
    can customize the color, the divider character, how many spaces appear before
    the heading, and the minimum length of the divider (otherwise will expand to
    match the length of the string, up to a maximum length).

    Args:
        string (str): The string to print as the heading (or object to convert to astring)
        args (list): Additional strings to print
        color (str): The color to use for the heading (default blue)
        divider (str): The symbol to use for the divider (default em dash)
        spaces (int): The number of spaces to put before the heading
        minlength (int): The minimum length of the divider
        maxlength (int): The maximum length of the divider
        sep (str): If multiple arguments are supplied, use this separator to join them
        output (bool): Whether to return the string as output (else, print)
        kwargs (dict): Arguments to pass to sc.colorize()

    Returns:
        String, unless output=False.

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

    # Convert to single string
    args = list(args)
    if string is not None:
        args = [string] + args
    string = sep.join(str(item) for item in args)

    # Add header and footer
    length = int(np.median([minlength, len(string), maxlength]))
    space = '\n'*spaces
    if divider and length: fulldivider = '\n'+divider*length+'\n'
    else:                  fulldivider = ''
    fullstring = space + fulldivider + string + fulldivider

    # Create output
    outputstring = colorize(color=color, string=fullstring, **kwargs)

    if output:
        return outputstring
    else:
        print(outputstring)
        return


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
#%% Type functions
##############################################################################

__all__ += ['flexstr', 'isiterable', 'checktype', 'isnumber', 'isstring', 'isarray',
            'promotetoarray', 'promotetolist', 'toarray', 'tolist', 'transposelist',
            'mergedicts', 'mergelists']

def flexstr(arg, force=True):
    '''
    Try converting any object to a "regular" string (i.e. ``str``), but proceed
    if it fails. Note: this function calls ``repr()`` rather than ``str()`` to
    ensure a more robust representation of objects.
    '''
    if isinstance(arg, str):
        return arg
    elif isinstance(arg, bytes):
        try:
            output = arg.decode() # If it's bytes, decode to unicode
        except: # pragma: no cover
            if force: output = repr(arg) # If that fails, just print its representation
            else:     output = arg
    else: # pragma: no cover
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
    be one of the following strings:

        - 'str', 'string': string or bytes object
        - 'num', 'number': any kind of number
        - 'arr', 'array': a Numpy array (equivalent to np.ndarray)
        - 'listlike': a list, tuple, or array
        - 'arraylike': a list, tuple, or array with numeric entries

    If subtype is not None, then checktype will iterate over the object and check
    recursively that each element matches the subtype.

    Args:
        obj     (any):         the object to check the type of
        objtype (str or type): the type to confirm the object belongs to
        subtype (str or type): optionally check the subtype if the object is iterable
        die     (bool):        whether or not to raise an exception if the object is the wrong type

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
    else: # pragma: no cover
        errormsg = f'Could not understand what type you want to check: should be either a string or a type, not "{objtype}"'
        raise ValueError(errormsg)

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
            errormsg = f'Incorrect type: object is {type(obj)}, but {objtype} is required'
            raise TypeError(errormsg)
        else:
            return None # It's fine, do nothing
    else: # Return the result of the comparison
        return result


def isnumber(obj, isnan=None):
    '''
    Determine whether or not the input is a number.

    Args:
        obj (any): the object to check if it's a number
        isnan (bool): an optional additional check to determine whether the number is/isn't NaN

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


def promotetoarray(x, keepnone=False, **kwargs):
    '''
    Small function to ensure consistent format for things that should be arrays
    (note: toarray()/promotetoarray() are identical).

    Very similar to ``np.array``, with the main difference being that ``sc.promotetoarray(3)``
    will return ``np.array([3])`` (i.e. a 1-d array that can be iterated over), while
    ``np.array(3)`` will return a 0-d array that can't be iterated over.

    Args:
        keepnone (bool): whether ``sc.promotetoarray(None)`` should return ``np.array([])`` or ``np.array([None], dtype=object)``
        kwargs (dict): passed to ``np.array()``

    **Examples**::

        sc.promotetoarray(5) # Returns np.array([5])
        sc.promotetoarray([3,5]) # Returns np.array([3,5])
        sc.promotetoarray(None, skipnone=True) # Returns np.array([])

    New in version 1.1.0: replaced "skipnone" with "keepnone"; allowed passing
    kwargs to ``np.array()``.
    '''
    skipnone = kwargs.pop('skipnone', None)
    if skipnone is not None: # pragma: no cover
        keepnone = not(skipnone)
        warnmsg = 'sc.promotetoarray() argument "skipnone" has been deprecated as of v1.1.0; use keepnone instead'
        warnings.warn(warnmsg, category=DeprecationWarning, stacklevel=2)
    if isnumber(x) or (isinstance(x, np.ndarray) and not np.shape(x)): # e.g. 3 or np.array(3)
        x = [x]
    elif x is None and not keepnone:
        x = []
    output = np.array(x, **kwargs)
    return output


def promotetolist(obj=None, objtype=None, keepnone=False, coerce='default'):
    '''
    Make sure object is always a list (note: tolist()/promotetolist() are identical).

    Used so functions can handle inputs like ``'a'``  or ``['a', 'b']``. In other
    words, if an argument can either be a single thing (e.g., a single dict key)
    or a list (e.g., a list of dict keys), this function can be used to do the
    conversion, so it's always safe to iterate over the output.

    While this usually wraps objects in a list rather than converts them to a list,
    the "coerce" argument can be used to change this behavior. Options are:

    - 'none' or None: do not coerce
    - 'default': coerce objects that were lists in Python 2 (range, map, dict_keys, dict_values, dict_items)
    - 'full': all the types in default, plus tuples and arrays

    Args:
        obj (anything): object to ensure is a list
        objtype (anything): optional type to check for each element; see ``sc.checktype()`` for details
        keepnone (bool): if ``keepnone`` is false, then ``None`` is converted to ``[]``; else, it's converted to ``[None]``
        coerce (str/tuple):  tuple of additional types to coerce to a list (as opposed to wrapping in a list)

    **Examples**::

        sc.promotetolist(5) # Returns [5]
        sc.promotetolist(np.array([3,5])) # Returns [np.array([3,5])] -- not [3,5]!
        sc.promotetolist(np.array([3,5]), coerce=np.ndarray) # Returns [3,5], since arrays are coerced to lists
        sc.promotetolist(None) # Returns []
        sc.promotetolist(range(3)) # Returns [0,1,2] since range is coerced by default
        sc.promotetolist(['a', 'b', 'c'], objtype='number') # Raises exception

        def myfunc(data, keys):
            keys = sc.promotetolist(keys)
            for key in keys:
                print(data[key])

        data = {'a':[1,2,3], 'b':[4,5,6]}
        myfunc(data, keys=['a', 'b']) # Works
        myfunc(data, keys='a') # Still works, equivalent to needing to supply keys=['a'] without promotetolist()

    New in version 1.1.0: "coerce" argument
    New in version 1.2.2: default coerce values
    '''
    # Handle coerce
    default_coerce = (range, map, type({}.keys()), type({}.values()), type({}.items()))
    if isinstance(coerce, str):
        if coerce == 'none':
            coerce = None
        elif coerce == 'default':
            coerce = default_coerce
        elif coerce == 'full':
            coerce = default_coerce + (tuple, np.ndarray)
        else:
            errormsg = f'Option "{coerce}"; not recognized; must be "none", "default", or "full"'

    if objtype is None: # Don't do type checking
        if isinstance(obj, list):
            output = obj # If it's already a list and we're not doing type checking, just return
        elif obj is None:
            if keepnone:
                output = [None] # Wrap in a list
            else:
                output = [] # Return an empty list, the "none" equivalent for a list
        else:
            if coerce is not None and isinstance(obj, coerce):
                output = list(obj) # Coerce to list
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
                errormsg = f'promotetolist(): type mismatch, expecting type {objtype}'
                raise TypeError(errormsg) from E
    return output


# Aliases for core functions
toarray = promotetoarray
tolist = promotetolist


def transposelist(obj):
    '''
    Convert e.g. a list of key-value tuples into a list of keys and a list of values.

    **Example**::

        o = sc.odict(a=1, b=4, c=9, d=16)
        itemlist = o.enumitems()
        inds, keys, vals = sc.transposelist(itemlist)

    New in version 1.1.0.
    '''
    return list(map(list, zip(*obj)))


def mergedicts(*args, strict=False, overwrite=True, copy=False):
    '''
    Small function to merge multiple dicts together. By default, skips things
    that are not, dicts (e.g., None), and allows keys to be set multiple times.
    Similar to dict.update(), except returns a value. The first dictionary supplied
    will be used for the output type (e.g. if the first dictionary is an odict,
    an odict will be returned).

    Useful for cases, e.g. function arguments, where the default option is ``None``
    but you will need a dict later on.

    Args:
        strict    (bool): if True, raise an exception if an argument isn't a dict
        overwrite (bool): if False, raise an exception if multiple keys are found
        copy      (bool): whether or not to deepcopy the merged dictionary
        *args     (dict): the sequence of dicts to be merged

    **Examples**::

        d0 = sc.mergedicts(user_args) # Useful if user_args might be None, but d0 is always a dict
        d1 = sc.mergedicts({'a':1}, {'b':2}) # Returns {'a':1, 'b':2}
        d2 = sc.mergedicts({'a':1, 'b':2}, {'b':3, 'c':4}) # Returns {'a':1, 'b':3, 'c':4}
        d3 = sc.mergedicts(sc.odict({'b':3, 'c':4}), {'a':1, 'b':2}) # Returns sc.odict({'b':2, 'c':4, 'a':1})
        d4 = sc.mergedicts({'b':3, 'c':4}, {'a':1, 'b':2}, overwrite=False) # Raises exception

    New in version 1.1.0: "copy" argument
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
                    errormsg = f'Could not merge dicts since keys "{strjoin(intersection)}" overlap and overwrite=False'
                    raise KeyError(errormsg)
            outputdict.update(arg)
    if copy:
        outputdict = dcp(outputdict)
    return outputdict


def mergelists(*args, copy=False, **kwargs):
    '''
    Merge multiple lists together.

    Args:
        args (any): the lists, or items, to be joined together into a list
        copy (bool): whether to deepcopy the resultant object
        kwargs (dict): passed to ``sc.promotetolist()``, which is called on each argument

    **Examples**::

        sc.mergelists(None) # Returns []
        sc.mergelists([1,2,3], [4,5,6]) # Returns [1, 2, 3, 4, 5, 6]
        sc.mergelists([1,2,3], 4, 5, 6) # Returns [1, 2, 3, 4, 5, 6]
        sc.mergelists([(1,2), (3,4)], (5,6)) # Returns [(1, 2), (3, 4), (5, 6)]
        sc.mergelists((1,2), (3,4), (5,6)) # Returns [(1, 2), (3, 4), (5, 6)]
        sc.mergelists((1,2), (3,4), (5,6), coerce=tuple) # Returns [1, 2, 3, 4, 5, 6]

    New in version 1.1.0.
    '''
    obj = []
    for arg in args:
        arg = promotetolist(arg, **kwargs)
        obj.extend(arg)
    if copy:
        obj = dcp(obj)
    return obj


##############################################################################
#%% Time/date functions
##############################################################################

__all__ += ['now', 'getdate', 'readdate', 'date', 'day', 'daydiff', 'daterange', 'datedelta', 'datetoyear',
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
            if isstring(obj):
                return obj # Return directly if it's a string
            obj.timetuple() # Try something that will only work if it's a date object
            dateobj = obj # Test passed: it's a date object
        except Exception as E: # pragma: no cover # It's not a date object
            errormsg = f'Getting date failed; date must be a string or a date object: {repr(E)}'
            raise TypeError(errormsg)

        if   astype == 'str':     output = dateobj.strftime(dateformat)
        elif astype == 'int':     output = time.mktime(dateobj.timetuple()) # So ugly!! But it works -- return integer representation of time
        elif astype == 'dateobj': output = dateobj
        else: # pragma: no cover
            errormsg = f'"astype={astype}" not understood; must be "str" or "int"'
            raise ValueError(errormsg)
        return output


def _sanitize_iterables(obj, *args):
    '''
    Take input as a list, array, or non-iterable type, along with one or more
    arguments, and return a list, along with information on what the input types
    were.

    **Examples**::

        _sanitize_iterables(1, 2, 3)             # Returns [1,2,3], False, False
        _sanitize_iterables([1, 2], 3)           # Returns [1,2,3], True, False
        _sanitize_iterables(np.array([1, 2]), 3) # Returns [1,2,3], True, True
        _sanitize_iterables(np.array([1, 2, 3])) # Returns [1,2,3], False, True
    '''
    is_list = isinstance(obj, list) or len(args)>0 # If we're given a list of args, treat it like a list
    is_array = isinstance(obj, np.ndarray) # Check if it's an array
    if is_array: # If it is, convert it to a list
        obj = obj.tolist()
    objs = dcp(promotetolist(obj)) # Ensure it's a list, and deepcopy to avoid mutability
    objs.extend(args) # Add on any arguments
    return objs, is_list, is_array


def _sanitize_output(obj, is_list, is_array, dtype=None):
    '''
    The companion to _sanitize_iterables, convert the object back to the original
    type supplied.
    '''
    if is_array:
        output = np.array(obj, dtype=dtype)
    elif not is_list and len(obj) == 1:
        output = obj[0]
    else:
        output = obj
    return output


def readdate(datestr=None, *args, dateformat=None, return_defaults=False):
    '''
    Convenience function for loading a date from a string. If dateformat is None,
    this function tries a list of standard date types.

    By default, a numeric date is treated as a POSIX (Unix) timestamp. This can be changed
    with the ``dateformat`` argument, specifically:

    - 'posix'/None: treat as a POSIX timestamp, in seconds from 1970
    - 'ordinal'/'matplotlib': treat as an ordinal number of days from 1970 (Matplotlib default)

    Args:
        datestr (int, float, str or list): the string containing the date, or the timestamp (in seconds), or a list of either
        args (list): additional dates to convert
        dateformat (str or list): the format for the date, if known; if 'dmy' or 'mdy', try as day-month-year or month-day-year formats; can also be a list of options
        return_defaults (bool): don't convert the date, just return the defaults

    Returns:
        dateobj (date): a datetime object

    **Examples**::

        dateobj  = sc.readdate('2020-03-03') # Standard format, so works
        dateobj  = sc.readdate('04-03-2020', dateformat='dmy') # Date is ambiguous, so need to specify day-month-year order
        dateobj  = sc.readdate(1611661666) # Can read timestamps as well
        dateobj  = sc.readdate(16166, dateformat='ordinal') # Or ordinal numbers of days, as used by Matplotlib
        dateobjs = sc.readdate(['2020-06', '2020-07'], dateformat='%Y-%m') # Can read custom date formats
        dateobjs = sc.readdate('20200321', 1611661666) # Can mix and match formats
    '''

    # Define default formats
    formats_to_try = {
        'date':           '%Y-%m-%d', # 2020-03-21
        'date-slash':     '%Y/%m/%d', # 2020/03/21
        'date-dot':       '%Y.%m.%d', # 2020.03.21
        'date-space':     '%Y %m %d', # 2020 03 21
        'date-alpha':     '%Y-%b-%d', # 2020-Mar-21
        'date-alpha-rev': '%d-%b-%Y', # 21-Mar-2020
        'date-alpha-sp':  '%d %b %Y', # 21 Mar 2020
        'date-Alpha':     '%Y-%B-%d', # 2020-March-21
        'date-Alpha-rev': '%d-%B-%Y', # 21-March-2020
        'date-Alpha-sp':  '%d %B %Y', # 21 March 2020
        'date-numeric':   '%Y%m%d',   # 20200321
        'datetime':       '%Y-%m-%d %H:%M:%S',    # 2020-03-21 14:35:21
        'datetime-alpha': '%Y-%b-%d %H:%M:%S',    # 2020-Mar-21 14:35:21
        'default':        '%Y-%m-%d %H:%M:%S.%f', # 2020-03-21 14:35:21.23483
        'ctime':          '%a %b %d %H:%M:%S %Y', # Sat Mar 21 23:09:29 2020
        }

    # Define day-month-year formats
    dmy_formats = {
        'date':           '%d-%m-%Y', # 21-03-2020
        'date-slash':     '%d/%m/%Y', # 21/03/2020
        'date-dot':       '%d.%m.%Y', # 21.03.2020
        'date-space':     '%d %m %Y', # 21 03 2020
    }

    # Define month-day-year formats
    mdy_formats = {
        'date':           '%m-%d-%Y', # 03-21-2020
        'date-slash':     '%m/%d/%Y', # 03/21/2020
        'date-dot':       '%m.%d.%Y', # 03.21.2020
        'date-space':     '%m %d %Y', # 03 21 2020
    }

    # To get the available formats
    if return_defaults:
        return formats_to_try

    # Handle date formats
    format_list = promotetolist(dateformat, keepnone=True) # Keep none which signifies default
    if dateformat is not None:
        if dateformat == 'dmy':
            formats_to_try = dmy_formats
        elif dateformat == 'mdy':
            formats_to_try = mdy_formats
        else:
            formats_to_try = {}
            for f,fmt in enumerate(format_list):
                formats_to_try[f'User supplied {f}'] = fmt

    # Ensure everything is in a consistent format
    datestrs, is_list, is_array = _sanitize_iterables(datestr, *args)

    # Actually process the dates
    dateobjs = []
    for datestr in datestrs: # Iterate over them
        dateobj = None
        exceptions = {}
        if isinstance(datestr, dt.datetime):
            dateobj = datestr # Nothing to do
        elif isnumber(datestr):
            if 'posix' in format_list or None in format_list:
                dateobj = dt.datetime.fromtimestamp(datestr)
            elif 'ordinal' in format_list or 'matplotlib' in format_list:
                dateobj = pl.num2date(datestr)
            else:
                errormsg = f'Could not convert numeric date {datestr} using available formats {strjoin(format_list)}; must be "posix" or "ordinal"'
                raise ValueError(errormsg)
        else:
            for key,fmt in formats_to_try.items():
                try:
                    dateobj = dt.datetime.strptime(datestr, fmt)
                    break # If we find one that works, we can stop
                except Exception as E:
                    exceptions[key] = str(E)
            if dateobj is None:
                formatstr = newlinejoin([f'{item[1]}' for item in formats_to_try.items()])
                errormsg = f'Was unable to convert "{datestr}" to a date using the formats:\n{formatstr}'
                if dateformat not in ['dmy', 'mdy']:
                    errormsg += '\n\nNote: to read day-month-year or month-day-year dates, use dateformat="dmy" or "mdy" respectively.'
                raise ValueError(errormsg)
        dateobjs.append(dateobj)

    # If only a single date was supplied, return just that; else return the list/array
    output = _sanitize_output(dateobjs, is_list, is_array, dtype=object)
    return output


def date(obj, *args, start_date=None, readformat=None, outformat=None, as_date=True, **kwargs):
    '''
    Convert any reasonable object -- a string, integer, or datetime object, or
    list/array of any of those -- to a date object. To convert an integer to a
    date, you must supply a start date.

    Caution: while this function and readdate() are similar, and indeed this function
    calls readdate() if the input is a string, in this function an integer is treated
    as a number of days from start_date, while for readdate() it is treated as a
    timestamp in seconds. To change

    Args:
        obj (str, int, date, datetime, list, array): the object to convert
        args (str, int, date, datetime): additional objects to convert
        start_date (str, date, datetime): the starting date, if an integer is supplied
        readformat (str/list): the format to read the date in; passed to sc.readdate()
        outformat (str): the format to output the date in, if returning a string
        as_date (bool): whether to return as a datetime date instead of a string

    Returns:
        dates (date or list): either a single date object, or a list of them (matching input data type where possible)

    **Examples**::

        sc.date('2020-04-05') # Returns datetime.date(2020, 4, 5)
        sc.date([35,36,37], start_date='2020-01-01', as_date=False) # Returns ['2020-02-05', '2020-02-06', '2020-02-07']
        sc.date(1923288822, readformat='posix') # Interpret as a POSIX timestamp

    New in version 1.0.0.
    New in version 1.2.2: "readformat" argument; renamed "dateformat" to "outformat"
    '''
    # Handle deprecation
    dateformat = kwargs.pop('dateformat', None)
    if dateformat is not None: # pragma: no cover
        outformat = dateformat
        warnmsg = 'sc.date() argument "dateformat" has been deprecated as of v1.2.2; use "outformat" instead'
        warnings.warn(warnmsg, category=DeprecationWarning, stacklevel=2)

    # Convert to list and handle other inputs
    if obj is None:
        return None
    if outformat is None:
        outformat = '%Y-%m-%d'
    obj, is_list, is_array = _sanitize_iterables(obj, *args)

    dates = []
    for d in obj:
        if d is None:
            dates.append(d)
            continue
        try:
            if type(d) == dt.date: # Do not use isinstance, since must be the exact type
                pass
            elif isinstance(d, dt.datetime):
                d = d.date()
            elif isstring(d):
                d = readdate(d, dateformat=readformat).date()
            elif isnumber(d):
                if readformat is not None:
                    d = readdate(d, dateformat=readformat).date()
                else:
                    if start_date is None:
                        errormsg = f'To convert the number {d} to a date, you must either specify "posix" or "ordinal" read format, or supply start_date'
                        raise ValueError(errormsg)
                    d = date(start_date) + dt.timedelta(days=int(d))
            else: # pragma: no cover
                errormsg = f'Cannot interpret {type(d)} as a date, must be date, datetime, or string'
                raise TypeError(errormsg)
            if as_date:
                dates.append(d)
            else:
                dates.append(d.strftime(outformat))
        except Exception as E:
            errormsg = f'Conversion of "{d}" to a date failed: {str(E)}'
            raise ValueError(errormsg)

    # Return an integer rather than a list if only one provided
    output = _sanitize_output(dates, is_list, is_array, dtype=object)
    return output


def day(obj, *args, start_date=None, **kwargs):
    '''
    Convert a string, date/datetime object, or int to a day (int), the number of
    days since the start day. See also sc.date() and sc.daydiff(). If a start day
    is not supplied, it returns the number of days into the current year.

    Args:
        obj (str, date, int, list, array): convert any of these objects to a day relative to the start day
        args (list): additional days
        start_date (str or date): the start day; if none is supplied, return days since (supplied year)-01-01.

    Returns:
        days (int or list): the day(s) in simulation time (matching input data type where possible)

    **Examples**::

        sc.day(sc.now()) # Returns how many days into the year we are
        sc.day(['2021-01-21', '2024-04-04'], start_date='2022-02-22') # Days can be positive or negative

    New in version 1.0.0.
    New in version 1.2.2: renamed "start_day" to "start_date"
    '''

    # Handle deprecation
    start_day = kwargs.pop('start_day', None)
    if start_day is not None: # pragma: no cover
        start_date = start_day
        warnmsg = 'sc.day() argument "start_day" has been deprecated as of v1.2.2; use "start_date" instead'
        warnings.warn(warnmsg, category=DeprecationWarning, stacklevel=2)

    # Do not process a day if it's not supplied, and ensure it's a list
    if obj is None:
        return None
    obj, is_list, is_array = _sanitize_iterables(obj, *args)

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
                if start_date:
                    start_date = date(start_date)
                else:
                    start_date = date(f'{d.year}-01-01')
                d_day = (d - start_date).days # Heavy lifting -- actually compute the day
                days.append(d_day)
            except Exception as E: # pragma: no cover
                errormsg = f'Could not interpret "{d}" as a date: {str(E)}'
                raise ValueError(errormsg)

    # Return an integer rather than a list if only one provided
    output = _sanitize_output(days, is_list, is_array)
    return output


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
    end_day = day(end_date, start_date=start_date)
    if inclusive:
        end_day += 1
    days = list(range(end_day))
    dates = date(days, start_date=start_date, as_date=as_date, dateformat=dateformat)
    return dates


def datedelta(datestr, days=0, months=0, years=0, weeks=0, as_date=None, **kwargs):
    '''
    Perform calculations on a date string (or date object), returning a string (or a date).
    Wrapper to dateutil.relativedelta().

    Args:
        datestr (str/date): the starting date (typically a string)
        days (int): the number of days (positive or negative) to increment
        months (int): as above
        years (int): as above
        weeks (int): as above
        as_date (bool): if True, return a date object; otherwise, return as input type
        kwargs (dict): passed to ``sc.readdate()``

    **Examples**::

        sc.datedelta('2021-07-07', 3) # Add 3 days
        sc.datedelta('2021-07-07', days=-4) # Subtract 4 days
        sc.datedelta('2021-07-07', weeks=4, months=-1, as_date=True) # Add 4 weeks but subtract a month, and return a dateobj
    '''
    if as_date is None and isinstance(datestr, str): # Typical case
        as_date = False
    dateobj = readdate(datestr, **kwargs)
    newdate = dateobj + dateutil.relativedelta.relativedelta(days=days, months=months, years=years, weeks=weeks)
    newdate = date(newdate, as_date=as_date)
    return newdate


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


def elapsedtimestr(pasttime, maxdays=5, minseconds=10, shortmonths=True):
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

    **Examples**::

        yesterday = sc.datedelta(sc.now(), days=-1)
        sc.elapsedtimestr(yesterday)
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
            pasttime = readdate(pasttime)
        except ValueError as E: # pragma: no cover
            errormsg = f"User supplied string {pasttime} is not in a readable format."
            raise ValueError(errormsg) from E
    elif isinstance(pasttime, dt.datetime):
        pass
    else: # pragma: no cover
        errormsg = f"User-supplied value {pasttime} is neither a datetime object nor an ISO 8601 string."
        raise TypeError(errormsg)

    # It doesn't make sense to measure time elapsed between now and a future date, so we'll just print the date
    if pasttime > now_time:
        includeyear = (pasttime.year != now_time.year)
        time_str = print_date(pasttime, includeyear=includeyear, shortmonths=shortmonths)

    # Otherwise, start by getting the elapsed time as a datetime object
    else:
        elapsed_time = now_time - pasttime

        # Check if the time is within the last minute
        if elapsed_time < dt.timedelta(seconds=60):
            if elapsed_time.seconds <= minseconds:
                time_str = "just now"
            else:
                time_str = f"{elapsed_time.seconds} secs ago"

        # Check if the time is within the last hour
        elif elapsed_time < dt.timedelta(seconds=60 * 60):

            # We know that seconds > 60, so we can safely round down
            minutes = int(elapsed_time.seconds / 60)
            if minutes == 1:
                time_str = "a minute ago"
            else:
                time_str = f"{minutes} mins ago"

        # Check if the time is within the last day
        elif elapsed_time < dt.timedelta(seconds=60 * 60 * 24 - 1):

            # We know that it's at least an hour, so we can safely round down
            hours = int(elapsed_time.seconds / (60 * 60))
            if hours == 1:
                time_str = "1 hour ago"
            else:
                time_str = f"{hours} hours ago"

        # Check if it's within the last N days, where N is a user-supplied argument
        elif elapsed_time < dt.timedelta(days=maxdays):
            if elapsed_time.days == 1:
                time_str = "yesterday"
            else:
                time_str = f"{elapsed_time.days} days ago"

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
    else:         base = f'Elapsed time for {label}: '
    logmessage = base + f'{sigfig(elapsed, sigfigs=sigfigs)} s'

    # Optionally reset the counter
    if reset:
        _tictime = time.time()  # Store the present time in the global

    if output:
        return elapsed
    else:
        if filename is not None: printtologfile(logmessage, filename) # If we passed in a filename, append the message to that file.
        else: print(logmessage) # Otherwise, print the message.
        return


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
                print(f'Pausing for {remaining:0.1f} s')
            time.sleep(remaining)
        else:
            if verbose:
                print(f'Warning, delay less than elapsed time ({delay:0.1f} vs. {elapsed:0.1f})')
    return None



##############################################################################
#%% Misc. functions
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
        saveobj(filename, variable, die=False)
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

    if plot: # pragma: no cover
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
    except KeyError: # pragma: no cover
        raise KeyNotFoundError(f'Unit {unit} not found among {strjoin(mapping.keys())}')
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
    except Exception as E: # pragma: no cover
        output = f'runcommand(): shell command failed: {str(E)}' # This is for a Python error, not a shell error -- those get passed to output
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

    Note: if direct directory reading fails, it will attempt to use the gitpython
    library.

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
            else: # pragma: no cover
                parent, _ = os.path.split(curpath)
                if parent == curpath:
                    curpath = None
                else:
                    curpath = parent
        else: # pragma: no cover
            raise RuntimeError("Could not find .git directory")

        # Then, get the branch and commit
        with open(os.path.join(gitdir, "HEAD"), "r") as f1:
            ref = f1.read()
            if ref.startswith("ref:"):
                refdir = ref.split(" ")[1].strip()  # The path to the file with the commit
                gitbranch = refdir.replace("refs/heads/", "")  # / is always used (not os.sep)
                with open(os.path.join(gitdir, refdir), "r") as f2:
                    githash = f2.read().strip()  # The hash of the commit
            else: # pragma: no cover
                gitbranch = "Detached head (no branch)"
                githash = ref.strip()

        # Now read the time from the commit
        with open(os.path.join(gitdir, "objects", githash[0:2], githash[2:]), "rb") as f3:
            compressed_contents = f3.read()
            decompressed_contents = zlib.decompress(compressed_contents).decode()
            for line in decompressed_contents.split("\n"):
                if line.startswith("author"):
                    _re_actor_epoch = re.compile(r"^.+? (.*) (\d+) ([+-]\d+).*$")
                    m = _re_actor_epoch.search(line)
                    actor, epoch, offset = m.groups()
                    t = time.gmtime(int(epoch))
                    gitdate = time.strftime("%Y-%m-%d %H:%M:%S UTC", t)

    except Exception as E: # pragma: no cover
        try: # Second, try importing gitpython
            import git
            rootdir = os.path.abspath(path) # e.g. /user/username/my/folder
            repo = git.Repo(path=rootdir, search_parent_directories=True)
            try:
                gitbranch = str(repo.active_branch.name)  # Just make sure it's a string
            except TypeError:
                gitbranch = 'Detached head (no branch)'
            githash = str(repo.head.object.hexsha)
            gitdate = str(repo.head.object.authored_datetime.isoformat())
        except Exception as E2:
            errormsg = f'''Could not extract git info; please check paths:
  Method 1 (direct read) error: {str(E)}
  Method 2 (gitpython) error:   {str(E2)}'''
            if die:
                raise RuntimeError(errormsg) from E
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
    format 1.2.3, but numeric works too. Returns 0 for equality, -1 for v1<v2, and
    1 for v1>v2.

    If ``version2`` starts with >, >=, <, <=, or ==, the function returns True or
    False depending on the result of the comparison.

    **Examples**::

        sc.compareversions('1.2.3', '2.3.4') # returns -1
        sc.compareversions(2, '2') # returns 0
        sc.compareversions('3.1', '2.99') # returns 1
        sc.compareversions('3.1', '>=2.99') # returns True
        sc.compareversions(mymodule.__version__, '>=1.0') # common usage pattern
        sc.compareversions(mymodule, '>=1.0') # alias to the above

    New in version 1.2.1: relational operators
    '''
    # Handle inputs
    if isinstance(version1, types.ModuleType):
        try:
            version1 = version1.__version__
        except Exception as E:
            errormsg = f'{version1} is a module, but does not have a __version__ attribute'
            raise AttributeError(errormsg) from E
    v1 = str(version1)
    v2 = str(version2)

    # Process version2
    valid = None
    if   v2.startswith('>'):  valid = [1]
    elif v2.startswith('>='): valid = [0,1]
    elif v2.startswith('='):  valid = [0]
    elif v2.startswith('=='): valid = [0]
    elif v2.startswith('~='): valid = [-1,1]
    elif v2.startswith('!='): valid = [-1,1]
    elif v2.startswith('<='): valid = [0,-1]
    elif v2.startswith('<'):  valid = [-1]
    v2 = v2.lstrip('<>=!~')

    # Do comparison
    if LooseVersion(v1) > LooseVersion(v2):
        comparison =  1
    elif LooseVersion(v1) < LooseVersion(v2):
        comparison =  -1
    else:
        comparison =  0

    # Return
    if valid is None:
        return comparison
    else:
        tf = (comparison in valid)
        return tf


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
    except Exception as E: # pragma: no cover
        errormsg = f'Cannot use "{name}" since {name} is not installed.\nPlease install {name} and try again.'
        print(errormsg)
        if die: raise E
        else:   return False
    if output: return module
    else:      return True


def suggest(user_input, valid_inputs, n=1, threshold=None, fulloutput=False, die=False, which='damerau'):
    """
    Return suggested item

    Returns item with lowest Levenshtein distance, where case substitution and stripping
    whitespace are not included in the distance. If there are ties, then the additional operations
    will be included.

    Args:
        user_input (str): User's input
        valid_inputs (list): List/collection of valid strings
        n (int): Maximum number of suggestions to return
        threshold (int): Maximum number of edits required for an option to be suggested (by default, two-thirds the length of the input; for no threshold, set to -1)
        die (bool): If True, an informative error will be raised (to avoid having to implement this in the calling code)
        which (str): Distance calculation method used; options are "damerau" (default), "levenshtein", or "jaro"

    Returns:
        suggestions (str or list): Suggested string. Returns None if no suggestions with edit distance less than threshold were found. This helps to make
             suggestions more relevant.

    **Examples**::

        >>> sc.suggest('foo',['Foo','Bar'])
        'Foo'
        >>> sc.suggest('foo',['FOO','Foo'])
        'Foo'
        >>> sc.suggest('foo',['Foo ','boo'])
        'Foo '
    """
    try:
        import jellyfish # To allow as an optional import
    except ModuleNotFoundError as e: # pragma: no cover
        raise ModuleNotFoundError('The "jellyfish" Python package is not available; please install via "pip install jellyfish"') from e

    valid_inputs = promotetolist(valid_inputs, objtype='string')

    mapping = {
        'damerau':     jellyfish.damerau_levenshtein_distance,
        'levenshtein': jellyfish.levenshtein_distance,
        'jaro':        jellyfish.jaro_distance,
        }

    keys = list(mapping.keys())
    if which not in keys: # pragma: no cover
        errormsg = f'Method {which} not available; options are {strjoin(keys)}'
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
    suggestionstr = strjoin([f'"{sugg}"' for sugg in suggestions[:n]])

    # Handle threshold
    if threshold is None:
        threshold = np.ceil(len(user_input)*2/3)
    if threshold < 0:
        threshold = np.inf

    # Output
    if min(distance) > threshold:
        if die: # pragma: no cover
            errormsg = f'"{user_input}" not found'
            raise ValueError(errormsg)
        else:
            return None
    elif die:
        errormsg = f'"{user_input} not found - did you mean {suggestionstr}'
        raise ValueError(errormsg)
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
    except ModuleNotFoundError as E: # pragma: no cover
        if 'win' in sys.platform:
            errormsg = 'The "line_profiler" package is not included by default on Windows;' \
                        'please install using "pip install line_profiler" (note: you will need a ' \
                        'C compiler installed, e.g. Microsoft Visual Studio)'
        else:
            errormsg = 'The "line_profiler" Python package is required to perform profiling'
        raise ModuleNotFoundError(errormsg) from E

    if follow is None:
        follow = run
    orig_func = run

    lp = LineProfiler()
    follow = promotetolist(follow)
    for f in follow:
        lp.add_function(f)
    lp.enable_by_count()
    wrapper = lp(run)

    if print_stats: # pragma: no cover
        print('Profiling...')
    wrapper(*args, **kwargs)
    run = orig_func
    if print_stats: # pragma: no cover
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
    except ModuleNotFoundError as E: # pragma: no cover
        if 'win' in sys.platform:
            errormsg = 'The "memory_profiler" package is not included by default on Windows;' \
                        'please install using "pip install memory_profiler" (note: you will need a ' \
                        'C compiler installed, e.g. Microsoft Visual Studio)'
        else:
            errormsg = 'The "memory_profiler" Python package is required to perform profiling'
        raise ModuleNotFoundError(errormsg) from E

    if follow is None:
        follow = run

    lp = mp.LineProfiler()
    follow = promotetolist(follow)
    for f in follow:
        lp.add_function(f)
    lp.enable_by_count()
    try:
        wrapper = lp(run)
    except TypeError as e: # pragma: no cover
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
    except Exception as E: # pragma: no cover
        if tostring:
            output = f'Calling function information not available ({str(E)})'
        else:
            output = {'filename':'N/A', 'lineno':'N/A'}
    return output



##############################################################################
#%% Nested dictionary functions
##############################################################################

__all__ += ['getnested', 'setnested', 'makenested', 'iternested', 'mergenested',
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



##############################################################################
#%% Classes
##############################################################################

__all__ += ['KeyNotFoundError', 'LinkException', 'prettyobj', 'autolist', 'Link', 'Timer']


class KeyNotFoundError(KeyError):
    '''
    A tiny class to fix repr for KeyErrors. KeyError prints the repr of the error
    message, rather than the actual message, so e.g. newline characters print as
    the character rather than the actual newline.

    **Example**::

        raise sc.KeyNotFoundError('The key "foo" is not available, but these are: "bar", "cat"')
    '''

    def __str__(self): # pragma: no cover
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


class autolist(list):
    '''
    A simple extension to a list that defines add methods to simplify appending
    and extension.

    **Examples**::

        ls = sc.autolist(3) # Quickly convert a scalar to a list

        ls = sc.autolist()
        for i in range(5):
            ls += i # No need for ls += [i]
    '''
    def __init__(self, *args):
        arglist = mergelists(*args) # Convert non-iterables to iterables
        return super().__init__(arglist)

    def __add__(self, obj=None):
        ''' Allows non-lists to be concatenated '''
        obj = promotetolist(obj)
        new = super().__add__(obj)
        return new

    def __radd__(self, obj):
        ''' Allows sum() to work correctly '''
        return self.__add__(obj)

    def __iadd__(self, obj):
        ''' Allows += to work correctly '''
        obj = promotetolist(obj)
        self.extend(obj)
        return self



class Link(object):
    '''
    A class to differentiate between an object and a link to an object. The idea
    is that this object is parsed differently from other objects -- most notably,
    a recursive method (such as a pickle) would skip over Link objects, and then
    would fix them up after the other objects had been reinstated.

    Version: 2017jan31
    '''

    def __init__(self, obj=None):
        self.obj = obj # Store the object -- or rather a reference to it, if it's mutable
        try:    self.uid = obj.uid # If the object has a UID, store it separately
        except: self.uid = None # If not, just use None

    def __repr__(self): # pragma: no cover
        ''' Just use default '''
        output  = prepr(self)
        return output

    def __call__(self, obj=None):
        ''' If called with no argument, return the stored object; if called with argument, update object '''
        if obj is None:
            if type(self.obj)==LinkException: # If the link is broken, raise it now
                raise self.obj
            return self.obj
        else: # pragma: no cover
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

    This wraps ``tic`` and ``toc`` with the formatting arguments and
    the start time (at construction)
    Use this in a ``with...as`` block to automatically print
    elapsed time when the block finishes.

    Implementation based on https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/

    Example making repeated calls to the same Timer::

        >>> timer = Timer()
        >>> timer.toc()
        Elapsed time: 2.63 s
        >>> timer.toc()
        Elapsed time: 5.00 s

    Example wrapping code using with-as::

        >>> with Timer(label='mylabel') as t:
        >>>     foo()

    '''
    def __init__(self,**kwargs):
        self.tic()
        self.kwargs = kwargs #: Store kwargs to pass to :func:`toc` at the end of the block
        return

    def __enter__(self):
        ''' Reset start time when entering with-as block '''
        self.tic()
        return self

    def __exit__(self, *args):
        ''' Print elapsed time when leaving a with-as block '''
        self.toc()

    def tic(self):
        ''' Set start time '''
        self.start = tic()

    def toc(self):
        ''' Print elapsed time '''
        toc(self.start,**self.kwargs)
