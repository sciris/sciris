"""
Miscellaneous utilities for type checking, printing, dates and times, etc.

Note: there are a lot! The design philosophy has been that it's easier to
ignore a function that you don't need than write one from scratch that you
do need.

Highlights:
    - :func:`sc.dcp() <dcp>`: shortcut to :func:`copy.deepcopy()`
    - :func:`sc.pp() <pp>`: shortcut to :func:`pprint.pprint()`
    - :func:`sc.isnumber() <isnumber>`: checks if something is any number type
    - :func:`sc.tolist() <tolist>`: converts any object to a list, for easy iteration
    - :func:`sc.toarray() <toarray>`: tries to convert any object to an array, for easy use with numpy
    - :func:`sc.mergedicts() <mergedicts>`: merges any set of inputs into a dictionary
    - :func:`sc.mergelists() <mergelists>`: merges any set of inputs into a list
    - :func:`sc.runcommand() <runcommand>`: simple way of executing a shell command
    - :func:`sc.download() <download>`: download multiple URLs in parallel
"""

##############################################################################
#%% Imports
##############################################################################

import re
import sys
import types
import copy
import json
import string
import numbers
import pprint
import hashlib
import getpass
import inspect
import warnings
import importlib
import subprocess
import unicodedata
import numpy as np
import pandas as pd
import random as rnd
import uuid as py_uuid
import contextlib as cl
import traceback as py_traceback
from pathlib import Path
import sciris as sc

# Handle types
_stringtypes = (str, bytes)
_numtype     = (numbers.Number,)
_booltypes   = (bool, np.bool_)

# Store these for access by other modules
__all__ = ['_stringtypes', '_numtype', '_booltypes']


##############################################################################
#%% Adaptations from other libraries
##############################################################################

# Define the modules being loaded
__all__ += ['fast_uuid', 'uuid', 'dcp', 'cp', 'pp', 'sha', 'traceback', 'getuser',
           'getplatform', 'iswindows', 'islinux', 'ismac', 'isjupyter', 'asciify']


def fast_uuid(which=None, length=None, n=1, secure=False, forcelist=False, safety=1000, recursion=0, recursion_limit=10, verbose=True):
    """
    Create a fast UID or set of UIDs. Note: for certain applications, :func:`sc.uuid() <uuid>`
    is faster than :func:`sc.fast_uuid() <fast_uuid>`!

    Args:
        which (str): the set of characters to choose from (default ascii)
        length (int): length of UID (default 6)
        n (int): number of UIDs to generate
        secure (bool): whether to generate random numbers from sources provided by the operating system
        forcelist (bool): whether or not to return a list even for a single UID (used for recursive calls)
        safety (float): ensure that the space of possible UIDs is at least this much larger than the number requested
        recursion (int): the recursion level of the call (since the function calls itself if not all UIDs are unique)
        recursion_limit (int): Maximum number of times to try regeneraring keys
        verbose (bool): whether to show progress

    Returns:
        uid (str or list): a string UID, or a list of string UIDs

    **Example**::

        uuids = sc.fast_uuid(n=100) # Generate 100 UUIDs

    Inspired by https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits/30038250#30038250
    """

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
    if secure: # pragma: no cover
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
    """
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
    """

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


def dcp(obj, die=True, memo=None):
    """
    Shortcut to perform a deep copy operation

    Almost identical to :func:`copy.deepcopy()`, but optionally fall back to :func:`copy.copy()`
    if deepcopy fails.

    Args:
        die (bool): if False, fall back to :func:`copy.copy()`

    | *New in version 2.0.0:* default die=True instead of False
    | *New in version 3.1.4:* die=False passed to sc.cp(); "verbose" argument removed; warning raised
    | *New in version 3.2.0:* "memo" argument
    """
    try:
        output = copy.deepcopy(obj, memo=memo)
    except Exception as E: # pragma: no cover
        errormsg = f'Warning: could not perform deep copy of {type(obj)}: {str(E)}'
        if die:
            raise ValueError(errormsg) from E
        else:
            output = cp(obj, die=False)
            warnmsg = errormsg + '\nPerforming shallow copy instead...'
            warnings.warn(warnmsg, category=RuntimeWarning, stacklevel=2)
    return output


def cp(obj, die=True):
    """
    Shortcut to perform a shallow copy operation

    Almost identical to :func:`copy.copy()`, but optionally allow failures

    | *New in version 2.0.0:* default die=True instead of False
    | *New in version 3.1.4:* "verbose" argument removed; warning raised
    """
    try:
        output = copy.copy(obj)
    except Exception as E:
        output = obj
        errormsg = f'Could not perform shallow copy of {type(obj)}'
        if die:
            raise ValueError(errormsg) from E
        else:
            warnmsg = errormsg + ', returning original object...'
            warnings.warn(warnmsg, category=RuntimeWarning, stacklevel=2)
    return output


def _printout(string=None, doprint=None, output=False):
    """
    Short for "print or output string". Not for public use.

    Private helper function to handle the logic of two boolean flags: by default, print;
    but if output is requested, return the output and do not print; unless doprint
    is set to True.
    """
    # Default for doprint is opposite of output
    if doprint is None: doprint = not(output)

    # Handle actions/output
    if doprint: print(string)
    if output: return string
    else:      return


def pp(obj, jsonify=False, doprint=None, output=False, sort_dicts=False, **kwargs):
    """
    Shortcut for pretty-printing the object.

    Almost identical to :func:`pprint.pprint()`, but can also be used as an alias for
    :func:`pprint.pformat()`.

    Args:
        obj     (any):  object to print
        jsonify (bool): whether to first convert the object to JSON, to handle things like ordered dicts nicely
        doprint (bool): whether to show output (default true)
        output  (bool): whether to return output as a string (default false)
        sort_dicts (bool): whether to sort dictionary keys (default false, unlike :func:`pprint.pprint()`)
        kwargs  (dict): passed to :func:`pprint.pprint()`
    
    **Example**::
        
        d = {'my very': {'large': 'and', 'unwieldy': {'nested': 'dictionary', 'cannot': 'be', 'easily': 'printed'}}}
        sc.pp(d)
        
    *New in version 1.3.1:* output argument
    *New in version 3.0.0:* "jsonify" defaults to False; sort_dicts defaults to False; removed "verbose" argument
    """

    # Get object
    if jsonify:
        try:
            toprint = json.loads(json.dumps(obj, default=lambda x: f'{x}')) # This is to handle things like OrderedDicts
        except Exception as E: # pragma: no cover
            print(f'Could not jsonify object ("{str(E)}"), printing default...')
            toprint = obj # If problems are encountered, just return the object
    else: # pragma: no cover
        toprint = obj

    # Decide what to do with object
    string = pprint.pformat(toprint, sort_dicts=sort_dicts, **kwargs)
    return _printout(string=string, doprint=doprint, output=output)


def sha(obj, digest=False, asint=False, encoding='utf-8'):
    """
    Shortcut for the standard hashing (SHA) method

    Equivalent to :obj:`hashlib.sha224()`.

    Args:
        obj (any): the object to be hashed; if not a string, converted to one
        digest (bool): if True, return the hex digest instead of the hash object
        asint (bool): if True, return the (very large) integer corresponding to the hex digest
        encoding (str): the encoding to use

    **Example**::

        sha1 = sc.sha(dict(foo=1, bar=2), True)
        sha2 = sc.sha(dict(foo=1, bar=2), digest=True)
        sha3 = sc.sha(dict(foo=1, bar=3), digest=True)
        assert sha1 == sha2
        assert sha2 != sha3
    
    | *New in version 3.2.0:* "asint" argument; changed argument order
    """
    # Prepare argument
    if not isinstance(obj, (str, bytes)): # Ensure it's actually a string
        obj = repr(obj) # More robust than str()
    if isinstance(obj, str): # If it's unicode, encode it to bytes first
        obj = obj.encode(encoding)
        
    # Do the hashing and prepare the output
    out = hashlib.sha224(obj)
    if digest or asint:
        out = out.hexdigest()
        if asint:
            out = int(out, 16) # Base-16 integer
    return out


def traceback(exc=None, value=None, tb=None, verbose=False, *args, **kwargs):
    """
    Shortcut for accessing the traceback

    Alias for :obj:`traceback.format_exc()`.
    
    If no argument is provided, then use the last exception encountered.
    
    Args:
        exc (Exception, tuple/list, or type): the exception to get the traceback from
        value (Exception): the actual exception
        tb (Traceback): the traceback
        verbose (bool): whether to print the exception
    
    **Examples**::
        
        # Use automatic exception info
        mylist = [0,1]
        try:
            mylist[2]
        except:
            print(f'Error: {sc.traceback()}')
        
        # Supply exception manually (also illustrating sc.tryexcept())
        with sc.tryexcept() as te1:
            dict(a=3)['b']

        with sc.tryexcept() as te2:
            [0,1][2]

        tb1 = sc.traceback(te1.exception)
        tb2 = sc.traceback(te2.exception)
        print(f'Tracebacks were:\n\n{tb1}\n{tb2}')
    """
    if exc is not None:
        if isinstance(exc, Exception): # Usual case: an exception is supplied
            exc_info = (exc.__class__, exc, exc.__traceback__)
        elif isinstance(exc, (tuple, list)): # Alternately, all three are supplied as a list # pragma: no cover
            exc_info = exc
        elif value is not None and tb is not None: # ... or separately # pragma: no cover
            exc_info = (exc, value, tb)
        else: # pragma: no cover
            errormsg = f'Unexpected exception type "{type(exc)}": expecting Exception, or list/tuple of exc_info'
            raise TypeError(errormsg)
        out = ''.join(py_traceback.format_exception(*exc_info, **kwargs))
    else:
        out = py_traceback.format_exc(*args, **kwargs)
    if verbose: # pragma: no cover
        print(out)
    return out


def getuser():
    """
    Get the current username 
    
    Alias to :func:`getpass.getuser()` -- see https://docs.python.org/3/library/getpass.html#getpass.getuser
    
    **Example**::
        
        sc.getuser()
    
    *New in version 3.0.0.*
    """
    return getpass.getuser()


def getplatform(expected=None, platform=None, die=False):
    """
    Return the name of the current "main" platform (e.g. 'mac')
    
    Alias to ``sys.platform``, except maps entries onto one of 'linux', 'windows', 
    'mac', or 'other'.

    Args:
        expected (str): if not None, check if the current platform is this
        platform (str): if supplied, map this onto one of the "main" platforms, rather than determine it
        die (bool): if True and expected is defined, raise an exception
    
    Returns:
        String, one of: 'linux', 'windows', 'mac', or 'other'

    **Examples**::

        sc.getplatform() # Get current name of platform
        sc.getplatform('windows', die=True) # Raise an exception if not on Windows
        sc.getplatform(platform='darwin') # Normalize to 'mac'
    """
    # Define different aliases for each operating system
    mapping = dict(
        linux   = ['linux', 'posix'],
        windows = ['windows', 'win', 'win32', 'cygwin', 'nt'],
        mac     = ['mac', 'macos', 'darwin', 'osx']
    )

    # Check to see what system it is
    sys_plat = sys.platform if platform is None else platform
    plat = 'other'
    for key,aliases in mapping.items():
        if sys_plat.lower() in aliases:
            plat = key
            break

    # Handle output
    if expected is not None:
        output = (expected.lower() in mapping[plat]) # Check if it's as expecte
        if not output and die: # pragma: no cover
            errormsg = f'System is "{plat}", not "{expected}"'
            raise EnvironmentError(errormsg)
    else: # pragma: no cover
        output = plat
    return output


def iswindows(die=False):
    """ Alias to :func:`sc.getplatform('windows') <getplatform>` """
    return getplatform('windows', die=die)

def islinux(die=False):
    """ Alias to :func:`sc.getplatform('linux') <getplatform>` """
    return getplatform('linux', die=die)

def ismac(die=False):
    """ Alias to :func:`sc.getplatform('mac') <getplatform>` """
    return getplatform('mac', die=die)


def isjupyter(detailed=False):
    """
    Check if a command is running inside a Jupyter notebook.
    
    Returns true/false if detailed=False, or a string for the exact type of notebook
    (e.g., Google Colab) if detailed=True.
    
    Args:
        detailed (bool): return a string of IPython/Jupyter type instead of true/false
        verbose (bool): print out additional information if IPython can't be imported
        
    **Examples**::
        
        if sc.isjupyter():
            sc.options(jupyter=True)
        
        if sc.isjupyter(detailed=True) == 'colab':
            print('You are running on Google Colab')
    
    *New in version 3.0.0.*
    """
    # First check if we can import it
    output = None
    is_jupyter = False
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except Exception as E: # pragma: no cover
        output = f'Python (Jupyter/IPython is not installed) ({E})'
        return output if detailed else is_jupyter
    
    # It's not IPython
    if ipython is None:
        output = 'Python (Jupyter/IPython is installed but not running)'
        return output if detailed else is_jupyter

    # It is IPython, keep checking
    else: # pragma: no cover
    
        # Get the modules and classes
        classes = inspect.getmro(ipython.__class__)
        names = [str(c).lower() for c in classes] # Names as a list
        firstname = names[0]
        allnames = ', '.join(names)
    
        # Define strings to look for
        mapping = dict(
            jupyter = 'zmqinteractive',
            ipython = 'terminalinteractive',
            spyder  = 'spyder',
            colab   = 'colab',
        )
        
        # First check if it's Jupyter, and return if it is
        if not detailed:
            if mapping['jupyter'] in allnames and mapping['spyder'] not in allnames: # Spyder inherits from ZMQ, unlike e.g. Colab shell
                is_jupyter = True
            return is_jupyter
        
        # Otherwise, get the full name
        else:
            if   mapping['spyder']  in firstname: output = 'Spyder'
            elif mapping['colab']   in firstname: output = 'Google Colab'
            elif mapping['ipython'] in firstname: output = 'IPython'
            elif mapping['jupyter'] in firstname: output = 'Jupyter'
            else:
                output = names[0]
            
            return output


def asciify(string, form='NFKD', encoding='ascii', errors='ignore', **kwargs):
    """
    Convert an arbitrary Unicode string to ASCII.
    
    Args:
        form (str): the type of Unicode normalization to use
        encoding (str): the output string to encode to
        errors (str): how to handle errors
        kwargs (dict): passed to :meth:`string.decode()`
    
    **Example**::
        sc.asciify('föö→λ ∈ ℝ') # Returns 'foo  R'
    
    *New in version 2.0.1.*
    """
    normalized = unicodedata.normalize(form, string) # First, normalize Unicode encoding
    encoded = normalized.encode(encoding, errors) # Then, convert to ASCII
    decoded = encoded.decode(**kwargs) # Finally, decode back to utf-8
    return decoded

##############################################################################
#%% Web/HTML functions
##############################################################################

__all__ += ['urlopen', 'wget', 'download', 'htmlify']

def urlopen(url, filename=None, save=None, headers=None, params=None, data=None,
            prefix='http', convert=True, die=False, response='text', verbose=False):
    """
    Download a single URL.

    Alias to ``urllib.request.urlopen(url).read()``. See also :func:`sc.download() <download>`
    for downloading multiple URLs. Note: :func:`sc.urlopen() <urlopen>`/:func:`sc.wget() <wget>` are aliases.

    Args:
        url (str): the URL to open, either as GET or POST
        filename (str): if supplied, save to file instead of returning output
        save (bool): if supplied instead of ``filename``, then use the default filename
        headers (dict): a dictionary of headers to pass
        params (dict): a dictionary of parameters to pass to the GET request
        data (dict) a dictionary of parameters to pass to a POST request
        prefix (str): the string to ensure the URL starts with (else, add it)
        convert (bool): whether to convert from bytes to string
        die (bool): whether to raise an exception if converting to text failed
        response (str): what to return: 'text' (default), 'json' (dictionary version of the data), 'status' (the HTTP status), or 'full' (the full response object)
        verbose (bool): whether to print progress

    **Examples**::

        html = sc.urlopen('wikipedia.org') # Retrieve into variable html
        sc.urlopen('http://wikipedia.org', filename='wikipedia.html') # Save to file wikipedia.html
        sc.urlopen('https://wikipedia.org', save=True, headers={'User-Agent':'Custom agent'}) # Save to the default filename (here, wikipedia.org), with headers
        sc.urlopen('wikipedia.org', response='status') # Only return the HTTP status of the site

    | *New in version 2.0.0:* renamed from ``wget`` to ``urlopen``; new arguments
    | *New in version 2.0.1:* creates folders by default if they do not exist
    | *New in version 2.0.4:* "prefix" argument, e.g. prepend "http://" if not present
    | *New in version 3.1.4:* renamed "return_response" to "response"; additional options
    """
    from urllib import request as ur # Need to import these directly, not via urllib
    from urllib import parse as up

    T = sc.timer()

    # Handle headers
    default_headers = {
        'User-Agent': 'Python/Sciris', # Some URLs require this
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8', # Default Chrome/Safari header
    }
    headers = mergedicts(default_headers, headers)

    # Handle parameters and data
    full_url = url
    if prefix is not None:
        if not full_url.startswith(prefix):
            full_url = prefix + '://' + full_url # Leaves https alone, but adds http:// otherwise
    if params is not None: # pragma: no cover
        full_url = full_url + '?' + up.urlencode(params)
    if data is not None: # pragma: no cover
        data = up.urlencode(data).encode(encoding='utf-8', errors='ignore')

    if verbose: print(f'Downloading {url}...')
    request = ur.Request(full_url, headers=headers, data=data)
    resp = ur.urlopen(request) # Actually open the URL
    if response in ['text', 'json']:
        output = resp.read()
        if response == 'json':
            try:
                output = sc.loadjson(string=output)
            except Exception as E:
                errormsg = f'Could not convert HTTP response beginning "{output[:20]}" to JSON'
                raise ValueError(errormsg) from E
        elif convert:
            if verbose>1: print('Converting from bytes to text...')
            try:
                output = output.decode()
            except Exception as E: # pragma: no cover
                if die:
                    raise E
                elif verbose:
                    errormsg = f'Could not decode to text: {str(E)}'
                    print(errormsg)
    elif response == 'status':
        output = resp.status
    elif response == 'full':
        output = resp
    else:
        errormsg = f'Unrecognized option "{response}"; must be one of "text", "json", "status", or "full"'
        raise ValueError(errormsg)

    # Set filename -- from https://stackoverflow.com/questions/31804799/how-to-get-pdf-filename-with-python-requests
    if filename is None and save: # pragma: no cover
        headers = dict(resp.getheaders())
        string = "Content-Disposition"
        if string in headers.keys():
            filename = re.findall("filename=(.+)", headers[string])[0]
        else:
            filename = url.rstrip('/').split('/')[-1] # Remove trailing /, then pull out the last chunk

    if filename is not None and save is not False:
        if verbose: print(f'Saving to {filename}...')
        filename = sc.makefilepath(filename, makedirs=True)
        if isinstance(output, bytes): # Raw HTML data
            with open(filename, 'wb') as f: # pragma: no cover
                f.write(output)
        elif isinstance(output, dict): # If it's a JSON
            sc.savejson(filename, output)
        else: # Other text
            with open(filename, 'w', encoding='utf-8') as f: # Explicit encoding to avoid issues on Windows
                f.write(output)
        output = filename

    if verbose:
        T.toc(f'Time to download {url}')

    return output

# Alias for backwards compatibility
wget = urlopen


def download(url, *args, filename=None, save=True, parallel=True, die=True, verbose=True, **kwargs):
    """
    Download one or more URLs in parallel and return output or save them to disk.

    A parallelized wrapper for :func:`sc.urlopen() <urlopen>`, except with ``save=True`` by default.

    Args:
        url (str/list/dict): either a single URL, a list of URLs, or a dict of key:URL or filename:URL pairs
        *args (list): additional URLs to download
        filename (str/list): either a string or a list of the same length as ``url`` (if not supplied, return output)
        save (bool): if supplied instead of ``filename``, then use the default filename
        parallel (bool): whether to download multiple URLs in parallel
        die (bool): whether to raise an exception if a URL can't be retrieved (default true)
        verbose (bool): whether to print progress (if verbose=2, print extra detail on each downloaded URL)
        **kwargs (dict): passed to :func:`sc.urlopen() <urlopen>`

    **Examples**::

        html = sc.download('http://sciris.org') # Download a single URL
        data = sc.download('http://sciris.org', 'http://covasim.org', save=False) # Download two in parallel
        sc.download({'sciris.html':'http://sciris.org', 'covasim.html':'http://covasim.org'}) # Download two and save to disk
        sc.download(['http://sciris.org', 'http://covasim.org'], filename=['sciris.html', 'covasim.html']) # Ditto
        data = sc.download(dict(sciris='http://sciris.org', covasim='http://covasim.org'), save=False) # Download and store in memory


    | *New in version 2.0.0.*
    | *New in version 3.0.0:* "die" argument
    | *New in version 3.1.1:* default order switched from URL:filename to filename:URL pairs
    | *New in version 3.1.3:* output as objdict instead of odict
    """
    T = sc.timer()

    # Parse arguments
    if isinstance(url, dict):
        keys = list(url.keys())
        urls = list(url.values())
        if any(['http' in key for key in keys]) and not any(['http' in url for url in urls]):
            keys,urls = urls,keys # Swap order, expect http to be in the URLs but not the keys
    else:
        keys = mergelists(filename)
        urls = mergelists(url, *args)
        
    # Ensure consistency
    n_urls = len(urls)
    n_keys = len(keys)
    if not n_keys:
        if save: # Filenames generated by urlopen(), pass None
            keys = [None]*n_urls
        else: # Otherwise, turn the URL into a filename
            keys = [url.split('/')[-1] for url in urls] # Turn e.g. 'http://mysite.com/tree/index.html' into 'index.html'
    elif n_keys != n_urls: # pragma: no cover
        errormsg = f'Cannot process {n_urls} URLs and {n_keys} filenames'
        raise ValueError(errormsg)

    if verbose and n_urls > 1:
        print(f'Downloading {n_urls} URLs...')

    # Get results in parallel
    wget_verbose = (verbose>1) or (verbose and n_urls == 1) # By default, don't print progress on each download
    iterkwargs = dict(url=urls, filename=keys)
    func_kwargs = mergedicts(dict(save=save, verbose=wget_verbose), kwargs)
    if n_urls > 1 and parallel:
        outputs = sc.parallelize(urlopen, iterkwargs=iterkwargs, kwargs=func_kwargs, parallelizer='thread', die=die)
    else:
        outputs = []
        for key,url in zip(keys, urls):
            try:
                output = urlopen(url=url, filename=key, **func_kwargs)
            except Exception as E: # pragma: no cover
                if die:
                    raise E
                else:
                    warnmsg = f'Could not download {url}: {E}'
                    warnings.warn(warnmsg, category=RuntimeWarning, stacklevel=2)
                    output = E
                    
            outputs.append(output)
    
    # If we're returning the data rather than saving the files, convert to an odict
    if not save:
        outputs = sc.objdict({k:v for k,v in zip(keys, outputs)})

    if verbose and n_urls > 1:
        T.toc(f'Time to download {n_urls} URLs')
    if n_urls == 1:
        outputs = outputs[0]

    return outputs


def htmlify(string, reverse=False, tostring=False):
    """
    Convert a string to its HTML representation by converting unicode characters,
    characters that need to be escaped, and newlines. If reverse=True, will convert
    HTML to string. If tostring=True, will convert the bytestring back to Unicode.

    **Examples**::

        output = sc.htmlify('foo&\\nbar') # Returns b'foo&amp;<br>bar'
        output = sc.htmlify('föö&\\nbar', tostring=True) # Returns 'f&#246;&#246;&amp;&nbsp;&nbsp;&nbsp;&nbsp;bar'
        output = sc.htmlify('foo&amp;<br>bar', reverse=True) # Returns 'foo&\\nbar'
    """
    import html
    if not reverse: # Convert to HTML
        output = html.escape(string).encode('ascii', 'xmlcharrefreplace') # Replace non-ASCII characters
        output = output.replace(b'\n', b'<br>') # Replace newlines with <br>
        output = output.replace(b'\t', b'&nbsp;&nbsp;&nbsp;&nbsp;') # Replace tabs with 4 spaces
        if tostring: # Convert from bytestring to unicode # pragma: no cover
            output = output.decode()
    else: # Convert from HTML
        output = html.unescape(string)
        output = output.replace('<br>','\n').replace('<br />','\n').replace('<BR>','\n')
    return output


##############################################################################
#%% Type functions
##############################################################################

__all__ += ['flexstr', 'sanitizestr', 'isiterable', 'checktype', 'isnumber', 'isstring', 'isarray', 'isfunc',
            'toarray', 'tolist', 'promotetoarray', 'promotetolist', 'transposelist',
            'swapdict', 'mergedicts', 'mergelists', 'ifelse']

def flexstr(arg, *args, force=True, join=''):
    """
    Try converting any object to a "regular" string (i.e. ``str``), but proceed
    if it fails. Note: this function calls ``repr()`` rather than ``str()`` to
    ensure a more robust representation of objects.
    
    Args:
        arg (any): the object to convert to a string
        args (list): additional arguments
        force (bool): whether to force it to be a string
        join (str): if multiple arguments are provided, the character to use to join
    
    **Example**::
        
        sc.flexstr(b'foo', 'bar', [1,2]) # Returns 'foobar[1, 2]'
    
    *New in version 3.0.0:* handle multiple inputs
    """
    arglist = mergelists(arg, list(args))
    outlist = []
    for arg in arglist:
        if isinstance(arg, str):
            output = arg
        elif isinstance(arg, bytes):
            try:
                output = arg.decode() # If it's bytes, decode to unicode
            except: # pragma: no cover
                if force: output = repr(arg) # If that fails, just print its representation
                else:     output = arg
        else: # pragma: no cover
            if force: output = repr(arg)
            else:     output = arg # Optionally don't do anything for non-strings
        outlist.append(output)
    
    if len(outlist) == 1:
        outstr = outlist[0]
    else: # pragma: no cover
        if force:
            outstr = join.join(outlist)
        else:
            outstr = outlist
    
    return outstr


def sanitizestr(string=None, alphanumeric=False, nospaces=False, asciify=False, 
                lower=False, validvariable=False, spacechar='_', symchar='?'):
    """
    Remove all non-"standard" characters from a string
    
    Can be used to e.g. generate a valid variable name from arbitrary input, remove
    non-ASCII characters (replacing with equivalent ASCII ones if possible), etc.
    
    Args:
        string (str): the string to sanitize
        alphanumeric (bool): allow only alphanumeric characters
        nospaces (bool): remove spaces
        asciify (bool): remove non-ASCII characters
        lower (bool): convert uppercase characters to lowercase
        validvariable (bool): convert to a valid Python variable name (similar to alphanumeric=True, nospaces=True; uses spacechar to substitute)
        spacechar (str): if nospaces is True, character to replace spaces with (can be blank)
        symchar (str): character to replace non-alphanumeric characters with (can be blank)
    
    **Examples**::
        
        string1 = 'This Is a String'
        sc.sanitizestr(string1, lower=True) # Returns 'this is a string'
        
        string2 = 'Lukáš wanted €500‽'
        sc.sanitizestr(string2, asciify=True, nospaces=True, symchar='*') # Returns 'Lukas_wanted_*500*'
        
        string3 = '"Ψ scattering", María said, "at ≤5 μm?"'
        sc.sanitizestr(string3, asciify=True, alphanumeric=True, nospaces=True, spacechar='') # Returns '??scattering??Mariasaid??at?5?m??'
    
        string4 = '4 path/names/to variable!'
        sc.sanitizestr(string4, validvariable=True, spacechar='') # Returns '_4pathnamestovariable'
    
    *New in version 3.0.0.*
    """
    string = flexstr(string)
    if asciify:
        newstr = ''
        for char in string:
            newchar = unicodedata.normalize('NFKD', char).encode('ascii', 'ignore').decode()
            if not len(newchar):
                newchar = symchar
            newstr += newchar
        string = newstr
    if nospaces:
        string = string.replace(' ', spacechar)
    if lower:
        string = string.lower()
    if alphanumeric:
        space = spacechar if nospaces else ' '
        string = re.sub(f'[^0-9a-zA-Z{space}]', symchar, string) # If not for the space string, could use /w
    if validvariable:
        string = re.sub(r'\W', spacechar, string) # /W matches an non-alphanumeric character; use a raw string to avoid warnings
        if not string or str.isdecimal(string[0]): # Don't allow leading decimals: here in case spacechar is None
            string = '_' + string
    return string


def isiterable(obj, *args, exclude=None, minlen=None):
    """
    Determine whether or not the input is iterable, with optional types to exclude.
    
    Args:
        obj (any): object to check for iterability
        args (any): additional objects to check for iterability
        exclude (list): list of iterable objects to exclude (e.g., strings)
        minlen (int): if not None, check that an object has a defined length as well
    
    **Examples**::
        
        obj1 = [1,2,3]
        obj2 = 'abc'
        obj3 = set()
        
        sc.isiterable(obj1) # Returns True
        sc.isiterable(obj1, obj2, obj3, exclude=str, minlen=1) # returns [True, False, False]
        
    See also :func:`numpy.iterable()` for a simpler version.
    
    *New in version 3.0.0:* "exclude" and "minlen" args; support multiple arguments
    """
    
    # Handle arguments
    objlist = [obj]
    n_args = len(args)
    exclude = tuple(tolist(exclude))
    if n_args: # pragma: no cover
        objlist.extend(args)
        
    # Determine iterability
    output = []
    for obj in objlist:
        
        # Basic test of iterability
        tf = np.iterable(obj)
        
        # Additional checks
        if tf:
            
            # Explicitly exclude types
            if exclude and checktype(obj, exclude): 
                tf = False
            
            # Check length
            if minlen is not None: # pragma: no cover
                try:
                    assert len(obj) >= minlen
                except:
                    tf = False
                    
        output.append(tf)
    
    if not n_args:
        output = output[0]
    
    return output


def checktype(obj=None, objtype=None, subtype=None, die=False):
    """
    A convenience function for checking instances. If objtype is a type,
    then this function works exactly like isinstance(). But, it can also
    be one of the following strings:

        - 'str', 'string': string or bytes object
        - 'num', 'number': any kind of number
        - 'arr', 'array': a Numpy array (equivalent to np.ndarray)
        - 'listlike': a list, tuple, or array
        - 'arraylike': a list, tuple, or array with numeric entries
        - 'none': a None object

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
    
    | *New in version 2.0.1:* ``pd.Series`` considered 'array-like'
    | *New in version 3.0.0:* allow list (in addition to tuple) of types; allow checking for NoneType
    | *New in version 3.1.3:* handle exceptions when casting to "arraylike"
    """

    # Handle "objtype" input
    if isinstance(objtype, str): # Ensure it's lowercase (e.g. 'None' → 'none')
        objtype = objtype.lower() 
    if   objtype in ['none']:                  objinstance = type(None) # NoneType not available in Python <3.10
    elif objtype in ['str','string']:          objinstance = _stringtypes
    elif objtype in ['num', 'number']:         objinstance = _numtype
    elif objtype in ['bool', 'boolean']:       objinstance = _booltypes
    elif objtype in ['arr', 'array']:          objinstance = np.ndarray
    elif objtype in ['listlike', 'arraylike']: objinstance = (list, tuple, np.ndarray, pd.Series) # Anything suitable as a numerical array
    elif type(objtype) == type:                objinstance = objtype # Don't need to do anything
    elif isinstance(objtype, tuple):           objinstance = objtype # Ditto
    elif isinstance(objtype, list):            objinstance = tuple(objtype) # Convert from a list to a tuple # pragma: no cover
    elif objtype is None: # pragma: no cover
        errormsg = "No object type was supplied; did you mean to use objtype='none' instead?"
        raise ValueError(errormsg)
    else: # pragma: no cover
        errormsg = f'Could not understand what type you want to check: should be either a string or a type, not "{objtype}"'
        raise ValueError(errormsg)

    # Do first-round checking
    result = isinstance(obj, objinstance)

    # Do second round checking
    if result and objtype == 'arraylike': # Special case for handling arrays which may be multi-dimensional
        try:
            obj = toarray(obj).flatten() # Flatten all elements
        except: # If it can't be cast to an array, it's not array-like
            result = False
        if subtype is None: # Add additional check for numeric entries
            subtype = _numtype + _booltypes
    if isiterable(obj) and subtype is not None:
        for item in obj:
            result = result and checktype(item, subtype)

    # Decide what to do with the information thus gleaned
    if die: # Either raise an exception or do nothing if die is True
        if not result: # It's not an instance
            errormsg = f'Incorrect type: object is {type(obj)}, but {objtype} is required'
            raise TypeError(errormsg)
        else:
            return # It's fine, do nothing
    else: # Return the result of the comparison
        return result


def isnumber(obj, isnan=None):
    """
    Determine whether or not the input is a number.
    
    Identical to isinstance(obj, numbers.Number) unless isnan is specified.

    Args:
        obj (any): the object to check if it's a number
        isnan (bool): an optional additional check to determine whether the number is/isn't NaN
    
    | *New in version 3.2.0:* use ``isinstance()`` directly
    """
    output = isinstance(obj, _numtype)
    if output and isnan is not None: # It is a number, so can check for nan # pragma: no cover
        output = (np.isnan(obj) == isnan) # See if they match
    return output


def isstring(obj):
    """
    Determine whether or not the input is string-like (i.e., str or bytes).

    Equivalent to ``isinstance(obj, (str, bytes))``
    """
    return checktype(obj, 'string')


def isarray(obj, dtype=None):
    """
    Check whether something is a Numpy array, and optionally check the dtype.

    Almost the same as ``isinstance(obj, np.ndarray)``.

    **Example**::

        sc.isarray(np.array([1,2,3]), dtype=float) # False, dtype is int

    | *New in version 1.0.0.*
    | *New in version 3.1.6:* explicit False return
    """
    if isinstance(obj, np.ndarray):
        if dtype is None:
            return True
        else:
            if obj.dtype == dtype: # pragma: no cover
                return True
            else:
                return False
    return False


def isfunc(obj):
    """
    Quickly check if something is a function.
    
    | *New in version 3.2.0.*    
    """
    return isinstance(obj, (types.MethodType, types.FunctionType))


def toarray(x, keepnone=False, asobject=True, dtype=None, **kwargs):
    """
    Small function to ensure consistent format for things that should be arrays
    (note: :func:`sc.toarray() <toarray>` and :func:`sc.promotetoarray() <promotetoarray>` are identical).

    Very similar to :func:`numpy.array`, with the main difference being that :func:`sc.toarray(3) <toarray>`
    will return ``np.array([3])`` (i.e. a 1-d array that can be iterated over), while
    ``np.array(3)`` will return a 0-d array that can't be iterated over.

    Args:
        x (any): a number or list of numbers
        keepnone (bool): whether ``sc.toarray(None)`` should return ``np.array([])`` or ``np.array([None], dtype=object)``
        asobject (bool): whether to prefer to coerce arrays to object type rather than string
        kwargs (dict): passed to :func:`numpy.array()`

    **Examples**::

        sc.toarray(5) # Returns np.array([5])
        sc.toarray([3,5]) # Returns np.array([3,5])
        sc.toarray(None, skipnone=True) # Returns np.array([])
        sc.toarray([1, 'foo']) # Returns np.array([1, 'foo'], dtype=object)

    | *New in version 1.1.0:* replaced "skipnone" with "keepnone"; allowed passing kwargs to ``np.array()``.
    | *New in version 2.0.1:* added support for pandas Series and DataFrame
    | *New in version 3.1.0:* "asobject" argument; cast mixed-type arrays to object rather than string by default 
    """
    # Handle None
    skipnone = kwargs.pop('skipnone', None)
    if skipnone is not None: # pragma: no cover
        keepnone = not(skipnone)
        warnmsg = 'sc.toarray() argument "skipnone" has been deprecated as of v1.1.0; use keepnone instead'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)
    
    # Handle different inputs
    if isnumber(x) or (isinstance(x, np.ndarray) and not np.shape(x)): # e.g. 3 or np.array(3)
        x = [x]
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    elif x is None and not keepnone:
        x = []
    
    # Actually convert to an array
    output = np.array(x, dtype=dtype, **kwargs)
    if asobject and (dtype is None) and issubclass(output.dtype.type, np.str_):
        output = np.array(x, dtype=object, **kwargs) # Cast to object instead of string
    
    return output


def tolist(obj=None, objtype=None, keepnone=False, coerce='default'):
    """
    Make sure object is always a list (note: :func:`sc.tolist() <tolist>`/:func:`sc.promotetolist() <promotetolist>` are identical).

    Used so functions can handle inputs like ``'a'``  or ``['a', 'b']``. In other
    words, if an argument can either be a single thing (e.g., a single dict key)
    or a list (e.g., a list of dict keys), this function can be used to do the
    conversion, so it's always safe to iterate over the output.

    While this usually wraps objects in a list rather than converts them to a list,
    the "coerce" argument can be used to change this behavior. Options are:

    - 'none' or None: do not coerce
    - 'default': coerce objects that were lists in Python 2 (range, map, dict_keys, dict_values, dict_items)
    - 'tuple': all the types in default, plus tuples
    - 'full': all the types in default, plus tuples and arrays

    Args:
        obj (anything): object to ensure is a list
        objtype (anything): optional type to check for each element; see :func:`sc.checktype() <checktype>` for details
        keepnone (bool): if ``keepnone`` is false, then ``None`` is converted to ``[]``; else, it's converted to ``[None]``
        coerce (str/tuple):  tuple of additional types to coerce to a list (as opposed to wrapping in a list)

    See also :func:`sc.mergelists() <mergelists>` to handle multiple input arguments.

    **Examples**::

        sc.tolist(5) # Returns [5]
        sc.tolist(np.array([3,5])) # Returns [np.array([3,5])] -- not [3,5]!
        sc.tolist(np.array([3,5]), coerce=np.ndarray) # Returns [3,5], since arrays are coerced to lists
        sc.tolist(None) # Returns []
        sc.tolist(range(3)) # Returns [0,1,2] since range is coerced by default
        sc.tolist(['a', 'b', 'c'], objtype='number') # Raises exception

        def myfunc(data, keys):
            keys = sc.tolist(keys)
            for key in keys:
                print(data[key])

        data = {'a':[1,2,3], 'b':[4,5,6]}
        myfunc(data, keys=['a', 'b']) # Works
        myfunc(data, keys='a') # Still works, equivalent to needing to supply keys=['a'] without tolist()

    | *New in version 1.1.0:* "coerce" argument
    | *New in version 1.2.2:* default coerce values
    | *New in version 2.0.2:* tuple coersion
    """
    # Handle coerce
    default_coerce = (range, map, type({}.keys()), type({}.values()), type({}.items()))
    if isinstance(coerce, str):
        if coerce == 'none':
            coerce = None
        elif coerce == 'default':
            coerce = default_coerce
        elif coerce == 'tuple': # pragma: no cover
            coerce = default_coerce + (tuple,)
        elif coerce == 'array': # pragma: no cover
            coerce = default_coerce + (np.ndarray,)
        elif coerce == 'full':
            coerce = default_coerce + (tuple, np.ndarray)
        else: # pragma: no cover
            errormsg = f'Option "{coerce}"; not recognized; must be "none", "default", "tuple", or "full"'

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
                errormsg = f'tolist(): type mismatch, expecting type {objtype}'
                raise TypeError(errormsg) from E
    return output


# Aliases for core functions
promotetoarray = toarray
promotetolist = tolist


def transposelist(obj, fix_uneven=True):
    """
    Convert e.g. a list of key-value tuples into a list of keys and a list of values.
    
    Args:
        obj (list): the list-of-lists to be transposed
        fix_uneven (bool): append None values where needed so all input lists have the same length

    **Examples**::

        o = sc.odict(a=1, b=4, c=9, d=16)
        itemlist = o.enumitems()
        inds, keys, vals = sc.transposelist(itemlist)
        
        listoflists = [
            ['a', 1, 3],
            ['b', 4, 5],
            ['c', 7, 8, 9, 10]
        ]
        trans = sc.transposelist(listoflists, fix_uneven=True)

    *New in version 1.1.0.*
    """
    if fix_uneven:
        maxlen = max([len(ls) for ls in obj])
        newobj = [] # Create a manual copy
        for ls in obj:
            row = [item for item in ls] + [None]*(maxlen - len(ls))
            newobj.append(row)
        obj = newobj
        
    return list(map(list, zip(*obj)))
        


def swapdict(d):
    """
    Swap the keys and values of a dictionary. Equivalent to {v:k for k,v in d.items()}

    Args:
        d (dict): dictionary

    **Example**::
        d1 = {'a':'foo', 'b':'bar'}
        d2 = sc.swapdict(d1) # Returns {'foo':'a', 'bar':'b'}

    *New in version 1.3.0.*
    """
    if not isinstance(d, dict):
        errormsg = f'Not a dictionary: {type(d)}'
        raise TypeError(errormsg)
    try:
        output = {v:k for k,v in d.items()}
    except Exception as E:
        exc = type(E)
        errormsg = 'Could not swap keys and values: ensure all values are of hashable type'
        raise exc(errormsg) from E
    return output


def mergedicts(*args, _strict=False, _overwrite=True, _copy=False, _sameclass=True, _die=True, **kwargs):
    """
    Small function to merge multiple dicts together.
    
    By default, skips any input arguments that are ``None``, and allows keys to be set 
    multiple times. This function is similar to dict.update(), except it returns a value.
    The first dictionary supplied will be used for the output type (e.g. if the
    first dictionary is an odict, an odict will be returned).

    Note that arguments start with underscores to avoid possible collisions with
    keywords (e.g. :func:`sc.mergedicts(dict(loose=True, strict=True), strict=False, _strict=True) <mergedicts>`).

    **Note**: This function is similar to the "|" operator introduced in Python 3.9.
    However, ``sc.mergedicts()`` is useful for cases such as function arguments, where 
    the default option is ``None`` but you will need a dict later on.

    Args:
        _strict    (bool): if True, raise an exception if an argument isn't a dict
        _overwrite (bool): if False, raise an exception if multiple keys are found
        _copy      (bool): whether or not to deepcopy the merged dictionary
        _sameclass (bool): whether to ensure the output has the same type as the first dictionary merged
        _die       (bool): whether to raise an exception if something goes wrong
        *args      (list): the sequence of dicts to be merged
        **kwargs   (dict): merge these into the dict as well

    **Examples**::

        d0 = sc.mergedicts(user_args) # Useful if user_args might be None, but d0 is always a dict
        d1 = sc.mergedicts({'a':1}, {'b':2}) # Returns {'a':1, 'b':2}
        d2 = sc.mergedicts({'a':1, 'b':2}, {'b':3, 'c':4}, None) # Returns {'a':1, 'b':3, 'c':4}
        d3 = sc.mergedicts(sc.odict({'b':3, 'c':4}), {'a':1, 'b':2}) # Returns sc.odict({'b':2, 'c':4, 'a':1})
        d4 = sc.mergedicts({'b':3, 'c':4}, {'a':1, 'b':2}, _overwrite=False) # Raises exception

    | *New in version 1.1.0:* "copy" argument
    | *New in version 1.3.3:* keywords allowed
    | *New in version 2.0.0:* keywords fully enabled; "_sameclass" argument
    | *New in version 2.0.1:* fixed bug with "_copy" argument
    """
    # Warn about deprecated keys
    renamed = ['strict', 'overwrite', 'copy']
    if any([k in kwargs for k in renamed]): # pragma: no cover
        warnmsg = f'sc.mergedicts() arguments "{strjoin(renamed)}" have been renamed with underscores as of v1.3.3; using these as keywords is undesirable'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)

    # Try to get the output type from the arguments, but revert to a standard dict if that fails
    outputdict = {}
    if _sameclass:
        for arg in args:
            if isinstance(arg, dict):
                try:
                    outputdict = arg.__class__() # This creates a new instance of the class
                    break
                except Exception as E: # pragma: no cover
                    errormsg = f'Could not create new dict of {type(args[0])} from first argument ({str(E)}); set _sameclass=False if this is OK'
                    if _die:
                        raise TypeError(errormsg) from E
                    else:
                        warnings.warn(errormsg, category=UserWarning, stacklevel=2)

    # Merge over the dictionaries in order
    args = list(args)
    args.append(kwargs) # Include any kwargs as the final dict
    for a,arg in enumerate(args):
        is_dict = isinstance(arg, dict)
        if _strict and not is_dict:
            errormsg = f'Argument {a} has {type(arg)}; must be dict since _strict=True'
            raise TypeError(errormsg)
        if is_dict:
            if not _overwrite:
                intersection = set(outputdict.keys()).intersection(arg.keys())
                if len(intersection):
                    errormsg = f'Could not merge dicts since keys "{strjoin(intersection)}" overlap and _overwrite=False'
                    raise KeyError(errormsg)
            outputdict.update(arg)
        else:
            if arg is not None and _die: # pragma: no cover
                errormsg = f'Could not handle argument {a} of {type(arg)}: expecting dict or None'
                raise TypeError(errormsg)

    if _copy:
        outputdict = dcp(outputdict, die=_die)
    return outputdict


def mergelists(*args, coerce='default', copy=False, **kwargs):
    """
    Merge multiple lists together.
    
    Often used to flexible handle the input arguments to functions; see example
    below.

    Args:
        args (any): the lists, or items, to be joined together into a list
        coerce (str): what types of objects to treat as lists; see :func:`sc.tolist() <tolist>` for details
        copy (bool): whether to deepcopy the resultant object
        kwargs (dict): passed to :func:`sc.tolist() <tolist>`, which is called on each argument

    **Examples**::

        # Simple usage
        sc.mergelists(None)                                # Returns []
        sc.mergelists([1,2,3], [4,5,6])                    # Returns [1, 2, 3, 4, 5, 6]
        sc.mergelists([1,2,3], 4, 5, 6)                    # Returns [1, 2, 3, 4, 5, 6]
        sc.mergelists([(1,2), (3,4)], (5,6))               # Returns [(1, 2), (3, 4), (5, 6)]
        sc.mergelists((1,2), (3,4), (5,6))                 # Returns [(1, 2), (3, 4), (5, 6)]
        sc.mergelists((1,2), (3,4), (5,6), coerce='tuple') # Returns [1, 2, 3, 4, 5, 6]
        
        # Usage for handling flexible input arguments
        def my_func(arg=None, *args):
            arglist = sc.mergelists(arg, list(args))
            return arglist

        a = my_func()           # Returns []
        b = my_func([1,2,3])    # Returns [1,2,3]
        c = my_func(1,2,3)      # Returns [1,2,3]
        d = my_func([1,2], 3)   # Returns [1,2,3]
        f = my_func(1, *[2,3])  # Returns [1,2,3]
        e = my_func(1, [2,3])   # Returns [1,[2,3]] since second argument is ambiguous
        g = my_func([[1,2]], 3) # Returns [[1,2],3] since first argument is nested

    *New in version 1.1.0.*
    """
    obj = []
    for arg in args:
        arg = tolist(arg, coerce=coerce, **kwargs)
        obj.extend(arg)
    if copy:
        obj = dcp(obj)
    return obj


def ifelse(*args, default=None, check=None):
    """
    For a list of inputs, return the first one that meets the condition
    
    By default, returns the first non-None item, but can also check truth value
    or an arbitrary function.
    
    Args:
        args (any): the arguments to check (note: cannot supply a single list, use * to unpack)
        default (any): the default value to use if no arguments meet the condition
        check (func): must be None (check if arguments are not None), bool (check if arguments evaluate True), or a callable (which returns True/False)
    
    Equivalent to ``next((arg for arg in args if check(arg)), default)``
    
    **Examples**::
        
        # 1. Standard usage
        a = None
        b = 3
        out = sc.ifelse(a, b) 
        
        ## Equivalent to:
        out = a if a is not None else b
        
        # 2. Boolean usage
        args = ['', False, {}, 'ok']
        out = sc.ifelse(*args, check=bool)
        
        ## Equivalent to:
        out = next((arg for arg in args if arg), None)
        
        # 3. Custom function
        args = [1, 3, 5, 7]
        out = sc.ifelse(*args, check=lambda x: x>5)
        
        ## Equivalent to:
        out = None
        for arg in args:
          if arg > 5:
            out = val
            break
    
    | *New in version 3.1.5.*
    """
    # Handle check
    if check is None:
        check = lambda x: x is not None
    elif check in [bool, True]:
        check = bool
    elif not callable(check): # pragma: no cover
        errormsg = f'"check" must be None, bool, or callable, not "{check}"'
        raise ValueError(errormsg)
        
    # Actually calculate it
    out = next((arg for arg in args if check(arg)), default)
    return out
    

def _sanitize_iterables(obj, *args):
    """
    Take input as a list, array, pandas Series, or non-iterable type, along with
    one or more arguments, and return a list, along with information on what the
    input types were (is_list and is_array).
    
    Not intended for the user, but used internally.

    **Examples**::

        _sanitize_iterables(1, 2, 3)             # Returns [1,2,3], False, False
        _sanitize_iterables([1, 2], 3)           # Returns [1,2,3], True, False
        _sanitize_iterables(np.array([1, 2]), 3) # Returns [1,2,3], True, True
        _sanitize_iterables(np.array([1, 2, 3])) # Returns [1,2,3], False, True
    """
    is_list   = isinstance(obj, list) or len(args)>0 # If we're given a list of args, treat it like a list
    is_array  = isinstance(obj, (np.ndarray, pd.Series)) # Check if it's an array
    if is_array: # If it is, convert it to a list
        obj = obj.tolist()
    objs = dcp(tolist(obj)) # Ensure it's a list, and deepcopy to avoid mutability
    objs.extend(args) # Add on any arguments
    return objs, is_list, is_array


def _sanitize_output(obj, is_list, is_array, dtype=None):
    """
    The companion to _sanitize_iterables, convert the object back to the original
    type supplied. Not for the user.
    """
    if is_array:
        output = np.array(obj, dtype=dtype)
    elif not is_list and len(obj) == 1:
        output = obj[0]
    else:
        output = obj
    return output


##############################################################################
#%% Misc. functions
##############################################################################

__all__ += ['strjoin', 'newlinejoin', 'strsplit', 'runcommand', 'uniquename', 
            'suggest', 'importbyname', 'importbypath']


def strjoin(*args, sep=', '):
    """
    Like string ``join()``, but handles more flexible inputs, converts items to
    strings. By default, join with commas.

    Args:
        args (list): the list of items to join
        sep (str): the separator string

    **Example**::

        sc.strjoin([1,2,3], 4, 'five')

    *New in version 1.1.0.*
    """
    obj = []
    for arg in args:
        if isstring(arg):
            obj.append(arg)
        elif isiterable(arg):
            obj.extend([str(item) for item in arg])
        else: # pragma: no cover
            obj.append(str(arg))
    output = sep.join(obj)
    return output


def newlinejoin(*args):
    """
    Alias to ``strjoin(*args, sep='\\n')``.

    **Example**::

        sc.newlinejoin([1,2,3], 4, 'five')

    *New in version 1.1.0.*
    """
    return strjoin(*args, sep='\n')


def strsplit(string, sep=None, skipempty=True, lstrip=True, rstrip=True):
    """
    Convenience function to split common types of strings.

    Note: to use regular expressions, use :func:`re.split()` instead.

    Args:
        string    (str):      the string to split
        sep       (str/list): the types of separator to accept (default space or comma, i.e. [' ', ','])
        skipempty (bool):     whether to skip empty entries (i.e. from consecutive delimiters)
        lstrip    (bool):     whether to strip any extra spaces on the left
        rstrip    (bool):     whether to strip any extra spaces on the right

    Examples:

        sc.strsplit('a b c') # Returns ['a', 'b', 'c']
        sc.strsplit('a,b,c') # Returns ['a', 'b', 'c']
        sc.strsplit('a, b, c') # Returns ['a', 'b', 'c']
        sc.strsplit('  foo_bar  ', sep='_') # Returns ['foo', 'bar']

    *New in version 2.0.0.*
    """
    strlist = []
    if sep is None:
        sep = [' ', ',']

    # Generate a character sequence that isn't in the string
    special = '∙' # Pick an obscure Unicode character
    while special in string: # If it exists in the string nonetheless, keep going until it doesn't # pragma: no cover
        special += special

    # Convert all separators to the special character
    for s in sep:
        string = string.replace(s, special)

    # Split the string, filter, and trim
    strlist = string.split(special)
    if lstrip:
        strlist = [s.lstrip() for s in strlist]
    if rstrip:
        strlist = [s.rstrip() for s in strlist]
    if skipempty:
        strlist = [s for s in strlist if s != '']

    return strlist


def runcommand(command, printinput=False, printoutput=None, wait=True, **kwargs):
    """
    Make it easier to run shell commands.

    Alias to :obj:`subprocess.Popen()`. Returns captured output if wait=True, else
    returns the subprocess.
    
    Args:
        command (str): the command to run
        printinput (bool): whether to print the input string
        printoutput (bool): whether to print the output (default: False if wait=True, True if wait=False)
        wait (bool): whether to wait for the process to return (else, return immediately with the subprocess)

    **Examples**::

        myfiles = sc.runcommand('ls').split('\\n') # Get a list of files in the current folder
        sc.runcommand('sshpass -f %s scp myfile.txt me@myserver:myfile.txt' % 'pa55w0rd', printinput=True, printoutput=True) # Copy a file remotely
        sc.runcommand('sleep 600; mkdir foo', wait=False) # Waits 10 min, then creates the folder "foo", but the function returns immediately
        sc.runcommand('find', wait=False) # Equivalent to executing 'find' in a terminal
    
    *New in version 3.1.1:* print real-time output if ``wait=False``
    """
    # Handle defaults and inputs
    if wait:
        defaults = dict(shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) # Redirect both to the pipe
    else:
        defaults = dict(shell=True, bufsize=0)
    if printoutput is None:
        printoutput = False if wait else True
    kwargs = mergedicts(defaults, kwargs)
    if printinput:
        print(command)
    
    # Actually run the command
    try:
        p = subprocess.Popen(command, **kwargs)
        if wait: # Run in foreground, blocking
            output, _ = p.communicate() # ...and then retrieve stdout from the pipe
            try: # Try to decode
                output = output.decode('utf-8')
            except Exception as E: # If something goes wrong, just leave it # pragma: no cover
                warnmsg = f'Could not decode bytestring: {E}'
                warnings.warn(warnmsg, category=RuntimeWarning, stacklevel=2)
        else: # Run in background
            if printoutput: # ...but print output as it comes if asked
                while p.returncode is None:
                    stdout, stderr = p.communicate()
                    if stdout: print(stdout) # Usually None, but just in case
                    if stderr: print(stderr)
            output = p # Return the subprocess instead of the output
    except Exception as E: # pragma: no cover
        output = f'runcommand(): shell command failed: {str(E)}' # This is for a Python error, not a shell error -- those get passed to output
    
    if printoutput and wait:
        print(output)
    
    return output


def uniquename(name=None, namelist=None, style=None, human=False, suffix=None):
    """
    Given a name and a list of other names, add a counter to the name so that
    it doesn't conflict with the other names.
    
    Useful for auto-incrementing filenames, etc.
    
    Args:
        name (str): the string to ensure is unique
        namelist (list): the list of strings that are taken
        style (str): a custom style for appending the counter
        human (bool): if True, use ' (%d)' as the style instead of '%d'
        suffix (str): if provided, remove this suffix from each name and add it back to the unique name

    **Examples**::

        sc.uniquename('out', ['out', 'out1']) # Returns 'out2'
        sc.uniquename(name='file', namelist=['file', 'file (1)', 'file (2)', 'myfile'], human=True) # Returns 'file (3)'
        sc.uniquename('results.csv', ['results.csv', 'results1.csv'], suffix='.csv') # Returns 'results2.csv'
    
    | *New in version 3.2.0:* "human" and "suffix" arguments, simpler default style
    """
    # Set the style if not specified
    if style is None:
        if human:
            style = ' (%d)'
        else:
            style = '%d'
    
    # Prepare the inputs
    name = str(name) # Ensure it's a string
    namelist = tolist(namelist)
    if suffix:
        name = name.removesuffix(suffix)
        namelist = [n.removesuffix(suffix) for n in namelist]
        
    # Do the loop
    i = 0
    unique_name = name # Start with the passed in name
    while unique_name in namelist: # Try adding an index (i) to the name until we find one that's unique
        i += 1
        unique_name = name + style%i # Convert from  the original name
    if suffix:
        unique_name += suffix
    return unique_name # Return the found name.


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
        fulloutput (bool): Whether to return suggestions and distances.
        die (bool): If True, an informative error will be raised (to avoid having to implement this in the calling code)
        which (str): Distance calculation method used; options are "damerau" (default), "levenshtein", or "jaro"

    Returns:
        suggestions (str or list): Suggested string. Returns None if no suggestions with edit distance less than threshold were found. This helps to make
             suggestions more relevant.

    **Examples**::

        >>> sc.suggest('foo', ['Foo','Bar'])
        'Foo'
        >>> sc.suggest('foo', ['FOO','Foo'])
        'Foo'
        >>> sc.suggest('foo', ['Foo ','boo'])
        'Foo '
    """
    try:
        import jellyfish # To allow as an optional import
    except ModuleNotFoundError as e: # pragma: no cover
        raise ModuleNotFoundError('The "jellyfish" Python package is not available; please install via "pip install jellyfish"') from e

    valid_inputs = tolist(valid_inputs, objtype='string')
    
    # Handle version incompatibility
    try: # jellyfish >= 1.0
        jaro = jellyfish.jaro_similarity
    except:
        jaro = jellyfish.jaro_distance

    mapping = {
        'damerau':     jellyfish.damerau_levenshtein_distance,
        'levenshtein': jellyfish.levenshtein_distance,
        'jaro':        jaro,
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
    if threshold < 0: # pragma: no cover
        threshold = np.inf

    # Output
    if min(distance) > threshold:
        if die: # pragma: no cover
            errormsg = f'"{user_input}" not found'
            raise ValueError(errormsg)
        else:
            return
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


def _assign_to_namespace(var, obj, namespace=None, overwrite=True): # pragma: no cover
    """ Helper function to assign an object to the global namespace """
    if namespace is None:
        namespace = globals()
    if var in namespace and not overwrite:
        errormsg = f'Cannot assign to variable "{var}" since it already exists and overwrite=False'
        raise NameError(errormsg)
    namespace[var] = obj
    return


def importbyname(module=None, variable=None, path=None, namespace=None, lazy=False, overwrite=True, die=True, verbose=True, **kwargs):
    """
    Import modules by name.
    
    ``sc.importbyname(x='y')`` is equivalent to "import y as x", but allows module
    importing to be done programmatically.

    See https://peps.python.org/pep-0690/ for a proposal for incorporating something
    similar into Python by default.

    Args:
        module (str): name of the module to import
        variable (str): the name of the variable to assign the module to (by default, the module's name)
        path (str/path): optionally load from path instead of by name
        namespace (dict): the namespace to load the modules into (by default, globals)
        lazy (bool): whether to create a LazyModule object instead of load the actual module
        overwrite (bool): whether to allow overwriting an existing variable (by default, yes)
        die (bool): whether to raise an exception if encountered
        verbose (bool): whether to print a warning if an module can't be imported
        **kwargs (dict): additional variable:modules pairs to import (see examples below)

    **Examples**::

        np = sc.importbyname('numpy') # Standard usage
        sc.importbyname(pd='pandas', np='numpy') # Use dictionary syntax to assign to namespace
        plt = sc.importbyname(plt='matplotlib.pyplot', lazy=True) # Won't actually import until e.g. plt.figure() is called
        mymod = sc.importbyname(path='/path/to/mymod') # Import by path rather than name
    
    See also :func:`sc.importbypath() <importbypath>`.

    | *New in version 2.1.0:* "verbose" argument
    | *New in version 3.0.0:* "path" argument
    """
    # Initialize
    if variable is None:
        variable = module

    # Map modules to variables
    mapping = {}
    if module is not None or path is not None:
        mapping[variable] = (path or module) # In this order so path takes precedence
    mapping.update(kwargs)

    # Load modules
    libs = []
    for variable,module in mapping.items():
        if lazy:
            lib = LazyModule(module=module, variable=variable, namespace=namespace)
        else:
            try:
                if path is not None:
                    lib = importbypath(path, variable)
                else:
                    lib = importlib.import_module(module)
            except Exception as E: # pragma: no cover
                errormsg = f'Cannot import "{module}" since {module} is not installed. Please install {module} and try again.'
                if verbose and not die:
                    print(errormsg)
                lib = None
                if die: raise E
                else:   return False

        _assign_to_namespace(var=variable, obj=lib, namespace=namespace, overwrite=overwrite)
        if namespace:
            namespace[variable] = lib
        libs.append(lib)

    if len(libs) == 1:
        libs = libs[0]

    return libs


def importbypath(path, name=None):
    """
    Import a module by path.
    
    Useful for importing multiple versions of the same module for comparison purposes.
    
    Args:
        path (str/path): the path to load the module from (with or without __init__.py)
        name (str): the name of the loaded module (by default, the file or folder name from the path)
    
    **Examples**::
        
        # Load a module that isn't importable otherwise
        mymod = sc.importbypath('my file with spaces.py')
        
        # Load two versions of the same module
        old = sc.importbypath('/path/to/old/mylib')
        new = sc.importbypath('/path/to/new/mylib')
        assert new.__version__ > old.__version__ # Example version comparison (see also sc.compareverisons())
    
    See also :func:`sc.importbyname() <importbyname>`.

    | *New in version 3.0.0.*
    | *New in version 3.2.0:* Allow importing two modules that have self imports (e.g. "import mylib" from within mylib)
    """
    # Sanitize the path and filename
    default_file='__init__.py'
    filepath = Path(path)
    parts = list(filepath.parts)
    if filepath.is_dir():
        filepath = filepath / default_file # Append default filename
    elif not filepath.is_file():
        errormsg = f'Could not import {filepath}: is not a valid file or folder'
        raise FileNotFoundError(errormsg)
    
    # Get the original name (what Python would know it as) from either the filename or the folder name
    if parts[-1] == default_file:
        basename = parts[-2] # If __init__.py was included, strip it now
    else:
        basename = parts[-1].removesuffix('.py')
    orig_name = sanitizestr(basename, validvariable=True) # Ensure it's a valid name
    if name is None:
        name = orig_name
        if name in sys.modules: # If a name isn't provided, ensure it doesn't conflict with an existing module name
            name = uniquename(name, sys.modules.keys())
    renamed = name != orig_name # Flag to track if we've renamed the module
    
    # If the original name is already present, make a copy to restore later
    if orig_name in sys.modules:
        orig_module = sys.modules[orig_name]
        restore = True
    else:
        restore = False
    
    # Actually import the module -- from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    
    # We need to call it by both orig_name and name to handle self-referential imports
    sys.modules[name] = module
    if renamed:
        sys.modules[orig_name] = module
    
    # Now actually "execute" the module, i.e. load it into memory
    spec.loader.exec_module(module)
    
    # Finally, clean up the original module name
    if restore:
        sys.modules[orig_name] = orig_module
    elif renamed:
        del sys.modules[orig_name]
    
    return module


##############################################################################
#%% Classes
##############################################################################

__all__ += ['KeyNotFoundError', 'LinkException', 'autolist', 'Link', 'LazyModule', 'tryexcept']


class KeyNotFoundError(KeyError):
    """
    A tiny class to fix repr for KeyErrors. KeyError prints the repr of the error
    message, rather than the actual message, so e.g. newline characters print as
    the character rather than the actual newline.

    **Example**::

        raise sc.KeyNotFoundError('The key "foo" is not available, but these are: "bar", "cat"')
    """

    def __str__(self): # pragma: no cover
        return Exception.__str__(self)


class LinkException(Exception): # pragma: no cover
    """
    An exception to raise when links are broken, for exclusive use with the Link
    class.
    """
    pass


class autolist(list):
    """
    A simple extension to a list that defines add methods to simplify appending
    and extension.

    **Examples**::

        ls = sc.autolist(3) # Quickly convert a scalar to a list

        ls = sc.autolist()
        for i in range(5):
            ls += i # No need for ls += [i]
    """
    def __init__(self, *args):
        arglist = mergelists(*args) # Convert non-iterables to iterables
        list.__init__(self, arglist)
        return 

    def __add__(self, obj=None):
        """ Allows non-lists to be concatenated """
        obj = tolist(obj)
        new = self.__class__(list.__add__(self, obj)) # Ensure it returns an autolist
        return new

    def __iadd__(self, obj):
        """ Allows += to work correctly -- key feature of autolist """
        obj = tolist(obj)
        self.extend(obj)
        return self

    def __getitem__(self, key):
        try:
            return list.__getitem__(self, key)
        except IndexError: # pragma: no cover
            errormsg = f'list index {key} is out of range for list of length {len(self)}'
            raise IndexError(errormsg) from None # Don't show the traceback


class Link:
    """
    A class to differentiate between an object and a link to an object. The idea
    is that this object is parsed differently from other objects -- most notably,
    a recursive method (such as a pickle) would skip over Link objects, and then
    would fix them up after the other objects had been reinstated.

    Version: 2017jan31
    """

    def __init__(self, obj=None):
        self.obj = obj # Store the object -- or rather a reference to it, if it's mutable
        try:    self.uid = obj.uid # If the object has a UID, store it separately
        except: self.uid = None # If not, just use None

    def __repr__(self): # pragma: no cover
        """ Just use default """
        from . import sc_printing as scp # To avoid circular import
        output  = sc.prepr(self)
        return output

    def __call__(self, obj=None):
        """ If called with no argument, return the stored object; if called with argument, update object """
        if obj is None:
            if type(self.obj)==LinkException: # If the link is broken, raise it now
                raise self.obj
            return self.obj
        else: # pragma: no cover
            self.__init__(obj)
            return

    def __copy__(self, *args, **kwargs):
        """ Do NOT automatically copy link objects!! """
        return Link(LinkException('Link object copied but not yet repaired'))

    def __deepcopy__(self, *args, **kwargs):
        """ Same as copy """
        return self.__copy__(*args, **kwargs)


class LazyModule:
    """
    Create a "lazy" module that is loaded if and only if an attribute is called.

    Typically not for use by the user, but is used by :func:`sc.importbyname() <importbyname>`.

    Args:
        module (str): name of the module to (not) load
        variable (str): variable name to assign the module to
        namespace (dict): the namespace to use (if not supplied, globals())
        overwrite (bool): whether to allow overwriting an existing variable (by default, yes)

    **Example**::

        pd = sc.LazyModule('pandas', 'pd') # pd is a LazyModule, not actually pandas
        df = pd.DataFrame() # Not only does this work, but pd is now actually pandas

    *New in version 2.0.0.*
    """

    def __init__(self, module, variable, namespace=None, overwrite=True):
        self._variable  = variable
        self._module    = module
        self._namespace = namespace
        self._overwrite = overwrite
        return


    def __repr__(self):
        output = f"<sc.LazyModule({self._variable}='{self._module}') at {hex(id(self))}>"
        return output


    def __getattr__(self, attr):
        """ In most cases, when an attribute is retrieved we want to replace this module with the actual one """
        _builtin_keys = ['_variable', '_module', '_namespace', '_overwrite', '_load']
        if attr in _builtin_keys: # pragma: no cover
            obj = object.__getattribute__(self, attr)
        else:
            obj = self._load(attr)
        return obj


    def _load(self, attr=None):
        """ Stop being lazy and load the module """
        var = self._variable
        lib = importlib.import_module(self._module)
        _assign_to_namespace(var, lib, namespace=self._namespace, overwrite=self._overwrite)
        if attr:
            obj = getattr(lib, attr)
        else: # pragma: no cover
            obj = lib
        return obj
    

class tryexcept(cl.suppress):
    """
    Simple class to catch exceptions in a single line
    
    Effectively an alias to :obj:`contextlib.suppress()`, which itself is a programmatic
    equivalent to using try-except blocks.
    
    By default, all errors are caught. If ``catch`` is not None, then by default 
    raise all other exceptions; if ``die`` is an exception (list of exceptions, 
    then by default suppress all other exceptions.
    
    Due to Python's fundamental architecture, exceptions can only be caught inside
    a with statement, and the with block will exit immediately as soon as the first
    exception is encountered.
    
    Args:
        die (bool/exception): default behavior of whether to raise caught exceptions
        catch (exception): one or more exceptions to catch regardless of "die"
        verbose (bool): whether to print caught exceptions (0 = silent, 1 = error type, 2 = full error information)
        history (list/tryexcept): a ``tryexcept`` object, or a list of exceptions, to keep the history (see example below)
    
    **Examples**::

        # Basic usage
        values = [0,1]
        with sc.tryexcept(): # Equivalent to contextlib.suppress(Exception)
            values[2]
            
        # Raise only certain errors
        with sc.tryexcept(die=IndexError): # Catch everything except IndexError
            values[2]

        # Catch (do not raise) only certain errors, and print full error information
        with sc.tryexcept(catch=IndexError, verbose=2): # Raise everything except IndexError
            values[2]
            
        # Storing the history of multiple exceptions
        tryexc = None
        for i in range(5):
            with sc.tryexcept(history=tryexc) as tryexc:
                print(values[i])
        tryexc.traceback()
            
    | *New in version 2.1.0.*
    | *New in version 3.0.0:* renamed "print" to "traceback"; added "to_df" and "disp" options
    | *New in version 3.1.0:* renamed "exceptions" to "data"; added "exceptions" property
    """

    def __init__(self, die=None, catch=None, verbose=1, history=None):
        
        # Handle defaults
        dietypes   = []
        catchtypes = []
        if die is None and catch is None: # Default: do not die
            self.defaultdie = False
        elif die in [True, False, 0, 1]: # It's truthy: use it directly # pragma: no cover
            self.defaultdie = die
        elif die is None and catch is not None: # We're asked to catch some things, so die otherwise
            self.defaultdie = True
            catchtypes = tolist(catch)
        elif die is not None and catch is None: # Vice versa
            self.defaultdie = False
            dietypes = tolist(die)
        else: # pragma: no cover
            errormsg = 'Unexpected input to "die" and "catch": typically only one or the other should be provided'
            raise ValueError(errormsg)
        
        # Finish initialization
        self.dietypes   = tuple(dietypes)
        self.catchtypes = tuple(catchtypes)
        self.verbose    = verbose
        self.data = []
        if history is not None:
            if isinstance(history, (list, tuple)): # pragma: no cover
                self.data.extend(list(history))
            elif isinstance(history, tryexcept):
                self.data.extend(history.data)
            else: # pragma: no cover
                errormsg = f'Could not understand supplied history: must be a list or a tryexcept object, not {type(history)}'
                raise TypeError(errormsg)
        return
    
    
    def __repr__(self):
        from . import sc_printing as scp # To avoid circular import
        return sc.prepr(self)
    
    
    def __len__(self):
        return len(self.data)


    def __enter__(self):
        return self
    
    
    def __exit__(self, exc_type, exc_val, traceback):
        """ If a context manager returns True from exit, the exception is caught """
        if exc_type is not None:
            self.data.append([exc_type, exc_val, traceback])
            die = (self.defaultdie or issubclass(exc_type, self.dietypes))
            live = issubclass(exc_type, self.catchtypes)
            if die and not live:
                return
            else:
                if self.verbose > 1: # Print everything # pragma: no cover
                    self.print()
                elif self.verbose: # Just print the exception type
                    print(exc_type, exc_val)
                return True

    
    def traceback(self, which=None, tostring=False):
        """
        Print the exception (usually the last)
        
        Args:
            which (int/list): which exception(s) to print; if None, print all
            tostring (bool): whether to return as a string (otherwise print)
        
        *New in version 3.1.0:* optionally print multiple tracebacks
        """
        string = ''
        if self.died:
            if which is None:
                which = np.arange(len(self))
            inds = toarray(which)
            for ind in inds:
                string += f'Traceback {ind+1} of {len(self)}:\n'
                string += traceback(*self.data[ind]) + '\n'
        else: # pragma: no cover
            print('No exceptions were encountered; nothing to trace')
        if tostring:
            return string
        else:
            print(string)
            return
    
    
    def to_df(self):
        """ Convert the exceptions to a dataframe """
        df = sc.dataframe(self.data, columns=['type','value','traceback'])
        self.df = df
        return df
    
    
    def disp(self):
        """ Display all exceptions as a table """
        df = self.to_df()
        df.disp()
        return
    
    @property
    def exceptions(self):
        """ Retrieve the last exception, if any """
        return [entry[1] for entry in self.data]

    @property
    def exception(self):
        """ Retrieve the last exception, if any """
        if len(self.data): # Exceptions were encountered
            return self.data[-1][1] # This is just the exception
        else:
            return None # Else, return None
    
    @property
    def died(self):
        """ Whether or not any exceptions were encountered """
        return len(self) > 0
