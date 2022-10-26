'''
Miscellaneous utilities for type checking, printing, dates and times, etc.

Note: there are a lot! The design philosophy has been that it's easier to
ignore a function that you don't need than write one from scratch that you
do need.

Highlights:
    - :func:`dcp`: shortcut to ``copy.deepcopy()``
    - :func:`pp`: shortcut to ``pprint.pprint()``
    - :func:`isnumber`: checks if something is any number type
    - :func:`tolist`: converts any object to a list, for easy iteration
    - :func:`toarray`: tries to convert any object to an array, for easy use with numpy
    - :func:`mergedicts`: merges any set of inputs into a dictionary
    - :func:`mergelists`: merges any set of inputs into a list
    - :func:`runcommand`: simple way of executing a shell command
    - :func:`download`: download multiple URLs in parallel
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
import string
import numbers
import pprint
import hashlib
import warnings
import subprocess
import unicodedata
import numpy as np
import pandas as pd
import random as rnd
import uuid as py_uuid
import packaging.version
import traceback as py_traceback

# Handle types
_stringtypes = (str, bytes)
_numtype     = (numbers.Number,)
_booltypes   = (bool, np.bool_)


##############################################################################
#%% Adaptations from other libraries
##############################################################################

# Define the modules being loaded
__all__ = ['fast_uuid', 'uuid', 'dcp', 'cp', 'pp', 'sha', 'freeze', 'require',
           'traceback', 'getplatform', 'iswindows', 'islinux', 'ismac', 'asciify']


def fast_uuid(which=None, length=None, n=1, secure=False, forcelist=False, safety=1000, recursion=0, recursion_limit=10, verbose=True):
    '''
    Create a fast UID or set of UIDs. Note: for certain applications, ``sc.uuid()``
    is faster than ``sc.fast_uuid()``!

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


def dcp(obj, die=True, verbose=True):
    '''
    Shortcut to perform a deep copy operation

    Almost identical to ``copy.deepcopy()``, but optionally fall back to copy()
    if deepcopy fails.

    Args:
        die (bool): if False, fall back to copy()
        verbose (bool): if die is False, then print a warning if deepcopy() fails

    New in version 2.0.0: default die=True instead of False
    '''
    try:
        output = copy.deepcopy(obj)
    except Exception as E: # pragma: no cover
        output = cp(obj)
        errormsg = f'Warning: could not perform deep copy: {str(E)}'
        if die: raise RuntimeError(errormsg)
        else:   print(errormsg + '\nPerforming shallow copy instead...')
    return output


def cp(obj, die=True, verbose=True):
    '''
    Shortcut to perform a shallow copy operation

    Almost identical to ``copy.copy()``, but optionally allow failures

    New in version 2.0.0: default die=True instead of False
    '''
    try:
        output = copy.copy(obj)
    except Exception as E:
        output = obj
        errormsg = 'Could not perform shallow copy'
        if die: raise ValueError(errormsg) from E
        else:   print(errormsg + '\nReturning original object...')
    return output


def _printout(string=None, doprint=None, output=False):
    '''
    Short for "print or output string". Not for public use.

    Private helper function to handle the logic of two boolean flags: by default, print;
    but if output is requested, return the output and do not print; unless doprint
    is set to True.
    '''
    # Default for doprint is opposite of output
    if doprint is None: doprint = not(output)

    # Handle actions/output
    if doprint: print(string)
    if output: return string
    else:      return


def pp(obj, jsonify=True, doprint=None, output=False, verbose=False, **kwargs):
    '''
    Shortcut for pretty-printing the object.

    Almost identical to ``pprint.pprint()``, but can also be used as an alias for
    ``pprint.pformat()``.

    Args:
        obj     (any):  object to print
        jsonify (bool): whether to first convert the object to JSON, to handle things like ordered dicts nicely
        doprint (bool): whether to show output (default true)
        output  (bool): whether to return output as a string (default false)
        verbose (bool): whether to show warnings when jsonifying the object
        kwargs  (dict): passed to ``pprint.pprint()``

    New in version 1.3.1: output argument
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
    string = pprint.pformat(toprint, **kwargs)
    return _printout(string=string, doprint=doprint, output=output)


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


def freeze(lower=False):
    '''
    Alias for pip freeze.

    Args:
        lower (bool): convert all keys to lowercase

    **Example**::

        assert 'numpy' in sc.freeze() # One way to check for versions

    New in version 1.2.2.
    '''
    import pkg_resources as pkgr # Imported here since slow (>0.1 s)
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
    import pkg_resources as pkgr # Imported here since slow (>0.1 s)

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
        errkeys = list(errs.keys())
        errormsg = '\nThe following requirement(s) were not met:'
        count = 0
        for k,v in data.items():
            if not v:
                count += 1
                errormsg += f'\n• "{k}": {str(errs[k])}'
        errormsg += f'''\n\nIf this is a valid module, you might want to try "pip install {strjoin(errkeys, sep=' ')} --upgrade".'''
        if die:
            err = errs[errkeys[-1]]
            raise ModuleNotFoundError(errormsg) from err
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

    **Example**::

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
    ''' Alias to ``sc.getplatform('windows')`` '''
    return getplatform('windows', die=die)

def islinux(die=False):
    ''' Alias to ``sc.getplatform('linux')`` '''
    return getplatform('linux', die=die)

def ismac(die=False):
    ''' Alias to ``sc.getplatform('mac')`` '''
    return getplatform('mac', die=die)


def asciify(string, form='NFKD', encoding='ascii', errors='ignore', **kwargs):
    '''
    Convert an arbitrary Unicode string to ASCII.
    
    Args:
        form (str): the type of Unicode normalization to use
        encoding (str): the output string to encode to
        errors (str): how to handle errors
        kwargs (dict): passed to ``string.decode()``
    
    **Example**:
        sc.asciify('föö→λ ∈ ℝ') # Returns 'foo  R'
    
    New in version 2.0.1.
    '''
    normalized = unicodedata.normalize(form, string) # First, normalize Unicode encoding
    encoded = normalized.encode(encoding, errors) # Then, convert to ASCII
    decoded = encoded.decode(**kwargs) # Finally, decode back to utf-8
    return decoded

##############################################################################
#%% Web/HTML functions
##############################################################################

__all__ += ['urlopen', 'wget', 'download', 'htmlify']

def urlopen(url, filename=None, save=False, headers=None, params=None, data=None,
            prefix='http', convert=True, die=False, return_response=False, verbose=False):
    '''
    Download a single URL.

    Alias to ``urllib.request.urlopen(url).read()``. See also ``sc.download()``
    for downloading multiple URLs. Note: ``sc.urlopen()``/``sc.wget()`` are aliases.

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
        return_response (bool): whether to return the response object instead of the output
        verbose (bool): whether to print progress

    **Examples**::

        html = sc.urlopen('sciris.org') # Retrieve into variable html
        sc.urlopen('http://sciris.org', filename='sciris.html') # Save to file sciris.html
        sc.urlopen('http://sciris.org', save=True, headers={'User-Agent':'Custom agent'}) # Save to the default filename (here, sciris.org), with headers

    | New in version 2.0.0: renamed from ``wget`` to ``urlopen``; new arguments
    | New in version 2.0.1: creates folders by default if they do not exist
    | New in version 2.0.4: "prefix" argument, e.g. prepend "http://" if not present
    '''
    from urllib import request as ur # Need to import these directly, not via urllib
    from urllib import parse as up
    from . import sc_datetime as scd  # To avoid circular import
    from . import sc_fileio as scf # To avoid circular import

    T = scd.timer()

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
    if params is not None:
        full_url = full_url + '?' + up.urlencode(params)
    if data is not None:
        data = up.urlencode(data).encode(encoding='utf-8', errors='ignore')

    if verbose: print(f'Downloading {url}...')
    request = ur.Request(full_url, headers=headers, data=data)
    response = ur.urlopen(request)
    output = response.read()
    if convert:
        if verbose: print('Converting from bytes to text...')
        try:
            output = output.decode()
        except Exception as E: # pragma: no cover
            if die:
                raise E
            elif verbose:
                errormsg = f'Could not decode to text: {str(E)}'
                print(errormsg)

    # Set filename -- from https://stackoverflow.com/questions/31804799/how-to-get-pdf-filename-with-python-requests
    if filename is None and save:
        headers = dict(response.getheaders())
        string = "Content-Disposition"
        if string in headers.keys():
            filename = re.findall("filename=(.+)", headers[string])[0]
        else:
            filename = url.rstrip('/').split('/')[-1] # Remove trailing /, then pull out the last chunk

    if filename is not None:
        if verbose: print(f'Saving to {filename}...')
        filename = scf.makefilepath(filename)
        if isinstance(output, bytes):
            with open(filename, 'wb') as f:
                f.write(output)
        else:
            with open(filename, 'w') as f:
                f.write(output)
        output = filename

    if verbose:
        T.toc(f'Time to download {url}')
    if return_response:
        output = response

    return output

# Alias for backwards compatibility
wget = urlopen


def download(url, *args, filename=None, save=True, parallel=True, verbose=True, **kwargs):
    '''
    Download one or more URLs in parallel and return output or save them to disk.

    A wrapper for ``sc.urlopen()``, except with ``save=True`` by default.

    Args:
        url (str/list/dict): either a single URL, a list of URLs, or a dict of URL:filename pairs
        *args (list): additional URLs to download
        filename (str/list): either a string or a list of the same length as ``url`` (if not supplied, return output)
        save (bool): if supplied instead of ``filename``, then use the default filename
        parallel (bool): whether to download multiple URLs in parallel
        verbose (bool): whether to print progress (if verbose=2, print extra detail on each downloaded URL)
        **kwargs (dict): passed to ``sc.urlopen()``

    **Examples**::

        html = sc.download('http://sciris.org') # Download a single URL
        data = sc.download('http://sciris.org', 'http://covasim.org') # Download two in parallel
        sc.download({'http://sciris.org':'sciris.html', 'http://covasim.org':'covasim.html'}) # Downlaod two and save to disk
        sc.download(['http://sciris.org', 'http://covasim.org'], filename=['sciris.html', 'covasim.html']) # Ditto

    New in version 2.0.0.
    '''
    from . import sc_parallel as scp # To avoid circular import
    from . import sc_datetime as scd

    T = scd.timer()

    # Parse arguments
    if isinstance(url, dict):
        urls = list(url.keys())
        filenames = list(url.values())
    else:
        urls = mergelists(url, *args)
        filenames = mergelists(filename)

    # Ensure consistency
    n_urls = len(urls)
    n_filenames = len(filenames)
    if not n_filenames:
        filenames = [None]*n_urls
    elif n_filenames != n_urls: # pragma: no cover
        errormsg = f'Cannot process {n_urls} URLs and {n_filenames} filenames'
        raise ValueError(errormsg)

    if verbose:
        print(f'Downloading {n_urls} URL(s)...')

    # Get results in parallel
    wget_verbose = (verbose>1) or (verbose and n_urls == 1) # By default, don't print progress on each download
    iterkwargs = dict(url=urls, filename=filenames)
    func_kwargs = mergedicts(dict(save=save, verbose=wget_verbose), kwargs)
    if n_urls > 1 and parallel:
        outputs = scp.parallelize(urlopen, iterkwargs=iterkwargs, kwargs=func_kwargs, parallelizer='thread')
    else:
        outputs = []
        for url,filename in zip(urls, filenames):
            output = urlopen(url=url, filename=filename, **func_kwargs)
            outputs.append(output)

    if verbose:
        T.toc(f'Time to download {n_urls} URLs')
    if n_urls == 1:
        outputs = outputs[0]

    return outputs


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


##############################################################################
#%% Type functions
##############################################################################

__all__ += ['flexstr', 'isiterable', 'checktype', 'isnumber', 'isstring', 'isarray',
            'promotetoarray', 'promotetolist', 'toarray', 'tolist', 'transposelist',
            'swapdict', 'mergedicts', 'mergelists']

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
    
    New in version 2.0.1: ``pd.Series`` considered 'array-like'
    '''

    # Handle "objtype" input
    if   objtype in ['str','string']:          objinstance = _stringtypes
    elif objtype in ['num', 'number']:         objinstance = _numtype
    elif objtype in ['bool', 'boolean']:       objinstance = _booltypes
    elif objtype in ['arr', 'array']:          objinstance = np.ndarray
    elif objtype in ['listlike', 'arraylike']: objinstance = (list, tuple, np.ndarray, pd.Series) # Anything suitable as a numerical array
    elif type(objtype) == type:                objinstance = objtype # Don't need to do anything
    elif isinstance(objtype, tuple):           objinstance = objtype # Ditto
    elif objtype is None:                      return # If not supplied, exit
    else: # pragma: no cover
        errormsg = f'Could not understand what type you want to check: should be either a string or a type, not "{objtype}"'
        raise ValueError(errormsg)

    # Do first-round checking
    result = isinstance(obj, objinstance)

    # Do second round checking
    if result and objtype in ['listlike', 'arraylike']: # Special case for handling arrays which may be multi-dimensional
        obj = promotetoarray(obj).flatten() # Flatten all elements
        if objtype == 'arraylike' and subtype is None: # Add additional check for numeric entries
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
    Determine whether or not the input is string-like (i.e., str or bytes).

    Equivalent to ``isinstance(obj, (str, bytes))``
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
    (note: ``toarray()`` and ``promotetoarray()`` are identical).

    Very similar to ``np.array``, with the main difference being that ``sc.promotetoarray(3)``
    will return ``np.array([3])`` (i.e. a 1-d array that can be iterated over), while
    ``np.array(3)`` will return a 0-d array that can't be iterated over.

    Args:
        x (any): a number or list of numbers
        keepnone (bool): whether ``sc.promotetoarray(None)`` should return ``np.array([])`` or ``np.array([None], dtype=object)``
        kwargs (dict): passed to ``np.array()``

    **Examples**::

        sc.promotetoarray(5) # Returns np.array([5])
        sc.promotetoarray([3,5]) # Returns np.array([3,5])
        sc.promotetoarray(None, skipnone=True) # Returns np.array([])

    | New in version 1.1.0: replaced "skipnone" with "keepnone"; allowed passing
    kwargs to ``np.array()``.
    | New in version 2.0.1: added support for pandas Series and DataFrame
    '''
    skipnone = kwargs.pop('skipnone', None)
    if skipnone is not None: # pragma: no cover
        keepnone = not(skipnone)
        warnmsg = 'sc.promotetoarray() argument "skipnone" has been deprecated as of v1.1.0; use keepnone instead'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)
    if isnumber(x) or (isinstance(x, np.ndarray) and not np.shape(x)): # e.g. 3 or np.array(3)
        x = [x]
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
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
    - 'tuple': all the types in default, plus tuples
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

    | New in version 1.1.0: "coerce" argument
    | New in version 1.2.2: default coerce values
    | New in version 2.0.2: tuple coersion
    '''
    # Handle coerce
    default_coerce = (range, map, type({}.keys()), type({}.values()), type({}.items()))
    if isinstance(coerce, str):
        if coerce == 'none':
            coerce = None
        elif coerce == 'default':
            coerce = default_coerce
        elif coerce == 'tuple':
            coerce = default_coerce + (tuple,)
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


def swapdict(d):
    '''
    Swap the keys and values of a dictionary. Equivalent to {v:k for k,v in d.items()}

    Args:
        d (dict): dictionary

    **Example**::
        d1 = {'a':'foo', 'b':'bar'}
        d2 = sc.swapdict(d1) # Returns {'foo':'a', 'bar':'b'}

    New in version 1.3.0.
    '''
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
    '''
    Small function to merge multiple dicts together.

    By default, skips any input arguments that are ``None``, and allows keys to be set 
    multiple times. This function is similar to dict.update(), except it returns a value.
    The first dictionary supplied will be used for the output type (e.g. if the
    first dictionary is an odict, an odict will be returned).

    Note that arguments start with underscores to avoid possible collisions with
    keywords (e.g. ``sc.mergedicts(dict(loose=True, strict=True), strict=False, _strict=True)``).

    This function is useful for cases, e.g. function arguments, where the default 
    option is ``None`` but you will need a dict later on.

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

    | New in version 1.1.0: "copy" argument
    | New in version 1.3.3: keywords allowed
    | New in version 2.0.0: keywords fully enabled; "_sameclass" argument
    | New in version 2.0.1: fixed bug with "_copy" argument
    '''
    # Warn about deprecated keys
    renamed = ['strict', 'overwrite', 'copy']
    if any([k in kwargs for k in renamed]):
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
                except Exception as E:
                    errormsg = f'Could not create new dict of {type(args[0])} from first argument ({str(E)}); set _sameclass=False if this is OK'
                    if _die: raise TypeError(errormsg) from E
                    else:    print(errormsg)

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
            if arg is not None and _die:
                errormsg = f'Could not handle argument {a} of {type(arg)}: expecting dict or None'
                raise TypeError(errormsg)

    if _copy:
        outputdict = dcp(outputdict, die=_die)
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


def _sanitize_iterables(obj, *args):
    '''
    Take input as a list, array, pandas Series, or non-iterable type, along with
    one or more arguments, and return a list, along with information on what the
    input types were.

    **Examples**::

        _sanitize_iterables(1, 2, 3)             # Returns [1,2,3], False, False
        _sanitize_iterables([1, 2], 3)           # Returns [1,2,3], True, False
        _sanitize_iterables(np.array([1, 2]), 3) # Returns [1,2,3], True, True
        _sanitize_iterables(np.array([1, 2, 3])) # Returns [1,2,3], False, True
    '''
    is_list   = isinstance(obj, list) or len(args)>0 # If we're given a list of args, treat it like a list
    is_array  = isinstance(obj, (np.ndarray, pd.Series)) # Check if it's an array
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


##############################################################################
#%% Misc. functions
##############################################################################

__all__ += ['strjoin', 'newlinejoin', 'strsplit', 'runcommand', 'gitinfo', 'compareversions',
            'uniquename', 'suggest', 'getcaller', 'importbyname']


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


def strsplit(string, sep=None, skipempty=True, lstrip=True, rstrip=True):
    '''
    Convenience function to split common types of strings.

    Note: to use regular expressions, use ``re.split()`` instead.

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

    New in version 2.0.0.
    '''
    strlist = []
    if sep is None:
        sep = [' ', ',']

    # Generate a character sequence that isn't in the string
    special = '∙' # Pick an obscure character
    while special in string: # If it exists in the string nonetheless, keep going until it doesn't
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
    if packaging.version.parse(v1) > packaging.version.parse(v2):
        comparison =  1
    elif packaging.version.parse(v1) < packaging.version.parse(v2):
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


def getcaller(frame=2, tostring=True, includelineno=False, includeline=False):
    '''
    Try to get information on the calling function, but fail gracefully.

    Frame 1 is the file calling this function, so not very useful. Frame 2 is
    the default assuming, it is being called directly. Frame 3 is used if
    another function is calling this function internally.

    Args:
        frame (int): how many frames to descend (e.g. the caller of the caller of the...), default 2
        tostring (bool): whether to return a string instead of a dict with filename and line number
        includelineno (bool): if ``tostring``, whether to also include the line number
        includeline (bool): if not ``tostring``, also store the line contents

    Returns:
        output (str/dict): the filename (and line number) of the calling function, either as a string or dict

    **Examples**::

        sc.getcaller()
        sc.getcaller(tostring=False)['filename'] # Equivalent to sc.getcaller()
        sc.getcaller(frame=3) # Descend one level deeper than usual
        sc.getcaller(frame=1, tostring=False, includeline=True) # See the line that called sc.getcaller()

    | New in version 1.0.0.
    | New in version 1.3.3: do not include line by default
    '''
    try:
        import inspect
        result = inspect.getouterframes(inspect.currentframe(), 2)
        fname = str(result[frame][1])
        lineno = result[frame][2]
        if tostring:
            output = f'{fname}'
            if includelineno:
                output += f', line {lineno}'
        else:
            output = {'filename':fname, 'lineno':lineno}
            if includeline:
                try:
                    with open(fname) as f:
                        lines = f.read().splitlines()
                        line = lines[lineno-1] # -1 since line numbers start at 1
                    output['line'] = line
                except: # Fail silently
                    output['line'] = 'N/A'
    except Exception as E: # pragma: no cover
        if tostring:
            output = f'Calling function information not available ({str(E)})'
        else:
            output = {'filename':'N/A', 'lineno':'N/A'}
    return output


def _assign_to_namespace(var, obj, namespace=None, overwrite=True):
    ''' Helper function to assign an object to the global namespace '''
    if namespace is None:
        namespace = globals()
    if var in namespace and not overwrite:
        errormsg = f'Cannot assign to variable "{var}" since it already exists and overwrite=False'
        raise NameError(errormsg)
    namespace[var] = obj
    return


def importbyname(module=None, variable=None, namespace=None, lazy=False, overwrite=True, die=True, **kwargs):
    '''
    Import modules by name.

    See https://peps.python.org/pep-0690/ for a proposal for incorporating something
    similar into Python by default.

    Args:
        module (str): name of the module to import
        variable (str): the name of the variable to assign the module to (by default, the module's name)
        namespace (dict): the namespace to load the modules into (by default, globals)
        lazy (bool): whether to create a LazyModule object instead of load the actual module
        overwrite (bool): whether to allow overwriting an existing variable (by default, yes)
        die (bool): whether to raise an exception if encountered
        **kwargs (dict): additional variable:modules pairs to import (see examples below)

    **Examples**::

        np = sc.importbyname('numpy')
        sc.importbyname(pd='pandas', np='numpy')
        pl = sc.importbyname(pl='matplotlib.pyplot', lazy=True) # Won't actually import until e.g. pl.figure() is called
    '''
    # Initialize
    import importlib
    if variable is None:
        variable = module

    # Map modules to variables
    mapping = {}
    if module is not None:
        mapping[variable] = module
    mapping.update(kwargs)

    # Load modules
    libs = []
    for variable,module in mapping.items():
        if lazy:
            lib = LazyModule(module=module, variable=variable, namespace=namespace)
        else:
            try:
                lib = importlib.import_module(module)
            except Exception as E: # pragma: no cover
                errormsg = f'Cannot import "{module}" since {module} is not installed. Please install {module} and try again.'
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



##############################################################################
#%% Classes
##############################################################################

__all__ += ['KeyNotFoundError', 'LinkException', 'prettyobj', 'autolist', 'Link', 'LazyModule']


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

    | New in version 2.0.0: allow positional arguments
    '''

    def __init__(self, *args, **kwargs):
        kwargs = mergedicts(*args, kwargs)
        for k,v in kwargs.items():
            self.__dict__[k] = v
        return


    def __repr__(self):
        from . import sc_printing as scp # To avoid circular import
        output  = scp.prepr(self)
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
        list.__init__(self, arglist)
        return 

    def __add__(self, obj=None):
        ''' Allows non-lists to be concatenated '''
        obj = promotetolist(obj)
        new = autolist(list.__add__(self, obj)) # Ensure it returns an autolist
        return new

    def __iadd__(self, obj):
        ''' Allows += to work correctly -- key feature of autolist '''
        obj = promotetolist(obj)
        self.extend(obj)
        return self

    def __getitem__(self, key):
        try:
            return list.__getitem__(self, key)
        except IndexError: # pragma: no cover
            errormsg = f'list index {key} is out of range for list of length {len(self)}'
            raise IndexError(errormsg) from None # Don't show the traceback


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
        from . import sc_printing as scp # To avoid circular import
        output  = scp.prepr(self)
        return output

    def __call__(self, obj=None):
        ''' If called with no argument, return the stored object; if called with argument, update object '''
        if obj is None:
            if type(self.obj)==LinkException: # If the link is broken, raise it now
                raise self.obj
            return self.obj
        else: # pragma: no cover
            self.__init__(obj)
            return

    def __copy__(self, *args, **kwargs):
        ''' Do NOT automatically copy link objects!! '''
        return Link(LinkException('Link object copied but not yet repaired'))

    def __deepcopy__(self, *args, **kwargs):
        ''' Same as copy '''
        return self.__copy__(*args, **kwargs)


class LazyModule:
    '''
    Create a "lazy" module that is loaded if and only if an attribute is called.

    Typically not for use by the user, but is used by ``sc.importbyname()``.

    Args:
        module (str): name of the module to (not) load
        variable (str): variable name to assign the module to
        namespace (dict): the namespace to use (if not supplied, globals())
        overwrite (bool): whether to allow overwriting an existing variable (by default, yes)

    **Example**::

        pd = sc.LazyModule('pandas', 'pd') # pd is a LazyModule, not actually pandas
        df = pd.DataFrame() # Not only does this work, but pd is now actually pandas

    New in version 2.0.0.
    '''

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
        ''' In most cases, when an attribute is retrieved we want to replace this module with the actual one '''
        _builtin_keys = ['_variable', '_module', '_namespace', '_overwrite', '_load']
        if attr in _builtin_keys:
            obj = object.__getattribute__(self, attr)
        else:
            obj = self._load(attr)
        return obj


    def _load(self, attr=None):
        ''' Stop being lazy and load the module '''
        import importlib
        var = self._variable
        lib = importlib.import_module(self._module)
        _assign_to_namespace(var, lib, namespace=self._namespace, overwrite=self._overwrite)
        if attr:
            obj = getattr(lib, attr)
        else:
            obj = lib
        return obj