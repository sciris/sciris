"""
Functions for reading/writing to files, including pickles, JSONs, and Excel.

Highlights:
    -  ``sc.saveobj()/sc.loadobj()``: efficiently save/load any Python object (via pickling)
    -  ``sc.savejson()/sc.loadjson()``: likewise, for JSONs
    -  ``sc.thisdir()``: get current folder
    -  ``sc.getfilelist()``: easy way to access glob
"""

##############################################################################
#%% Imports
##############################################################################

# Basic imports
import io
import os
import re
import json
import uuid
import types
import inspect
import importlib
import traceback
import warnings
import numpy as np
import datetime as dt
from glob import glob
from zipfile import ZipFile
from contextlib import closing
from io import BytesIO as IO
from pathlib import Path
import copyreg as cpreg
import pickle as pkl
import gzip as gz
from . import sc_settings as scs
from . import sc_utils as scu
from . import sc_printing as scp
from . import sc_datetime as scd
from . import sc_odict as sco
from . import sc_dataframe as scdf


##############################################################################
#%% Pickling functions
##############################################################################

__all__ = ['loadobj', 'loadstr', 'saveobj', 'dumpstr', 'load', 'save']


def loadobj(filename=None, folder=None, verbose=False, die=None, remapping=None, method='pickle', **kwargs):
    '''
    Load a file that has been saved as a gzipped pickle file, e.g. by ``sc.saveobj()``.
    Accepts either a filename (standard usage) or a file object as the first argument.
    Note that ``loadobj()``/``load()`` are aliases of each other.

    Note: be careful when loading pickle files, since a malicious pickle can be
    used to execute arbitrary code.

    When a pickle file is loaded, Python imports any modules that are referenced
    in it. This is a problem if module has been renamed. In this case, you can
    use the ``remapping`` argument to point to the new modules or classes.

    Args:
        filename  (str/Path): the filename (or full path) to load
        folder    (str/Path): the folder
        verbose   (bool):     print details
        die       (bool):     whether to raise an exception if errors are encountered (otherwise, load as much as possible)
        remapping (dict):     way of mapping old/unavailable module names to new
        method    (str):      method for loading (usually pickle or dill)
        kwargs    (dict):     passed to pickle.loads()/dill.loads()

    **Examples**::

        obj = sc.loadobj('myfile.obj') # Standard usage
        old = sc.loadobj('my-old-file.obj', method='dill', ignore=True) # Load classes from saved files
        old = sc.loadobj('my-old-file.obj', remapping={'foo.Bar':cat.Mat}) # If loading a saved object containing a reference to foo.Bar that is now cat.Mat
        old = sc.loadobj('my-old-file.obj', remapping={'foo.Bar':('cat', 'Mat')}) # Equivalent to the above

    | New in version 1.1.0: "remapping" argument
    | New in version 1.2.2: ability to load non-gzipped pickles; support for dill; arguments passed to loader
    '''

    # Handle loading of either filename or file object
    if isinstance(filename, Path):
        filename = str(filename)
    if scu.isstring(filename):
        argtype = 'filename'
        filename = makefilepath(filename=filename, folder=folder, makedirs=False) # If it is a file, validate the folder (but don't create one if it's missing)
    elif isinstance(filename, io.BytesIO):
        argtype = 'fileobj'
    else: # pragma: no cover
        errormsg = f'First argument to loadobj() must be a string or file object, not {type(filename)}'
        raise TypeError(errormsg)
    fileargs = {'mode': 'rb', argtype: filename}

    # Define common error messages
    gziperror = f'''
Unable to load
    {filename}
as either a gzipped or regular pickle file. Ensure that it is actually a pickle file.
'''
    unpicklingerror = f'''
Unable to load
    {filename}
as a gzipped pickle file. Loading pickles can fail if Python modules have changed
since the object was saved. If you are loading a custom class that has failed,
you can use the remapping argument; see the sc.loadobj() docstring for details.
Otherwise, you might need to revert to the previous version of that module (e.g.
in a virtual environment), save in a format other than pickle, reload in a different
environment with the new version of that module, and then re-save as a pickle.

For general information on unpickling errors, see e.g.:
https://wiki.python.org/moin/UsingPickle
https://stackoverflow.com/questions/41554738/how-to-load-an-old-pickle-file
'''

    # Load the file
    try:
        with gz.GzipFile(**fileargs) as fileobj:
            filestr = fileobj.read() # Convert it to a string
    except Exception as E: # pragma: no cover
        exc = type(E) # Figure out what kind of error it is
        if exc == FileNotFoundError: # This is simple, just raise directly
            raise E
        elif exc == gz.BadGzipFile:
            try: # If the gzip file failed, first try as a regular object
                with open(filename, 'rb') as fileobj:
                    filestr = fileobj.read() # Convert it to a string
            except:
                raise exc(gziperror) from E

    # Unpickle it
    try:
        obj = _unpickler(filestr, filename=filename, verbose=verbose, die=die, remapping=remapping, method=method, **kwargs) # Actually load it
    except Exception as E: # pragma: no cover
        exc = type(E) # Figure out what kind of error it is
        errormsg = unpicklingerror + '\n\nSee the stack trace above for more information on this specific error.'
        raise exc(errormsg) from E

    # If it loaded but with errors, print them here
    if isinstance(obj, Failed):
        print(unpicklingerror)
    elif verbose:
        print(f'Object loaded from "{filename}"')

    return obj


def saveobj(filename=None, obj=None, compresslevel=5, verbose=0, folder=None, method='pickle', die=True, *args, **kwargs):
    '''
    Save an object to file as a gzipped pickle -- use compression 5 by default,
    since more is much slower but not much smaller. Once saved, can be loaded
    with sc.loadobj(). Note that saveobj()/save() are identical.

    Args:
        filename (str or Path): the filename to save to; if str, passed to sc.makefilepath()
        obj (literally anything): the object to save
        compresslevel (int): the level of gzip compression
        verbose (int): detail to print
        folder (str): passed to sc.makefilepath()
        method (str): whether to use pickle (default) or dill
        die (bool): whether to fail if no object is provided
        args (list): passed to pickle.dumps()
        kwargs (dict): passed to pickle.dumps()

    **Example**::

        myobj = ['this', 'is', 'a', 'weird', {'object':44}]
        sc.saveobj('myfile.obj', myobj)
        sc.saveobj('myfile.obj', myobj, method='dill') # Use dill instead, to save custom classes as well

    | New in version 1.1.1: removed Python 2 support.
    | New in version 1.2.2: automatic swapping of arguments if order is incorrect; correct passing of arguments
    '''

    # Handle path
    filetypes = (str, type(Path()), type(None))
    if isinstance(filename, Path): # If it's a path object, convert to string
        filename = str(filename)
    if filename is None: # If it doesn't exist, just create a byte stream
        bytesobj = io.BytesIO()
    if not isinstance(filename, filetypes): # pragma: no cover
        if isinstance(obj, filetypes):
            print(f'Warning: filename was not supplied as a valid type ({type(filename)}) but the object was ({type(obj)}); automatically swapping order')
            real_obj = filename
            real_file = obj
            filename = real_file
            obj = real_obj
        else:
            errormsg = f'Filename type {type(filename)} is not valid: must be one of {filetypes}'
    else: # Normal use case: make a file path
        bytesobj = None
        filename = makefilepath(filename=filename, folder=folder, default='default.obj', sanitize=True)

    # Handle object
    if obj is None: # pragma: no cover
        errormsg = 'No object was supplied to saveobj(), or the object was empty'
        if die:
            raise ValueError(errormsg)
        elif verbose:
            print(errormsg)

    # Actually save
    with gz.GzipFile(filename=filename, fileobj=bytesobj, mode='wb', compresslevel=compresslevel) as fileobj:
        if method == 'dill': # pragma: no cover # If dill is requested, use that
            if verbose>=2: print('Saving as dill...')
            _savedill(fileobj, obj)
        else: # Otherwise, try pickle
            try:
                if verbose>=2: print('Saving as pickle...')
                _savepickle(fileobj, obj, *args, **kwargs) # Use pickle
            except Exception as E: # pragma: no cover
                if verbose>=2: print(f'Exception when saving as pickle ({repr(E)}), saving as dill...')
                _savedill(fileobj, obj, *args, **kwargs) # ...but use Dill if that fails

    if verbose and filename:
        print(f'Object saved to "{filename}"')

    if filename:
        return filename
    else: # pragma: no cover
        bytesobj.seek(0)
        return bytesobj


# Aliases to make these core functions even easier to use
load = loadobj
save = saveobj

def loadstr(string, verbose=False, die=None, remapping=None):
    '''
    Like loadobj(), but for a bytes-like string (rarely used).

    **Example**::

        obj = sc.objdict(a=1, b=2)
        string1 = sc.dumpstr(obj)
        string2 = sc.loadstr(string1)
        assert string1 == string2
    '''
    with closing(IO(string)) as output: # Open a "fake file" with the Gzip string pickle in it.
        with gz.GzipFile(fileobj=output, mode='rb') as fileobj: # Set a Gzip reader to pull from the "file."
            picklestring = fileobj.read() # Read the string pickle from the "file" (applying Gzip decompression).
    obj = _unpickler(picklestring, filestring=string, verbose=verbose, die=die, remapping=remapping) # Return the object gotten from the string pickle.
    return obj


def dumpstr(obj=None):
    ''' Dump an object as a bytes-like string (rarely used); see ``sc.loadstr()`` '''
    with closing(IO()) as output: # Open a "fake file."
        with gz.GzipFile(fileobj=output, mode='wb') as fileobj:  # Open a Gzip-compressing way to write to this "file."
            try:    _savepickle(fileobj, obj) # Use pickle
            except: _savedill(fileobj, obj) # ...but use Dill if that fails
        output.seek(0) # Move the mark to the beginning of the "file."
        result = output.read() # Read all of the content into result.
    return result



##############################################################################
#%% Other file functions
##############################################################################

__all__ += ['loadtext', 'savetext', 'savezip', 'getfilelist', 'sanitizefilename', 'makefilepath', 'path', 'thisdir']


def loadtext(filename=None, folder=None, splitlines=False):
    '''
    Convenience function for reading a text file

    **Example**::

        mytext = sc.loadtext('my-document.txt')
    '''
    filename = makefilepath(filename=filename, folder=folder)
    with open(filename) as f: output = f.read()
    if splitlines: output = output.splitlines()
    return output


def savetext(filename=None, string=None):
    '''
    Convenience function for saving a text file -- accepts a string or list of strings.

    **Example**::

        text = ['Here', 'is', 'a', 'poem']
        sc.savetext('my-document.txt', text)
    '''
    if isinstance(string, list): string = '\n'.join(string) # Convert from list to string)
    if not scu.isstring(string):  string = str(string)
    filename = makefilepath(filename=filename)
    with open(filename, 'w') as f: f.write(string)
    return


def savezip(filename=None, filelist=None, folder=None, basename=True, verbose=True):
    '''
    Create a zip file from the supplied list of files

    **Example**::

        scripts = sc.getfilelist('./code/*.py')
        sc.savezip('scripts.zip', scripts)
    '''
    fullpath = makefilepath(filename=filename, folder=folder, sanitize=True)
    filelist = scu.promotetolist(filelist)
    with ZipFile(fullpath, 'w') as zf: # Create the zip file
        for thisfile in filelist:
            thispath = makefilepath(filename=thisfile, abspath=False)
            if basename: thisname = os.path.basename(thispath)
            else:        thisname = thispath
            zf.write(thispath, thisname)
    if verbose: print(f'Zip file saved to "{fullpath}"')
    return fullpath


def getfilelist(folder=None, pattern=None, abspath=False, nopath=False, filesonly=False, foldersonly=False, recursive=False, aspath=None):
    '''
    A shortcut for using glob.

    Args:
        folder      (str):  the folder to find files in (default, current)
        pattern     (str):  the pattern to match (default, wildcard); can be excluded if part of the folder
        abspath     (bool): whether to return the full path
        nopath      (bool): whether to return no path
        filesonly   (bool): whether to only return files (not folders)
        foldersonly (bool): whether to only return folders (not files)
        recursive   (bool): passed to glob()
        aspath      (bool): whether to return Path objects

    Returns:
        List of files/folders

    **Examples**::

        sc.getfilelist() # return all files and folders in current folder
        sc.getfilelist('~/temp', '*.py', abspath=True) # return absolute paths of all Python files in ~/temp folder
        sc.getfilelist('~/temp/*.py') # Like above

    New in version 1.1.0: "aspath" argument
    '''
    if folder is None:
        folder = '.'
    folder = os.path.expanduser(folder)
    if abspath:
        folder = os.path.abspath(folder)
    if os.path.isdir(folder) and pattern is None:
        pattern = '*'
    if aspath is None: aspath = scs.options.aspath
    globstr = os.path.join(folder, pattern) if pattern else folder
    filelist = sorted(glob(globstr, recursive=recursive))
    if filesonly:
        filelist = [f for f in filelist if os.path.isfile(f)]
    elif foldersonly:
        filelist = [f for f in filelist if os.path.isdir(f)]
    if nopath:
        filelist = [os.path.basename(f) for f in filelist]
    if aspath:
        filelist = [Path(f) for f in filelist]
    return filelist


def sanitizefilename(rawfilename):
    '''
    Takes a potentially Linux- and Windows-unfriendly candidate file name, and
    returns a "sanitized" version that is more usable.

    **Example**::

        bad_name = 'How*is*this*even*a*filename?!.doc'
        good_name = sc.sanitizefilename(bad_name) # Returns 'How_is_this_even_a_filename.doc'
    '''
    filtername = re.sub(r'[\!\?\"\'<>]', '', rawfilename) # Erase certain characters we don't want at all: !, ?, ", ', <, >
    filtername = re.sub(r'[:/\\\*\|,]', '_', filtername) # Change certain characters that might be being used as separators from what they were to underscores: space, :, /, \, *, |, comma
    return filtername # Return the sanitized file name.


def makefilepath(filename=None, folder=None, ext=None, default=None, split=False, aspath=None, abspath=True, makedirs=True, checkexists=None, sanitize=False, die=True, verbose=False):
    '''
    Utility for taking a filename and folder -- or not -- and generating a
    valid path from them. By default, this function will combine a filename and
    folder using os.path.join, create the folder(s) if needed with os.makedirs,
    and return the absolute path.

    Args:
        filename    (str or Path)   : the filename, or full file path, to save to -- in which case this utility does nothing
        folder      (str/Path/list) : the name of the folder to be prepended to the filename; if a list, fed to ``os.path.join()``
        ext         (str)           : the extension to ensure the file has
        default     (str or list)   : a name or list of names to use if filename is None
        split       (bool)          : whether to return the path and filename separately
        aspath      (bool)          : whether to return a Path object
        makedirs    (bool)          : whether or not to make the folders to save into if they don't exist
        checkexists (bool)          : if False/True, raises an exception if the path does/doesn't exist
        sanitize    (bool)          : whether or not to remove special characters from the path; see ``sc.sanitizefilename()`` for details
        verbose     (bool)          : how much detail to print

    Returns:
        filepath (str or Path): the validated path (or the folder and filename if split=True)

    **Simple example**::

        filepath = sc.makefilepath('myfile.obj') # Equivalent to os.path.abspath(os.path.expanduser('myfile.obj'))

    **Complex example**::

        filepath = makefilepath(filename=None, folder='./congee', ext='prj', default=[project.filename, project.name], split=True, abspath=True, makedirs=True)

    Assuming project.filename is None and project.name is "recipe" and ./congee
    doesn't exist, this will makes folder ./congee and returns e.g. ('/home/myname/congee', 'recipe.prj')

    New in version 1.1.0: "aspath" argument
    '''

    # Initialize
    filefolder = '' # The folder the file will be located in
    filebasename = '' # The filename
    if aspath is None: aspath = scs.options.aspath

    if isinstance(filename, Path):
        filename = str(filename)
    if isinstance(folder, Path): # If it's a path object, convert to string
        folder = str(folder)
    if isinstance(folder, list): # It's a list, join together
        folder = os.path.join(*folder)

    # Process filename
    if filename is None: # pragma: no cover
        defaultnames = scu.promotetolist(default) # Loop over list of default names
        for defaultname in defaultnames:
            if not filename and defaultname: filename = defaultname # Replace empty name with default name
    if filename is not None: # If filename exists by now, use it
        filebasename = os.path.basename(filename)
        filefolder = os.path.dirname(filename)
    if not filebasename: filebasename = 'default' # If all else fails

    # Add extension if it's defined but missing from the filebasename
    if ext and not filebasename.endswith(ext):
        filebasename += '.'+ext
    if verbose:
        print(f'From filename="{filename}", default="{default}", extension="{ext}", made basename "{filebasename}"')

    # Sanitize base filename
    if sanitize: filebasename = sanitizefilename(filebasename)

    # Process folder
    if folder is not None: # Replace with specified folder, if defined
        filefolder = folder
    if abspath: # Convert to absolute path
        filefolder = os.path.abspath(os.path.expanduser(filefolder))

    # Make sure folder exists
    if makedirs:
        try:
            os.makedirs(filefolder, exist_ok=True)
        except Exception as E: # pragma: no cover
            if die:
                raise E
            else:
                print(f'Could not create folders: {str(E)}')

    # Create the full path
    filepath = os.path.join(filefolder, filebasename)

    # Optionally check if it exists
    if checkexists is not None: # pragma: no cover
        exists = os.path.exists(filepath)
        errormsg = ''
        if exists and not checkexists:
            errormsg = f'File {filepath} should not exist, but it does'
            if die:
                raise FileExistsError(errormsg)
        if not exists and checkexists:
            errormsg = f'File {filepath} should exist, but it does not'
            if die:
                raise FileNotFoundError(errormsg)
        if errormsg:
            print(errormsg)

    # Decide on output
    if verbose:
        print(f'From filename="{filename}", folder="{folder}", made path name "{filepath}"')
    if split:
        output = filefolder, filebasename
    elif aspath:
        output = Path(filepath)
    else:
        output = filepath

    return output


def path(*args, **kwargs):
    ''' Alias to pathlib.Path(). New in version 1.2.2. '''
    return Path(*args, **kwargs)
path.__doc__ += '\n\n' + Path.__doc__


def thisdir(file=None, path=None, *args, aspath=None, **kwargs):
    '''
    Tiny helper function to get the folder for a file, usually the current file.
    If not supplied, then use the current file.

    Args:
        file (str): the file to get the directory from; usually __file__
        path (str/list): additional path to append; passed to os.path.join()
        args  (list): also passed to os.path.join()
        aspath (bool): whether to return a Path object instead of a string
        kwargs (dict): passed to Path()

    Returns:
        filepath (str): the full path to the folder (or filename if additional arguments are given)

    **Examples**::

        thisdir = sc.thisdir() # Get folder of calling file
        thisdir = sc.thisdir('.') # Ditto (usually)
        thisdir = sc.thisdir(__file__) # Ditto (usually)
        file_in_same_dir = sc.thisdir(path='new_file.txt')
        file_in_sub_dir = sc.thisdir('..', 'tests', 'mytests.py') # Merge parent folder with sufolders and a file
        np_dir = sc.thisdir(np) # Get the folder that Numpy is loaded from (assuming "import numpy as np")

    | New in version 1.1.0: "as_path" argument renamed "aspath"
    | New in version 1.2.2: "path" argument
    | New in version 1.3.0: allow modules
    '''
    if file is None: # No file: use the current folder
         file = str(Path(inspect.stack()[1][1])) # Adopted from Atomica
    elif hasattr(file, '__file__'): # It's actually a module
        file = file.__file__
    if aspath is None: aspath = scs.options.aspath
    folder = os.path.abspath(os.path.dirname(file))
    path = scu.mergelists(path, *args)
    filepath = os.path.join(folder, *path)
    if aspath:
        filepath = Path(filepath, **kwargs)
    return filepath


##############################################################################
#%% JSON functions
##############################################################################

__all__ += ['sanitizejson', 'jsonify', 'loadjson', 'savejson', 'jsonpickle', 'jsonunpickle']


def sanitizejson(obj, verbose=True, die=False, tostring=False, **kwargs):
    """
    This is the main conversion function for Python data-structures into
    JSON-compatible data structures (note: sanitizejson/jsonify are identical).

    Args:
        obj (any): almost any kind of data structure that is a combination of list, numpy.ndarray, odicts, etc.
        verbose (bool): level of detail to print
        die (bool): whether or not to raise an exception if conversion failed (otherwise, return a string)
        tostring (bool): whether to return a string representation of the sanitized object instead of the object itself
        kwargs (dict): passed to json.dumps() if tostring=True

    Returns:
        object (any or str): the converted object that should be JSON compatible, or its representation as a string if tostring=True

    Version: 2020apr11
    """
    if obj is None: # Return None unchanged
        output = None

    elif isinstance(obj, (bool, np.bool_)): # It's true/false
        output = bool(obj)

    elif scu.isnumber(obj): # It's a number
        if np.isnan(obj): # It's nan, so return None
            output = None
        else:
            if isinstance(obj, (int, np.int64)):
                output = int(obj) # It's an integer
            else:
                output = float(obj)# It's something else, treat it as a float

    elif scu.isstring(obj): # It's a string of some kind
        try:    string = str(obj) # Try to convert it to ascii
        except: string = obj # Give up and use original
        output = string

    elif isinstance(obj, np.ndarray): # It's an array, iterate recursively
        if obj.shape: output = [sanitizejson(p) for p in list(obj)] # Handle most cases, incluing e.g. array([5])
        else:         output = [sanitizejson(p) for p in list(np.array([obj]))] # Handle the special case of e.g. array(5)

    elif isinstance(obj, (list, set, tuple)): # It's another kind of interable, so iterate recurisevly
        output = [sanitizejson(p) for p in list(obj)]

    elif isinstance(obj, dict): # It's a dictionary, so iterate over the items
        output = {str(key):sanitizejson(val) for key,val in obj.items()}

    elif isinstance(obj, (dt.time, dt.date, dt.datetime)):
        output = str(obj)

    elif isinstance(obj, uuid.UUID):
        output = str(obj)

    elif callable(getattr(obj, 'to_dict', None)): # Handle e.g. pandas, where we want to return the object, not the string
        output = obj.to_dict()

    elif callable(getattr(obj, 'to_json', None)):
        output = obj.to_json()

    elif callable(getattr(obj, 'toJSON', None)):
        output = obj.toJSON()

    else: # None of the above
        try:
            output = jsonpickle(obj)
        except Exception as E: # pragma: no cover
            errormsg = f'Could not sanitize "{obj}" {type(obj)} ({str(E)}), converting to string instead'
            if die:       raise TypeError(errormsg)
            elif verbose: print(errormsg)
            output = str(obj)

    # Convert to string if desired
    if tostring:
        output = json.dumps(output, **kwargs)

    return output

# Define alias
jsonify = sanitizejson


def loadjson(filename=None, folder=None, string=None, fromfile=True, **kwargs):
    '''
    Convenience function for reading a JSON file (or string).

    Args:
        filename (str): the file to load, or the JSON object if using positional arguments
        folder (str): folder if not part of the filename
        string (str): if not loading from a file, a string representation of the JSON
        fromfile (bool): whether or not to load from file
        kwargs (dict): passed to json.load()

    Returns:
        output (dict): the JSON object

    **Examples**::

        json = sc.loadjson('my-file.json')
        json = sc.loadjson(string='{"a":null, "b":[1,2,3]}')
    '''
    if string is not None or not fromfile:
        if string is None and filename is not None:
            string = filename # Swap arguments
        output = json.loads(string, **kwargs)
    else:
        filepath = makefilepath(filename=filename, folder=folder)
        try:
            with open(filepath) as f:
                output = json.load(f, **kwargs)
        except FileNotFoundError as E: # pragma: no cover
            errormsg = f'No such file "{filename}". Use fromfile=False if loading a JSON string rather than a file.'
            raise FileNotFoundError(errormsg) from E
    return output



def savejson(filename=None, obj=None, folder=None, die=True, indent=2, keepnone=False, **kwargs):
    '''
    Convenience function for saving to a JSON file.

    Args:
        filename (str): the file to save
        obj (anything): the object to save; if not already in JSON format, conversion will be attempted
        folder (str): folder if not part of the filename
        die (bool): whether or not to raise an exception if saving an empty object
        indent (int): indentation to use for saved JSON
        keepnone (bool): allow ``sc.savejson(None)`` to return 'null' rather than raising an exception
        kwargs (dict): passed to json.dump()

    Returns:
        None

    **Example**::

        json = {'foo':'bar', 'data':[1,2,3]}
        sc.savejson('my-file.json', json)
    '''

    filename = makefilepath(filename=filename, folder=folder)

    if obj is None and not keepnone: # pragma: no cover
        errormsg = 'No object was supplied to savejson(), or the object was empty'
        if die: raise ValueError(errormsg)
        else:   print(errormsg)

    with open(filename, 'w') as f:
        json.dump(sanitizejson(obj), f, indent=indent, **kwargs)

    return


def jsonpickle(obj, tostring=False):
    '''
    Save any Python object to a JSON file using jsonpickle.

    Args:
        obj (any): the object to pickle as a JSON
        tostring (bool): whether to return a string (rather than the JSONified Python object)

    Returns:
        Either a string or a Python object for the JSON

    Wrapper for the jsonpickle library: https://jsonpickle.github.io/
    '''
    import jsonpickle as jp
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle.ext.pandas as jsonpickle_pandas
    jsonpickle_numpy.register_handlers()
    jsonpickle_pandas.register_handlers()

    if tostring:
        output = jp.dumps(obj)
    else:
        pickler = jp.pickler.Pickler()
        output = pickler.flatten(obj)

    return output


def jsonunpickle(json):
    ''' Use jsonpickle to restore an object (see jsonpickle()) '''
    import jsonpickle as jp
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle.ext.pandas as jsonpickle_pandas
    jsonpickle_numpy.register_handlers()
    jsonpickle_pandas.register_handlers()

    if isinstance(json, str):
        output = jp.loads(json)
    else:
        unpickler = jp.unpickler.Unpickler()
        output = unpickler.restore(json)

    return output


##############################################################################
#%% Spreadsheet functions
##############################################################################

__all__ += ['Blobject', 'Spreadsheet', 'loadspreadsheet', 'savespreadsheet']


class Blobject(object):
    '''
    A wrapper for a binary file -- rarely used directly.

    So named because it's an object representing a blob.

    "source" is a specification of where to get the data from. It can be anything
    supported by Blobject.load() which are (a) a filename, which will get loaded,
    or (b) a io.BytesIO which will get dumped into this instance

    Alternatively, can specify ``blob`` which is a binary string that gets stored directly
    in the ``blob`` attribute
    '''

    def __init__(self, source=None, name=None, filename=None, blob=None):
        # Handle inputs
        if source   is None and filename is not None: source   = filename # Reset the source to be the filename, e.g. Spreadsheet(filename='foo.xlsx')
        if filename is None and scu.isstring(source):  filename = source   # Reset the filename to be the source, e.g. Spreadsheet('foo.xlsx')
        if name     is None and filename is not None: name     = os.path.basename(filename) # If not supplied, use the filename
        if blob is not None and source is not None: raise ValueError('Can initialize from either source or blob, but not both')

        # Define quantities
        self.name      = name # Name of the object
        self.filename  = filename # Filename (used as default for load/save)
        self.created   = scd.now() # When the object was created
        self.modified  = scd.now() # When it was last modified
        self.blob  = blob # The binary data
        self.bytes = None # The filestream representation of the binary data
        if source is not None: self.load(source)
        return


    def __repr__(self):
        return scp.prepr(self, skip=['blob','bytes'])


    def load(self, source=None):
        '''
        This function loads the spreadsheet from a file or object. If no input argument is supplied,
        then it will read self.bytes, assuming it exists.
        '''
        def read_bin(source):
            ''' Helper to read a binary stream '''
            source.flush()
            source.seek(0)
            output = source.read()
            return output

        def read_file(filename):
            ''' Helper to read an actual file '''
            filepath = makefilepath(filename=filename)
            self.filename = filename
            with open(filepath, mode='rb') as f:
                output = f.read()
            return output

        if isinstance(source, Path):  # If it's a path object, convert to string
            source = str(source)

        if source is None:
            if self.bytes is not None:
                self.blob = read_bin(self.bytes)
                self.bytes = None # Once read in, delete
            else:
                if self.filename is not None:
                    self.blob = read_file(self.filename)
                else:
                    print('Nothing to load: no source or filename supplied and self.bytes is empty.')
        else:
            if isinstance(source,io.BytesIO):
                self.blob = read_bin(source)
            elif scu.isstring(source):
                self.blob = read_file(source)
            else:
                errormsg = f'Input source must be type string (for a filename) or BytesIO, not {type(source)}'
                raise TypeError(errormsg)

        self.modified = scd.now()
        return


    def save(self, filename=None):
        ''' This function writes the spreadsheet to a file on disk. '''
        filepath = makefilepath(filename=filename)
        with open(filepath, mode='wb') as f:
            f.write(self.blob)
        self.filename = filename
        print(f'Object saved to {filepath}.')
        return filepath


    def tofile(self, output=True):
        '''
        Return a file-like object with the contents of the file.

        This can then be used to open the workbook from memory without writing anything to disk e.g.

            - book = openpyxl.load_workbook(self.tofile())
            - book = xlrd.open_workbook(file_contents=self.tofile().read())
        '''
        bytesblob = io.BytesIO(self.blob)
        if output:
            return bytesblob
        else:
            self.bytes = bytesblob
            return


    def freshbytes(self):
        ''' Refresh the bytes object to accept new data '''
        self.bytes = io.BytesIO()
        return self.bytes



class Spreadsheet(Blobject):
    '''
    A class for reading and writing Excel files in binary format. No disk IO needs
    to happen to manipulate the spreadsheets with openpyxl (or xlrd or pandas).

    New in version 1.3.0: Changed default from xlrd to openpyxl and added self.wb
    attribute to avoid the need to reload workbooks.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wb = None
        return


    def _reload_wb(self, reload=None):
        ''' Helper function to check if workbook is already loaded '''
        output = (not hasattr(self, 'wb')) or (self.wb is None) or reload
        return output


    def xlrd(self, reload=False, store=True, **kwargs): # pragma: no cover
        ''' Return a book as opened by xlrd '''
        wb = self.wb
        if self._reload_wb(reload=reload):
            try:
                import xlrd # Optional import
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError('The "xlrd" Python package is not available; please install manually') from e
            wb = xlrd.open_workbook(file_contents=self.tofile().read(), **kwargs)
        if store:
            self.wb = wb
        return wb


    def openpyxl(self, reload=False, store=True, **kwargs):
        ''' Return a book as opened by openpyxl '''
        wb = self.wb
        if self._reload_wb(reload=reload):
            import openpyxl # Optional import
            if self.blob is not None:
                self.tofile(output=False)
                wb = openpyxl.load_workbook(self.bytes, **kwargs) # This stream can be passed straight to openpyxl
            else:
                wb = openpyxl.Workbook(**kwargs)
        if store:
            self.wb = wb
        return wb


    def openpyexcel(self, *args, **kwargs):
        ''' Legacy name for openpyxl() '''
        warnmsg = '''
Spreadsheet() no longer supports openpyexcel as of v1.3.1. To load using it anyway, you can manually do:

    %pip install openpyexcel
    import openpyexcel
    spreadsheet = sc.Spreadsheet()
    spreadsheet.wb = openpyexcel.Workbook(...)

Falling back to openpyxl, which is identical except for how cached cell values are handled.
'''
        warnings.warn(warnmsg, category=DeprecationWarning, stacklevel=2)
        return self.openpyxl(*args, **kwargs)


    def pandas(self, reload=False, store=True, **kwargs): # pragma: no cover
        ''' Return a book as opened by pandas '''
        wb = self.wb
        if self._reload_wb(reload=reload):
            import pandas as pd # Optional import
            if self.blob is not None:
                self.tofile(output=False)
                wb = pd.ExcelFile(self.bytes, **kwargs)
            else:
                errormsg = 'For pandas, must load an existing workbook; use openpyxl to create a new workbook'
                raise FileNotFoundError(errormsg)
        if store:
            self.wb = wb
        return wb


    def update(self, book): # pragma: no cover
        ''' Updated the stored spreadsheet with book instead '''
        self.tofile(output=False)
        book.save(self.freshbytes())
        self.load()
        return


    def _getsheet(self, sheetname=None, sheetnum=None):
        if   sheetname is not None: sheet = self.wb[sheetname]
        elif sheetnum  is not None: sheet = self.wb[self.wb.sheetnames[sheetnum]]
        else:                       sheet = self.wb.active
        return sheet


    def readcells(self, wbargs=None, *args, **kwargs):
        ''' Alias to loadspreadsheet() '''
        method = kwargs.pop('method', 'openpyxl')
        wbargs = scu.mergedicts(wbargs)
        f = self.tofile()
        kwargs['fileobj'] = f

        # Read in sheetoutput (sciris dataframe object for xlrd, 2D numpy array for openpyxl).
        if method == 'xlrd': # pragma: no cover
            sheetoutput = loadspreadsheet(*args, **kwargs, method='xlrd')  # returns sciris dataframe object
        elif method in ['openpyxl', 'openpyexcel']:
            wb_reader = self.openpyxl if method == 'openpyxl' else self.openpyexcel
            wb_reader(**wbargs)
            ws = self._getsheet(sheetname=kwargs.get('sheetname'), sheetnum=kwargs.get('sheetname'))
            rawdata = tuple(ws.rows)
            sheetoutput = np.empty(np.shape(rawdata), dtype=object)
            for r,rowdata in enumerate(rawdata):
                for c,val in enumerate(rowdata):
                    sheetoutput[r][c] = rawdata[r][c].value
        else: # pragma: no cover
            errormsg = f'Reading method not found; must be openpyxl or xlrd, not {method}'
            raise ValueError(errormsg)

        # Return the appropriate output
        cells = kwargs.get('cells')
        if cells is None:  # If no cells specified, return the whole sheet.
            return sheetoutput
        else:
            results = []
            for cell in cells:  # Loop over all cells
                rownum = cell[0]
                colnum = cell[1]
                if method == 'xlrd':  # If we're using xlrd, reduce the row number by 1.
                    rownum -= 1
                results.append(sheetoutput[rownum][colnum])  # Grab and append the result at the cell.
            return results


    def writecells(self, cells=None, startrow=None, startcol=None, vals=None, sheetname=None, sheetnum=None, verbose=False, wbargs=None):
        '''
        Specify cells to write. Can supply either a list of cells of the same length
        as the values, or else specify a starting row and column and write the values
        from there.

        **Examples**::

            S = sc.Spreadsheet()
            S.writecells(cells=['A6','B7'], vals=['Cat','Dog']) # Method 1
            S.writecells(cells=[np.array([2,3])+i for i in range(2)], vals=['Foo', 'Bar']) # Method 2
            S.writecells(startrow=14, startcol=1, vals=np.random.rand(3,3)) # Method 3
            S.save('myfile.xlsx')
        '''
        # Load workbook
        if wbargs is None: wbargs = {}
        wb = self.openpyxl(**wbargs)
        if verbose: print(f'Workbook loaded: {wb}')

        # Get right worksheet
        ws = self._getsheet(sheetname=sheetname, sheetnum=sheetnum)
        if verbose: print(f'Worksheet loaded: {ws}')

        # Determine the cells
        if cells is not None: # A list of cells is supplied
            cells = scu.promotetolist(cells)
            vals  = scu.promotetolist(vals)
            if len(cells) != len(vals): # pragma: no cover
                errormsg = f'If using cells, cells and vals must have the same length ({len(cells)} vs. {len(vals)})'
                raise ValueError(errormsg)
            for cell,val in zip(cells,vals):
                try:
                    if scu.isstring(cell): # Handles e.g. cell='A1'
                        cellobj = ws[cell]
                    elif scu.checktype(cell, 'arraylike','number') and len(cell)==2: # Handles e.g. cell=(0,0)
                        cellobj = ws.cell(row=cell[0], column=cell[1])
                    else:
                        errormsg = f'Cell must be formatted as a label or row-column pair, e.g. "A1" or (3,5); not "{cell}"'
                        raise TypeError(errormsg)
                    if verbose: print(f'  Cell {cell} = {val}')
                    if isinstance(val,tuple):
                        cellobj.value = val[0]
                        cellobj.cached_value = val[1]
                    else:
                        cellobj.value = val
                except Exception as E: # pragma: no cover
                    errormsg = f'Could not write "{val}" to cell "{cell}": {repr(E)}'
                    raise RuntimeError(errormsg)
        else:# Cells aren't supplied, assume a matrix
            if startrow is None: startrow = 1 # Excel uses 1-based indexing
            if startcol is None: startcol = 1
            valarray = np.atleast_2d(np.array(vals, dtype=object))
            for i,rowvals in enumerate(valarray):
                row = startrow + i
                for j,val in enumerate(rowvals):
                    col = startcol + j
                    try:
                        key = f'row:{row} col:{col}'
                        ws.cell(row=row, column=col, value=val)
                        if verbose: print(f'  Cell {key} = {val}')
                    except Exception as E: # pragma: no cover
                        errormsg = f'Could not write "{val}" to {key}: {repr(E)}'
                        raise RuntimeError(errormsg)

        # Save
        wb.save(self.freshbytes())
        self.load()

        return


    def save(self, filename='spreadsheet.xlsx'):
        filepath = makefilepath(filename=filename, ext='xlsx')
        Blobject.save(self, filepath)



def loadspreadsheet(filename=None, folder=None, fileobj=None, sheet=0, asdataframe=None, header=True, method='pandas', **kwargs):
    '''
    Load a spreadsheet as a dataframe or a list of lists.

    By default, an alias to ``pandas.read_excel()`` with a header, but also supports loading
    via openpyxl or xlrd. Read from either a filename or a file object.

    Args:
        filename (str): filename or path to read
        folder (str): optional folder to use with the filename
        fileobj (obj): load from file object rather than path
        sheet (str/int/list): name or number of sheet(s) to use (default 0)
        asdataframe (bool): whether to return as a pandas/Sciris dataframe (default True)
        method (str): how to read (default 'pandas', other choices 'openpyxl' and 'xlrd')
        kwargs (dict): passed to pd.read_excel(), openpyxl(), etc.

    **Examples**::

        df = sc.loadspreadsheet('myfile.xlsx') # Alias to pd.read_excel(header=1)
        wb = sc.loadspreadsheet('myfile.xlsx', method='openpyxl') # Returns workbook
        data = sc.loadspreadsheet('myfile.xlsx', method='xlrd', asdataframe=False) # Returns raw data; requires xlrd

    New in version 1.3.0: change default from xlrd to pandas; renamed sheetname and sheetnum arguments to sheet.
    '''

    # Handle path and sheet name/number
    fullpath = makefilepath(filename=filename, folder=folder)
    for key in ['sheetname', 'sheetnum', 'sheet_name']:
        sheet = kwargs.pop('sheetname', sheet)

    # Load using pandas
    if method == 'pandas':
        import pandas as pd # Optional import
        if fileobj is not None: fullpath = fileobj # Substitute here for reading
        if header  is not None: header = np.arange(header)
        data = pd.read_excel(fullpath, sheet_name=sheet, header=header, **kwargs)
        if asdataframe is False:
            pass
        return data

    # Load using openpyxl
    elif method == 'openpyxl': # pragma: no cover
        spread = Spreadsheet(fullpath)
        wb = spread.openpyxl(**kwargs)
        return wb

    # Legacy method -- xlrd
    elif method == 'xlrd': # pragma: no cover
        try:
            import xlrd # Optional import

            # Handle inputs
            if asdataframe is None: asdataframe = True
            if isinstance(filename, io.BytesIO): fileobj = filename # It's actually a fileobj
            if fileobj is None:
                book = xlrd.open_workbook(fullpath)
            else:
                book = xlrd.open_workbook(file_contents=fileobj.read())

            if scu.isnumber(sheet):
                ws = book.sheet_by_index(sheet)
            else:
                ws = book.sheet_by_name(sheet)

            # Load the raw data
            rawdata = []
            for rownum in range(ws.nrows-header):
                rawdata.append(sco.odict())
                for colnum in range(ws.ncols):
                    if header: attr = ws.cell_value(0,colnum)
                    else:      attr = f'Column {colnum}'
                    attr = scu.uniquename(attr, namelist=rawdata[rownum].keys(), style='(%d)')
                    val = ws.cell_value(rownum+header,colnum)
                    try:
                        val = float(val) # Convert it to a number if possible
                    except:
                        try:    val = str(val)  # But give up easily and convert to a string (not Unicode)
                        except: pass # Still no dice? Fine, we tried
                    rawdata[rownum][str(attr)] = val

            # Convert to dataframe
            if asdataframe:
                cols = rawdata[0].keys()
                reformatted = []
                for oldrow in rawdata:
                    newrow = list(oldrow[:])
                    reformatted.append(newrow)
                dfdata = scdf.dataframe(cols=cols, data=reformatted)
                return dfdata

            # Or leave in the original format
            else:
                return rawdata
        except Exception as E:
            if isinstance(E, ModuleNotFoundError):
                errormsg = 'The "xlrd" Python package is not available; please install manually via "pip install xlrd==1.2.0"'
                raise ModuleNotFoundError(errormsg) from E
            elif isinstance(E, AttributeError):
                errormsg = '''
Warning! xlrd has been deprecated in Python 3.9 and can no longer read XLSX files.
If the error you got above is

    "AttributeError: 'ElementTree' object has no attribute 'getiterator'"

then this should fix it:

    import xlrd
    xlrd.xlsx.ensure_elementtree_imported(False, None)
    xlrd.xlsx.Element_has_iter = True

Then try again to load your Excel file.
'''
                raise AttributeError(errormsg) from E
            else:
                raise E

    else: # pragma: no cover
        errormsg = f'Method "{method}" not found: must be one of pandas, openpyxl, or xlrd'
        raise ValueError(errormsg)


def savespreadsheet(filename=None, data=None, folder=None, sheetnames=None, close=True, formats=None, formatdata=None, verbose=False):
    '''
    Not-so-little function to format data nicely for Excel.

    Note: this function, while not deprecated, is not actively maintained.

    **Examples**::

        import sciris as sc
        import pylab as pl

        # Simple example
        testdata1 = pl.rand(8,3)
        sc.savespreadsheet(filename='test1.xlsx', data=testdata1)

        # Include column headers
        test2headers = [['A','B','C']] # Need double to get right shape
        test2values = pl.rand(8,3).tolist()
        testdata2 = test2headers + test2values
        sc.savespreadsheet(filename='test2.xlsx', data=testdata2)

        # Multiple sheets
        testdata3 = [pl.rand(10,10), pl.rand(20,5)]
        sheetnames = ['Ten by ten', 'Twenty by five']
        sc.savespreadsheet(filename='test3.xlsx', data=testdata3, sheetnames=sheetnames)

        # Supply data as an odict
        testdata4 = sc.odict([('First sheet', pl.rand(6,2)), ('Second sheet', pl.rand(3,3))])
        sc.savespreadsheet(filename='test4.xlsx', data=testdata4, sheetnames=sheetnames)

        # Include formatting
        nrows = 15
        ncols = 3
        formats = {
            'header':{'bold':True, 'bg_color':'#3c7d3e', 'color':'#ffffff'},
            'plain': {},
            'big':   {'bg_color':'#ffcccc'}}
        testdata5  = pl.zeros((nrows+1, ncols), dtype=object) # Includes header row
        formatdata = pl.zeros((nrows+1, ncols), dtype=object) # Format data needs to be the same size
        testdata5[0,:] = ['A', 'B', 'C'] # Create header
        testdata5[1:,:] = pl.rand(nrows,ncols) # Create data
        formatdata[1:,:] = 'plain' # Format data
        formatdata[testdata5>0.7] = 'big' # Find "big" numbers and format them differently
        formatdata[0,:] = 'header' # Format header
        sc.savespreadsheet(filename='test5.xlsx', data=testdata5, formats=formats, formatdata=formatdata)
    '''
    try:
        import xlsxwriter # Optional import
    except ModuleNotFoundError as e: # pragma: no cover
        raise ModuleNotFoundError('The "xlsxwriter" Python package is not available; please install manually') from e
    fullpath = makefilepath(filename=filename, folder=folder, default='default.xlsx')
    datadict   = sco.odict()
    formatdict = sco.odict()
    hasformats = (formats is not None) and (formatdata is not None)

    # Handle input arguments
    if isinstance(data, dict) and sheetnames is None:
        if verbose: print('Data is a dict, taking sheetnames from keys')
        sheetnames = data.keys()
        datadict   = sco.odict(data) # Use directly, converting to odict first
        if hasformats: formatdict = sco.odict(formatdata) #  NB, might be None but should be ok
    elif isinstance(data, dict) and sheetnames is not None:
        if verbose: print('Data is a dict, taking sheetnames from input')
        if len(sheetnames) != len(data):
            errormsg = f'If supplying data as a dict as well as sheetnames, length must match ({len(data)} vs {len(sheetnames)})'
            raise ValueError(errormsg)
        datadict   = sco.odict(data) # Use directly, but keep original sheet names
        if hasformats: formatdict = sco.odict(formatdata)
    elif not isinstance(data, dict):
        if sheetnames is None:
            if verbose: print('Data is a simple array')
            sheetnames = ['Sheet1']
            datadict[sheetnames[0]]   = data # Set it explicitly
            formatdict[sheetnames[0]] = formatdata # Set it explicitly -- NB, might be None but should be ok
        else:
            if verbose: print('Data is a list, taking matching sheetnames from inputs')
            if len(sheetnames) == len(data):
                for s,sheetname in enumerate(sheetnames):
                    datadict[sheetname] = data[s] # Assume there's a 1-to-1 mapping
                    if hasformats: formatdict[sheetname] = formatdata[s]
            else:
                errormsg = f'Unable to figure out how to match {len(sheetnames)} sheet names with data of length {len(data)}'
                raise ValueError(errormsg)
    else:
        errormsg = f'Cannot figure out how to handle data of type: {type(data)}'
        raise TypeError(errormsg) # This shouldn't happen!

    # Create workbook
    if verbose: print(f'Creating file {fullpath}')
    workbook = xlsxwriter.Workbook(fullpath)

    # Optionally add formats
    if formats is not None:
        if verbose: print(f'  Adding {len(formats)} formats')
        workbook_formats = dict()
        for formatkey,formatval in formats.items():
            workbook_formats[formatkey] = workbook.add_format(formatval)
    else:
        thisformat = workbook.add_format({}) # Plain formatting

    # Actually write the data
    for sheetname in datadict.keys():
        if verbose: print(f'  Creating sheet {sheetname}')
        sheetdata   = datadict[sheetname]
        if hasformats: sheetformat = formatdict[sheetname]
        worksheet = workbook.add_worksheet(sheetname)
        for r,row_data in enumerate(sheetdata):
            if verbose: print(f'    Writing row {r}/{len(sheetdata)}')
            for c,cell_data in enumerate(row_data):
                if verbose: print(f'      Writing cell {c}/{len(row_data)}')
                if hasformats:
                    thisformat = workbook_formats[sheetformat[r][c]] # Get the actual format
                try:
                    if not scu.isnumber(cell_data):
                        cell_data = str(cell_data) # Convert everything except numbers to strings -- use formatting to handle the rest
                    worksheet.write(r, c, cell_data, thisformat) # Write with or without formatting
                except Exception as E: # pragma: no cover
                    errormsg = f'Could not write cell [{r},{c}] data="{cell_data}", format="{thisformat}": {str(E)}'
                    raise RuntimeError(errormsg)

    # Either close the workbook and write to file, or return it for further working
    if close:
        if verbose: print(f'Saving file {filename} and closing')
        workbook.close()
        return fullpath
    else:
        if verbose: print('Returning workbook')
        return workbook



##############################################################################
#%% Pickling support methods
##############################################################################

__all__ += ['Failed', 'Empty', 'loadobj2or3']


class Failed(object):
    ''' An empty class to represent a failed object loading '''
    failure_info = sco.odict()

    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        output = scp.objrepr(self) # This does not include failure_info since it's a class attribute
        output += self.showfailures(verbose=False, tostring=True)
        return output

    def showfailures(self, verbose=True, tostring=False):
        output = ''
        for f,failure in self.failure_info.enumvals():
            output += f'\nFailure {f+1} of {len(self.failure_info)}:\n'
            output += f'Module: {failure["module"]}\n'
            output += f'Class: {failure["class"]}\n'
            output += f'Error: {failure["error"]}\n'
            if verbose:
                output += '\nTraceback:\n'
                output += failure['exception']
                output += '\n\n'
        if tostring:
            return output
        else:
            print(output)
            return


class Empty(object):
    ''' Another empty class to represent a failed object loading, but do not proceed with setstate '''

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class UniversalFailed(Failed):
    ''' A universal failed object class, that preserves as much data as possible '''

    def __init__(self, *args, **kwargs):
        if args:
            self.args = args
        if kwargs:
            self.kwargs = kwargs
        self.__set_empty()
        return

    def __set_empty(self):
        if not hasattr(self, 'state'):
            self.state = {}
        if not hasattr(self, 'dict'):
            self.dict = {}
        return

    def __repr__(self):
        output = scp.objrepr(self) # This does not include failure_info since it's a class attribute
        return output

    def __setstate__(self, state):
        self.__set_empty()
        self.state = state
        return

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value
        return

    def disp(self, *args, **kwargs):
        return scu.pr(self, *args, **kwargs)


def makefailed(module_name=None, name=None, error=None, exception=None, universal=False):
    ''' Create a class -- not an object! -- that contains the failure info for a pickle that failed to load '''
    base = UniversalFailed if universal else Failed
    key = f'Failure {len(base.failure_info)+1}'
    base.failure_info[key] = sco.odict()
    base.failure_info[key]['module']    = module_name
    base.failure_info[key]['class']     = name
    base.failure_info[key]['error']     = error
    base.failure_info[key]['exception'] = exception
    return base


class _RobustUnpickler(pkl.Unpickler):
    ''' Try to import an object, and if that fails, return a Failed object rather than crashing '''

    def __init__(self, bytesio, fix_imports=True, encoding="latin1", errors="ignore", remapping=None):
        pkl.Unpickler.__init__(self, bytesio, fix_imports=fix_imports, encoding=encoding, errors=errors)
        self.remapping = remapping if remapping is not None else {}
        return

    def find_class(self, module_name, name, verbose=True):
        key = f'{module_name}.{name}'
        obj = self.remapping.get(key) # If the user has supplied the module directly
        if obj is None or (isinstance(obj, tuple) and len(obj)==2): # Either it's not in the remapping, or it's a tuple
            try:
                if obj is None: # Key not in remapping
                    load_module = module_name
                    load_name = name
                else: # It's a tuple of names, process
                    load_module = obj[0]
                    load_name = obj[1]
                module = importlib.import_module(load_module)
                obj = getattr(module, load_name)
            except Exception as E:
                if verbose: print(f'Unpickling warning: could not import {load_module}.{load_name}: {str(E)}')
                exception = traceback.format_exc() # Grab the trackback stack
                obj = makefailed(module_name=module_name, name=name, error=E, exception=exception)
        return obj


class _UltraRobustUnpickler(pkl.Unpickler):
    ''' If all else fails, just make a default object '''

    def __init__(self, bytesio, *args, unpicklingerrors=None, **kwargs):
        pkl.Unpickler.__init__(self, bytesio, *args, **kwargs)
        self.unpicklingerrors = unpicklingerrors if unpicklingerrors is not None else []
        return

    def find_class(self, module_name, name):
        ''' Ignore all attempts to use the actual class and always make a UniversalFailed class '''
        error = scu.strjoin(self.unpicklingerrors)
        exception = self.unpicklingerrors
        obj = makefailed(module_name=module_name, name=name, error=error, exception=exception, universal=True)
        return obj


def _unpickler(string=None, filename=None, filestring=None, die=None, verbose=False, remapping=None, method='pickle', **kwargs):
    ''' Not invoked directly; used as a helper function for saveobj/loadobj '''

    if die is None: die = False
    try: # Try pickle first
        if method == 'pickle':
            obj = pkl.loads(string, **kwargs) # Actually load it -- main usage case
        elif method == 'dill':
            import dill # Optional Sciris dependency
            obj = dill.loads(string, **kwargs) # Actually load it, with dill
        else:
            errormsg = f'Method "{method}" not recognized, must be pickle or dill'
            raise ValueError(errormsg)
    except Exception as E1:
        if die:
            raise E1
        else:
            try:
                if verbose: print(f'Standard unpickling failed ({str(E1)}), trying encoding...')
                obj = pkl.loads(string, encoding='latin1', **kwargs) # Try loading it again with different encoding
            except Exception as E2:
                try:
                    if verbose: print(f'Encoded unpickling failed ({str(E2)}), trying dill...')
                    import dill # Optional Sciris dependency
                    obj = dill.loads(string, **kwargs) # If that fails, try dill
                except Exception as E3:
                    try:
                        if verbose: print(f'Dill failed ({str(E3)}), trying robust unpickler...')
                        obj = _RobustUnpickler(io.BytesIO(string), remapping=remapping).load() # And if that fails, throw everything at it
                    except Exception as E4:
                        try:
                            if verbose: print(f'Robust failed ({str(E4)}), trying ultrarobust unpickler...')
                            obj = _UltraRobustUnpickler(io.BytesIO(string), unpicklingerrors=[E1, E2, E3, E4]).load() # And if that fails, really throw everything at it
                        except Exception as E5: # pragma: no cover
                            errormsg = f'''
All available unpickling methods failed:
    Standard: {E1}
     Encoded: {E2}
        Dill: {E3}
      Robust: {E4}
 Ultrarobust: {E5}'''
                            raise Exception(errormsg)

    if isinstance(obj, Failed):
        print('Warning, the following errors were encountered during unpickling:')
        obj.showfailures(verbose=False)

    return obj


def _savepickle(fileobj=None, obj=None, protocol=None, *args, **kwargs):
        ''' Use pickle to do the salty work. '''
        if protocol is None:
            protocol = 4 # Use protocol 4 for backwards compatibility
        fileobj.write(pkl.dumps(obj, protocol=protocol, *args, **kwargs))
        return


def _savedill(fileobj=None, obj=None, *args, **kwargs): # pragma: no cover
    ''' Use dill to do the sour work (note: this function is not actively maintained) '''
    try:
        import dill # Optional Sciris dependency
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('The "dill" Python package is not available; please install manually') from e
    fileobj.write(dill.dumps(obj, protocol=-1, *args, **kwargs))
    return



##############################################################################
#%% Python 2 legacy support
##############################################################################

not_string_pickleable = ['datetime', 'BytesIO']
byte_objects = ['datetime', 'BytesIO', 'odict', 'spreadsheet', 'blobject']


def loadobj2or3(filename=None, filestring=None, recursionlimit=None, **kwargs):  # pragma: no cover
    '''
    Try to load as a (Sciris-saved) Python 3 pickle; if that fails, try to load
    as a Python 2 pickle. For legacy support only.

    For available keyword arguments, see sc.load().

    Args:
        filename (str): the name of the file to load
        filestring (str): alternatively, specify an already-loaded bytestring
        recursionlimit (int): how deeply to parse objects before failing (default 1000)
    '''
    try:
        output = loadobj(filename=filename, **kwargs)
    except:
        output = _loadobj2to3(filename=filename, filestring=filestring, recursionlimit=recursionlimit)
    return output


def _loadobj2to3(filename=None, filestring=None, recursionlimit=None): # pragma: no cover
    '''
    Used by loadobj2or3() to load Python2 objects in Python3 if all other
    loading methods fail. Uses a recursive approach, so can set a recursion limit.
    '''

    class Placeholder():
        ''' Replace these corrupted classes with properly loaded ones '''
        def __init__(*args):
            return

        def __setstate__(self, state):
            if isinstance(state,dict):
                self.__dict__ = state
            else:
                self.state = state
            return

    class StringUnpickler(pkl.Unpickler):
        def find_class(self, module, name, verbose=False):
            if verbose: print('Unpickling string module %s , name %s' % (module, name))
            if name in not_string_pickleable:
                return Empty
            else:
                try:
                    output = pkl.Unpickler.find_class(self,module,name)
                except Exception as E:
                    print('Warning, string unpickling could not find module %s, name %s: %s' % (module, name, str(E)))
                    output = Empty
                return output

    class BytesUnpickler(pkl.Unpickler):
        def find_class(self, module, name, verbose=False):
            if verbose: print('Unpickling bytes module %s , name %s' % (module, name))
            if name in byte_objects:
                try:
                    output = pkl.Unpickler.find_class(self,module,name)
                except Exception as E:
                    print('Warning, bytes unpickling could not find module %s, name %s: %s' % (module, name, str(E)))
                    output = Placeholder
                return output
            else:
                return Placeholder

    def recursive_substitute(obj1, obj2, track=None, recursionlevel=0, recursionlimit=None):
        if recursionlimit is None: # Recursion limit
            recursionlimit = 1000 # Better to die here than hit Python's recursion limit

        def recursion_warning(count, obj1, obj2):
            output = 'Warning, internal recursion depth exceeded, aborting: depth=%s, %s -> %s' % (count, type(obj1), type(obj2))
            return output

        recursionlevel += 1

        if track is None:
            track = []

        if isinstance(obj1, Blobject): # Handle blobjects (usually spreadsheets)
            obj1.blob  = obj2.__dict__[b'blob']
            obj1.bytes = obj2.__dict__[b'bytes']

        if isinstance(obj2, dict): # Handle dictionaries
            for k,v in obj2.items():
                if isinstance(v, dt.datetime):
                    setattr(obj1, k.decode('latin1'), v)
                elif isinstance(v, dict) or hasattr(v,'__dict__'):
                    if isinstance(k, (bytes, bytearray)):
                        k = k.decode('latin1')
                    track2 = track.copy()
                    track2.append(k)
                    if recursionlevel<=recursionlimit:
                        recursionlevel = recursive_substitute(obj1[k], v, track2, recursionlevel, recursionlimit)
                    else:
                        print(recursion_warning(recursionlevel, obj1, obj2))
        else:
            for k,v in obj2.__dict__.items():
                if isinstance(v, dt.datetime):
                    setattr(obj1,k.decode('latin1'), v)
                elif isinstance(v,dict) or hasattr(v,'__dict__'):
                    if isinstance(k, (bytes, bytearray)):
                        k = k.decode('latin1')
                    track2 = track.copy()
                    track2.append(k)
                    if recursionlevel<=recursionlimit:
                        recursionlevel = recursive_substitute(getattr(obj1,k), v, track2, recursionlevel, recursionlimit)
                    else:
                        print(recursion_warning(recursionlevel, obj1, obj2))
        return recursionlevel

    def loadintostring(fileobj):
        unpickler1 = StringUnpickler(fileobj, encoding='latin1')
        try:
            stringout = unpickler1.load()
        except Exception as E:
            print('Warning, string pickle loading failed: %s' % str(E))
            exception = traceback.format_exc() # Grab the trackback stack
            stringout = makefailed(module_name='String unpickler failed', name='n/a', error=E, exception=exception)
        return stringout

    def loadintobytes(fileobj):
        unpickler2 = BytesUnpickler(fileobj,  encoding='bytes')
        try:
            bytesout  = unpickler2.load()
        except Exception as E:
            print('Warning, bytes pickle loading failed: %s' % str(E))
            exception = traceback.format_exc() # Grab the trackback stack
            bytesout = makefailed(module_name='Bytes unpickler failed', name='n/a', error=E, exception=exception)
        return bytesout

    # Load either from file or from string
    if filename:
        with gz.GzipFile(filename) as fileobj:
            stringout = loadintostring(fileobj)
        with gz.GzipFile(filename) as fileobj:
            bytesout = loadintobytes(fileobj)

    elif filestring:
        with closing(IO(filestring)) as output:
            with gz.GzipFile(fileobj=output, mode='rb') as fileobj:
                stringout = loadintostring(fileobj)
        with closing(IO(filestring)) as output:
            with gz.GzipFile(fileobj=output, mode='rb') as fileobj:
                bytesout = loadintobytes(fileobj)
    else:
        errormsg = 'You must supply either a filename or a filestring for loadobj() or loadstr(), respectively'
        raise Exception(errormsg)

    # Actually do the load, with correct substitution
    recursive_substitute(stringout, bytesout, recursionlevel=0, recursionlimit=recursionlimit)
    return stringout



##############################################################################
#%% Twisted pickling methods
##############################################################################

# NOTE: The code below is part of the Twisted package, and is included
# here to allow functools.partial() objects (among other things) to be
# pickled; they are not for public consumption. --CK

# From: twisted/persisted/styles.py
# -*- test-case-name: twisted.test.test_persisted -*-
# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE for details.

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

_UniversalPicklingError = pkl.PicklingError

def _pickleMethod(method):
    return (_unpickleMethod, (method.__name__,         method.__self__, method.__self__.__class__))

def _methodFunction(classObject, methodName):
    methodObject = getattr(classObject, methodName)
    return methodObject

def _unpickleMethod(im_name, im_self, im_class):
    if im_self is None:
        return getattr(im_class, im_name)
    try:
        methodFunction = _methodFunction(im_class, im_name)
    except AttributeError: # pragma: no cover
        assert im_self is not None, "No recourse: no instance to guess from."
        if im_self.__class__ is im_class:
            raise
        return _unpickleMethod(im_name, im_self, im_self.__class__)
    else:
        maybeClass = ()
        bound = types.MethodType(methodFunction, im_self, *maybeClass)
        return bound

cpreg.pickle(types.MethodType, _pickleMethod, _unpickleMethod)

# Legacy support for loading Sciris <1.0 objects; may be removed in future
pickleMethod = _pickleMethod
unpickleMethod = _unpickleMethod
