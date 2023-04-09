'''
Functions for checking and saving versioning information, such as Python package
versions, git versions, etc.

Highlights:
    - :func:`freeze`: programatically store "pip freeze" output
    - :func:`require`: require a specific version of a package
    - :func:`gitinfo`: gets the git information (if available) of a given file
    - :func:`compareversions`: easy way to compare version numbers
    - :func:`storemetadata`: collects relevant metadata into a dictionary
    - :func:`savewithmetadata`: saves data as a zip file including versioning metadata
'''

import os
import re
import time
import zlib
import types
import packaging.version
from zipfile import ZipFile
from . import sc_utils as scu
from . import sc_fileio as scf
from . import sc_datetime as scd

__all__ = ['freeze', 'require', 'gitinfo', 'compareversions', 'getcaller', 
           'storemetadata', 'loadmetadata', 'savewithmetadata', 'loadwithmetadata']


# Define shared variables
_metadataflag = 'sciris_metadata' # Key name used to identify metadata saved in a figure
_mdata_fn = 'metadata.json' # Default filenames for metadata and data
_data_fn  = 'data.obj'


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
        reqlist = scu.mergelists(reqs, reqlist)

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
        errormsg += f'''\n\nIf this is a valid module, you might want to try "pip install {scu.strjoin(errkeys, sep=' ')} --upgrade".'''
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


def getcaller(frame=2, tostring=True, includelineno=False, includeline=False, relframe=0, die=False):
    '''
    Try to get information on the calling function, but fail gracefully. See also
    :func:`thisfile`.

    Frame 1 is the file calling this function, so not very useful. Frame 2 is
    the default assuming it is being called directly. Frame 3 is used if
    another function is calling this function internally.

    Args:
        frame (int): how many frames to descend (e.g. the caller of the caller of the...), default 2
        tostring (bool): whether to return a string instead of a dict with filename and line number
        includelineno (bool): if ``tostring``, whether to also include the line number
        includeline (bool): if not ``tostring``, also store the line contents
        relframe (int): relative frame -- another way of specifying the frame; added to "frame"
        die (bool): whether to raise an exception if calling information cannot be retrieved

    Returns:
        output (str/dict): the filename (and line number) of the calling function, either as a string or dict

    **Examples**::

        sc.getcaller()
        sc.getcaller(tostring=False)['filename'] # Equivalent to sc.getcaller()
        sc.getcaller(frame=3) # Descend one level deeper than usual
        sc.getcaller(frame=1, tostring=False, includeline=True) # See the line that called sc.getcaller()

    | New in version 1.0.0.
    | New in version 1.3.3: do not include line by default
    | New in version 2.2.0: "relframe" argument; "die" argument
    '''
    try:
        import inspect
        frame = frame + relframe
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
        if die:
            raise E
        else:
            if tostring:
                output = f'Calling function information not available ({str(E)})'
            else:
                output = {'filename':'N/A', 'lineno':'N/A'}
    return output


def storemetadata(paths=True, caller=True, git=True, pipfreeze=True, comments=None, outfile=None, relframe=0, **kwargs):
    '''
    Store common metadata: useful for exactly recreating (or remembering) the environment
    at a moment in time.
    
    Args:
        paths (bool): store the path to the calling file and Python executable (may include sensitive information)
        caller (bool): store info on the calling file
        git (bool): store git information on the calling file
        pipfreeze (bool): store the current Python environment, equivalent to "pip freeze"
        comments (str/dict): any other metadata to store
        outfile (str): if not None, then save as JSON to this filename
        relframe (int): how far to descend into the calling stack (if used directly, use 0; if called by another function, use 1; etc)
        kwargs (dict): additional data to store
        
    Returns:
        A dictionary with information on the date, plateform, executable, versions
        of key libraries (Sciris, Numpy, pandas, and Matplotlib), and the Python environment
        
    **Example**::
        
        metadata = sc.getmetadata()
        sc.compareversions(metadata['pandas_version'], '1.5.0')
    
    New in version 2.2.0.
    '''
    
    # Additional imports
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import sys
    from . import sc_version as scver
    
    # Store key metadata
    md = dict()
    md['date']       = scd.getdate()
    md['platform']   = scu.getplatform()
    md['executable'] = sys.executable if paths else None # NB, will likely include username info
    md['comments']   = comments
    
    # Store key version info
    md['python_version']     = sys.version
    md['sciris_version']     = scver.__version__
    md['numpy_version']      = np.__version__
    md['pandas_version']     = pd.__version__
    md['matplotlib_version'] = mpl.__version__
    
    # Store optional metadata
    caller = scu.getcaller(relframe=relframe+1, tostring=False)
    if caller and paths:
        md['calling_info'] = caller
    if git:
        md['git_info'] = scu.gitinfo(caller['filename'], die=False, verbose=False)
    if pipfreeze:
        md['pipfreeze'] = scu.freeze()
        
    # Store any additional data
    md.update(kwargs)
    
    if outfile is not None:
        outfile = scf.makefilepath(outfile)
        scf.savejson(outfile, md)
    
    return md


def loadmetadata(filename, load_all=False, die=True):
    '''
    Read metadata from a saved image; currently only PNG and SVG are supported.

    Only for use with images saved with ``sc.savefig()``. Metadata retrieval for PDF
    is not currently supported. To load metadata saved with ``sc.getmetadata()``, 
    use ``sc.loadjson()`` instead.

    Args:
        filename (str): the name of the file to load the data from
        load_all (bool): whether to load all metadata available in an image (else, just load what was saved by Sciris)
        die (bool): whether to raise an exception if the metadata can't be found

    **Example**::

        pl.plot([1,2,3], [4,2,6])
        sc.savefig('example.png')
        sc.loadmetadata('example.png')
    '''
    from . import sc_fileio as scf # To avoid circular import

    # Initialize
    metadata = {}
    lcfn = str(filename).lower() # Lowercase filename

    # Handle bitmaps
    is_png = lcfn.endswith('png')
    is_jpg = lcfn.endswith('jpg') or lcfn.endswith('jpeg')
    if is_png or is_jpg:
        try:
            import PIL
        except ImportError as E: # pragma: no cover
            errormsg = f'Pillow import failed ({str(E)}), please install first (pip install pillow)'
            raise ImportError(errormsg) from E
        im = PIL.Image.open(filename)
        keys = im.info.keys()

        # Usual case, can find metadata and is PNG
        if is_png and (load_all or _metadataflag in keys):
            if load_all:
                metadata = im.info
            else:
                jsonstr = im.info[_metadataflag]
                metadata = scf.loadjson(string=jsonstr)

        # JPG -- from https://www.thepythoncode.com/article/extracting-image-metadata-in-python
        elif is_jpg: # pragma: no cover
            from PIL.ExifTags import TAGS # Must be imported directly
            exifdata = im.getexif()
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode()
                metadata[tag] = data

        # Can't find metadata
        else: # pragma: no cover
            errormsg = f'Could not find "{_metadataflag}": metadata can only be extracted from figures saved with sc.savefig().\nAvailable keys are: {scu.strjoin(keys)}'
            if die:
                raise ValueError(errormsg)
            else:
                print(errormsg)
                metadata = im.info


    # Handle SVG
    elif lcfn.endswith('svg'): # pragma: no cover

        # Load SVG as text and parse it
        svg = scf.loadtext(filename).splitlines()
        flag = _metadataflag + '=' # Start of the line
        end = '</'

        found = False
        for line in svg:
            if flag in line:
                found = True
                break

        # Usual case, can find metadata
        if found:
            jsonstr = line[line.find(flag)+len(flag):line.find(end)]
            metadata = scf.loadjson(string=jsonstr)

        # Can't find metadata
        else:
            errormsg = f'Could not find the string "{_metadataflag}" anywhere in "{filename}": metadata can only be extracted from figures saved with sc.savefig()'
            if die:
                raise ValueError(errormsg)
            else:
                print(errormsg)
    
    elif lcfn.endswith('json'):
        metadata = scf.loadjson(filename) # Overkill, but ok

    # Other formats not supported
    else: # pragma: no cover
        errormsg = f'Filename "{filename}" has unsupported type: must be PNG, JPG, or SVG (PDF is not supported)'
        raise ValueError(errormsg)

    return metadata



__all__ += ['savewithmetadata', 'loadwithmetadata']


def savewithmetadata(filename, data, paths=True, caller=True, git=True, pip=True, comments=None,
                  mdata_fn=_mdata_fn, data_fn=_data_fn, **kwargs):

    # Get the metadata
    metadata = storemetadata(paths=paths, caller=caller, git=git, pip=pip, comments=comments, frame=3)
    
    # Convert both to strings
    metadatastr = scf.jsonify(metadata, tostring=True, indent=2)
    datastr     = scf.dumpstr(data)
    
    # Construct output
    datadict = {mdata_fn:metadatastr, data_fn:datastr}
    
    return scf.savezip(filename=filename, data=datadict, tobytes=False, **kwargs)
    

def loadwithmetadata(filename, folder=None, loadmetadata=False, mdata_fn=_mdata_fn, data_fn=_data_fn, **kwargs):
    
    filename = scf.makefilepath(filename=filename, folder=folder)
   
    with ZipFile(filename, 'r') as zf: # Create the zip file
        
        # Read in the strings
        try:
            metadatastr = zf.read(mdata_fn)
            datastr     = zf.read(data_fn)
        except Exception as E:
            errormsg = f'Could not load metadata file "{mdata_fn}" and/or data file "{data_fn}": are you sure this is a Sciris-versioned data zipfile?'
            raise FileNotFoundError(errormsg) from E
        
        # Convert into Python objects
        metadata = scf.loadjson(string=metadatastr)
        data     = scf.loadstr(datastr, **kwargs)

    if loadmetadata:
        output = dict(metadata=metadata, data=data)
        return output
    else:
        return data