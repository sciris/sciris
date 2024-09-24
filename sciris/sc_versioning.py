"""
Functions for checking and saving versioning information, such as Python package
versions, git versions, etc.

Highlights:
    - :func:`sc.freeze() <freeze>`: programmatically store "pip freeze" output
    - :func:`sc.require() <require>`: require a specific version of a package
    - :func:`sc.gitinfo() <gitinfo>`: gets the git information (if available) of a given file
    - :func:`sc.compareversions() <compareversions>`: easy way to compare version numbers
    - :func:`sc.metadata() <metadata>`: collects relevant metadata into a dictionary
    - :func:`sc.savearchive() <savearchive>`: saves data as a zip file including versioning metadata
"""

import os
import re
import time
import zlib
import types
import warnings
import importlib.metadata as imd
import packaging.version as pkgv
import packaging.specifiers as pkgs
import packaging.requirements as pkgr
from zipfile import ZipFile
import sciris as sc

__all__ = ['freeze', 'require', 'gitinfo', 'compareversions', 'getcaller', 
           'metadata', 'loadmetadata', 'savearchive', 'loadarchive']


# Define shared variables
_metadataflag      = 'sciris_metadata' # Key name used to identify metadata saved in a figure
_metadata_filename = 'sciris_metadata.json' # Default filenames for metadata and data
_obj_filename      = 'sciris_data.obj'


def freeze(lower=False):
    """
    Alias for pip freeze.

    Args:
        lower (bool): convert all keys to lowercase

    **Example**::

        assert 'numpy' in sc.freeze() # One way to check for versions

    | *New in version 1.2.2.*
    | *New in version 3.1.3:* use ``importlib`` instead of ``pkg_resources``
    """
    raw = {dist.metadata['Name']:dist.version for dist in imd.distributions()}
    keys = sorted(raw.keys())
    if lower: # pragma: no cover
        labels = {k:k.lower() for k in keys}
    else:
        labels = {k:k for k in keys}
    data = {labels[k]:raw[k] for k in keys} # Sort alphabetically
    return data


def pkg_require(req):
    """ Replacement for pkg_resources.require(); used by sc.require(), not for external use """
    r = pkgr.Requirement(req)
    version = imd.version(r.name)
    allowed = pkgs.SpecifierSet(str(r.specifier))
    if version not in allowed:
        string = f'{req} (available: {version})'
        raise imd.PackageNotFoundError(string)
    return


def require(reqs=None, *args, message=None, exact=False, detailed=False, die=True, warn=True, verbose=True, **kwargs):
    """
    Check whether environment requirements are met. Alias to pkg_resources.require().

    Args:
        reqs (list/dict): a list of strings, or a dict of package names and versions
        args (list): additional requirements
        message (str): optionally provide a custom error message if requirements are not met; "<MISSING>" will be replaced with list of missing requirements
        kwargs (dict): additional requirements
        exact (bool): use '==' instead of '>=' as the default comparison operator if not specified
        detailed (bool): return a dict of which requirements are/aren't met
        die (bool): whether to raise an exception if requirements aren't met
        warn (bool): if not die, raise a warning if requirements aren't met
        verbose (bool): print out the exception if it's not being raised or warned

    **Examples**::

        sc.require('numpy')
        sc.require(numpy='')
        sc.require(reqs={'numpy':'1.19.1', 'matplotlib':'3.2.2'})
        sc.require('numpy>=1.19.1', 'matplotlib==3.2.2', die=False, message='Requirements <MISSING> not met, but continuing anyway')
        sc.require(numpy='1.19.1', matplotlib='==4.2.2', die=False, detailed=True)

    | *New in version 1.2.2.*
    | *New in version 3.0.0:* "warn" argument
    | *New in version 3.1.3:* "message" argument
    | *New in version 3.1.6:* replace pkg_resources dependency with packaging
    """

    # Handle inputs
    reqlist = list(args)
    reqdict = kwargs
    if isinstance(reqs, dict):
        reqdict.update(reqs)
    else:
        reqlist = sc.mergelists(reqs, reqlist)

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
            pkg_require(entry) # Drop-in replacement for pkg_resources.require()
            data[entry] = True
        except Exception as E:
            data[entry] = False
            errs[entry] = E

    # Figure out output
    met = all([e==True for e in data.values()])

    # Handle exceptions
    if not met:
        errkeys = list(errs.keys())
        count = 0
        errorstrings = ''
        for key,valid in data.items():
            if not valid:
                count += 1
                errorstrings += f'\n• {key} (error: {errs[key]})'
        missing = sc.strjoin(errkeys, sep=' ')
        if message is not None:
            errormsg = message.replace('<MISSING>', missing)
        else:
            errormsg = '\nThe following requirement(s) were not met:'
            errormsg += errorstrings
            errormsg += f'\nTry "pip install {missing}".'
        if die:
            err = errs[errkeys[-1]]
            raise ModuleNotFoundError(errormsg) from err
        elif warn:
            warnings.warn(errormsg, category=UserWarning, stacklevel=2)
        elif verbose:
            print(errormsg)
        else:
            pass

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
    """
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

    *New in version 1.2.1:* relational operators
    """
    # Handle inputs
    if isinstance(version1, types.ModuleType):
        try:
            version1 = version1.__version__
        except Exception as E: # pragma: no cover
            errormsg = f'{version1} is a module, but does not have a __version__ attribute'
            raise AttributeError(errormsg) from E
    v1 = str(version1)
    v2 = str(version2)

    # Process version2 -- note that order matters, and two-char prefixes have to be handled first
    valid = None
    if   v2.startswith('<='): valid = [0,-1]
    elif v2.startswith('>='): valid = [0,1]
    elif v2.startswith('=='): valid = [0]
    elif v2.startswith('~='): valid = [-1,1]
    elif v2.startswith('!='): valid = [-1,1]
    elif v2.startswith('<'):  valid = [-1]
    elif v2.startswith('>'):  valid = [1]
    elif v2.startswith('='):  valid = [0]
    elif v2.startswith('!'):  valid = [-1,1]
    elif v2.startswith('~'): # pragma: no cover
        errormsg = 'Loose version pinning is not supported; for not, use "~="'
        raise ValueError(errormsg)
    
    v2 = v2.lstrip('<>=!~')

    # Do comparison
    if pkgv.parse(v1) > pkgv.parse(v2):
        comparison =  1
    elif pkgv.parse(v1) < pkgv.parse(v2):
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
    """
    Try to get information on the calling function, but fail gracefully. See also
    :func:`sc.thisfile() <sciris.sc_fileio.thisfile>`.

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

    | *New in version 1.0.0.*
    | *New in version 1.3.3:* do not include line by default
    | *New in version 3.0.0:* "relframe" argument; "die" argument
    """
    try:
        import inspect
        frame = frame + relframe
        result = inspect.getouterframes(inspect.currentframe(), 2)
        fname = str(result[frame][1])
        lineno = result[frame][2]
        if tostring:
            output = f'{fname}'
            if includelineno: # pragma: no cover
                output += f', line {lineno}'
        else:
            output = {'filename':fname, 'lineno':lineno}
            if includeline: # pragma: no cover
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


def metadata(outfile=None, version=None, comments=None, require=None, pipfreeze=True, user=True, caller=True, 
             git=True, asdict=False, tostring=False, relframe=0, **kwargs):
    """
    Collect common metadata: useful for exactly recreating (or remembering) the environment
    at a moment in time.
    
    Args:
        outfile (str): if not None, then save as JSON to this filename
        version (str): if supplied, the user-supplied version of the data being stored
        comments (str/dict): additional comments on the data to store
        require (str/dict): if provided, an additional manual set of requirements
        pipfreeze (bool): store the current Python environment, equivalent to "pip freeze"
        user (bool): store the username
        caller (bool): store info on the calling file
        git (bool): store git information on the calling file (branch, hash, etc.)
        asdict (bool): construct as a dict instead of an objdict
        tostring (bool): return a string rather than a dict
        relframe (int): how far to descend into the calling stack (if used directly, use 0; if called by another function, use 1; etc)
        kwargs (dict): any additional data to store (can be anything JSON-compatible)
        
    Returns:
        A dictionary with information on the date, plateform, executable, versions
        of key libraries (Sciris, Numpy, pandas, and Matplotlib), and the Python environment
        
    **Examples**::
        
        metadata = sc.metadata()
        sc.compareversions(metadata.versions.pandas, '1.5.0')
        
        sc.metadata('my-metadata.json') # Save to disk
    
    *New in version 3.0.0.*
    """
    
    # Additional imports
    import sys
    import platform
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    from .sc_version import __version__
    
    # Handle type
    dict_fn = dict if asdict else sc.objdict
    
    # Get calling info
    calling_info = dict_fn(getcaller(relframe=relframe+1, tostring=False))
    
    # Store metadata
    md = dict_fn(
        version   = version,
        timestamp = sc.getdate(),
        user      = sc.getuser() if user else None,
        system = dict_fn(
            platform   = platform.platform(),
            executable = sys.executable,
            version    = sys.version,
        ),
        versions = dict_fn(
            python     = platform.python_version(),
            sciris     = __version__,
            numpy      = np.__version__,
            pandas     = pd.__version__,
            matplotlib = mpl.__version__,
        ),
        calling_info = calling_info if caller else None,
        git_info     = dict_fn(gitinfo(calling_info['filename'], die=False, verbose=False)) if git else None,
        pipfreeze    = freeze() if pipfreeze else None,
        require      = require,
        comments     = comments,
    )
    
    # Store any additional data
    md.update(kwargs)
    
    if outfile is not None:
        outfile = sc.makepath(outfile, makedirs=True)
        sc.savejson(outfile, md)
    
    if tostring:
        md = sc.jsonify(md, tostring=True, indent=2)
    
    return md


def _md_to_objdict(md):
    """ Convert a metadata dictionary into an objdict -- descend two levels but no deeper """
    md = sc.objdict(md)
    for k,v in md.items():
        if isinstance(v, dict):
            md[k] = sc.objdict(v)
    return md
        

def loadmetadata(filename, load_all=False, die=True):
    """
    Read metadata from a saved image; currently only PNG and SVG are supported.

    Only for use with images saved with :func:`sc.savefig() <sciris.sc_plotting.savefig>`. Metadata retrieval for PDF
    is not currently supported. To load metadata saved with :func:`sc.metadata() <metadata>`, 
    you can also use :func:`sc.loadjson() <sciris.sc_fileio.loadjson>` instead. To load metadata saved with :func:`sc.savearchive() <savearchive>`,
    use :func:`sc.loadarchive() <loadarchive>` instead.

    Args:
        filename (str): the name of the file to load the data from
        load_all (bool): whether to load all metadata available in an image (else, just load what was saved by Sciris)
        die (bool): whether to raise an exception if the metadata can't be found

    **Example**::

        plt.plot([1,2,3], [4,2,6])
        sc.savefig('example.png')
        sc.loadmetadata('example.png')
    """

    # Initialize
    md = {}
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
            if load_all: # pragma: no cover
                md = im.info
            else:
                jsonstr = im.info[_metadataflag]
                md = sc.loadjson(string=jsonstr)

        # JPG -- from https://www.thepythoncode.com/article/extracting-image-metadata-in-python
        elif is_jpg: # pragma: no cover
            from PIL.ExifTags import TAGS # Must be imported directly
            exifdata = im.getexif()
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode()
                md[tag] = data

        # Can't find metadata
        else: # pragma: no cover
            errormsg = f'Could not find "{_metadataflag}": metadata can only be extracted from figures saved with sc.savefig().\nAvailable keys are: {sc.strjoin(keys)}'
            if die:
                raise ValueError(errormsg)
            else:
                print(errormsg)
                md = im.info


    # Handle SVG
    elif lcfn.endswith('svg'): # pragma: no cover

        # Load SVG as text and parse it
        svg = sc.loadtext(filename).splitlines()
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
            md = sc.loadjson(string=jsonstr)

        # Can't find metadata
        else:
            errormsg = f'Could not find the string "{_metadataflag}" anywhere in "{filename}": metadata can only be extracted from figures saved with sc.savefig()'
            if die:
                raise ValueError(errormsg)
            else:
                print(errormsg)
    
    # Load metadata saved with sc.metadata(), and convert it from dict to objdict
    elif lcfn.endswith('json'):
        md = _md_to_objdict(sc.loadjson(filename))
    
    # Load metadata saved with sc.savearchive()
    elif lcfn.endswith('zip'):
        md = loadarchive(filename, loadobj=False, loadmetadata=True)

    # Other formats not supported
    else: # pragma: no cover
        errormsg = f'Filename "{filename}" has unsupported type: must be PNG, JPG, or SVG (PDF is not supported)'
        raise ValueError(errormsg)

    return md



def savearchive(filename, obj, files=None, folder=None, comments=None, require=None, 
                     user=True, caller=True, git=True, pipfreeze=True, method='dill', 
                     allow_nonzip=False, dumpargs=None, **kwargs):
    """
    Save any object as a pickled zip file, including metadata as a separate JSON file.
    
    Pickles are usually not good for long-term data storage, since they rely on
    importing the libraries that were used to create the pickled object. This function
    partly addresses that by storing metadata along with the saved pickle. While
    there may still be issues opening the pickle, the metadata (which is stored separately)
    should give enough information to figure out how to reconstruct the original 
    environment (allowing the pickle to be loaded, and then re-saved in a more persistent 
    format if desired).
    
    Note: Since this function relies on pickle, it can potentially execute arbitrary
    code, so you should only use it with sources you trust. For more information, see:
    https://docs.python.org/3/library/pickle.html
    
    Args:
        filename (str/path): the file to save to (must end in .zip)
        obj (any): the object to save
        files (str/list): any additional files or folders to save
        comments (str/dict): other comments/information to store in the metadata (must be JSON-compatible)
        require (str/dict): if provided, an additional manual set of requirements
        caller (bool): store information on the current user in the metadata (see :func:`sc.metadata() <metadata>`)
        caller (bool): store information on the calling file in the metadata (see :func:`sc.metadata() <metadata>`)
        git (bool): store the git version in the metadata (see :func:`sc.metadata() <metadata>`)
        pipfreeze (bool): store the output of "pip freeze" in the metadata (see :func:`sc.metadata() <metadata>`)
        method (str): the method to use saving the data; default "dill" for more robustness, but "pickle" is faster
        allow_nonzip (bool): whether to permit extensions other than .zip (note, may cause problems!)
        dumpargs (dict): passed to :func:`sc.dumpstr() <sciris.sc_fileio.dumpstr>`
        kwargs (dict): passed to :func:`sc.savezip() <sciris.sc_fileio.savezip>`
    
    **Example**::
        
        obj = MyClass() # Create an arbitrary object
        sc.savearchive('my-class.zip', obj)
        
        # Much later...
        obj = sc.loadarchive('my-class.zip')
    
    *New in version 3.0.0.*
    """
    filename = sc.makepath(filename=filename, folder=folder, makedirs=True)
    
    # Check filename
    if not allow_nonzip: # pragma: no cover
        if filename.suffix != '.zip':
            errormsg = f'Your filename ends with "{filename.suffix}" rather than ".zip". If you are sure you want to do this, set allow_nonzip=True.'
            raise ValueError(errormsg)

    # Create the metadata, including storing the custom "method" attribute
    md = metadata(caller=caller, git=git, pipfreeze=pipfreeze, comments=comments, 
                  require=require, frame=3, method=method)
    
    # Convert both to strings
    dumpargs    = sc.mergedicts({'method':method}, dumpargs)
    metadatastr = sc.jsonify(md, tostring=True, indent=2)
    datastr     = sc.dumpstr(obj, **dumpargs)
    
    # Construct output
    datadict = {_metadata_filename:metadatastr, _obj_filename:datastr}
    
    return sc.savezip(filename=filename, files=files, data=datadict, tobytes=False, **kwargs)


def loadarchive(filename, folder=None, loadobj=True, loadmetadata=False, 
                     remapping=None, die=True, **kwargs):
    """
    Load a zip file saved with :func:`sc.savearchive() <savearchive>`.
    
    **Note**: Since this function relies on pickle, it can potentially execute arbitrary
    code, so you should only use it with sources you trust. For more information, see:
    https://docs.python.org/3/library/pickle.html
    
    Args:
        filename (str/path): the file load to (usually ends in .zip)
        folder (str): optional additional folder to load from
        loadobj (bool): whether to load the saved object
        loadmetadata (bool): whether to load the metadata as well
        remapping (dict): any known module remappings between the saved pickle version and the current libraries
        die (bool): whether to fail if an exception is raised (else, just return the metadata)
        kwargs (dict): passed to :func:`sc.load() <sciris.sc_fileio.load>`
    
    Returns:
        If loadobj=True and loadmetadata=False, return the object;
        If loadobj=False and loadmetadata=True, return the metadata
        If loadobj=True and loadmetadata=True, return a dictionary of both
    
    **Example**::
        
        obj = MyClass() # Create an arbitrary object
        sc.savearchive('my-class.zip', obj)
        
        # Much later...
        data = sc.loadarchive('my-class.zip', loadmetadata=True)
        metadata, obj = data['metadata'], data['obj']
    
    Note: This function expects the zip file to contain two files in it, one called 
    "metadata.json" and one called "sciris_pickle.obj". If you need to change these,
    you can manually modify ``sc.sc_versioning._metadata_filename`` and 
    ``sc.sc_versioning._obj_filename``, respectively. However, you almost certainly
    should not do so!
    
    *New in version 3.0.0.*
    """
    filename = sc.makefilepath(filename=filename, folder=folder, makedirs=False)
    
    # Open the zip file
    try:
        zf = ZipFile(filename, 'r')  # Create the zip file
    except Exception as E: # pragma: no cover
        exc = type(E)
        errormsg = 'Could not open zip file: ensure the filename is correct and of the right type; see error above for details'
        raise exc(errormsg) from E
   
    # Read the zip file
    obj = None
    with zf:
    
        # Read in the strings
        try:
            metadatastr = zf.read(_metadata_filename)
        except Exception as E: # pragma: no cover
            exc = type(E)
            errormsg = f'Could not load metadata file "{_metadata_filename}": are you sure this is a Sciris-versioned data zipfile?'
            raise exc(errormsg) from E
        
        # Load the metadata first
        try:
            md = _md_to_objdict(sc.loadjson(string=metadatastr))
        except Exception as E: # pragma: no cover
            errormsg = 'Could not parse the metadata as a JSON file; see error above for details'
            raise ValueError(errormsg) from E
            
        # Now try loading the actual data
        if loadobj:
            try:
                datastr = zf.read(_obj_filename)
            except Exception as E: # pragma: no cover
                exc = type(E)
                errormsg = f'Could not load object file "{_obj_filename}": are you sure this is a Sciris-versioned data zipfile? To debug using metadata, set die=False'
                if die:
                    raise exc(errormsg) from E
                else:
                    warnmsg = 'Exception encountered opening the object, returning metadata only'
                    warnings.warn(warnmsg, category=UserWarning, stacklevel=2)
                    return md
                
            # Convert into Python objects -- the most likely step where this can go wrong
            try:
                method = md.get('method', None)
                reqs   = md.get('require', None)
                obj    = sc.loadstr(datastr, method=method, remapping=remapping, **kwargs) # Load with original remappings
                if reqs:
                    require(reqs=reqs, die=False, warn=True) # Don't die, but print warnings
            except: # pragma: no cover
                try:
                    remapping = sc.mergedicts(remapping, sc.known_fixes)
                    obj = sc.loadstr(datastr, remapping=remapping, verbose=True, **kwargs) # Load with all remappings
                except Exception as E:
                    exc = type(E)
                    errormsg = 'Could not unpickle the object: to debug using metadata, set die=False'
                    if die:
                        raise exc(errormsg) from E
                    else:
                        warnmsg = 'Exception encountered unpickling the object, returning metadata only'
                        warnings.warn(warnmsg, category=UserWarning, stacklevel=2)
                        return md

    # Handle output
    if loadobj and not loadmetadata:
        return obj
    elif loadmetadata and not loadobj:
        return md
    elif loadmetadata and loadobj:
        return dict(metadata=md, obj=obj)
    else: # pragma: no cover
        errormsg = 'No return value specified; you must load the object, metadata, or both'
        raise ValueError(errormsg)