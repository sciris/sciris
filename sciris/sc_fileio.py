"""
fileio.py -- code for file management in Sciris

Last update: 2019feb18
"""

##############################################################################
### Imports
##############################################################################

# Basic imports
import io
import os
import re
import json
import uuid
import pickle
import types
import datetime
import importlib
import traceback
import numpy as np
from glob import glob
from gzip import GzipFile
from zipfile import ZipFile
from contextlib import closing
from collections import OrderedDict
from pathlib import Path
from . import sc_utils as ut
from .sc_odict import odict
from .sc_dataframe import dataframe

# Handle types and Python 2/3 compatibility
import six
if six.PY3: # Python 3
    from io import BytesIO as IO
    import pickle as pkl
    import copyreg as cpreg
else: # Python 2
    from cStringIO import StringIO as IO
    import cPickle as pkl
    import copy_reg as cpreg




##############################################################################
### Pickling functions
##############################################################################

__all__ = ['loadobj', 'loadstr', 'saveobj', 'dumpstr']


def loadobj(filename=None, folder=None, verbose=False, die=None):
    '''
    Load a file that has been saved as a gzipped pickle file. Accepts either
    a filename (standard usage) or a file object as the first argument.

    Usage:
        obj = loadobj('myfile.obj')
    '''

    # Handle loading of either filename or file object
    if isinstance(filename, Path):
        filename = str(filename)
    if ut.isstring(filename):
        argtype = 'filename'
        filename = makefilepath(filename=filename, folder=folder, makedirs=False) # If it is a file, validate the folder (but don't create one if it's missing)
    elif isinstance(filename, io.BytesIO):
        argtype = 'fileobj'
    else:
        errormsg = 'First argument to loadobj() must be a string or file object, not %s' % type(filename)
        raise Exception(errormsg)
    kwargs = {'mode': 'rb', argtype: filename}
    with GzipFile(**kwargs) as fileobj:
        filestr = fileobj.read() # Convert it to a string
        obj = unpickler(filestr, filename=filename, verbose=verbose, die=die) # Actually load it
    if verbose: print('Object loaded from "%s"' % filename)
    return obj


def loadstr(string=None, verbose=False, die=None):
    if string is None:
        errormsg = 'loadstr() error: input received was None, not a string'
        raise Exception(errormsg)
    with closing(IO(string)) as output: # Open a "fake file" with the Gzip string pickle in it.
        with GzipFile(fileobj=output, mode='rb') as fileobj: # Set a Gzip reader to pull from the "file."
            picklestring = fileobj.read() # Read the string pickle from the "file" (applying Gzip decompression).
    obj = unpickler(picklestring, filestring=string, verbose=verbose, die=die) # Return the object gotten from the string pickle.
    return obj


def saveobj(filename=None, obj=None, compresslevel=5, verbose=0, folder=None, method='pickle'):
    '''
    Save an object to file -- use compression 5 by default, since more is much slower but not much smaller.
    Once saved, can be loaded with loadobj() (q.v.).

    Usage:
        myobj = ['this', 'is', 'a', 'weird', {'object':44}]
        saveobj('myfile.obj', myobj)
    '''
    if isinstance(filename, Path): # If it's a path object, convert to string
        filename = str(filename)
    if filename is None: # If it doesn't exist, just create a byte stream
        bytesobj = io.BytesIO()
    else: # Normal use case: make a file path
        bytesobj = None
        filename = makefilepath(filename=filename, folder=folder, default='default.obj', sanitize=True)

    with GzipFile(filename=filename, fileobj=bytesobj, mode='wb', compresslevel=compresslevel) as fileobj:
        if method == 'dill': # If dill is requested, use that
            if verbose>=2: print('Saving as dill...')
            savedill(fileobj, obj)
        else: # Otherwise, try pickle
            try:
                if verbose>=2: print('Saving as pickle...')
                savepickle(fileobj, obj) # Use pickle
            except Exception as E:
                if verbose>=2: print('Exception when saving as pickle (%s), saving as dill...' % repr(E))
                savedill(fileobj, obj) # ...but use Dill if that fails

    if verbose and filename: print('Object saved to "%s"' % filename)

    if filename:
        return filename
    else:
        bytesobj.seek(0)
        return bytesobj


def dumpstr(obj=None):
    with closing(IO()) as output: # Open a "fake file."
        with GzipFile(fileobj=output, mode='wb') as fileobj:  # Open a Gzip-compressing way to write to this "file."
            try:    savepickle(fileobj, obj) # Use pickle
            except: savedill(fileobj, obj) # ...but use Dill if that fails
        output.seek(0) # Move the mark to the beginning of the "file."
        result = output.read() # Read all of the content into result.
    return result




##############################################################################
### Other file functions
##############################################################################

__all__ += ['loadtext', 'savetext', 'savezip', 'getfilelist', 'sanitizefilename', 'makefilepath']



def loadtext(filename=None, folder=None, splitlines=False):
    ''' Convenience function for reading a text file '''
    filename = makefilepath(filename=filename, folder=folder)
    with open(filename) as f: output = f.read()
    if splitlines: output = output.splitlines()
    return output


def savetext(filename=None, string=None):
    ''' Convenience function for saving a text file -- accepts a string or list of strings '''
    if isinstance(string, list): string = '\n'.join(string) # Convert from list to string)
    if not ut.isstring(string):  string = str(string)
    filename = makefilepath(filename=filename)
    with open(filename, 'w') as f: f.write(string)
    return None


def savezip(filename=None, filelist=None, folder=None, basename=True, verbose=True):
    ''' Create a zip file from the supplied lst of files '''
    fullpath = makefilepath(filename=filename, folder=folder, sanitize=True)
    filelist = ut.promotetolist(filelist)
    with ZipFile(fullpath, 'w') as zf: # Create the zip file
        for thisfile in filelist:
            thispath = makefilepath(filename=thisfile, abspath=False)
            if basename: thisname = os.path.basename(thispath)
            else:        thisname = thispath
            zf.write(thispath, thisname)
    if verbose: print('Zip file saved to "%s"' % fullpath)
    return fullpath


def getfilelist(folder=None, ext=None, pattern=None):
    ''' A short-hand since glob is annoying '''
    if folder is None: folder = os.getcwd()
    folder = os.path.expanduser(folder)
    if pattern is None:
        if ext is None: ext = '*'
        pattern = '*.'+ext
    filelist = sorted(glob(os.path.join(folder, pattern)))
    return filelist


def sanitizefilename(rawfilename):
    '''
    Takes a potentially Linux- and Windows-unfriendly candidate file name, and
    returns a "sanitized" version that is more usable.
    '''
    filtername = re.sub('[\!\?\"\'<>]', '', rawfilename) # Erase certain characters we don't want at all: !, ?, ", ', <, >
    filtername = re.sub('[:/\\\*\|,]', '_', filtername) # Change certain characters that might be being used as separators from what they were to underscores: space, :, /, \, *, |, comma
    return filtername # Return the sanitized file name.


def makefilepath(filename=None, folder=None, ext=None, default=None, split=False, abspath=True, makedirs=True, verbose=False, sanitize=False):
    '''
    Utility for taking a filename and folder -- or not -- and generating a valid path from them.

    Inputs:
        filename = the filename, or full file path, to save to -- in which case this utility does nothing
        folder = the name of the folder to be prepended to the filename
        ext = the extension to ensure the file has
        default = a name or list of names to use if filename is None
        split = whether to return the path and filename separately
        makedirs = whether or not to make the folders to save into if they don't exist
        verbose = how much detail to print

    Example:
        makefilepath(filename=None, folder='./congee', ext='prj', default=[project.filename, project.name], split=True, abspath=True, makedirs=True)

    Assuming project.filename is None and project.name is "soggyrice" and ./congee doesn't exist:
        * Makes folder ./congee
        * Returns e.g. ('/home/myname/congee', 'soggyrice.prj')

    Actual code example from project.py:
        fullpath = makefilepath(filename=filename, folder=folder, default=[self.filename, self.name], ext='prj')

    Version: 2018sep22
    '''

    # Initialize
    filefolder = '' # The folder the file will be located in
    filebasename = '' # The filename

    # Process filename
    if filename is None:
        defaultnames = ut.promotetolist(default) # Loop over list of default names
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
        print('From filename="%s", default="%s", extension="%s", made basename "%s"' % (filename, default, ext, filebasename))

    # Sanitize base filename
    if sanitize: filebasename = sanitizefilename(filebasename)

    # Process folder
    if folder is not None: # Replace with specified folder, if defined
        filefolder = folder
    if abspath: # Convert to absolute path
        filefolder = os.path.abspath(os.path.expanduser(filefolder))
    if makedirs: # Make sure folder exists
        try: os.makedirs(filefolder)
        except: pass
    if verbose:
        print('From filename="%s", folder="%s", abspath="%s", makedirs="%s", made folder name "%s"' % (filename, folder, abspath, makedirs, filefolder))

    fullfile = os.path.join(filefolder, filebasename) # And the full thing

    if split: return filefolder, filebasename
    else:     return fullfile # Or can do os.path.split() on output



##############################################################################
### JSON functions
##############################################################################

__all__ += ['sanitizejson', 'jsonify', 'loadjson', 'savejson']


def sanitizejson(obj, verbose=True, die=False, tostring=False, **kwargs):
    """
    This is the main conversion function for Python data-structures into
    JSON-compatible data structures.
    Use this as much as possible to guard against data corruption!
    Args:
        obj: almost any kind of data structure that is a combination
            of list, numpy.ndarray, odicts, etc.
    Returns:
        A converted dict/list/value that should be JSON compatible
    """
    if obj is None: # Return None unchanged
        output = None

    elif isinstance(obj, (bool, np.bool_)): # It's true/false
        output = bool(obj)

    elif ut.isnumber(obj): # It's a number
        if np.isnan(obj): # It's nan, so return None
            output = None
        else:
            if isinstance(obj, (int, np.int64)): output = int(obj) # It's an integer
            else:                                output = float(obj)# It's something else, treat it as a float

    elif isinstance(obj, np.ndarray): # It's an array, iterate recursively
        if obj.shape: output = [sanitizejson(p) for p in list(obj)] # Handle most cases, incluing e.g. array([5])
        else:         output = [sanitizejson(p) for p in list(np.array([obj]))] # Handle the special case of e.g. array(5)

    elif isinstance(obj, (list, set, tuple)): # It's another kind of interable, so iterate recurisevly
        output = [sanitizejson(p) for p in list(obj)]

    elif isinstance(obj, dict): # Treat all dictionaries as ordered dictionaries
        output = OrderedDict()
        for key,val in obj.items():
            output[str(key)] = sanitizejson(val)

    elif isinstance(obj, datetime.datetime):
        output = str(obj)

    elif isinstance(obj, uuid.UUID):
        output = str(obj)

    elif ut.isstring(obj): # It's a string of some kind
        try:    string = str(obj) # Try to convert it to ascii
        except: string = obj # Give up and use original
        output = string

    else: # None of the above
        try:
            output = json.loads(json.dumps(obj)) # Try passing it through jsonification
        except Exception as E:
            errormsg = 'Could not sanitize "%s" %s (%s), converting to string instead' % (obj, type(obj), str(E))
            if die:       raise Exception(errormsg)
            elif verbose: print(errormsg)
            output = str(obj)

    if tostring: return json.dumps(output, **kwargs)
    else:        return output



def jsonify(*args, **kwargs):
    ''' Alias to sanitizejson() '''
    return sanitizejson(*args, **kwargs)


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

    '''
    if fromfile:
        filename = makefilepath(filename=filename, folder=folder)
        try:
            with open(filename) as f:
                output = json.load(f, **kwargs)
        except FileNotFoundError as E:
            print('Warning: file not found. Use fromfile=False if not loading from a file')
            raise E
    else:
        if string is None and filename is not None:
            string = filename # Swap arguments
        output = json.loads(string, **kwargs)

    return output



def savejson(filename=None, obj=None, folder=None, **kwargs):
    ''' Convenience function for saving a JSON '''
    filename = makefilepath(filename=filename, folder=folder)
    with open(filename, 'w') as f:
        json.dump(sanitizejson(obj), f, **kwargs)
    return None



##############################################################################
### Spreadsheet functions
##############################################################################

__all__ += ['Blobject', 'Spreadsheet', 'loadspreadsheet', 'savespreadsheet']


class Blobject(object):
    ''' A wrapper for a binary file '''

    def __init__(self, source=None, name=None, filename=None, blob=None):
        # "source" is a specification of where to get the data from
        # It can be anything supported by Blobject.load() which are
        # - A filename, which will get loaded
        # - A io.BytesIO which will get dumped into this instance
        # Alternatively, can specify `blob` which is a binary string that gets stored directly
        # in the `blob` attribute

        # Handle inputs
        if source   is None and filename is not None: source   = filename # Reset the source to be the filename, e.g. Spreadsheet(filename='foo.xlsx')
        if filename is None and ut.isstring(source):  filename = source   # Reset the filename to be the source, e.g. Spreadsheet('foo.xlsx')
        if name     is None and filename is not None: name     = os.path.basename(filename) # If not supplied, use the filename
        if blob is not None and source is not None: raise Exception('Can initialize from either source or blob, but not both')

        # Define quantities
        self.name      = name # Name of the object
        self.filename  = filename # Filename (used as default for load/save)
        self.created   = ut.now() # When the object was created
        self.modified  = ut.now() # When it was last modified
        self.blob  = blob # The binary data
        self.bytes = None # The filestream representation of the binary data
        if source is not None: self.load(source)
        return None

    def __repr__(self):
        return ut.prepr(self, skip=['blob','bytes'])

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
            elif ut.isstring(source):
                self.blob = read_file(source)
            else:
                errormsg = 'Input source must be type string (for a filename) or BytesIO, not %s' % type(source)
                raise Exception(errormsg)

        self.modified = ut.now()
        return None

    def save(self, filename=None):
        ''' This function writes the spreadsheet to a file on disk. '''
        if filename is None:
            if self.filename is not None:
                filename = self.filename
            elif self.name is not None:
                if self.name.endswith('.xlsx'):
                    filename = self.name
                else:
                    filename = self.name + '.xlsx'
            else:
                filename = 'spreadsheet.xlsx' # Come up with a terrible default name
        filepath = makefilepath(filename=filename)
        with open(filepath, mode='wb') as f:
            f.write(self.blob)
        self.filename = filename
        print('Spreadsheet saved to %s.' % filepath)
        return filepath

    def tofile(self, output=True):
        '''
        Return a file-like object with the contents of the file.
        This can then be used to open the workbook from memory without writing anything to disk e.g.
        - book = openpyexcel.load_workbook(self.tofile())
        - book = xlrd.open_workbook(file_contents=self.tofile().read())
        '''
        bytesblob = io.BytesIO(self.blob)
        if output:
            return bytesblob
        else:
            self.bytes = bytesblob
            return None

    def freshbytes(self):
        ''' Refresh the bytes object to accept new data '''
        self.bytes = io.BytesIO()
        return self.bytes



class Spreadsheet(Blobject):
    '''
    A class for reading and writing Excel files in binary format.No disk IO needs
    to happen to manipulate the spreadsheets with openpyexcel (or xlrd or pandas).

    Version: 2018sep03
    '''

    def xlrd(self, *args, **kwargs):
        ''' Return a book as opened by xlrd '''
        try:
            import xlrd # Optional import
        except ModuleNotFoundError as e:
            raise Exception('The "xlrd" Python package is not available; please install manually') from e
        book = xlrd.open_workbook(file_contents=self.tofile().read(), *args, **kwargs)
        return book

    def openpyexcel(self, *args, **kwargs):
        ''' Return a book as opened by openpyexcel '''
        try:
            import openpyexcel # Optional import
        except ModuleNotFoundError as e:
            raise Exception('The "openpyexcel" Python package is not available; please install manually') from e
        self.tofile(output=False)
        book = openpyexcel.load_workbook(self.bytes, *args, **kwargs) # This stream can be passed straight to openpyexcel
        return book

    def pandas(self, *args, **kwargs):
        ''' Return a book as opened by pandas '''
        try:
            import pandas # Optional import
        except ModuleNotFoundError as e:
            raise Exception('The "pandas" Python package is not available; please install manually') from e
        self.tofile(output=False)
        book = pandas.ExcelFile(self.bytes, *args, **kwargs)
        return book

    def update(self, book):
        ''' Updated the stored spreadsheet with book instead '''
        self.tofile(output=False)
        book.save(self.freshbytes())
        self.load()
        return None

    @staticmethod
    def _getsheet(book, sheetname=None, sheetnum=None):
        if   sheetname is not None: sheet = book[sheetname]
        elif sheetnum  is not None: sheet = book[book.sheetnames[sheetnum]]
        else:                       sheet = book.active
        return sheet

    def readcells(self, wbargs=None, *args, **kwargs):
        ''' Alias to loadspreadsheet() '''
        if 'method' in kwargs:
            method = kwargs['method']
            kwargs.pop('method')
        else:
            method = None
        if method is None: method = 'xlrd'
        f = self.tofile()
        kwargs['fileobj'] = f

        # Read in sheetoutput (sciris dataframe object for xlrd, 2D numpy array for openpyexcel).
        if method == 'xlrd':
            sheetoutput = loadspreadsheet(*args, **kwargs)  # returns sciris dataframe object
        elif method == 'openpyexcel':
            if wbargs is None: wbargs = {}
            book = self.openpyexcel(**wbargs)
            ws = self._getsheet(book=book, sheetname=kwargs.get('sheetname'), sheetnum=kwargs.get('sheetname'))
            rawdata = tuple(ws.rows)
            sheetoutput = np.empty(np.shape(rawdata), dtype=object)
            for r,rowdata in enumerate(rawdata):
                for c,val in enumerate(rowdata):
                    sheetoutput[r][c] = rawdata[r][c].value
        else:
            errormsg = 'Reading method not found; must be one of xlrd, openpyexcel, or pandas, not %s' % method
            raise Exception(errormsg)

        # Return the appropriate output.
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
        '''
        # Load workbook
        if wbargs is None: wbargs = {}
        wb = self.openpyexcel(**wbargs)
        if verbose: print('Workbook loaded: %s' % wb)

        # Get right worksheet
        ws = self._getsheet(book=wb, sheetname=sheetname, sheetnum=sheetnum)
        if verbose: print('Worksheet loaded: %s' % ws)

        # Determine the cells
        if cells is not None: # A list of cells is supplied
            cells = ut.promotetolist(cells)
            vals  = ut.promotetolist(vals)
            if len(cells) != len(vals):
                errormsg = 'If using cells, cells and vals must have the same length (%s vs. %s)' % (len(cells), len(vals))
                raise Exception(errormsg)
            for cell,val in zip(cells,vals):
                try:
                    if ut.isstring(cell): # Handles e.g. cell='A1'
                        cellobj = ws[cell]
                    elif ut.checktype(cell, 'arraylike','number') and len(cell)==2: # Handles e.g. cell=(0,0)
                        cellobj = ws.cell(row=cell[0], column=cell[1])
                    else:
                        errormsg = 'Cell must be formatted as a label or row-column pair, e.g. "A1" or (3,5); not "%s"' % cell
                        raise Exception(errormsg)
                    if verbose: print('  Cell %s = %s' % (cell,val))
                    if isinstance(val,tuple):
                        cellobj.value = val[0]
                        cellobj.cached_value = val[1]
                    else:
                        cellobj.value = val
                except Exception as E:
                    errormsg = 'Could not write "%s" to cell "%s": %s' % (val, cell, repr(E))
                    raise Exception(errormsg)
        else:# Cells aren't supplied, assume a matrix
            if startrow is None: startrow = 1 # Excel uses 1-based indexing
            if startcol is None: startcol = 1
            valarray = np.atleast_2d(np.array(vals, dtype=object))
            for i,rowvals in enumerate(valarray):
                row = startrow + i
                for j,val in enumerate(rowvals):
                    col = startcol + j
                    try:
                        key = 'row:%s col:%s' % (row,col)
                        ws.cell(row=row, column=col, value=val)
                        if verbose: print('  Cell %s = %s' % (key, val))
                    except Exception as E:
                        errormsg = 'Could not write "%s" to %s: %s' % (val, key, repr(E))
                        raise Exception(errormsg)

        # Save
        wb.save(self.freshbytes())
        self.load()

        return None


    pass


def loadspreadsheet(filename=None, folder=None, fileobj=None, sheetname=None, sheetnum=None, asdataframe=None, header=True, cells=None):
    '''
    Load a spreadsheet as a list of lists or as a dataframe. Read from either a filename or a file object.
    '''
    try:
        import xlrd # Optional import
    except ModuleNotFoundError as e:
        raise Exception('The "xlrd" Python package is not available; please install manually') from e

    # Handle inputs
    if asdataframe is None: asdataframe = True
    if isinstance(filename, io.BytesIO): fileobj = filename # It's actually a fileobj
    if fileobj is None:
        fullpath = makefilepath(filename=filename, folder=folder)
        book = xlrd.open_workbook(fullpath)
    else:
        book = xlrd.open_workbook(file_contents=fileobj.read())
    if sheetname is not None:
        sheet = book.sheet_by_name(sheetname)
    else:
        if sheetnum is None: sheetnum = 0
        sheet = book.sheet_by_index(sheetnum)

    # Load the raw data
    rawdata = []
    for rownum in range(sheet.nrows-header):
        rawdata.append(odict())
        for colnum in range(sheet.ncols):
            if header: attr = sheet.cell_value(0,colnum)
            else:      attr = 'Column %s' % colnum
            attr = ut.uniquename(attr, namelist=rawdata[rownum].keys(), style='(%d)')
            val = sheet.cell_value(rownum+header,colnum)
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
        dfdata = dataframe(cols=cols, data=reformatted)
        return dfdata

    # Or leave in the original format
    else:
        return rawdata



def savespreadsheet(filename=None, data=None, folder=None, sheetnames=None, close=True, formats=None, formatdata=None, verbose=False):
    '''
    Little function to format an output results nicely for Excel. Examples:

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
    except ModuleNotFoundError as e:
            raise Exception('The "xlsxwriter" Python package is not available; please install manually') from e
    fullpath = makefilepath(filename=filename, folder=folder, default='default.xlsx')
    datadict   = odict()
    formatdict = odict()
    hasformats = (formats is not None) and (formatdata is not None)

    # Handle input arguments
    if isinstance(data, dict) and sheetnames is None:
        if verbose: print('Data is a dict, taking sheetnames from keys')
        sheetnames = data.keys()
        datadict   = odict(data) # Use directly, converting to odict first
        if hasformats: formatdict = odict(formatdata) #  NB, might be None but should be ok
    elif isinstance(data, dict) and sheetnames is not None:
        if verbose: print('Data is a dict, taking sheetnames from input')
        if len(sheetnames) != len(data):
            errormsg = 'If supplying data as a dict as well as sheetnames, length must matc (%s vs %s)' % (len(data), len(sheetnames))
            raise Exception(errormsg)
        datadict   = odict(data) # Use directly, but keep original sheet names
        if hasformats: formatdict = odict(formatdata)
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
                errormsg = 'Unable to figure out how to match %s sheet names with data of length %s' % (len(sheetnames), len(data))
                raise Exception(errormsg)
    else:
        raise Exception('Cannot figure out the format of the data') # This shouldn't happen!

    # Create workbook
    if verbose: print('Creating file %s' % fullpath)
    workbook = xlsxwriter.Workbook(fullpath)

    # Optionally add formats
    if formats is not None:
        if verbose: print('  Adding %s formats' % len(formats))
        workbook_formats = dict()
        for formatkey,formatval in formats.items():
            workbook_formats[formatkey] = workbook.add_format(formatval)
    else:
        thisformat = workbook.add_format({}) # Plain formatting

    # Actually write the data
    for sheetname in datadict.keys():
        if verbose: print('  Creating sheet %s' % sheetname)
        sheetdata   = datadict[sheetname]
        if hasformats: sheetformat = formatdict[sheetname]
        worksheet = workbook.add_worksheet(sheetname)
        for r,row_data in enumerate(sheetdata):
            if verbose: print('    Writing row %s/%s' % (r, len(sheetdata)))
            for c,cell_data in enumerate(row_data):
                if verbose: print('      Writing cell %s/%s' % (c, len(row_data)))
                if hasformats:
                    thisformat = workbook_formats[sheetformat[r][c]] # Get the actual format
                try:
                    if not ut.isnumber(cell_data):
                        cell_data = str(cell_data) # Convert everything except numbers to strings -- use formatting to handle the rest
                    worksheet.write(r, c, cell_data, thisformat) # Write with or without formatting
                except Exception as E:
                    errormsg = 'Could not write cell [%s,%s] data="%s", format="%s": %s' % (r, c, cell_data, thisformat, str(E))
                    raise Exception(errormsg)

    # Either close the workbook and write to file, or return it for further working
    if close:
        if verbose: print('Saving file %s and closing' % filename)
        workbook.close()
        return fullpath
    else:
        if verbose: print('Returning workbook')
        return workbook



##############################################################################
### Pickling support methods
##############################################################################

class Failed(object):
    ''' An empty class to represent a failed object loading '''
    failure_info = odict()

    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        output = ut.prepr(self) # This does not include failure_info since it's a class attribute
        output += self.showfailures(verbose=False, tostring=True)
        return output

    def showfailures(self, verbose=True, tostring=False):
        output = ''
        for f,failure in self.failure_info.enumvals():
            output += '\nFailure %s of %s:\n' % (f+1, len(self.failure_info))
            output += 'Module: %s\n' % failure['module']
            output += 'Class: %s\n' % failure['class']
            output += 'Error: %s\n' % failure['error']
            if verbose:
                output += '\nTraceback:\n'
                output += failure['exception']
                output += '\n\n'
        if tostring:
            return output
        else:
            print(output)
            return None


class Empty(object):
    ''' Another empty class to represent a failed object loading, but do not proceed with setstate '''

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def makefailed(module_name=None, name=None, error=None, exception=None):
    ''' Create a class -- not an object! -- that contains the failure info '''
    key = 'Failure %s' % (len(Failed.failure_info)+1)
    Failed.failure_info[key] = odict()
    Failed.failure_info[key]['module'] = module_name
    Failed.failure_info[key]['class'] = name
    Failed.failure_info[key]['error'] = error
    Failed.failure_info[key]['exception'] = exception
    return Failed


class RobustUnpickler(pickle.Unpickler):
    ''' Try to import an object, and if that fails, return a Failed object rather than crashing '''

    def __init__(self, tmpfile, fix_imports=True, encoding="latin1", errors="ignore"):
        pickle.Unpickler.__init__(self, tmpfile, fix_imports=fix_imports, encoding=encoding, errors=errors)

    def find_class(self, module_name, name, verbose=True):
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, name)
        except Exception as E:
            if verbose: print('Unpickling warning: could not import %s.%s: %s' % (module_name, name, str(E)))
            exception = traceback.format_exc() # Grab the trackback stack
            obj = makefailed(module_name=module_name, name=name, error=E, exception=exception)
        return obj


def unpickler(string=None, filename=None, filestring=None, die=None, verbose=False):

    if die is None: die = False
    try: # Try pickle first
        obj = pkl.loads(string) # Actually load it -- main usage case
    except Exception as E1:
        if die:
            raise E1
        else:
            try:
                if verbose: print('Standard unpickling failed (%s), trying encoding...' % str(E1))
                obj = pkl.loads(string, encoding='latin1') # Try loading it again with different encoding
            except Exception as E2:
                try:
                    if verbose: print('Encoded unpickling failed (%s), trying dill...' % str(E2))
                    import dill # Optional Sciris dependency
                    obj = dill.loads(string) # If that fails, try dill
                except Exception as E3:
                    try:
                        if verbose: print('Dill failed (%s), trying robust unpickler...' % str(E3))
                        obj = RobustUnpickler(io.BytesIO(string)).load() # And if that trails, throw everything at it
                    except Exception as E4:
                        try:
                            if verbose: print('Robust unpickler failed (%s), trying Python 2->3 conversion...' % str(E4))
                            obj = loadobj2to3(filename=filename, filestring=filestring)
                        except Exception as E5:
                            if verbose: print('Python 2->3 conversion failed (%s), giving up...' % str(E5))
                            errormsg = 'All available unpickling methods failed:\n    Standard: %s\n     Encoded: %s\n        Dill: %s\n      Robust: %s\n  Python2->3: %s' % (E1, E2, E3, E4, E5)
                            raise Exception(errormsg)

    if isinstance(obj, Failed):
        print('Warning, the following errors were encountered during unpickling:')
        obj.showfailures(verbose=False)

    return obj


def savepickle(fileobj=None, obj=None):
        ''' Use pickle to do the salty work '''
        fileobj.write(pkl.dumps(obj, protocol=-1))
        return None


def savedill(fileobj=None, obj=None):
    ''' Use dill to do the sour work '''
    try:
        import dill # Optional Sciris dependency
    except ModuleNotFoundError as e:
        raise Exception('The "dill" Python package is not available; please install manually') from e
    fileobj.write(dill.dumps(obj, protocol=-1))
    return None



##############################################################################
### Python 2 legacy support
##############################################################################

not_string_pickleable = ['datetime', 'BytesIO']
byte_objects = ['datetime', 'BytesIO', 'odict', 'spreadsheet', 'blobject']

def loadobj2to3(filename=None, filestring=None, recursionlimit=None):
    '''
    Used automatically by loadobj() to load Python2 objects in Python3 if all other
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

    class StringUnpickler(pickle.Unpickler):
        def find_class(self, module, name, verbose=False):
            if verbose: print('Unpickling string module %s , name %s' % (module, name))
            if name in not_string_pickleable:
                return Empty
            else:
                try:
                    output = pickle.Unpickler.find_class(self,module,name)
                except Exception as E:
                    print('Warning, string unpickling could not find module %s, name %s: %s' % (module, name, str(E)))
                    output = Empty
                return output

    class BytesUnpickler(pickle.Unpickler):
        def find_class(self, module, name, verbose=False):
            if verbose: print('Unpickling bytes module %s , name %s' % (module, name))
            if name in byte_objects:
                try:
                    output = pickle.Unpickler.find_class(self,module,name)
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
                if isinstance(v, datetime.datetime):
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
                if isinstance(v,datetime.datetime):
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
        with GzipFile(filename) as fileobj:
            stringout = loadintostring(fileobj)
        with GzipFile(filename) as fileobj:
            bytesout = loadintobytes(fileobj)

    elif filestring:
        with closing(IO(filestring)) as output:
            with GzipFile(fileobj=output, mode='rb') as fileobj:
                stringout = loadintostring(fileobj)
        with closing(IO(filestring)) as output:
            with GzipFile(fileobj=output, mode='rb') as fileobj:
                bytesout = loadintobytes(fileobj)
    else:
        errormsg = 'You must supply either a filename or a filestring for loadobj() or loadstr(), respectively'
        raise Exception(errormsg)

    # Actually do the load, with correct substitution
    recursive_substitute(stringout, bytesout, recursionlevel=0, recursionlimit=recursionlimit)
    return stringout




##############################################################################
### Twisted pickling methods
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

if six.PY2:
    import cPickle
    class _UniversalPicklingError(pickle.PicklingError, cPickle.PicklingError):
        pass
else:
    _UniversalPicklingError = pickle.PicklingError

def pickleMethod(method):
    if six.PY3: return (unpickleMethod, (method.__name__,         method.__self__, method.__self__.__class__))
    else:       return (unpickleMethod, (method.im_func.__name__, method.im_self,  method.im_class))

def _methodFunction(classObject, methodName):
    methodObject = getattr(classObject, methodName)
    if six.PY3: return methodObject
    else:       return methodObject.im_func

def unpickleMethod(im_name, im_self, im_class):
    if im_self is None:
        return getattr(im_class, im_name)
    try:
        methodFunction = _methodFunction(im_class, im_name)
    except AttributeError:
        assert im_self is not None, "No recourse: no instance to guess from."
        if im_self.__class__ is im_class:
            raise
        return unpickleMethod(im_name, im_self, im_self.__class__)
    else:
        if six.PY3: maybeClass = ()
        else:       maybeClass = tuple([im_class])
        bound = types.MethodType(methodFunction, im_self, *maybeClass)
        return bound

cpreg.pickle(types.MethodType, pickleMethod, unpickleMethod)