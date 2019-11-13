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
import traceback
import numpy as np
from glob import glob
from gzip import GzipFile
from zipfile import ZipFile
from contextlib import closing
from collections import OrderedDict
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
    if ut.isstring(filename): 
        argtype = 'filename'
        filename = makefilepath(filename=filename, folder=folder) # If it is a file, validate the folder
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


    if filename is None:
        bytesobj = io.BytesIO()
    else:
        filename = makefilepath(filename=filename, folder=folder, default='default.obj', sanitize=True)
        bytesobj = None

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

__all__ += ['sanitizejson', 'loadjson', 'savejson']


def sanitizejson(obj, verbose=True, die=False, tostring=False):
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

    if tostring: return json.dumps(output)
    else:        return output



def loadjson(filename=None, folder=None):
    ''' Convenience function for reading a JSON file '''
    filename = makefilepath(filename=filename, folder=folder)
    with open(filename) as f:
        output = json.load(f)
    return output



def savejson(filename=None, obj=None, folder=None):
    ''' Convenience function for saving a JSON '''
    filename = makefilepath(filename=filename, folder=folder)
    with open(filename, 'w') as f:
        json.dump(sanitizejson(obj), f)
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
        import xlrd # Optional import
        book = xlrd.open_workbook(file_contents=self.tofile().read(), *args, **kwargs)
        return book
    
    def openpyexcel(self, *args, **kwargs):
        ''' Return a book as opened by openpyexcel '''
        import openpyexcel # Optional import
        self.tofile(output=False)
        book = openpyexcel.load_workbook(self.bytes, *args, **kwargs) # This stream can be passed straight to openpyexcel
        return book
        
    def pandas(self, *args, **kwargs):
        ''' Return a book as opened by pandas '''
        import pandas # Optional import
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
    import xlrd # Optional import
    
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
    sc.savespreadsheet(filename='test4.xlsx', data=testdata4)
    
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
    import xlsxwriter # Optional import
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
### Powerpoint functions
##############################################################################
        
__all__ += ['savepptx']


def savepptx(filename=None, template=None, slides=None, image_path='', close=True, verbose=False):
    '''
    :param filename: A name for the desired output document. It should end in .pptx.
    :param template: The filepath to the powerpoint template which is to be used.
    :param slides: A list of dicts, with each dict specifying the attributes to be added in a slide. The format of each
    dict should match with the construction of an appropriate slide within your template.
    :param image_path: The filepath to any images which are to be added to a slide.

    If no template powerpoint is provided then an automatic template will be used, but the functionality will be limited
    and the prescribed slide layouts will have to match either a NxM grid of images (with or without a supplementary
    image such as a legend and with or without a small amount of text) or a single section of text. See
    'allowed_features' below for the allowed attributes, where the 'style' can be 'text' or a 1x1 to 3x3 grid as defined
    in 'image_sets'. Note that images will be scaled to fit on the slide which will likely impact aspect ratios, and if
    an image slide with text is input then the images may be placed over the text.

    If a template is provided then each input slide 'style' must match with the layout name of the appropriate slide
    within the template's Slide Master. For example, if a slide layout was constructed to hold a title, 3 images and a
    section of text with the layout name set as '3x1 with text' then an appropriate dict for the slide would be:
    {'style': '3x1 with text', 'title': 'The title of this slide is...', 'img1': 'firstpic.png', 'img2': 'secondpic.jpg,
    'img3': 'thirdpic.pdf', 'text1': 'Words words words, etc etc etc.'}
    Note that the images (and text if multiple sets of text are defined) will fill the appropriate placeholders from
    left to right, top to bottom.
    '''
    import pptx as pp
    
    template_provided = False
    allowed_features = ['style', 'title', 'legend', 'img1', 'img2', 'img3', 'img4', 'img5', 'img6', 'img7', 'img8',
                        'img9', 'text1'] #these are the allowed features for the default XxY layouts of images if no template is provided, more generally any feature names can be added
    image_sets = {'1x1': [1, 1], '1x2': [1, 2], '1x3': [1, 3], '2x1': [2, 1], '2x2': [2, 2], '2x3': [2, 3],
                  '3x1': [3, 1], '3x2': [3, 2], '3x3': [3, 3]}
    if filename is None:
        filename = 'output_file.pptx'
        if verbose: print("A filename was not specified. As such, the output will be saved as: %s" %filename)
    if template is None:
        prs = pp.Presentation()
        if verbose: print("A presentation template was not specified. As such, an automatic template is being used.")
    else:
        if os.path.exists(template):
            prs = pp.Presentation(template)
            template_provided = True
        else:
            prs = pp.Presentation()
            if verbose: print("The specified template file could not be found. As such, an automatic template is being used.")
    if slides is None or slides == []:
        if verbose: print("No slide data was provided. As such, the blank template has been saved as: %s" %filename)
    else:
        if template_provided:
            for s, slide in enumerate(slides):
                for entry in slide.keys():
                    entry.lower()
                num_features = len(slide)
                if num_features > 0:
                    prs = update_custom(prs, slide, s, image_path=image_path, verbose=verbose)
        else:
            for slide in slides:
                for attr in slide.keys():
                    if attr not in allowed_features:
#                        del slide[attr]
                        if verbose: print("Slide contained an invalid attribute: %s, which has been ignored." %attr)
                num_features = len(slide)
                if num_features > 0:
                    try:
                        format = slide['style']
                    except KeyError:
                        prs = update_fail(prs, verbose=verbose)
                    if format in image_sets:
                        prs = update_image(prs, slide, arrange=image_sets[format], image_path=image_path, verbose=verbose)
                    elif format == 'text':
                        prs = update_text(prs, slide, verbose=verbose)
                    else:
                        prs = update_fail(prs, verbose=verbose)
                else:
                    prs = update_fail(prs, verbose=verbose)
    if close:
        prs.save(filename)
        if verbose: print("The powerpoint has been saved as: %s" %filename)
        return filename
    else:
        return prs


def update_image(presentation, slide_details, arrange=[1, 1], image_path=None, verbose=False):
    num_attributes = len(slide_details) - 1
    num_image = arrange[0]*arrange[1]
    slide_layout = presentation.slide_layouts[1]
    slide = presentation.slides.add_slide(slide_layout)
    text_present = False
    for entry in slide_details.keys():
        if 'text' in entry or 'par' in entry or 'txt' in entry:
            text_present = True
            break
    if 'title' in slide_details.keys():
        num_attributes -= 1
        title_placeholder = slide.shapes.title
        title_placeholder.text = slide_details['title']
        del slide_details['title']
        title_present = True
    else:
        title_present = False
    if 'legend' in slide_details.keys():
        num_attributes -= 1
        if title_present:
            if text_present:
                left, top, height, width = getproperties(ncols=arrange[0] + 1, nrows=arrange[1], minver=8)
            else:
                left, top, height, width = getproperties(ncols=arrange[0] + 1, nrows=arrange[1], minver=4)
        else:
            if text_present:
                left, top, height, width = getproperties(ncols=arrange[0] + 1, nrows=arrange[1], minver=8)
            else:
                left, top, height, width = getproperties(ncols=arrange[0] + 1, nrows=arrange[1])
        image_name = slide_details['legend']
        img_path = os.path.join(image_path, image_name)
        pic = slide.shapes.add_picture(img_path, left[-1], top[-1], height=height)
        del slide_details['legend']
        legend_present = True
    else:
        if title_present:
            if text_present:
                left, top, height, width = getproperties(ncols=arrange[0], nrows=arrange[1], minver=8)
            else:
                left, top, height, width = getproperties(ncols=arrange[0], nrows=arrange[1], minver=4)
        else:
            if text_present:
                left, top, height, width = getproperties(ncols=arrange[0], nrows=arrange[1], minver=8)
            else:
                left, top, height, width = getproperties(ncols=arrange[0], nrows=arrange[1])
        legend_present = False
    counter = 0
    for image in list(range(num_image)):
        num_attributes -= 1
        image_key = 'img' + str(image + 1)
        if image_key in slide_details.keys():
            if legend_present and (image + 1) % (arrange[0] + 1) == 0:
                counter += 1
            image_name = slide_details[image_key]
            img_path = os.path.join(image_path, image_name)
            pic = slide.shapes.add_picture(img_path, left[counter], top[counter], height=height, width=width)
            del slide_details[image_key]
            counter += 1
        else:
            if verbose: print("%s image(s) expected in the slide, but %s was either keyed incorrectly or does not exist."
                  %(num_image, image_key))
    if num_attributes > 0:
        presentation = update_text(presentation, slide_details, slide=slide, verbose=verbose)
    return presentation

def update_text(presentation, slide_details, slide=None, verbose=False):
    num_attributes = len(slide_details) - 1
    if slide is None:
        slide_layout = presentation.slide_layouts[1]
        slide = presentation.slides.add_slide(slide_layout)
    if 'title' in slide_details.keys():
        num_attributes -= 1
        title_placeholder = slide.shapes.title
        title_placeholder.text = slide_details['title']
    if num_attributes > 1:
        for text in list(range(num_attributes)):
            text_key = 'text' + str(text + 1)
            if text_key in slide_details.keys():
                text_info = slide_details[text_key]
                text_frame = slide.placeholders[1].text_frame
                if text == 0:
                    text_frame.clear()
                    p = text_frame.paragraphs[text]
                    run = p.add_run()
                    run.text = text_info
                else:
                    p = text_frame.paragraphs[0]
                    run = p.add_run()
                    run.text = text_info
            else:
                if verbose: print("%s paragraphs were expected in the slide, but '%s' was either keyed incorrectly or does not "
                      "exist." % (num_attributes, text_key))
    elif num_attributes > 0:
        text_key = 'text1'
        if text_key in slide_details.keys():
            text_info = slide_details[text_key]
            text_frame = slide.placeholders[1].text_frame
            text_frame.clear()
            p = text_frame.paragraphs[0]
            run = p.add_run()
            run.text = text_info
        else:
            if verbose: print("A paragraph was expected in the slide, but 'text1' was either keyed incorrectly or does not exist.")
    else:
        if verbose: print("At least one piece of text was expected for the slide, but none were found.")
    return presentation

def update_fail(presentation, verbose=False):
    slide_layout = presentation.slide_layouts[6]
    slide = presentation.slides.add_slide(slide_layout)
    if verbose: print("The slide format provided was not understood. A blank slide was added in its place. Please provide slide"
          " attributes according to the specifications of savepptx.")
    return presentation

def update_custom(presentation, slide_details, slide_num, image_path='', verbose=False):
    name = slide_details['style']
    slide = False
    if len(presentation.slide_masters) > 1:
        if verbose: print('There are %d Slide Masters saved in this template, by default the first is being used. If this was not'
              ' intended then please ensure that only one Slide Master is present in the template, or that the'
              ' intended Slide Master is at the top of the order.' % len(presentation.slide_masters))
        master_layouts = presentation.slide_masters[0].slide_layouts
    elif len(presentation.slide_masters) == 1:
        master_layouts = presentation.slide_masters[0].slide_layouts
    else:
        presentation = update_fail(presentation, verbose=verbose)
        return presentation
    for layout in master_layouts:
        if layout.name == name:
            slide = presentation.slides.add_slide(layout)
    if slide:
        for shape in slide.placeholders:
            if 'Title' in shape.name:
                if 'title' in slide_details.keys():
                    if not slide_details['title'] is None:
                        shape.text = slide_details['title']
                else:
                    shape.element.getparent().remove(shape.element)
            elif 'Picture' in shape.name: #NOTE: this means that 'Content' placeholders will always be treated only as 'Picture' placeholders, edits would have to be made to enable them to be flexible
                for entry in slide_details.keys():
                    if 'im' in entry or 'pic' in entry:
                        if not slide_details[entry] is None:
                            image_name = slide_details[entry]
                            img_path = os.path.join(image_path, image_name)
                            pic = slide.shapes.add_picture(img_path, shape.left, shape.top, width=shape.width
    #                                                       , height=shape.height, width=shape.width
                                                           )
                            shape.element.getparent().remove(shape.element)
                        del slide_details[entry]
                        break
            elif 'Content' in shape.name: #Assume that 'content' is for legends specifically and only!
                for entry in slide_details.keys():
                    if 'leg' in entry:
                        if not slide_details[entry] is None:
                            image_name = slide_details[entry]
                            img_path = os.path.join(image_path, image_name)
                            pic = slide.shapes.add_picture(img_path, shape.left, shape.top) #don't resize legends
                            shape.element.getparent().remove(shape.element)
                        del slide_details[entry]
                        break
            elif 'Text' in shape.name:
                filled = False
                for entry in slide_details.keys():
                    if 'text' in entry or 'txt' in entry:
                        if not slide_details[entry] is None:
                            shape.text = slide_details[entry]
                        filled = True
                        del slide_details[entry]
                        break
                    elif 'par' in entry or 'paras' in entry: #it should be a list of tuples giving paragraphs and the level of each
                        if not slide_details[entry] is None:
                            p = shape.text_frame.paragraphs[0]    
                            for pn, par in enumerate(slide_details[entry]):
                                if pn>0:
                                    p = shape.text_frame.add_paragraph()
                                p.text = par[1]
                                p.level = par[0]
                                
                        filled = True  
                        del slide_details[entry]
                        break
                if not filled:
                    shape.element.getparent().remove(shape.element)
            elif 'Table' in shape.name:
                for entry in slide_details.keys():
                    if 'tab' in entry :
                        if not slide_details[entry] is None:
                            data = slide_details[entry]
                            graphic_frame = shape.insert_table(rows=len(data), cols=len(data[0]))
                            table = graphic_frame.table
                            for row in range(len(data)):
                                for col in range(len(data[0])):
                                    table.cell(row, col).text = str(data[row][col])
#                            shape.element.getparent().remove(shape.element)
                        del slide_details[entry]
                        break
                    

        if len(slide_details) > 1:
            for entry in slide_details.keys():
                if 'im' in entry or 'pic' in entry or 'leg' in entry:
                    if verbose: print('Note: More images were provided for slide %d (%s) than there were Picture Placeholders available. '
                          'As such, some images may have been omitted.' % (slide_num, slide_details['title']))
                elif 'text' in entry or 'par' in entry or 'txt' in entry:
                    if verbose: print('Note: More text paragraphs were provided for slide %d (%s) than there were Text Placeholders available'
                          '. As such, some text may have been omitted.' % (slide_num, slide_details['title']))
        elif len(slide_details) > 0:
            if 'im' in slide_details.keys()[0] or 'pic' in slide_details.keys()[0] or 'leg' in slide_details.keys()[0]:
                if verbose: print('Note: More images were provided for slide %d (%s) than there were Picture Placeholders available. '
                      'As such, some images may have been omitted.' % (slide_num, slide_details['title']))
            elif 'text' in slide_details.keys()[0] or 'par' in slide_details.keys()[0] or 'txt' in slide_details.keys()[0]:
                if verbose: print(
                    'Note: More text paragraphs were provided for slide %d (%s) than there were Text Placeholders available'
                    '. As such, some text may have been omitted.' % (slide_num, slide_details['title']))
        return presentation
    else:
        if verbose: print('Note: The style (%s) attributed to slide %d does not match any layout name in the Slide Master'
              % (name, slide_num))
        return presentation


def getproperties(ncols, nrows, minver=2):
    import numpy
    import pptx as pp
    # image settings for slides
    aspectratio = 1.75
    minhor = 1.0
    availablewidth = 25
    availableheight = 19
    minpercentgap = 0.05

    # image placement calculations
    figheight = float((1 - (nrows - 1) * minpercentgap) * availableheight - minver) / nrows
    figwidth = float((1 - (ncols - 1) * minpercentgap) * availablewidth - minhor) / ncols
    availableaspectratio = figwidth / figheight

    if availableaspectratio > aspectratio:
        ygap = minpercentgap * float(availableheight) / nrows
        figheight = float(availableheight - (nrows - 1) * ygap) / nrows
        figwidth = aspectratio * figheight
        if ncols == 1:
            figwidth = aspectratio * figheight
            xgap = 0
        else:
            xgap = float(availablewidth - ncols * figwidth) / (ncols - 1)
    else:
        xgap = minpercentgap * float(availablewidth) / ncols
        figwidth = float(availablewidth - (ncols - 1) * xgap) / ncols
        figheight = figwidth / float(aspectratio)
        if nrows == 1:
            figheight = figwidth / float(aspectratio)
            ygap = 0
        else:
            ygap = minpercentgap * float(availableheight) / nrows
    while nrows * figheight + (nrows - 1) * ygap + minver > availableheight:
        figheight *= 0.95
        ygap *= 0.95
    while ncols * figwidth + (ncols - 1) * xgap + minhor > availablewidth:
        figwidth *= 0.95
        xgap *= 0.95
    leftcoords = []
    temptop = []
    for j in range(nrows):
        temptop.append(pp.util.Cm(minver + j * (figheight + ygap)))
        for i in range(ncols):
            leftcoords.append(pp.util.Cm(minhor + i * (figwidth + xgap)))

    topcoords = numpy.repeat(temptop, ncols)
    figheight = pp.util.Cm(figheight)
    figwidth = pp.util.Cm(figwidth)

    return leftcoords, topcoords, figheight, figwidth

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
    
    def find_class(self, module_name, name, verbose=False):
        try:
            module = __import__(module_name)
            obj = getattr(module, name)
        except:
            try:
                string = 'from %s import %s as obj' % (module_name, name)
                exec(string)
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
    import dill # Optional Sciris dependency
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