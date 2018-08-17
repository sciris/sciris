"""
fileio.py -- code for file management in Sciris 
    
Last update: 5/31/18 (gchadder3)
"""

#
# Imports
#

try: # Python 2
    import cPickle as pickle
    from cStringIO import StringIO
except: # Python 3
    import pickle
    from io import StringIO
from gzip import GzipFile
from contextlib import closing
from xlrd import open_workbook
import os
from tempfile import mkdtemp
from shutil import rmtree
import atexit
from .utils import makefilepath
from .odict import odict
from .dataframe import dataframe
  
#
# Globals
#

# These will get set by calling code.

# Directory (FileSaveDirectory object) for saved files.
file_save_dir = None

# Directory (FileSaveDirectory object) for file uploads to be routed to.
uploads_dir = None

# Directory (FileSaveDirectory object) for file downloads to be routed to.
downloads_dir = None

#
# Classes
#
    
class FileSaveDirectory(object):
    """
    An object wrapping a directory where files may get saved by the web 
    application.
    
    Methods:
        __init__(dir_path: str [None], temp_dir: bool [False]): void -- 
            constructor
        cleanup(): void -- clean up after web app is exited
        clear(): void -- erase the contents of the directory
        delete(): void -- delete the entire directory
                    
    Attributes:
        dir_path (str) -- the full path of the directory on disk
        is_temp_dir (bool) -- is the directory to be spawned on startup and 
            erased on exit?
        
    Usage:
        >>> new_dir = FileSaveDirectory(transfer_dir_path, temp_dir=True)
    """
    
    def __init__(self, dir_path=None, temp_dir=False):
        # Set whether we are a temp directory.
        self.is_temp_dir = temp_dir
               
        # If no path is specified, create the temp directory.
        if dir_path is None:
            self.dir_path = mkdtemp()
            
        # Otherwise...
        else:
            # Set the path to what was passed in.
            self.dir_path = dir_path
            
            # If the directory doesn't exist yet, create it.
            if not os.path.exists(dir_path):            
                os.mkdir(dir_path)
            
        # Register the cleanup method to be called on web app exit.
        atexit.register(self.cleanup)
            
    def cleanup(self):
        # If we are a temp directory and the directory still exists, do the cleanup.
        if self.is_temp_dir and os.path.exists(self.dir_path):
            # Show cleanup message.
            print('>> Cleaning up FileSaveDirectory at %s' % self.dir_path)
            
            # Delete the entire directory (file contents included).
            self.delete()
            
    def clear(self):
        # Delete the entire directory (file contents included).
        rmtree(self.dir_path)
        
        # Create a fresh direcgtory
        os.mkdir(self.dir_path)
    
    def delete(self):
        # Delete the entire directory (file contents included).
        rmtree(self.dir_path)
        

#
# Pickle / unpickle functions
#

def object_to_string_pickle(obj):
    return pickle.dumps(obj, protocol=-1)

def string_pickle_to_object(string_pickle):
    return pickle.loads(string_pickle)

def object_to_gzip_string_pickle_file(full_file_name, obj, compresslevel=5):
    # Object a Gzip file object to write to and set the compression level 
    # (which the function defaults to 5, since higher is much slower, but 
    # not much more compact).
    with GzipFile(full_file_name, 'wb', compresslevel=compresslevel) as fileobj:
        # Write the string pickle conversion of the object to the file.
        fileobj.write(object_to_string_pickle(obj))

def gzip_string_pickle_file_to_object(full_file_name):
    # Object a Gzip file object to read from.
    with GzipFile(full_file_name, 'rb') as fileobj:
        # Read the string pickle from the file.
        string_pickle = fileobj.read()
        
    # Return the object gotten from the string pickle.    
    return string_pickle_to_object(string_pickle)

def object_to_gzip_string_pickle(obj):
    # Start with a null result.
    result = None
    
    # Open a "fake file."
    with closing(StringIO()) as output:
        # Open a Gzip-compressing way to write to this "file."
        with GzipFile(fileobj=output, mode='wb') as fileobj: 
            # Write the string pickle conversion of the object to the "file."
            fileobj.write(object_to_string_pickle(obj))
            
        # Move the mark to the beginning of the "file."
        output.seek(0)
        
        # Read all of the content into result.
        result = output.read()
        
    # Return the read-in result.
    return result

# Alias for above function.
def dumpstr(obj):
    return object_to_gzip_string_pickle(obj)

def gzip_string_pickle_to_object(gzip_string_pickle):
    # Open a "fake file" with the Gzip string pickle in it.
    with closing(StringIO(gzip_string_pickle)) as output:
        # Set a Gzip reader to pull from the "file."
        with GzipFile(fileobj=output, mode='rb') as fileobj: 
            # Read the string pickle from the "file" (applying Gzip 
            # decompression).
            string_pickle = fileobj.read()  
            
            # Extract the object from the string pickle.
            obj = string_pickle_to_object(string_pickle)
            
    # Return the object.
    return obj

# Alias for above function.
def loadstr(gzip_string_pickle):
    return gzip_string_pickle_to_object(gzip_string_pickle)

#
# Excel spreadsheet functions
#
    
def loadspreadsheet(filename=None, folder=None, sheetname=None, sheetnum=None, asdataframe=True):
    '''
    Load a spreadsheet
    '''

    fullpath = makefilepath(filename=filename, folder=folder)
    workbook = open_workbook(fullpath)
    if sheetname is not None: 
        sheet = workbook.sheet_by_name(sheetname)
    else:
        if sheetnum is None: sheetnum = 0
        sheet = workbook.sheet_by_index(sheetnum)
    
    # Load the raw data
    rawdata = []
    for rownum in range(sheet.nrows-1):
        rawdata.append(odict())
        for colnum in range(sheet.ncols):
            attr = sheet.cell_value(0,colnum)
            val = sheet.cell_value(rownum+1,colnum)
            try:    val = float(val) # Convert it to a number if possible
            except: 
                try:    val = str(val)  # But give up easily and convert to a string (not Unicode)
                except: pass # Still no dice? Fine, we tried
            rawdata[rownum][attr] = val
    
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

def export_xlsx(filename=None, data=None, folder=None, sheetnames=None, close=True, formats=None, formatdata=None, verbose=False):
    '''
    Little function to format an output results nicely for Excel. Examples:
    
    import sciris.core as sc
    import pylab as pl
    
    # Simple example
    testdata1 = pl.rand(8,3)
    sc.export_file(filename='test1.xlsx', data=testdata1)
    
    # Include column headers
    test2headers = [['A','B','C']] # Need double to get right shape
    test2values = pl.rand(8,3).tolist()
    testdata2 = test2headers + test2values
    sc.export_file(filename='test2.xlsx', data=testdata2)
    
    # Multiple sheets
    testdata3 = [pl.rand(10,10), pl.rand(20,5)]
    sheetnames = ['Ten by ten', 'Twenty by five']
    sc.export_file(filename='test3.xlsx', data=testdata3, sheetnames=sheetnames)
    
    # Supply data as an odict
    testdata4 = sc.odict([('First sheet', pl.rand(6,2)), ('Second sheet', pl.rand(3,3))])
    sc.export_file(filename='test4.xlsx', data=testdata4, sheetnames=sheetnames)
    
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
    sc.export_file(filename='test5.xlsx', data=testdata5, formats=formats, formatdata=formatdata)
    '''
    from xlsxwriter import Workbook
    
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
    workbook = Workbook(fullpath)
    
    # Optionally add formats
    if formats is not None:
        if verbose: print('  Adding %s formats' % len(formats))
        workbook_formats = dict()
        for formatkey,formatval in formats.items():
            workbook_formats[formatkey] = workbook.add_format(formatval)
       
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
                    worksheet.write(r, c, cell_data, thisformat) # Write with formatting
                else:
                    worksheet.write(r, c, cell_data) # Write without formatting
    
    # Either close the workbook and write to file, or return it for further working
    if close:
        if verbose: print('Saving file %s and closing' % filename)
        workbook.close()
        return None
    else:
        if verbose: print('Returning workbook')
        return workbook