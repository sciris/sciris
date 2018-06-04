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
from .utils import makefilepath
from .odict import odict
from .dataframe import dataframe
from xlrd import open_workbook
import os
from tempfile import mkdtemp
from shutil import rmtree
import atexit
  
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
            print '>> Cleaning up FileSaveDirectory at %s' % self.dir_path
            
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

def export_file(filename=None, data=None, sheetname=None, close=True):
    '''
    Little function to format an output results nicely for Excel
    '''
    from xlsxwriter import Workbook
    
    if filename  is None: filename  = 'default.xlsx'
    if sheetname is None: sheetname = 'Sheet1'
    
    workbook = Workbook(filename)
    worksheet = workbook.add_worksheet(sheetname)
    
    for r,row_data in enumerate(data):
        for c,cell_data in enumerate(row_data):
            worksheet.write(r, c, cell_data)
        
    if close:
        workbook.close()
        return None
    else:
        return workbook