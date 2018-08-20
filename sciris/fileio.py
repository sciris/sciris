"""
fileio.py -- code for file management in Sciris 
    
Last update: 5/31/18 (gchadder3)
"""

##############################################################################
### Imports
##############################################################################

# Basic imports
from glob import glob
import os
import re
import dill
from gzip import GzipFile
from contextlib import closing
from xlrd import open_workbook
from xlsxwriter import Workbook
from .utils import promotetolist
from .odict import odict
from .dataframe import dataframe

# Handle types and Python 2/3 compatibility
from six import PY2 as _PY2
if _PY2: # Python 2
    _stringtype = basestring 
    import cPickle as pickle
    from cStringIO import StringIO
else: # Python 3
    _stringtype = str
    import pickle
    from io import StringIO



##############################################################################
### Key file functions
##############################################################################

def saveobj(filename=None, obj=None, compresslevel=5, verbose=True, folder=None, method='pickle'):
    '''
    Save an object to file -- use compression 5 by default, since more is much slower but not much smaller.
    Once saved, can be loaded with loadobj() (q.v.).

    Usage:
        myobj = ['this', 'is', 'a', 'weird', {'object':44}]
        saveobj('myfile.obj', myobj)
    '''
    
    def savepickle(fileobj, obj):
        ''' Use pickle to do the salty work '''
        fileobj.write(pickle.dumps(obj, protocol=-1))
        return None
    
    def savedill(fileobj, obj):
        ''' Use dill to do the sour work '''
        fileobj.write(dill.dumps(obj, protocol=-1))
        return None
    
    fullpath = makefilepath(filename=filename, folder=folder, sanitize=True)
    with GzipFile(fullpath, 'wb', compresslevel=compresslevel) as fileobj:
        if method == 'dill': # If dill is requested, use that
            savedill(fileobj, obj)
        else: # Otherwise, try pickle
            try:    savepickle(fileobj, obj) # Use pickle
            except: savedill(fileobj, obj) # ...but use Dill if that fails
        
    if verbose: print('Object saved to "%s"' % fullpath)
    return fullpath



def loadobj(filename=None, folder=None, verbose=True):
    '''
    Load a saved file.

    Usage:
        obj = loadobj('myfile.obj')
    '''
    
    # Handle loading of either filename or file object
    if isinstance(filename, _stringtype): 
        argtype = 'filename'
        filename = makefilepath(filename=filename, folder=folder) # If it is a file, validate the folder
    else: 
        argtype = 'fileobj'
    kwargs = {'mode': 'rb', argtype: filename}
    with GzipFile(**kwargs) as fileobj:
        filestr = fileobj.read() # Convert it to a string
        try: # Try pickle first
            obj = pickle.loads(filestr) # Actually load it
        except:
            obj = dill.loads(filestr)
    if verbose: print('Object loaded from "%s"' % filename)
    return obj



def dumpstr(obj):
    with closing(StringIO()) as output: # Open a "fake file."
        with GzipFile(fileobj=output, mode='wb') as fileobj:  # Open a Gzip-compressing way to write to this "file."
            fileobj.write(pickle.dumps(obj, protocol=-1)) # Write the string pickle conversion of the object to the "file."
        output.seek(0) # Move the mark to the beginning of the "file."
        result = output.read() # Read all of the content into result.
    return result



def loadstr(gzip_string_pickle):
    with closing(StringIO(gzip_string_pickle)) as output: # Open a "fake file" with the Gzip string pickle in it.
        with GzipFile(fileobj=output, mode='rb') as fileobj: # Set a Gzip reader to pull from the "file."
            string_pickle = fileobj.read() # Read the string pickle from the "file" (applying Gzip decompression).
    obj = pickle.loads(string_pickle) # Return the object gotten from the string pickle.   
    return obj
    
    
    
def loadtext(filename=None, splitlines=False):
    ''' Convenience function for reading a text file '''
    with open(filename) as f: output = f.read()
    if splitlines: output = output.splitlines()
    return output



def savetext(filename=None, string=None):
    ''' Convenience function for reading a text file -- accepts a string or list of strings '''
    if isinstance(string, list): string = '\n'.join(string) # Convert from list to string)
    with open(filename, 'w') as f: f.write(string)
    return None



def getfilelist(folder=None, ext=None, pattern=None):
    ''' A short-hand since glob is annoying '''
    if folder is None: folder = os.getcwd()
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
        * Returns e.g. ('/home/optima/congee', 'soggyrice.prj')
    
    Actual code example from project.py:
        fullpath = makefilepath(filename=filename, folder=folder, default=[self.filename, self.name], ext='prj')
    
    Version: 2017apr04    
    '''
    
    # Initialize
    filefolder = '' # The folder the file will be located in
    filebasename = '' # The filename
    
    # Process filename
    if filename is None:
        defaultnames = promotetolist(default) # Loop over list of default names
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
        filefolder = os.path.abspath(filefolder) 
    if makedirs: # Make sure folder exists
        try: os.makedirs(filefolder)
        except: pass
    if verbose:
        print('From filename="%s", folder="%s", abspath="%s", makedirs="%s", made folder name "%s"' % (filename, folder, abspath, makedirs, filefolder))
    
    fullfile = os.path.join(filefolder, filebasename) # And the full thing
    
    if split: return filefolder, filebasename
    else:     return fullfile # Or can do os.path.split() on output




##############################################################################
### Spreadsheet functions
##############################################################################
    
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



def savespreadsheet(filename=None, data=None, folder=None, sheetnames=None, close=True, formats=None, formatdata=None, verbose=False):
    '''
    Little function to format an output results nicely for Excel. Examples:
    
    import sciris as sc
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