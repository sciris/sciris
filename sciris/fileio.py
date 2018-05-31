#############################################################################################################################
### Imports
#############################################################################################################################

try: # Python 2
    import cPickle as pickle
    from cStringIO import StringIO
except: # Python 3
    import pickle
    from io import StringIO
from gzip import GzipFile
from contextlib import closing
from sciris.utils import makefilepath, odict, dataframe
from xlrd import open_workbook



#############################################################################################################################
### Basic I/O functions
#############################################################################################################################

def dumpstr(obj):
    ''' Write data to a fake file object,then read from it -- used on the FE '''
    result = None
    with closing(StringIO()) as output:
        with GzipFile(fileobj = output, mode = 'wb') as fileobj: 
            fileobj.write(pickle.dumps(obj, protocol=-1))
        output.seek(0)
        result = output.read()
    return result


def loadstr(source):
    ''' Load data from a fake file object -- also used on the FE '''
    with closing(StringIO(source)) as output:
        with GzipFile(fileobj = output, mode = 'rb') as fileobj:
            filestr = fileobj.read() # Convert it to a string
            obj = pickle.loads(filestr) # Actually load it
    return obj


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