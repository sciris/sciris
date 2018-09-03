"""
Version:
"""

import sciris as sc
import pylab as pl
import openpyxl


torun = [
#'savespreadsheet',
#'loadspreadsheet',
#'Blobject',
'Spreadsheet',
#'saveobj',
#'loadobj',
#'savetext',
#'loadtext',
#'getfilelist',
]


def check(test, dependencies=None):
    if dependencies == None: dependencies = []
    tf = any([t in torun for t in [test]+dependencies])
    return tf
    
# Define filenames
files = sc.odict()
files['excel']  = 'test.xlsx'
files['binary'] = 'test.obj'
files['text']   = 'text.txt'

# Define the test data
nrows = 15
ncols = 3
testdata   = pl.zeros((nrows+1, ncols), dtype=object) # Includes header row
testdata[0,:] = ['A', 'B', 'C'] # Create header
testdata[1:,:] = pl.rand(nrows,ncols) # Create data

# Test spreadsheet writing, and create the file for later
if check('savespreadsheet', ['loadspreadsheet', 'Spreadsheet']):
    
    formats = {
        'header':{'bold':True, 'bg_color':'#3c7d3e', 'color':'#ffffff'},
        'plain': {},
        'big':   {'bg_color':'#ffcccc'}}
    formatdata = pl.zeros((nrows+1, ncols), dtype=object) # Format data needs to be the same size
    formatdata[1:,:] = 'plain' # Format data
    formatdata[testdata>0.7] = 'big' # Find "big" numbers and format them differently
    formatdata[0,:] = 'header' # Format header
    sc.savespreadsheet(filename=files.excel, data=testdata, formats=formats, formatdata=formatdata)


# Test loading
if check('loadspreadsheet'):
    data = sc.loadspreadsheet(files.excel)
    print(data)


if check('Blobject'):
    blob = sc.Blobject(files.excel)
    f = blob.tofile()
    wb = openpyxl.load_workbook(f)
    ws = wb.active
    ws['B7'] = 'Hi!     '
    wb.save(f)
    blob.load(f)
    blob.tofile(output=False)
    data = sc.loadspreadsheet(fileobj=blob.bytes)
    print(blob)
    sc.pp(data)
    

# Test spreadsheet saving
if check('Spreadsheet'):
    S = sc.Spreadsheet(files.excel)
    S.writecells(cells=['A1','B2','C3','D4'], vals=['this','is','a','test'])
    S.save()
    data = S.readcells()
    print(S)
    sc.pp(data)
    


if check('saveobj', ['loadobj']):
    sc.saveobj(files.binary, testdata)


if check('loadobj'):
    obj = sc.loadobj(files.binary)
    print(obj)


if check('savetext', ['loadtext']):
    sc.savetext(files.text, testdata)


if check('loadtext'):
    obj = sc.loadtext(files.text)
    print(obj)
    

if check('getfilelist'):
    print('Files in current folder:')
    sc.pp(sc.getfilelist())

print('Done.')