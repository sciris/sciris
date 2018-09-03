"""
Version:
"""

import sciris as sc
import pylab as pl

testfile = 'test.xlsx'

# Test spreadsheet writing
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
sc.savespreadsheet(filename=testfile, data=testdata5, formats=formats, formatdata=formatdata)


# Test spreadsheet saving
S = sc.Spreadsheet(testfile)

print('Done.')