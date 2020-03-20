'''
Test Sciris file I/O functions.
'''

import pytest
import pylab as pl
import sciris as sc


def legacy_tests():
    '''
    Preserved for completeness, but too fragile to be used in automated unit testing
    due to reliance on openpyxl (which is not a required Sciris dependency).
    '''

    import openpyxl
    import os

    torun = [
    'savespreadsheet',
    'loadspreadsheet',
    'readcells',
    'Blobject',
    'Spreadsheet',
    'saveobj',
    'loadobj',
    'savetext',
    'loadtext',
    'getfilelist',
    'savezip',
    'loadfailed',
    ]


    def check(test, dependencies=None):
        if dependencies == None: dependencies = []
        tf = any([t in torun for t in [test]+dependencies])
        return tf

    # Define filenames
    filedir = 'files' + os.sep
    files = sc.prettyobj()
    files.excel  = filedir+'test.xlsx'
    files.binary = filedir+'test.obj'
    files.text   = filedir+'text.txt'
    files.zip    = filedir+'test.zip'
    tidyup = True

    # Define the test data
    nrows = 15
    ncols = 3
    testdata   = pl.zeros((nrows+1, ncols), dtype=object) # Includes header row
    testdata[0,:] = ['A', 'B', 'C'] # Create header
    testdata[1:,:] = pl.rand(nrows,ncols) # Create data

    # Test spreadsheet writing, and create the file for later
    if check('savespreadsheet', ['loadspreadsheet', 'Spreadsheet', 'savezip']):

        formats = {
            'header':{'bold':True, 'bg_color':'#3c7d3e', 'color':'#ffffff'},
            'plain': {},
            'big':   {'bg_color':'#ffcccc'}}
        formatdata = pl.zeros((nrows+1, ncols), dtype=object) # Format data needs to be the same size
        formatdata[1:,:] = 'plain' # Format data
        formatdata[1:,:][testdata[1:,:]>0.7] = 'big' # Find "big" numbers and format them differently
        formatdata[0,:] = 'header' # Format header
        sc.savespreadsheet(filename=files.excel, data=testdata, formats=formats, formatdata=formatdata)


    # Test loading
    if check('loadspreadsheet'):
        sc.heading('Loading spreadsheet')
        data = sc.loadspreadsheet(files.excel)
        print(data)


    if check('readcells'):
        sc.heading('Reading cells')
        wb = sc.Spreadsheet(filename=filedir+'exampledata.xlsx') # Load a sample databook to try pulling cells from
        celltest = wb.readcells(method='xlrd', sheetname='Baseline year population inputs', cells=[[46, 2], [47, 2]]) # Grab cells using xlrd
        celltest2 = wb.readcells(method='openpyexcel', wbargs={'data_only': True}, sheetname='Baseline year population inputs', cells=[[46, 2], [47, 2]]) # Grab cells using openpyexcel.  You have to set wbargs={'data_only': True} to pull out cached values instead of formula strings
        print('xlrd output: %s' % celltest)
        print('openpyxl output: %s' % celltest2)


    if check('Blobject'):
        sc.heading('Loading a blobject')
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
        sc.heading('Using a Spreadsheet')
        S = sc.Spreadsheet(files.excel)
        S.writecells(cells=['A6','B7','C8','D9'], vals=['This','is','a','test']) # Method 1
        S.writecells(cells=[pl.array([7,1])+i for i in range(4)], vals=['And','so','is','this']) # Method 2
        newdata = (pl.rand(3,3)*100).round()
        S.writecells(startrow=14, startcol=1, vals=newdata, verbose=True) # Method 3
        S.save()
        data = S.readcells(header=False)
        print(S)
        sc.pp(data)


    if check('saveobj', ['loadobj']):
        sc.heading('Saveobj/loadobj')
        sc.saveobj(files.binary, testdata)


    if check('loadobj'):
        obj = sc.loadobj(files.binary)
        print(obj)


    if check('savetext', ['loadtext', 'savezip']):
        sc.heading('Savetext/loadtext')
        sc.savetext(files.text, testdata)


    if check('loadtext'):
        obj = sc.loadtext(files.text)
        print(obj)


    if check('getfilelist'):
        sc.heading('Get files')
        print('Files in current folder:')
        sc.pp(sc.getfilelist())


    if check('savezip'):
        sc.heading('Save zip')
        sc.savezip(files.zip, [files.text, files.excel])


    if check('loadfailed'):
        '''
        Check that loading an object with a non-existent class works. The file
        deadclass.obj was created with:

        deadclass.py:
        -------------------------------------------------
        class DeadClass():
            def __init__(self, x):
                self.x = x
        -------------------------------------------------

        then:
        -------------------------------------------------
        import deadclass as dc
        import sciris as sc
        deadclass = dc.DeadClass(238473)
        sc.saveobj('deadclass.obj', deadclass)
        -------------------------------------------------
        '''
        sc.heading('Intentionally loading corrupted file')
        obj = sc.loadobj(filedir+'deadclass.obj')
        print('Loading corrupted object succeeded, x=%s' % obj.x)


    # Tidy up
    if tidyup:
        sc.blank()
        sc.heading('Tidying up')
        for fn in [files.excel, files.binary, files.text, files.zip]:
            try:
                os.remove(fn)
                print('Removed %s' % fn)
            except:
                pass

    print('Done, all fileio tests succeeded')


def test_json():
    sc.heading('Testing JSON read/write functions')

    not_jsonifiable = sc.Blobject() # Create an object that can't be JSON serialized

    print('Testing jsonifying a NON-jsonifiable object:')
    sc.jsonify(not_jsonifiable, die=False) # Will return a string representation
    with pytest.raises(Exception):
        sc.sanitizejson(not_jsonifiable, die=True) # Will die

    jsonifiable = sc.objdict().make(keys=['a','b'], vals=pl.rand(10))
    json_obj = sc.jsonify(jsonifiable)
    json_str = sc.jsonify(jsonifiable, tostring=True, indent=2) # kwargs are passed to json.dumps()

    print('JSON as sanitized object:')
    print(json_obj)
    print('JSON as string:')
    print(json_str)

    return json_str



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    try:
        legacy_tests()
    except:
        print('\n\nNOTE: Legacy tests failed -- this is no major cause for concern, but the cause was:\n')
        print(sc.traceback())
        print('Continuing with other tests...')

    json = test_json()



    sc.toc()