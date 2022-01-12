'''
Test Sciris file I/O functions.
'''

import os
import pylab as pl
import pandas as pd
import openpyxl
import sciris as sc

def test_spreadsheets():
    '''
    Preserved for completeness, but fairly fragile since relies on not-well-trodden
    Excel libraries.
    '''

    # Define filenames
    filedir = 'files' + os.sep
    files = sc.prettyobj()
    files.excel  = filedir + 'test.xlsx'
    files.binary = filedir + 'test.obj'
    files.text   = filedir + 'text.txt'
    files.zip    = filedir + 'test.zip'
    tidyup = True

    # Define the test data
    nrows = 15
    ncols = 3
    testdata   = pl.zeros((nrows+1, ncols), dtype=object) # Includes header row
    testdata[0,:] = ['A', 'B', 'C'] # Create header
    testdata[1:,:] = pl.rand(nrows,ncols) # Create data

    # Test spreadsheet writing, and create the file for later
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
    sc.heading('Loading spreadsheet')
    data = sc.loadspreadsheet(files.excel)
    print(data)

    excel_path = filedir+'exampledata.xlsx'
    if os.path.exists(excel_path):
        sc.heading('Reading cells')
        wb = sc.Spreadsheet(filename=excel_path) # Load a sample databook to try pulling cells from
        celltest = wb.readcells(method='openpyxl', wbargs={'data_only': True}, sheetname='Baseline year population inputs', cells=[[46, 2], [47, 2]]) # Grab cells using openpyxl.  You have to set wbargs={'data_only': True} to pull out cached values instead of formula strings
        print(f'openpyxl output: {celltest}')
    else:
        print(f'{excel_path} not found, skipping...')


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

    sc.heading('Saveobj/loadobj')
    sc.saveobj(files.binary, testdata)

    obj = sc.loadobj(files.binary)
    print(obj)

    sc.heading('Savetext/loadtext')
    sc.savetext(files.text, testdata)

    obj = sc.loadtext(files.text)
    print(obj)

    sc.heading('Get files')
    print('Files in current folder:')
    TF = [True,False]
    for tf in TF:
        sc.pp(sc.getfilelist(abspath=tf, filesonly=tf, foldersonly=not(tf), nopath=tf, aspath=tf))

    sc.heading('Save zip')
    sc.savezip(files.zip, [files.text, files.excel])


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

    class LiveClass():
        def __init__(self, x):
            self.x = x

    dead_path = filedir+'deadclass.obj'
    if os.path.exists(dead_path):
        sc.heading('Intentionally loading corrupted file')
        print('Loading with no remapping...')
        obj = sc.loadobj(dead_path)
        print(f'Loading corrupted object succeeded, x={obj.x}')

        print('Loading with remapping...')
        obj2 = sc.loadobj(dead_path, remapping={'deadclass.DeadClass':LiveClass})
        print(f'Loading remapped object succeeded, x={obj2.x}, object: {obj2}')
    else:
        print(f'{dead_path} not found, skipping...')

    # Tidy up
    if tidyup:
        sc.blank()
        sc.heading('Tidying up')
        for fn in [files.excel, files.binary, files.text, files.zip, 'spreadsheet.xlsx']:
            try:
                os.remove(fn)
                print('Removed %s' % fn)
            except:
                pass

    print('Done, all fileio tests succeeded')

    return S


def test_json():
    sc.heading('Testing JSON read/write functions')

    not_jsonifiable = sc.Blobject() # Create an object that can't be JSON serialized

    print('Testing jsonifying a NON-jsonifiable object:')
    notjson = sc.jsonify(not_jsonifiable, die=False) # Will return a string representation
    sc.sanitizejson(not_jsonifiable, die=True) # Will still not die thanks to jsonpickle

    jsonifiable = sc.objdict().make(keys=['a','b'], vals=pl.rand(10), coerce='none')
    json_obj = sc.jsonify(jsonifiable)
    json_str = sc.jsonify(jsonifiable, tostring=True, indent=2) # kwargs are passed to json.dumps()

    print('Not-a-JSON as sanitized object:')
    print(notjson)
    print('JSON as sanitized object:')
    print(json_obj)
    print('JSON as string:')
    print(json_str)

    # Test JSON load/save
    sc.thisdir() # Just put this here for testing
    jsonfile = 'test.json'
    testdata = {'key1':pl.rand(5,5).tolist(), 'key2':['test1', None]}
    sc.savejson(jsonfile, testdata)
    testdata2 = sc.loadjson(jsonfile)
    assert testdata == testdata2
    os.remove(jsonfile)

    return json_str


def test_jsonpickle():
    sc.heading('Testing JSON read/write functions')

    myobj = sc.prettyobj()
    myobj.a = 3
    myobj.b = pd.DataFrame.from_dict({'a':[3,5,23]})

    jp = sc.jsonpickle(myobj)
    jps = sc.jsonpickle(myobj, tostring=True)
    myobj2 = sc.jsonunpickle(jp)
    myobj3 = sc.jsonunpickle(jps)

    assert myobj.b.equals(myobj2.b)
    assert myobj.b.equals(myobj3.b)

    return jp


def test_load_dump_str():
    sc.heading('Testing load/dump string')
    obj1 = sc.objdict(a=1, b=2)
    string = sc.dumpstr(obj1)
    obj2 = sc.loadstr(string)
    assert obj1 == obj2
    return string


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    spread = test_spreadsheets()
    json   = test_json()
    jp     = test_jsonpickle()
    string = test_load_dump_str()

    sc.toc()
    print('Done.')