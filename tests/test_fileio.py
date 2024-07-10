'''
Test Sciris file I/O functions.
'''

import os
import numpy as np
import pylab as pl
import openpyxl
import sciris as sc
import pytest
ut = sc.importbypath(sc.thispath() / 'sc_test_utils.py')

# Define filenames
files = sc.prettyobj()
filedir = sc.thispath() / 'files'
files.excel  = filedir / 'test.xlsx'
files.binary = filedir / 'test.obj'
files.text   = filedir / 'text.txt'
files.zip    = filedir / 'test.zip'
tidyup = True

# Define the test data
nrows = 15
ncols = 3
testdata   = np.zeros((nrows+1, ncols), dtype=object) # Includes header row
testdata[0,:] = ['A', 'B', 'C'] # Create header
testdata[1:,:] = np.random.rand(nrows,ncols) # Create data


def test_spreadsheets():
    '''
    Preserved for completeness, but fairly fragile since relies on not-well-trodden
    Excel libraries.
    '''

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

    excel_path = filedir / 'exampledata.xlsx'
    if os.path.exists(excel_path):
        sc.heading('Reading cells')
        wb = sc.Spreadsheet(filename=excel_path) # Load a sample databook to try pulling cells from
        kw = dict(sheetname='Baseline year population inputs', cells=[[46, 2], [47, 2]])
        celltest_opyxl = wb.readcells(method='openpyxl', **kw, wbargs={'data_only': True}) # Grab cells using openpyxl.  You have to set wbargs={'data_only': True} to pull out cached values instead of formula strings
        celltest_pd    = wb.readcells(method='pandas',   **kw)  # Grab cells using pandas
        print(f'openpyxl output: {celltest_opyxl}')
        print(f'pandas output: {celltest_pd}')
        assert celltest_opyxl == celltest_pd
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
    wb = sc.Spreadsheet().new()
    S = sc.Spreadsheet(files.excel)
    S.writecells(cells=['A6','B7','C8','D9'], vals=['This','is','a','test']) # Method 1
    S.writecells(cells=[pl.array([7,1])+i for i in range(4)], vals=['And','so','is','this']) # Method 2
    newdata = (np.random.rand(3,3)*100).round()
    S.writecells(startrow=14, startcol=1, vals=newdata, verbose=True) # Method 3
    S.save()
    data = S.readcells(header=False)
    print(S)
    sc.pp(data)

    if tidyup:
        sc.rmpath(files.excel)
        sc.rmpath('spreadsheet.xlsx')

    return S


def test_load_save():
    '''
    Test file loading and saving
    '''
    o = sc.objdict()
    
    sc.heading('Save/load')
    sc.save(files.binary, testdata)
    o.obj1 = sc.load(files.binary)
    print(o.obj1)
    
    sc.heading('Testing other load/save options')
    sc.save(files.binary, testdata, compresslevel=9)
    sc.save(files.binary, testdata, compression='none')
    sc.zsave(files.binary, testdata)
    zobj = sc.load(files.binary, verbose=True)
    assert np.all(o.obj1 == zobj)
    
    sc.heading('Savetext/loadtext')
    sc.savetext(files.text, testdata)
    sc.savetext(files.text, testdata.tolist())
    o.obj2 = sc.loadtext(files.text)
    sc.loadtext(files.text, splitlines=True)
    print(o.obj2)
    
    sc.heading('Save/load zip')
    sc.savezip(files.zip, data={'fake_datafile.obj':testdata})
    sc.savezip(files.zip, [files.text, files.binary], basename=True)
    o.zip = sc.loadzip(files.zip) # Load into memory
    sc.unzip(files.zip, outfolder=filedir) # Unpack onto disk
    
    # Tidy up
    if tidyup:
        sc.blank()
        sc.heading('Tidying up')
        sc.rmpath([files.binary, files.text, files.zip], die=False)
    
    return o


def test_load_corrupted():
    '''
    Test that loading a corrupted object still works -- specifically, one with
    a no-longer-existant class. See the ``files`` folder for the scripts used
    to create this file.
    '''
    sc.heading('Intentionally loading corrupted file')
    
    o = sc.objdict()

    class LiveClass():
        def __init__(self, x):
            self.x = x
        def disp(self):
            return f'x is {self.x}'

    dead_path = filedir / 'deadclass.obj'
    
    print('Loading with no remapping...')
    with pytest.warns(sc.UnpicklingWarning):
        o.obj1 = sc.load(dead_path)
    print(o.obj1)
    print(f'Loading corrupted object succeeded, x={o.obj1.x}')

    print('Loading with remapping...')
    o.obj2 = sc.load(dead_path, remapping={'deadclass.DeadClass': LiveClass})
    o.obj3 = sc.load(dead_path, remapping={('deadclass', 'DeadClass'): LiveClass}, verbose=True)
    for obj in [o.obj2, o.obj3]:
        assert isinstance(obj, LiveClass)
    print(f'Loading remapped object succeeded, {o.obj2.disp()}, object: {o.obj2}')
    
    print('Loading with error on initialization')
    class DyingClass():
        def __new__(self):
            raise Exception('Intentional exception')
    with pytest.warns(sc.UnpicklingWarning):
        o.obj4 = sc.load(dead_path, remapping={'deadclass.DeadClass': DyingClass}, method='robust', verbose=True)
    with pytest.raises(sc.UnpicklingError):
        sc.load(dead_path, die=True)
        
    print('Testing Failed directly')
    f = sc.Failed()
    len(f)
    f['foo'] = 3
    assert f['foo'] == 3
    f()
    f.disp()
    f.showfailure()
    
    return o


def test_fileio():
    '''
    Test other file I/O functions
    '''
    o = sc.objdict()

    # Test thisdir
    sc.heading('Testing thisdir')
    a = sc.thisdir()
    b = sc.thispath()
    assert a == str(b)
    assert 'sciris' in a
    assert 'tests' in a
    sc.thisdir(sc) # Test with a module
    o.thisdir = a

    sc.heading('Get files')
    print(f'Files in "{filedir}":')
    TF = [True,False]
    for tf in TF:
        sc.pp(sc.getfilepaths(folder=filedir, abspath=tf, filesonly=tf, foldersonly=not(tf), nopath=tf, aspath=tf))
    o.filelist = sc.getfilelist(fnmatch='*.py', nopath=True) # Test alias
    assert all(['py' in f for f in o.filelist])
    assert 'test_fileio.py' in o.filelist
    
    
    sc.heading('Sanitizing filenames')
    bad = 'Nöt*a   file&name?!.doc'
    good = sc.sanitizefilename(bad)
    assert str(sc.sanitizepath(bad)) == good
    sc.sanitizefilename(bad, strict=True, aspath=None)
    o.sanitized = good
    
    sc.heading('Testing other')
    path1 = sc.path('/a/folder', 'a_file.txt')
    path2 = sc.path(['/a', 'folder'], 'a_file.txt')
    path3 = sc.path('/a/folder', None, 'a_file.txt')
    assert str(path1) == str(path2) == str(path3) == os.sep.join(['', 'a', 'folder', 'a_file.txt'])
    assert sc.ispath(path1)
    o.thisfile = sc.thisfile(aspath=True)

    return o


def test_json():
    sc.heading('Testing JSON read/write functions')

    not_jsonifiable = sc.Blobject() # Create an object that can't be JSON serialized

    print('Testing jsonifying a NON-jsonifiable object')
    notjson = sc.jsonify(not_jsonifiable, die=False) # Will return a string representation
    sc.sanitizejson(not_jsonifiable, die=True) # Will still not die thanks to jsonpickle

    jsonifiable = sc.objdict().make(keys=['a','b'], vals=np.random.rand(10), coerce='none')
    json_obj = sc.jsonify(jsonifiable)
    json_str = sc.jsonify(jsonifiable, tostring=True, indent=2) # kwargs are passed to json.dumps()
    json = sc.readjson(json_str)
    assert json_obj == json

    print('Not-a-JSON as sanitized object:')
    print(notjson)
    print('JSON as sanitized object:')
    print(json_obj)
    print('JSON as string:')
    print(json_str)
    
    # jsonify() docstring tests
    data = dict(a=dict(b=np.arange(4)), c=dict(foo='cat', bar='dog'))
    json = sc.jsonify(data)
    jsonstr = sc.jsonify(data, tostring=True, indent=2)
    assert json['a']['b'][-1] == 3
    assert '"foo": "cat"' in jsonstr
    
    # Use a custom function for parsing the data
    custom = {np.ndarray: lambda x: f'It was an array: {x}'}
    j2 = sc.jsonify(data, custom=custom, tostring=True)
    assert 'It was an array' in j2

    # Test JSON load/save
    print('Testing JSON load/save...')
    jsonfile = 'test.json'
    testdata = {'key1':np.random.rand(5,5).tolist(), 'key2':['test1', None]}
    sc.savejson(jsonfile, testdata)
    testdata2 = sc.loadjson(jsonfile)
    assert testdata == testdata2
    
    # Test YAML load/save
    print('Testing YAML load/save...')
    yamlfile = 'test.yaml'
    sc.saveyaml(yamlfile, testdata)
    testdata2 = sc.loadyaml(yamlfile)
    testdata3 = sc.readyaml(sc.loadtext(yamlfile))
    assert testdata == testdata2 == testdata3
    
    # Tidy up
    sc.rmpath(jsonfile, yamlfile)

    return json_str


def test_jsonpickle():
    sc.heading('Testing jsonpickle functions')

    myobj = sc.prettyobj()
    myobj.a = 3
    myobj.b = ut.MyClass(nan=False, mixed=False, pandas=False) # jsonpickle can't handle mixed data types

    jp = sc.jsonpickle(myobj)
    jps = sc.jsonpickle(myobj, tostring=True)
    myobj2 = sc.jsonunpickle(jp)
    myobj3 = sc.jsonunpickle(jps)
    
    jpath = 'my-data.json'
    sc.jsonpickle(myobj, jpath)
    myobj4 = sc.jsonunpickle(jpath)
    
    # Tidy up
    sc.rmpath(jpath)
    assert sc.equal(myobj, myobj2, myobj3, myobj4, leaf=True)

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

    spread  = test_spreadsheets()
    ls      = test_load_save()
    corrupt = test_load_corrupted()
    fileio  = test_fileio()
    json    = test_json()
    jp      = test_jsonpickle()
    string  = test_load_dump_str()

    sc.toc()
    print('Done.')