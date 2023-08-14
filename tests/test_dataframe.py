'''
Test the Sciris dataframe
'''

import numpy as np
import sciris as sc
import pytest

def dfprint(label, val=None):
    sc.colorize('cyan', f'\n{label}')
    if val is not None:
        print(val)
    return None

def subheading(label):
    return sc.printgreen('\n\n' + label)

def test_dataframe():
    sc.heading('Testing dataframe')

    df = sc.dataframe(cols=['x','y'], data=[[1238,2],[384,5],[666,7]]); dfprint('Create dataframe', df)
    dfprint('Print out a column', df['x'])
    dfprint('Print out a row', df[0])
    dfprint('Print out an element', df['x',0])
    df[0,:] = [123,6]; dfprint('Set values for a whole row', df)
    df['y'] = [8,5,0]; dfprint('Set values for a whole column', df)
    df['z'] = [14,14,14]; dfprint('Add new column', df)
    df.addcol('m', [14,15,16]); dfprint('Alternate way to add new column', df)
    df.popcols('z'); dfprint('Remove a column', df)
    df.poprows(1); dfprint('Remove a row', df)
    df.appendrow([555,2,-1]); dfprint('Append a new row', df)
    df = df.concat([[1,2,3],[4,5,6]], [9,9,9]); dfprint('Concatenate', df)
    df.insertrow(0, [660,3,-2]); dfprint('Insert a new row', df)
    df.sortrows(); dfprint('Sort by the first column', df)
    df.sortrows('y'); dfprint('Sort by the second column', df)
    dfprint('Return the row starting with value "555"', df.findrow(555))
    df.poprows(); dfprint('Remove last row', df)
    df.poprows(value=666); dfprint('Remove the row starting with element "666"', df)
    p = df.to_pandas(); dfprint('Convert to pandas', p)
    df2 = df.filtercols(['m','x']); dfprint('Filter to columns m and x', df2)
    
    # Do tests on the final dataframe
    assert df.x.sum() == 1343
    assert df.y.sum() == 20
    assert df.m.sum() == 20
    assert df.shape == (5, 3)
    
    return df


def test_init():
    sc.heading('Testing dataframe initialization')
    
    subheading('Initialization')
    df = sc.dataframe(cols=['a', 'b'], nrows=3)
    assert df.shape == (3,2)
    df += np.random.random(df.shape)
    dfprint('To start', df)
    
    # Merge
    dfm  = sc.dataframe(dict(x=[1,2,3], y=[4,5,6])) 
    dfm2 = sc.dataframe(dict(x=[1,2,3], z=[9,8,7])) 
    dfm.merge(dfm2, on='x', inplace=True) 
    
    # Define with keywords
    sc.dataframe(a=[1,2,3], b=[4,5,6])
    sc.dataframe(dict(a=[1,2,3]), b=[4,5,6])

    subheading('Append row')
    df.append([7,7]) # Alias
    df.appendrow(dict(a=4, b=4)); dfprint('Append row as dict', df)
    with pytest.raises(ValueError):
        df.appendrow([1,2,3])
    
    return df


def test_get_set():
    sc.heading('Testing dataframe get/set')
    
    df = sc.dataframe(cols=['a', 'b'], data=np.random.random((4,2)))

    subheading('Get')
    dfprint('Key get', df['a'])
    dfprint('Array get', df[[0,2]])
    dfprint('Tuple get 1', df[0,'a'])
    dfprint('Tuple get 2', df[0,1])
    dfprint('Slice get 0', df[:2])
    dfprint('Slice get 1', df[0,:])
    dfprint('Slice get 2', df[:,'a'])
    dfprint('Slice get 3', df[:,:])
    with pytest.raises(sc.KeyNotFoundError):
        df['not_a_column']
    with pytest.raises(sc.KeyNotFoundError):
        df[sc.prettyobj({'wrong':'type'})]
    df.get('a') # Test pandas method

    subheading('Set and flexget')
    df['c'] = np.random.randn(df.nrows); dfprint('Set column', df)
    df[2,:] = [17,15,13]; dfprint('Insert row', df)
    df[0,'a'] = 300; dfprint('Tuple set 1', df)
    df[1,1]   = 400; dfprint('Tuple set 2', df)
    out = df.flexget(cols=['a','c'], rows=[0,2]); dfprint('Flexget', out)
    out2 = df.flexget('c', 2)
    assert isinstance(out2, float)
    
    return df


def test_other():
    sc.heading('Testing other dataframe methods')
    
    df = sc.dataframe(dict(
        a = [0.3, 0.5, 0.66, 0.33, 300, 17], 
        b = [0.6, 400, 0.66, 0.33, 0.3, 15], 
        c = [0.7, 0.4, 0.66, 0.33, 0.5, 13])
    )
    
    subheading('Other')
    df.poprows([1,3]); dfprint('Removing rows', df)
    df.replacecol('a', 300, 333); dfprint('Replacing 300→333', df)
    od = df.to_odict(); dfprint('To dict', od)
    df.appendrow(np.random.rand(2,3)); dfprint('Adding more rows', df)
    df2 = df.filterin([0,1]);     dfprint('Filtering in 0-1', df2)
    df3 = df.filterout([0,1]);    dfprint('Filtering out 0-1', df3)
    df4 = df.filterin(value=17);  dfprint('Filtering in 17', df4)
    df5 = df.filterout(value=17); dfprint('Filtering out 17', df5)
    assert df4[0,0] == 17
    assert df5[0,0] != 17
    df.insert(0, 'f', np.random.rand(df.nrows)); dfprint('Inserting column', df)
    df.sortcols(reverse=True); dfprint('Sorting columns', df)
    df.sort()
    assert df.cols[-1] == 'a'
    df.poprow(); dfprint('Removing one row', df)
    
    dfnew = sc.dataframe(cols=['x','y'], data=[['a',2],['b',5],['c',7]])
    
    print('df.col_index()')
    assert dfnew.col_index('y') == dfnew.col_index(1)
    
    print('df.col_name()')
    dfx = sc.dataframe(dict(a=[1,2,3], b=[4,5,6], c=[7,8,9])) 
    assert dfx.col_name(1)    == 'b' 
    assert dfx.col_name('b')  == 'b' 
    assert dfx.col_name(0, 2) == ['a', 'c'] 
    
    print('df.set()')
    dfnew.set('x', ['d','e','f'])
    assert dfnew.x[2] == 'f'
    
    print('Equality')
    df1 = sc.dataframe(a=[1, 2, np.nan])
    df2 = sc.dataframe(a=[1, 2, 4])
    df3 = sc.dataframe(a=[1, 2, 4], b=['a', 'b','c'])
    df4 = sc.dataframe(a=[1, 'foo', np.nan])
    
    assert sc.dataframe.equal(df1, df1) # Returns True
    assert not sc.dataframe.equal(df1, df1, equal_nan=False) # Returns False
    assert not sc.dataframe.equal(df1, df2) # Returns False
    assert not sc.dataframe.equal(df1, df1, df2) # Also returns False
    assert sc.dataframe.equal(df4, df4, df4) # Returns True
    assert df4.equals(df4)
    assert not df4.equals(df4, equal_nan=False)
    
    subheading('Printing')
    dfprint('Custom display')
    df2 = sc.dataframe(data=np.random.rand(100,10))
    df2.disp(precision=2, ncols=5, nrows=5, colheader_justify='left')

    return df


def test_io():
    sc.heading('Testing dataframe I/O')
    
    np.random.seed(1)
    
    f = sc.objdict(csv='my-df.csv', excel='my-df.xlsx')
    df = sc.dataframe(
        a = np.round(np.random.rand(3), decimals=6),  # Handle floats, but not to the limits of precision
        b = [1,2,3], 
        c = ['x','y','z'],
    )
    
    df.to_csv(f.csv, index=False)
    df.to_excel(f.excel, index=False)
    
    df2 = sc.dataframe.read_csv(f.csv)
    df3 = sc.dataframe.read_excel(f.excel)
    
    assert sc.dataframe.equal(df, df2, df3)

    sc.rmpath(f.values())
    
    return df


def test_errors():
    sc.heading('Testing dataframe error handling')
    
    df = sc.dataframe(cols=['x','y'], data=[['a',2],['b',5],['c',7]])
    
    print('Duplicate dtype definitions ✓')
    data = [
        ['a',1],
        ['b',2]
    ]
    columns = {'str':str,'int':int}
    dtypes = [str, int]
    dd = sc.dataframe(data=data, columns=columns) # dtypes are ok
    sc.dataframe(data=data, columns=columns.keys(), dtypes=dtypes) # dtypes are ok
    sc.dataframe(data=data, columns=columns.keys()) # dtypes are ok
    dd.set_dtypes(dtypes)
    with pytest.raises(ValueError):
        sc.dataframe(data=data, columns=columns, dtypes=dtypes) # duplicate dtypes
        
    print('Incompatible columns ✓')
    with pytest.warns(RuntimeWarning):
        data = {'str':['a','b'], 'int':[1,2]}
        columns = ['wrong', 'name']
        sc.dataframe(data=data, columns=columns)
    
    print('Incompatible data ✓')
    with pytest.raises(TypeError):
        data = [['a','b'], [1,2]]
        columns = ['wrong', 'type']
        sc.dataframe(data=data, columns=columns, error=['c',3])
        
    print('Invalid key ✓')
    with pytest.raises(TypeError):
        df[dict(not_a='key')] = 4
        
    return df
    


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    a = test_dataframe()
    b = test_init()
    c = test_get_set()
    d = test_io()
    e = test_other()
    f = test_errors()

    print('\n\n')
    sc.toc()
    print('Done.')