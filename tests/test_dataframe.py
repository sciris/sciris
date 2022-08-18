'''
Test the Sciris dataframe
'''

import numpy as np
import sciris as sc
import pytest

def dfprint(label, val):
    sc.colorize('cyan', f'\n{label}')
    print(val)
    return None

def test_dataframe():
    sc.heading('Testing dataframe')


    a = sc.dataframe(cols=['x','y'], data=[[1238,2],[384,5],[666,7]]); dfprint('Create dataframe', a)
    dfprint('Print out a column', a['x'])
    dfprint('Print out a row', a[0])
    dfprint('Print out an element', a['x',0])
    a[0] = [123,6]; dfprint('Set values for a whole row', a)
    a['y'] = [8,5,0]; dfprint('Set values for a whole column', a)
    a['z'] = [14,14,14]; dfprint('Add new column', a)
    a.addcol('m', [14,15,16]); dfprint('Alternate way to add new column', a)
    a.rmcol('z'); dfprint('Remove a column', a)
    a.poprow(1); dfprint('Remove a row', a)
    a.appendrow([555,2,-1]); dfprint('Append a new row', a)
    a.concat([[1,2,3],[4,5,6]], [9,9,9]); dfprint('Concatenate', a)
    a.insertrow(0, [660,3,-2]); dfprint('Insert a new row', a)
    a.sortrows(); dfprint('Sort by the first column', a)
    a.sortrows('y'); dfprint('Sort by the second column', a)
    dfprint('Return the row starting with value "555"', a.findrow(555))
    a.rmrow(); dfprint('Remove last row', a)
    a.rmrow(value=666); dfprint('Remove the row starting with element "666"', a)
    p = a.to_pandas(); dfprint('Convert to pandas', p)
    b = a.filtercols(['m','x']); dfprint('Filter to columns m and x', b)
    return a


def test_methods():
    sc.heading('Testing dataframe methods')
    df = sc.dataframe(cols=['a', 'b'], nrows=3)
    assert df.shape == (3,2)
    df += np.random.random(df.shape)
    dfprint('To start', df)

    # Append row
    df.appendrow(dict(a=4, b=4)); dfprint('Append row as dict', df)
    with pytest.raises(IndexError):
        df.appendrow([1,2,3])

    # Get
    dfprint('Key get', df['a'])
    dfprint('Array get', df[[0,2]])
    dfprint('Tuple get 1', df[0,'a'])
    dfprint('Tuple get 2', df[0,1])
    dfprint('Slice get 1', df[0,:])
    dfprint('Slice get 2', df[:,'a'])
    dfprint('Slice get 3', df[:,:])
    with pytest.raises(sc.KeyNotFoundError):
        df['not_a_column']
    with pytest.raises(sc.KeyNotFoundError):
        df[sc.prettyobj({'wrong':'type'})]

    # Set
    df['c'] = np.random.randn(df.nrows); dfprint('Set column', df)
    df[2] = [17,15,13]; dfprint('Insert row', df)
    df[0,'a'] = 300; dfprint('Tuple set 1', df)
    df[1,1]   = 400; dfprint('Tuple set 2', df)
    out = df.flexget(cols=['a','c'], rows=[0,2]); dfprint('Flexget', out)

    # Other
    df.rmrows([1,3]); dfprint('Removing rows', df)
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
    assert df.cols[-1] == 'a'

    return df



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    a  = test_dataframe()
    df = test_methods()

    sc.toc()
    print('Done.')