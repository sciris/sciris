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

    # Get/set
    dfprint('Key get', df['a'])
    dfprint('Array get', df[[0,2]])
    dfprint('Tuple get 1', df[0,'a'])
    dfprint('Tuple get 2', df[0,1])
    dfprint('Slice get 1', df[0,:])
    dfprint('Slice get 2', df[:,'a'])
    dfprint('Slice get 3', df[:,:])

    return df



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    df = test_dataframe()
    df2 = test_methods()

    sc.toc()
    print('Done.')