'''
Test the Sciris dataframe
'''

import sciris as sc

def test_dataframe():
    sc.heading('Testing dataframe')

    def dfprint(label, val):
        sc.colorize('cyan', f'\n{label}')
        print(val)
        return None


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
    df = sc.dataframe(cols=['a', 'b', 'c'], nrows=8)
    assert df.shape == (8,3)
    return df



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    df = test_dataframe()
    df2 = test_methods()

    sc.toc()
    print('Done.')