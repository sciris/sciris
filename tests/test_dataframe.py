"""
Version: 2020apr27
"""

import sciris as sc

def test_dataframe():

    def dfprint(label, val):
        sc.colorize('cyan', f'\n{label}')
        print(val)
        return None

    print('Testing dataframe:')
    a = sc.dataframe(cols=['x','y'],data=[[1238,2],[384,5],[666,7]]); dfprint('Create dataframe', a)
    dfprint('Print out a column', a['x'])
    dfprint('Print out a row', a[0])
    dfprint('Print out an element', a['x',0])
    a[0] = [123,6]; dfprint('Set values for a whole row', a)
    a['y'] = [8,5,0]; dfprint('Set values for a whole column', a)
    a['z'] = [14,14,14]; dfprint('Add new column', a)
    a.addcol('m', [14,15,16]); dfprint('Alternate way to add new column', a)
    a.rmcol('z'); dfprint('Remove a column', a)
    a.pop(1); dfprint('Remove a row', a)
    a.append([555,2,-1]); dfprint('Append a new row', a)
    a.insert(1,[660,3,-1]); dfprint('Insert a new row', a)
    a.sort(); dfprint('Sort by the first column', a)
    a.sort('y'); dfprint('Sort by the second column', a)
    a.addrow([770,4,-1]); dfprint('Replace the previous row and sort', a)
    dfprint('Return the row starting with value "555"', a.findrow(555))
    a.rmrow(); dfprint('Remove last row', a)
    a.rmrow(123); dfprint('Remove the row starting with element "123"', a)
    p = a.pandas(); dfprint('Convert to pandas', p)
    q = p.add(p); dfprint('Do a pandas operation', q)
    a.pandas(q); dfprint('Convert back', a)

    a.filtercols(['m','x']); dfprint('Filter to columns m and x', a)
    b = sc.dcp(a); dfprint('Dataframe copying:', a==b)
    return a


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    df = test_dataframe()

    sc.toc()
    print('Done.')