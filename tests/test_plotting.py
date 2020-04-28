"""
Version: 2020apr27
"""

import pylab as pl
import sciris as sc


if 'doplot' not in locals(): doplot = True

def test_plotting():

    torun = [
    'hex2rgb',
    'arraycolors',
    'gridcolors',
    'surf3d',
    'bar3d',
    ]

    if 'hex2rgb' in torun:
        c1 = sc.hex2rgb('#fff')
        c2 = sc.hex2rgb('fabcd8')
        print(c1)
        print(c2)

    if 'arraycolors' in torun:
        n = 1000
        ncols = 5
        arr = pl.rand(n,ncols)
        for c in range(ncols):
            arr[:,c] += c
        x = pl.rand(n)
        y = pl.rand(n)
        colors = sc.arraycolors(arr)
        if doplot:
            pl.figure(figsize=(20,16))
            for c in range(ncols):
                pl.scatter(x+c, y, s=50, c=colors[:,c])

    if 'gridcolors' in torun:
        colors_a = sc.gridcolors(ncolors=8,  demo=doplot)
        colors_b = sc.gridcolors(ncolors=18, demo=doplot)
        colors_c = sc.gridcolors(ncolors=28, demo=doplot)
        print('\n8 colors:', colors_a)
        print('\n18 colors:', colors_b)
        print('\n28 colors:', colors_c)

    if 'surf3d' in torun:
        data = pl.randn(50,50)
        smoothdata = sc.smooth(data,20)
        if doplot:
            sc.surf3d(smoothdata)

    if 'bar3d' in torun:
        data = pl.rand(20,20)
        smoothdata = sc.smooth(data)
        if doplot:
            sc.bar3d(smoothdata)

    return pl.gcf()


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    fig = test_plotting()

    sc.toc()
    print('Done.')
