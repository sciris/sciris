"""
Version: 2020apr27
"""

import pylab as pl
import sciris as sc


if 'doplot' not in locals(): doplot = False


def test_colors(doplot):
    sc.heading('Testing colors')
    o = sc.objdict()

    print('Testing hex')
    o.c1 = sc.hex2rgb('#fff')
    o.c2 = sc.hex2rgb('fabcd8')
    print(o.c1)
    print(o.c2)

    print('Testing arraycolors')
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
    o.arraycolors = colors

    print('Testing gridcolors')
    colors_a = sc.gridcolors(ncolors=8,  demo=doplot)
    colors_b = sc.gridcolors(ncolors=18, demo=doplot)
    colors_c = sc.gridcolors(ncolors=28, demo=doplot)
    print('\n8 colors:', colors_a)
    print('\n18 colors:', colors_b)
    print('\n28 colors:', colors_c)
    o.gridcolors = colors_a

    return o


def test_colormaps(doplot):
    sc.heading('Testing colormaps')
    o = sc.objdict()
    return o


def test_3d(doplot):
    sc.heading('Testing 3D')
    o = sc.objdict()

    print('Testing surf3d')
    data = pl.randn(50,50)
    smoothdata = sc.smooth(data,20)
    if doplot:
        sc.surf3d(smoothdata)

    print('Testing bar3d')
    data = pl.rand(20,20)
    smoothdata = sc.smooth(data)
    if doplot:
        sc.bar3d(smoothdata)

    return o


def test_other(doplot):
    sc.heading('Testing other')
    o = sc.objdict()
    return o


def test_saving(doplot):
    sc.heading('Testing saving')
    o = sc.objdict()
    return o





#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    doplot = True

    colors    = test_colors(doplot)
    colormaps = test_colormaps(doplot)
    threed    = test_3d(doplot)
    other     = test_other(doplot)
    saved     = test_saving(doplot)

    if doplot:
        pl.show()

    sc.toc()
    print('Done.')
