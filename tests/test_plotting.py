"""
Test color and plotting functions -- warning, opens up many windows!
"""

import os
import numpy as np
import pylab as pl
import sciris as sc


if 'doplot' not in locals():
    doplot = True


def test_colors(doplot=doplot):
    sc.heading('Testing colors')
    o = sc.objdict()

    print('Testing shifthue')
    o.hue = sc.shifthue(colors=[(1,0,0),(0,1,0)], hueshift=0.5)

    print('Testing hex2rgb and rgb2hex')
    hx = '#87bc26'
    o.rgb = sc.hex2rgb(hx)
    o.hx = sc.rgb2hex(o.rgb)
    assert o.hx == hx

    print('Testing rgb2hsv and hsv2rgb')
    rgb = np.array([0.53, 0.74, 0.15])
    o.hsv = sc.rgb2hsv(rgb)
    o.rgb2 = sc.hsv2rgb(o.hsv)
    assert np.all(np.isclose(rgb, o.rgb2))

    return o


def test_colormaps(doplot=doplot):
    sc.heading('Testing colormaps')
    o = sc.objdict()

    print('Testing vectocolor')
    x = np.random.rand(10)
    o.veccolors = sc.vectocolor(x, cmap='turbo')

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
    o.gridcolors = sc.gridcolors(ncolors=8,  demo=doplot)
    sc.gridcolors(ncolors=28, demo=doplot)
    print('\n8 colors:', o.gridcolors)

    print('Testing colormapdemo')
    if doplot:
        sc.colormapdemo('parula', doshow=False)

    return o


def test_3d(doplot=doplot):
    sc.heading('Testing 3D')
    o = sc.objdict()

    print('Testing surf3d')
    if doplot:
        o.fig = sc.fig3d()

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


def test_other(doplot=doplot):
    sc.heading('Testing other')
    o = sc.objdict()

    data = np.random.rand(10)*1e4

    nrows,ncols = sc.get_rows_cols(100, ratio=0.5) # Returns 8,13 since rows are prioritized

    if doplot:
        o.fig = pl.figure()

        pl.subplot(2,1,1)
        pl.plot(data)
        sc.boxoff()
        sc.setxlim()
        sc.setylim()
        sc.commaticks()

        pl.subplot(2,1,2)
        pl.plot(data)
        sc.SIticks()

        try:
            sc.maximize()
        except Exception as E:
            print(f'sc.maximize() failed with {str(E)}:')
            print(sc.traceback())
            print('↑↑↑ Ignoring since sc.maximize() unlikely to work via e.g. automated testing')

    return o


def test_saving(doplot=doplot):
    sc.heading('Testing saving')
    o = sc.objdict()

    filename = 'testfig.fig'
    moviename = 'testmovie.gif'

    if doplot:

        print('Testing save figs')
        o.fig = pl.figure()
        pl.plot(pl.rand(10))

        sc.savefigs(o.fig, filetype='fig', filename=filename)
        sc.loadfig(filename)

        print('Testing save movie')
        frames = [pl.plot(pl.cumsum(pl.randn(100))) for i in range(10)] # Create frames
        sc.savemovie(frames, moviename) # Save movie as medium-quality gif

        os.remove(filename)
        os.remove(moviename)

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
