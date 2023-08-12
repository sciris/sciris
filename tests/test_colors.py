"""
Test color and plotting functions -- warning, opens up many windows!
"""

import numpy as np
import pylab as pl
import sciris as sc
import pytest


#%% Functions

if 'doplot' not in locals():
    doplot = False
    sc.options(interactive=doplot)


def test_colors():
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
    
    print('Testing sanitizecolors')
    o.green1 = sc.sanitizecolor('g')
    o.green2 = sc.sanitizecolor('tab:green')
    o.crimson1 = sc.sanitizecolor('crimson')
    o.crimson2 = sc.sanitizecolor((220, 20, 60))
    assert o.crimson1 == o.crimson2
    o.midgrey = sc.sanitizecolor(0.5)
    with pytest.raises(ValueError):
        sc.sanitizecolor('not-a-color')

    return o


def test_colormaps():
    sc.heading('Testing colormaps')
    o = sc.objdict()

    print('Testing vectocolor')
    nanpos = 5
    nancolor = 'sienna'
    x = np.random.rand(10)
    x = sc.normalize(x) # To ensure the values span 0-1
    x[nanpos] = np.nan
    o.veccolors = sc.vectocolor(x, nancolor=nancolor, midpoint=0.3, cmap='turbo')
    assert (o.veccolors[nanpos,:] == sc.sanitizecolor(nancolor, asarray=True, alpha=1)).all()

    print('Testing arraycolors')
    n = 1000
    ncols = 5
    arr = pl.rand(n,ncols)
    for c in range(ncols):
        arr[:,c] += c
    x = pl.rand(n)
    y = pl.rand(n)
    colors = sc.arraycolors(arr)
    pl.figure('Array colors', figsize=(20,16))
    for c in range(ncols):
        pl.scatter(x+c, y, s=50, c=colors[:,c])
    o.arraycolors = colors

    print('Testing gridcolors')
    o.gridcolors = sc.gridcolors(ncolors=8,  demo=True)
    sc.gridcolors(ncolors=28, demo=True)
    print('\n8 colors:', o.gridcolors)

    print('Testing colormapdemo')
    sc.colormapdemo('parula', doshow=False)

    return o


def test_colorbars():
    sc.heading('Testing colorbars')
    o = sc.objdict()
    
    print('Create a default colorbar')
    o.cb1 = sc.manualcolorbar()
    
    print('Add a colorbar to non-mappable data (e.g. a scatterplot)')
    pl.figure()
    n = 1000
    x = pl.randn(n)
    y = pl.randn(n)
    c = x**2 + y**2
    pl.scatter(x, y, c=c)
    o.cb2 = sc.manualcolorbar(c)
    
    print('Create a custom colorbar with a custom label')
    pl.figure()
    sc.manualcolorbar(
        vmin=-20,
        vmax=40,
        vcenter=0,
        cmap='orangeblue',
        label='Cold/hot',
        orientation='horizontal',
        labelkwargs=dict(rotation=10, fontweight='bold'),
        axkwargs=[0.1,0.5,0.8,0.1],
    )
    
    print('Create a completely custom colorbar')
    pl.figure()
    n = 12
    x = np.arange(n)
    values = np.sqrt(np.arange(n))
    colors = sc.gridcolors(n)
    pl.scatter(x, values, c=colors)
    pl.grid(True)

    ticklabels = ['' for i in range(n)]
    for i in [0, 2, 4, 10, 11]:
        ticklabels[i] = f'Color {i} is nice'
    o.cb3 = sc.manualcolorbar(
        colors=colors, 
        values=values, 
        ticks=values, 
        ticklabels=ticklabels, 
        spacing='proportional'
    )
    
    return o
    

#%% Run as a script
if __name__ == '__main__':
    T = sc.timer()

    doplot = True
    sc.options(interactive=True)

    c  = test_colors()
    cm = test_colormaps()
    cb = test_colorbars()

    if doplot:
        pl.show()
    else:
        pl.close('all')

    T.toc()
    print('Done.')
