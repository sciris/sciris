"""
Test color and plotting functions -- warning, opens up many windows!
"""

import os
import numpy as np
import pylab as pl
import sciris as sc
import pytest

#%% Functions

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
        pl.figure('Array colors', figsize=(20,16))
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
        o.fig = sc.fig3d(num='Blank 3D')

    print('Testing surf3d')
    data = pl.randn(50,50)
    smoothdata = sc.smooth(data,20)
    if doplot:
        sc.surf3d(smoothdata, figkwargs=dict(num='surf3d'))

    print('Testing bar3d')
    data = pl.rand(20,20)
    smoothdata = sc.smooth(data)
    if doplot:
        sc.bar3d(smoothdata, figkwargs=dict(num='bar3d'))

    return o


def test_other(doplot=doplot):
    sc.heading('Testing other')
    o = sc.objdict()

    data = np.random.rand(10)*1e4

    nrows,ncols = sc.get_rows_cols(100, ratio=0.5) # Returns 8,13 since rows are prioritized

    if doplot:
        sc.emptyfig()
        o.fig = pl.figure('Limits')

        pl.subplot(2,1,1)
        pl.plot(data)
        sc.boxoff()
        sc.setxlim()
        sc.setylim()
        sc.commaticks()

        pl.subplot(2,1,2)
        pl.plot(data)
        sc.SIticks()
        pl.title('SI ticks')

        try:
            sc.maximize()
        except Exception as E:
            print(f'sc.maximize() failed with {str(E)}:')
            print(sc.traceback())
            print('↑↑↑ Ignoring since sc.maximize() unlikely to work via e.g. automated testing')

        # Need keep=True to avoid refresh which crashes when run via pytest-parallel
        sc.figlayout(keep=True)

        # Test legends
        pl.figure('Legends')
        pl.plot([1,4,3], label='A')
        pl.plot([5,7,8], label='B')
        pl.plot([2,5,2], label='C')
        sc.orderlegend(reverse=True) # Legend order C, B, A
        sc.orderlegend([1,0,2], frameon=False) # Legend order B, A, C with no frame
        sc.separatelegend()

    return o


def test_saving(doplot=doplot):
    sc.heading('Testing saving')
    o = sc.objdict()

    fn = sc.objdict()
    fn.fig   = 'testfig.fig'
    fn.movie = 'testmovie.gif'
    fn.anim  = 'testanim.gif' # mp4 only available if ffmpeg is installed

    if doplot:

        print('Testing save figs')
        o.fig = pl.figure('Save figs')
        pl.plot(pl.rand(10))

        sc.savefigs(o.fig, filetype='fig', filename=fn.fig)
        sc.loadfig(fn.fig)

        print('Testing save movie')
        frames = [pl.plot(pl.cumsum(pl.randn(100))) for i in range(3)] # Create frames
        sc.savemovie(frames, fn.movie) # Save movie as medium-quality gif

        print('Testing animation')
        anim = sc.animation(filename=fn.anim)
        pl.figure('Animation')
        for i in range(3):
            pl.plot(pl.cumsum(pl.randn(100)))
            anim.addframe()
        anim.save()

        print('Tidying...')
        for f in fn.values():
            os.remove(f)

    return o


def test_dates(doplot=doplot):
    sc.heading('Testing dates')
    o = sc.objdict()

    if doplot:
        x = np.array(sc.daterange('2020-12-24', '2021-01-15', asdate=True))
        y = sc.smooth(pl.rand(len(x)))
        o.fig = pl.figure('Date formatters', figsize=(6,6))

        for i,style in enumerate(['auto', 'concise', 'sciris']):
            pl.subplot(3,1,i+1)
            pl.plot(x, y)
            pl.title('Date formatter: ' + style.title())
            sc.dateformatter(style=style, rotation=1)

        pl.figure('Datenum formatter')
        pl.plot(np.arange(500), pl.randn(500))
        sc.datenumformatter(start_date='2021-01-01')

    return o


def test_fonts(doplot=doplot):
    sc.heading('Testing font functions')

    # Test getting fonts
    fonts = sc.fonts()

    # Test setting fonts
    orig = pl.rcParams['font.family']
    sc.fonts(add=sc.path('files/examplefont.ttf'), use=True, die=True, verbose=True)

    if doplot:
        pl.figure('Fonts')
        pl.plot([1,2,3], [4,5,6])
        pl.xlabel('Example label in new font')

    # Reset
    pl.rcParams['font.family'] = orig

    return fonts


def test_saveload(doplot=doplot):
    sc.heading('Testing figure save/load')

    fig = pl.figure('Save/load')
    pl.plot([1,3,7])
    fn = sc.objdict()
    fn.png1 = 'example1.png'
    fn.png2 = 'example2.png'
    fn.jpg  = 'example.jpg'

    # Basic usage
    sc.savefig(fn.png1)
    md1 = sc.loadmetadata(fn.png1)
    sc.pp(md1)

    # Complex usage
    comments = 'My figure'
    sc.savefig(fn.png2, fig=fig, comments=comments, freeze=True)
    md2 = sc.loadmetadata(fn.png2)
    assert md2['modules']['numpy'] == np.__version__ # Check version information was stored correctly
    assert md2['comments'] == comments

    # Should print a warning
    with sc.capture() as txt:
        sc.savefig(fn.jpg, die=False)
    assert 'Warning' in txt

    with pytest.raises(ValueError):
        sc.savefig(fn.jpg)

    with pytest.raises(ValueError):
        sc.loadmetadata(fn.jpg)

    # Tidy up
    pl.close(fig)
    for f in fn.values():
        try:
            os.remove(f)
            print(f'Removed temporary file {f}')
        except Exception as E:
            print(f'Could not remove {f}: {E}')

    return md2


#%% Run as a script
if __name__ == '__main__':
    T = sc.timer()

    doplot = True

    colors    = test_colors(doplot)
    colormaps = test_colormaps(doplot)
    threed    = test_3d(doplot)
    other     = test_other(doplot)
    saved     = test_saving(doplot)
    dates     = test_dates(doplot)
    fonts     = test_fonts(doplot)
    metadata  = test_saveload(doplot)

    if doplot:
        pl.show()

    T.toc()
    print('Done.')
