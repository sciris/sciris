"""
Test color and plotting functions -- warning, opens up many windows!
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sciris as sc
import pytest

#%% Functions

if 'doplot' not in locals():
    doplot = False
    sc.options(interactive=doplot)


def test_3d(doplot=doplot):
    sc.heading('Testing 3D')
    o = sc.objdict()

    print('Testing fig3d')
    o.fig = sc.fig3d(num='Blank 3D')

    print('Testing surf3d')
    data = np.random.randn(50,50)
    smoothdata = sc.smooth(data,20)
    sc.surf3d(smoothdata, figkwargs=dict(num='surf3d'))

    print('Testing bar3d')
    data = np.random.rand(20,20)
    smoothdata = sc.smooth(data)
    sc.bar3d(smoothdata, figkwargs=dict(num='bar3d'))

    print('Testing scatter3d')
    sc.scatter3d(smoothdata, figkwargs=dict(num='scatter3d'))

    if not doplot:
        plt.close('all')

    return o


def test_stackedbar(doplot=doplot):
    sc.heading('Testing stacked bar')

    plt.figure(num='Stacked bar')

    bmt = ['bottom','middle','top']
    lmr = ['left','middle','right']

    plt.subplot(1,2,1)
    values = np.random.rand(3,4)
    artists = sc.stackedbar(values, labels=bmt)
    plt.legend()

    plt.subplot(1,2,2)
    artists = sc.stackedbar(np.cumsum(values, axis=0).T, labels=lmr, is_cum=True, transpose=True, barh=True)
    plt.legend()

    if not doplot:
        plt.close('all')

    return artists


def test_other(doplot=doplot):
    sc.heading('Testing other')
    o = sc.objdict()

    data = np.random.rand(10)*1e4

    nrows,ncols = sc.get_rows_cols(100, ratio=0.5) # Returns 8,13 since rows are prioritized
    nrows2,ncols2 = sc.getrowscols(1)
    assert (nrows,ncols) == (8,13)
    assert (nrows2,ncols2) == (1,1) # Check that it works with a single row

    sc.emptyfig()
    o.fig = plt.figure('Limits')

    plt.subplot(2,1,1)
    plt.plot(data)
    sc.boxoff()
    sc.setxlim()
    sc.setylim()
    sc.commaticks()

    plt.subplot(2,1,2)
    plt.plot(data)
    sc.SIticks()
    plt.title('SI ticks')

    try:
        sc.maximize()
    except Exception as E:
        print(f'sc.maximize() failed with {str(E)}:')
        print(sc.traceback())
        print('↑↑↑ Ignoring since sc.maximize() unlikely to work via e.g. automated testing')

    # Need keep=True to avoid refresh which crashes when run via pytest-parallel
    sc.figlayout(keep=True)

    # Test legends
    plt.figure('Legends')
    plt.plot([1,4,3], label='A')
    plt.plot([5,7,8], label='B')
    plt.plot([2,5,2], label='C')
    sc.orderlegend(reverse=True) # Legend order C, B, A
    sc.orderlegend([1,0,2], frameon=False) # Legend order B, A, C with no frame
    sc.separatelegend()

    # Test styles
    style = 'sciris.fancy' if sc.compareversions(mpl, '>=3.7') else sc.style_fancy # Matplotlib regression support
    with plt.style.context(style):
        plt.figure('Custom style')
        plt.plot(np.random.rand(10))

    if not doplot:
        plt.close('all')

    return o


def test_saving(doplot=doplot):
    sc.heading('Testing saving')
    o = sc.objdict()

    fn = sc.objdict()
    fn.fig   = 'testfig.fig'
    fn.movie = 'testmovie.gif'
    fn.anim  = 'testanim.gif' # mp4 only available if ffmpeg is installed

    print('Testing save figs')
    o.fig = plt.figure('Save figs')
    plt.plot(np.random.rand(10))

    sc.savefigs(o.fig, filetype='fig', filename=fn.fig)
    sc.loadfig(fn.fig)

    print('Testing save movie')
    frames = [plt.plot(np.cumsum(np.random.randn(100))) for i in range(2)] # Create frames
    sc.savemovie(frames, fn.movie) # Save movie as medium-quality gif

    print('Testing animation')
    anim = sc.animation(filename=fn.anim)
    plt.figure('Animation')
    for i in range(2):
        plt.plot(np.cumsum(np.random.randn(100)))
        anim.addframe()
    anim.save()

    print('Tidying...')
    for f in fn.values():
        os.remove(f)

    if not doplot:
        plt.close('all')

    return o


def test_dates(doplot=doplot):
    sc.heading('Testing dates')
    o = sc.objdict()

    x = np.array(sc.daterange('2020-12-24', '2021-01-15', asdate=True))
    y = sc.smooth(np.random.rand(len(x)))
    o.fig = plt.figure('Date formatters', figsize=(6,6))

    for i,style in enumerate(['auto', 'concise', 'sciris']):
        plt.subplot(3,1,i+1)
        plt.plot(x, y)
        plt.title('Date formatter: ' + style.title())
        sc.dateformatter(style=style, rotation=1)

    o.fig.tight_layout()

    plt.figure('Datenum formatter')
    plt.plot(np.arange(500), np.random.randn(500))
    sc.datenumformatter(start_date='2021-01-01')

    # Plot more use cases
    x1 = np.arange(2020, 2050) # Numerical years
    x2 = np.linspace(2020, 2022) # Fractional years
    x3 = pd.date_range('2020-01-01', periods=24, freq='MS') # 3. pd.Timestamp objects
    x4 = np.array(x3, dtype='datetime64') # 4. numpy.datetime64 objects

    x_vals = [x1, x2, x3, x4]
    titles = ['Integer years', 'Fractional years', 'pd.Timestamp', 'np.datetime64']

    o.fig2 = plt.figure('More dateformatter formats', figsize=(10, 12))
    for i, (x, title) in enumerate(zip(x_vals, titles)):
        ax = plt.subplot(len(x_vals), 1, i+1)
        y = np.random.rand(len(x))
        ax.plot(x, y, marker='o')
        ax.set_title(title)
        sc.dateformatter(ax)

    o.fig2.tight_layout()

    if not doplot:
        plt.close('all')

    return o


def test_fonts(doplot=doplot):
    sc.heading('Testing font functions')

    # Test getting fonts
    fonts = sc.fonts()

    # Test setting fonts
    orig = plt.rcParams['font.family']
    sc.fonts(add=sc.path(sc.thispath() / 'files/examplefont.ttf'), use=True, die=True, verbose=True)

    plt.figure('Fonts')
    plt.plot([1,2,3], [4,5,6])
    plt.xlabel('Example label in new font')

    # Reset
    plt.rcParams['font.family'] = orig

    if not doplot:
        plt.close('all')

    return fonts


def test_saveload(doplot=doplot):
    sc.heading('Testing figure save/load')

    fig = plt.figure('Save/load')
    plt.plot([1,3,7])
    fn = sc.objdict()
    fn.png1a = 'example1.png'
    fn.png1b = os.path.abspath('example1.png')
    fn.png2 = 'example2.png'
    fn.jpg  = 'example.jpg'

    # Basic usage
    sc.savefig(fn.png1a)
    sc.savefig(fn.png1b)
    md1a = sc.loadmetadata(fn.png1a)
    md1b = sc.loadmetadata(fn.png1b)
    sc.pp(md1a)
    sc.pp(md1b)

    # Complex usage
    comments = 'My figure'
    sc.savefig(fn.png2, fig=fig, comments=comments, freeze=True)
    md2 = sc.loadmetadata(fn.png2)
    assert md2['pipfreeze']['numpy'] == np.__version__ # Check version information was stored correctly
    assert md2['comments'] == comments

    # Should print a warning
    with sc.capture() as txt:
        sc.savefig(fn.jpg, die=False)
    assert 'Warning' in txt

    with pytest.raises(ValueError):
        sc.savefig(fn.jpg)

    # Tidy up
    for f in fn.values():
        try:
            os.remove(f)
            print(f'Removed temporary file {f}')
        except Exception as E:
            print(f'Could not remove {f}: {E}')

    if not doplot:
        plt.close('all')

    return md2


#%% Run as a script
if __name__ == '__main__':
    T = sc.timer()

    doplot = True
    sc.options(interactive=True)

    threed    = test_3d(doplot)
    artists   = test_stackedbar(doplot)
    other     = test_other(doplot)
    saved     = test_saving(doplot)
    dates     = test_dates(doplot)
    fonts     = test_fonts(doplot)
    metadata  = test_saveload(doplot)

    if doplot:
        plt.show()

    T.toc()
    print('Done.')
