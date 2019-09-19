"""
Version: 2019jan10
"""

import pylab as pl
import sciris as sc

torun = [
'hex2rgb',
'gridcolors',
'surf3d',
'bar3d',
]

if 'doplot' not in locals(): doplot = True

if 'hex2rgb' in torun:
    c1 = sc.hex2rgb('#fff')
    c2 = sc.hex2rgb('fabcd8')

if 'gridcolors' in torun:
    colors_a = sc.gridcolors(ncolors=8,  demo=doplot)
    colors_b = sc.gridcolors(ncolors=18, demo=doplot)
    colors_c = sc.gridcolors(ncolors=28, demo=doplot)

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

