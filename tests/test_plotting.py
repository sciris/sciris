"""
Version: 2018oct30
"""

import sciris as sc

torun = [
'hex2rgb',
'gridcolors',
]

doplot = False

if 'hex2rgb' in torun:
    c1 = sc.hex2rgb('#fff')
    c2 = sc.hex2rgb('fabcd8')

if 'gridcolors' in torun:
    colors_a = sc.gridcolors(ncolors=8,  doplot=doplot)
    colors_b = sc.gridcolors(ncolors=18, doplot=doplot)
    colors_c = sc.gridcolors(ncolors=28, doplot=doplot)



