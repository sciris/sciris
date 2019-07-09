"""
Version: 2019jan10
"""

import pylab as pl
import sciris as sc

torun = [
'smooth',
]

if 'doplot' not in locals(): doplot = True

if 'smooth' in torun:
    data = pl.randn(200,100)
    smoothdata = sc.smooth(data,10)
    if doplot:
        pl.subplot(1,2,1)
        pl.pcolor(data)
        pl.subplot(1,2,2)
        pl.pcolor(smoothdata)
