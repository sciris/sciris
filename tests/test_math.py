"""
Version: 2020apr27
"""

import pylab as pl
import sciris as sc


if 'doplot' not in locals(): doplot = True


def test_smooth():
    data = pl.randn(200,100)
    smoothdata = sc.smooth(data,10)
    if doplot:
        pl.subplot(1,2,1)
        pl.pcolor(data)
        pl.subplot(1,2,2)
        pl.pcolor(smoothdata)
    return smoothdata


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    smoothed = test_smooth()

    sc.toc()
    print('Done.')