"""
Version:
"""
import pylab as pl
import scirisweb as sw


torun = [
'browser'
]


if 'browser' in torun:
    figs = []
    for n in [10, 50]:
        fig = pl.figure()
        pl.plot(pl.rand(n), pl.rand(n))
        figs.append(fig)
    
    barfig = pl.figure()
    pl.bar(pl.arange(10), pl.rand(10))
    barjson = sw.mpld3ify(barfig)
    sw.browser(figs=figs, jsons=barjson)