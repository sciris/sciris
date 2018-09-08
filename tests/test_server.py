"""
Version:
"""
import pylab as pl
import mpld3
import scirisweb as sw
import json



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
    graph_dict = mpld3.fig_to_dict(barfig)
    json = str(json.dumps(sw.sanitize_json(mpld3.fig_to_dict(barfig)))) # This shouldn't be necessary, but it is...
    sw.browser(figs=figs+[barfig])