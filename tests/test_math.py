'''
Test Sciris math functions.
'''

import numpy as np
import pylab as pl
import sciris as sc


if 'doplot' not in locals(): doplot = False


def test_utils():
    sc.heading('Testing utilities')

    o = sc.objdict()

    print('Testing sc.approx()')
    assert sc.approx(2*6, 11.9999999, eps=1e-6) # Returns True
    o.approx = sc.approx([3,12,11.9], 12) # Returns array([False, True, False], dtype=bool)
    assert not o.approx[0]

    print('Testing sc.savedivide()')
    assert sc.safedivide(numerator=0, denominator=0, default=1, eps=0) == 1 # Returns 1
    o.safedivide = sc.safedivide(3, np.array([1,3,0]),-1, warn=False)  # Returns array([ 3,  1, -1])
    assert o.safedivide[-1] == -1

    print('Testing sc.isprime()')
    o.isprime = [[i,sc.isprime(i)] for i in range(10)]

    print('Testing sc.perturb()')
    o.perturb = sc.perturb(10, 0.3)

    print('Testing sc.normsum() and sc.normalize()')
    o.normsum   = sc.normsum([2,5,3,6,2,6,7,2,3,4], 100) # Scale so sum equals 100
    o.normalize = sc.normalize([2,3,7,27])
    assert o.normsum[0] == 5.0
    assert o.normalize[2] == 0.2

    print('Testing sc.inclusiverange()')
    o.inclusiverange = sc.inclusiverange(3,5,0.2)
    assert o.inclusiverange[-1] == 5

    print('Testing sc.randround()')
    o.randround = sc.randround(np.random.randn(20))

    print('Testing sc.cat()')
    o.cat = sc.cat(np.array([1,2,3]), [4,5], 6)
    assert o.cat[3] == 4

    return o


def test_find():
    sc.heading('Testing find functions')

    found = sc.objdict()

    print('Testing sc.findinds()')
    found.vals = sc.findinds([2,3,6,3], 6)
    assert found.vals[0] == 2

    print('Testing sc.findfirst(), sc.findlast()')
    found.first = sc.findfirst(pl.rand(10))
    found.last = sc.findlast(pl.rand(10))

    print('Testing sc.findnearest()')
    found.nearest = sc.findnearest([0,2,5,8], 3)
    assert found.nearest == 1

    print('Testing sc.getvalidinds(), sc.getvaliddata()')
    found.inds = sc.getvalidinds([3,5,8,13], [2000, np.nan, np.nan, 2004]) # Returns array([0,3])
    found.data = sc.getvaliddata(np.array([3,5,8,13]), np.array([2000, np.nan, np.nan, 2004])) # Returns array([3,13])
    assert found.inds[-1] == 3
    assert found.data[-1] == 13

    print('Testing sc.sanitize()')
    sanitized,inds = sc.sanitize(np.array([3, 4, np.nan, 8, 2, np.nan, np.nan, np.nan, 8]), returninds=True)
    sanitized      = sc.sanitize(np.array([3, 4, np.nan, 8, 2, np.nan, np.nan, np.nan, 8]), replacenans=True)
    sanitized      = sc.sanitize(np.array([3, 4, np.nan, 8, 2, np.nan, np.nan, np.nan, 8]), replacenans=0)
    found.sanitized = sanitized

    return found


def test_smooth(doplot=doplot):
    sc.heading('Testing smoothing')

    print('Testing sc.smooth()')
    data = pl.randn(200,100)
    smoothdata = sc.smooth(data,10)

    print('Testing sc.smoothinterp()')
    n = 20
    x1 = pl.arange(n)
    y1 = pl.randn(n)
    x2 = pl.linspace(0,n+5,100)
    y2 = sc.smoothinterp(x2, x1, y1)

    if doplot:
        pl.subplot(3,1,1)
        pl.pcolor(data)
        pl.subplot(3,1,2)
        pl.pcolor(smoothdata)
        pl.subplot(3,1,3)
        pl.scatter(x1, y1)
        pl.plot(x2, y2)
        pl.show()

    return smoothdata


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    doplot = True

    other = test_utils()
    found = test_find()
    smoothed = test_smooth(doplot)

    sc.toc()
    print('Done.')