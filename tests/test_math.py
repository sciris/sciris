'''
Test Sciris math functions.
'''

import numpy as np
import pylab as pl
import sciris as sc
import pytest


if 'doplot' not in locals(): doplot = False


def test_utils():
    sc.heading('Testing mathematical utilities')
    np.random.seed(1) # Ensure reproducibility

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
    o.isprime = [[i**2+1,sc.isprime(i**2+1)] for i in range(10)]
    assert sc.isprime(3) == True
    assert sc.isprime(9) == False
    assert sc.isprime(49) == False

    print('Testing sc.perturb()')
    o.perturb = sc.perturb(10, 0.3)
    δ = 0.1
    x = sc.perturb(1000, δ, randseed=1)
    assert np.all(x>1-δ)
    assert np.all(x<1+δ)
    assert abs(x.mean() - 1) < δ
    y = sc.perturb(1000, δ, normal=True)
    assert y.min() < x.min()
    assert y.max() > x.max()
    
    print('Testing sc.normsum() and sc.normalize()')
    o.normsum   = sc.normsum([2,5,3,6,2,6,7,2,3,4], 100) # Scale so sum equals 100
    o.normalize = sc.normalize([2,3,7,27])
    assert o.normsum[0] == 5.0
    assert o.normalize[2] == 0.2

    print('Testing sc.inclusiverange()')
    o.inclusiverange = sc.inclusiverange(3,5,0.2)
    sc.inclusiverange()
    sc.inclusiverange(3)
    sc.inclusiverange(3,5)
    assert o.inclusiverange[-1] == 5

    print('Testing sc.randround()')
    base = np.random.randn(20)
    o.randround = sc.randround(base)
    sc.randround(base.tolist())
    sc.randround(base[0])

    print('Testing sc.cat()')
    o.cat = sc.cat(np.array([1,2,3]), [4,5], 6, copy=True)
    assert o.cat[3] == 4
    assert len(sc.cat()) == 0
    
    print('Testing sc.linregress()')
    x = range(10)
    y = sorted(2*np.random.rand(10) + 1)
    o.out = sc.linregress(x, y, full=True) # Has out.m, out.b, out.x, out.y, out.corr, etc.
    assert o.out.r2 > 0

    return o


def test_find():
    sc.heading('Testing find functions')
    np.random.seed(1) # Ensure reproducibility

    found = sc.objdict()

    print('Testing sc.findinds()')
    found.vals = sc.findinds([2,3,6,3], 6)
    assert found.vals[0] == 2
    
    data = np.random.rand(1000) 
    f1 = sc.findinds(data<0.5) # Standard usage; returns e.g. array([2, 4, 5, 9]) 
    f2 = sc.findinds(data>0.3, data<0.5) # Multiple arguments 
    assert len(f2) < len(f1)
    
    maxval = 0.1
    tabdata = np.random.rand(20,20)
    loc = sc.findinds(tabdata<maxval, ind=3)
    assert tabdata[loc] < maxval

    print('Testing sc.count()')
    found.count = sc.count([1,2,2,3], 2.0)
    assert found.count == 2

    print('Testing sc.findfirst(), sc.findlast()')
    found.first = sc.findfirst(pl.rand(10))
    found.last = sc.findlast(pl.rand(10))
    sc.findlast([1,2,3], 4, die=False)
    with pytest.raises(IndexError):
        sc.findlast([1,2,3], 4)

    print('Testing sc.findnearest()')
    found.nearest = sc.findnearest([0,2,5,8], 3)
    sc.findnearest([0,2,5,8], [3,6])
    assert found.nearest == 1

    print('Testing sc.numdigits()')
    found.numdigits = sc.numdigits(1234)
    found.numdigits_max = max(sc.numdigits([10, 200, 30000]))
    found.numdigits_dec = sc.numdigits(0.01)
    assert found.numdigits == 4
    assert found.numdigits_max == 5
    assert found.numdigits_dec == -2
    assert sc.numdigits(-1, count_minus=True) == 2
    assert sc.numdigits(0.02, count_decimal=True) == -4

    return found


def test_nan():
    sc.heading('Testing NaN functions')
    o = sc.objdict()
    
    print('Testing sc.getvalidinds(), sc.getvaliddata()')
    o.inds = sc.getvalidinds([3,5,8,13], [2000, np.nan, np.nan, 2004]) # Returns array([0,3])
    o.data = sc.getvaliddata(np.array([3,5,8,13]), np.array([2000, np.nan, np.nan, 2004])) # Returns array([3,13])
    assert o.inds[-1] == 3
    assert o.data[-1] == 13
    
    print('Testing sc.sanitize()')
    data = [3, 4, np.nan, 8, 2, np.nan, np.nan, 8]
    sanitized, inds = sc.sanitize(data, returninds=True) # Remove NaNs
    assert len(sanitized) == 5
    assert 1 in inds
    
    s2 = sc.sanitize(data, replacenans=True) # Replace NaNs using nearest neighbor interpolation
    s3 = sc.sanitize(data, replacenans='nearest') # Eequivalent to replacenans=True
    s4 = sc.sanitize(data, replacenans='linear') # Replace NaNs using linear interpolation
    s5 = sc.sanitize(data, replacenans=0) # Replace NaNs with 0
    assert s2[2] == 4
    assert s3[2] == 4
    assert s4[2] == 6
    assert s5[2] == 0
    
    allnans = np.full(5, np.nan)
    assert len(sc.sanitize(allnans)) == 0
    assert sc.sanitize(allnans, defaultval=7) == 7
    
    print('Testing fillnan and rmnan')
    data2d = np.random.rand(3,3)
    data2d[1,1] = np.nan
    data2d[2,2] = np.nan
    with pytest.raises(ValueError):
        sc.rmnans(data2d)
    with pytest.raises(NotImplementedError):
        sc.fillnans(data2d, replacenans='nearest')
    filled = sc.fillnans(data2d, 100)
    assert 200 < filled.sum() < 207
    o.sanitized = sanitized
    o.filled = filled
    
    print('Testing findnan')
    data = [0, 1, 2, np.nan, 4, np.nan, 6, np.nan, np.nan, np.nan, 10]
    assert len(sc.findnans(data)) == 5
    
    print('Testing nanequal')
    arr1 = np.array([1, 2, np.nan, np.nan]) 
    arr2 = [1, 2, np.nan, np.nan] 
    assert np.all(sc.nanequal(arr1, arr2)) # Returns array([ True,  True,  True, True]) 
     
    arr3 = [3, np.nan, 'foo'] 
    assert sc.nanequal(arr3, arr3, arr3, scalar=True) # Returns True 
    
    arr4 = [np.nan, np.nan, 'foo']
    assert not sc.nanequal(arr3, arr4, scalar=True)
    
    arr5 = [np.nan, np.nan, 'foo', 3]
    assert not sc.nanequal(arr4, arr5, scalar=True)
    
    
    return o


def test_smooth(doplot=doplot):
    sc.heading('Testing smoothing')
    np.random.seed(1) # Ensure reproducibility
    
    o = sc.objdict()

    print('Testing sc.smooth()')
    data = pl.randn(200,100)
    o.smoothdata = sc.smooth(data,10)

    print('Testing sc.smoothinterp()')
    n = 50
    x1 = pl.arange(n)
    y1 = pl.randn(n)
    y1[10] = np.nan
    x2 = pl.linspace(-5,n+5,100)
    o.smoothinterp = sc.smoothinterp(x2, x1, y1)
    assert np.isnan(o.smoothinterp).sum() == 0
    assert np.nanmin(y1) < o.smoothinterp.min()
    assert np.nanmax(y1) > o.smoothinterp.max()
    
    si = sc.smoothinterp(x2, x1, y1, ensurefinite=False)
    assert np.isnan(si).sum() > 0
    
    print('Testing sc.rolling()')
    d = pl.randn(365)
    o.r = sc.rolling(d, replacenans=False)
    assert sum(abs(np.diff(d))) > sum(abs(np.diff(o.r))) # Rolling should make the diffs smaller
    for op in ['none', 'median', 'sum']:
        sc.rolling(d, operation=op)
    
    if doplot:
        pl.subplot(3,1,1)
        pl.pcolor(data)
        pl.subplot(3,1,2)
        pl.pcolor(o.smoothdata)
        pl.subplot(3,1,3)
        pl.scatter(x1, y1)
        pl.plot(x2, o.smoothinterp)
        pl.show()

    return o


def test_gauss1d(doplot=doplot):
    sc.heading('Testing Gaussian 1D smoothing')

    # Setup
    n = 20
    x = np.sort(pl.rand(n))
    y = (x-0.3)**2 + 0.2*pl.rand(n)

    # Smooth
    yi = sc.gauss1d(x, y)
    yi2 = sc.gauss1d(x, y, scale=0.3)
    xi3 = pl.linspace(0, 1, n)
    yi3 = sc.gauss1d(x, y, xi3)

    # Plot oiginal and interpolated versions
    if doplot:
        kw = dict(alpha=0.5, lw=2, marker='o')
        pl.figure()
        pl.scatter(x, y,  label='Original')
        pl.plot(x, yi,    label='Default smoothing', **kw)
        pl.plot(x, yi2,   label='More smoothing', **kw)
        pl.plot(xi3, yi3, label='Uniform spacing', **kw)
        pl.legend()
        pl.show()

    return yi3


def test_gauss2d(doplot=doplot):
    sc.heading('Testing Gaussian 2D smoothing')

    # Parameters
    x = np.random.rand(40)
    y = np.random.rand(40)
    z = 1-(x-0.5)**2 + (y-0.5)**2

    # Method 1 -- form grid
    xi = np.linspace(0,1,20)
    yi = np.linspace(0,1,20)
    zi = sc.gauss2d(x, y, z, xi, yi, scale=0.1, grid=True)

    # Method 2 -- use points directly
    xi2 = np.random.rand(400)
    yi2 = np.random.rand(400)
    zi2 = sc.gauss2d(x, y, z, xi2, yi2, scale=0.1, grid=False)

    if doplot:
        sc.scatter3d(x, y, z, c=z)
        sc.surf3d(zi)
        sc.scatter3d(xi2, yi2, zi2, c=zi2)
        pl.show()

    return zi


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    doplot = True

    other   = test_utils()
    found   = test_find()
    nan     = test_nan()
    smooth  = test_smooth(doplot)
    gauss1d = test_gauss1d(doplot)
    gauss2d = test_gauss2d(doplot)

    sc.toc()
    print('Done.')
