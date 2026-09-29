'''
Version: 2019jul09
'''

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc


if 'doplot' not in locals():
    doplot = True


def test_simple():
    result = sc.asd(np.linalg.norm, [1, 2, 3])
    print('Result:')
    print(result.x)
    return result


def test_args():
    def my_func(x, scale=1.0, weight=1.0): # Example function with keywords
        return abs((x[0] - 1)) + abs(x[1] + 2)*scale + abs(x[2] + 3)*weight

    result = sc.asd(my_func, x=[0, 0, 1], args=[0.5, 0.1]) # Option 1 for passing arguments
    result = sc.asd(my_func, x=[0, 0, 1], args=dict(scale=0.5, weight=0.1)) # Option 1 for passing arguments
    result = sc.asd(my_func, x=[0, 0, 1], scale=0.5, weight=0.1) # Option 2 for passing arguments
    return result


def test_complex():

    def objective(pars):
        x, y, z = pars
        err = 50*(y - x**2)**2 + (0.5 - x)**2 + 2*abs(z-0.5); # Rosenbrock's valley
        return err

    randseed = 239847
    startvals = [-1, -1, -1]
    minvals = [-1, -1, -1]
    maxvals = [1, 1, 1]
    result = sc.asd(objective, startvals, xmin=minvals, xmax=maxvals, randseed=randseed)

    if doplot:

        # Plot parameter space
        np.random.seed(randseed)
        npts = 15
        perturb = 0.05
        xvec = np.linspace(-1,1,npts)
        yvec = np.linspace(-1,1,npts)
        zvec = np.linspace(-1,1,npts)
        alldata = []
        for x in xvec:
            for y in yvec:
                for z in zvec:
                    xp = x + perturb*np.random.randn()
                    yp = y + perturb*np.random.randn()
                    zp = z + perturb*np.random.randn()
                    o = np.log10(objective([xp, yp, zp]))
                    alldata.append([xp, yp, zp, o])
        alldata = np.array(alldata)
        X = alldata[:,0]
        Y = alldata[:,1]
        Z = alldata[:,2]
        O = alldata[:,3]
        fig = plt.figure(figsize=(16,12))
        ax = sc.scatter3d(X, Y, Z, O, fig=fig, alpha=0.3)
        ax.view_init(elev=50, azim=-45)
        plt.xlabel('x')
        plt.ylabel('y')

        # Plot trajectory
        X2 = result.details.xvals[:,0]
        Y2 = result.details.xvals[:,1]
        Z2 = result.details.xvals[:,2]
        O2 = np.log10(result.details.fvals)
        ax = sc.scatter3d(X2, Y2, Z2, O2, fig=fig, marker='d')
        ax = sc.plot3d(X2, Y2, Z2, fig=fig, c='k', lw=3)

    return result


def test_inputs():
    """ Check that inputs of different shapes and lengths are handled correctly """
    def sq(x): return float(np.sum((np.asarray(x) - 1.0)**2))
    x0 = [1.5, 2.5, 3.5]
    kw = dict(maxiters=200, verbose=0, randseed=1)

    # One pinitial per parameter is the same as two per parameter
    r1 = sc.asd(sq, x0, pinitial=[1,1,1], **kw)
    r2 = sc.asd(sq, x0, pinitial=[1,1,1,1,1,1], **kw)
    assert np.array_equal(r1.x, r2.x) and r1.fval < 1e-3

    # Scalar bounds and step sizes are broadcast
    for key,val in dict(xmin=0, xmax=10, sinitial=0.1).items():
        r = sc.asd(sq, x0, **{key:val}, **kw)
        assert r.fval < 1e-3

    # An out-of-bounds starting point raises, or is clipped if die=False
    def far(x): return float((x[0] - 4.9)**2 + (x[1] - 4.9)**2)
    with pytest.raises(ValueError):
        sc.asd(far, [4.8, 4.8], xmin=[0, 0], xmax=[1, 1], **kw)
    with pytest.warns(UserWarning):
        r = sc.asd(far, [4.8, 4.8], xmin=[0, 0], xmax=[1, 1], die=False, **kw)
    assert np.array_equal(r.x, [1, 1])

    # Objective values that are 0-d or 2-d size-1 arrays are accepted
    for func in [lambda x: np.array(sq(x)), lambda x: np.array([[sq(x)]])]:
        r = sc.asd(func, [3., 4.], **kw)
        assert r.fval < 1e-3

    # The objective is called with x in its original shape
    target = np.array([[1., 2.], [3., 4.]])
    r = sc.asd(lambda x: float(np.sum((x - target)**2)), np.zeros((2, 2)), **kw)
    assert r.x.shape == (2,2) and r.fval < 1e-3
    r = sc.asd(lambda x: float(np.sum((x - target)**2)), np.zeros((2, 2)), xmin=np.zeros((2,2)), xmax=5*np.ones((2,2)), **kw)
    assert r.x.shape == (2,2) and r.fval < 1e-3
    r = sc.asd(lambda x: float((x - 2)**2), 5., **kw)
    assert r.x.shape == () and abs(r.x - 2) < 1e-2

    # maxiters=0 does nothing
    r = sc.asd(sq, x0, maxiters=0, verbose=0)
    assert np.array_equal(r.x, x0) and r.exitreason == 'Maximum iterations reached'

    return r


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    r1 = test_simple()
    r2 = test_args()
    r3 = test_complex()
    r4 = test_inputs()

    sc.toc()
    print('Done.')