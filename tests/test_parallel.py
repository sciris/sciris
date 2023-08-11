'''
Test parallelization functions and classes
'''

import sciris as sc
import numpy as np
import pylab as pl
import multiprocessing as mp
import pytest
ut = sc.importbypath(sc.thispath() / 'sc_test_utils.py')

if 'doplot' not in locals(): doplot = False


def subheading(label):
    return sc.printgreen('\n\n' + label)

def f(i):
    ''' Test function for parallelization -- here to avoid pickling errors '''
    return i**2


def test_simple():
    sc.heading('Example 1 -- simple usage as a shortcut to multiprocessing.map()')

    def f(x):
        return x*x

    results = sc.parallelize(f, [1,2,3])
    print(results)
    return


def test_embarrassing():
    sc.heading('Example 2 -- simple usage for "embarrassingly parallel" processing')

    def rnd():
        import pylab as pl
        return pl.rand()

    results = sc.parallelize(rnd, 10)
    print(results)
    return


def test_multiargs():
    sc.heading('Example 3 -- using multiple arguments')

    def f(x,y):
        return x*y

    results1 = sc.parallelize(func=f, iterarg=[(1,2),(2,3),(3,4)])
    results2 = sc.parallelize(func=f, iterkwargs={'x':[1,2,3], 'y':[2,3,4]})
    results3 = sc.parallelize(func=f, iterkwargs=[{'x':1, 'y':2}, {'x':2, 'y':3}, {'x':3, 'y':4}])
    assert results1 == results2 == results3
    print(results1)
    return


def test_noniterated(doplot=doplot):
    sc.heading('Example 4 -- using non-iterated arguments and dynamic load balancing')

    def myfunc(i, x, y):
        xy = [x+i*pl.randn(100), y+i*pl.randn(100)]
        return xy

    xylist1 = sc.parallelize(myfunc, kwargs={'x':3, 'y':8}, iterarg=range(5), maxcpu=0.8, interval=0.1) # Use kwargs dict
    xylist2 = sc.parallelize(myfunc, x=5, y=10, iterarg=[5,10,15]) # Supply kwargs directly

    if doplot:
        for p,xylist in enumerate([xylist1, xylist2]):
            pl.subplot(2,1,p+1)
            for i,xy in enumerate(reversed(xylist)):
                pl.scatter(xy[0], xy[1], label='Run %i'%i)
            pl.legend()
    return


def test_exceptions():
    sc.heading('Test that exceptions are being handled correctly')

    def good_func(x, y):
        return sum([(i+x*y)**2 for i in range(int(1e3))])

    def bad_func(x=0):
        raise ValueError('Intentional failure')

    # Should preserve original exception type
    with pytest.raises(ValueError):
        sc.parallelize(bad_func, iterarg=5)
    with pytest.raises(TypeError):
        sc.parallelize(bad_func, iterarg=5, kwargs=dict(y='bad kwarg'))

    # Test serial (debug) mode
    n = 10
    iterkwargs = dict(x=pl.arange(n), y=pl.linspace(0,1,n))
    res1 = sc.parallelize(good_func, iterkwargs=iterkwargs)
    res2 = sc.parallelize(good_func, iterkwargs=iterkwargs, serial=True)
    assert res1 == res2
    print(res1)

    return


def test_class():
    sc.heading('Testing sc.Parallel class')
    
    def slowfunc(i):
        np.random.seed(i)
        sc.timedsleep(0.2*np.random.rand())
        return i**2
    
    subheading('Creation and display')
    P = sc.Parallel(slowfunc, iterarg=range(4), parallelizer='multiprocess-async', ncpus=2) # NB, multiprocessing-async fails with a pickle error
    print(P)
    P.disp()
    
    subheading('Running asynchronously and monitoring')
    P.run_async()
    print(P.running)
    print(P.ready)
    print(P.status)
    P.monitor(interval=0.05)
    P.finalize()
    
    
    subheading('Checking parallelizers')
    
    iterarg = np.arange(4)
    
    print('Checking serial with copy and with progress')
    r1 = sc.parallelize(f, iterarg, parallelizer='serial-copy', progress=True)
    
    print('Checking multiprocessing')
    r2 = sc.parallelize(f, iterarg, parallelizer='multiprocessing')
    
    print('Checking fast (concurrent.futures)')
    r3 = sc.parallelize(f, iterarg, parallelizer='fast')
    
    print('Checking thread')
    r4 = sc.parallelize(f, iterarg, parallelizer='thread')
    
    print('Checking custom')
    with mp.Pool(processes=2) as pool:
        r5 = sc.parallelize(f, iterarg, parallelizer=pool.imap)
    
    assert r1 == r2 == r3 == r4 == r5

    
    subheading('Other')
    
    print('Checking CPUs')
    sc.Parallel(f, 10, ncpus=0.7)
    
    print('Validation: no jobs to run')
    with pytest.raises(ValueError):
        sc.Parallel(f, iterarg=[])
        
    print('Validation: no CPUs')
    with pytest.raises(ValueError):
        sc.Parallel(f, 10, ncpus=-3)
    
    print('Validation: invalid parallelizer')
    with pytest.raises(sc.KeyNotFoundError):
        sc.Parallel(f, 10, parallelizer='invalid-parallelizer')
        
    print('Validation: invalid async')
    with pytest.raises(ValueError):
        sc.Parallel(f, 10, parallelizer='serial-async')
        
    print('Validation: checking call signatures')
    ut.check_signatures(sc.parallelize, sc.Parallel.__init__, extras=['self', 'label'], die=True)
    
    return P


def test_components():
    sc.heading('Testing subcomponents directly')

    print('Testing TaskArgs and _task')
    a = sc.dictobj()
    a.func         = lambda: None
    a.index        = 0
    a.njobs        = 0
    a.iterval      = 0
    a.iterdict     = None
    a.args         = None
    a.kwargs       = None
    a.maxcpu       = 0
    a.maxmem       = 0
    a.interval     = 0
    a.embarrassing = True
    a.callback     = None
    a.progress     = None
    a.globaldict   = None
    a.useglobal    = None
    a.started      = None
    a.die          = None
    taskargs = sc.sc_parallel.TaskArgs(*a.values())
    task = sc.sc_parallel._task(taskargs)
    
    print('Testing progress bar')
    globaldict = {0:1, 1:1, 2:0, 3:0, 4:0}
    njobs = 5
    sc.sc_parallel._progressbar(globaldict, njobs, started=sc.now())
    return task



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    doplot = True

    test_simple()
    test_embarrassing()
    test_multiargs()
    test_noniterated(doplot)
    test_exceptions()
    P = test_class()
    test_components()

    sc.toc()
    print('Done.')