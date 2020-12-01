'''
Test parallelization. Not written as pytest tests because it conflicts with pytest's
built-in parallelization, and since functions-within-functions can't be pickled.
'''

import sciris as sc
import pylab as pl


if 'doplot' not in locals(): doplot = False


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

    xylist1 = sc.parallelize(myfunc, kwargs={'x':3, 'y':8}, iterarg=range(5), maxload=0.8, interval=0.2) # Use kwargs dict
    xylist2 = sc.parallelize(myfunc, x=5, y=10, iterarg=[5,10,15]) # Supply kwargs directly

    if doplot:
        for p,xylist in enumerate([xylist1, xylist2]):
            pl.subplot(2,1,p+1)
            for i,xy in enumerate(reversed(xylist)):
                pl.scatter(xy[0], xy[1], label='Run %i'%i)
            pl.legend()
    return


def test_parallelcmd():
    sc.heading('Using a string-based command')

    const = 4
    parfor = {'val':[3,5,9]}
    returnval = 'result'
    cmd = """
newval = val+const # Note that this can't be indented
result = newval**2
    """
    results = sc.parallelcmd(cmd=cmd, parfor=parfor, returnval=returnval, const=const, maxload=0)
    print(results)
    return


def test_components():
    sc.heading('Testing subcomponents directly')
    sc.loadbalancer()

    def empty(): pass

    args = [0]*9
    args[0] = empty
    args[3] = None # Set iterdict to None
    args[4] = None # Set args to empty list
    args[5] = None # Set kwargs to empty dict
    args[8] = True # Set embarrassing
    taskargs = sc.sc_parallel.TaskArgs(*args)
    task = sc.sc_parallel._parallel_task(taskargs)
    return task



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    doplot = True

    test_simple()
    test_embarrassing()
    test_multiargs()
    test_noniterated(doplot)
    test_parallelcmd()
    test_components()

    sc.toc()
    print('Done.')