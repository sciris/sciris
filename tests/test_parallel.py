import sciris as sc
import pylab as pl

torun = [
#'simple',
#'embarrassing',
'multiargs',
#'noniterated'
]



#Example 1 -- simple usage as a shortcut to multiprocessing.map():
if 'simple' in torun:
    
    def f(x):
        return x*x
    
    results = sc.parallelize(f, [1,2,3])
    print(results)



#Example 2 -- simple usage for "embarrassingly parallel" processing:
if 'embarrassing' in torun:
    
    def rnd(i):
        import pylab as pl
        return pl.rand()
    
    results = sc.parallelize(rnd, 10)
    print(results)


#Example 3 -- using multiple arguments:
if 'multiargs' in torun:

    def f(x,y):
        return x*y
    
    results1 = sc.parallelize(func=f, iterarg=[(1,2),(2,3),(3,4)])
    results2 = sc.parallelize(func=f, iterkwargs={'x':[1,2,3], 'y':[2,3,4]})
    results3 = sc.parallelize(func=f, iterkwargs=[{'x':1, 'y':2}, {'x':2, 'y':3}, {'x':3, 'y':4}])
    assert results1 == results2 == results3
    print(results1)


#Example 4 -- using non-iterated arguments and dynamic load balancing:
if 'noniterated' in torun:

    def myfunc(i, x, y):
        xy = [x+i*pl.rand(100), y+i*pl.randn(100)]
        return xy
    
    xylist = sc.parallelize(myfunc, kwargs={'x':3, 'y':8}, iterarg=range(5), maxload=0.8)
    
    for xy in xylist:
        pl.scatter(xy[0], xy[1])