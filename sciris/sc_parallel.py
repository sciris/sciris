'''
Functions to allow parallelization to be performed easily.

NB: Uses ``multiprocess`` instead of ``multiprocessing`` under the hood for
broadest support  across platforms (e.g. Jupyter notebooks).

Highlights:
    - :func:`parallelize`: as-easy-as-possible parallelization
'''

import warnings
import numpy as np
import multiprocess as mp
import multiprocessing as mpi
import concurrent.futures as cf
from . import sc_utils as scu
from . import sc_profiling as scp


##############################################################################
#%% Parallelization functions
##############################################################################

__all__ = ['parallelize']


def parallelize(func, iterarg=None, iterkwargs=None, args=None, kwargs=None, ncpus=None, 
                maxcpu=None, maxmem=None, interval=None, parallelizer=None, serial=False, 
                returnpool=False, progress=False, callback=None, die=True, **func_kwargs):
    '''
    Execute a function in parallel.

    Most simply, ``sc.parallelize()`` acts as an shortcut for using ``Pool.map()``.
    However, it also provides flexibility in how arguments are passed to the function,
    load balancing, etc.

    Either or both of ``iterarg`` or ``iterkwargs`` can be used. ``iterarg`` can
    be an iterable or an integer; if the latter, it will run the function that number
    of times and not pass the argument to the function (which may be useful for
    running "embarrassingly parallel" simulations). ``iterkwargs`` is a dict of
    iterables; each iterable must be the same length (and the same length of ``iterarg``,
    if it exists), and each dict key will be used as a kwarg to the called function.
    Any other kwargs passed to ``sc.parallelize()`` will also be passed to the function.

    This function can either use a fixed number of CPUs or allocate dynamically
    based on load. If ``ncpus`` is ``None`` and ``maxcpu`` is ``None``, then it
    will use the number of CPUs returned by ``multiprocessing``; if ``ncpus`` is
    not ``None``, it will use the specified number of CPUs; if ``ncpus`` is ``None``
    and ``maxcpu`` is not ``None``, it will allocate the number of CPUs dynamically.

    Args:
        func         (func)      : the function to parallelize
        iterarg      (list)      : the variable(s) to provide to each process (see examples below)
        iterkwargs   (dict)      : another way of providing variables to each process (see examples below)
        args         (list)      : positional arguments for each process, the same for all processes
        kwargs       (dict)      : keyword arguments for each process, the same for all processes
        ncpus        (int/float) : number of CPUs to use (if <1, treat as a fraction of the total available; if None, use loadbalancer)
        maxcpu       (float)     : maximum CPU load; otherwise, delay the start of the next process (not used if ``ncpus`` is specified)
        maxmem       (float)     : maximum fraction of virtual memory (RAM); otherwise, delay the start of the next process
        interval     (float)     : number of seconds to pause between starting processes for checking load
        parallelizer (str/func)  : parallelization function; default 'multiprocess' (see below for details)
        serial       (bool)      : whether to skip parallelization and run in serial (useful for debugging; equivalent to ``parallelizer='serial'``)
        returnpool   (bool)      : whether to return the process pool as well as the results
        progress     (bool)      : whether to show a progress bar
        callback     (func)      : an optional function to call from each worker
        die          (bool)      : whether to stop immediately if an exception is encountered (otherwise, store the exception as the result)
        func_kwargs  (dict)      : merged with kwargs (see above)

    Returns:
        List of outputs from each process


    **Example 1 -- simple usage as a shortcut to multiprocess.map()**::

        def f(x):
            return x*x

        results = sc.parallelize(f, [1,2,3])

    **Example 2 -- simple usage for "embarrassingly parallel" processing**::

        import numpy as np

        def rnd():
            np.random.seed()
            return np.random.random()

        results = sc.parallelize(rnd, 10, ncpus=4)

    **Example 3 -- three different equivalent ways to use multiple arguments**::

        def f(x,y):
            return x*y

        results1 = sc.parallelize(func=f, iterarg=[(1,2),(2,3),(3,4)])
        results2 = sc.parallelize(func=f, iterkwargs={'x':[1,2,3], 'y':[2,3,4]})
        results3 = sc.parallelize(func=f, iterkwargs=[{'x':1, 'y':2}, {'x':2, 'y':3}, {'x':3, 'y':4}])
        assert results1 == results2 == results3

    **Example 4 -- using non-iterated arguments and dynamic load balancing**::

        def myfunc(i, x, y):
            np.random.seed()
            xy = [x+i*np.random.randn(100), y+i*np.random.randn(100)]
            return xy

        xylist1 = sc.parallelize(myfunc, kwargs={'x':3, 'y':8}, iterarg=range(5), maxcpu=0.8, interval=0.2) # Use kwargs dict
        xylist2 = sc.parallelize(myfunc, x=5, y=10, iterarg=[0,1,2], parallelizer='multiprocessing') # Supply kwargs directly and use a different parallelizer

        for p,xylist in enumerate([xylist1, xylist2]):
            pl.subplot(2,1,p+1)
            for i,xy in enumerate(reversed(xylist)):
                pl.scatter(xy[0], xy[1], label='Run %i'%i)
            pl.legend()

    **Example 5 -- using a custom parallelization function**::

        def f(x,y):
            return [x]*y

        import multiprocessing as mp
        pool = mp.Pool(processes=2)
        results = sc.parallelize(f, iterkwargs=dict(x=[1,2,3], y=[4,5,6]), parallelizer=pool.map) # Note: parallelizer is pool.map, not pool


    **Example 6 -- using Sciris as an interface to Dask**::
    
        def f(x,y):
            return [x]*y
    
        def dask_map(task, argslist):
            import dask
            queued = [dask.delayed(task)(args) for args in argslist]
            return list(dask.compute(*queued))
    
        results = sc.parallelize(f, iterkwargs=dict(x=[1,2,3], y=[4,5,6]), parallelizer=dask_map)


    **Note 1**: the default parallelizer ``"multiprocess"`` uses ``dill`` for pickling, so
    is the most versatile (e.g., it can pickle non-top-level functions). However,
    it is also the slowest for passing large amounts of data. You can switch between
    these with ``parallelizer='fast'`` (``concurrent.futures``) and ``parallelizer='robust'``
    (``multiprocess``).

    The ``parallelizer`` argument allows a wide range of different parallelizers
    (including different aliases for each), and also supports user-supplied ones.
    Note that in most cases, the default parallelizer will suffice. However, the
    full list of options is:
        
        - ``None``, ``'default'``, ``'robust'``, ``'multiprocess'``: the slow but robust dill-based parallelizer ``multiprocess``
        - ``'fast'``, ``'concurrent'``, ``'concurrent.futures'``: the faster but more fragile pickle-based Python-default parallelizer ``concurrent.futures``
        - ``'multiprocessing'``: the previous pickle-based Python default parallelizer, ``multiprocessing``
        - ``'serial'``, ``'serial-nocopy'``: no parallelization (single-threaded); with "-nocopy", do not force pickling
        - ``'thread'``', ``'threadpool'``', ``'thread-nocopy'``': thread- rather than process-based parallelization ("-nocopy" as above)
        - User supplied: any ``map()``-like function that takes in a function and an argument list


    **Note 2**: If parallelizing figure generation, use a non-interactive backend,
    or make sure (a) figure is closed inside the function call, and (b) the figure
    object is not returned. Otherwise, parallelization won't increase speed (and
    might even be slower than serial!).

    
    **Note 3**: to use on Windows, parallel calls must contained with an ``if __name__ == '__main__'`` block.

    For example::

        import sciris as sc

        def f(x,y):
            return x*y

        if __name__ == '__main__':
            results = sc.parallelize(func=f, iterarg=[(1,2),(2,3),(3,4)])
            print(results)

    | New in version 1.1.1: "serial" argument.
    | New in version 2.0.0: changed default parallelizer from ``multiprocess.Pool`` to ``concurrent.futures.ProcessPoolExecutor``; replaced ``maxload`` with ``maxcpu``/``maxmem``; added ``returnpool`` argument
    | New in version 2.0.4: added "die" argument; changed exception handling
    | New in version 2.2.0: propagated "die" to tasks
    '''
    # Handle maxload
    maxload = func_kwargs.pop('maxload', None)
    if maxload is not None: # pragma: no cover
        maxcpu = maxload
        warnmsg = 'sc.loadbalancer() argument "maxload" has been renamed "maxcpu" as of v2.0.0'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)
    if ncpus is None and maxcpu is None:
        ncpus = scp.cpu_count()
    if ncpus is not None and ncpus < 1: # Less than one, treat as a fraction of total
        ncpus = int(scp.cpu_count()*ncpus)

    # Handle kwargs
    kwargs = scu.mergedicts(kwargs, func_kwargs)

    # Handle iterarg and iterkwargs
    niters = 0
    embarrassing = False # Whether or not it's an embarrassingly parallel optimization
    if iterarg is not None and iterkwargs is not None: # pragma: no cover
        errormsg = 'You can only use one of iterarg or iterkwargs as your iterable, not both'
        raise ValueError(errormsg)
    if iterarg is not None:
        if not(scu.isiterable(iterarg)):
            try:
                iterarg = np.arange(iterarg)
                embarrassing = True
            except Exception as E: # pragma: no cover
                errormsg = f'Could not understand iterarg "{iterarg}": not iterable and not an integer: {str(E)}'
                raise TypeError(errormsg)
        niters = len(iterarg)
    if iterkwargs is not None: # Check that iterkwargs has the right format
        if isinstance(iterkwargs, dict): # It's a dict of lists, e.g. {'x':[1,2,3], 'y':[2,3,4]}
            for key,val in iterkwargs.items():
                if not scu.isiterable(val): # pragma: no cover
                    errormsg = f'iterkwargs entries must be iterable, not {type(val)}'
                    raise TypeError(errormsg)
                if not niters:
                    niters = len(val)
                else:
                    if len(val) != niters: # pragma: no cover
                        errormsg = f'All iterkwargs iterables must be the same length, not {niters} vs. {len(val)}'
                        raise ValueError(errormsg)
        elif isinstance(iterkwargs, list): # It's a list of dicts, e.g. [{'x':1, 'y':2}, {'x':2, 'y':3}, {'x':3, 'y':4}]
            niters = len(iterkwargs)
            for item in iterkwargs:
                if not isinstance(item, dict): # pragma: no cover
                    errormsg = f'If iterkwargs is a list, each entry must be a dict, not {type(item)}'
                    raise TypeError(errormsg)
        else: # pragma: no cover
            errormsg = f'iterkwargs must be a dict of lists, a list of dicts, or None, not {type(iterkwargs)}'
            raise TypeError(errormsg)

    if niters == 0:
        errormsg = 'Nothing found to parallelize: please supply an iterarg, iterkwargs, or both'
        raise ValueError(errormsg)

    # Construct argument list
    argslist = []
    for index in range(niters):
        if iterarg is None:
            iterval = None
        else:
            iterval = iterarg[index]
        if iterkwargs is None:
            iterdict = None
        else:
            if isinstance(iterkwargs, dict): # Dict of lists
                iterdict = {}
                for key,val in iterkwargs.items():
                    iterdict[key] = val[index]
            elif isinstance(iterkwargs, list): # List of dicts
                iterdict = iterkwargs[index]
            else:  # pragma: no cover # Should be caught by previous checking, so shouldn't happen
                errormsg = f'iterkwargs type not understood ({type(iterkwargs)})'
                raise TypeError(errormsg)
        taskargs = TaskArgs(func=func, index=index, niters=niters, iterval=iterval, iterdict=iterdict,
                            args=args, kwargs=kwargs, maxcpu=maxcpu, maxmem=maxmem,
                            interval=interval, embarrassing=embarrassing, callback=callback,
                            die=die)
        argslist.append(taskargs)
    
    # Set up the run
    pool = None # Defined here so it can be returned
    if ncpus is not None:
        ncpus = min(ncpus, len(argslist)) # Don't use more CPUs than there are things to process
    if serial: # This is a separate keyword argument, but make it consistent
        parallelizer = 'serial'
        
    # Handle the choice of parallelizer
    fast   = 'concurrent.futures'
    robust = 'multiprocess'
    if parallelizer is None or scu.isstring(parallelizer):

        # Map parallelizer to consistent choices
        mapping = {
            None                 : robust,
            'default'            : robust,
            'robust'             : robust,
            'fast'               : fast,
            'serial'             : 'serial',
            'serial-nocopy'      : 'serial',
            'concurrent.futures' : 'concurrent.futures',
            'concurrent'         : 'concurrent.futures',
            'multiprocess'       : 'multiprocess',
            'multiprocessing'    : 'multiprocessing',
            'thread'             : 'thread',
            'threadpool'         : 'thread',
            'thread-nocopy'      : 'thread',
        }
        try:
            pname = mapping[parallelizer]
        except:
            errormsg = f'Parallelizer "{parallelizer}" not found: must be one of {scu.strjoin(mapping.keys())}'
            raise scu.KeyNotFoundError(errormsg)
    else:
        pname = 'custom' # If a custom parallelizer is provided
        
    
    def run_parallel(pname, parallelizer, argslist):
        ''' Choose how to run in parallel '''
        
        # Choose which parallelizer to use
        if pname == 'serial':
            if not 'nocopy' in parallelizer: # Niche use case of running without deepcopying
                argslist = scu.dcp(argslist) # Need to deepcopy here, since effectively deecopied by other parallelization methods
            outputlist = list(map(_parallel_task, argslist))
        
        elif pname == 'multiprocess': # Main use case
            with mp.Pool(processes=ncpus) as pool:
                outputlist = list(pool.map(_parallel_task, argslist))
        
        elif pname == 'multiprocessing':
            with mpi.Pool(processes=ncpus) as pool:
                outputlist = pool.map(_parallel_task, argslist)
        
        elif pname == 'concurrent.futures':
            with cf.ProcessPoolExecutor(max_workers=ncpus) as pool:
                outputlist = list(pool.map(_parallel_task, argslist))
        
        elif pname == 'thread':
            if not 'nocopy' in parallelizer: # Niche use case of running without deepcopying
                argslist = scu.dcp(argslist) # Also need to deepcopy here
            with cf.ThreadPoolExecutor(max_workers=ncpus) as pool:
                outputlist = list(pool.map(_parallel_task, argslist))

        elif pname == 'custom':
            outputlist = parallelizer(_parallel_task, argslist)
            
        else: # Should be unreachable; exception should have already been caught
            errormsg = f'Invalid parallelizer "{parallelizer}"'
            raise ValueError(errormsg)
        
        return outputlist

    # Actually run the parallelization
    try:
        outputlist = run_parallel(pname, parallelizer, argslist)
        
    # Handle if run outside of __main__ on Windows
    except RuntimeError as E: # pragma: no cover
        if 'freeze_support' in E.args[0]: # For this error, add additional information
            errormsg = '''
 Uh oh! It appears you are trying to run with multiprocessing on Windows outside
 of the __main__ block; please see https://docs.python.org/3/library/multiprocessing.html
 for more information. The correct syntax to use is e.g.

 import sciris as sc

 def my_func(x):
     return

 if __name__ == '__main__':
     sc.parallelize(my_func)
 '''
            raise RuntimeError(errormsg) from E
        else: # For all other runtime errors, raise the original exception
            raise E

    # Tidy up
    if returnpool:
        return pool, outputlist
    else:
        return outputlist


##############################################################################
#%% Helper functions/classes
##############################################################################

class TaskArgs(scu.prettyobj):
        '''
        A class to hold the arguments for the parallel task -- not to be invoked by the user.

        Arguments and ordering must match both ``sc.parallelize()`` and ``sc._parallel_task()`` '''
        def __init__(self, func, index, niters, iterval, iterdict, args, kwargs, 
                     maxcpu, maxmem, interval, embarrassing, callback=None, die=True):
            self.func         = func         # The function being called
            self.index        = index        # The place in the queue
            self.niters       = niters       # The total number of iterations
            self.iterval      = iterval      # The value being iterated (may be None if iterdict is not None)
            self.iterdict     = iterdict     # A dictionary of values being iterated (may be None if iterval is not None)
            self.args         = args         # Arguments passed directly to the function
            self.kwargs       = kwargs       # Keyword arguments passed directly to the function
            self.maxcpu       = maxcpu       # Maximum CPU load (ignored if ncpus is not None in sc.parallelize())
            self.maxmem       = maxmem       # Maximum memory
            self.interval     = interval     # Interval to check load (only used with maxcpu/maxmem)
            self.embarrassing = embarrassing # Whether or not to pass the iterarg to the function (no if it's embarrassing)
            self.callback     = callback     # A function to call after each task finishes
            self.die          = die          # Whether to raise an exception if the child task encounters one
            return


def _parallel_task(taskargs, outputqueue=None):
    ''' Task called by parallelize() -- not to be called directly '''

    # Handle inputs
    func   = taskargs.func
    index  = taskargs.index
    args   = taskargs.args
    kwargs = taskargs.kwargs
    if args   is None: args   = ()
    if kwargs is None: kwargs = {}
    if taskargs.iterval is not None:
        if not isinstance(taskargs.iterval, tuple): # Ensure it's a tuple
            taskargs.iterval = (taskargs.iterval,)
        if not taskargs.embarrassing:
            args = taskargs.iterval + args # If variable name is not supplied, prepend it to args
    if taskargs.iterdict is not None:
        for key,val in taskargs.iterdict.items():
            kwargs[key] = val # Otherwise, include it in kwargs

    # Handle load balancing
    maxcpu = taskargs.maxcpu
    maxmem = taskargs.maxmem
    if maxcpu or maxmem:
        scp.loadbalancer(maxcpu=maxcpu, maxmem=maxmem, index=index, interval=taskargs.interval)

    # Call the function!
    try:
        output = func(*args, **kwargs)
    except Exception as E:
        if taskargs.die: # Usual case, raise an exception and stop
            raise E
        else: # Alternatively, keep going and just let this trial fail
            warnmsg = f'sc.parallelize(): Task {index} failed, but die=False so continuing.\n{scu.traceback()}'
            warnings.warn(warnmsg, category=RuntimeWarning, stacklevel=2)
            output = E
    
    # Handle callback, if present
    if taskargs.callback:
        data = dict(index=index, niters=taskargs.niters, args=args, kwargs=kwargs, output=output)
        taskargs.callback(data)

    # Handle output
    if outputqueue:
        outputqueue.put((index,output))
        return
    else:
        return output