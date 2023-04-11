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
from . import sc_odict as sco
from . import sc_printing as scp
from . import sc_profiling as scpro


##############################################################################
#%% Parallelization class
##############################################################################

__all__ = ['Parallel', 'parallelize']

# Define a common error message
freeze_support_error = '''
Uh oh! It appears you are trying to run with multiprocessing on Windows outside
of the __main__ block; please see https://docs.python.org/3/library/multiprocessing.html
for more information. The correct syntax to use is e.g.

import sciris as sc

def my_func(x):
    return

if __name__ == '__main__':
    sc.parallelize(my_func)
'''


class Parallel:
    '''
    Parallelization manager
    
    # TODO!!!!!
    
    '''
    def __init__(self, func, iterarg=None, iterkwargs=None, args=None, kwargs=None, ncpus=None, 
                 maxcpu=None, maxmem=None, interval=None, parallelizer=None, serial=False, 
                 returnpool=False, progress=False, callback=None, label=None, use_async=True,
                 die=True, **func_kwargs):
        
        # Store input arguments
        self.func         = func
        self.iterarg      = iterarg
        self.iterkwargs   = iterkwargs
        self.args         = args
        self.kwargs       = scu.mergedicts(kwargs, func_kwargs)
        self.ncpus        = ncpus
        self.maxcpu       = maxcpu
        self.maxmem       = maxmem
        self.interval     = interval
        self.parallelizer = parallelizer
        self.serial       = serial
        self.returnpool   = returnpool
        self.progress     = progress
        self.callback     = callback
        self.label        = label
        self.use_async    = use_async
        self.die          = die
        
        # Pre-allocate other results
        self.njobs       = None
        self.embarrassing = None
        self.argslist     = None
        self.outputlist   = None
        self.already_run  = False
        
        # Additional setup
        self.configure()
        return
    
    
    def __repr__(self):
        # Set label string
        labelstr = f'"{self.label}"' if self.label else '<no label>'
        string   = f'Parallel({labelstr}; {start} to {end}; pop: {pop_size:n} {pop_type}; epi: {results})'

        return 'TODO'
    
    
    def disp(self):
        ''' Display the full representation of the object '''
        return scp.pr(self)
        
    
    def configure(self):
        '''
        Perform all configuration steps; this can safely be called after initialization
        '''
        self.set_defaults()
        self.set_balancer()
        self.validate_iterable()
        self.make_argslist()
        return
    
    
    def set_defaults(self):
        ''' Define defaults for parallelization '''
        self.defaults = sco.objdict()
        
        # Define default parallelizers
        self.defaults.fast   = 'concurrent.futures'
        self.defaults.robust = 'multiprocess'
        
        # Map parallelizer to consistent choices
        self.defaults.mapping = {
            None                 : self.defaults.robust,
            'default'            : self.defaults.robust,
            'robust'             : self.defaults.robust,
            'fast'               : self.defaults.fast,
            'serial'             : 'serial',
            'serial-copy'        : 'serial',
            'concurrent.futures' : 'concurrent.futures',
            'concurrent'         : 'concurrent.futures',
            'multiprocess'       : 'multiprocess',
            'multiprocessing'    : 'multiprocessing',
            'thread'             : 'thread',
            'threadpool'         : 'thread',
            'thread-copy'        : 'thread',
        }
        
        return
    
    
    def set_balancer(self):
        ''' Configure load balancer settings '''
        maxload = self.kwargs.pop('maxload', None)
        if maxload is not None: # pragma: no cover
            self.maxcpu = maxload
            warnmsg = 'sc.loadbalancer() argument "maxload" has been renamed "maxcpu" as of v2.0.0'
            warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)
        if self.ncpus is None and self.maxcpu is None:
            self.ncpus = scpro.cpu_count()
        if self.ncpus is not None and self.ncpus < 1: # Less than one, treat as a fraction of total
            self.ncpus = int(scpro.cpu_count()*self.ncpus)
        return
    
    
    def validate_iterable(self):
        ''' Handle iterarg and iterkwargs '''
        iterarg = self.iterarg
        iterkwargs = self.iterkwargs
        njobs = 0
        embarrassing = False # Whether or not it's an embarrassingly parallel optimization
        
        # Check that only one was provided
        if iterarg is not None and iterkwargs is not None: # pragma: no cover
            errormsg = 'You can only use one of iterarg or iterkwargs as your iterable, not both'
            raise ValueError(errormsg)
        
        # Validate iterarg
        if iterarg is not None:
            if not(scu.isiterable(iterarg)):
                try:
                    iterarg = np.arange(iterarg)
                    embarrassing = True
                except Exception as E: # pragma: no cover
                    errormsg = f'Could not understand iterarg "{iterarg}": not iterable and not an integer: {str(E)}'
                    raise TypeError(errormsg)
            njobs = len(iterarg)
            
        # Validate iterkwargs
        if iterkwargs is not None: 
            
            if isinstance(iterkwargs, dict): # It's a dict of lists, e.g. {'x':[1,2,3], 'y':[2,3,4]}
                for key,val in iterkwargs.items():
                    if not scu.isiterable(val): # pragma: no cover
                        errormsg = f'iterkwargs entries must be iterable, not {type(val)}'
                        raise TypeError(errormsg)
                    if not njobs:
                        njobs = len(val)
                    else:
                        if len(val) != njobs: # pragma: no cover
                            errormsg = f'All iterkwargs iterables must be the same length, not {njobs} vs. {len(val)}'
                            raise ValueError(errormsg)
            
            elif isinstance(iterkwargs, list): # It's a list of dicts, e.g. [{'x':1, 'y':2}, {'x':2, 'y':3}, {'x':3, 'y':4}]
                njobs = len(iterkwargs)
                for item in iterkwargs:
                    if not isinstance(item, dict): # pragma: no cover
                        errormsg = f'If iterkwargs is a list, each entry must be a dict, not {type(item)}'
                        raise TypeError(errormsg)
            
            else: # pragma: no cover
                errormsg = f'iterkwargs must be a dict of lists, a list of dicts, or None, not {type(iterkwargs)}'
                raise TypeError(errormsg)
    
        # Final error checking
        if njobs == 0:
            errormsg = 'Nothing found to parallelize: please supply an iterarg, iterkwargs, or both'
            raise ValueError(errormsg)
        else:
            self.njobs = njobs
            self.embarrassing = embarrassing
            
        return
    
    
    def make_argslist(self):
        ''' Construct argument list '''
        iterarg = self.iterarg
        iterkwargs = self.iterkwargs
        argslist = []
        for index in range(self.njobs):
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
            taskargs = TaskArgs(func=self.func, index=index, njobs=self.njobs, iterval=iterval, iterdict=iterdict,
                                args=self.args, kwargs=self.kwargs, maxcpu=self.maxcpu, maxmem=self.maxmem,
                                interval=self.interval, embarrassing=self.embarrassing, callback=self.callback, 
                                die=self.die)
            argslist.append(taskargs)
        
        self.argslist = argslist
        return
    
        
    def run_parallel(self):
        ''' Choose how to run in parallel, and do it '''
        
        # Shorten common variables
        ncpus = self.ncpus
        parallelizer = self.parallelizer
        argslist = self.argslist
        
        # Handle number of CPUs
        if ncpus is not None:
            ncpus = min(ncpus, len(self.argslist)) # Don't use more CPUs than there are things to process
        if self.serial: # This is a separate keyword argument, but make it consistent
            self.parallelizer = 'serial'
            
        # Handle the choice of parallelizer
        if self.parallelizer is None or scu.isstring(self.parallelizer):
            try:
                pname = self.defaults.mapping[self.parallelizer]
            except:
                errormsg = f'Parallelizer "{self.parallelizer}" not found: must be one of {scu.strjoin(self.defaults.mapping.keys())}'
                raise scu.KeyNotFoundError(errormsg)
        else:
            pname = 'custom' # If a custom parallelizer is provided
        
        # Choose which parallelizer to use
        pool = None
        if pname == 'serial':
            if 'copy' in parallelizer: # Niche use case of running without deepcopying
                argslist = scu.dcp(argslist) # Need to deepcopy here, since effectively deecopied by other parallelization methods
            outputlist = list(map(_task, argslist))
        
        elif pname == 'multiprocess': # Main use case
            with mp.Pool(processes=ncpus) as pool:
                outputlist = list(pool.map(_task, argslist))
        
        elif pname == 'multiprocessing':
            with mpi.Pool(processes=ncpus) as pool:
                outputlist = pool.map(_task, argslist)
        
        elif pname == 'concurrent.futures':
            with cf.ProcessPoolExecutor(max_workers=ncpus) as pool:
                outputlist = list(pool.map(_task, argslist))
        
        elif pname == 'thread':
            if 'copy' in parallelizer: # Niche use case of running without deepcopying
                argslist = scu.dcp(argslist) # Also need to deepcopy here
            with cf.ThreadPoolExecutor(max_workers=ncpus) as pool:
                outputlist = list(pool.map(_task, argslist))

        elif pname == 'custom':
            outputlist = parallelizer(_task, argslist)
            
        else: # Should be unreachable; exception should have already been caught
            errormsg = f'Invalid parallelizer "{parallelizer}"'
            raise ValueError(errormsg)
        
        # Store the pool; do not store the output list here
        self.pool = pool
        
        return outputlist
    
    
    def run(self):
        ''' Actually run the parallelization '''
        try:
            self.outputlist = self.run_parallel()
            
        # Handle if run outside of __main__ on Windows
        except RuntimeError as E: # pragma: no cover
            if 'freeze_support' in E.args[0]: # For this error, add additional information
                raise RuntimeError(freeze_support_error) from E
            else: # For all other runtime errors, raise the original exception
                raise E
    
        # Tidy up
        return self.outputlist
    


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
        - ``'serial'``, ``'serial-copy'``: no parallelization (single-threaded); with "-copy", force pickling
        - ``'thread'``', ``'threadpool'``', ``'thread-copy'``': thread- rather than process-based parallelization ("-copy" as above)
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
    | New in version 2.2.0: new Parallel class; propagated "die" to tasks
    '''
    # Create the parallel instance
    P = Parallel(func, iterarg=iterarg, iterkwargs=iterkwargs, args=args, kwargs=kwargs, 
                 ncpus=ncpus, maxcpu=maxcpu, maxmem=maxmem, interval=interval, 
                 parallelizer=parallelizer, serial=serial, returnpool=returnpool, 
                 progress=progress, callback=callback, die=die, **func_kwargs)
    
    # Run it
    output = P.run()
    return output



##############################################################################
#%% Helper functions/classes
##############################################################################

class TaskArgs(scu.prettyobj):
        '''
        A class to hold the arguments for the parallel task -- not to be invoked by the user.

        Arguments must match both ``sc.parallelize()`` and ``sc._task()``
        '''
        def __init__(self, func, index, njobs, iterval, iterdict, args, kwargs, 
                     maxcpu, maxmem, interval, embarrassing, callback=None, die=True):
            self.func         = func         # The function being called
            self.index        = index        # The place in the queue
            self.njobs        = njobs        # The total number of iterations
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


def _task(taskargs, outputqueue=None):
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
        scpro.loadbalancer(maxcpu=maxcpu, maxmem=maxmem, index=index, interval=taskargs.interval)

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
        data = dict(index=index, njobs=taskargs.njobs, args=args, kwargs=kwargs, output=output)
        taskargs.callback(data)

    # Handle output
    if outputqueue:
        outputqueue.put((index,output))
        return
    else:
        return output