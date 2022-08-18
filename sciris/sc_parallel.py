'''
Parallelization functions, allowing multiprocessing to be used simply.

NB: Uses ``multiprocess`` instead of ``multiprocessing`` under the hood for
broadest support  across platforms (e.g. Jupyter notebooks).

Highlights:
    - :func:`parallelize`: as-easy-as-possible parallelization
'''

import numpy as np
import multiprocess as mp
import multiprocessing as mpi
import concurrent.futures as cf
from functools import partial
import textwrap
import warnings
from . import sc_utils as scu
from . import sc_profiling as scp


##############################################################################
#%% Parallelization functions
##############################################################################

__all__ = ['parallelize', 'parallelcmd', 'parallel_progress']


def parallelize(func, iterarg=None, iterkwargs=None, args=None, kwargs=None, ncpus=None, maxcpu=None, maxmem=None,
                interval=None, parallelizer='multiprocess', serial=False, returnpool=False, **func_kwargs):
    '''
    Execute a function in parallel.

    Most simply, ``sc.parallelize()`` acts as an shortcut for using ``multiprocess.Pool()``.
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

    Note: the default parallelizer ``"multiprocess"`` uses ``dill`` for pickling, so
    is the most versatile (e.g., it can pickle non-top-level functions). However,
    it is also the slowest for passing large amounts of data. If you don't need
    ``dill``, you might get better performance using ``"concurrent.futures"`` as
    the parallelizer.

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
        parallelizer (str/func)  : parallelization function; default 'multiprocess' (other choices are 'concurrent.futures', 'multiprocessing', or user-supplied; see example below)
        serial       (bool)      : whether to skip parallelization run in serial (useful for debugging)
        returnpool   (bool)      : whether to return the process pool as well as the results
        func_kwargs  (dict)      : merged with kwargs (see above)

    Returns:
        List of outputs from each process


    **Example 1 -- simple usage as a shortcut to multiprocessing.map()**::

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


    **Note**: to use on Windows, parallel calls must contained with an ``if __name__ == '__main__'`` block.

    For example::

        import sciris as sc

        def f(x,y):
            return x*y

        if __name__ == '__main__':
            results = sc.parallelize(func=f, iterarg=[(1,2),(2,3),(3,4)])
            print(results)

    | New in version 1.1.1: "serial" argument.
    | New in version 2.0.0: changed default parallelizer from ``multiprocess.Pool`` to ``concurrent.futures.ProcessPoolExecutor``;
    replaced ``maxload`` with ``maxcpu``/``maxmem``; added ``returnpool`` argument
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
        taskargs = TaskArgs(func=func, index=index, iterval=iterval, iterdict=iterdict,
                            args=args, kwargs=kwargs, maxcpu=maxcpu, maxmem=maxmem,
                            interval=interval, embarrassing=embarrassing)
        argslist.append(taskargs)

    # Actually run the parallelization
    try:

        # Set up the run
        pool = None
        if ncpus is not None:
            ncpus = min(ncpus, len(argslist)) # Don't use more CPUs than there are things to process

        # Run in serial for debugging
        if serial:
            outputlist = list(map(_parallel_task, argslist))

         # Standard usage: use the default map() function
        elif parallelizer is None or scu.isstring(parallelizer):

            # Map parallelizer to consistent choices
            default = 'multiprocess'
            mapping = {
                None: default,
                'default': default,
                'multiprocess': 'multiprocess',
                'multiprocessing': 'multiprocessing',
                'concurrent.futures': 'concurrent.futures',
                'concurrent': 'concurrent.futures',
                'thread': 'thread',
                'threadpool': 'thread',
            }
            try:
                parallelizer = mapping[parallelizer]
            except:
                errormsg = f'Parallelizer "{parallelizer}" not found: must be one of {scu.strjoin(mapping.keys())}'
                raise scu.KeyNotFoundError(errormsg)

            # Choose which parallelizer to use
            if parallelizer == 'multiprocess': # Main use case
                with mp.Pool(processes=ncpus) as pool:
                    outputlist = pool.map(_parallel_task, argslist)
            elif parallelizer == 'multiprocessing':
                with mpi.Pool(processes=ncpus) as pool:
                    outputlist = pool.map(_parallel_task, argslist)
            elif parallelizer == 'concurrent.futures':
                with cf.ProcessPoolExecutor(max_workers=ncpus) as pool:
                    outputlist = list(pool.map(_parallel_task, argslist))
            elif parallelizer == 'thread':
                with cf.ThreadPoolExecutor(max_workers=ncpus) as pool:
                    outputlist = list(pool.map(_parallel_task, argslist))
            else: # Should be unreachable; exception should have already been caught
                errormsg = f'Invalid parallelizer "{parallelizer}"'
                raise ValueError(errormsg)

        # Use a custom parallelization method
        else:
            outputlist = parallelizer(_parallel_task, argslist)

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


def parallelcmd(cmd=None, parfor=None, returnval=None, maxcpu=None, maxmem=None, interval=None, **kwargs):
    '''
    A function to parallelize any block of code. Note: this is intended for quick
    prototyping; since it uses exec(), it is not recommended for use in production
    code.

    Args:
        cmd       (str):   a string representation of the code to be run in parallel
        parfor    (dict):  a dictionary of lists of the variables to loop over
        returnval (str):   the name of the output variable
        maxcpu    (float): maximum CPU load; used by ``sc.loadbalancer()``
        maxmem    (float): maximum fraction of virtual memory (RAM); used by ``sc.loadbalancer()``
        interval  (float): the time delay to poll to see if load is OK,  used in ``sc.loadbalancer()``
        kwargs    (dict):  variables to pass into the code

    **Example**::

        const = 4
        parfor = {'val':[3,5,9]}
        returnval = 'result'
        cmd = """
        newval = val+const
        result = newval**2
        """
        results = sc.parallelcmd(cmd=cmd, parfor=parfor, returnval=returnval, const=const)

    New in version 2.0.0: replaced ``maxload`` with ``maxcpu``/``maxmem``; automatically de-indent the command
    '''

    # Handle maxload
    maxload = kwargs.pop('maxload', None)
    if maxload is not None: # pragma: no cover
        maxcpu = maxload
        warnmsg = 'sc.loadbalancer() argument "maxload" has been renamed "maxcpu" as of v2.0.0'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)

    # Deindent the command
    cmd = textwrap.dedent(cmd)

    # Create queue
    nfor = len(list(parfor.values())[0])
    outputqueue = mp.Queue()
    outputlist = np.empty(nfor, dtype=object)
    processes = []
    for i in range(nfor):
        args = (cmd, parfor, returnval, i, outputqueue, maxcpu, maxmem, interval, kwargs)
        prc = mp.Process(target=_parallelcmd_task, args=args)
        prc.start()
        processes.append(prc)
    for i in range(nfor):
        _i,returnval = outputqueue.get()
        outputlist[_i] = returnval
    for prc in processes:
        prc.join() # Wait for them to finish

    outputlist = outputlist.tolist()

    return outputlist


def parallel_progress(fcn, inputs, num_workers=None, show_progress=True, initializer=None): # pragma: no cover
    """
    Run a function in parallel with a optional single progress bar

    The result is essentially equivalent to::

        >>> list(map(fcn, inputs))

    But with execution in parallel and with a single progress bar being shown.

    Args:
        fcn (function): Function object to call, accepting one argument, OR a function with zero arguments in which case inputs should be an integer
        inputs (list): A collection of inputs that will each be passed to the function OR a number, if the fcn() has no input arguments
        num_workers (int): Number of processes, defaults to the number of CPUs
        show_progress (bool): Whether to show a progress bar
        initializer (func): A function that each worker process will call when it starts

    Returns:
        A list of outputs

    New in version 1.0.0.
    """
    try:
        from tqdm import tqdm
    except ModuleNotFoundError as E:
        errormsg = 'Module tqdm not found; please install with "pip install tqdm"'
        raise ModuleNotFoundError(errormsg) from E

    pool = mp.pool.Pool(num_workers, initializer=initializer)

    results = [None]
    if scu.isnumber(inputs):
        results *= inputs
        pbar = tqdm(total=inputs) if show_progress else None
    else:
        results *= len(inputs)
        pbar = tqdm(total=len(inputs)) if show_progress else None

    def callback(result, idx):
        results[idx] = result
        if show_progress:
            pbar.update(1)

    if scu.isnumber(inputs):
        for i in range(inputs):
            pool.apply_async(fcn, callback=partial(callback, idx=i))
    else:
        for i, x in enumerate(inputs):
            pool.apply_async(fcn, args=(x,), callback=partial(callback, idx=i))

    pool.close()
    pool.join()

    if show_progress:
        pbar.close()

    return results



##############################################################################
#%% Helper functions/classes
##############################################################################

class TaskArgs(scu.prettyobj):
        '''
        A class to hold the arguments for the parallel task -- not to be invoked by the user.

        Arguments and ordering must match both ``sc.parallelize()`` and ``sc._parallel_task()`` '''
        def __init__(self, func, index, iterval, iterdict, args, kwargs, maxcpu, maxmem, interval, embarrassing):
            self.func         = func         # The function being called
            self.index        = index        # The place in the queue
            self.iterval      = iterval      # The value being iterated (may be None if iterdict is not None)
            self.iterdict     = iterdict     # A dictionary of values being iterated (may be None if iterval is not None)
            self.args         = args         # Arguments passed directly to the function
            self.kwargs       = kwargs       # Keyword arguments passed directly to the function
            self.maxcpu       = maxcpu       # Maximum CPU load (ignored if ncpus is not None in sc.parallelize())
            self.maxmem       = maxmem       # Maximum memory
            self.interval     = interval     # Interval to check load (only used with maxcpu/maxmem)
            self.embarrassing = embarrassing # Whether or not to pass the iterarg to the function (no if it's embarrassing)
            return


def _parallel_task(taskargs, outputqueue=None):
    ''' Task called by parallelize() -- not to be called directly '''

    # Handle inputs
    taskargs = scu.dcp(taskargs)
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

    # Call the function
    output = func(*args, **kwargs)

    # Handle output
    if outputqueue:
        outputqueue.put((index,output))
        return
    else:
        return output


def _parallelcmd_task(_cmd, _parfor, _returnval, _i, _outputqueue, _maxcpu, _maxmem, _interval, _kwargs): # pragma: no cover # No coverage since pickled
    '''
    The task to be executed by ``sc.parallelcmd()``. All internal variables start with
    underscores to avoid possible collisions in the ``exec()`` statements. Not to be called
    directly.
    '''
    if _maxcpu or _maxmem:
        scp.loadbalancer(maxcpu=_maxcpu, maxmem=_maxmem, index=_i, interval=_interval)

    # Set the loop variables
    for _key in _parfor.keys():
        _thisval = _parfor[_key][_i] # analysis:ignore
        exec(f'{_key} = _thisval') # Set the value of this variable

    # Set the keyword arguments
    for _key in _kwargs.keys():
        _thisval = _kwargs[_key] # analysis:ignore
        exec(f'{_key} = _thisval') # Set the value of this variable

    # Calculate the command
    try:
        exec(_cmd) # The meat of the matter!
    except Exception as E:
        print(f'WARNING, parallel task failed:\n{str(E)}')
        exec(f'{_returnval} = None')

    # Append results
    _outputqueue.put((_i,eval(_returnval)))

    return