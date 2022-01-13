'''
Parallelization functions, allowing multiprocessing to be used simply.

NB: Uses ``multiprocess`` instead of ``multiprocessing`` under the hood for
broadest support  across platforms (e.g. Jupyter notebooks).

Highlights:
    - ``sc.parallelize()``: as-easy-as-possible parallelization
    - ``sc.loadbalancer()``: very basic load balancer
'''

import time
import psutil
import multiprocess as mp
import numpy as np
from functools import partial
from . import sc_utils as scu


##############################################################################
#%% Parallelization functions
##############################################################################

__all__ = ['cpu_count', 'loadbalancer', 'parallelize', 'parallelcmd', 'parallel_progress']


def cpu_count():
    ''' Alias to mp.cpu_count() '''
    return mp.cpu_count()


def loadbalancer(maxload=None, index=None, interval=None, maxtime=None, label=None, verbose=True):
    '''
    A little function to delay execution while CPU load is too high -- a very simple load balancer.

    Arguments:
        maxload:  the maximum load to allow for the task to still start (default 0.5)
        index:    the index of the task -- used to start processes asynchronously (default None)
        interval: the time delay to poll to see if CPU load is OK (default 5 seconds)
        maxtime:  maximum amount of time to wait to start the task (default 36000 seconds (10 hours))
        label:    the label to print out when outputting information about task delay or start (default None)
        verbose:  whether or not to print information about task delay or start (default True)

    **Examples**::

        loadbalancer() # Simplest usage -- delay while load is >80%
        for nproc in processlist: loadbalancer(maxload=0.9, index=nproc) # Use a maximum load of 90%, and stagger the start by process number

    Version: 2018nov01
    '''

    # Set up processes to start asynchronously
    if maxload  is None: maxload = 0.8
    if interval is None: interval = 1.0
    if maxtime  is None: maxtime = 36000
    if label    is None: label = ''
    else: label += ': '
    if index is None:
        pause = np.random.rand()*interval
        index = ''
    else:
        pause = index*interval
    if maxload>1: maxload/100. # If it's >1, assume it was given as a percent
    if not maxload>0:
        return # Return immediately if no max load
    else:
        time.sleep(pause) # Give it time to asynchronize

    # Loop until load is OK
    toohigh = True # Assume too high
    count = 0
    maxcount = maxtime/float(interval)
    while toohigh and count<maxcount:
        count += 1
        currentload = psutil.cpu_percent(interval=0.1)/100. # If interval is too small, can give very inaccurate readings
        if currentload>maxload:
            if verbose: print(label+f'CPU load too high ({currentload:0.2f}/{maxload:0.2f}); process {index} queued {count} times')
            time.sleep(interval*2*np.random.rand()) # Sleeps for an average of refresh seconds, but do it randomly so you don't get locking
        else:
            toohigh = False
            if verbose: print(label+f'CPU load fine ({currentload:0.2f}/{maxload:0.2f}), starting process {index} after {count} tries')
    return


def parallelize(func, iterarg=None, iterkwargs=None, args=None, kwargs=None, ncpus=None, maxload=None, interval=None, parallelizer=None, serial=False, **func_kwargs):
    '''
    Main method for parallelizing a function.

    Most simply, acts as an shortcut for using multiprocess.Pool(). However, this
    function can also iterate over more complex arguments.

    Either or both of iterarg or iterkwargs can be used. iterarg can be an iterable or an integer;
    if the latter, it will run the function that number of times and not pass the argument to the
    function (which may be useful for running "embarrassingly parallel" simulations). iterkwargs
    is a dict of iterables; each iterable must be the same length (and the same length of iterarg,
    if it exists), and each dict key will be used as a kwarg to the called function. Any other kwargs
    passed to parallelize() will also be passed to the function.

    This function can either use a fixed number of CPUs or allocate dynamically based
    on load. If ncpus is None and maxload is None, then it will use the number of CPUs
    returned by multiprocessing; if ncpus is not None, it will use the specified number of CPUs;
    if ncpus is None and maxload is not None, it will allocate the number of CPUs dynamically.

    Args:
        func (function): the function to parallelize
        iterarg (list): the variable(s) to provide to each process (see examples below)
        iterkwargs (dict): another way of providing variables to each process (see examples below)
        args (list): positional arguments, the same for all processes
        kwargs (dict): keyword arguments, the same for all processes
        ncpus (int or float): number of CPUs to use (if <1, treat as a fraction of the total available; if None, use loadbalancer)
        maxload (float): maximum CPU load to use (not used if ncpus is specified)
        interval (float): number of seconds to pause between starting processes for checking load (not used if ncpus is specified)
        parallelizer (func): alternate parallelization function instead of multiprocess.Pool.map()
        serial (bool): whether to skip parallelization run in serial (useful for debugging)
        func_kwargs (dict): merged with kwargs (see above)

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

        xylist1 = sc.parallelize(myfunc, kwargs={'x':3, 'y':8}, iterarg=range(5), maxload=0.8, interval=0.2) # Use kwargs dict
        xylist2 = sc.parallelize(myfunc, x=5, y=10, iterarg=[0,1,2]) # Supply kwargs directly

        for p,xylist in enumerate([xylist1, xylist2]):
            pl.subplot(2,1,p+1)
            for i,xy in enumerate(reversed(xylist)):
                pl.scatter(xy[0], xy[1], label='Run %i'%i)
            pl.legend()

    **Example 5 -- using a custom parallelization function**::

        def f(x,y):
            return [x]*y

        import multiprocessing as mp
        multipool = mp.Pool(processes=2)
        results = sc.parallelize(f, iterkwargs=dict(x=[1,2,3], y=[4,5,6]), parallelizer=multipool.map)
        multipool.close() # NB, in this case, close and join are not strictly required
        multipool.join()

    **Note**: to use on Windows, parallel calls must contained with an ``if __name__ == '__main__'`` block.

    For example::

        import sciris as sc

        def f(x,y):
            return x*y

        if __name__ == '__main__':
            results = sc.parallelize(func=f, iterarg=[(1,2),(2,3),(3,4)])
            print(results)

    New in version 1.1.1: "serial" argument.
    '''
    # Handle maxload
    if ncpus is None and maxload is None:
        ncpus = mp.cpu_count()
    if ncpus is not None and ncpus < 1: # Less than one, treat as a fraction of total
        ncpus = int(mp.cpu_count()*ncpus)

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
        taskargs = TaskArgs(func, index, iterval, iterdict, args, kwargs, maxload, interval, embarrassing)
        argslist.append(taskargs)

    # Run simply using map -- no advantage here to using Process/Queue
    try:
        if serial: # Run in serial
            outputlist = list(map(_parallel_task, argslist))
        elif parallelizer is None: # Standard usage: use the default map() function
            if ncpus is not None:
                ncpus = min(ncpus, len(argslist)) # Don't use more CPUs than there are things to process
            multipool = mp.Pool(processes=ncpus)
            outputlist = multipool.map(_parallel_task, argslist)
            multipool.close()
            multipool.join()
        else: # Use a custom parallelization method
            outputlist = parallelizer(_parallel_task, argslist)
    except RuntimeError as E: # pragma: no cover # Handle if run outside of __main__ on Windows
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

    return outputlist


def parallelcmd(cmd=None, parfor=None, returnval=None, maxload=None, interval=None, **kwargs):
    '''
    A function to parallelize any block of code. Note: this is intended for quick
    prototyping; since it uses exec(), it is not recommended for use in production
    code.

    Args:
        cmd (str): a string representation of the code to be run in parallel
        parfor (dict): a dictionary of lists of the variables to loop over
        returnval (str): the name of the output variable
        maxload (float): the maximum CPU load, used in ``sc.loadbalancer()``
        kwargs (dict): variables to pass into the code

    **Example**::

        const = 4
        parfor = {'val':[3,5,9]}
        returnval = 'result'
        cmd = """newval = val+const # Note that this can't be indented
        result = newval**2
        """
        results = sc.parallelcmd(cmd=cmd, parfor=parfor, returnval=returnval, const=const)

    Version: 2018nov01
    '''

    nfor = len(list(parfor.values())[0])
    outputqueue = mp.Queue()
    outputlist = np.empty(nfor, dtype=object)
    processes = []
    for i in range(nfor):
        prc = mp.Process(target=_parallelcmd_task, args=(cmd, parfor, returnval, i, outputqueue, maxload, interval, kwargs))
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

    Returns:
        A list of outputs

    New in version 1.0.0.
    """
    from multiprocess import pool
    try:
        from tqdm import tqdm
    except ModuleNotFoundError as E:
        errormsg = 'Module tqdm not found; please install with "pip install tqdm"'
        raise ModuleNotFoundError(errormsg) from E

    pool = pool.Pool(num_workers, initializer=initializer)

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
        def __init__(self, func, index, iterval, iterdict, args, kwargs, maxload, interval, embarrassing):
            self.func         = func         # The function being called
            self.index        = index        # The place in the queue
            self.iterval      = iterval      # The value being iterated (may be None if iterdict is not None)
            self.iterdict     = iterdict     # A dictionary of values being iterated (may be None if iterval is not None)
            self.args         = args         # Arguments passed directly to the function
            self.kwargs       = kwargs       # Keyword arguments passed directly to the function
            self.maxload      = maxload      # Maximum CPU load (ignored if ncpus is not None in parallelize()
            self.interval     = interval     # Interval to check load (only used with maxload)
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
    maxload = taskargs.maxload
    if maxload:
        loadbalancer(maxload=maxload, index=index, interval=taskargs.interval)

    # Call the function
    output = func(*args, **kwargs)

    # Handle output
    if outputqueue:
        outputqueue.put((index,output))
        return
    else:
        return output


def _parallelcmd_task(_cmd, _parfor, _returnval, _i, _outputqueue, _maxload, _interval, _kwargs): # pragma: no cover # No coverage since pickled
    '''
    The task to be executed by ``sc.parallelcmd()``. All internal variables start with
    underscores to avoid possible collisions in the ``exec()`` statements. Not to be called
    directly.
    '''
    loadbalancer(maxload=_maxload, index=_i, interval=_interval)

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