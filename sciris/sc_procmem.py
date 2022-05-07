'''
Memory monitoring function
NB: Uses ``multiprocess`` instead of ``multiprocessing`` under the hood for
broadest support  across platforms (e.g. Jupyter notebooks).

Highlights:
    - ``sc.parallelize()``: as-easy-as-possible parallelization
    - ``sc.loadbalancer()``: very basic load balancer
'''
import contextlib
import time
import psutil
import tracemalloc
import asyncio
import numpy as np
from functools import partial

import sciris
from . import sc_utils as scu

##############################################################################
# %% Parallelization functions
##############################################################################


__all__ = ['resourcelimit', 'limit_memory', 'limit_malloc', 'async_limit_memory']


@contextlib.contextmanager
def limit_memory(max_size=0.5, die=True, verbose=True):
    """

    :param max_size:
    :param die:
    :param verbose:
    :return:
    """
    # Exclusion file patterns
    # start
    snapshot1 = _take_snapshot()

    yield

    snapshot2 = _take_snapshot()
    print(snapshot2)

    #snapshot  = snapshot2 - snapshot1
    try:
        current_size = snapshot2

        if current_size  > max_size:
            if die:
                raise AttributeError(f"Memory usage exceeded the threshold: "
                                     f"{current_size} > {max_size}")
            else:
                pass
        else:
            if verbose: print(f"Memory usage is fine ({current_size:0.2f}/{max_size:0.2f})")

    finally:
        return


def _take_snapshot():
    snapshot = psutil.virtual_memory().percent / 100.0
    return snapshot


class async_limit_memory:
    def __init__(self, max_mem, die=True, verbose=True):
        self.max_mem = max_mem
        self.die = die
        self.verbose = verbose

    async def __aenter__(self):
        self.enter_mem = self._take_snapshot() # returns percentage memory via psutil
        self.exit_mem   = 0.0
        return self


    async def __aexit__(self):
        # See percentege of memory used at the end
        self.exit_mem =  self._take_snapshot()
        if self.exit_mem > self.max_mem:
             print(f"Memory usage exceeded the threshold: {self.exit_mem} > {self.max_mem}")
             # callback to slack or other

    async def _check_mem(self):
           current_mem = self._take_snapshot()
           while current_mem < self.max_mem:
               # Do something about this hardcoded value
               await asyncio.sleep(1.0)
               current_mem = self._take_snapshot()
               if self.verbose:
                   print(f"memory status internal -> {current_mem}")
           if self.die:
                raise RuntimeError(f"Memory usage exceeded the threshold: {current_mem} > {self.max_mem}")
           return current_mem

    def _take_snapshot(self):
        snapshot = psutil.virtual_memory().percent / 100.0
        return snapshot


def resourcelimit(maxmem=None, die=True, interval=None, maxtime=None, label=None, index=None, die_callback=None, verbose=True, **callback_kwargs): # pragma: no cover
    """
    Check if we are using more than maxmem, die if true and post to slack

    Args:
        maxmem:  the maximum memory that a task can use


    Returns:

    """

    if maxmem is None: maxmem = 0.8
    if interval is None: interval = 1.0
    if maxtime  is None: maxtime = 36000
    if label is None:
        label = ''
    else:
        label += ': '
    if index is not None:
        pause = index*interval
    else:
        pause = np.random.rand()*interval
        index = ''
    if maxmem>1: maxmem/100. # If it's >1, assume it was given as a percent

    #What to do if we stop execution
    if die_callback is None:
        callback = die_callback
    else:
        callback = die_callback(**callback_kwargs)
    if not maxmem>0:
        return # Return immediately if no max load
    else:
        time.sleep(pause) # Give it time to asynchronize

    # Loop until load is OK
    toohigh = True  # Assume too high
    count = 0
    maxcount = maxtime / float(interval)

    while toohigh and count < maxcount:
        count += 1
        currentmem = psutil.virtual_memory().percent / 100.0
        if currentmem > maxmem:
            if verbose: print(label + f'Memory usage too high ({currentmem:0.2f}/{maxmem:0.2f}).')
            if die:
                errormsg = f'Error: Stopping process {index}. Using more memory than requested ({currentmem:0.2f}/{maxmem:0.2f}). Callback to {callback}'
                raise RuntimeError(errormsg)# Do something
            else:
                time.sleep(interval * 2 * np.random.rand())  # Sleeps for an average of refresh seconds, but do it randomly so you don't get locking
        else:
            toohigh = False
            if verbose: print(label + f'Memory usage is fine ({currentmem:0.2f}/{maxmem:0.2f}), starting process {index} after {count} tries')
    return



import contextlib
import logging
import tracemalloc


LOG = logging.getLogger(__name__)

TRACE_FILTERS = (
    tracemalloc.Filter(False, '<frozen importlib._bootstrap>'),
    tracemalloc.Filter(False, '<frozen importlib._bootstrap_external>'),
    tracemalloc.Filter(False, __file__),
    tracemalloc.Filter(False, tracemalloc.__file__),
    tracemalloc.Filter(False, '<unknown>'),
)


@contextlib.contextmanager
def limit_malloc(size):
    """
    https://gist.github.com/adalekin/2b4219808ac72cafda6cff896739a11d
    """
    TRACE_FILTERS = (
        tracemalloc.Filter(False, __file__),
        tracemalloc.Filter(False, tracemalloc.__file__),
        tracemalloc.Filter(False, '<unknown>'),
    )

    if not tracemalloc.is_tracing():
        tracemalloc.start()

    snapshot1 = tracemalloc.take_snapshot()

    yield

    snapshot2 = tracemalloc.take_snapshot().filter_traces(TRACE_FILTERS)
    snapshot1 = snapshot1.filter_traces(TRACE_FILTERS)

    snapshot = snapshot2.compare_to(snapshot1, 'lineno')

    try:
        current_size = sum(stat.size_diff for stat in snapshot)

        if current_size > size:
            for stat in snapshot[]:
                print(stat)
            raise AttributeError(f'Memory usage exceeded the threshold: '
                                 f'{current_size} > size')
    finally:
        tracemalloc.stop()

##############################################################################
# %% Helper functions/classes
##############################################################################

class TaskArgs(scu.prettyobj):
    '''
        A class to hold the arguments for the parallel task -- not to be invoked by the user.

        Arguments and ordering must match both ``sc.parallelize()`` and ``sc._parallel_task()`` '''

    def __init__(self, func, index, iterval, iterdict, args, kwargs, maxload, interval, embarrassing):
        self.func = func  # The function being called
        self.index = index  # The place in the queue
        self.iterval = iterval  # The value being iterated (may be None if iterdict is not None)
        self.iterdict = iterdict  # A dictionary of values being iterated (may be None if iterval is not None)
        self.args = args  # Arguments passed directly to the function
        self.kwargs = kwargs  # Keyword arguments passed directly to the function
        self.maxload = maxload  # Maximum CPU load (ignored if ncpus is not None in parallelize()
        self.interval = interval  # Interval to check load (only used with maxload)
        self.embarrassing = embarrassing  # Whether or not to pass the iterarg to the function (no if it's embarrassing)
        return


def _parallel_task(taskargs, outputqueue=None):
    ''' Task called by parallelize() -- not to be called directly '''

    # Handle inputs
    taskargs = scu.dcp(taskargs)
    func = taskargs.func
    index = taskargs.index
    args = taskargs.args
    kwargs = taskargs.kwargs
    if args is None: args = ()
    if kwargs is None: kwargs = {}
    if taskargs.iterval is not None:
        if not isinstance(taskargs.iterval, tuple):  # Ensure it's a tuple
            taskargs.iterval = (taskargs.iterval,)
        if not taskargs.embarrassing:
            args = taskargs.iterval + args  # If variable name is not supplied, prepend it to args
    if taskargs.iterdict is not None:
        for key, val in taskargs.iterdict.items():
            kwargs[key] = val  # Otherwise, include it in kwargs

    # Handle load balancing
    maxload = taskargs.maxload
    if maxload:
        loadbalancer(maxload=maxload, index=index, interval=taskargs.interval)

    # Call the function
    output = func(*args, **kwargs)

    # Handle output
    if outputqueue:
        outputqueue.put((index, output))
        return
    else:
        return output


def _parallelcmd_task(_cmd, _parfor, _returnval, _i, _outputqueue, _maxload, _interval,
                      _kwargs):  # pragma: no cover # No coverage since pickled
    '''
    The task to be executed by ``sc.parallelcmd()``. All internal variables start with
    underscores to avoid possible collisions in the ``exec()`` statements. Not to be called
    directly.
    '''
    loadbalancer(maxload=_maxload, index=_i, interval=_interval)

    # Set the loop variables
    for _key in _parfor.keys():
        _thisval = _parfor[_key][_i]  # analysis:ignore
        exec(f'{_key} = _thisval')  # Set the value of this variable

    # Set the keyword arguments
    for _key in _kwargs.keys():
        _thisval = _kwargs[_key]  # analysis:ignore
        exec(f'{_key} = _thisval')  # Set the value of this variable

    # Calculate the command
    try:
        exec(_cmd)  # The meat of the matter!
    except Exception as E:
        print(f'WARNING, parallel task failed:\n{str(E)}')
        exec(f'{_returnval} = None')

    # Append results
    _outputqueue.put((_i, eval(_returnval)))

    return
