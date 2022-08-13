"""
Profiling and CPU/memory management functions.

NB: Uses ``multiprocess`` instead of ``multiprocessing`` under the hood for
broadest support  across platforms (e.g. Jupyter notebooks).

Highlights:
    - ``sc.cpuload()``: alias to ``psutil.cpu_percent()``
    - ``sc.loadbalancer()``: very basic load balancer
    - ``sc.profile()``: a line profiler
"""

import os
import sys
import time
import psutil
import contextlib
import tracemalloc
import resource
import tempfile
import warnings
import numpy as np
import multiprocess as mp
from . import sc_utils as scu


##############################################################################
#%% Load balancing functions
##############################################################################

__all__ = ['cpu_count', 'cpuload', 'memload', 'loadbalancer']


def cpu_count():
    ''' Alias to mp.cpu_count() '''
    return mp.cpu_count()


def cpuload(interval=0.1):
    """
    Takes a snapshot of current CPU usage via psutil

    Args:
        interval (float): number of seconds over which to estimate CPU load

    Returns:
        a float between 0-1 representing the fraction of ``psutil.cpu_percent()`` currently used.
    """
    return psutil.cpu_percent(interval=interval)/100


def memload():
    """
    Takes a snapshot of current fraction of memory usage via psutil

    Returns:
        a float between 0-1 representing the fraction of ``psutil.virtual_memory()`` currently used.
    """
    return psutil.virtual_memory().percent / 100


def loadbalancer(maxcpu=0.8, maxmem=0.8, index=None, interval=1.0, cpu_interval=0.1, maxtime=36000, label=None, verbose=True, **kwargs):
    '''
    Delay execution while CPU load is too high -- a very simple load balancer.

    Arguments:
        maxcpu       (float) : the maximum CPU load to allow for the task to still start
        maxmem       (float) : the maximum memory usage to allow for the task to still start
        index        (int)   : the index of the task -- used to start processes asynchronously (default None)
        interval     (float) : the time delay to poll to see if CPU load is OK (default 1 second)
        cpu_interval (float) : number of seconds over which to estimate CPU load (default 0.1; to small gives inaccurate readings)
        maxtime      (float) : maximum amount of time to wait to start the task (default 36000 seconds (10 hours))
        label        (str)   : the label to print out when outputting information about task delay or start (default None)
        verbose      (bool)  : whether or not to print information about task delay or start (default True)

    **Examples**::

        # Simplest usage -- delay if CPU or memory load is >80%
        sc.loadbalancer()

        # Use a maximum CPU load of 50%, maximum memory of 90%, and stagger the start by process number
        for nproc in processlist:
            sc.loadbalancer(maxload=0.5, maxmem=0.9, index=nproc)

    | New in version 2.0.0: ``maxmem`` argument; ``maxload`` renamed ``maxcpu``
    '''

    def sleep(interval=None, pause=None):
        ''' Sleeps for an average of ``interval`` seconds, but do it randomly so you don't get locking '''
        if interval:
            pause = interval*2*np.random.rand()
        time.sleep(pause)
        return

    # Handle deprecation
    maxload = kwargs.pop('maxload', None)
    if maxload is not None: # pragma: no cover
        maxcpu = maxload
        warnmsg = 'sc.loadbalancer() argument "maxload" has been renamed "maxcpu" as of v2.0.0'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)

    # Set up processes to start asynchronously
    if maxcpu   is None or maxcpu  is False: maxcpu  = 1.0
    if maxmem   is None or maxmem  is False: maxmem  = 1.0
    if maxtime  is None or maxtime is False: maxtime = 36000

    if label is None:
        label = ''
    else:
        label += ': '

    if index is None:
        pause = interval*2*np.random.rand()
        index = ''
    else:
        pause = index*interval

    if maxcpu>1: maxcpu = maxcpu/100 # If it's >1, assume it was given as a percent
    if maxmem>1: maxmem = maxmem/100
    if (not 0 < maxcpu < 1) and (not 0 < maxmem < 1):
        return # Return immediately if no max load
    else:
        sleep(pause=pause) # Give it time to asynchronize

    # Loop until load is OK
    toohigh = True # Assume too high
    count = 0
    maxcount = maxtime/float(interval)
    while toohigh and count<maxcount:
        count += 1
        cpu_current = cpuload(interval=cpu_interval) # If interval is too small, can give very inaccurate readings
        mem_current = memload()
        cpu_toohigh = cpu_current > maxcpu
        mem_toohigh = mem_current > maxmem
        cpu_compare = ['<', '>'][cpu_toohigh]
        mem_compare = ['<', '>'][mem_toohigh]
        cpu_str = f'{cpu_current:0.2f}{cpu_compare}{maxcpu:0.2f}'
        mem_str = f'{mem_current:0.2f}{mem_compare}{maxmem:0.2f}'
        process_str = f'process {index}' if index else 'process'
        if cpu_toohigh:
            if verbose:
                print(label+f'CPU load too high ({cpu_str}); {process_str} queued {count} times')
            sleep(interval=interval)
        elif mem_toohigh:
            if verbose:
                print(label+f'Memory load too high ({mem_str}); {process_str} queued {count} times')
            sleep(interval=interval)
        else:
            toohigh = False
            if verbose: print(label+f'CPU & memory fine ({cpu_str} & {mem_str}), starting {process_str} after {count} tries')
    return



##############################################################################
#%% Profiling functions
##############################################################################

__all__ += ['profile', 'mprofile', 'checkmem', 'checkram']


def profile(run, follow=None, print_stats=True, *args, **kwargs):
    '''
    Profile the line-by-line time required by a function.

    Args:
        run (function): The function to be run
        follow (function): The function or list of functions to be followed in the profiler; if None, defaults to the run function
        print_stats (bool): whether to print the statistics of the profile to stdout
        args, kwargs: Passed to the function to be run

    Returns:
        LineProfiler (by default, the profile output is also printed to stdout)

    **Example**::

        def slow_fn():
            n = 10000
            int_list = []
            int_dict = {}
            for i in range(n):
                int_list.append(i)
                int_dict[i] = i
            return

        class Foo:
            def __init__(self):
                self.a = 0
                return

            def outer(self):
                for i in range(100):
                    self.inner()
                return

            def inner(self):
                for i in range(1000):
                    self.a += 1
                return

        foo = Foo()
        sc.profile(run=foo.outer, follow=[foo.outer, foo.inner])
        sc.profile(slow_fn)

        # Profile the constructor for Foo
        f = lambda: Foo()
        sc.profile(run=f, follow=[foo.__init__])
    '''
    try:
        from line_profiler import LineProfiler
    except ModuleNotFoundError as E: # pragma: no cover
        if 'win' in sys.platform:
            errormsg = 'The "line_profiler" package is not included by default on Windows;' \
                        'please install using "pip install line_profiler" (note: you will need a ' \
                        'C compiler installed, e.g. Microsoft Visual Studio)'
        else:
            errormsg = 'The "line_profiler" Python package is required to perform profiling'
        raise ModuleNotFoundError(errormsg) from E

    if follow is None:
        follow = run
    orig_func = run

    lp = LineProfiler()
    follow = scu.promotetolist(follow)
    for f in follow:
        lp.add_function(f)
    lp.enable_by_count()
    wrapper = lp(run)

    if print_stats: # pragma: no cover
        print('Profiling...')
    wrapper(*args, **kwargs)
    run = orig_func
    if print_stats: # pragma: no cover
        lp.print_stats()
        print('Done.')
    return lp


def mprofile(run, follow=None, show_results=True, *args, **kwargs):
    '''
    Profile the line-by-line memory required by a function. See profile() for a
    usage example.

    Args:
        run (function): The function to be run
        follow (function): The function or list of functions to be followed in the profiler; if None, defaults to the run function
        show_results (bool): whether to print the statistics of the profile to stdout
        args, kwargs: Passed to the function to be run

    Returns:
        LineProfiler (by default, the profile output is also printed to stdout)
    '''

    try:
        import memory_profiler as mp
    except ModuleNotFoundError as E: # pragma: no cover
        if 'win' in sys.platform:
            errormsg = 'The "memory_profiler" package is not included by default on Windows;' \
                        'please install using "pip install memory_profiler" (note: you will need a ' \
                        'C compiler installed, e.g. Microsoft Visual Studio)'
        else:
            errormsg = 'The "memory_profiler" Python package is required to perform profiling'
        raise ModuleNotFoundError(errormsg) from E

    if follow is None:
        follow = run

    lp = mp.LineProfiler()
    follow = scu.promotetolist(follow)
    for f in follow:
        lp.add_function(f)
    lp.enable_by_count()
    try:
        wrapper = lp(run)
    except TypeError as e: # pragma: no cover
        raise TypeError('Function wrapping failed; are you profiling an already-profiled function?') from e

    if show_results:
        print('Profiling...')
    wrapper(*args, **kwargs)
    if show_results:
        mp.show_results(lp)
        print('Done.')
    return lp


##############################################################################
#%% Memory management
##############################################################################


def checkmem(var, descend=None, alphabetical=False, plot=False, verbose=False):
    '''
    Checks how much memory the variable or variables in question use by dumping
    them to file. See also checkram().

    Args:
        var (any): the variable being checked
        descend (bool): whether or not to descend one level into the object
        alphabetical (bool): if descending into a dict or object, whether to list items by name rather than size
        plot (bool): if descending, show the results as a pie chart
        verbose (bool or int): detail to print, if >1, print repr of objects along the way

    **Example**::

        import sciris as sc
        sc.checkmem(['spiffy',rand(2483,589)], descend=True)
    '''
    from .sc_fileio import saveobj # Here to avoid recursion

    def check_one_object(variable):
        ''' Check the size of one variable '''

        if verbose>1:
            print(f'  Checking size of {variable}...')

        # Create a temporary file, save the object, check the size, remove it
        filename = tempfile.mktemp()
        saveobj(filename, variable, die=False)
        filesize = os.path.getsize(filename)
        os.remove(filename)

        # Convert to string
        factor = 1
        label = 'B'
        labels = ['KB','MB','GB']
        for i,f in enumerate([3,6,9]):
            if filesize>10**f:
                factor = 10**f
                label = labels[i]
        humansize = float(filesize/float(factor))
        sizestr = f'{humansize:0.3f} {label}'
        return filesize, sizestr

    # Initialize
    varnames  = []
    variables = []
    sizes     = []
    sizestrs  = []

    # Create the object(s) to check the size(s) of
    varnames = [''] # Set defaults
    variables = [var]
    if descend or descend is None:
        if hasattr(var, '__dict__'): # It's an object
            if verbose>1: print('Iterating over object')
            varnames = sorted(list(var.__dict__.keys()))
            variables = [getattr(var, attr) for attr in varnames]
        elif np.iterable(var): # Handle dicts and lists
            if isinstance(var, dict): # Handle dicts
                if verbose>1: print('Iterating over dict')
                varnames = list(var.keys())
                variables = var.values()
            else: # Handle lists and other things
                if verbose>1: print('Iterating over list')
                varnames = [f'item {i}' for i in range(len(var))]
                variables = var
        else:
            if descend: # Could also be None
                print('Object is not iterable: cannot descend') # Print warning and use default

    # Compute the sizes
    for v,variable in enumerate(variables):
        if verbose:
            print(f'Processing variable {v} of {len(variables)}')
        filesize, sizestr = check_one_object(variable)
        sizes.append(filesize)
        sizestrs.append(sizestr)

    if alphabetical:
        inds = np.argsort(varnames)
    else:
        inds = np.argsort(sizes)[::-1]

    for i in inds:
        varstr = f'Variable "{varnames[i]}"' if varnames[i] else 'Variable'
        print(f'{varstr} is {sizestrs[i]}')

    if plot: # pragma: no cover
        import pylab as pl # Optional import
        pl.axes(aspect=1)
        pl.pie(pl.array(sizes)[inds], labels=pl.array(varnames)[inds], autopct='%0.2f')

    return


def checkram(unit='mb', fmt='0.2f', start=0, to_string=True):
    '''
    Unlike checkmem(), checkram() looks at actual memory usage, typically at different
    points throughout execution.

    **Example**::

        import sciris as sc
        import numpy as np
        start = sc.checkram(to_string=False)
        a = np.random.random((1_000, 10_000))
        print(sc.checkram(start=start))

    New in version 1.0.0.
    '''
    process = psutil.Process(os.getpid())
    mapping = {'b':1, 'kb':1e3, 'mb':1e6, 'gb':1e9}
    try:
        factor = mapping[unit.lower()]
    except KeyError: # pragma: no cover
        raise scu.KeyNotFoundError(f'Unit {unit} not found among {scu.strjoin(mapping.keys())}')
    mem_use = process.memory_info().rss/factor - start
    if to_string:
        output = f'{mem_use:{fmt}} {unit.upper()}'
    else:
        output = mem_use
    return output


__all__ += ['limit_malloc', 'memory', 'ResourceLimit', 'MemoryMonitor']

class ResourceLimit:
    """
    DOCME
    """
    def __init__(self, limit, verbose=False):
        self.percentage_limit = limit
        self.verbose = verbose
        self.LIMITS = [('RLIMIT_DATA', 'heap size'),
                       ('RLIMIT_AS', 'address size'),
]
    def __enter__(self):
        # New soft limit
        totalmem = psutil.virtual_memory().available
        new_soft = int(round(totalmem * self.percentage_limit))

        self.old_softie = []
        self.old_hardie = []

        for name, description in self.LIMITS:
            limit_num = getattr(resource, name)
            soft, hard = resource.getrlimit(limit_num)
            self.old_softie.append(soft)
            self.old_hardie.append(hard)
            resource.setrlimit(limit_num, (new_soft, hard))
            if self.verbose:
                sl, unit_sl = self.human_readable(new_soft)
                hl, unit_hl = self.human_readable(hard)
                print('Setting {:<23} {:<23} {:6} {}{}/{}{}'.format(name, description, "to", sl, unit_sl, hl, unit_hl))

    def __exit__(self, exc_type, exc_value, exc_tb):
        # TODO: Deal with exceptions here
        for (name, description), soft, hard in zip(self.LIMITS, self.old_softie, self.old_hardie):
            limit_num = getattr(resource, name)
            resource.setrlimit(limit_num, (soft, hard))
            if self.verbose:
                sl, unit_sl = self.human_readable(soft)
                hl, unit_hl = self.human_readable(hard)
                print('Resetting {:<23} {:<23} {:6} {}{}/{}{}'.format(name, description, "to", sl, unit_sl, hl, unit_hl))

    def human_readable(self, limit):
        """
        Deal with limits that are -1, implies unlimited
        """
        if limit < 0:
            unit = ""
            limit = "max"
        else:
            unit = "GB"
            limit >>= 30
        return limit, unit


class MemoryMonitor(mp.Process):
    """
    DOCME

    def function_that_needs_a_lot_of_ram():
       l1 = []
       for i in range(2000):
           l1.append(x for x in range(1000000))
       return l1

    with sc.MemoryMonitor(max_mem=0.35) as monitor:
       # Start operation of interest
        ptask = multiprocess.Process(target=function_that_needs_a_lot_of_ram)
        ptask.start()
      # Let the memory monitor track the process of interest
        monitor.task_id(ptask.pid)
      # Start monitoring memory
        monitor.start()
      # If the process of interest finished, stop monitoring
        monitor.stop(ptask.join())


    """
    def __init__(self, max_mem, verbose=True, verbose_monitor=False):
        mp.Process.__init__(self, name='MemoryLimiter')
        self.max_mem = max_mem
        self.current_mem = take_mem_snapshot()
        self.daemon = True # Make this a deamon process
        self.reached_memory_limit = False
        self.verbose = verbose
        self.verbose_monitor = verbose_monitor # To be removed, just for debugging

    def run(self):
        while not self.reached_memory_limit:
            # TODO: add interval attr
            # time.sleep(1)
            self.current_mem = take_mem_snapshot()
            if self.current_mem > self.max_mem:
                self.reached_memory_limit = True
            if self.verbose_monitor:
                print(f"Measuring memory: {self.current_mem:.3f}")

        # Terminate task
        self.reached_memory_limit = True
        self.stop_task()

    def stop(self, join_output):
        if join_output is None:
            if self.verbose:
                print("Terminate memory monitoring")
            self.terminate()

    def task_id(self, pid):
        """
        Track the process of interest
        """
        self.task_id = pid
        self.p = psutil.Process(pid)

    def stop_task(self):
        """
        Terminate the process of interest
        """
        if self.verbose:
            print(f"Terminating task because reached max memory limit: {self.current_mem:.3f}/{self.max_mem:.3f}")
        self.p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join()
        return


def take_mem_snapshot():
    """
    Take a snapshot of current memory usage (in %) via psutil
    Arguments: None

    Returns: a float between 0-1 representing the fraction of psutil.virtuak memory currently used.
    """
    snapshot = psutil.virtual_memory().percent / 100.0
    return snapshot


def memory(percentage=0.8, verbose=True):
    """
    Arguments:
        percentage: a float between 0 and 1
        verbose: whether to print more info
    @sc.memory(0.05)
    def function_that_needs_a_lot_of_ram():
       l1 = []
       for i in range(2000):
           l1.append(x for x in range(1000000))
       return l1

    function_that_needs_a_lot_of_ram()
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            with ResourceLimit(percentage_limit=percentage, verbose=verbose):
                try:
                    return function(*args, **kwargs)
                except MemoryError:
                    if verbose:
                        print("Aborting. Memory limit reached.")
                    return
        return wrapper
    return decorator


@contextlib.contextmanager
def limit_malloc(size):
    """
    Context manager to trace memory block allocation.
    Useful for debuggin purposes.

    Argument:
       size (in B)

    **Example**::

        import sciris as sc
        with sc.limit_malloc(500):
           l1 = []
           for x in range(200000):
               # print(x)
               l1.append(x)

    Source:
    https://gist.github.com/adalekin/2b4219808ac72cafda6cff896739a11d
    https://docs.python.org/3.9/library/tracemalloc.html
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
            for stat in snapshot:
                print(stat)
            raise RuntimeError(f'Memory usage exceeded the threshold: {current_size} > {size}')
    finally:
        tracemalloc.stop()
