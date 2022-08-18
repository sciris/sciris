"""
Profiling and CPU/memory management functions.

Highlights:
    - :func:`cpuload`: alias to ``psutil.cpu_percent()``
    - :func:`loadbalancer`: very basic load balancer
    - :func:`profile`: a line profiler
    - :func:`resourcemonitor`: a monitor to kill processes that exceed memory or other limits
"""

import os
import sys
import time
import psutil
import signal
import _thread
import threading
import tempfile
import warnings
import numpy as np
import pandas as pd
import pylab as pl
import multiprocessing as mp
from . import sc_utils as scu
from . import sc_datetime as scd
from . import sc_odict as sco
from . import sc_fileio as scf
from . import sc_nested as scn


##############################################################################
#%% Load balancing functions
##############################################################################

__all__ = ['cpu_count', 'cpuload', 'memload', 'loadbalancer']


def cpu_count():
    ''' Alias to ``mp.cpu_count()`` '''
    return mp.cpu_count()


def cpuload(interval=0.1):
    """
    Takes a snapshot of current CPU usage via ``psutil``

    Args:
        interval (float): number of seconds over which to estimate CPU load

    Returns:
        a float between 0-1 representing the fraction of ``psutil.cpu_percent()`` currently used.
    """
    return psutil.cpu_percent(interval=interval)/100


def memload():
    """
    Takes a snapshot of current fraction of memory usage via ``psutil``

    Note on the different functions:

        - ``sc.memload()`` checks current total system memory consumption
        - ``sc.checkram()`` checks RAM (virtual memory) used by the current Python process
        - ``sc.checkmem()`` checks memory consumption by a given object

    Returns:
        a float between 0-1 representing the fraction of ``psutil.virtual_memory()`` currently used.
    """
    return psutil.virtual_memory().percent / 100



def checkmem(var, descend=None, alphabetical=False, plot=False, doprint=True, verbose=False):
    '''
    Checks how much memory the variable or variables in question use by dumping
    them to file.

    Note on the different functions:

        - ``sc.memload()`` checks current total system memory consumption
        - ``sc.checkram()`` checks RAM (virtual memory) used by the current Python process
        - ``sc.checkmem()`` checks memory consumption by a given object

    Args:
        var (any): the variable being checked
        descend (bool): whether or not to descend one level into the object
        alphabetical (bool): if descending into a dict or object, whether to list items by name rather than size
        plot (bool): if descending, show the results as a pie chart
        doprint (bool): whether to print out results
        verbose (bool or int): detail to print, if >1, print repr of objects along the way

    **Example**::

        import sciris as sc
        sc.checkmem(['spiffy',rand(2483,589)], descend=True)
    '''

    def check_one_object(variable):
        ''' Check the size of one variable '''

        if verbose>1:
            print(f'  Checking size of {variable}...')

        # Create a temporary file, save the object, check the size, remove it
        filename = tempfile.mktemp()
        scf.saveobj(filename, variable, die=False)
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
    varnames = ['Variable'] # Set defaults
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

    data = sco.objdict({varnames[i]:[sizes[i], sizestrs[i]] for i in inds})

    if plot: # pragma: no cover
        pl.axes(aspect=1)
        pl.pie(np.array(sizes)[inds], labels=np.array(varnames)[inds], autopct='%0.2f')

    return data


def checkram(unit='mb', fmt='0.2f', start=0, to_string=True):
    '''
    Measure actual memory usage, typically at different points throughout execution.

    Note on the different functions:

        - ``sc.memload()`` checks current total system memory consumption
        - ``sc.checkram()`` checks RAM (virtual memory) used by the current Python process
        - ``sc.checkmem()`` checks memory consumption by a given object

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


def loadbalancer(maxcpu=0.8, maxmem=0.8, index=None, interval=0.5, cpu_interval=0.1,
                 maxtime=36000, label=None, verbose=True, **kwargs):
    '''
    Delay execution while CPU load is too high -- a very simple load balancer.

    Arguments:
        maxcpu       (float) : the maximum CPU load to allow for the task to still start
        maxmem       (float) : the maximum memory usage to allow for the task to still start
        index        (int)   : the index of the task -- used to start processes asynchronously (default None)
        interval     (float) : the time delay to poll to see if CPU load is OK (default 0.5 seconds)
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
    if interval is None: interval = 0

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
        time.sleep(pause) # Give it time to asynchronize, with a predefined delay

    # Loop until load is OK
    toohigh = True # Assume too high
    count = 0
    maxcount = maxtime/float(interval)
    string = ''
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
            string = label+f'CPU load too high ({cpu_str}); {process_str} queued {count} times'
            scd.randsleep(interval)
        elif mem_toohigh:
            string = label+f'Memory load too high ({mem_str}); {process_str} queued {count} times'
            scd.randsleep(interval)
        else:
            toohigh = False
            string = label+f'CPU & memory fine ({cpu_str} & {mem_str}), starting {process_str} after {count} tries'
        if verbose:
            print(string)
    return string



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
#%% Resource monitor
##############################################################################

__all__ += ['LimitExceeded', 'resourcemonitor']



class LimitExceeded(MemoryError, KeyboardInterrupt):
    '''
    Custom exception for use with the ``sc.resourcemonitor()`` monitor.

    It inherits from ``MemoryError`` since this is the most similar built-in Python
    except, and it inherits from ``KeyboardInterrupt`` since this is the means by
    which the monitor interrupts the main Python thread.
    '''
    pass


class resourcemonitor(scu.prettyobj):
    """
    Asynchronously monitor resource (e.g. memory) usage and terminate the process
    if the specified threshold is exceeded.

    Args:
        mem (float): maximum virtual memory allowed (as a fraction of total RAM)
        cpu (float): maximum CPU usage (NB: included for completeness only; typically one would not terminate a process just due to high CPU usage)
        time (float): maximum time limit in seconds
        interval (float): how frequently to check memory/CPU usage (in seconds)
        label (str): an optional label to use while printing out progress
        start (bool): whether to start the resource monitor on initialization (else call ``start()``)
        die (bool): whether to raise an exception if the resource limit is exceeded
        kill_children (bool): whether to kill child processes (if False, will not work with multiprocessing)
        kill_parent (bool): whether to also kill the parent process (will usually exit Python interpreter in the process)
        callback (func): optional callback if the resource limit is exceeded
        verbose (bool): detail to print out (default: if exceeded; True: every step; False: no output)

    **Examples**::

        # Using with-as:
        with sc.resourcemonitor(mem=0.8) as resmon:
            memory_heavy_job()

        # As a standalone (don't forget to call stop!)
        resmon = sc.resourcemonitor(mem=0.95, cpu=0.9, time=3600, label='Load checker', die=False, callback=post_to_slack)
        long_cpu_heavy_job()
        resmon.stop()
        print(resmon.to_df())
        ,
    """
    def __init__(self, mem=0.9, cpu=None, time=None, interval=1.0, label=None, start=True,
                 die=True, kill_children=True, kill_parent=False, callback=None, verbose=None):
        self.mem  = mem  if mem  else 1.0    # Memory limit
        self.cpu  = cpu  if cpu  else 1.0    # CPU limit
        self.time = time if time else np.inf # Time limit
        self.interval = interval
        self.label    = label if label else 'Monitor'
        self.die      = die
        self.kill_children = kill_children
        self.kill_parent   = kill_parent
        self.callback   = callback
        self.verbose    = verbose
        self.running    = False # Whether the monitor is running
        self.count      = 0 # Count number of iterations the monitor has been running for
        self.start_time = 0 # When the monitor started running
        self.elapsed    = 0 # How long the monitor has been running for
        self.log        = [] # Log of output
        self.parent     = os.getpid() # ID of the current process (parent of thread)
        self.thread     = None # Store the separate thread that will be running the monitor
        self.exception  = None # Store the exception if raised
        self._orig_sigint = signal.getsignal(signal.SIGINT)
        if start:
            self.start()
        return


    def start(self, label=None):
        '''
        Start the monitor running

        Args:
            label (str): optional label for printing progress
        '''

        def handler(signum, frame):
            ''' Custom exception handler '''
            if self.exception is not None:
                raise self.exception
            else:
                return self._orig_sigint()

        if not self.running:

            # Overwrite default KeyboardInterrupt handling when we start
            try:
                signal.signal(signal.SIGINT, handler)
            except ValueError:
                errormsg = 'Could not set signal, probably not calling from main thread'
                print(errormsg)

            # Create a thread and start running
            self.start_time = time.time()
            self.running = True
            self.thread = threading.Thread(target=self.monitor, daemon=True)
            self.thread.start()

        return self


    def stop(self):
        ''' Stop the monitor from running '''
        if self.verbose:
            print(f'{self.label}: done')
        self.running = False
        try:
            signal.signal(signal.SIGINT, self._orig_sigint) # Restore original KeyboardInterrupt handling
        except ValueError:
            errormsg = 'Could not reset signal, probably not calling from main thread'
            print(errormsg)
        if self.exception is not None and self.die: # This exception has likely already been raised, but if not, raise it now
            raise self.exception
        return self


    def __enter__(self, *args, **kwargs):
        ''' For use in a context block '''
        return self.start()


    def __exit__(self, *args, **kwargs):
        ''' For use in a context block '''
        return self.stop()


    def monitor(self, label=None, *args, **kwargs):
        ''' Actually run the resource monitor '''
        while self.running:
            self.count += 1
            is_ok, checkdata, checkstr = self.check()
            if self.verbose:
                updatestr = f"{self.label} step {self.count}: {checkstr}"
                print(updatestr)
                if self.callback:
                    self.callback(checkdata, updatestr)
            if not is_ok:
                self.running = False
                self.exception = LimitExceeded(checkstr)
                if self.callback:
                    self.callback(checkdata, checkstr)
                if self.die:
                    self.kill()
            time.sleep(self.interval)

        return


    def check(self):
        ''' Check if any limits have been exceeded '''
        time_now = time.time()
        self.elapsed = time_now - self.start_time

        # Define the limits
        lim = sco.objdict(
            cpu  = self.cpu,
            mem  = self.mem,
            time = self.time,
        )

        # Check current load
        now = sco.objdict(
            cpu  = cpuload(),
            mem  = memload(),
            time = self.elapsed,
        )

        # Check if limits are OK, and the ratios
        ok    = sco.objdict()
        ratio = sco.objdict()
        for k in lim.keys():
            ok[k]    = now[k] <= lim[k]
            ratio[k] = now[k] / lim[k]
        is_ok = ok[:].all()

        # Gather into output form
        checkdata = sco.objdict(
            count    = self.count,
            elapsed  = self.elapsed,
            is_ok    = is_ok,
            load     = now,
            limit    = lim,
            limit_ok = ok,
            ratio    = ratio
        )
        self.log.append(checkdata)

        # Convert to string
        prefix = 'Limits OK' if is_ok else 'Limits exceeded'
        datalist = scu.autolist()
        if self.cpu < 1:
            datalist += f'CPU: {now.cpu:0.2f} vs {lim.cpu:0.2f}'
        if self.mem < 1:
            datalist += f'Memory: {now.mem:0.2f} vs {lim.mem:0.2f}'
        if np.isfinite(self.time):
            datalist += f'Time: {now.time:n} vs {lim.time:n}'
        datastr = '; '.join(datalist)
        checkstr = f'{prefix}: {datastr}'

        return is_ok, checkdata, checkstr


    def kill(self):
        ''' Kill all processes '''
        kill_verbose = self.verbose is not False # Print if self.verbose is True or None (just not False)
        if kill_verbose:
            print(self.exception)
            print('Killing processes...')

        parent   = psutil.Process(self.parent)
        children = parent.children(recursive=True)

        if self.kill_children:
            for c,child in enumerate(children):
                if kill_verbose:
                    print(f'Killing child {c+1} of {len(children)}...')
                child.kill()

        if self.kill_parent:
            if kill_verbose:
                print(f'Killing parent (PID={self.parent_pid})')
            parent.kill()

        # Finally, interrupt the main thread -- usually not recoverable, but the only way to interrupt it
        _thread.interrupt_main()

        return


    def to_df(self):
        ''' Convert the log into a pandas dataframe '''
        entries = []
        for entry in self.log:
            flat = scn.flattendict(entry, sep='_')
            entries.append(flat)
        self.df = pd.DataFrame(entries)
        return self.df

