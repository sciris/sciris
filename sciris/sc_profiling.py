"""
Profiling and CPU/memory management functions.

Highlights:
    - :func:`sc.profile() <profile>`: a line profiler
    - :func:`sc.cprofile() <cprofile>`: a function profiler
    - :func:`sc.benchmark() <benchmark>`: quickly check your computer's performance
    - :func:`sc.loadbalancer() <loadbalancer>`: very basic load balancer
    - :func:`sc.resourcemonitor() <resourcemonitor>`: a monitor to kill processes that exceed memory or other limits
"""

import re
import os
import sys
import time
import types
import psutil
import signal
import pstats
import cProfile
import _thread
import threading
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import sciris as sc


##############################################################################
#%% Basic performance functions
##############################################################################

__all__ = ['checkmem', 'checkram', 'benchmark']


def checkmem(var, descend=1, order='size', compresslevel=0, maxitems=1000,
             subtotals=True, plot=False, verbose=False, **kwargs):
    """
    Checks how much memory the variable or variables in question use by dumping
    them to file.

    Note on the different functions:

        - :func:`sc.memload() <memload>` checks current total system memory consumption
        - :func:`sc.checkram() <checkram>` checks RAM (virtual memory) used by the current Python process
        - :func:`sc.checkmem() <checkmem>` checks memory consumption by a given object

    Args:
        var (any): the variable being checked
        descend (bool): whether or not to descend one level into the object
        order (str): order in which to list items: "size" (default), "alphabetical", or "none"
        compresslevel (int): level of compression to use when saving to file (typically 0)
        maxitems (int): the maximum number of separate entries to check the size of
        subtotals (bool): whether to include subtotals for different levels of depth
        plot (bool): if descending, show the results as a pie chart
        verbose (bool or int): detail to print, if >1, print repr of objects along the way
        **kwargs (dict): passed to :func:`sc.load() <sciris.sc_fileio.load>`

    **Examples**::

        import numpy as np
        import sciris as sc

        list_obj = ['label', np.random.rand(2483,589)])
        sc.checkmem(list_obj)


        nested_dict = dict(
            foo = dict(
                a = np.random.rand(5,10),
                b = np.random.rand(5,20),
                c = np.random.rand(5,50),
            ),
            bar = [
                np.random.rand(5,100),
                np.random.rand(5,200),
                np.random.rand(5,500),
            ],
            cat = np.random.rand(5,10),
        )
        sc.checkmem(nested_dict)

    *New in version 3.0.0:* descend multiple levels; dataframe output; "alphabetical" renamed "order"
    """

    # Handle input arguments -- used for recursion
    _depth  = kwargs.pop('_depth', 0)
    _prefix = kwargs.pop('_prefix', '')
    _join   = kwargs.pop('_join', '→')

    def check_one_object(variable):
        """ Check the size of one variable """

        if verbose>1: # pragma: no cover
            print(f'  Checking size of {variable}...')

        # Create a temporary file, save the object, check the size, remove it
        filename = tempfile.mktemp()
        sc.save(filename, variable, allow_empty=True, compresslevel=compresslevel)
        filesize = os.path.getsize(filename)
        os.remove(filename)

        sizestr = sc.humanize_bytes(filesize)
        return filesize, sizestr

    # Initialize
    varnames  = []
    variables = []
    columns=dict(
        variable  = object,
        humansize = object,
        bytesize  = int,
        depth     = int,
        is_total  = bool,
    )
    df = sc.dataframe(columns=columns)

    if descend:
        if isinstance(var, dict): # Handle dicts
            if verbose>1: print('Iterating over dict')
            varnames = list(var.keys())
            variables = var.values()
        elif hasattr(var, '__dict__'): # It's an object
            if verbose>1: print('Iterating over class-like object')
            varnames = sorted(list(var.__dict__.keys()))
            variables = [getattr(var, attr) for attr in varnames]
        elif sc.isiterable(var, exclude=str): # Handle lists, and be sure to skip strings
            if verbose>1: print('Iterating over list-like object')
            varnames = [f'item {i}' for i in range(len(var))]
            variables = var
        else:
            descend = 0 # Can't descend

    # Create the object(s) to check the size(s) of
    if not descend:
        varname = _prefix if _prefix else 'Variable'
        bytesize, sizestr = check_one_object(var)
        df.appendrow(dict(variable=varname, humansize=sizestr, bytesize=bytesize, depth=_depth, is_total=False))

    else:
        # Error checking
        n_variables = len(variables)
        if n_variables > maxitems: # pragma: no cover
            errormsg = f'Cannot compute the sizes of {n_variables} items since maxitems is set to {maxitems}'
            raise RuntimeError(errormsg)

        # Compute the sizes recursively
        for v,(varname,variable) in enumerate(zip(varnames, variables)):
            if verbose: # pragma: no cover
                print(f'Processing variable {v} of {len(variables)}')
            label = _join.join([_prefix, varname]) if _prefix else varname
            this_df = checkmem(variable, descend=descend-1, compresslevel=compresslevel, maxitems=maxitems, plot=False, verbose=False, _prefix=label, _depth=_depth+1)
            df.concat(this_df, inplace=True)

    # Handle subtotals
    if subtotals and len(df) > 1:
        total_label = _prefix + ' (total)' if _prefix else 'Total'
        total = df[np.logical_not(df.is_total)].bytesize.sum()
        human_total = sc.humanize_bytes(total)
        df.appendrow(dict(variable=total_label, humansize=human_total, bytesize=total, depth=_depth, is_total=True))

    # Only sort if we're at the highest level
    if _depth == 0 and len(df) > 1:
        # if subtotals:

        if order == 'alphabetical': # pragma: no cover
            df.sortrows(col='variable')
        elif order == 'size':
            df.sortrows(col='bytesize', reverse=True)

    if plot: # pragma: no cover
        plt.axes(aspect=1)
        plt.pie(df.bytesize, labels=df.variable, autopct='%0.2f')

    return df


def checkram(unit='mb', fmt='0.2f', start=0, to_string=True):
    """
    Measure actual memory usage, typically at different points throughout execution.

    Note on the different functions:

        - :func:`sc.memload() <memload>` checks current total system memory consumption
        - :func:`sc.checkram() <checkram>` checks RAM (virtual memory) used by the current Python process
        - :func:`sc.checkmem() <checkmem>` checks memory consumption by a given object

    **Example**::

        import sciris as sc
        import numpy as np
        start = sc.checkram(to_string=False)
        a = np.random.random((1_000, 10_000))
        print(sc.checkram(start=start))

    *New in version 1.0.0.*
    """
    process = psutil.Process(os.getpid())
    mapping = {'b':1, 'kb':1e3, 'mb':1e6, 'gb':1e9}
    try:
        factor = mapping[unit.lower()]
    except KeyError: # pragma: no cover
        raise sc.KeyNotFoundError(f'Unit {unit} not found among {sc.strjoin(mapping.keys())}')
    mem_use = process.memory_info().rss/factor - start
    if to_string:
        output = f'{mem_use:{fmt}} {unit.upper()}'
    else: # pragma: no cover
        output = mem_use
    return output


def benchmark(repeats=5, scale=1, verbose=False, python=True, numpy=True, parallel=False, return_timers=False):
    """
    Benchmark Python performance

    Performs a set of standard operations in both Python and Numpy and times how
    long they take. Results are returned in terms of millions of operations per second (MOPS).
    With default settings, this function should take very approximately 0.1 s to
    run (depending on the machine, of course!).

    For Python, these operations are: for loops, list append/indexing, dict set/get,
    and arithmetic. For Numpy, these operations are: random floats, random ints,
    addition, and multiplication.

    Args:
        repeats (int): the number of times to repeat each test
        scale (float): the scale factor to use for the size of the loops/arrays
        verbose (bool): print out the results after each repeat
        python (bool): whether to run the Python tests
        numpy (bool): whether to run the Numpy tests
        parallel (bool/int): whether to run the tests across all cores
        return_timers (bool): if True, return the timer objects instead of the "MOPS" results

    Returns:
        A dict with keys "python" and "numpy" for the number of MOPS for each

    **Examples**::

        sc.benchmark() # Returns e.g. {'python': 11.43, 'numpy': 236.595}

        numpy_mops = sc.benchmark(python=False)['numpy']
        if numpy_mops < 100:
            print('Your computer is slow')
        elif numpy_mops > 400:
            print('Your computer is fast')
        else:
            print('Your computer is normal')

        sc.benchmark(parallel=True) # Use all CPUs

    | *New in version 3.0.0.*
    | *New in version 3.1.0:* "parallel" argument; increased default scale
    """

    # Calculate the number of operations
    py_outer = 10
    np_outer = 1
    py_inner = scale*1e3
    np_inner = scale*1e6
    py_ops = (py_outer * py_inner * 18)/1e6
    np_ops = (np_outer * np_inner * 4)/1e6


    # Define the benchmarking functions

    def bm_python(prefix=''):
        P = sc.timer(verbose=verbose)
        for r in range(repeats):
            l = list()
            d = dict()
            result = 0
            P.tic()
            for i in range(py_outer):
                for j in range(int(py_inner)):
                    l.append([i,j]) # Operation 1: list append
                    d[str((i,j))] = [i,j] # Operations 2-3: convert to string and dict assign
                    v1 = l[-1][0] + l[-1][0] # Operations 4-8: list get (x4) and sum
                    v2 = d[str((i,j))][0] + d[str((i,j))][1] # Operations 9-16: convert to string (x2), list get (x2), dict get (x2), sum
                    result += v1 + v2 # Operations 17-18: sum (x2)
            P.toc(f'{prefix}Python, {py_ops}m operations')
        return P

    def bm_numpy(prefix=''):
        N = sc.timer(verbose=verbose)
        for r in range(repeats):
            N.tic()
            for i in range(np_outer):
                a = np.random.random(int(np_inner)) # Operation 1: random floats
                b = np.random.randint(10, size=int(np_inner)) # Operation 2: random integers
                a + b # Operation 3: addition
                a*b # Operation 4: multiplication
            N.toc(f'{prefix}Numpy, {np_ops}m operations')
        return N


    # Do the benchmarking

    if not parallel:

        ncpus = 1

        # Benchmark plain Python
        if python:
            P = bm_python()

        # Benchmark Numpy
        if numpy:
            N = bm_numpy()

    else:

        from . import sc_parallel as scpar

        if parallel == 1: # Probably "True"
            ncpus = cpu_count()
        else:
            try:
                ncpus = int(parallel)
            except Exception as E:
                errormsg = f'Could not interpret "{parallel}" as a number of cores'
                raise ValueError(errormsg) from E

        arglist = [f'Run {i}: ' for i in range(ncpus)]

        if python:
            Plist = scpar.parallelize(bm_python, arglist, ncpus=ncpus)
            P = sum(Plist)
        if numpy:
            Nlist = scpar.parallelize(bm_numpy, arglist, ncpus=ncpus)
            N = sum(Nlist)

    # Handle output
    if return_timers: # pragma: no cover
        out = sc.objdict(python=P, numpy=N)
    else:
        pymops = py_ops/P.mean()*ncpus if len(P) else None # Handle if one or the other isn't run
        npmops = np_ops/N.mean()*ncpus if len(N) else None
        out = dict(python=pymops, numpy=npmops)

    return out


##############################################################################
#%% Load balancing functions
##############################################################################

__all__ += ['cpu_count', 'cpuload', 'memload', 'loadbalancer']


def cpu_count():
    """ Alias to :func:`multiprocessing.cpu_count()` """
    return mp.cpu_count()


def cpuload(interval=0.1):
    """
    Takes a snapshot of current CPU usage via :mod:`psutil`

    Args:
        interval (float): number of seconds over which to estimate CPU load

    Returns:
        a float between 0-1 representing the fraction of :func:`psutil.cpu_percent()` currently used.
    """
    return psutil.cpu_percent(interval=interval)/100


def memload():
    """
    Takes a snapshot of current fraction of memory usage via :mod:`psutil`

    Note on the different functions:

        - :func:`sc.memload() <memload>` checks current total system memory consumption
        - :func:`sc.checkram() <checkram>` checks RAM (virtual memory) used by the current Python process
        - :func:`sc.checkmem() <checkmem>` checks memory consumption by a given object

    Returns:
        a float between 0-1 representing the fraction of :func:`psutil.virtual_memory()` currently used.
    """
    return psutil.virtual_memory().percent / 100





def loadbalancer(maxcpu=0.9, maxmem=0.9, index=None, interval=None, cpu_interval=0.1,
                 maxtime=36_000, label=None, verbose=True, **kwargs):
    """
    Delay execution while CPU load is too high -- a very simple load balancer.

    Arguments:
        maxcpu       (float) : the maximum CPU load to allow for the task to still start
        maxmem       (float) : the maximum memory usage to allow for the task to still start
        index        (int)   : the index of the task -- used to start processes asynchronously (default None)
        interval     (float) : the time delay to poll to see if CPU load is OK (default 0.5 seconds)
        cpu_interval (float) : number of seconds over which to estimate CPU load (default 0.1; too small gives inaccurate readings)
        maxtime      (float) : maximum amount of time to wait to start the task (default 36000 seconds (10 hours))
        label        (str)   : the label to print out when outputting information about task delay or start (default None)
        verbose      (bool)  : whether or not to print information about task delay or start (default True)

    **Examples**::

        # Simplest usage -- delay if CPU or memory load is >80%
        sc.loadbalancer()

        # Use a maximum CPU load of 50%, maximum memory of 90%, and stagger the start by process number
        for nproc in processlist:
            sc.loadbalancer(maxload=0.5, maxmem=0.8, index=nproc)

    | *New in version 2.0.0:* ``maxmem`` argument; ``maxload`` renamed ``maxcpu``
    | *New in version 3.0.0:* ``maxcpu`` and ``maxmem`` set to 0.9 by default
    """

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

    # Handle the interval
    default_interval = 0.5
    min_interval = 1e-3 # Don't allow intervals of less than 1 ms
    if interval is None: # pragma: no cover
        interval = default_interval
        default_interval = None # Used as a flag below
    if interval < min_interval: # pragma: no cover
        interval = min_interval
        warnmsg = f'sc.loadbalancer() "interval" should not be less than {min_interval} s'
        warnings.warn(warnmsg, category=UserWarning, stacklevel=2)

    if label is None:
        label = ''
    else: # pragma: no cover
        label += ': '

    if index is None:
        pause = interval*2*np.random.rand()
        index = ''
    else: # pragma: no cover
        pause = index*interval

    if maxcpu>1: maxcpu = maxcpu/100 # If it's >1, assume it was given as a percent
    if maxmem>1: maxmem = maxmem/100
    if (not 0 < maxcpu < 1) and (not 0 < maxmem < 1) and (default_interval is None): # pragma: no cover
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
        process_str = f'process {index}' if index is not None else 'process'
        if cpu_toohigh: # pragma: no cover
            string = label+f'CPU load too high ({cpu_str}); {process_str} queued {count} times'
            sc.randsleep(interval)
        elif mem_toohigh: # pragma: no cover
            string = label+f'Memory load too high ({mem_str}); {process_str} queued {count} times'
            sc.randsleep(interval)
        else:
            ok = 'OK' if sc.getplatform() == 'windows' else '✓' # Windows doesn't support unicode (!)
            toohigh = False
            string = label+f'CPU {ok} ({cpu_str}), memory {ok} ({mem_str}): starting {process_str} after {count} tries'
        if verbose:
            print(string)
    return string



##############################################################################
#%% Profiling functions
##############################################################################

__all__ += ['profile', 'mprofile', 'cprofile', 'listfuncs', 'tracecalls']


class profile(sc.prettyobj):
    """
    Profile the line-by-line time required by a function.

    Interface to the line_profiler library.

    Note: :func:`sc.profile() <profile>` shows the time taken by each line of code, in the order
    the code appears in. :class:`sc.cprofile() <cprofile>` shows the time taken by each function,
    regardless of where in the code it appears.

    Args:
        run (function): The function to be run
        follow (function): The function, list of functions, class, or module to be followed in the profiler; if None, defaults to the run function
        private (bool/str/list): if True and a class is supplied, follow private functions; if a string/list, follow only those private functions (default ``'__init__'``)
        include (str): if a class/module is supplied, include only functions matching this string
        exclude (str): if a class/module is supplied, exclude functions matching this string
        do_run (bool): whether to run immediately (default: true)
        print_stats (bool): whether to print the statistics of the profile to stdout (default True)
        verbose (bool): list the functions to be profiled
        args (list): Passed to the function to be run
        kwargs (dict): Passed to the function to be run

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
            def __init__(self, a=0):
                self.a = a

            def outer(self):
                for i in range(100):
                    self.inner()

            def inner(self):
                for i in range(1000):
                    self.a += 1

        # Profile a function
        sc.profile(slow_fn)

        # Profile a class or class instance
        foo = Foo()
        sc.profile(run=foo.outer, follow=foo)

        # Profile the constructor for Foo
        f = lambda a: Foo(a)
        sc.profile(run=f, follow=Foo.__init__, a=10) # "a" is passed to the function

    | *New in version 3.2.0:* allow class and module arguments for "follow"; "private" argument
    | *New in version 3.2.2:* converted to a class
    """
    def __init__(self, run, follow=None, private='__init__', include=None, exclude=None,
                do_run=True, verbose=True, *args, **kwargs):
        self.run_func = run
        self.follow = follow
        self.private = private
        self.include = include
        self.exclude = exclude
        self.verbose = verbose
        self.args = args
        self.kwargs = kwargs
        if do_run:
            self.run()
            if verbose:
                self.disp()
        return

    def run(self):
        """ Run profiling """
        try:
            from line_profiler import LineProfiler
        except ModuleNotFoundError as E: # pragma: no cover
            errormsg = 'The "line_profiler" package is not installed; try "pip install line_profiler". (Note: it is not compatible with Python 3.12)'
            raise ModuleNotFoundError(errormsg) from E

        # Figure out follow
        if self.follow is None: # pragma: no cover
            follow_funcs = [self.run_func]
        else:
            follow_funcs = listfuncs(self.follow, private=self.private, include=self.include, exclude=self.exclude)

        if self.verbose:
            print(f'Profiling {len(follow_funcs)} function(s):\n', sc.newlinejoin(follow_funcs), '\n')

        # Construct the wrapper
        orig_func = self.run_func # Needed for argument passing
        prof = LineProfiler()
        for f in follow_funcs:
            prof.add_function(f)
        prof.enable_by_count()
        wrapper = prof(self.run_func) # pragma: no cover

        # Run the profiling
        with sc.timer() as T:
            wrapper(*self.args, **self.kwargs) # pragma: no cover
        self.run_func = orig_func # Restore run for argument passing

        self.follow_funcs = follow_funcs
        self.prof = prof
        self.total = T.total
        self._parse()
        if self.verbose:
            self.disp()

        return

    def __add__(self, other, copy=True):
        """ Allow multiple instances to be combined """
        if not isinstance(other, profile):
            errormsg = f'Can only add two sc.profile() instances, not {type(other)}'
            raise TypeError(errormsg)

        # Merge simple lists and timings
        if copy:
            out = self.__class__.__new__(self.__class__) # New empty class
        else:
            out = self
        out.run_func = sc.mergelists(self.run_func, other.run_func)
        out.follow = sc.mergelists(self.follow, other.follow)
        out.follow_funcs = sc.mergelists(self.follow_funcs, other.follow_funcs)
        out.prof = sc.mergelists(self.prof, other.prof)
        out.total = self.total + other.total

        # Merge outputs
        out.output = self.output.copy(deep=True)
        orig_len = len(self.output)
        for key,entry in other.output.copy(deep=True).items():
            entry.order += orig_len # Update ordering
            out.output[key] = entry

        # Recalculate percentages
        for entry in out.output.values():
            entry.percent = entry.time / out.total * 100

        # Regenerate dataframe
        out.to_df()
        return out

    def __iadd__(self, other):
        return self.__add__(other, copy=False)

    @staticmethod
    def _path_to_name(filepath, lineno):
        """ Helper function to convert a file/line number to a qualified path (via ChatGPT) """
        import ast # Not used elsewhere

        try:
            with open(filepath, 'r') as f:
                source = f.read()

            tree = ast.parse(source, filename=filepath)
            context_stack = []
            name = None

            class ContextVisitor(ast.NodeVisitor):
                def visit_ClassDef(self, node):
                    context_stack.append(node.name)
                    self.generic_visit(node)
                    context_stack.pop()

                def visit_FunctionDef(self, node):
                    if node.lineno == lineno:
                        context_stack.append(node.name)
                        nonlocal name
                        name = '.'.join(context_stack)
                        context_stack.pop()
                    else:
                        context_stack.append(node.name)
                        self.generic_visit(node)
                        context_stack.pop()

            # Do the mapping
            ContextVisitor().visit(tree)
        except:
            name = None

        try:
            module_name = os.path.splitext(os.path.basename(filepath))[0]
        except:
            module_name = None

        if module_name:
            if name:
                return f'{module_name}.{name}'
            else:
                return f'{module_name}.py:{lineno}'
        else:
            return f'{filepath}:{lineno}'

    def _parse(self):
        """ Parse text output into a dictionary (with ChatGPT) """
        with sc.capture() as txt:
            self.prof.print_stats()

        token = 'Total time:' # Define the token used to signify the start of a chunk
        chunks = txt.strip().split(f'\n{token}')
        self.output = sc.objdict()

        for i,chunk in enumerate(chunks[1:]):  # Skip the first split part (before the first "Total time")
            section = 'Total time:' + chunk  # Re-add the delimiter for consistency

            # Extract time
            time_match = re.search(r'Total time:\s*([0-9.+-eE]+)', section)
            time = float(time_match.group(1)) if time_match else None
            percent = time/self.total*100

            # Extract file
            file_match = re.search(r'File:\s*(.+)', section)
            file = file_match.group(1).strip() if file_match else None

            # Extract function name and line number
            func_match = re.search(r'Function:\s*(\w+)\s+at line\s+(\d+)', section)
            function = func_match.group(1) if func_match else None
            lineno = int(func_match.group(2)) if func_match else None

            # Get the qualified name if available
            func_name = self._path_to_name(file, lineno)
            name = sc.uniquename(func_name, self.output.keys(), style='_%d')

            # Store
            entry = sc.objdict(
                name = name,
                order = i,
                time = time,
                percent = percent,
                file = file,
                function = function,
                lineno = lineno,
                data = section,
            )
            self.output[name] = entry

        # The last name extracted is the name of the function that was run
        self.run_func_name = name

        # Sort and convert to dataframe
        self.sort()
        self.to_df()
        return

    def sort(self, bytime=1, copy=False):
        """
        Sort or unsort by time.

        Args:
            bytime (int): if 1, sort by increasing time (default); if -1, sort by decreasing; if 0, do not sort by time
        """
        entries = self.output.values()
        if bytime:
            times = bytime*np.array([e.time for e in entries]) # Multiply by bytime to flip the order if needed
            order = np.argsort(times)
        else:
            order = np.argsort([e.order for e in entries])
        out = self.output.sort(order, copy=copy)
        if copy:
            return out
        else:
            return self

    def _get_entries(self, bytime, maxentries, skiprun):
        """ Helper function to return correct number of entries in correct order """
        entries = self.sort(bytime=bytime, copy=True)
        entries = entries.values()
        if skiprun:
            entries = [e for e in entries if e.name != self.run_func_name]
        if maxentries:
            if bytime == 1: # Skip the first entries
                entries = entries[-maxentries:]
            elif bytime == -1: # Skip the last entries
                entries = entries[:maxentries]
        return entries

    def disp(self, bytime=1, maxentries=10, skiprun=False):
        """ Display the results of the profiling """
        entries = self._get_entries(bytime, maxentries, skiprun)
        for e in entries:
            sc.heading(f'Profile of {e.name}: {e.time:n} s ({e.percent:n}%)')
            print(e.data)
        return

    def to_df(self, bytime=1, maxentries=None):
        data = []
        entries = self._get_entries(bytime, maxentries, skiprun=False)
        for e in entries:
            cols = ['name', 'time', 'percent', 'function', 'file', 'lineno', 'order']
            row = {col:e[col] for col in cols}
            data.append(row)
        df = sc.dataframe(data)
        self.df = df
        return df

    def plot(self, bytime=1, maxentries=10, figkwargs=None, barkwargs=None):
        """
        Plot the time spent on each function.

        Args:
            bytime (bool): if True, order events by total time rather than actual order
            maxentries (int): how many entries to show
            figkwargs (dict): passed to ``plt.figure()``
            barkwargs (dict): passed to ``plt.bar()``
        """
        # Assemble data
        df = self.to_df(bytime, maxentries)
        ylabels = df.name.values
        if bytime:
            y = np.arange(len(ylabels))
        else:
            y = df.order.values

        x = df.time.values
        pcts = df.percent.values

        if x.max() < 1:
            x *= 1e3
            unit = 'ms'
        else:
            unit = 's'

        # Assemble labels
        for i in range(len(df)):
            timestr = sc.sigfig(x[i], 3) + f' {unit}'
            pctstr = sc.sigfig(pcts[i], 3) + '%'
            ylabels[i] += f'()\n{timestr}, {pctstr}'

        # Trim if needed
        if maxentries:
            x = x[:maxentries]
            y = y[:maxentries]
            ylabels = ylabels[:maxentries]

        # Do the plotting
        barkwargs = sc.mergedicts(barkwargs)
        fig = plt.figure(**sc.mergedicts(figkwargs))
        plt.barh(y, width=x, **barkwargs)
        plt.yticks(y, ylabels)
        plt.xlabel(f'Time ({unit})')
        plt.ylabel('Function')
        plt.grid(True)
        sc.figlayout()
        sc.boxoff()
        return fig


def mprofile(run, follow=None, show_results=True, *args, **kwargs):
    """
    Profile the line-by-line memory required by a function. See profile() for a
    usage example.

    Args:
        run (function): The function to be run
        follow (function): The function or list of functions to be followed in the profiler; if None, defaults to the run function
        show_results (bool): whether to print the statistics of the profile to stdout
        args, kwargs: Passed to the function to be run

    Returns:
        LineProfiler (by default, the profile output is also printed to stdout)
    """

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

    if follow is None: # pragma: no cover
        follow = run

    lp = mp.LineProfiler()
    follow = sc.tolist(follow)
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


class cprofile(sc.prettyobj):
    """
    Function profiler, built off Python's built-in cProfile

    Note: :func:`sc.profile() <profile>` shows the time taken by each line of code, in the order
    the code appears in. :class:`sc.cprofile() <cprofile>` shows the time taken by each function,
    regardless of where in the code it appears.

    The profiler can be used either with the ``enable()`` and ``disable()`` commands,
    or as a context block. See examples below for details.

    Default columns of output are:

        - 'func': the function being called
        - 'cumpct': the cumulative percentage of time taken by this function (including subfunctions)
        - 'selfpct': the percentage of time taken just by this function (excluding subfunctions)
        - 'cumtime': the cumulative time taken by this function
        - 'selftime': the time taken just by this function
        - 'calls': the number of calls
        - 'path': the file and line number

    Args:
        sort (str): the column to sort by (default "cumpct")
        columns (str): what columns to show; options are "default" (above), "brief" (just func, cumtime, selftime), and "full" (as default plus percall and separate line numbers)
        mintime (float): exclude function times below this value
        maxitems (int): only include up to this many functions in the output
        maxfunclen (int): maximum length of the function name to print
        maxpathlen (int): maximum length of the function path to print
        stripdirs (bool): whether to strip folder information from the file paths
        show (bool): whether to show results of the profiling as soon as it's complete

    **Examples**::

        import sciris as sc
        import numpy as np

        class Slow:

            def math(self):
                n = 1_000_000
                self.a = np.arange(n)
                self.b = sum(self.a)

            def plain(self):
                n = 100_000
                self.int_list = []
                self.int_dict = {}
                for i in range(n):
                    self.int_list.append(i)
                    for j in range(10):
                        self.int_dict[i+j] = i+j

            def run(self):
                self.math()
                self.plain()

        # Option 1: as a context block
        with sc.cprofile() as cpr:
            slow = Slow()
            slow.run()

        # Option 2: with start and stop
        cpr = sc.cprofile()
        cpr.start()
        slow = Slow()
        slow.run()
        cpr.stop()

    | *New in version 3.1.6.*
    """
    def __init__(self, sort='cumtime', columns='default', mintime=1e-3, maxitems=100,
                 maxfunclen=40, maxpathlen=40, stripdirs=True, show=True):
        self.sort = sort
        self.columns = columns
        self.mintime = mintime
        self.maxitems = maxitems
        self.maxfunclen = maxfunclen
        self.maxpathlen = maxpathlen
        self.show = show
        self.stripdirs = stripdirs
        self.parsed = None
        self.df = None
        self.profile = cProfile.Profile()
        return

    def parse_stats(self, stripdirs=None, force=False):
        """ Parse the raw data into a dictionary """
        stripdirs = sc.ifelse(stripdirs, self.stripdirs)
        if self.parsed is None or force:
            self.parsed = pstats.Stats(self.profile)
            if stripdirs:
                self.parsed = self.parsed.strip_dirs()
            self.n_functions = len(self.parsed.stats)
            self.total = self.parsed.total_tt
        return

    def to_df(self, sort=None, mintime=None, maxitems=None, maxfunclen=None,
              maxpathlen=None, columns=None):
        """
        Parse data into a dataframe

        See class docstring for arguments
        """
        sort       = sc.ifelse(sort, self.sort)
        mintime    = sc.ifelse(mintime, self.mintime)
        maxitems   = sc.ifelse(maxitems, self.maxitems)
        maxfunclen = sc.ifelse(maxfunclen, self.maxfunclen)
        maxpathlen = sc.ifelse(maxpathlen, self.maxpathlen)
        columns    = sc.ifelse(columns, self.columns)

        def trim(name, maxlen):
            if maxlen is None or len(name) <= maxlen:
                return name
            else:
                return name[:maxlen//2-1] + '...' + name[-maxlen//2-1:]

        # Settings
        cols = dict(
            brief = ['func', 'cumtime', 'selftime'],
            default = ['func', 'cumpct', 'selfpct', 'cumtime', 'selftime', 'calls', 'path'],
            full = ['calls', 'percall', 'selftime', 'cumtime', 'selfpct', 'cumpct', 'func', 'file', 'line'],
        )
        cols = cols[columns]

        # Parse the stats
        self.parse_stats()
        d = sc.dictobj()
        for key in ['calls', 'selftime', 'cumtime', 'file', 'line', 'func']:
            d[key] = []
        for key,entry in self.parsed.stats.items():
            _, ecall, eself, ecum, _ = entry
            if ecum >= mintime:
                efile,eline,efunc = key
                d.calls.append(ecall)
                d.selftime.append(eself)
                d.cumtime.append(ecum)
                d.file.append(efile)
                d.line.append(eline)
                d.func.append(trim(efunc, maxfunclen))

        # Convert to arrays
        for key in ['calls', 'selftime', 'cumtime']:
            d[key] = np.array(d[key])

        # Calculate additional columns
        d.percall = d.cumtime/d.calls
        d.cumpct  = d.cumtime/self.total*100
        d.selfpct = d.selftime/self.total*100
        d.path = []
        for fi,ln in zip(d.file, d.line):
            entry = fi if ln==0 else f'{fi}:{ln}'
            d.path.append(trim(entry, maxpathlen))

        # Convert to a dataframe
        data = {key:d[key] for key in cols}
        self.df = sc.dataframe(**data)
        reverse = sc.isarray(d[sort]) # If numeric, assume we want the highest first
        self.df = self.df.sortrows(sort, reverse=reverse)
        self.df = self.df[:self.maxitems]
        return self.df

    def disp(self, *args, **kwargs):
        """ Display the results of the profiling; arguments are passed to ``to_df()`` """
        if self.df is None or args or kwargs:
            self.to_df(*args, **kwargs)
        self.df.disp()
        print(f'Total time: {self.total:n} s')
        if len(self.df) < self.n_functions:
            print(f'Note: {self.n_functions-len(self.df)} functions with time < {self.mintime} not shown.')
        return

    def start(self):
        """ Start profiling """
        return self.profile.enable()

    def stop(self):
        """ Stop profiling """
        self.profile.disable()
        if self.show:
            self.disp()
        return

    def __enter__(self):
        """ Start profiling in a context block """
        self.start()
        return self

    def __exit__(self, *exc_info):
        """ Stop profiling in a context block """
        self.stop()
        return


def listfuncs(*args, private='__init__', include=None, exclude=None):
    """
    Enumerate all functions in the supplied arguments; used in sc.profile().

    If module(s) are supplied, recursively search themfor functions and classes.
    If class(es) are supplied, search them for methods. Otherwise, search input(s) for
    functions.

    Args:
        args (list): the arguments to parse for functions; can be modules, classes, or functions.
        private (bool/str/list): if True and a class is supplied, follow private functions; if a string/list, follow only those private functions (default ``'__init__'``)
        include (str): if a class/module is supplied, include only functions matching this string
        exclude (str): if a class/module is supplied, exclude functions matching this string

    *New in version 3.2.2.*
    """
    orig_list = sc.mergelists(*args)

    m_list = [] # List of modules (to be turned into functions/classes)
    c_list = [] # List of classes (to be turned into functions)
    f_list = [] # List of functions (to be used)
    include = sc.tolist(include)
    exclude = sc.tolist(exclude)

    # First parse into the different categories
    for obj in orig_list:
        if sc.isfunc(obj): # Simple: just a function
            f_list.append(obj)
        elif isinstance(obj, types.ModuleType):
            m_list.append(obj)
        elif isinstance(obj, type):
            c_list.append(obj)
        else:
            c_list.append(obj.__class__) # Everything is a class

    def get_attrs(parent):
        """ Safely get the attributes of a module/class """
        attrs = sc.objatt(parent, private=private, return_keys=True)
        objs = []
        for attr in attrs:
            try:
                obj = getattr(parent, attr)
                objs.append(obj)
            except: # Don't worry too much if some fail
                pass
        return objs

    # First take a pass and convert any modules to functions and classes
    for mod in m_list:
        objs = get_attrs(mod)
        for obj in objs:
            if sc.isfunc(obj): # It's a function, use it directly
                f_list.append(obj)
            elif isinstance(obj, type): # It's a class, add it to the class list
                c_list.append(obj)

    # Then convert any classes to functions/objects
    for cla in c_list:
        objs = get_attrs(cla)
        for obj in objs:
            if sc.isfunc(obj):
                f_list.append(obj)

    # Finally, do any filtering
    if include:
        f_list = [f for f in f_list if any(inc in str(f) for inc in include)]
    if exclude:
        f_list = [f for f in f_list if not any(exc in str(f) for exc in exclude)]

    return f_list


class tracecalls(sc.prettyobj):
    """
    Trace all function calls.

    Alias to ``sys.steprofile()``.

    Args:
        trace (str/list/regex): the module(s)/file(s) to trace calls from ('' matches all, but this is usually undesirable)
        exclude (str/list/regex): a list of modules/files to exclude (default excludes builtins; set to None to not exclude anything)
        regex (bool): whether to interpret trace and exclude as regexes rather than simple string matching
        repeats (bool): whether to record repeat calls of the same function (default False)
        custom (func): if provided, use this rather than the built in logic for checking for matches
        verbose (bool): how much information to print (False=silent, None=default, True=debug)

    **Examples**::

        import mymodule as mm

        # In context block
        with sc.tracecalls('mymodule'):
            mm.big_operation()

        # Explicitly
        tc = sc.tracecalls('*mysubmodule*', exclude='^init*', regex=True, repeats=True)
        tc.start()
        mm.big_operation()
        tc.stop()
        tc.df.disp()

    | *New in version 3.2.0.*
    | *New in version 3.2.1:* "custom" argument added; "kwargs" removed
    """
    def __init__(self, trace='<default>', exclude='<default>', regex=False, repeats=False,
                 custom=None, verbose=None):
        default_trace   = '.*' if regex else ''
        default_exclude = {'.*<', '.*tracecalls'} if regex else {'<', 'tracecalls'}
        trace   = default_trace   if trace   == '<default>' else trace
        exclude = default_exclude if exclude == '<default>' else exclude
        self.trace   = trace   if isinstance(trace, set)   else set(sc.tolist(trace))
        self.exclude = exclude if isinstance(exclude, set) else set(sc.tolist(exclude))
        self.regex = regex
        self.repeats = repeats
        self.custom = custom
        self.verbose = verbose
        self.parsed = set()
        self.entries = []
        self.entry = sc.dictobj()
        self.entry.stack = 10 # Default stack
        self.n_calls = 0
        return

    def start(self):
        """ Start profiling """
        sys.setprofile(self._call)
        return

    def stop(self, disp=None):
        """ Stop profiling """
        sys.setprofile(None)
        if disp is None:
            disp = self.verbose != False
        self.func_names = set([e['name'] for e in self.entries])
        self.to_df()
        if disp:
            self.disp()
        return

    def __enter__(self):
        """ Start when entering with-as block; start profiling """
        self.start()
        return self

    def __exit__(self, *args):
        """ Stop when leaving a with-as block; stop profiling """
        self.stop()
        return

    def __len__(self):
        return len(self.entries)

    def _call(self, frame, event, arg):
        """ Used internally to track calls """
        self.n_calls += 1
        e = self.entry # Shorten
        if event == 'call':
            e.stack += 1
            e.filename = frame.f_code.co_filename
            e.lineno = str(frame.f_lineno)

            # Check if it's already parsed
            uid = e.filename + e.lineno
            if uid in self.parsed and not self.repeats:
                return

            # If not, keep processing
            e.frame = frame
            e.co_name = frame.f_code.co_name
            e.name = self._get_name()
            e.full_name = '_'.join([e.filename, e.name])
            if self.verbose:
                sc.printgreen(f'Processing call {self.n_calls}:')
                sc.pp(dict(e))
                print()

            # Check if it's excluded, and if not, store it
            if self._check():
                self._store_name()
                self.parsed.add(uid)

        elif event == 'return':
            if self.verbose:
                sc.printgreen(f'  (Returning call {self.n_calls}: {event})')
            e.stack -= 1

        elif self.verbose:
            sc.printgreen(f'  (Skipping call {self.n_calls}: {event})')

        return

    def _check(self):
        """ Used internally to check calls """
        e = self.entry
        if self.custom is not None:
            return self.custom(e)
        elif self.regex:
            allow = any(bool(re.match(token, e.full_name)) for token in self.trace)
            exclude = any(bool(re.match(token, e.full_name)) for token in self.exclude)
        else:
            allow = any(token in e.full_name for token in self.trace)
            exclude = any(token in e.full_name for token in self.exclude)
        result = allow and not exclude
        return result

    def _get_name(self):
        """ Get the name of the class, if available """
        e = self.entry
        f_locals = e.frame.f_locals
        class_name = ''
        if '__class__' in f_locals:
            class_name = f_locals['__class__'].__name__
        elif 'self' in f_locals:
            class_name = f_locals['self'].__class__.__name__

        name = class_name + '.' + e.co_name if class_name else e.co_name
        return name

    def _store_name(self):
        """ Used internally to store the name """
        e = self.entry
        entry = sc.dictobj(name=e.name, filename=e.filename, lineno=e.lineno, stack=e.stack)
        self.entries.append(entry)
        return

    def disp(self, maxlen=60):
        """ Display the results """
        ddf = self.df.copy()
        ddf['indent'] = ['.'*i for i in self.df['stack'].values]
        ddf['label'] = ddf.indent + ddf.name
        maxlen = min(maxlen, max([len(label) for label in ddf.label.values]))
        out = ''
        for i,(label,file,line) in ddf.enumrows(['label', 'filename', 'lineno'], tuple):
            out += f'{i} {label:{maxlen}s} # {file}:L{line}\n'
        print(out)
        return

    def to_df(self):
        """ Convert to a dataframe; if repeats=True, also count repeats """
        df = sc.dataframe(self.entries)
        df['stack'] -= df['stack'].min()
        self.df = df

        # Count duplicates
        if self.repeats:
            cols = ['name', 'filename', 'lineno']
            df_counts = sc.dataframe(self.df.groupby(cols).size().reset_index(name='count'))
            self.df_counts = df_counts

        return df

    def check_expected(self, expected, die=False):
        """
        Check function calls against a list of expected function calls.

        Args:
            expected (set/list/any): if a list of set of strings, check those function names; if object(s) or classes are supplied, check each method
            die (bool): raise an exception if any expected function calls were not called

        **Example**::

            # Check which methods of a class are called
            with sc.tracecalls() as tc:
                my_obj = MyObj()
                my_obj.run()

            expected = tc.check_expected(MyObj) # Equivalent to tc.check_expected(my_obj)
            print(expected)
        """
        # Handle input
        if not isinstance(expected, set):
            raw = sc.tolist(expected)
            expected = []
            for item in raw:
                if isinstance(item, str): # Handle a string
                    expected.append(str)
                    continue
                else: # Handle an object
                    if not isinstance(item, type): # Handle an object instance
                        item = item.__class__
                    cls_name = item.__name__
                    items = item.__dict__.items()
                    methods = [f'{cls_name}.{f_name}' for f_name,obj in items if sc.isfunc(obj)]
                    expected.extend(methods)
            expected = set(expected)

        # Do processing
        out = sc.objdict()
        out.called = expected & self.func_names
        out.not_called = expected - self.func_names
        if die and len(out.not_called):
            errormsg = f'The following {len(out.not_called)} functions were not called:\n{out.not_called}'
            raise RuntimeError(errormsg)

        self.expected = out
        return out



##############################################################################
#%% Resource monitor
##############################################################################

__all__ += ['LimitExceeded', 'resourcemonitor']



class LimitExceeded(MemoryError, KeyboardInterrupt):
    """
    Custom exception for use with the :func:`sc.resourcemonitor() <resourcemonitor>` monitor.

    It inherits from :obj:`MemoryError` since this is the most similar built-in Python
    except, and it inherits from :obj:`KeyboardInterrupt` since this is the means by
    which the monitor interrupts the main Python thread.
    """
    pass


class resourcemonitor(sc.prettyobj): # pragma: no cover # For some reason pycov doesn't catch this class
    """
    Asynchronously monitor resource (e.g. memory) usage and terminate the process
    if the specified threshold is exceeded.

    Args:
        mem (float): maximum virtual memory allowed (as a fraction of total RAM)
        cpu (float): maximum CPU usage (NB: included for completeness only; typically one would not terminate a process just due to high CPU usage)
        time (float): maximum time limit in seconds
        interval (float): how frequently to check memory/CPU usage (in seconds)
        label (str): an optional label to use while printing out progress
        start (bool): whether to start the resource monitor on initialization (else call :meth:`start() <resourcemonitor.start>`)
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
        """
        Start the monitor running

        Args:
            label (str): optional label for printing progress
        """

        def handler(signum, frame): # pragma: no cover
            """ Custom exception handler """
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
        """ Stop the monitor from running """
        if self.verbose:
            print(f'{self.label}: done')
        self.running = False
        try:
            signal.signal(signal.SIGINT, self._orig_sigint) # Restore original KeyboardInterrupt handling
        except ValueError:
            errormsg = 'Could not reset signal, probably not calling from main thread'
            print(errormsg)
        if self.exception is not None and self.die: # This exception has likely already been raised, but if not, raise it now
            raise self.exception # pragma: no cover
        return self


    def __enter__(self, *args, **kwargs):
        """ For use in a context block """
        return self.start()


    def __exit__(self, *args, **kwargs):
        """ For use in a context block """
        return self.stop()


    def monitor(self, label=None, *args, **kwargs):
        """ Actually run the resource monitor """
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
                if self.die: # pragma: no cover
                    self.kill()
            time.sleep(self.interval)

        return


    def check(self):
        """ Check if any limits have been exceeded """
        time_now = time.time()
        self.elapsed = time_now - self.start_time

        # Define the limits
        lim = sc.objdict(
            cpu  = self.cpu,
            mem  = self.mem,
            time = self.time,
        )

        # Check current load
        now = sc.objdict(
            cpu  = cpuload(),
            mem  = memload(),
            time = self.elapsed,
        )

        # Check if limits are OK, and the ratios
        ok    = sc.objdict()
        ratio = sc.objdict()
        for k in lim.keys():
            ok[k]    = now[k] <= lim[k]
            ratio[k] = now[k] / lim[k]
        is_ok = ok[:].all()

        # Gather into output form
        checkdata = sc.objdict(
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
        datalist = sc.autolist()
        if self.cpu < 1:
            datalist += f'CPU: {now.cpu:0.2f} vs {lim.cpu:0.2f}'
        if self.mem < 1:
            datalist += f'Memory: {now.mem:0.2f} vs {lim.mem:0.2f}'
        if np.isfinite(self.time):
            datalist += f'Time: {now.time:n} vs {lim.time:n}'
        datastr = '; '.join(datalist)
        checkstr = f'{prefix}: {datastr}'

        return is_ok, checkdata, checkstr


    def kill(self): # pragma: no cover
        """ Kill all processes """
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
        """ Convert the log into a pandas dataframe """
        entries = []
        for entry in self.log:
            flat = sc.flattendict(entry, sep='_')
            entries.append(flat)
        self.df = pd.DataFrame(entries)
        return self.df
