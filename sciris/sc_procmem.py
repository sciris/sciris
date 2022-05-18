"""
Memory monitoring functions
"""

import contextlib
import psutil
from multiprocess import Process
import tracemalloc
import resource


##############################################################################
# %% Memory managment functions
##############################################################################


__all__ = ['limit_malloc', 'memory', 'ResourceLimit', 'MemoryMonitor']

class ResourceLimit:
    """
    DOCME
    """
    def __init__(self, percentage_limit, verbose=False):
        self.percentage_limit = percentage_limit
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


class MemoryMonitor(Process):
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
        Process.__init__(self, name='MemoryLimiter')
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
    Take a snapshot of current memory usage (in %) via psuti
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

    Example:

    import sciris as sc
    with sc.limit_malloc(500):
       l1 = []
       l1.append(x for x in range(200000))

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
            raise AttributeError(f'Memory usage exceeded the threshold: '
                                 f'{current_size} > {size}')
    finally:
        tracemalloc.stop()
