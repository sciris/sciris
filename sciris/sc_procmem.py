'''
Memory monitoring functions
NB: Uses ``multiprocess`` instead of ``multiprocessing`` under the hood for
broadest support  across platforms (e.g. Jupyter notebooks).

'''
import contextlib
import psutil
import multiprocess
import tracemalloc
import sys
import resource

from . import sc_utils as scu

##############################################################################
# %% Memory managment functions
##############################################################################


__all__ = ['limit_malloc', 'get_free_memory', 'change_resource_limit', 'memory']
__all__ += ['ResourceLimit']


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
            limit = limit >> 30
        return limit, unit


def take_mem_snapshot():
    """
    Take a snapshot of current memory usage (in %) via psuti
    Arguments: None

    Returns: a float between 0-1 representing the fraction of psutil.virtuak memory currently used.
    """
    snapshot = psutil.virtual_memory().percent / 100.0
    return snapshot


def get_free_memory():
    """
    Helper function to get amount of free memory

    Returns:
        Free memmory expressed in kb
    """
    if not(scu.islinux()):
        return

    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in 'MemTotal:':
                free_memory += int(sline[1])
        return free_memory  # expressed in kB


def change_resource_limit(percentage):
    """
    Helper function to change the the (soft) limit of address memory (resource.RLIMIT_AS)
    as a fraction of current available free memory.

    TODO: to generalise to any resource?

    Arguments:
        percentage: a float between 0 and 1
    """
    if scu.iswindows():
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_free_memory() * 1024 * percentage, hard))


def memory(percentage=0.8):
    """

    Arguments:
        percentage: a float between 0 and 1

    @memory(percentage=0.8)
    def main():
        print('My memory is limited to 80% of free memory.')
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            change_resource_limit(percentage)
            try:
                return function(*args, **kwargs)
            except MemoryError:
                mem = get_free_memory() / 1024 / 1024
                print('Remain: %.2f GB' % mem)
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                sys.exit(1)
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
