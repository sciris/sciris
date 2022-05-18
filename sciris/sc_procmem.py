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
    def __init__(self, percentage_limit):
        self.percentage_limit = percentage_limit

    def __enter__(self):
        self.old_heap_limit = resource.getrlimit(resource.RLIMIT_DATA)
        self.old_address_limit = resource.getrlimit(resource.RLIMIT_AS)

        for rsrc in (resource.RLIMIT_DATA, resource.RLIMIT_DATA):
            totalmem = psutil.virtual_memory().available
            hard = int(round(totalmem * 0.8))
            soft = hard
            resource.setrlimit(rsrc, (soft, hard))  # limit to 80% of total memory

    def __exit__(self, type, value, tb):
        resource.setrlimit(resource.RLIMIT_DATA, self.old_heap_limit)
        resource.setrlimit(resource.RLIMIT_AS, self.old_address_limit)


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
            if str(sline[0]) in ('MemTotal:'):
                free_memory += int(sline[1])
        return free_memory # expressed in kB


def change_resource_limit(percentage):
    """
    Helper function to change the the limit of address memory (resource.RLIMIT_AS)
    as a fraction of current limit.

    TODO: to generalise to any resource?

    Arguments:
        percentage: a float between 0 and 1 -- though could be higher than 1 if we had previosuly decreased the limit
    """
    if scu.iswindows():
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_free_memory() * 1024 * percentage, hard))


def memory(percentage=0.8):
    def decorator(function):
        def wrapper(*args, **kwargs):
            change_resource_limit(percentage)
            try:
                return function(*args, **kwargs)
            except MemoryError:
                mem = get_free_memory() / 1024 /1024
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

