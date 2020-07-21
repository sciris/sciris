##############################################################################
### IMPORTS
##############################################################################

import time
import psutil
import multiprocessing as mp
import numpy as np
from . import sc_utils as ut



##############################################################################
### PARALLELIZATION FUNCTIONS
##############################################################################

__all__ = ['loadbalancer', 'parallelize', 'parallelcmd']


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

    Usage examples:
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
        return None # Return immediately if no max load
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
            if verbose: print(label+'CPU load too high (%0.2f/%0.2f); process %s queued %i times' % (currentload, maxload, index, count))
            time.sleep(interval*2*np.random.rand()) # Sleeps for an average of refresh seconds, but do it randomly so you don't get locking
        else: 
            toohigh = False 
            if verbose: print(label+'CPU load fine (%0.2f/%0.2f), starting process %s after %i tries' % (currentload, maxload, index, count))
    return None



def parallelize(func, iterarg=None, iterkwargs=None, args=None, kwargs=None, ncpus=None, maxload=None, interval=None, **func_kwargs):
    '''
    Shortcut for parallelizing a function. Most simply, acts as an shortcut for using
    multiprocessing.Pool() or Queue(). However, this function can also iterate over more
    complex arguments.
    
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
    
    Example 1 -- simple usage as a shortcut to multiprocessing.map():
        def f(x):
            return x*x
        
        results = sc.parallelize(f, [1,2,3])
    
    Example 2 -- simple usage for "embarrassingly parallel" processing:
        def rnd():
            import pylab as pl
            return pl.rand()
        
        results = sc.parallelize(rnd, 10, ncpus=4)
    
    Example 3 -- three different equivalent ways to use multiple arguments:
        def f(x,y):
            return x*y
        
        results1 = sc.parallelize(func=f, iterarg=[(1,2),(2,3),(3,4)])
        results2 = sc.parallelize(func=f, iterkwargs={'x':[1,2,3], 'y':[2,3,4]})
        results3 = sc.parallelize(func=f, iterkwargs=[{'x':1, 'y':2}, {'x':2, 'y':3}, {'x':3, 'y':4}])
        assert results1 == results2 == results3
    
    Example 4 -- using non-iterated arguments and dynamic load balancing:
        def myfunc(i, x, y):
            xy = [x+i*pl.randn(100), y+i*pl.randn(100)]
            return xy
        
        xylist1 = sc.parallelize(myfunc, kwargs={'x':3, 'y':8}, iterarg=range(5), maxload=0.8, interval=0.2) # Use kwargs dict
        xylist2 = sc.parallelize(myfunc, x=5, y=10, iterarg=[0,1,2]) # Supply kwargs directly
        
        for p,xylist in enumerate([xylist1, xylist2]):
            pl.subplot(2,1,p+1)
            for i,xy in enumerate(reversed(xylist)):
                pl.scatter(xy[0], xy[1], label='Run %i'%i)
            pl.legend()
    
    Note: to use on Windows, parallel calls must contained with an `if __name__ == '__main__'` block,
    for example:

        import sciris as sc

        def f(x,y):
            return x*y

        if __name__ == '__main__':   
            results = sc.parallelize(func=f, iterarg=[(1,2),(2,3),(3,4)])
            print(results)
    
    Version: 2019dec03
    '''
    # Handle maxload
    if ncpus is None and maxload is None:
        ncpus = mp.cpu_count()
    
    # Handle kwargs
    if func_kwargs:
        if kwargs is None:
            kwargs = func_kwargs
        else:
            kwargs.update(func_kwargs)
    
    # Handle iterarg and iterkwargs
    niters = 0
    embarrassing = False # Whether or not it's an embarrassingly parallel optimization
    if iterarg is not None and iterkwargs is not None:
        errormsg = 'You can only use one of iterarg or iterkwargs as your iterable, not both'
        raise Exception(errormsg)
    if iterarg is not None:
        if not(ut.isiterable(iterarg)):
            try:
                iterarg = np.arange(iterarg)
                embarrassing = True
            except Exception as E:
                errormsg = 'Could not understand iterarg "%s": not iterable and not an integer: %s' % (iterarg, str(E))
                raise Exception(errormsg)
        niters = len(iterarg)
    if iterkwargs is not None: # Check that iterkwargs has the right format
        if isinstance(iterkwargs, dict): # It's a dict of lists, e.g. {'x':[1,2,3], 'y':[2,3,4]}
            for key,val in iterkwargs.items():
                if not ut.isiterable(val):
                    errormsg = 'iterkwargs entries must be iterable, not %s' % type(val)
                    raise Exception(errormsg)
                if not niters:
                    niters = len(val)
                else:
                    if len(val) != niters:
                        errormsg = 'All iterkwargs iterables must be the same length, not %s vs. %s' % (niters, len(val))
                        raise Exception(errormsg)
        elif isinstance(iterkwargs, list): # It's a list of dicts, e.g. [{'x':1, 'y':2}, {'x':2, 'y':3}, {'x':3, 'y':4}]
            niters = len(iterkwargs)
            for item in iterkwargs:
                if not isinstance(item, dict):
                    errormsg = 'If iterkwargs is a list, each entry must be a dict, not %s' % type(item)
                    raise Exception(errormsg)
        else:
            errormsg = 'iterkwargs must be a dict of lists, a list of dicts, or None, not %s' % type(iterkwargs)
            raise Exception(errormsg)
    
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
            else: # Should be caught by previous checking, so shouldn't happen
                errormsg = 'iterkwargs type not understood (%s)' % type(iterkwargs)
                raise Exception(errormsg)
        taskargs = TaskArgs(func, index, iterval, iterdict, args, kwargs, maxload, interval, embarrassing)
        argslist.append(taskargs)
        
    # Run simply using map -- no advantage here to using Process/Queue
    multipool = mp.Pool(processes=ncpus)
    outputlist = multipool.map(parallel_task, argslist)
    multipool.close()
    multipool.join()

    return outputlist



def parallelcmd(cmd=None, parfor=None, returnval=None, maxload=None, interval=None, **kwargs):
    '''
    A function to parallelize any block of code. Note: this is intended for quick
    prototyping; since it uses exec(), it is not recommended for use in production
    code.
    
    Arguments:
        cmd -- a string representation of the code to be run in parallel
        parfor -- a dictionary of lists of the variables to loop over
        returnval -- the name of the output variable
        maxload -- the maxmium CPU load, used in loadbalancer()
        **kwargs -- variables to pass into the code
    
    Example:
        const = 4
        parfor = {'val':[3,5,9]}
        returnval = 'result'
        cmd = """
newval = val+const # Note that this can't be indented
result = newval**2
        """
        results = parallelcmd(cmd=cmd, parfor=parfor, returnval=returnval, const=const)
        
    Version: 2018nov01
    '''
    
    nfor = len(list(parfor.values())[0])
    outputqueue = mp.Queue()
    outputlist = np.empty(nfor, dtype=object)
    processes = []
    for i in range(nfor):
        prc = mp.Process(target=parallelcmd_task, args=(cmd, parfor, returnval, i, outputqueue, maxload, interval, kwargs))
        prc.start()
        processes.append(prc)
    for i in range(nfor):
        _i,returnval = outputqueue.get()
        outputlist[_i] = returnval
    for prc in processes:
        prc.join() # Wait for them to finish
    
    outputlist = outputlist.tolist()
    
    return outputlist
    


##############################################################################
### HELPER FUNCTIONS/CLASSES
##############################################################################

class TaskArgs(ut.prettyobj):
        ''' A class to hold the arguments for the task -- must match both parallelize() and parallel_task() '''
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
            return None



def parallel_task(taskargs, outputqueue=None):
    ''' Task called by parallelize() -- not to be called directly '''
    
    # Handle inputs
    taskargs = ut.dcp(taskargs)
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
        return None
    else:
        return output


def parallelcmd_task(_cmd, _parfor, _returnval, _i, _outputqueue, _maxload, _interval, _kwargs):
    '''
    The task to be executed by parallelcmd(). All internal variables start with
    underscores to avoid possible collisions in the exec() statements. Not to be called
    directly.
    '''
    loadbalancer(maxload=_maxload, index=_i, interval=_interval)
    
    # Set the loop variables
    for _key in _parfor.keys(): 
        _thisval = _parfor[_key][_i] # analysis:ignore
        exec('%s = _thisval' % _key) # Set the value of this variable
    
    # Set the keyword arguments
    for _key in _kwargs.keys(): 
        _thisval = _kwargs[_key] # analysis:ignore
        exec('%s = _thisval' % _key) # Set the value of this variable
    
    # Calculate the command
    try:
        exec(_cmd) # The meat of the matter!
    except Exception as E:
        print('WARNING, parallel task failed:\n%s' % str(E))
        exec('%s = None' % _returnval)
    
    # Append results
    _outputqueue.put((_i,eval(_returnval)))
    
    return None