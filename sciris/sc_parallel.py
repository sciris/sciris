"""
Version:
"""

import time
import psutil
import multiprocessing as mp
import numpy as np
from . import sc_utils as ut




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
        loadbalancer() # Simplest usage -- delay while load is >50%
        for nproc in processlist: loadbalancer(maxload=0.9, index=nproc) # Use a maximum load of 90%, and stagger the start by process number

    Version: 2018nov01
     '''
    # Set up processes to start asynchronously
    if maxload  is None: maxload = 0.8
    if interval is None: interval = 0.5
    if maxtime  is None: maxtime = 36000
    if label    is None: label = ''
    else: label += ': '
    if index is None:
        pause = np.random.rand()*interval
        index = ''
    else:              
        pause = index*interval
    if maxload>1: maxload/100. # If it's >1, assume it was given as a percent
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



def parallelize(func=None, iterarg=None, itervar=None, args=None, kwargs=None, ncpus=None, maxload=None):
    '''
    Shortcut for parallelizing a function. iterarg can be an iterable or an integer;
    if the latter, it will run the function that number of times.
    
    The function can either handle a fixed number of CPUs or allocate dynamically based
    on load. If ncpus is None and maxload is None, then it will use the number of CPUs
    returned by multiprocessing; if ncpus is not None, it will use the specified number;
    if maxload is not None, it will allocate dynamically.
    
    Example 1:
        def f(x):
            return x*x
        
        results = parallelize(func=f, iterarg=[1,2,3], ncpus=3)
    
    Example 2:
        import pylab as pl
        def myfunc(x, y, i):
            xy = [x+i*pl.rand(1000), y+i*pl.randn(1000)]
            return xy
        n = 5
        xy = parallelize(myfunc, kwargs={'x':3, 'y':8}, iterarg=n, itervar='i', maxload=0.8)
        for i in range(n):
            pl.scatter(xy[i][0], xy[i][1])
    '''
    # Handle maxload
    if ncpus is None and maxload is None:
        ncpus = mp.cpu_count()
    
    # Handle iterarg
    if not(ut.isiterable(iterarg)):
        try:
            iterarg = np.arange(iterarg)
        except Exception as E:
            errormsg = 'Could not understand iterarg "%s": not iterable and not an integer: %s' % (iterarg, str(E))
            raise Exception(errormsg)
    
    # Construct argument list
    argslist = []
    niters = len(iterarg)
    for i,ival in enumerate(iterarg):
        args = ({'func':func, 'index':i, 'iterval':ival, 'itervar':itervar, 'args':args, 'kwargs':kwargs, 'maxload':maxload},)
        argslist.append(args)
        
    # Decide how to run -- not load-balanced, use map
    if ncpus:
        multipool = mp.Pool(processes=ncpus)
        outputlist = multipool.map(parallel_task, argslist)
        return outputlist
    
    # It is load-balanced, use Process/Queue
    else:
        outputqueue = mp.Queue()
        outputlist = np.empty(niters, dtype=object)
        processes = []
        for i in range(niters):
            prc = mp.Process(target=parallel_task, args=(argslist[i], outputqueue))
            prc.start()
            processes.append(prc)
        for i in range(niters):
            _i,returnval = outputqueue.get()
            outputlist[_i] = returnval
        for prc in processes:
            prc.join() # Wait for them to finish
        
        outputlist = outputlist.tolist()
        
        return outputlist



def parallel_task(taskargs):
    ''' Task called by parallelize() -- not to be called directly '''
    
    # Handle inputs
    taskargs = taskargs[0]
    func = taskargs['func']
    index = taskargs['index']
    args = taskargs['args']
    kwargs = taskargs['kwargs']
    if args   is None: args   = ()
    if kwargs is None: kwargs = {}
    if taskargs['itervar'] is None:
        args = (taskargs['iterval'],) + args # If variable name is not supplied, prepend it to args
    else: 
        kwargs[taskargs['itervar']] = taskargs['iterval'] # Otherwise, include it in kwargs
    
    # Handle load balancing
    maxload = taskargs['maxload']
    if maxload:
        loadbalancer(maxload=maxload, index=index)
    
    # Call the function
    print('MY NAME IS ISHMAEL')
    print(args)
    print(kwargs)
    print('DONE')
    output = func(*args, **kwargs)
        
    # Handle output
    outputqueue = None
    if outputqueue:
        outputqueue.put((index,output))
        return None
    else:
        return output
    
    


def parallelcmd(cmd=None, parfor=None, returnval=None, maxload=None, **kwargs):
    '''
    A function to parallelize any block of code. Yes. seriously.
    
    Arguments:
        cmd -- a string representation of the code to be run in parallel
        parfor -- a dictionary of lists of the variables to loop over
        returnval -- the name of the output variable
        maxload -- the maxmium CPU load, used in loadbalancer()
        **kwargs -- variables to pass into the code
    
    Example:
        vals = [3,5,9]
        const = 4
        parfor = {'val':vals}
        returnval = 'result'
        cmd = """
newval = val+const # Note that this can't be indented
result = newval**2
        """
        results = parallelize(cmd=cmd, parfor=parfor, returnval=returnval, const=const)
        '''
    
    nfor = len(list(parfor.values())[0])
    outputqueue = mp.Queue()
    outputlist = np.empty(nfor, dtype=object)
    processes = []
    for i in range(nfor):
        prc = mp.Process(target=parallelcmd_task, args=(cmd, parfor, returnval, i, outputqueue, maxload, kwargs))
        prc.start()
        processes.append(prc)
    for i in range(nfor):
        _i,returnval = outputqueue.get()
        outputlist[_i] = returnval
    for prc in processes:
        prc.join() # Wait for them to finish
    
    outputlist = outputlist.tolist()
    
    return outputlist


def parallelcmd_task(_cmd, _parfor, _returnval, _i, _outputqueue, _maxload, _kwargs):
    '''
    The task to be executed by parallelcmd(). All internal variables start with
    underscores to avoid possible collisions in the exec() statements. Not to be called
    directly.
    '''
    loadbalancer(maxload=_maxload, index=_i)
    
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
    
    print('...done.')
    return None