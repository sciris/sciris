"""
tasks.py -- code related to Sciris task queue management
    
Last update: 2018sep23
"""

import traceback
from functools import wraps
from celery import Celery
from time import sleep
import sciris as sc
import scirisweb as sw
from . import sc_rpcs as rpcs


################################################################################
### Globals
################################################################################

__all__ = ['celery_instance'] # Others for internal use only

datastore = None
task_func_dict = {} # Dictionary to hold registered task functions to be callable from run_task().
RPC_dict = {} # Dictionary to hold all of the registered RPCs in this module.
RPC = rpcs.RPCwrapper(RPC_dict) # RPC registration decorator factory created using call to make_RPC().
celery_instance = None # Celery instance.



################################################################################
### Classes
################################################################################

__all__ += ['Task']


class Task(sc.prettyobj):
    '''
    A Sciris record for an asynchronous task.
    
    Attributes:
        task_id (str)         -- the ID / name for the task that typically is chosen by the client
        status (str)          -- the status of the task:
                                    'unknown': unknown status, for example, just initialized
                                    'error':   task failed with an actual error
        error_text (str)      -- string giving an idea of what error has transpired
        func_name (str)       -- string of the function name for what's called
        args (list)           -- list containing args for the function
        kwargs (dict)         -- dict containing kwargs for the function
        result_id (str)       -- string for the Redis ID of the AsyncResult
        queue_time (datetime) -- the time the task was queued for Celery
        start_time (datetime) -- the time the task was actually started
        stop_time (datetime)  -- the time the task completed
        pending_time (int)    -- the time the process has been waiting to be executed on the server in seconds        
        execution_time (int)  -- the time the process required to complete
    '''
    
    def  __init__(self, task_id):
        self.task_id        = task_id # Set the task ID (what the client typically knows as the task).
        self.uid            = task_id # Make it the same as the task ID...WARNING, fix
        self.status         = 'unknown' # Start the status at 'unknown'.
        self.error_text     = None # Start the error_text at None.
        self.func_name      = None # Start the func_name at None.
        self.args           = None # Start the args and kwargs at None.
        self.kwargs         = None
        self.result_id      = None # Start with no result_id.
        self.queue_time     = None # Start the queue, start, and stop times at None.
        self.start_time     = None
        self.stop_time      = None
        self.pending_time   = None # Start the pending and execution times at None.
        self.execution_time = None
        return None
    
    def show(self):
        print('-----------------------------')
        print('        Task ID: %s'   % self.task_id)
        print('         Status: %s'   % self.status)
        print('     Error text: %s'   % self.error_text)
        print('  Function name: %s'   % self.func_name)
        print('  Function args: %s'   % self.args)
        print('Function kwargs: %s'   % self.kwargs)        
        print('      Result ID: %s'   % self.result_id)
        print('     Queue time: %s'   % self.queue_time)        
        print('     Start time: %s'   % self.start_time)
        print('      Stop time: %s'   % self.stop_time)
        print('   Pending time: %s s' % self.pending_time)        
        print(' Execution time: %s s' % self.execution_time)  
        print('-----------------------------')
    
    def jsonify(self):
        output = {'task':
                     {'UID':           self.uid,                    
                      'taskId':        self.task_id,
                      'status':        self.status,
                      'errorText':     self.error_text,
                      'funcName':      self.func_name,
                      'funcArgs':      self.args,
                      'funcKwargs':    self.kwargs,
                      'resultId':      self.result_id,
                      'queueTime':     self.queue_time,                
                      'startTime':     self.start_time,
                      'stopTime':      self.stop_time,
                      'pendingTime':   self.pending_time,
                      'executionTime': self.execution_time     
                      }
                  }
        return output



################################################################################
### Functions
################################################################################

__all__ += ['get_datastore', 'make_celery', 'add_task_funcs', 'check_task', 'get_task_result', 'delete_task', 'taskwrapper']


def get_datastore(config=None):
    ''' Only if we have not already done so, create the DataStore object, setting up Redis. '''
    from . import sc_datastore as ds # This needs to be here to avoid a circular import
    try:
        datastore = sw.flaskapp.datastore
        assert datastore is not None
    except:
        if isinstance(config, dict): redis_url = config['REDIS_URL']
        else:                        redis_url = config.REDIS_URL
        datastore = ds.DataStore(redis_url=redis_url)
    return datastore


# Function for creating the Celery instance, resetting the global and also 
# passing back the same result. for the benefit of callers in non-Sciris 
# modules.
def make_celery(config=None, verbose=True):
    global celery_instance
    global run_task_lock
    global datastore # So it's accessible in other functions

    run_task_lock = False
    celery_instance = Celery('tasks') # Define the Celery instance.
    if config is not None: # Configure Celery with config.py.
        celery_instance.config_from_object(config)
    datastore = get_datastore(config=config)

    # Define subfunctions available to the Celery instance
    
    def lock_run_task(task_id):
        global run_task_lock
        sleepduration = 5 # Define how lon to sleep before trying again
        if verbose: print('C>> Checking lock on run_task() for %s' % task_id)
        while run_task_lock: # Until there is no lock, sit and sleep.
            if verbose: print('C>> Detected lock on %s. Waiting...' % task_id)
            sleep(sleepduration)  # sleep before trying again
        if verbose: print('C>> No lock detected on %s, locking run_task()' % task_id)
        run_task_lock = True # Set the lock to keep other run_task() instances from co-occurring on the same Celery worker.
        return None
    
    
    def unlock_run_task(task_id):
        global run_task_lock
        if verbose: print('C>> Unlocking run_task() for %s' % task_id)
        run_task_lock = False # Remove the lock for this Celery worker.
        return None
    
    
    @celery_instance.task
    def run_task(task_id, func_name, args, kwargs):
        if kwargs is None: kwargs = {} # So **kwargs works below
        
        if verbose: print('C>> Starting run_task() for %s' % task_id)
        
        # We need to load in the whole DataStore here because the Celery worker 
        # (in which this function is running) will not know about the same context 
        # from the datastore.py module that the server code will.
        
        # Check if run_task() locked and wait until it isn't, then lock it for 
        # other run_task() instances in this Celery worker.
        lock_run_task(task_id)
        
        # Find a matching task record (if any) to the task_id.
        match_taskrec = datastore.loadtask(task_id)
        if match_taskrec is None:
            if verbose: print('C>> Failed to find task record for %s' % task_id)
            unlock_run_task(task_id)
            return { 'error': 'Could not access Task' }
    
        # Set the TaskRecord to indicate start of the task.
        match_taskrec.status = 'started'
        match_taskrec.start_time = sc.now()
        match_taskrec.pending_time = (match_taskrec.start_time - match_taskrec.queue_time).total_seconds()
            
        # Do the actual update of the TaskRecord.
        # NOTE: At the moment the TaskDict on disk is not modified here, which 
        # which is a good thing because that could disrupt actiities in other 
        # run_task() instances.
        datastore.savetask(match_taskrec)
        if verbose: print('C>> Saved task for %s' % task_id)
        
        # Make the actual function call, inside a try block in case there is 
        # an exception thrown.
        # NOTE: This block is likely to run for several seconds or even 
        # minutes or hours, depending on the task.
        try:
            result = task_func_dict[func_name](*args, **kwargs)
            match_taskrec.status = 'completed'
            if verbose: print('C>> Successfully completed task %s! :)' % task_id)
        except Exception: # If there's an exception, grab the stack track and set the TaskRecord to have stopped on in error.
            error_text = traceback.format_exc()
            match_taskrec.status = 'error'
            match_taskrec.error_text = error_text
            result = error_text
            if verbose: print('C>> Failed task %s! :(' % task_id)
        
        # Set the TaskRecord to indicate end of the task.
        # NOTE: Even if the browser has ordered the deletion of the task 
        # record, it will be "resurrected" during this update, so the 
        # delete_task() RPC may not always work as expected.
        match_taskrec.stop_time = sc.now()
        match_taskrec.execution_time = (match_taskrec.stop_time - match_taskrec.start_time).total_seconds()
        
        # Do the actual update of the TaskRecord.  Do this in a try / except 
        # block because this step may fail.  For example, if a TaskRecord is 
        # deleted by the webapp, the update here will crash.
        # NOTE: At the moment the TaskDict on disk is not modified here, which 
        # which is a good thing because that could disrupt actiities in other 
        # run_task() instances.
        try:
            datastore.savetask(match_taskrec)         
        except Exception:
            error_text = traceback.format_exc()
            match_taskrec.status = 'error'
            match_taskrec.error_text = error_text
            result = error_text
            if verbose: print('C>> Failed to save task %s! :(' % task_id)            
            
        if verbose: print('C>> End of run_task() for %s' % task_id)
        
        # Unlock run-task() for other run_task() instances running on the same 
        # Celery worker.
        unlock_run_task(task_id)
        
        # Return the result.
        return result 
    
    # The launch-task() RPC is the only one included here because it is the 
    # only one that makes a direct call to run_task().  Any other RPCs that 
    # would call run_task() would have to be placed in make_celery() 
    # as well.
    @RPC(validation='named') 
    def launch_task(task_id='', func_name='', args=[], kwargs={}):
        
        match_taskrec = datastore.loadtask(task_id) # Find a matching task record (if any) to the task_id.
        
        if match_taskrec is None: # If we did not find a match...
            if not func_name in task_func_dict: # If the function name is not in the task function dictionary, return an error.
                return_dict = {'error': 'Could not find requested async task function "%s"' % func_name}
            
            else:
                new_task_record = Task(task_id) # Create a new TaskRecord.
                
                # Initialize the TaskRecord with available information.
                new_task_record.status = 'queued'
                new_task_record.queue_time = sc.now()
                new_task_record.func_name = func_name
                new_task_record.args = args
                new_task_record.kwargs = kwargs
                datastore.savetask(new_task_record)  # Add the TaskRecord to the TaskDict.
                
                # Queue up run_task() for Celery.
                my_result = run_task.delay(task_id, func_name, args, kwargs)
                new_task_record.result_id = my_result.id # Add the result ID to the TaskRecord, and update the DataStore.
                datastore.savetask(new_task_record)
                return_dict = new_task_record.jsonify() # Create the return dict from the user repr.
        
        else: # Otherwise (there is a matching task)...
            if match_taskrec.status == 'completed' or match_taskrec.status == 'error': # If the TaskRecord indicates the task has been completed or thrown an error...   
                if match_taskrec.result_id is not None: # If we have a result ID, erase the result from Redis.
                    result = celery_instance.AsyncResult(match_taskrec.result_id)
                    result.forget()
                    match_taskrec.result_id = None
                           
                # Initialize the TaskRecord to start the task again (though possibly with a new function and arguments).
                match_taskrec.status = 'queued'
                match_taskrec.queue_time = sc.now()
                match_taskrec.start_time = None
                match_taskrec.stop_time = None
                match_taskrec.pending_time = None
                match_taskrec.execution_time = None                
                match_taskrec.func_name = func_name
                match_taskrec.args = args
                match_taskrec.kwargs = kwargs
                
                # Queue up run_task() for Celery.   
                my_result = run_task.delay(task_id, func_name, args, kwargs)             
                match_taskrec.result_id = my_result.id # Add the new result ID to the TaskRecord, and update the DataStore.
                datastore.savetask(match_taskrec)
                return_dict = match_taskrec.jsonify() # Create the return dict from the user repr.
            
            else: # Else (the task is not completed)...
                return_dict = {'error': 'Task is already %s' % match_taskrec.status}
        
        return return_dict   # Return our result.
    
    return celery_instance # Return the new instance.


def add_task_funcs(new_task_funcs):
    ''' Function for adding new task functions to those that run_task() can see '''
    global task_func_dict
    for key in new_task_funcs: # For all of the keys in the dict passed in, put the key/value pairs in the global dict.
        task_func_dict[key] = new_task_funcs[key]
    return None


@RPC(validation='named') 
def check_task(task_id, verbose=False): 
    match_taskrec = datastore.loadtask(task_id) # Find a matching task record (if any) to the task_id.
    if match_taskrec is None: # Check to see if the task exists, and if not, return an error.
        errormsg = {'error': 'No task found for specified task ID (%s)' % task_id}
        if verbose: print(errormsg)
        return errormsg
    else: # Update the elapsed times.
        if match_taskrec.pending_time is not None: # If we are no longer pending...
            pending_time = match_taskrec.pending_time # Use the existing pending_time.
            if match_taskrec.execution_time is not None: # If we have finished executing...
                execution_time = match_taskrec.execution_time # Use the execution time in the record.
            else: # Else (we are still executing)...
                execution_time = (sc.now() - match_taskrec.start_time).total_seconds()
        else: # Else (we are still pending)...
            pending_time = (sc.now() - match_taskrec.queue_time).total_seconds()
            execution_time = 0
        taskrec_dict = match_taskrec.jsonify() # Create the return dict from the user repr.
        taskrec_dict['pendingTime'] = pending_time
        taskrec_dict['executionTime'] = execution_time
        if verbose: sc.pp(taskrec_dict)
        return taskrec_dict    # Return the has record information and elapsed times.     
   
   
@RPC(validation='named') 
def get_task_result(task_id):
    match_taskrec = datastore.loadtask(task_id) # Find a matching task record (if any) to the task_id.
    if match_taskrec is None: # Check to see if the task exists, and if not, return an error.
        return {'error': 'No task found for specified task ID'}
    else:
        if match_taskrec.result_id is not None: # If we have a result ID...
            result = celery_instance.AsyncResult(match_taskrec.result_id) # Get the result itself from Celery.
            if not result.ready(): # If the result is not ready, return an error.
                return {'error': 'Task not completed'}
            elif result.failed(): # Else, if we have failed, return the exception.
                return {'error': 'Task failed with an exception'}
            else: # Else (task is ready)...
                return {'result': result.get()}
        else: # Else (no result ID)...
            return {'error': 'No result ID'}
    
    
@RPC(validation='named') 
def delete_task(task_id): 
    match_taskrec = datastore.loadtask(task_id) # Find a matching task record (if any) to the task_id.
    if match_taskrec is None: # Check to see if the task exists, and if not, return an error.
        return {'error': 'No task found for specified task ID'}
    else: # Otherwise (matching task).
        if match_taskrec.result_id is not None: # If we have a result ID, erase the result from Redis.
            result = celery_instance.AsyncResult(match_taskrec.result_id)
            result.revoke(terminate=True) # This commmand works under Linux, but not under the Windows Celery setup.  It allows terminating a task in mid-run.
            result.forget()
        datastore.delete(task_id, objtype='task') # Erase the TaskRecord.
        return 'success'


def taskwrapper(task_func_dict):
    ''' Function for making a register_async_task decorator in other modules '''
    def task_func_decorator(task_func):
        @wraps(task_func)
        def wrapper(*args, **kwargs):        
            output = task_func(*args, **kwargs)
            return output
            
        task_func_dict[task_func.__name__] = task_func # Add the function to the dictionary.
        return wrapper    
    
    return task_func_decorator

