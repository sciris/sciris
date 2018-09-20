"""
tasks.py -- code related to Sciris task queue management
    
Last update: 2018sep02
"""

import traceback
from functools import wraps
from celery import Celery
from time import sleep
import sciris as sc
from . import sc_datastore as ds
from . import sc_rpcs as rpcs


################################################################################
### Globals
################################################################################

__all__ = ['task_dict', 'celery_instance'] # Others for internal use only

task_dict = None # The TaskDict object for all of the app's existing asynchronous tasks. Gets initialized by and loaded by init_tasks().
task_func_dict = {} # Dictionary to hold registered task functions to be callable from run_task().
RPC_dict = {} # Dictionary to hold all of the registered RPCs in this module.
RPC = rpcs.makeRPCtag(RPC_dict) # RPC registration decorator factory created using call to make_RPC().
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
        output = {'UID':           self.uid,                    
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
        return output



################################################################################
### Functions
################################################################################

__all__ += ['make_celery_instance', 'add_task_funcs', 'check_task', 'get_task_result', 'delete_task', 'make_async_tag']
        
# Function for creating the Celery instance, resetting the global and also 
# passing back the same result. for the benefit of callers in non-Sciris 
# modules.
def make_celery_instance(config=None):
    global celery_instance
    global run_task_lock
    
    run_task_lock = False
    
    # Define the Celery instance.
    celery_instance = Celery('tasks')
    
    # Configure Celery with config.py.
    if config is not None:
        celery_instance.config_from_object(config)
    
    # Configure so that the actual start of a task is tracked.
    # This may work only under version 3.1.25
#    celery_instance.conf.CELERY_TRACK_STARTED = True
    
    def lock_run_task(task_id):
        global run_task_lock
        
        print('>>> CHECKING LOCK ON run_task() FOR %s' % task_id)
        
        # Until there is no lock, sit and sleep.
        while run_task_lock:
            print('>>> DETECTED LOCK ON %s.  WAITING...' % task_id)
            sleep(5)  # sleep 5 seconds before trying again
        
        print('>>> NO LOCK DETECTED ON %s' % task_id)
        print('>>> LOCKING run_task() ON %s' % task_id)
        
        # Set the lock to keep other run_task() instances from co-occurring on 
        # the same Celery worker.
        run_task_lock = True
        
    def unlock_run_task(task_id):
        global run_task_lock
        
        print('>>> UNLOCKING run_task() FOR %s' % task_id)
        # Remove the lock for this Celery worker.
        run_task_lock = False
        
    @celery_instance.task
    def run_task(task_id, func_name, args, kwargs):
        if kwargs is None: kwargs = {} # So **kwargs works below
        
        print('>>> START OF run_task() FOR %s' % task_id)
        
        # We need to load in the whole DataStore here because the Celery worker 
        # (in which this function is running) will not know about the same context 
        # from the datastore.py module that the server code will.
        
        # Only if we have not already done so, create the DataStore object, 
        # setting up Redis.
        if ds.globalvars.data_store is None:
            # Create the DataStore object, setting up Redis.
            ds.globalvars.data_store = ds.DataStore(redis_db_URL=config.REDIS_URL)
            
        # Check if run_task() locked and wait until it isn't, then lock it for 
        # other run_task() instances in this Celery worker.
        lock_run_task(task_id)
        
        print('>>> EFFECTIVE START OF run_task() FOR %s' % task_id)
        
        # Load the DataStore state from disk. 
        # NOTE: This could corrupt DataStore access by concurrently active 
        # run_task() instances running in the same Celery worker because it 
        # overwrites ds.globalvars.data_store with the new disk state.
        # For this reason, locking needs to be done to prevent this, although 
        # the locking should allow run_task() instances on separate Celery 
        # workers to run concurrently.
        print('>>> LOADING DATASTORE FOR %s' % task_id)
        ds.globalvars.data_store.load()
        
        # Look for an existing tasks dictionary.
        task_dict_uid = ds.globalvars.data_store.get_uid('taskdict', 'Task Dictionary')
        
        # Create the task dictionary object.
        task_dict = TaskDict(task_dict_uid)
        
        # Load the TaskDict tasks from Redis.
        task_dict.load_from_data_store()
        print('>>> FINISHED LOADING DATASTORE FOR %s' % task_id)
        task_dict.show()  # TODO: remove this post-debugging
            
        # Find a matching task record (if any) to the task_id.
        match_taskrec = task_dict.get_task_record_by_task_id(task_id)
        if match_taskrec is None:
            print('>>> FAILED TO FIND TASK RECORD FOR %s' % task_id)
            unlock_run_task(task_id)
            return { 'error': 'Could not access TaskRecord' }
    
        # Set the TaskRecord to indicate start of the task.
        match_taskrec.status = 'started'
        match_taskrec.start_time = sc.now()
        match_taskrec.pending_time = \
            (match_taskrec.start_time - match_taskrec.queue_time).total_seconds()
            
        print('>>> BEFORE STARTED UPDATE OF TASK RECORD FOR %s' % task_id)
        
        # Do the actual update of the TaskRecord.
        # NOTE: At the moment the TaskDict on disk is not modified here, which 
        # which is a good thing because that could disrupt actiities in other 
        # run_task() instances.
        task_dict.update(match_taskrec)
        print('>>> AFTER STARTED UPDATE OF TASK RECORD FOR %s' % task_id)
        
        # Make the actual function call, inside a try block in case there is 
        # an exception thrown.
        # NOTE: This block is likely to run for several seconds or even 
        # minutes or hours, depending on the task.
        
        # WARNING: It is a very bad idea to have any calls in your task code 
        # that modify data_store.handle_dict.  That includes calls that add 
        # new entries to a BlobDict where the entries are kept external from 
        # the BlobDict object.  Writing to handle_dict will lead to conflicts 
        # with the webapp process, which has long stretches between reading in  
        # the handle_dict state and modifying it, during which Celery task 
        # worker writes to handle_dict will end up in lost modifications.
        
        # NOTE / WARNING: The function being called which the result is being 
        # taken from may rely on the contents of the DataStore.
#        print('Available task_funcs:')
#        print(task_func_dict)     
        try:
            result = task_func_dict[func_name](*args, **kwargs)
            match_taskrec.status = 'completed'
            print('>>> SUCCESSFULLY COMPLETED TASK %s' % task_id)

        # If there's an exception, grab the stack track and set the TaskRecord 
        # to have stopped on in error.
        except Exception:
            error_text = traceback.format_exc()
            match_taskrec.status = 'error'
            match_taskrec.error_text = error_text
            result = error_text
            print('>>> FAILED TASK %s' % task_id)
        
        # Set the TaskRecord to indicate end of the task.
        # NOTE: Even if the browser has ordered the deletion of the task 
        # record, it will be "resurrected" during this update, so the 
        # delete_task() RPC may not always work as expected.
        # TODO: Check to see if this is still true. (8/29/18)
        match_taskrec.stop_time = sc.now()
        match_taskrec.execution_time = \
            (match_taskrec.stop_time - match_taskrec.start_time).total_seconds()
        print('>>> BEFORE FINISHED UPDATE OF TASK RECORD FOR %s' % task_id)
        
        # Do the actual update of the TaskRecord.  Do this in a try / except 
        # block because this step may fail.  For example, if a TaskRecord is 
        # deleted by the webapp, the update here will crash.
        # NOTE: At the moment the TaskDict on disk is not modified here, which 
        # which is a good thing because that could disrupt actiities in other 
        # run_task() instances.
        try:
            task_dict.update(match_taskrec)           
        except Exception:
            error_text = traceback.format_exc()
            match_taskrec.status = 'error'
            match_taskrec.error_text = error_text
            result = error_text
            print('>>> FAILED TASK %s' % task_id)            
            
        print('>>> AFTER FINISHED UPDATE OF TASK RECORD FOR %s' % task_id)
        print('>>> END OF run_task() FOR %s' % task_id)
        
        # Unlock run-task() for other run_task() instances running on the same 
        # Celery worker.
        unlock_run_task(task_id)
        
        # Return the result.
        return result 
    
    # The launch-task() RPC is the only one included here because it is the 
    # only one that makes a direct call to run_task().  Any other RPCs that 
    # would call run_task() would have to be placed in make_celery_instance() 
    # as well.
    @RPC(validation='named') 
    def launch_task(task_id='', func_name='', args=[], kwargs={}):
        # NOTE: We might want to add a concurrency lock to this function.
        
#        print('Here is the celery_instance:')
#        print celery_instance
#        print('Here are the celery_instance tasks:')
#        print celery_instance.tasks

        # Reload the whole TaskDict from the DataStore because Celery may have 
        # modified its state in Redis (in particular, modified the TaskRecords).
        task_dict.load_from_data_store()
        
        # Find a matching task record (if any) to the task_id.
        match_taskrec = task_dict.get_task_record_by_task_id(task_id)
              
        # If we did not find a match...
        if match_taskrec is None:
            # If the function name is not in the task function dictionary, return an 
            # error.
            if not func_name in task_func_dict:
                return_dict = {
                    'error': 'Could not find requested async task function \'%s\'' % 
                        func_name
                }
            else:
                # Create a new TaskRecord.
                new_task_record = TaskRecord(task_id)
                
                # Initialize the TaskRecord with available information.
                new_task_record.status = 'queued'
                new_task_record.queue_time = sc.now()
                new_task_record.func_name = func_name
                new_task_record.args = args
                new_task_record.kwargs = kwargs
                
                # Add the TaskRecord to the TaskDict.
                task_dict.add(new_task_record) 
                
                # Queue up run_task() for Celery.
                my_result = run_task.delay(task_id, func_name, args, kwargs)
                
                # Add the result ID to the TaskRecord, and update the DataStore.
                new_task_record.result_id = my_result.id
                task_dict.update(new_task_record)
                
                # Create the return dict from the user repr.
                return_dict = new_task_record.get_user_front_end_repr()
            
        # Otherwise (there is a matching task)...
        else:
            # If the TaskRecord indicates the task has been completed or 
            # thrown an error...
            if match_taskrec.status == 'completed' or \
                match_taskrec.status == 'error':         
                # If we have a result ID, erase the result from Redis.
                if match_taskrec.result_id is not None:
                    result = celery_instance.AsyncResult(match_taskrec.result_id)
                    result.forget()
                    match_taskrec.result_id = None
                           
                # Initialize the TaskRecord to start the task again (though 
                # possibly with a new function and arguments).
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
                
                # Add the new result ID to the TaskRecord, and update the DataStore.
                match_taskrec.result_id = my_result.id
                task_dict.update(match_taskrec)
                
                # Create the return dict from the user repr.
                return_dict = match_taskrec.get_user_front_end_repr()
                
            # Else (the task is not completed)...
            else:
                return_dict = {
                    'error': 'Task is already %s' % match_taskrec.status
                }
        
        # Return our result.
        return return_dict  
    
    # Return the new instance.
    return celery_instance

# Function for adding new task functions to those that run_task() can see.
def add_task_funcs(new_task_funcs):
    global task_func_dict
    
    # For all of the keys in the dict passed in, put the key/value pairs in 
    # the global dict.
    for key in new_task_funcs:
        task_func_dict[key] = new_task_funcs[key]
  
@RPC(validation='named') 
def check_task(task_id): 
    # Reload the whole TaskDict from the DataStore because Celery may have 
    # modified its state in Redis (in particular, modified the TaskRecords).
    task_dict.load_from_data_store()
    
    # Find a matching task record (if any) to the task_id.
    match_taskrec = task_dict.get_task_record_by_task_id(task_id)
    
    # Check to see if the task exists, and if not, return an error.
    if match_taskrec is None:
        return {'error': 'No task found for specified task ID'}
    else:
        # Update the elapsed times.
        
        # If we are no longer pending...
        if match_taskrec.pending_time is not None:
            # Use the existing pending_time.
            pending_time = match_taskrec.pending_time
            
            # If we have finished executing...
            if match_taskrec.execution_time is not None:
                # Use the execution time in the record.
                execution_time = match_taskrec.execution_time
                
            # Else (we are still executing)...
            else:
                execution_time = (sc.now() - match_taskrec.start_time).total_seconds()
                
        # Else (we are still pending)...
        else:
            pending_time = (sc.now() - match_taskrec.queue_time).total_seconds()
            execution_time = 0
        
        # Create the return dict from the user repr.
        taskrec_dict = match_taskrec.get_user_front_end_repr()
        taskrec_dict['pendingTime'] = pending_time
        taskrec_dict['executionTime'] = execution_time
        
        # Return the has record information and elapsed times.
        return taskrec_dict        
    
@RPC(validation='named') 
def get_task_result(task_id):
    # Reload the whole TaskDict from the DataStore because Celery may have 
    # modified its state in Redis (in particular, modified the TaskRecords).
    task_dict.load_from_data_store()
    
    # Find a matching task record (if any) to the task_id.
    match_taskrec = task_dict.get_task_record_by_task_id(task_id)
    
    # Check to see if the task exists, and if not, return an error.
    if match_taskrec is None:
        return {'error': 'No task found for specified task ID'}
    else:
        # If we have a result ID...
        if match_taskrec.result_id is not None:
            # Get the result itself from Celery.
            result = celery_instance.AsyncResult(match_taskrec.result_id) 
            
            # If the result is not ready, return an error.
            if not result.ready():
                return {'error': 'Task not completed'}
            
            # Else, if we have failed, return the exception.
            elif result.failed():
                return {'error': 'Task failed with an exception'}
            
            # Else (task is ready)...
            else:
                return {'result': result.get()}
            
        # Else (no result ID)...
        else:
            return {'error': 'No result ID'}
    
@RPC(validation='named') 
def delete_task(task_id): 
    # NOTE: We might want to add a concurrency lock to this function.
    
    # Reload the whole TaskDict from the DataStore because Celery may have 
    # modified its state in Redis (in particular, modified the TaskRecords).
    task_dict.load_from_data_store()
    
    # Find a matching task record (if any) to the task_id.
    match_taskrec = task_dict.get_task_record_by_task_id(task_id)
    
    # Check to see if the task exists, and if not, return an error.
    if match_taskrec is None:
        return {'error': 'No task found for specified task ID'}
    
    # Otherwise (matching task).
    else:
        # If we have a result ID, erase the result from Redis.
        if match_taskrec.result_id is not None:
            result = celery_instance.AsyncResult(match_taskrec.result_id)
            # This commmand works under Linux, but not under the Windows 
            # Celery setup.  It allows terminating a task in mid-run.
            result.revoke(terminate=True)
            result.forget()
            
        # Erase the TaskRecord.
        task_dict.delete_by_task_id(task_id)
        
        # Return success.
        return 'success'


# Function for making a register_async_task decorator in other modules.
def make_async_tag(task_func_dict):
    def task_func_decorator(task_func):
        @wraps(task_func)
        def wrapper(*args, **kwargs):        
            output = task_func(*args, **kwargs)
            return output
            
        # Add the function to the dictionary.
        task_func_dict[task_func.__name__] = task_func
        
        return wrapper    
    
    return task_func_decorator

