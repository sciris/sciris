"""
apptasks.py -- The Celery tasks module for this webapp
    
Last update: 6/15/18 (gchadder3)
"""

#
# Imports
#

from celery import Celery
import config
import time
from functools import wraps
from sciris.corelib import utils as ut
from sciris.weblib import datastore as ds
from sciris.weblib import tasks
from sciris.weblib.rpcs import make_register_RPC

#
# Globals
#

# Define the Celery instance.
celery_instance = Celery('tasks')

# Configure Celery with config.py.
celery_instance.config_from_object('config')

# Configure so that the actual start of a task is tracked.
celery_instance.conf.CELERY_TRACK_STARTED = True

# Dictionary to hold all of the registered RPCs in this module.
RPC_dict = {}

# RPC registration decorator factory created using call to make_register_RPC().
register_RPC = make_register_RPC(RPC_dict)

# Dictionary to hold registered task functions to be callable from run_task().
task_func_dict = {}

#
# run_task() function
#
        
@celery_instance.task        
def run_task(task_id, func_name, args, kwargs):
    # We need to load in the whole DataStore here because the Celery worker 
    # (in which this function is running) will not know about the same context 
    # from the datastore.py module that the server code will.
    
    # Create the DataStore object, setting up Redis.
    ds.data_store = ds.DataStore(redis_db_URL=config.REDIS_URL)
    
    # Load the DataStore state from disk.
    ds.data_store.load()
    
    # Look for an existing tasks dictionary.
    task_dict_uid = ds.data_store.get_uid_from_instance('taskdict', 'Task Dictionary')
    
    # Create the task dictionary object.
    tasks.task_dict = tasks.TaskDict(task_dict_uid)
    
    # Load the TaskDict tasks from Redis.
    tasks.task_dict.load_from_data_store()
        
    # Find a matching task record (if any) to the task_id.
    match_taskrec = tasks.task_dict.get_task_record_by_task_id(task_id)

    # Set the TaskRecord to indicate start of the task.
    match_taskrec.status = 'started'
    match_taskrec.start_time = ut.today()
    tasks.task_dict.update(match_taskrec)
    
    # Make the actual function call.
    result = task_func_dict[func_name](*args, **kwargs)
    
    # Set the TaskRecord to indicate end of the task.
    match_taskrec.status = 'completed'
    match_taskrec.stop_time = ut.today()
    tasks.task_dict.update(match_taskrec)
    
    # Return the result.
    return result

#
# RPC functions
#
        
@register_RPC(validation_type='nonanonymous user') 
def launch_task(task_id='', func_name='', args=[], kwargs={}):
    # Find a matching task record (if any) to the task_id.
    match_taskrec = tasks.task_dict.get_task_record_by_task_id(task_id)
          
    # If we did not find a match...
    if match_taskrec is None:
        # If the function name is not in the task function dictionary, return an 
        # error.
        if not func_name in task_func_dict:
            return_dict = {
                'error': 'Could not find requested async task function'
            }
        else:
            # Create a new TaskRecord.
            new_task_record = tasks.TaskRecord(task_id)
            
            # Initialize the TaskRecord with available information.
            new_task_record.status = 'queued'
            new_task_record.queue_time = ut.today()
            new_task_record.func_name = func_name
            new_task_record.args = args
            new_task_record.kwargs = kwargs
            
            # Add the TaskRecord to the TaskDict.
            tasks.task_dict.add(new_task_record) 
            
            # Queue up run_task() for Celery.
            my_result = run_task.delay(task_id, func_name, args, kwargs)
            
            # Add the result ID to the TaskRecord, and update the DataStore.
            new_task_record.result_id = my_result.id
            tasks.task_dict.update(new_task_record)
            
            # Create the return dict from the user repr.
            return_dict = new_task_record.get_user_front_end_repr()
        
    # Otherwise (there is a matching task)...
    else:
        # If the TaskRecord indicates the task has been completed...
        if match_taskrec.status == 'completed':
            # If we have a result ID, erase the result from Redis.
            if match_taskrec.result_id is not None:
                result = celery_instance.AsyncResult(match_taskrec.result_id)
                result.forget()
                       
            # Initialize the TaskRecord to start the task again (though 
            # possibly with a new function and arguments).
            match_taskrec.status = 'queued'
            match_taskrec.queue_time = ut.today()
            match_taskrec.start_time = None
            match_taskrec.stop_time = None            
            match_taskrec.func_name = func_name
            match_taskrec.args = args
            match_taskrec.kwargs = kwargs
            
            # Queue up run_task() for Celery.
            my_result = run_task.delay(task_id, func_name, args, kwargs)
            
            # Add the new result ID to the TaskRecord, and update the DataStore.
            match_taskrec.result_id = my_result.id
            tasks.task_dict.update(match_taskrec)
            
            # Create the return dict from the user repr.
            return_dict = match_taskrec.get_user_front_end_repr()
            
        # Else (the task is not completed)...
        else:
            return_dict = {
                'error': 'Task is already running'
            }
    
    # Return our result.
    return return_dict

@register_RPC(validation_type='nonanonymous user') 
def check_task(task_id): 
    # Reload the whole TaskDict from the DataStore because Celery may have 
    # modified its state in Redis.
    tasks.task_dict.load_from_data_store()
    
    # Find a matching task record (if any) to the task_id.
    match_taskrec = tasks.task_dict.get_task_record_by_task_id(task_id)
    
    # Check to see if the task exists, and if not, return an error.
    if match_taskrec is None:
        return {'error': 'No task found for specified task ID'}
    else:
        # Update the elapsed time.
        
        # Create the return dict from the user repr.
        return match_taskrec.get_user_front_end_repr()        
    
@register_RPC(validation_type='nonanonymous user') 
def get_task_result(task_id):
    # Reload the whole TaskDict from the DataStore because Celery may have 
    # modified its state in Redis.
    tasks.task_dict.load_from_data_store()
    
    # Find a matching task record (if any) to the task_id.
    match_taskrec = tasks.task_dict.get_task_record_by_task_id(task_id)
    
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
            
            # Else (task is ready)...
            else:
                return {'result': result.get()}
            
        # Else (no result ID)...
        else:
            return {'error': 'No result ID'}
    
@register_RPC(validation_type='nonanonymous user') 
def delete_task(task_id): 
    # Find a matching task record (if any) to the task_id.
    match_taskrec = tasks.task_dict.get_task_record_by_task_id(task_id)
    
    # Check to see if the task exists, and if not, return an error.
    if match_taskrec is None:
        return {'error': 'No task found for specified task ID'}
    
    # Otherwise (matching task).
    else:
        # If we have a result ID, erase the result from Redis.
        if match_taskrec.result_id is not None:
            result = celery_instance.AsyncResult(match_taskrec.result_id)
            result.forget()
            
        # Erase the TaskRecord.
        tasks.task_dict.delete_by_task_id(task_id)
        
        # Return success.
        return 'success'

#
# Task functions
#

# Decorator function for registering an async task function, putting it in the 
# task_func_dict.
def register_async_task(task_func):
    @wraps(task_func)
    def wrapper(*args, **kwargs):        
        task_func(*args, **kwargs)

    # Add the function to the dictionary.
    task_func_dict[task_func.__name__] = task_func
    
    return wrapper

@register_async_task
def async_add(x, y):
    time.sleep(60)
    return x + y

@register_async_task
def async_sub(x, y):
    time.sleep(60)
    return x - y