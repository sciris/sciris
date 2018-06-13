"""
tasks.py -- code related to Sciris task queue management
    
Last update: 6/13/18 (gchadder3)
"""

#
# Imports
#

from numpy import argsort
from . import rpcs #import make_register_RPC
from . import scirisobjects as sobj
from ..corelib import utils as ut
import time

from celery import Celery

celery_instance = Celery('tasks', broker='redis://localhost:6379', 
    backend='redis://localhost:6379')
celery_instance.conf.CELERY_TRACK_STARTED = True

@celery_instance.task
def async_add(x, y):
    time.sleep(120)
    return x + y

#
# Globals
#

# The TaskDict object for all of the app's existing asynchronous tasks.  
# Gets initialized by and loaded by init_tasks().
task_dict = None

# Dictionary to hold all of the registered RPCs in this module.
RPC_dict = {}

# RPC registration decorator factory created using call to make_register_RPC().
register_RPC = rpcs.make_register_RPC(RPC_dict)

# Dictionary to hold all of the registered task functions in this module.
task_funcs_dict = {}

#
# Classes
#

class TaskRecord(sobj.ScirisObject):
    """
    A Sciris record for an asynchronous task.
    
    Methods:
        __init__(task_id: str, password: str, uid: UUID [None]): 
            void -- constructor
        load_from_copy(other_object): void -- assuming other_object is another 
            object of our type, copy its contents to us (calls the 
            ScirisObject superclass version of this method also)  
        show(): void -- print the contents of the object
        get_user_front_end_repr(): dict -- get a JSON-friendly dictionary 
            representation of the object state the front-end uses for non-
            admin purposes
        get_admin_front_end_repr(): dict -- get a JSON-friendly dictionary
            representation of the object state the front-end uses for admin
            purposes        
                    
    Attributes:
        task_id (str) -- the ID / name for the task that typically is chosen 
            by the client
        status (str) -- the status of the task:
            'unknown' : unknown status, for example, just initialized
        error_text (str) -- string giving an idea of what error has transpired
        result_id (str) -- string for the Redis ID of the AsyncResult
        start_time (???) -- the time the task was actually started
        stop_time (???) -- the time the task completed  
        elapsed_time (int) -- the time the process has been running on the 
            server in seconds
        
    Usage:
        >>> my_task = TaskRecord('my-special-task', uid=uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def  __init__(self, task_id, uid=None):
        # Set superclass parameters.
        super(TaskRecord, self).__init__(uid)
        
        # Set the task ID (what the client typically knows as the task).
        self.task_id = task_id
        
        # Start the status at 'unknown'.
        self.status = 'unknown'
        
        # Start the error_text at None.
        self.error_text = None
        
        # Start with no result_id.
        self.result_id = None
        
        # Start the start and stop times at None.
        self.start_time = None
        self.stop_time = None
        
        # Start the elapsed time at zero seconds.
        self.elapsed_time = 0
        
        # Set the type prefix to 'task'.
        self.type_prefix = 'task'
        
        # Set the file suffix to '.tsk'.
        self.file_suffix = '.tsk'
        
        # Set the instance label to the task_id.
        self.instance_label = task_id 
              
    def load_from_copy(self, other_object):
        if type(other_object) == type(self):
            # Do the superclass copying.
            super(TaskRecord, self).load_from_copy(other_object)

            self.task_id = other_object.task_id
            
    def show(self):
        # Show superclass attributes.
        super(TaskRecord, self).show()  
        
        print('---------------------')

        print('Task ID: %s' % self.task_id)
        print('Status: %s' % self.status)
        print('Error Text: %s' % self.error_text)
        print('Result ID: %s' % self.result_id)
        print('Start Time: %s' % self.start_time)
        print('Stop Time: %s' % self.stop_time)
        print('Elapsed Time: %s sec.' % self.elapsed_time)        
            
    def get_user_front_end_repr(self):
        obj_info = {
            'task': {
                'UID': self.uid.hex,                    
                'instanceLabel': self.instance_label,
                'taskId': self.task_id,
                'status': self.status,
                'errorText': self.error_text,
                'resultId': self.result_id,
                'startTime': self.start_time,
                'stopTime': self.stop_time,
                'elapsedTime': self.elapsed_time
            }
        }
        return obj_info
    
    def get_admin_front_end_repr(self):
        obj_info = {
            'task': {
                'UID': self.uid.hex, 
                'typePrefix': self.type_prefix, 
                'fileSuffix': self.file_suffix, 
                'instanceLabel': self.instance_label,
                'taskId': self.task_id,
                'status': self.status,
                'errorText': self.error_text,
                'resultId': self.result_id,                
                'startTime': self.start_time,
                'stopTime': self.stop_time,
                'elapsedTime': self.elapsed_time                
            }
        }
        return obj_info             
            
class TaskDict(sobj.ScirisCollection):
    """
    A dictionary of Sciris tasks.
    
    Methods:
        __init__(uid: UUID [None], type_prefix: str ['taskdict'], 
            file_suffix: str ['.td'], 
            instance_label: str ['Task Dictionary']): void -- constructor
        load_from_copy(other_object): void -- assuming other_object is another 
            object of our type, copy its contents to us (calls the 
            ScirisCollection superclass version of this method also)           
        get_task_record_by_uid(uid: UUID or str): TaskRecord or None -- 
            returns the TaskRecord object pointed to by uid
        get_task_record_by_task_id(task_id: str): TaskRecord or None -- 
            return the TaskRecord object pointed to by the task_id
        add(task_record: TaskRecord): void -- add a TaskRecord to the dictionary and update
            the dictionary's DataStore state
        update(task_record: TaskRecord): void -- update a TaskRecord in the dictionary and 
            update the dictionary's DataStore state
        delete_by_uid(uid: UUID or str): void -- delete a TaskRecord in the dictionary
            selected by the UID, and update the dictionary's DataStore state
        delete_by_task_id(task_id: str): void -- delete a TaskRecord in the 
            dictionary selected by a task_id, and update the dictionary's 
            DataStore state
        delete_all(): void -- delete the entire contents of the TaskDict and 
            update the dictionary's DataStore state            
        get_user_front_end_repr(): dict -- get a JSON-friendly dictionary 
            representation of the collection state the front-end uses for non-
            admin purposes
        get_admin_front_end_repr(): dict -- get a JSON-friendly dictionary
            representation of the collection state the front-end uses for admin
            purposes
                    
    Attributes:
        task_id_hashes (dict) -- a dict mapping task_ids to UIDs, so either
            indexing by UIDs or task_ids can be fast
        
    Usage:
        >>> task_dict = TaskDict(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def __init__(self, uid, type_prefix='taskdict', file_suffix='.td', 
        instance_label='Task Dictionary'):
        # Set superclass parameters.
        super(TaskDict, self).__init__(uid, type_prefix, file_suffix, 
             instance_label)
        
        # Create the Python dict to hold the hashes from task_ids to the UIDs.
        self.task_id_hashes = {}
        
    def load_from_copy(self, other_object):
        if type(other_object) == type(self):
            # Do the superclass copying.
            super(TaskDict, self).load_from_copy(other_object)
            
            self.task_id_hashes = other_object.task_id_hashes
            
    def get_task_record_by_uid(self, uid):
        return self.get_object_by_uid(uid)
    
    def get_task_record_by_task_id(self, task_id):
        # Get the task record's UID matching the task_id.
        id_index = self.task_id_hashes.get(task_id, None)
        
        # If we found at match, use the UID to try to fetch the task record; 
        # otherwise, return None.
        if id_index is not None:
            return self.get_task_record_by_uid(id_index)
        else:
            return None
        
    def add(self, task_record):
        # Add the object to the hash table, keyed by the UID.
        self.obj_dict[task_record.uid] = task_record
        
        # Add the task_id hash for this task record.
        self.task_id_hashes[task_record.task_id] = task_record.uid
        
        # Update our DataStore representation if we are there. 
        if self.in_data_store():
            self.update_data_store()
    
    def update(self, task_record):
        # Get the old task_id.
        old_task_id = self.obj_dict[task_record.uid].task_id
        
        # If we new task_id is different than the old one, delete the old 
        # task_id hash.
        if task_record.task_id != old_task_id:
            del self.task_id_hashes[old_task_id]
 
        # Add the task record to the hash table, keyed by the UID.
        self.obj_dict[task_record.uid] = task_record
        
        # Add the task_id hash for this task record.
        self.task_id_hashes[task_record.task_id] = task_record.uid
       
        # Update our DataStore representation if we are there. 
        if self.in_data_store():
            self.update_data_store()
    
    def delete_by_uid(self, uid):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        valid_uuid = ut.uuid(uid)
        
        # If we have a valid UUID...
        if valid_uuid is not None:
            # Get the object pointed to by the UID.
            obj = self.obj_dict[valid_uuid]
            
            # If a match is found...
            if obj is not None:
                # Remove entries from both task_dict and task_id_hashes 
                # attributes.
                task_id = obj.task_id
                del self.obj_dict[valid_uuid]
                del self.task_id_hashes[task_id]                
                
                # Update our DataStore representation if we are there. 
                if self.in_data_store():
                    self.update_data_store()
        
    def delete_by_task_id(self, task_id):
        # Get the UID of the task record matching task_id.
        id_index = self.task_id_hashes.get(task_id, None)
        
        # If we found a match, call delete_by_uid to complete the deletion.
        if id_index is not None:
            self.delete_by_uid(id_index)
    
    def delete_all(self):
        # Reset the Python dicts to hold the task record objects and hashes from 
        # task_ids to the UIDs.
        self.obj_dict = {}
        self.task_id_hashes = {}  
        
        # Update our DataStore representation if we are there. 
        if self.in_data_store():
            self.update_data_store()
            
    def get_user_front_end_repr(self):
        # Get dictionaries for each task record in the dictionary.
        taskrecs_info = [self.obj_dict[key].get_user_front_end_repr() 
            for key in self.obj_dict]
        return taskrecs_info
        
    def get_admin_front_end_repr(self):
        # Get dictionaries for each task record in the dictionary.       
        taskrecs_info = [self.obj_dict[key].get_admin_front_end_repr() 
            for key in self.obj_dict]
        
        # Extract just the task_ids.
        task_ids = [task_record['task']['task_id'] for task_record in taskrecs_info]
        
        # Get sorting indices with respect to the task_ids.
        sort_order = argsort(task_ids)
        
        # Created a list of the sorted users_info list.
        sorted_taskrecs_info = [taskrecs_info[ind] for ind in sort_order]

        # Return the sorted users info.      
        return sorted_taskrecs_info  

#
# Other functions (mostly helpers for the RPCs)
#

#def get_scirisdemo_user(name='_ScirisDemo'):    
#    # Get the user object matching (if any)...
#    the_user = user_dict.get_user_by_username(name)
#    
#    # If there is a match, return the UID, otherwise return None.
#    if the_user is not None:
#        return the_user.get_id()
#    else:
#        return None
        
#
# run_task() function
#
        
@celery_instance.task        
def run_task(task_id, func_name, args):
    pass

#
# RPC functions
#
        
@register_RPC(validation_type='nonanonymous user') 
def launch_task(task_id='', func_name='', args=[], kwargs={}):
    # Start with an empty return dict.
    return_dict = {}
    
    # Find a matching task record (if any) to the task_id.
    match_taskrec = task_dict.get_task_record_by_task_id(task_id)
    
    # If we did not find a match...
    if task_dict.get_task_record_by_task_id(task_id) is None:
        if func_name != 'async_add':
            return_dict = {
                'error': 'You can run any function as long as its async_add!'
            }
        else:
            # Create a new TaskRecord.
            new_task_record = TaskRecord(task_id)
        
            my_result = async_add.delay(args[0], args[1])
            
            new_task_record.result_id = my_result.id
            new_task_record.status = 'started'
            new_task_record.start_time = ut.today()
            
            # Add the TaskRecord to the TaskDict.
            task_dict.add(new_task_record)   
            
            # Create the return dict from the user repr.
            return_dict = new_task_record.get_user_front_end_repr()
        
    # Otherwise (there is a matching task)...
    else:
        # xxx
        if match_taskrec.status == 'completed':
            return_dict = {
                'status': 'started'
            }
        else:
            return_dict = {
                'error': 'blocked'
            }
    
    # Return our result.
    return return_dict

@register_RPC(validation_type='nonanonymous user') 
def check_task(task_id): 
    # Find a matching task record (if any) to the task_id.
    match_taskrec = task_dict.get_task_record_by_task_id(task_id)
    
    # Check to see if the task exists, and if not, return an error.
    if match_taskrec is None:
        return {'error': 'No task found for specified task ID'}
    else:
        # Update the elapsed time.
        
        # Create the return dict from the user repr.
        return match_taskrec.get_user_front_end_repr()        
    
@register_RPC(validation_type='nonanonymous user') 
def get_task_result(task_id):  
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
            
            # Else (task is ready)...
            else:
                return {'result': result.get()}
            
        # Else (no result ID)...
        else:
            return {'error': 'No result ID'}
    
@register_RPC(validation_type='nonanonymous user') 
def delete_task(task_id): 
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
            result.forget()
            
        # Erase the TaskRecord.
        task_dict.delete_by_task_id(task_id)
        
        # Return success.
        return 'success'

#
# Task functions
#

