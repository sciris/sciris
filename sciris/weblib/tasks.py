"""
tasks.py -- code related to Sciris task queue management
    
Last update: 6/14/18 (gchadder3)
"""

#
# Imports
#

from numpy import argsort
from . import rpcs #import make_register_RPC
from . import scirisobjects as sobj
from ..corelib import utils as ut

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
        func_name (str) -- string of the function name for what's called
        args (list) -- list containing args for the function
        kwargs (dict) -- dict containing kwargs for the function
        result_id (str) -- string for the Redis ID of the AsyncResult
        queue_time (???) -- the time the task was queued for Celery
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
        
        # Start the func_name at None.
        self.func_name = None
        
        # Start the args and kwargs at None.
        self.args = None
        self.kwargs = None
        
        # Start with no result_id.
        self.result_id = None
        
        # Start the queue, start, and stop times at None.
        self.queue_time = None
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
        print('Function Name: %s' % self.func_name)
        print('Function Args: %s' % self.args)
        print('Function Kwargs: %s' % self.kwargs)        
        print('Result ID: %s' % self.result_id)
        print('Queue Time: %s' % self.queue_time)        
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
                'funcName': self.func_name,
                'funcArgs': self.args,
                'funcKwargs': self.kwargs,
                'resultId': self.result_id,
                'queueTime': self.queue_time,                
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
                'funcName': self.func_name,
                'funcArgs': self.args,
                'funcKwargs': self.kwargs,                
                'resultId': self.result_id, 
                'queueTime': self.queue_time,                    
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
# RPC functions
#
