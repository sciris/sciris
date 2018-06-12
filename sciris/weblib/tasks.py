"""
tasks.py -- code related to Sciris task queue management
    
Last update: 6/11/18 (gchadder3)
"""

#
# Imports
#

from numpy import argsort
from . import rpcs #import make_register_RPC
from . import scirisobjects as sobj
from ..corelib import utils as ut

from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379', 
    backend='redis://localhost:6379')

@app.task
def add(x, y):
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
        
    Usage:
        >>> my_task = TaskRecord('my-special-task', uid=uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def  __init__(self, task_id, uid=None):
        # Set superclass parameters.
        super(TaskRecord, self).__init__(uid)
        
        # Set the task ID (what the client typically knows as the task).
        self.task_id = task_id
        
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
            
    def get_user_front_end_repr(self):
        obj_info = {
            'task': {
                'UID': self.uid.hex,                    
                'instanceLabel': self.instance_label,
                'taskId': self.task_id,               
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
                'taskId': self.task_id               
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
# RPC functions
#
 
@register_RPC(validation_type='nonanonymous user') 
def launch_task(task_id='', func_name='', args=[], kwargs={}):
    # Show arguments.
    print 'task_id'
    print task_id
    print 'my func'
    print func_name
    print 'my args'
    print args
    print 'my kwargs'
    print kwargs
    
    new_task_record = TaskRecord(task_id)
    task_dict.add(new_task_record)
    
    return {'error': 'Sorry, not ready!'}

@register_RPC(validation_type='nonanonymous user') 
def check_task(task_id): 
    # Check to see if the task exists, and if not, return an error.
    if task_dict.get_task_record_by_task_id(task_id) is None:
        return {'error': 'No task found for specified task ID'}
    else:
        return {'error': 'Sorry, not ready!'}
    
@register_RPC(validation_type='nonanonymous user') 
def get_task_result(task_id):  
    # Check to see if the task exists, and if not, return an error.
    if task_dict.get_task_record_by_task_id(task_id) is None:
        return {'error': 'No task found for specified task ID'}
    else:
        return {'error': 'Sorry, not ready!'}
    
@register_RPC(validation_type='nonanonymous user') 
def delete_task(task_id): 
    # Check to see if the task exists, and if not, return an error.
    if task_dict.get_task_record_by_task_id(task_id) is None:
        return {'error': 'No task found for specified task ID'}
    
    # Otherwise, erase the task and return success.
    else:
        task_dict.delete_by_task_id(task_id)
        return 'success'
       
#@register_RPC()
#def user_login(username, password):  
#    # Get the matching user (if any).
#    matching_user = user_dict.get_user_by_username(username)
#    
#    # If we have a match and the password matches, and the account is active,
#    # also, log in the user and return success; otherwise, return failure.
#    if matching_user is not None and matching_user.password == password and \
#        matching_user.is_active:
#        # Log the user in.
#        login_user(matching_user)
#        
#        return 'success'
#    else:
#        return 'failure'
#    
#@register_RPC(validation_type='nonanonymous user')       
#def user_logout():
#    # Log the user out and set the session to having an anonymous user.
#    logout_user()
#    
#    # Clear the session cookie.
#    session.clear()
#    
#    # Return nothing.
#    return None
#
#@register_RPC(validation_type='nonanonymous user') 
#def get_current_user_info():
#    return current_user.get_user_front_end_repr()
#
#@register_RPC(validation_type='admin user')
#def get_all_users():
#    # Return success.
#    return user_dict.get_admin_front_end_repr()
#
#@register_RPC()
#def user_register(username, password, displayname, email):
#    # Get the matching user (if any).
#    matching_user = user_dict.get_user_by_username(username)
#    
#    # If we have a match, fail because we don't want to register an existing 
#    # user.
#    if matching_user is not None:
#        return 'failure'
#    
#    # Create a new User object with the new information.
#    new_user = User(username, password, displayname, email, hash_the_password=False)
#    
#    # Set the user to be inactive (so an admin user needs to activate it, 
#    # even though the account has been created) or active (so that no admin
#    # action is necessary), according to the REGISTER_AUTOACTIVATE config 
#    # parameter.
#    new_user.is_active = current_app.config['REGISTER_AUTOACTIVATE']
#    
#    # Put the user right into the UserDict.
#    user_dict.add(new_user)
#    
#    # Return success.
#    return 'success'
#
#@register_RPC(validation_type='nonanonymous user') 
#def user_change_info(username, password, displayname, email):
#    # Make a copy of the current_user.
#    the_user = ut.dcp(current_user)
#    
#    # If the password entered doesn't match the current user password, fail.
#    if password != the_user.password:
#        return 'failure'
#       
#    # If the username entered by the user is different from the current_user
#    # name (meaning they are trying to change the name)...
#    if username != the_user.username:
#        # Get any matching user (if any) to the new username we're trying to 
#        # switch to.
#        matching_user = user_dict.get_user_by_username(username)
#    
#        # If we have a match, fail because we don't want to rename the user to 
#        # another existing user.
#        if matching_user is not None:
#            return 'failure'
#        
#    # Change the user name, display name, email, and instance label.
#    the_user.username = username
#    the_user.displayname = displayname
#    the_user.email = email
#    the_user.instance_label = username
#    
#    # Update the user in user_dict.
#    user_dict.update(the_user)
#    
#    # Return success.
#    return 'success'
#
#@register_RPC(validation_type='nonanonymous user') 
#def user_change_password(oldpassword, newpassword):
#    # Make a copy of the current_user.
#    the_user = ut.dcp(current_user)
#    
#    # If the password entered doesn't match the current user password, fail.
#    if oldpassword != the_user.password:
#        return 'failure' 
#    
#    # Change just the password.
#    the_user.password = newpassword
#    
#    # Update the user in user_dict.
#    user_dict.update(the_user)
#    
#    # Return success.
#    return 'success'
#
#@register_RPC(validation_type='admin user')
#def admin_get_user_info(username):
#    # Get the matching user (if any).
#    matching_user = user_dict.get_user_by_username(username)
#    
#    # If we don't have a match, fail.
#    if matching_user is None:
#        return 'failure'
#    
#    return matching_user.get_admin_front_end_repr()
#
#@register_RPC(validation_type='admin user')
#def admin_delete_user(username):
#    # Get the matching user (if any).
#    matching_user = user_dict.get_user_by_username(username)
#    
#    # If we don't have a match, fail.
#    if matching_user is None:
#        return 'failure'
#    
#    # Delete the user from the dictionary.
#    user_dict.delete_by_username(username)
#
#    # Return success.
#    return 'success'
#
#@register_RPC(validation_type='admin user')
#def admin_activate_account(username):
#    # Get the matching user (if any).
#    matching_user = user_dict.get_user_by_username(username)
#    
#    # If we don't have a match, fail.
#    if matching_user is None:
#        return 'failure'
#    
#    # If the account is already activated, fail.
#    if matching_user.is_active:
#        return 'failure'
#    
#    # Activate the account.
#    matching_user.is_active = True
#    user_dict.update(matching_user)
#    
#    # Return success.
#    return 'success'    
#
#@register_RPC(validation_type='admin user')
#def admin_deactivate_account(username):
#    # Get the matching user (if any).
#    matching_user = user_dict.get_user_by_username(username)
#    
#    # If we don't have a match, fail.
#    if matching_user is None:
#        return 'failure'
#    
#    # If the account is already deactivated, fail.
#    if not matching_user.is_active:
#        return 'failure'
#    
#    # Activate the account.
#    matching_user.is_active = False
#    user_dict.update(matching_user)
#    
#    # Return success.
#    return 'success'
#
#@register_RPC(validation_type='admin user')
#def admin_grant_admin(username):
#    # Get the matching user (if any).
#    matching_user = user_dict.get_user_by_username(username)
#    
#    # If we don't have a match, fail.
#    if matching_user is None:
#        return 'failure'
#    
#    # If the account already has admin access, fail.
#    if matching_user.is_admin:
#        return 'failure'
#    
#    # Grant admin access.
#    matching_user.is_admin = True
#    user_dict.update(matching_user)
#    
#    # Return success.
#    return 'success'
#
#@register_RPC(validation_type='admin user')
#def admin_revoke_admin(username):
#    # Get the matching user (if any).
#    matching_user = user_dict.get_user_by_username(username)
#    
#    # If we don't have a match, fail.
#    if matching_user is None:
#        return 'failure'
#    
#    # If the account has no admin access, fail.
#    if not matching_user.is_admin:
#        return 'failure'
#    
#    # Revoke admin access.
#    matching_user.is_admin = False
#    user_dict.update(matching_user)
#    
#    # Return success.
#    return 'success'
#
#@register_RPC(validation_type='admin user')
#def admin_reset_password(username):
#    # Get the matching user (if any).
#    matching_user = user_dict.get_user_by_username(username)
#    
#    # If we don't have a match, fail.
#    if matching_user is None:
#        return 'failure'
#    
#    # Set the password to the desired raw password for them to use.
#    raw_password = 'sciris'
#    matching_user.password = sha224(raw_password).hexdigest() 
#    user_dict.update(matching_user)
#    
#    # Return success.
#    return 'success'

#
# Task functions
#

