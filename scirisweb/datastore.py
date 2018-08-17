"""
datastore.py -- code related to Sciris database persistence
    
Last update: 7/11/18 (gchadder3)
"""

#
# Imports
#

import os
import redis
from ..corelib import utils as ut
from ..corelib import fileio as io


#
# Globals
#

# These will get set by calling code.

# The DataStore object for persistence for the app.  Gets initialized by
# and loaded by init_datastore().
data_store = None

#
# Classes
#


class StoreObjectHandle(object):
    """
    An object associated with a Python object which permits the Python object 
    to be stored in and retrieved from a DataStore object.
    
    Methods:
        __init__(uid: UUID [None], type_prefix: str ['obj'], 
            file_suffix: str ['.obj'], instance_label: str['']): void --
            constructor
        get_uid(): UUID -- return the StoreObjectHandle's UID
        file_store(dir_path: str, obj: Object): void -- store obj in 
            the dir_path directory
        file_retrieve(dir_path: str): void -- retrieve the stored object from 
            the dir_path directory
        file_delete(dir_path: str): void -- delete the stored object from the 
            dir_path directory
        redis_store(obj: Object, redis_db: redis.client.StrictRedis): 
            void -- store obj in Redis
        redis_retrieve(redis_db: redis.client.StrictRedis): void -- retrieve 
            the stored object from Redis
        redis_delete(redis_db: redis.client.StrictRedis): void -- delete the 
            stored object from Redis
        show(): void -- print the contents of the object
                    
    Attributes:
        uid (UUID) -- the unique ID for the handle (uuid Python library-related)
        type_prefix (str) -- a prefix that gets added to the UUID to give either 
            a file name or a Redis key
        file_suffix (str) -- a suffix that gets added to files
        instance_label (str) -- a name of the object which should at least be 
            unique across other handles of the save type_prefix
        
    Usage:
        >>> new_handle = StoreObjectHandle(uuid.UUID('12345678123456781234567812345678'), 
            type_prefix='project', file_suffix='.prj', instance_label='Project 1')
    """
    
    def __init__(self, uid=None, type_prefix='obj', file_suffix='.obj', 
        instance_label=''):
        # Set the UID to what was passed in, or if None was passed in, generate 
        # and use a new one.
        self.uid = ut.uuid(uid)
        
        self.type_prefix = type_prefix
        self.file_suffix = file_suffix
        self.instance_label = instance_label
        
    def get_uid(self):
        return self.uid
    
    def file_store(self, dir_path, obj):
        # Create a filename containing the type prefix, hex UID code, and the
        # appropriate file suffix.
        file_name = '%s-%s%s' % (self.type_prefix, self.uid.hex, self.file_suffix)
        
        # Generate the full file name with path.
        full_file_name = '%s%s%s' % (dir_path, os.sep, file_name)
        
        # Write the object to a Gzip string pickle file.
#        ut.tic()
        io.object_to_gzip_string_pickle_file(full_file_name, obj)
#        ut.toc(label='file_store (%s)' % self.instance_label)
    
    def file_retrieve(self, dir_path):
        # Create a filename containing the type prefix, hex UID code, and the
        # appropriate file suffix.
        file_name = '%s-%s%s' % (self.type_prefix, self.uid.hex, self.file_suffix)
        
        # Generate the full file name with path.
        full_file_name = '%s%s%s' % (dir_path, os.sep, file_name)
        
        # Return object from the Gzip string pickle file.
#        ut.tic()
        obj = io.gzip_string_pickle_file_to_object(full_file_name)
#        ut.toc(label='file_retrieve (%s)' % self.instance_label)
        return obj
    
    def file_delete(self, dir_path):
        # Create a filename containing the type prefix, hex UID code, and the
        # appropriate file suffix.
        file_name = '%s-%s%s' % (self.type_prefix, self.uid.hex, self.file_suffix)
        
        # Generate the full file name with path.
        full_file_name = '%s%s%s' % (dir_path, os.sep, file_name)
        
        # Remove the file if it's there.
        if os.path.exists(full_file_name):
#            ut.tic()
            os.remove(full_file_name)
#            ut.toc(label='file_delete (%s)' % self.instance_label)
    
    def redis_store(self, obj, redis_db):
        # Make the Redis key containing the type prefix, and the hex UID code.
        key_name = '%s-%s' % (self.type_prefix, self.uid.hex)
        
        # Put the object in Redis.
#        ut.tic()
        redis_db.set(key_name, io.object_to_gzip_string_pickle(obj))
#        ut.toc(label='redis_store (%s)' % self.instance_label)
    
    def redis_retrieve(self, redis_db):
        # Make the Redis key containing the type prefix, and the hex UID code.
        key_name = '%s-%s' % (self.type_prefix, self.uid.hex) 
        
        # Get and return the object with the key in Redis.
#        ut.tic()
        obj = io.gzip_string_pickle_to_object(redis_db.get(key_name))
#        ut.toc(label='redis_retrieve (%s)' % self.instance_label)
        return obj
    
    def redis_delete(self, redis_db):
        # Make the Redis key containing the type prefix, and the hex UID code.
        key_name = '%s-%s' % (self.type_prefix, self.uid.hex)
        
        # Delete the entry from Redis.
#        ut.tic()
        redis_db.delete(key_name)
#        ut.toc(label='redis_delete (%s)' % self.instance_label)
        
    def show(self):
        print('UUID: %s' % self.uid.hex)
        print('Type Prefix: %s' % self.type_prefix)
        print('File Suffix: %s' % self.file_suffix)
        print('Instance Label: %s' % self.instance_label)
        
class DataStore(object):
    """
    An object allowing storage and retrieval of Python objects using either 
    files or the Redis database.  You can think of it as being a generalized 
    key/value-pair-based database.
    
    Methods:
        __init__(db_mode: str ['redis'], redis_db_URL: str [None]): void -- 
            constructor
        save(): void -- save the state of the DataStore either to file or 
            Redis, depending on the mode
        load(): void -- load the state of the DataStore either from file or 
            Redis, depending on the mode
        get_handle_by_uid(uid: UUID or str): StoreObjectHandle -- get the 
            handle (if any) pointed to by an UID            
        get_uid_from_instance(type_prefix: str, instance_label: str): UUID -- 
            find the UID of the first matching case where a handle in the dict 
            has the same type prefix and instance label        
        add(obj: Object, uid: UUID or str [None], type_label: str ['obj'], 
            file_suffix: str ['.obj'], instance_label: str [''], 
            save_handle_changes: bool [True]): void -- add a Python object to 
            the DataStore, creating also a StoreObjectHandle for managing it, 
            and return the UUID (useful if no UID was passed in, and a 
            new one had to be generated)
        retrieve(uid: UUID or str): Object -- retrieve a Python object 
            stored in the DataStore, keyed by a UID
        update(uid: UUID or str, obj: Object): void -- update a Python object 
            stored in the DataStore, keyed by a UID
        delete(uid: UUID or str, save_handle_changes=True): void -- delete a 
            Python object stored in the DataStore, keyed by a UID
        delete_all(): void -- delete all of the Python objects in the DataStore
        show_handles(): void -- show all of the StoreObjectHandles in the 
            DataStore
        show_redis_keys(): void -- show all of the keys in the Redis database 
            we are using
        clear_redis_keys(): void -- delete all of the keys in the Redis database
            we are using
                    
    Attributes:
        handle_dict (dict) -- the Python dictionary holding the StoreObjectHandles
        db_mode (str) -- the mode of persistence the DataStore uses (either 
            'redis' or 'file')
        redis_db (redis.client.StrictRedis) -- link to the Redis database we 
            are using
        
    Usage:
        >>> data_store = DataStore(redis_db_URL='redis://localhost:6379/0/')                      
    """
    
    def __init__(self, db_mode='redis', redis_db_URL=None):
        # Start with an empty dictionary.
        self.handle_dict = {}
        
        if redis_db_URL is not None:
            self.db_mode = 'redis'
        else:
            self.db_mode = db_mode
        
        # If we are using Redis...
        if self.db_mode == 'redis':
            # Open up the Redis database we want to dedicate to Sciris.
            self.redis_db = redis.StrictRedis.from_url(redis_db_URL)
        
    def save(self):
        # If we are using Redis...
        if self.db_mode == 'redis':
            # Set the entries for all of the data items.
            self.redis_db.set('scirisdatastore-handle_dict', 
                io.object_to_gzip_string_pickle(self.handle_dict))
            self.redis_db.set('scirisdatastore-db_mode', 
                io.object_to_gzip_string_pickle(self.db_mode))
            
        # Otherwise (we are using files)...
        else:
            outfile = open('.\\sciris.ds', 'wb')
            io.pickle.dump(self.handle_dict, outfile)
            io.pickle.dump(self.db_mode, outfile)
    
    def load(self):
        # If we are using Redis...
        if self.db_mode == 'redis':
            if self.redis_db.get('scirisdatastore-handle_dict') is None:
                print('Error: DataStore object has not been saved yet.')
                return None
            
            # Get the entries for all of the data items.
            self.handle_dict = io.gzip_string_pickle_to_object(self.redis_db.get('scirisdatastore-handle_dict'))
            self.db_mode = io.gzip_string_pickle_to_object(self.redis_db.get('scirisdatastore-db_mode'))
        
        # Otherwise (we are using files)...
        else:    
            if not os.path.exists('.\\sciris.ds'):
                print('Error: DataStore object has not been saved yet.')
                return None
            
            infile = open('.\\sciris.ds', 'rb')
            self.handle_dict = io.pickle.load(infile)
            self.db_mode = io.pickle.load(infile)
    
    def get_handle_by_uid(self, uid):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        valid_uid = ut.uuid(uid)
        
        # If we have a valid UUID...
        if valid_uid is not None:
            return self.handle_dict.get(valid_uid, None)
        else:
            return None
        
    def get_uid_from_instance(self, type_prefix, instance_label):
        # Initialize an empty list to put the matches in.
        uid_matches = []
        
        # For each key in the dictionary...
        for key in self.handle_dict:
            # Get the handle pointed to.
            handle = self.handle_dict[key]
            
            # If both the type prefix and instance label match, add the UID
            # of the handle to the list.
            if handle.type_prefix == type_prefix and \
                handle.instance_label == instance_label:
                uid_matches.append(handle.uid)
                
        # If there is no match, return None.        
        if len(uid_matches) == 0:
            return None
        
        # Else, if there is more than one match, give a warning.
        elif len(uid_matches) > 1:
            print('Warning: get_uid_from_instance() only returning the first match.')
            
        # Return the first (and hopefully only) matching UID.  
        return uid_matches[0]
        
    def add(self, obj, uid=None, type_label='obj', file_suffix='.obj', 
        instance_label='', save_handle_changes=True):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.  If no UID is passed in, generate a new one.    
        valid_uid = ut.uuid(uid)
        
        # Create the new StoreObjectHandle.
        new_handle = StoreObjectHandle(valid_uid, type_label, file_suffix, 
            instance_label)
        
        # Add the handle to the dictionary.
        self.handle_dict[valid_uid] = new_handle
        
        # If we are using Redis...
        if self.db_mode == 'redis':
            # Put the object in Redis.
            new_handle.redis_store(obj, self.redis_db)
            
        # Otherwise (we are using files)...
        else:
            # Put the object in a file.
            new_handle.file_store('.', obj)
            
        # Do a save of the database so change is kept.
        if save_handle_changes:
            self.save()
            
        # Return the UUID.
        return valid_uid
    
    def retrieve(self, uid):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        valid_uid = ut.uuid(uid)
        
        # If we have a valid UUID...
        if valid_uid is not None: 
            # Get the handle (if any) matching the UID.
            handle = self.get_handle_by_uid(valid_uid)
            
            # If we found a matching handle...
            if handle is not None:
                # If we are using Redis...
                if self.db_mode == 'redis':        
                    # Return the object pointed to by the handle.
                    return handle.redis_retrieve(self.redis_db)
                
                # Otherwise (we are using files)...
                else:   
                    # Return the object pointed to by the handle.
                    return handle.file_retrieve('.')
                
        # Return None (a failure to find a match).
        return None
    
    def update(self, uid, obj):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        valid_uid = ut.uuid(uid)
        
        # If we have a valid UUID...
        if valid_uid is not None:  
            # Get the handle (if any) matching the UID.
            handle = self.get_handle_by_uid(valid_uid)
            
            # If we found a matching handle...
            if handle is not None:
                # If we are using Redis...
                if self.db_mode == 'redis':   
                    # Overwrite the old copy of the object using the handle.
                    handle.redis_store(obj, self.redis_db)
                    
                # Otherwise (we are using files)...
                else:
                    # Overwrite the old copy of the object using the handle.
                    handle.file_store('.', obj) 
     
    def delete(self, uid, save_handle_changes=True):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        valid_uid = ut.uuid(uid)
        
        # If we have a valid UUID...
        if valid_uid is not None: 
            # Get the handle (if any) matching the UID.
            handle = self.get_handle_by_uid(valid_uid) 
            
            # If we found a matching handle...
            if handle is not None:
                # If we are using Redis...
                if self.db_mode == 'redis': 
                    # Delete the key using the handle.
                    handle.redis_delete(self.redis_db)
                    
                # Otherwise (we are using files)...
                else:        
                    # Delete the file using the handle.
                    handle.file_delete('.')
                
                # Delete the handle from the dictionary.
                del self.handle_dict[valid_uid]
                
                # Do a save of the database so change is kept.
                if save_handle_changes:        
                    self.save()        
               
    def delete_all(self):
        # For each key in the dictionary, delete the key and handle, but 
        # don't do the save of the DataStore object until after the changes 
        # are made.
        all_keys = [key for key in self.handle_dict]
        for key in all_keys:
            self.delete(key, save_handle_changes=False)
        
        # Save the DataStore object.
        self.save()
    
    def show_handles(self):
        # For each key in the dictionary...
        for key in self.handle_dict:
            # Get the handle pointed to.
            handle = self.handle_dict[key]
            
            # Separator line.
            print('--------------------------------------------')
            
            # Show the handle contents.
            handle.show()
            
        # Separator line.
        print('--------------------------------------------')
    
    def show_redis_keys(self):
        # Show all of the keys in the Redis database we are using.
        print(self.redis_db.keys())
        
    def clear_redis_keys(self):
        # Delete all of the keys in the Redis database we are using.
        for key in self.redis_db.keys():
            self.redis_db.delete(key)