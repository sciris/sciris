"""
Blobs.py -- classes for Sciris objects which are generally managed
    
Last update: 2018aug20
"""

import sciris as sc
from . import sc_datastore as ds

__all__ = ['Blob', 'BlobDict']

class Blob(object):
    """
    A general Sciris object (base class for all such objects).  Objects of this 
    type are meant to be storable using Sciris' DataStore persistence 
    functionality.
    
    Methods:
        __init__(uid: UUID [None], type_prefix: str ['obj'], 
            file_suffix: str ['.obj'], instance_label: str ['']): 
            void -- constructor
        in_data_store(): bool -- is the object stored in the DataStore?
        load_from_copy(other_obj): void -- assuming other_obj is another 
            object of our type, copy its contents to us
        add_to_data_store(): void -- add ourselves to the Sciris DataStore 
            (theDataStore object)
        load_copy_from_data_store(uid: UUID [None]): Blob -- return 
            the Blob matching a UUID (usually ourselves in UUID is 
            None)
        load_from_data_store(self, uid=None): void -- overwrite our current 
            state with what's in the DataStore (usually from our stored 
            object, but you can pass in another UUID also to copy from another 
            stored object)
        update_data_store(): void -- update the DataStore representation from 
            our current state
        delete_from_data_store(): void -- delete ourselves from the DataStore        
        show(): void -- print the contents of the object       
        get_user_front_end_repr(): dict -- get a JSON-friendly dictionary 
            representation of the object state the front-end uses for non-
            admin purposes
        get_admin_front_end_repr(): dict -- get a JSON-friendly dictionary
            representation of the object state the front-end uses for admin
            purposes
                    
    Attributes:
        uid (UUID) -- the unique ID for the user (uuid Python library-related)
        type_prefix (str) -- a prefix that gets added to the UUID to give either 
            a file name or a Redis key
        file_suffix (str) -- a suffix that gets added to files
        instance_label (str) -- a name of the object which should at least be 
            unique across other handles of the save type_prefix
            
    Usage:
        >>> obj = Blob(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def  __init__(self, uid=None, type_prefix='obj', file_suffix='.obj', instance_label=''):       
        # Get a valid UUID from what is passed in, or if None is passed in, 
        # get a new ID.
        self.uid = sc.uuid(uid) 
            
        # Set the other variables that might be used with DataStore.
        self.type_prefix = type_prefix
        self.file_suffix = file_suffix
        self.instance_label = instance_label
        
    def in_data_store(self):
        return (ds.gv.data_store.retrieve(self.uid) is not None)
    
    def load_from_copy(self, other_obj):
        if type(other_obj) == type(self):
            self.uid = other_obj.uid
            self.type_prefix = other_obj.type_prefix
            self.instance_label = other_obj.instance_label
            
    def add_to_data_store(self):
        # Check to see if the object is already in the DataStore, and give an error if so.
        if self.in_data_store():
            print('Error: Object ''%s'' is already in DataStore.' % self.uid.hex)
            return
        
        # Add our representation to the DataStore.
        ds.gv.data_store.add(self, self.uid, self.type_prefix, self.file_suffix, 
            self.instance_label)
        
    def load_copy_from_data_store(self, uid=None):
        if uid is None:
            if not self.in_data_store():
                print('Error: Object ''%s'' is not in DataStore.' % self.uid.hex)
                return None
            else:
                return ds.gv.data_store.retrieve(self.uid)
        else:
            return ds.gv.data_store.retrieve(uid)
            
    def load_from_data_store(self, uid=None):
        # Get a copy from the DataStore.
        copy_from_store = self.load_copy_from_data_store(uid)
        
        # Copy the internal information over from the copy to ourselves.
        self.load_from_copy(copy_from_store)
        
    def update_data_store(self):
        # Give an error if the object is not in the DataStore.
        if not self.in_data_store():
            print('Error: Object ''%s'' is not in DataStore.' % self.uid.hex)
            return
        
        # Update our DataStore representation with our current state. 
        ds.gv.data_store.update(self.uid, self) 
        
    def delete_from_data_store(self):
        # Give an error if the object is not in the DataStore.
        if not self.in_data_store():
            print('Error: Object ''%s'' is not in DataStore.' % self.uid.hex)
            return
        
        # Delete ourselves from the DataStore.
        ds.gv.data_store.delete(self.uid)
    
    def show(self):
        print('--------------------------------------------')
        print('          UUID: %s' % self.uid.hex)
        print('   Type prefix: %s' % self.type_prefix)
        print('   File suffix: %s' % self.file_suffix)
        print('Instance label: %s' % self.instance_label)
        in_data_store = self.in_data_store()
        if in_data_store:
            print('In DataStore?: Yes')
        else:
            print('In DataStore?: No')
        #print '--------------------------------------------'
        
    def get_user_front_end_repr(self):
        obj_info = {
            'Blob': {
                'instance_label': self.instance_label                
            }
        }
        return obj_info
    
    def get_admin_front_end_repr(self):
        obj_info = {
            'Blob': {
                'UID': self.uid.hex, 
                'type_prefix': self.type_prefix, 
                'file_suffix': self.file_suffix, 
                'instance_label': self.instance_label                
            }
        }
        return obj_info 


class BlobDict(Blob):
    """
    A collection of Blobs (stored in an odict).
    
    Methods:
        __init__(uid: UUID [None], type_prefix: str ['collection'], 
            file_suffix: str ['.scl'], instance_label: str [''], 
            objs_within_dict: bool [False]): void -- constructor        
        load_from_copy(other_obj): void -- assuming other_obj is another 
            object of our type, copy its contents to us (calls the 
            Blob superclass version of this method also)
        get_object_by_uid(uid: UUID): Blob -- get a Blob out of 
            the collection by the UUID passed in
        get_all_objects(): list of Blobs -- get all of the Blobs 
           and put them in a list
        add_object(obj: Blob): void -- add a Blob to the 
            collection
        update_object(obj: Blob): void -- update a Blob 
            passed in to the collection
        delete_object_by_uid(uid: UUID): void -- delete the Blob 
            indexed by the UUID from the collection
        delete_all_objects(): void -- delete all Blobs from the 
            collection
        show(): void -- print the contents of the collection, including the 
            object information as well as the objects
                    
    Attributes:
        obj_dict (dict) -- the Python dictionary holding the Blobs
        objs_within_coll (bool) -- are the objects themselves within the 
            collection? If they are not, they are kept as separate data store 
            entries and handle_dict is used.
        ds_uuid_set (set) -- the Python set holding UUIDs to DataStore 
            entries for the Blobs (used only if objs_within_coll is False)
        
    Usage:
        >>> objs = BlobDict(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def __init__(self, uid, type_prefix='collection', 
        file_suffix='.scl', instance_label='', objs_within_coll=False):
        # Set superclass parameters.
        super(BlobDict, self).__init__(uid, type_prefix, 
             file_suffix, instance_label)
        
        # Create an odict to hold the Blobs.
        self.obj_dict = sc.odict()
        
        # Set the flag for whether the objects are stored within the collection 
        # itself (or outside in separate data store entries).
        self.objs_within_coll = objs_within_coll
        
        # Set the datastore UUID set to empty.
        self.ds_uuid_set = set()
        
    def load_from_copy(self, other_obj):
        if type(other_obj) == type(self):
            # Do the superclass copying.
            super(BlobDict, self).load_from_copy(other_obj)
            
            # Copy other items specific to this class.
            self.obj_dict = sc.odict(other_obj.obj_dict)
            self.objs_within_coll = other_obj.objs_within_coll
            self.ds_uuid_set = other_obj.ds_uuid_set
            
    def get_object_by_uid(self, uid):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.
        valid_uid = sc.uuid(uid)
        
        # If we have a valid UUID...
        if valid_uid is not None:
            # If we have the objects within the collection itself...
            if self.objs_within_coll:
                # Return the object.
                return self.obj_dict.get(valid_uid, None)
            
            # Otherwise, if the UUID is in the set...
            elif valid_uid in self.ds_uuid_set:
                # Return the object that is in the global DataStore object.
                return ds.gv.data_store.retrieve(valid_uid)
            
            # Otherwise (no match)...
            else:
                return None
        else:
            return None
        
    def get_all_objects(self):
        if self.objs_within_coll:
            return [self.obj_dict[key] for key in self.obj_dict]
        else:
            return [ds.gv.data_store.retrieve(uid) for uid in self.ds_uuid_set]
    
    def add_object(self, obj):
        # If we are storing things inside the obj_dict...
        if self.objs_within_coll:        
            # Add the object to the hash table, keyed by the UID.
            self.obj_dict[obj.uid] = obj
            
        # Otherwise, we are using the UUID set...
        else:
            # Put the UUID in the set.
            self.ds_uuid_set.add(obj.uid)
            
            # If the object is not actually in the DataStore, put it there.
            if not obj.in_data_store():
                obj.add_to_data_store()
        
        # Update our DataStore representation if we are there. 
        if self.in_data_store():
            self.update_data_store()
            
    def update_object(self, obj):
        # If we are storing things inside the obj_dict...
        if self.objs_within_coll:           
            # Do the same behavior as add_object().
            self.add_object(obj)
            
        # Otherwise, we are using the UUID set...
        else:
            # Update the object in the global DataStore object.
            ds.gv.data_store.update(obj.uid, obj)
            
    def delete_object_by_uid(self, uid):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        valid_uid = sc.uuid(uid)
        
        # If we have a valid UUID...
        if valid_uid is not None:
            # Default to not needing to update the data store.
            need_to_update = False
            
            # If we are storing things inside the obj_dict...
            if self.objs_within_coll:             
                # Get the object pointed to by the UID.
                obj = self.obj_dict[valid_uid]
                
                # If a match is found...
                if obj is not None:
                    # Remove entries from obj_dict.
                    del self.obj_dict[valid_uid]
                    
                    # Set to update the data store
                    need_to_update = True
                    
            # Otherwise, we are using the UUID set...
            else:
                # If the UUID is in the set...
                if valid_uid in self.ds_uuid_set:
                    # Remove the UUID from the set.
                    self.ds_uuid_set.remove(valid_uid)
                    
                    # Delete the object in the global DataStore object.
                    ds.gv.data_store.delete(valid_uid)
                    
                    # Set to update the data store
                    need_to_update = True                    
            
            # Update our DataStore representation if we are there. 
            if need_to_update and self.in_data_store():
                self.update_data_store()
    
    def delete_all_objects(self):
        # If we are storing things inside the obj_dict...
        if self.objs_within_coll:          
            # Reset the Python dicts.
            self.obj_dict = {}
            
        # Otherwise, we are using the UUID set...
        else:
            # For each item in the set...
            for uid in self.ds_uuid_set:
                # Delete the object with that UID in the DataStore.
                ds.gv.data_store.delete(uid)
                
            # Clear the UUID set.
            self.ds_uuid_set.clear()
         
        # Update our DataStore representation if we are there. 
        if self.in_data_store():
            self.update_data_store()  
        
    def show(self):
        super(BlobDict, self).show()   # Show superclass attributes.
        if self.objs_within_coll: print('Objects stored within dict?: Yes')
        else:                     print('Objects stored within dict?: No')
        print('---------------------')
        print('Contents')
        print('---------------------')
        
        if self.objs_within_coll: # If we are storing things inside the obj_dict...
            for key in self.obj_dict: # For each key in the dictionary...
                obj = self.obj_dict[key] # Get the object pointed to.
                obj.show() # Show the handle contents.
        else: # Otherwise, we are using the UUID set.
            for uid in self.ds_uuid_set: # For each item in the set...
                ds.gv.data_store.retrieve(uid).show() # Show the object with that UID in the DataStore.
        print('--------------------------------------------')