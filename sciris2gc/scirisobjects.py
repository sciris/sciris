"""
scirisobjects.py -- classes for Sciris objects which are generally managed
    
Last update: 5/21/18 (gchadder3)
"""

#
# Imports
#

import datastore as ds
import uuid

#
# Classes
#

class ScirisObject(object):
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
        load_copy_from_data_store(uid: UUID [None]): ScirisObject -- return 
            the ScirisObject matching a UUID (usually ourselves in UUID is 
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
        >>> theObj = ScirisObject(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def  __init__(self, uid=None, type_prefix='obj', file_suffix='.obj', 
        instance_label=''):        
        # If a UUID was passed in...
        if uid is not None:
            # Make sure the argument is a valid UUID, converting a hex text to a
            # UUID object, if needed.        
            valid_uid = get_valid_uuid(uid) 
            
            # If a valid_uid was found, use it.
            if valid_uid is not None:
                self.uid = valid_uid
            # Otherwise, generate a new random UUID using uuid4().
            else:
                self.uid = uuid.uuid4()
        # Otherwise, generate a new random UUID using uuid4().
        else:
            self.uid = uuid.uuid4()
            
        # Set the other variables that might be used with DataStore.
        self.type_prefix = type_prefix
        self.file_suffix = file_suffix
        self.instance_label = instance_label
        
    def in_data_store(self):
        return (ds.data_store.retrieve(self.uid) is not None)
    
    def load_from_copy(self, other_obj):
        if type(other_obj) == type(self):
            self.uid = other_obj.uid
            self.type_prefix = other_obj.type_prefix
            self.instance_label = other_obj.instance_label
            
    def add_to_data_store(self):
        # Check to see if the object is already in the DataStore, and give an 
        # error if so.
        if self.in_data_store():
            print 'Error: Object ''%s'' is already in DataStore.' % self.uid.hex
            return
        
        # Add our representation to the DataStore.
        ds.data_store.add(self, self.uid, self.type_prefix, self.file_suffix, 
            self.instance_label)
        
    def load_copy_from_data_store(self, uid=None):
        if uid is None:
            if not self.in_data_store():
                print 'Error: Object ''%s'' is not in DataStore.' % self.uid.hex
                return None
            else:
                return ds.data_store.retrieve(self.uid)
        else:
            return ds.data_store.retrieve(uid)
            
    def load_from_data_store(self, uid=None):
        # Get a copy from the DataStore.
        copy_from_store = self.load_copy_from_data_store(uid)
        
        # Copy the internal information over from the copy to ourselves.
        self.load_from_copy(copy_from_store)
        
    def update_data_store(self):
        # Give an error if the object is not in the DataStore.
        if not self.in_data_store():
            print 'Error: Object ''%s'' is not in DataStore.' % self.uid.hex
            return
        
        # Update our DataStore representation with our current state. 
        ds.data_store.update(self.uid, self) 
        
    def delete_from_data_store(self):
        # Give an error if the object is not in the DataStore.
        if not self.in_data_store():
            print 'Error: Object ''%s'' is not in DataStore.' % self.uid.hex
            return
        
        # Delete ourselves from the DataStore.
        ds.data_store.delete(self.uid)
    
    def show(self):
        print '--------------------------------------------'
        print 'UUID: %s' % self.uid.hex
        print 'Type Prefix: %s' % self.type_prefix
        print 'File Suffix: %s' % self.file_suffix
        print 'Instance Label: %s' % self.instance_label
        in_data_store = self.in_data_store()
        if in_data_store:
            print 'In DataStore?: Yes'
        else:
            print 'In DataStore?: No'
        #print '--------------------------------------------'
        
    def get_user_front_end_repr(self):
        objInfo = {
            'scirisobject': {
                'instance_label': self.instance_label                
            }
        }
        return objInfo
    
    def get_admin_front_end_repr(self):
        objInfo = {
            'scirisobject': {
                'UID': self.uid.hex, 
                'type_prefix': self.type_prefix, 
                'file_suffix': self.file_suffix, 
                'instance_label': self.instance_label                
            }
        }
        return objInfo 
          
class ScirisCollection(ScirisObject):
    """
    A collection of ScirisObjects (stored in a Python dict obj_dict).
    
    Methods:
        __init__(uid: UUID [None], type_prefix: str ['collection'], 
            file_suffix: str ['.scl'], instance_label: str ['']): 
            void -- constructor        
        load_from_copy(other_obj): void -- assuming other_obj is another 
            object of our type, copy its contents to us (calls the 
            ScirisObject superclass version of this method also)
        get_object_by_uid(uid: UUID): ScirisObject -- get a ScirisObject out of 
            the collection by the UUID passed in
        get_all_objects(): list of ScirisObjects -- get all of the ScirisObjects 
           and put them in a list
        add_object(obj: ScirisObject): void -- add a ScirisObject to the 
            collection
        update_object(obj: ScirisObject): void -- update a ScirisObject 
            passed in to the collection
        delete_object_by_uid(uid: UUID): void -- delete the ScirisObject 
            indexed by the UUID from the collection
        delete_all_objects(): void -- delete all ScirisObjects from the 
            collection
        show(): void -- print the contents of the collection, including the 
            object information as well as the objects
                    
    Attributes:
        obj_dict (dict) -- the Python dictionary holding the ScirisObjects
        
    Usage:
        >>> theObjs = ScirisCollection(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def __init__(self, uid, type_prefix='collection', 
        file_suffix='.scl', instance_label=''):
        # Set superclass parameters.
        super(ScirisCollection, self).__init__(uid, type_prefix, 
             file_suffix, instance_label)
        
        # Create a Python dict to hold the ScirisObjects.
        self.obj_dict = {}
        
    def load_from_copy(self, other_obj):
        if type(other_obj) == type(self):
            # Do the superclass copying.
            super(ScirisCollection, self).load_from_copy(other_obj)
            
            self.obj_dict = other_obj.obj_dict
            
    def get_object_by_uid(self, uid):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.
        valid_uid = get_valid_uuid(uid)
        
        # If we have a valid UUID, return the matching ScirisObject (if any); 
        # otherwise, return None.
        if valid_uid is not None:
            return self.obj_dict.get(valid_uid, None)
        else:
            return None
        
    def get_all_objects(self):
        return [self.obj_dict[key] for key in self.obj_dict]
    
    def add_object(self, obj):
        # Add the object to the hash table, keyed by the UID.
        self.obj_dict[obj.uid] = obj
        
        # Update our DataStore representation if we are there. 
        if self.in_data_store():
            self.update_data_store()
            
    def update_object(self, obj):
        # Do the same behavior as add_object().
        self.add_object(obj)
            
    def delete_object_by_uid(self, uid):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        valid_uid = get_valid_uuid(uid)
        
        # If we have a valid UUID...
        if valid_uid is not None:
            # Get the object pointed to by the UID.
            obj = self.obj_dict[valid_uid]
            
            # If a match is found...
            if obj is not None:
                # Remove entries from obj_dict.
                del self.obj_dict[valid_uid]
                
                # Update our DataStore representation if we are there. 
                if self.in_data_store():
                    self.update_data_store()
    
    def delete_all_objects(self):
        # Reset the Python dicts.
        self.obj_dict = {}
        
        # Update our DataStore representation if we are there. 
        if self.in_data_store():
            self.update_data_store()  
        
    def show(self):
        # Show superclass attributes.
        super(ScirisCollection, self).show()  
        
        print '---------------------'
        print 'Contents'
        print '---------------------'
        
        # For each key in the dictionary...
        for key in self.obj_dict:
            # Get the object pointed to.
            obj = self.obj_dict[key]
            
            # Separator line.
            #print '--------------------------------------------'
            
            # Show the handle contents.
            obj.show()
            
        # Separator line.
        print '--------------------------------------------'
        
#
# Other utility functions
#

def get_valid_uuid(uid_param):
    # Get the type of the parameter passed in.
    param_type = type(uid_param)
    
    # Return what was passed in if it is already the right type.
    if param_type == uuid.UUID:
        return uid_param
    
    # Try to do the conversion and if it fails, set the conversion to None.
    try:
        convert_param = uuid.UUID(uid_param)
    except:
        convert_param = None
    
    # Return the converted value.
    return convert_param 