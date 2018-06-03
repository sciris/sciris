"""
scirisobjects.py -- classes for Sciris objects which are generally managed
    
Last update: 11/22/17 (gchadder3)
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
        __init__(theUID: UUID [None], theTypePrefix: str ['obj'], 
            theFileSuffix: str ['.obj'], theInstanceLabel: str ['']): 
            void -- constructor
        inDataStore(): bool -- is the object stored in the DataStore?
        loadFromCopy(otherObject): void -- assuming otherObject is another 
            object of our type, copy its contents to us
        addToDataStore(): void -- add ourselves to the Sciris DataStore 
            (theDataStore object)
        loadCopyFromDataStore(theUID: UUID [None]): ScirisObject -- return 
            the ScirisObject matching a UUID (usually ourselves in UUID is 
            None)
        loadFromDataStore(self, theUID=None): void -- overwrite our current 
            state with what's in the DataStore (usually from our stored 
            object, but you can pass in another UUID also to copy from another 
            stored object)
        updateDataStore(): void -- update the DataStore representation from 
            our current state
        deleteFromDataStore(): void -- delete ourselves from the DataStore        
        show(): void -- print the contents of the object       
        getUserFrontEndRepr(): dict -- get a JSON-friendly dictionary 
            representation of the object state the front-end uses for non-
            admin purposes
        getAdminFrontEndRepr(): dict -- get a JSON-friendly dictionary
            representation of the object state the front-end uses for admin
            purposes
                    
    Attributes:
        uid (UUID) -- the unique ID for the user (uuid Python library-related)
        typePrefix (str) -- a prefix that gets added to the UUID to give either 
            a file name or a Redis key
        fileSuffix (str) -- a suffix that gets added to files
        instanceLabel (str) -- a name of the object which should at least be 
            unique across other handles of the save typePrefix
            
    Usage:
        >>> theObj = ScirisObject(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def  __init__(self, theUID=None, theTypePrefix='obj', theFileSuffix='.obj', 
        theInstanceLabel=''):        
        # If a UUID was passed in...
        if theUID is not None:
            # Make sure the argument is a valid UUID, converting a hex text to a
            # UUID object, if needed.        
            validUID = getValidUUID(theUID) 
            
            # If a validUID was found, use it.
            if validUID is not None:
                self.uid = validUID
            # Otherwise, generate a new random UUID using uuid4().
            else:
                self.uid = uuid.uuid4()
        # Otherwise, generate a new random UUID using uuid4().
        else:
            self.uid = uuid.uuid4()
            
        # Set the other variables that might be used with DataStore.
        self.typePrefix = theTypePrefix
        self.fileSuffix = theFileSuffix
        self.instanceLabel = theInstanceLabel
        
    def inDataStore(self):
        return (ds.theDataStore.retrieve(self.uid) is not None)
    
    def loadFromCopy(self, otherObject):
        if type(otherObject) == type(self):
            self.uid = otherObject.uid
            self.typePrefix = otherObject.typePrefix
            self.instanceLabel = otherObject.instanceLabel
            
    def addToDataStore(self):
        # Check to see if the object is already in the DataStore, and give an 
        # error if so.
        if self.inDataStore():
            print 'Error: Object ''%s'' is already in DataStore.' % self.uid.hex
            return
        
        # Add our representation to the DataStore.
        ds.theDataStore.add(self, self.uid, self.typePrefix, self.fileSuffix, 
            self.instanceLabel)
        
    def loadCopyFromDataStore(self, theUID=None):
        if theUID is None:
            if not self.inDataStore():
                print 'Error: Object ''%s'' is not in DataStore.' % self.uid.hex
                return None
            else:
                return ds.theDataStore.retrieve(self.uid)
        else:
            return ds.theDataStore.retrieve(theUID)
            
    def loadFromDataStore(self, theUID=None):
        # Get a copy from the DataStore.
        copyFromStore = self.loadCopyFromDataStore(theUID)
        
        # Copy the internal information over from the copy to ourselves.
        self.loadFromCopy(copyFromStore)
        
    def updateDataStore(self):
        # Give an error if the object is not in the DataStore.
        if not self.inDataStore():
            print 'Error: Object ''%s'' is not in DataStore.' % self.uid.hex
            return
        
        # Update our DataStore representation with our current state. 
        ds.theDataStore.update(self.uid, self) 
        
    def deleteFromDataStore(self):
        # Give an error if the object is not in the DataStore.
        if not self.inDataStore():
            print 'Error: Object ''%s'' is not in DataStore.' % self.uid.hex
            return
        
        # Delete ourselves from the DataStore.
        ds.theDataStore.delete(self.uid)
    
    def show(self):
        print '--------------------------------------------'
        print 'UUID: %s' % self.uid.hex
        print 'Type Prefix: %s' % self.typePrefix
        print 'File Suffix: %s' % self.fileSuffix
        print 'Instance Label: %s' % self.instanceLabel
        inDataStore = self.inDataStore()
        if inDataStore:
            print 'In DataStore?: Yes'
        else:
            print 'In DataStore?: No'
        #print '--------------------------------------------'
        
    def getUserFrontEndRepr(self):
        objInfo = {
            'scirisobject': {
                'instancelabel': self.instanceLabel                
            }
        }
        return objInfo
    
    def getAdminFrontEndRepr(self):
        objInfo = {
            'scirisobject': {
                'UID': self.uid.hex, 
                'typeprefix': self.typePrefix, 
                'filesuffix': self.fileSuffix, 
                'instancelabel': self.instanceLabel                
            }
        }
        return objInfo 
          
class ScirisCollection(ScirisObject):
    """
    A collection of ScirisObjects (stored in a Python dict theObjectDict).
    
    Methods:
        __init__(theUID: UUID [None], theTypePrefix: str ['collection'], 
            theFileSuffix: str ['.scl'], theInstanceLabel: str ['']): 
            void -- constructor        
        loadFromCopy(otherObject): void -- assuming otherObject is another 
            object of our type, copy its contents to us (calls the 
            ScirisObject superclass version of this method also)
        getObjectByUID(theUID: UUID): ScirisObject -- get a ScirisObject out of 
            the collection by the UUID passed in
        getAllObjects(): list of ScirisObjects -- get all of the ScirisObjects 
           and put them in a list
        addObject(theObject: ScirisObject): void -- add a ScirisObject to the 
            collection
        updateObject(theObject: ScirisObject): void -- update a ScirisObject 
            passed in to the collection
        deleteObjectByUID(theUID: UUID): void -- delete the ScirisObject 
            indexed by the UUID from the collection
        deleteAllObjects(): void -- delete all ScirisObjects from the 
            collection
        show(): void -- print the contents of the collection, including the 
            object information as well as the objects
                    
    Attributes:
        theObjectDict (dict) -- the Python dictionary holding the ScirisObjects
        
    Usage:
        >>> theObjs = ScirisCollection(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def __init__(self, theUID, theTypePrefix='collection', 
        theFileSuffix='.scl', theInstanceLabel=''):
        # Set superclass parameters.
        super(ScirisCollection, self).__init__(theUID, theTypePrefix, 
             theFileSuffix, theInstanceLabel)
        
        # Create a Python dict to hold the ScirisObjects.
        self.theObjectDict = {}
        
    def loadFromCopy(self, otherObject):
        if type(otherObject) == type(self):
            # Do the superclass copying.
            super(ScirisCollection, self).loadFromCopy(otherObject)
            
            self.theObjectDict = otherObject.theObjectDict
            
    def getObjectByUID(self, theUID):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.
        validUID = getValidUUID(theUID)
        
        # If we have a valid UUID, return the matching ScirisObject (if any); 
        # otherwise, return None.
        if validUID is not None:
            return self.theObjectDict.get(validUID, None)
        else:
            return None
        
    def getAllObjects(self):
        return [self.theObjectDict[theKey] for theKey in self.theObjectDict]
    
    def addObject(self, theObject):
        # Add the object to the hash table, keyed by the UID.
        self.theObjectDict[theObject.uid] = theObject
        
        # Update our DataStore representation if we are there. 
        if self.inDataStore():
            self.updateDataStore()
            
    def updateObject(self, theObject):
        # Do the same behavior as addObject().
        self.addObject(theObject)
            
    def deleteObjectByUID(self, theUID):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        validUID = getValidUUID(theUID)
        
        # If we have a valid UUID...
        if validUID is not None:
            # Get the object pointed to by the UID.
            theObject = self.theObjectDict[validUID]
            
            # If a match is found...
            if theObject is not None:
                # Remove entries from theObjectDict.
                del self.theObjectDict[validUID]
                
                # Update our DataStore representation if we are there. 
                if self.inDataStore():
                    self.updateDataStore()
    
    def deleteAllObjects(self):
        # Reset the Python dicts.
        self.theObjectDict = {}
        
        # Update our DataStore representation if we are there. 
        if self.inDataStore():
            self.updateDataStore()  
        
    def show(self):
        # Show superclass attributes.
        super(ScirisCollection, self).show()  
        
        print '---------------------'
        print 'Contents'
        print '---------------------'
        
        # For each key in the dictionary...
        for theKey in self.theObjectDict:
            # Get the object pointed to.
            theObject = self.theObjectDict[theKey]
            
            # Separator line.
            #print '--------------------------------------------'
            
            # Show the handle contents.
            theObject.show()
            
        # Separator line.
        print '--------------------------------------------'
        
#
# Other utility functions
#

def getValidUUID(uidParam):
    # Get the type of the parameter passed in.
    paramType = type(uidParam)
    
    # Return what was passed in if it is already the right type.
    if paramType == uuid.UUID:
        return uidParam
    
    # Try to do the conversion and if it fails, set the conversion to None.
    try:
        convertParam = uuid.UUID(uidParam)
    except:
        convertParam = None
    
    # Return the converted value.
    return convertParam 