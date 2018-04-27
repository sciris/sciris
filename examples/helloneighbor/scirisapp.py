"""
scirisapp.py -- classes for Sciris (Flask-based) apps 
    
Last update: 4/26/18 (gchadder3)
"""

#
# Classes
#

class ScirisApp(object):
    """
    [this text copied from ScirisObject class]
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