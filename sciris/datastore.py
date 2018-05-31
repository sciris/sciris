"""
datastore.py -- code related to Sciris persistence (both files and database)
    
Last update: 3/28/18 (gchadder3)
"""

"""
DEV NOTES: 
* We don't want Sciris users to have to customize this file much, or at all.
* We also don't want classes here for objects that are not intimately tied
to actual to-disk persistence.
* We may want to break out the file and directory related stuff into a 
different file, leaving DataStore-related store here, but maybe file and 
directory management stuff goes in a file called filemanagement.py.
"""

#
# Imports
#

import os
from tempfile import mkdtemp
from shutil import rmtree
import atexit
import redis
import scirisobjects as sobj

# Import cPickle if it is available in your Python setup because it is a 
# faster method.  If it's not available, import the regular pickle library.
try: 
    import cPickle as pickle
except: 
    import pickle
    
from gzip import GzipFile
from cStringIO import StringIO
from contextlib import closing

#
# Globals
#

# These will get set by calling code.

# The path of the Sciris repo.
scirisRootPath = None

# Directory (FileSaveDirectory object) for saved files.
fileSaveDir = None

# Directory (FileSaveDirectory object) for file uploads to be routed to.
uploadsDir = None

# Directory (FileSaveDirectory object) for file downloads to be routed to.
downloadsDir = None

# The DataStore object for persistence for the app.  Gets initialized by
# and loaded by init_datastore().
theDataStore = None

#
# Classes
#

class StoreObjectHandle(object):
    """
    An object associated with a Python object which permits the Python object 
    to be stored in and retrieved from a DataStore object.
    
    Methods:
        __init__(theUID: UUID, theTypePrefix: str ['obj'], 
            theFileSuffix: str ['.obj'], theInstanceLabel: str['']): void --
            constructor
        getUID(): UUID -- return the StoreObjectHandle's UID
        fileStore(dirPath: str, theObject: Object): void -- store theObject in 
            the dirPath directory
        fileRetrieve(dirPath: str): void -- retrieve the stored object from 
            the dirPath directory
        fileDelete(dirPath: str): void -- delete the stored object from the 
            dirPath directory
        redisStore(theObject: Object, redisDb: redis.client.StrictRedis): 
            void -- store theObject in Redis
        redisRetrieve(redisDb: redis.client.StrictRedis): void -- retrieve 
            the stored object from Redis
        redisDelete(redisDb: redis.client.StrictRedis): void -- delete the 
            stored object from Redis
        show(): void -- print the contents of the object
                    
    Attributes:
        uid (UUID) -- the unique ID for the handle (uuid Python library-related)
        typePrefix (str) -- a prefix that gets added to the UUID to give either 
            a file name or a Redis key
        fileSuffix (str) -- a suffix that gets added to files
        instanceLabel (str) -- a name of the object which should at least be 
            unique across other handles of the save typePrefix
        
    Usage:
        >>> newHandle = StoreObjectHandle(uuid.UUID('12345678123456781234567812345678'), 
            typePrefix='project', fileSuffix='.prj', instanceLabel='Project 1')
    """
    
    def __init__(self, theUID, theTypePrefix='obj', theFileSuffix='.obj', 
        theInstanceLabel=''):
        self.uid = theUID
        self.typePrefix = theTypePrefix
        self.fileSuffix = theFileSuffix
        self.instanceLabel = theInstanceLabel
        
    def getUID(self):
        return self.uid
    
    def fileStore(self, dirPath, theObject):
        # Create a filename containing the type prefix, hex UID code, and the
        # appropriate file suffix.
        fileName = '%s-%s%s' % (self.typePrefix, self.uid.hex, self.fileSuffix)
        
        # Generate the full file name with path.
        fullFileName = '%s%s%s' % (dirPath, os.sep, fileName)
        
        # Write the object to a Gzip string pickle file.
        objectToGzipStringPickleFile(fullFileName, theObject)
    
    def fileRetrieve(self, dirPath):
        # Create a filename containing the type prefix, hex UID code, and the
        # appropriate file suffix.
        fileName = '%s-%s%s' % (self.typePrefix, self.uid.hex, self.fileSuffix)
        
        # Generate the full file name with path.
        fullFileName = '%s%s%s' % (dirPath, os.sep, fileName)
        
        # Return object from the Gzip string pickle file.
        return gzipStringPickleFileToObject(fullFileName)
    
    def fileDelete(self, dirPath):
        # Create a filename containing the type prefix, hex UID code, and the
        # appropriate file suffix.
        fileName = '%s-%s%s' % (self.typePrefix, self.uid.hex, self.fileSuffix)
        
        # Generate the full file name with path.
        fullFileName = '%s%s%s' % (dirPath, os.sep, fileName)
        
        # Remove the file if it's there.
        if os.path.exists(fullFileName):
            os.remove(fullFileName)
    
    def redisStore(self, theObject, redisDb):
        # Make the Redis key containing the type prefix, and the hex UID code.
        keyName = '%s-%s' % (self.typePrefix, self.uid.hex)
        
        # Put the object in Redis.
        redisDb.set(keyName, objectToGzipStringPickle(theObject))
    
    def redisRetrieve(self, redisDb):
        # Make the Redis key containing the type prefix, and the hex UID code.
        keyName = '%s-%s' % (self.typePrefix, self.uid.hex) 
        
        # Get and return the object with the key in Redis.
        return gzipStringPickleToObject(redisDb.get(keyName))
    
    def redisDelete(self, redisDb):
        # Make the Redis key containing the type prefix, and the hex UID code.
        keyName = '%s-%s' % (self.typePrefix, self.uid.hex)
        
        # Delete the entry from Redis.
        redisDb.delete(keyName)
        
    def show(self):
        print 'UUID: %s' % self.uid.hex
        print 'Type Prefix: %s' % self.typePrefix
        print 'File Suffix: %s' % self.fileSuffix
        print 'Instance Label: %s' % self.instanceLabel
        
class DataStore(object):
    """
    An object allowing storage and retrieval of Python objects using either 
    files or the Redis database.  You can think of it as being a generalized 
    key/value-pair-based database.
    
    Methods:
        __init__(theDbMode: str ['redis'], redisDbURL: str [None]): void -- 
            constructor
        save(): void -- save the state of the DataStore either to file or 
            Redis, depending on the mode
        load(): void -- load the state of the DataStore either from file or 
            Redis, depending on the mode
        getHandleByUID(theUID: UUID or str): StoreObjectHandle -- get the 
            handle (if any) pointed to by an UID
        add(theObject: Object, theUID: UUID or str, theTypeLabel: str ['obj'], 
            theFileSuffix: str ['.obj'], theInstanceLabel: str [''], 
            saveHandleChanges: bool [True]): void -- add a Python object to 
            the DataStore, creating also a StoreObjectHandle for managing it
        retrieve(theUID: UUID or str): Object -- retrieve a Python object 
            stored in the DataStore, keyed by a UID
        update(theUID: UUID or str, theObject): void -- update a Python object 
            stored in the DataStore, keyed by a UID
        delete(theUID: UUID or str, saveHandleChanges=True): void -- delete a 
            Python object stored in the DataStore, keyed by a UID
        deleteAll(): void -- delete all of the Python objects in the DataStore
        showHandles(): void -- show all of the StoreObjectHandles in the 
            DataStore
        showRedisKeys(): void -- show all of the keys in the Redis database 
            we are using
        clearRedisKeys(): void -- delete all of the keys in the Redis database
            we are using
                    
    Attributes:
        handleDict (dict) -- the Python dictionary holding the StoreObjectHandles
        dbMode (str) -- the mode of persistence the DataStore uses (either 
            'redis' or 'file')
        redisDb (redis.client.StrictRedis) -- link to the Redis database we 
            are using
        
    Usage:
        >>> theDataStore = DataStore(redisDbURL='redis://localhost:6379/1/')                      
    """
    
    def __init__(self, theDbMode='redis', redisDbURL=None):
        # Start with an empty dictionary.
        self.handleDict = {}
        
        if redisDbURL is not None:
            self.dbMode = 'redis'
        else:
            self.dbMode = theDbMode
        
        # If we are using Redis...
        if self.dbMode == 'redis':
            # Open up the Redis database we want to dedicate to Sciris.
            self.redisDb = redis.StrictRedis.from_url(redisDbURL)
        
    def save(self):
        # If we are using Redis...
        if self.dbMode == 'redis':
            # Set the entries for all of the data items.
            self.redisDb.set('scirisdatastore-handleDict', 
                objectToGzipStringPickle(self.handleDict))
            self.redisDb.set('scirisdatastore-dbMode', 
                objectToGzipStringPickle(self.dbMode))
            
        # Otherwise (we are using files)...
        else:
            outfile = open('.\\sciris.ds', 'wb')
            pickle.dump(self.handleDict, outfile)
            pickle.dump(self.dbMode, outfile)
    
    def load(self):
        # If we are using Redis...
        if self.dbMode == 'redis':
            if self.redisDb.get('scirisdatastore-handleDict') is None:
                print 'Error: DataStore object has not been saved yet.'
                return
            
            # Get the entries for all of the data items.
            self.handleDict = gzipStringPickleToObject(self.redisDb.get('scirisdatastore-handleDict'))
            self.dbMode = gzipStringPickleToObject(self.redisDb.get('scirisdatastore-dbMode'))
        
        # Otherwise (we are using files)...
        else:    
            if not os.path.exists('.\\sciris.ds'):
                print 'Error: DataStore object has not been saved yet.'
                return
            
            infile = open('.\\sciris.ds', 'rb')
            self.handleDict = pickle.load(infile)
            self.dbMode = pickle.load(infile)
    
    def getHandleByUID(self, theUID):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        validUID = sobj.getValidUUID(theUID)
        
        # If we have a valid UUID...
        if validUID is not None:
            return self.handleDict.get(validUID, None)
        else:
            return None
        
    def getUIDFromInstance(self, typePrefix, instanceLabel):
        # Initialize an empty list to put the matches in.
        UIDmatches = []
        
        # For each key in the dictionary...
        for theKey in self.handleDict:
            # Get the handle pointed to.
            theHandle = self.handleDict[theKey]
            
            # If both the type prefix and instance label match, add the UID
            # of the handle to the list.
            if theHandle.typePrefix == typePrefix and \
                theHandle.instanceLabel == instanceLabel:
                UIDmatches.append(theHandle.uid)
                
        # If there is no match, return None.        
        if len(UIDmatches) == 0:
            return None
        
        # Else, if there is more than one match, give a warning.
        elif len(UIDmatches) > 1:
            print ('Warning: getUIDFromInstance() only returning the first match.')
            
        # Return the first (and hopefully only) matching UID.  
        return UIDmatches[0]
        
    def add(self, theObject, theUID, theTypeLabel='obj', theFileSuffix='.obj', 
        theInstanceLabel='', saveHandleChanges=True):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        validUID = sobj.getValidUUID(theUID)
        
        # If we have a valid UUID...
        if validUID is not None:       
            # Create the new StoreObjectHandle.
            newHandle = StoreObjectHandle(validUID, theTypeLabel, theFileSuffix, 
                theInstanceLabel)
            
            # Add the handle to the dictionary.
            self.handleDict[validUID] = newHandle
            
            # If we are using Redis...
            if self.dbMode == 'redis':
                # Put the object in Redis.
                newHandle.redisStore(theObject, self.redisDb)
                
            # Otherwise (we are using files)...
            else:
                # Put the object in a file.
                newHandle.fileStore('.', theObject)
                
            # Do a save of the database so change is kept.
            if saveHandleChanges:
                self.save()
    
    def retrieve(self, theUID):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        validUID = sobj.getValidUUID(theUID)
        
        # If we have a valid UUID...
        if validUID is not None: 
            # Get the handle (if any) matching the UID.
            theHandle = self.getHandleByUID(validUID)
            
            # If we found a matching handle...
            if theHandle is not None:
                # If we are using Redis...
                if self.dbMode == 'redis':        
                    # Return the object pointed to by the handle.
                    return theHandle.redisRetrieve(self.redisDb)
                
                # Otherwise (we are using files)...
                else:   
                    # Return the object pointed to by the handle.
                    return theHandle.fileRetrieve('.')
                
        # Return None (a failure to find a match).
        return None
    
    def update(self, theUID, theObject):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        validUID = sobj.getValidUUID(theUID)
        
        # If we have a valid UUID...
        if validUID is not None:  
            # Get the handle (if any) matching the UID.
            theHandle = self.getHandleByUID(validUID)
            
            # If we found a matching handle...
            if theHandle is not None:
                # If we are using Redis...
                if self.dbMode == 'redis':   
                    # Overwrite the old copy of the object using the handle.
                    theHandle.redisStore(theObject, self.redisDb)
                    
                # Otherwise (we are using files)...
                else:
                    # Overwrite the old copy of the object using the handle.
                    theHandle.fileStore('.', theObject)     
     
    def delete(self, theUID, saveHandleChanges=True):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        validUID = sobj.getValidUUID(theUID)
        
        # If we have a valid UUID...
        if validUID is not None: 
            # Get the handle (if any) matching the UID.
            theHandle = self.getHandleByUID(validUID) 
            
            # If we found a matching handle...
            if theHandle is not None:
                # If we are using Redis...
                if self.dbMode == 'redis': 
                    # Delete the key using the handle.
                    theHandle.redisDelete(self.redisDb)
                    
                # Otherwise (we are using files)...
                else:        
                    # Delete the file using the handle.
                    theHandle.fileDelete('.')
                
                # Delete the handle from the dictionary.
                del self.handleDict[validUID]
                
                # Do a save of the database so change is kept.
                if saveHandleChanges:        
                    self.save()        
               
    def deleteAll(self):
        # For each key in the dictionary, delete the key and handle, but 
        # don't do the save of the DataStore object until after the changes 
        # are made.
        allKeys = [key for key in self.handleDict]
        for theKey in allKeys:
            self.delete(theKey, saveHandleChanges=False)
        
        # Save the DataStore object.
        self.save()
    
    def showHandles(self):
        # For each key in the dictionary...
        for theKey in self.handleDict:
            # Get the handle pointed to.
            theHandle = self.handleDict[theKey]
            
            # Separator line.
            print '--------------------------------------------'
            
            # Show the handle contents.
            theHandle.show()
            
        # Separator line.
        print '--------------------------------------------'
    
    def showRedisKeys(self):
        # Show all of the keys in the Redis database we are using.
        print self.redisDb.keys()
        
    def clearRedisKeys(self):
        # Delete all of the keys in the Redis database we are using.
        for theKey in self.redisDb.keys():
            self.redisDb.delete(theKey)
 

# Directory-related classes
# NOTE: If enough code ends up here, we may want to break it out into another 
# file.
    
class FileSaveDirectory(object):
    """
    An object wrapping a directory where files may get saved by the web 
    application.
    
    Methods:
        __init__(dirPath: str [None], tempDir: bool [False]): void -- 
            constructor
        cleanup(): void -- clean up after web app is exited
        clear(): void -- erase the contents of the directory
        delete(): void -- delete the entire directory
                    
    Attributes:
        dirPath (str) -- the full path of the directory on disk
        isTempDir (bool) -- is the directory to be spawned on startup and 
            erased on exit?
        
    Usage:
        >>> newDir = FileSaveDirectory(transferDirPath, tempDir=True)
    """
    
    def __init__(self, dirPath=None, tempDir=False):
        # Set whether we are a temp directory.
        self.isTempDir = tempDir
               
        # If no path is specified, create the temp directory.
        if dirPath is None:
            self.dirPath = mkdtemp()
            
        # Otherwise...
        else:
            # Set the path to what was passed in.
            self.dirPath = dirPath
            
            # If the directory doesn't exist yet, create it.
            if not os.path.exists(dirPath):            
                os.mkdir(dirPath)
            
        # Register the cleanup method to be called on web app exit.
        atexit.register(self.cleanup)
            
    def cleanup(self):
        # If we are a temp directory, do the cleanup.
        if self.isTempDir:
            # Show cleanup message.
            print '>> Cleaning up FileSaveDirectory at %s' % self.dirPath
            
            # Delete the entire directory (file contents included).
            self.delete()
            
    def clear(self):
        # Delete the entire directory (file contents included).
        rmtree(self.dirPath)
        
        # Create a fresh direcgtory
        os.mkdir(self.dirPath)
    
    def delete(self):
        # Delete the entire directory (file contents included).
        rmtree(self.dirPath)
        
#
# Pickle / unpickle functions
#

def objectToStringPickle(theObject):
    return pickle.dumps(theObject, protocol=-1)

def stringPickleToObject(theStringPickle):
    return pickle.loads(theStringPickle)

def objectToGzipStringPickleFile(fullFileName, theObject, compresslevel=5):
    # Object a Gzip file object to write to and set the compression level 
    # (which the function defaults to 5, since higher is much slower, but 
    # not much more compact).
    with GzipFile(fullFileName, 'wb', compresslevel=compresslevel) as fileobj:
        # Write the string pickle conversion of the object to the file.
        fileobj.write(objectToStringPickle(theObject))

def gzipStringPickleFileToObject(fullFileName):
    # Object a Gzip file object to read from.
    with GzipFile(fullFileName, 'rb') as fileobj:
        # Read the string pickle from the file.
        theStringPickle = fileobj.read()
        
    # Return the object gotten from the string pickle.    
    return stringPickleToObject(theStringPickle)

def objectToGzipStringPickle(theObject):
    # Start with a null result.
    result = None
    
    # Open a "fake file."
    with closing(StringIO()) as output:
        # Open a Gzip-compressing way to write to this "file."
        with GzipFile(fileobj=output, mode='wb') as fileobj: 
            # Write the string pickle conversion of the object to the "file."
            fileobj.write(objectToStringPickle(theObject))
            
        # Move the mark to the beginning of the "file."
        output.seek(0)
        
        # Read all of the content into result.
        result = output.read()
        
    # Return the read-in result.
    return result

def gzipStringPickleToObject(theGzipStringPickle):
    # Open a "fake file" with the Gzip string pickle in it.
    with closing(StringIO(theGzipStringPickle)) as output:
        # Set a Gzip reader to pull from the "file."
        with GzipFile(fileobj=output, mode='rb') as fileobj: 
            # Read the string pickle from the "file" (applying Gzip 
            # decompression).
            theStringPickle = fileobj.read()  
            
            # Extract the object from the string pickle.
            theObject = stringPickleToObject(theStringPickle)
            
    # Return the object.
    return theObject

#
# RPC functions
#
