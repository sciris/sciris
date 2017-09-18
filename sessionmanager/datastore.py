"""
datastore.py -- code related to Sciris data storage (both files and database)
    
Last update: 9/17/17 (gchadder3)
"""

# NOTE: We don't want Sciris users to have to customize this file much.

#
# Imports
#

import os
import redis
import uuid

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
# Classes
#

class StoreObjectHandle(object):
    def __init__(self, theUID, theTypeLabel='obj', theFileSuffix='.obj', 
        theInstanceLabel=''):
        self.uid = theUID
        self.typeLabel = theTypeLabel
        self.fileSuffix = theFileSuffix
        self.instanceLabel = theInstanceLabel
        
    def getUID(self):
        return self.uid
    
    def fileStore(self, dirPath, theObject):
        # Create a filename containing the type label, hex UID code, and the
        # appropriate file suffix.
        fileName = '%s-%s%s' % (self.typeLabel, self.uid.hex, self.fileSuffix)
        
        # Generate the full file name with path.
        fullFileName = '%s%s%s' % (dirPath, os.sep, fileName)
        
        # Write the object to a Gzip string pickle file.
        objectToGzipStringPickleFile(fullFileName, theObject)
    
    def fileRetrieve(self, dirPath):
        # Create a filename containing the type label, hex UID code, and the
        # appropriate file suffix.
        fileName = '%s-%s%s' % (self.typeLabel, self.uid.hex, self.fileSuffix)
        
        # Generate the full file name with path.
        fullFileName = '%s%s%s' % (dirPath, os.sep, fileName)
        
        # Return object from the Gzip string pickle file.
        return gzipStringPickleFileToObject(fullFileName)
    
    def fileDelete(self, dirPath):
        # Create a filename containing the type label, hex UID code, and the
        # appropriate file suffix.
        fileName = '%s-%s%s' % (self.typeLabel, self.uid.hex, self.fileSuffix)
        
        # Generate the full file name with path.
        fullFileName = '%s%s%s' % (dirPath, os.sep, fileName)
        
        # Remove the file if it's there.
        if os.path.exists(fullFileName):
            os.remove(fullFileName)
    
    def redisStore(self, theObject):
        # Make the Redis key containing the type label, and the hex UID code.
        keyName = '%s-%s' % (self.typeLabel, self.uid.hex)
        
        # Put the object in Redis.
        redisDb.set(keyName, objectToGzipStringPickle(theObject))
    
    def redisRetrieve(self):
        # Make the Redis key containing the type label, and the hex UID code.
        keyName = '%s-%s' % (self.typeLabel, self.uid.hex) 
        
        # Get and return the object with the key in Redis.
        return gzipStringPickleToObject(redisDb.get(keyName))
    
    def redisDelete(self):
        # Make the Redis key containing the type label, and the hex UID code.
        keyName = '%s-%s' % (self.typeLabel, self.uid.hex)
        
        # Delete the entry from Redis.
        redisDb.delete(keyName)  
        
class DataStore(object):
    def __init__(self, theDbMode='redis'):
        self.handleList = []
        self.uuidHashes = {}
        self.dbMode = theDbMode
        
    def dump(self):
        # If we are using Redis...
        if self.dbMode == 'redis':
            # Set the entries for all of the data items.
            redisDb.set('scirisdatastore-handleList', 
                objectToGzipStringPickle(self.handleList))
            redisDb.set('scirisdatastore-uuidHashes', 
                objectToGzipStringPickle(self.uuidHashes))
            redisDb.set('scirisdatastore-dbMode', 
                objectToGzipStringPickle(self.dbMode))
            
        # Otherwise (we are using files)...
        else:
            outfile = open('.\\sciris.ds', 'wb')
            pickle.dump(self.handleList, outfile)
            pickle.dump(self.uuidHashes, outfile)
            pickle.dump(self.dbMode, outfile)
    
    def load(self):
        # If we are using Redis...
        if self.dbMode == 'redis':
            # Get the entries for all of the data items.
            self.handleList = gzipStringPickleToObject(redisDb.get('scirisdatastore-handleList'))
            self.uuidHashes = gzipStringPickleToObject(redisDb.get('scirisdatastore-uuidHashes'))
            self.dbMode = gzipStringPickleToObject(redisDb.get('scirisdatastore-dbMode'))
        
        # Otherwise (we are using files)...
        else:        
            infile = open('.\\sciris.ds', 'rb')
            self.handleList = pickle.load(infile)
            self.uuidHashes = pickle.load(infile)
            self.dbMode = pickle.load(infile)
    
    def getHandleByUID(self, theUID):
        return self.handleList[self.uuidHashes[theUID]]
    
    def add(self, theObject, theUID, theTypeLabel='obj', theFileSuffix='.obj', 
        theInstanceLabel=''):
        # Create the new StoreObjectHandle
        newHandle = StoreObjectHandle(theUID, theTypeLabel, theFileSuffix, 
            theInstanceLabel)
        
        # Add the handle to the list.
        self.handleList.append(newHandle)
        self.uuidHashes[theUID] = len(self.handleList) - 1
        
        # If we are using Redis...
        if self.dbMode == 'redis':
            # Put the object in Redis.
            newHandle.redisStore(theObject)
            
        # Otherwise (we are using files)...
        else:
            # Put the object in a file.
            newHandle.fileStore('.', theObject)
    
    def retrieve(self, theUID):
        # If we are using Redis...
        if self.dbMode == 'redis':        
            # Return the object pointed to by the handle.
            return self.getHandleByUID(theUID).redisRetrieve()
        
        # Otherwise (we are using files)...
        else:   
            # Return the object pointed to by the handle.
            return self.getHandleByUID(theUID).fileRetrieve('.')
    
    def update(self, theUID, theObject):
        # If we are using Redis...
        if self.dbMode == 'redis':        
            # Overwrite the old copy of the object using the handle.
            self.getHandleByUID(theUID).redisStore(theObject)
            
        # Otherwise (we are using files)...
        else:
            # Overwrite the old copy of the object using the handle.
            self.getHandleByUID(theUID).fileStore('.', theObject)     
     
    def delete(self, theUID):
        # If we are using Redis...
        if self.dbMode == 'redis': 
            # Delete the key using the handle.
            self.getHandleByUID(theUID).redisDelete()
            
        # Otherwise (we are using files)...
        else:        
            # Delete the file using the handle.
            self.getHandleByUID(theUID).fileDelete('.')
        
        # Delete the handle from the list.
        
    def deleteAll(self):
        # Delete all of the resources pointed to by all of the handlers.
        # Delete all handlers.
        # Delete the saved version of the store itself.
        pass
    
  
# Wraps a directory storage place for the session manager.
class DirectoryStore(DataStore):
    def __init__(self, dirPath):
        self.dirPath = dirPath
        
        
#
# Functions
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
# Script code
#

# Open up the Redis database we want to dedicate to Sciris.
redisDb = redis.StrictRedis.from_url('redis://localhost:6379/1/')
