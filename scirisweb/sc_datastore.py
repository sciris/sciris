"""
datastore.py -- code related to Sciris database persistence
    
Last update: 2018sep19
"""


#################################################################
### Imports and global variables
#################################################################

# Imports
import os
import atexit
import tempfile
import shutil
import redis
import sciris as sc
from .sc_objects import Blob, User, Task

# Global variables
default_url         = 'redis://localhost:6379/' # The default URL for the Redis database
default_settingskey = '!DataStoreSettings'      # Key for holding DataStore settings
default_separator   = '__|__'                   # Define the separator between a key type and uid


#################################################################
### Classes
#################################################################

__all__ = ['DataStoreSettings', 'DataStore']


class DataStoreSettings(sc.prettyobj):
    ''' Global settings for the DataStore '''
    
    def __init__(self, settings=None, tempfolder=None, separator=None):
        
        # Initialize
        self.tempfolder = None
        self.separator  = None
        self.is_new     = True
        
        # Are we creating this for the first time? If not, load from existing settings object
        if settings is not None:
            self.is_new     = False
            self.tempfolder = settings.tempfolder
            self.separator  = settings.separator
            
        # Handle the folder and separator
        if self.tempfolder is None: self.tempfolder = tempfile.mkdtemp()
        if self.separator  is None: self.separator  = default_separator
        
        return None


class DataStore(sc.prettyobj):
    """ Interface to the Redis database. """
    
    def __init__(self, redis_url=None, tempfolder=None, separator=None, settingskey=None, verbose=True):
        ''' Establishes data-structure wrapping a particular Redis URL. '''
        
        # Handle the Redis URL
        if redis_url is None:        redis_url = default_url + '0' # e.g. sw.DataStore()
        elif sc.isnumber(redis_url): redis_url = default_url + '%i'%redis_url # e.g. sw.DataStore(3)
        self.redis_url  = redis_url
        self.tempfolder = None # Populated by self.settings()
        self.separator  = None # Populated by self.settings()
        self.is_new     = None # Populated by self.settings()
        self.verbose    = verbose
        
        # Create Redis
        self.redis = redis.StrictRedis.from_url(self.redis_url)
        self.settings(settingskey=settingskey, redis_url=redis_url, tempfolder=tempfolder, separator=separator) # Set or get the settings
        if self.is_new: actionstring = 'created'
        else:           actionstring = 'loaded'
        if self.verbose: print('DataStore %s at %s with temp folder %s' % (actionstring, self.redis_url, self.tempfolder))
        return None
    
    
    def settings(self, settingskey=None, redis_url=None, tempfolder=None, separator=None):
        ''' Handle the DataStore settings '''
        if settingskey is None: settingskey = default_settingskey
        settings = DataStoreSettings(settings=self.get(settingskey), tempfolder=tempfolder, separator=separator)
        self.tempfolder = settings.tempfolder
        self.separator  = settings.separator
        self.is_new     = settings.is_new
        self.set(settingskey, settings) # Save back to the database
        
        # Handle the temporary folder
        if not os.path.exists(self.tempfolder):
            os.mkdir(self.tempfolder)
            atexit.register(self._rmtempfolder) # Only register this if we've just created the temp folder
        
        return settings

    
    def _rmtempfolder(self):
        ''' Remove the temporary folder that was created '''
        if os.path.exists(self.tempfolder):
            if self.verbose: print('Removing up temporary folder at %s' % self.tempfolder)
            shutil.rmtree(self.tempfolder)
        return None
        
    
    def _makekey(self, key=None, objtype=None, uid=None, obj=None, forcecreate=False, fulloutput=False):
        '''
        Construct a database key, either from a given key (do nothing), or else from
        a supplied objtype and uid, or else read them from the object (e.g. Blob) supplied.
        '''
        
        # Get any missing properties from the object
        if obj is not None: # Populate any missing values from the object
            if key     is None and hasattr(obj, 'key'):     key     = obj.key
            if objtype is None and hasattr(obj, 'objtype'): objtype = obj.objtype
            if uid     is None and hasattr(obj, 'uid'):     uid     = obj.uid
        
        
        # Optionally force non-None objtype and uid
        if forcecreate:
            if objtype is None: objtype = 'object'
            if uid     is None: uid     = str(sc.uuid())
        
        # Construct the key if it doesn't exist
        if key is None: # Only worry about the key if none is provided
            if objtype is not None and uid is not None:
                key = '%s%s%s' % (objtype, self.separator, uid)
            else:
                errormsg = 'To create a key, you must either specify the key, or the objtype and the uid, or use an object with these'
                raise Exception(errormsg)
        
        # Determine the objtype and uid from the key
        if fulloutput:
            splitkey = key.split(self.separator)
            if len(splitkey)==2:
                if objtype and objtype != splitkey[0]: print('Key warning: requested objtypes do not match (%s vs. %s)' % (objtype, splitkey[0]))
                if uid     and uid     != splitkey[1]: print('Key warning: requested UIDs do not match (%s vs. %s)'     % (uid,     splitkey[1]))
                objtype = splitkey[0]
                uid  = splitkey[1]
            elif (objtype is None or uid is None):
                errormsg = 'Warning: could not split key "%s" into objtype and uid using separator "%s"' % (key, self.separator)
                print(errormsg) # Still warn, but don't crash
        
        # Return what we need to return
        if fulloutput: return key, objtype, uid
        else:          return key
    
    
    
    def set(self, key=None, obj=None, objtype=None, uid=None):
        ''' Alias to redis.set() '''
        key = self._makekey(key, objtype, uid, obj)
        objstr = sc.dumpstr(obj)
        output = self.redis.set(key, objstr)
        return output
    
    
    def get(self, key=None, obj=None, objtype=None, uid=None, die=False):
        ''' Alias to redis.get() '''
        key = self._makekey(key, objtype, uid, obj)
        objstr = self.redis.get(key)
        if objstr is not None:
            output = sc.loadstr(objstr)
        else:                  
            if die:
                errormsg = 'Redis key "%s" not found (obj=%s, objtype=%s, uid=%s)' % (key, obj, objtype, uid)
                raise Exception(errormsg)
            else:
                output = None
        return output
    
    
    def delete(self, key=None, obj=None, objtype=None, uid=None):
        ''' Alias to redis.delete() '''
        key = self._makekey(key, objtype, uid, obj)
        output = self.redis.delete(key)
        if self.verbose: print('DataStore: deleted key %s' % key)
        return output
    
    
    def flushdb(self):
        ''' Alias to redis.flushdb() '''
        output = self.redis.flushdb()
        if self.verbose: print('DataStore flushed.')
        return output
    
    
    def keys(self, pattern=None):
        ''' Alias to redis.keys() '''
        if pattern is None: pattern = '*'
        output = self.redis.keys(pattern=pattern)
        return output
    
    
    def items(self, pattern=None):
        ''' Return all found items in an odict '''
        output = sc.odict()
        keys = self.keys(pattern=pattern)
        for key in keys:
            output[key] = self.get(key)
        return output
    
    
    def _checktype(self, key, obj, objtype):
        if   objtype == 'Blob': objclass = Blob
        elif objtype == 'User': objclass = User
        elif objtype == 'Task': objclass = Task
        else:
            errormsg = 'Unrecognized type "%s": must be Blob, User, or Task'
            raise Exception(errormsg)
        
        if obj is None:
            errormsg = 'Cannot load %s as a %s: key not found' % (key, objtype)
            raise Exception(errormsg)
        
        if not isinstance(obj, objclass):
            errormsg = 'Cannot load %s as a %s since it is %s' % (key, objtype, type(obj))
            raise Exception(errormsg)
        
        return None
        
    
    def saveblob(self, data, key=None, obj=None, objtype=None, uid=None, overwrite=True):
        '''
        Add a new or update existing Blob in Redis, returns key. If key is None, 
        constructs a key from the Blob (objtype:uid); otherwise, updates the Blob with the 
        provided key.
        '''
        key, objtype, uid = self._makekey(key, objtype, uid, obj, forcecreate=True, fulloutput=True)
        blob = self.get(key)
        if blob:
            self._checktype(key, blob, 'Blob')
            if overwrite:
                blob.save(data)
            else:
                errormsg = 'DataStore: Blob %s already exists and overwrite is set to False' % key
                raise Exception(errormsg)
        else:
            blob = Blob(key=key, objtype=objtype, uid=uid, data=data)
        self.set(key=key, obj=blob)
        if self.verbose: print('DataStore: Blob "%s" saved' % key)
        return key
    
    
    def loadblob(self, key=None, objtype=None, uid=None, die=True):
        ''' Load a blob from Redis '''
        key = self._makekey(key, objtype, uid)
        blob = self.get(key)
        if die: self._checktype(key, blob, 'Blob')
        data = blob.load()
        if self.verbose: print('DataStore: Blob "%s" loaded' % key)
        return data
    
    
    def saveuser(self, user, overwrite=False):
        '''
        Add a new or update existing User in Redis, returns key.
        '''
        key, objtype, username = self._makekey(objtype='user', uid=user.username, fulloutput=True)
        olduser = self.get(key)
        if olduser and not overwrite:
            errormsg = 'DataStore: User %s already exists' % key
            raise Exception(errormsg)
        else:
            self._checktype(key, user, 'User')
            self.set(key=key, obj=user)
        if self.verbose: print('DataStore: User "%s" saved' % key)
        return key
    
    
    def loaduser(self, username=None, key=None, die=True):
        ''' Load a user from Redis '''
        key = self._makekey(key=key, objtype='user', uid=username)
        user = self.get(key)
        if die: self._checktype(key, user, 'User')
        if self.verbose: print('DataStore: User "%s" loaded' % key)
        return user
        
    
    def savetask(self, task, key=None, uid=None, overwrite=False):
        '''
        Add a new or update existing Task in Redis, returns key.
        '''
        key, objtype, uid = self._makekey(key=key, objtype='task', uid=uid, obj=task, fulloutput=True)
        oldtask = self.get(key)
        if oldtask and not overwrite:
            errormsg = 'DataStore: Task %s already exists' % key
            raise Exception(errormsg)
        else:
            self._checktype(key, task, 'Task')
            self.set(key=key, obj=task)
        if self.verbose: print('DataStore: Task "%s" saved' % key)
        return key
    
    
    def loadtask(self, key=None, uid=None, die=True):
        ''' Load a user from Redis '''
        key = self._makekey(key=key, objtype='task', uid=uid)
        task = self.get(key)
        if die: self._checktype(key, task, 'Task')
        if self.verbose: print('DataStore: Task "%s" loaded' % key)
        return task