"""
datastore.py -- code related to Sciris database persistence
    
Last update: 2018sep20
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
from .sc_users import User
from .sc_tasks import Task

# Global variables
default_url         = 'redis://127.0.0.1:6379/' # The default URL for the Redis database
default_settingskey = '!DataStoreSettings'      # Key for holding DataStore settings
default_separator   = '::'                     # Define the separator between a key type and uid


#################################################################
### Classes
#################################################################

__all__ = ['Blob', 'DataStoreSettings', 'DataStore']


class Blob(sc.prettyobj):
    ''' Wrapper for any Python object we want to store in the DataStore. '''
    
    def __init__(self, obj=None, key=None, objtype=None, uid=None, force=True):
        # Handle input arguments
        if uid is None: 
            if force:
                uid = sc.uuid()
            else:
                errormsg = 'DataStore: Not creating a new Blob UUID since force is set to False: key=%s, objtype=%s, uid=%s, obj=%s' % (key, objtype, uid, obj)
                raise Exception(errormsg)
        if not key: key = '%s%s%s' % (objtype, default_separator, uid)
        
        # Set attributes
        self.key      = key
        self.objtype  = objtype
        self.uid      = uid
        self.created  = sc.now()
        self.modified = [self.created]
        self.obj      = obj
        return None
    
    def update(self):
        ''' When the object is updated, append the current time to the modified list '''
        now = sc.now()
        self.modified.append(now)
        return now
        
    def save(self, obj):
        ''' Save new object to the Blob '''
        self.obj = obj
        self.update()
        return None
    
    def load(self):
        ''' Load data from the Blob '''
        output = self.obj
        return output


class DataStoreSettings(sc.prettyobj):
    ''' Global settings for the DataStore '''
    
    def __init__(self, settings=None, tempfolder=None, separator=None):
        
        ''' Initialize with highest priority given to the inputs, then the stored settings, then the defaults '''
        
        # 1. Arguments
        self.tempfolder = tempfolder
        self.separator  = separator
        
        # 2. Existing settings
        if not settings:
            self.is_new    = True
            old_tempfolder = None
            old_separator  = None
        else:
            self.is_new    = False
            old_tempfolder = settings.tempfolder
            old_separator  = settings.separator
        
        # 3. Defaults
        def_tempfolder = tempfile.mkdtemp()
        def_separator  = default_separator
        
        # Iterate in order 
        tempfolder_list = [old_tempfolder, def_tempfolder]
        separator_list  = [old_separator,  def_separator]
        for folder in tempfolder_list:
            if not self.tempfolder:
                self.tempfolder = folder
        for sep in separator_list:
            if not self.separator:
                self.separator = sep
        
        return None


class DataStore(sc.prettyobj):
    """ Interface to the Redis database. """
    
    def __init__(self, redis_url=None, tempfolder=None, separator=None, settingskey=None, verbose=True):
        ''' Establishes data-structure wrapping a particular Redis URL. '''
        
        # Handle the Redis URL
        if not redis_url:            redis_url = default_url + '0' # e.g. sw.DataStore()
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
        if not settingskey: settingskey = default_settingskey
        settings = DataStoreSettings(settings=self.get(settingskey), tempfolder=tempfolder, separator=separator)
        self.tempfolder = settings.tempfolder
        self.separator  = settings.separator
        self.is_new     = settings.is_new
        self.set(settingskey, settings) # Save back to the database
        
        # Handle the temporary folder
        if not os.path.exists(self.tempfolder):
            os.makedirs(self.tempfolder)
            atexit.register(self._rmtempfolder) # Only register this if we've just created the temp folder
        
        return settings

    
    def _rmtempfolder(self):
        ''' Remove the temporary folder that was created '''
        if os.path.exists(self.tempfolder):
            if self.verbose: print('Removing up temporary folder at %s' % self.tempfolder)
            shutil.rmtree(self.tempfolder)
        return None
    
    
    def makekey(self, objtype, uid):
        ''' Create a key from an object type and UID '''
        if objtype: key = '%s%s%s' % (objtype, self.separator, uid) # Construct a key with an object type and separator
        else:       key = '%s'     % uid                            # ...or, just return the UID
        return key
        
    
    def getkey(self, key=None, objtype=None, uid=None, obj=None, fulloutput=None):
        '''
        Get a valid database key, either from a given key (do nothing), or else from
        a supplied objtype and uid, or else read them from the object supplied. The
        idea is for this method to be as forgiving as possible for different possible
        combinations of inputs.
        '''
        # Handle optional input arguments
        if fulloutput is None: fulloutput = False
        
        # Handle different sources for things
        props = ['key', 'objtype', 'uid']
        args    = {'key':key,  'objtype':objtype, 'uid':uid}  # These are what have been supplied by the user
        fromobj = {'key':None, 'objtype':None,    'uid':None} # This is from the object
        final   = {'key':None, 'objtype':None,    'uid':None} # These will eventually be the output values -- copy of args
        
        # Look for missing properties from the object
        if obj:
            if hasattr(obj, 'key'):     fromobj['key']     = obj.key
            if hasattr(obj, 'objtype'): fromobj['objtype'] = obj.objtype
            if hasattr(obj, 'uid'):     fromobj['uid']     = obj.uid
        
        # Populate all non-None entries from the input arguments
        for p in props:
            if args[p]:
                final[p] = str(args[p]) # Convert to string since you don't know what crazy thing might be passed

        # Populate what we can from the object, if it hasn't already been populated
        for p in props:
            if fromobj[p] and not final[p]:
                final[p] = fromobj[p]
        
        # If the key is supplied but other things aren't, try to create them now
        if final['key'] and (not final['objtype'] or not final['uid']):
            splitkey = final['key'].split(self.separator, 1)
            if len(splitkey)==2: # Check that the key split properly
                if not final['objtype']: final['objtype'] = splitkey[0]
                if not final['uid']:     final['uid']     = splitkey[1]
        
        # If we're still stuck, try making a new uid
        if not final['key'] and not final['uid']:
            final['uid'] = str(sc.uuid())
        
        # If everything is supplied except the key, create it
        if not final['key']:
            if final['objtype'] and final['uid']: # Construct a key from the object type and UID
                final['key'] = self.makekey(objtype=final['objtype'], uid=final['uid'])
            elif not final['objtype'] and final['uid']: # Otherwise, just use the UID
                final['key'] = final['uid']
            
        # Check that it's found, and if not, treat the key as a UID and try again
        keyexists = self.exists(final['key']) # Check to see whether a match has been found
        if not keyexists: # If not, treat the key as a UID instead
            newkey = self.makekey(objtype=final['objtype'], uid=final['key'])
            newkeyexists = self.exists(newkey) # Check to see whether a match has been found
            if newkeyexists:
                final['key'] = newkey
        
        # Return what we need to return
        if fulloutput: return final['key'], final['objtype'], final['uid']
        else:          return final['key']
    
    
    def set(self, key=None, obj=None, objtype=None, uid=None):
        ''' Alias to redis.set() '''
        key = self.getkey(key=key, objtype=objtype, uid=uid, obj=obj)
        objstr = sc.dumpstr(obj)
        output = self.redis.set(key, objstr)
        return output
    
    
    def get(self, key=None, obj=None, objtype=None, uid=None, die=False):
        ''' Alias to redis.get() '''
        key = self.getkey(key=key, objtype=objtype, uid=uid, obj=obj)
        objstr = self.redis.get(key)
        if objstr:
            output = sc.loadstr(objstr)
        else:                  
            if die:
                errormsg = 'Redis key "%s" not found (obj=%s, objtype=%s, uid=%s)' % (key, obj, objtype, uid)
                raise Exception(errormsg)
            else:
                output = None
        return output
    
    
    def delete(self, key=None, obj=None, objtype=None, uid=None, die=None):
        ''' Alias to redis.delete() '''
        if die is None: die = True
        key = self.getkey(key=key, objtype=objtype, uid=uid, obj=obj)
        output = self.redis.delete(key)
        if self.verbose: print('DataStore: deleted key %s' % key)
        return output
    
    
    def flushdb(self):
        ''' Alias to redis.flushdb() '''
        output = self.redis.flushdb()
        if self.verbose: print('DataStore flushed.')
        return output
    
    
    def exists(self, key):
        ''' Alias to redis.exists() '''
        output = self.redis.exists(key)
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
        
    
    def saveblob(self, obj, key=None, objtype=None, uid=None, overwrite=True, die=None):
        '''
        Add a new or update existing Blob in Redis, returns key. If key is None, 
        constructs a key from the Blob (objtype:uid); otherwise, updates the Blob with the 
        provided key.
        '''
        if die is None: die = True
        key, objtype, uid = self.getkey(key=key, objtype=objtype, uid=uid, obj=obj, fulloutput=True)
        blob = self.get(key)
        if blob:
            self._checktype(key, blob, 'Blob')
            if overwrite:
                blob.save(obj)
            else:
                errormsg = 'DataStore: Blob %s already exists and overwrite is set to False' % key
                if die: raise Exception(errormsg)
                else:   print(errormsg)
        else:
            blob = Blob(key=key, objtype=objtype, uid=uid, obj=obj)
        self.set(key=key, obj=blob)
        if self.verbose: print('DataStore: Blob "%s" saved' % key)
        return key
    
    
    def loadblob(self, key=None, objtype=None, uid=None, die=None):
        ''' Load a blob from Redis '''
        if die is None: die = True
        key = self.getkey(key=key, objtype=objtype, uid=uid)
        blob = self.get(key)
        if die: self._checktype(key, blob, 'Blob')
        if isinstance(blob, Blob):
            obj = blob.load()
            if self.verbose: print('DataStore: Blob "%s" loaded' % key)
            return obj
        else:
            if self.verbose: print('DataStore: Blob "%s" not found' % key)
            return None
    
    
    def saveuser(self, user, overwrite=True, die=None):
        '''
        Add a new or update existing User in Redis, returns key.
        '''
        if die is None: die = True
        key, objtype, username = self.getkey(objtype='user', uid=user.username, fulloutput=True)
        olduser = self.get(key)
        if olduser and not overwrite:
            errormsg = 'DataStore: User %s already exists, not overwriting' % key
            if die: raise Exception(errormsg)
            else:   print(errormsg)
        else:
            self._checktype(key, user, 'User')
            self.set(key=key, obj=user)
            if self.verbose:
                if olduser: print('DataStore: User "%s" updated' % key)
                else:       print('DataStore: User "%s" created' % key)
        return key
    
    
    def loaduser(self, username=None, key=None, die=None):
        ''' Load a user from Redis '''
        if die is None: die = True
        key = self.getkey(key=key, objtype='user', uid=username)
        user = self.get(key)
        if die: self._checktype(key, user, 'User')
        if isinstance(user, User):
            if self.verbose: print('DataStore: User "%s" loaded' % key)
            return user
        else:
            if self.verbose: print('DataStore: User "%s" not found' % key)
            return None
        
    
    def savetask(self, task, key=None, uid=None, overwrite=None):
        '''
        Add a new or update existing Task in Redis, returns key.
        '''
        if overwrite is None: overwrite = True
        key, objtype, uid = self.getkey(key=key, objtype='task', uid=uid, obj=task, fulloutput=True)
        oldtask = self.get(key)
        if oldtask and not overwrite:
            errormsg = 'DataStore: Task %s already exists' % key
            raise Exception(errormsg)
        else:
            self._checktype(key, task, 'Task')
            self.set(key=key, obj=task)
        if self.verbose: print('DataStore: Task "%s" saved' % key)
        return key
    
    
    def loadtask(self, key=None, uid=None, die=None):
        ''' Load a user from Redis '''
        if die is None: die = False # Here, we won't always know whether the task exists
        key = self.getkey(key=key, objtype='task', uid=uid)
        task = self.get(key)
        if die: self._checktype(key, task, 'Task')
        if isinstance(task, Task):
            if self.verbose: print('DataStore: Task "%s" loaded' % key)
            return task
        else:
            if self.verbose: print('DataStore: Task "%s" not found' % key)
            return None