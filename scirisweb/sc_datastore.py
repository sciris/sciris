"""
datastore.py -- code related to Sciris database persistence
    
Last update: 2018sep19
"""

# Imports
import os
import atexit
import tempfile
import shutil
import redis
import sciris as sc
from .sc_objects import Blob, User, Task

__all__ = ['DataStore']


class DataStore(object):
    """
    Interface to the Redis database.                   
    """
    
    def __init__(self, redis_url=None, tempfolder=None, separator=None):
        ''' Establishes data-structure wrapping a particular Redis URL. '''
        
        # Handle the Redis URL
        default_url = 'redis://localhost:6379/'
        if redis_url is None:        redis_url = default_url + '0' # e.g. sw.DataStore()
        elif sc.isnumber(redis_url): redis_url = default_url + '%i'%redis_url # e.g. sw.DataStore(3)
        self.redis_url = redis_url
        
        # Handle the temporary folder
        self.tempfolder = tempfolder if tempfolder is not None else tempfile.mkdtemp()
        if not os.path.exists(self.tempfolder):
            os.mkdir(self.tempfolder)
            atexit.register(self._rmtempfolder) # Only register this if we've just created the temp folder
        
        # Create Redis
        self.redis = redis.StrictRedis.from_url(self.redis_url)
        self.separator = separator if separator is not None else '__|__' # Define the separator between a key type and uid
        print('New DataStore initialized at %s with temp folder %s' % (self.redis_url, self.tempfolder))
        return None
    
    
    def _rmtempfolder(self):
        ''' Remove the temporary folder that was created '''
        if os.path.exists(self.tempfolder):
            print('Removing up temporary folder at %s' % self.tempfolder)
            shutil.rmtree(self.tempfolder)
        return None
        
    
    def _makekey(self, key=None, objtype=None, uid=None, obj=None, create=False, fulloutput=False):
        '''
        Construct a database key, either from a given key (do nothing), or else from
        a supplied objtype and uid, or else read them from the object (e.g. Blob) supplied.
        '''
        
        # Get any missing properties from the object
        if obj is not None: # Populate any missing values from the object
            if key     is None and hasattr(obj, 'key'):     key     = obj.key
            if objtype is None and hasattr(obj, 'objtype'): objtype = obj.objtype
            if uid     is None and hasattr(obj, 'uid'):     uid     = obj.uid
        
        # Construct the key if it doesn't exist
        if key is None: # Only worry about the key if none is provided
            if objtype is not None and uid is not None:
                key = '%s%s%s' % (objtype, self.separator, uid)
            elif create:
                if objtype is None: objtype = 'object'
                if uid     is None: uid     = str(sc.uuid())
                key = '%s%s%s' % (objtype, self.separator, uid)
            else:
                errormsg = 'To create a key, you must either specify the key, or the objtype and the uid, or use an object with these'
                raise Exception(errormsg)
        
        # Determine the objtype and uid from the key
        splitkey = key.split(self.separator)
        if len(splitkey)==2:
            if objtype and objtype != splitkey[0]: print('Key warning: requested objtypes do not match (%s vs. %s)' % (objtype, splitkey[0]))
            if uid     and uid     != splitkey[1]: print('Key warning: requested UIDs do not match (%s vs. %s)'     % (uid,     splitkey[1]))
            objtype = splitkey[0]
            uid  = splitkey[1]
        else:
            errormsg = 'Warning: could not split key "%s" into objtype and uid using separator "%s"' % (key, self.separator)
            if fulloutput and objtype is None or uid is None: # Don't worry about it unless full output is requested
                raise Exception(errormsg)
            else:
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
    
    
    def get(self, key=None, obj=None, objtype=None, uid=None):
        ''' Alias to redis.get() '''
        key = self._makekey(key, objtype, uid, obj)
        objstr = self.redis.get(key)
        if objstr is not None: output = sc.loadstr(objstr)
        else:                  output = None
        return output
    
    
    def delete(self, key=None, obj=None, objtype=None, uid=None):
        ''' Alias to redis.delete() '''
        key = self._makekey(key, objtype, uid, obj)
        output = self.redis.delete(key)
        return output
    
    
    def flushdb(self):
        ''' Alias to redis.flushdb() '''
        output = self.redis.flushdb()
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
        key, objtype, uid = self._makekey(key, objtype, uid, obj, fulloutput=True)
        blob = self.get(key)
        if blob:
            self._checktype(key, blob, 'Blob')
            if overwrite:
                blob.save(data)
            else:
                errormsg = 'Blob %s already exists and overwrite is set to False' % key
                raise Exception(errormsg)
        else:
            blob = Blob(key=key, objtype=objtype, uid=uid, data=data)
        self.set(key=key, obj=blob)
        print('Blob "%s" saved' % key)
        return key
    
    
    def loadblob(self, key=None, objtype=None, uid=None, die=True):
        ''' Load a blob from Redis '''
        key = self._makekey(key, objtype, uid)
        blob = self.get(key)
        if die: self._checktype(key, blob, 'Blob')
        data = blob.load()
        print('Blob "%s" loaded' % key)
        return data
    
    
    def saveuser(self, user, overwrite=False):
        '''
        Add a new or update existing User in Redis, returns key.
        '''
        key, objtype, username = self._makekey(objtype='user', uid=user.username, fulloutput=True)
        olduser = self.get(key)
        if olduser and not overwrite:
            errormsg = 'User %s already exists' % key
            raise Exception(errormsg)
        else:
            self._checktype(key, user, 'User')
            self.set(key=key, obj=user)
        print('User "%s" saved' % key)
        return key
    
    
    def loaduser(self, username=None, key=None, die=True):
        ''' Load a user from Redis '''
        key = self._makekey(key=key, objtype='user', uid=username)
        user = self.get(key)
        if die: self._checktype(key, user, 'User')
        print('User "%s" loaded' % key)
        return user
        
    
    def savetask(self, task, key=None, uid=None, overwrite=False):
        '''
        Add a new or update existing Task in Redis, returns key.
        '''
        key, objtype, uid = self._makekey(key=key, objtype='task', uid=uid, obj=task, fulloutput=True)
        oldtask = self.get(key)
        if oldtask and not overwrite:
            errormsg = 'Task %s already exists' % key
            raise Exception(errormsg)
        else:
            self._checktype(key, task, 'Task')
            self.set(key=key, obj=task)
        print('Task "%s" saved' % key)
        return key
    
    
    def loadtask(self, key=None, uid=None, die=True):
        ''' Load a user from Redis '''
        key = self._makekey(key=key, objtype='task', uid=uid)
        task = self.get(key)
        if die: self._checktype(key, task, 'Task')
        print('Task "%s" loaded' % key)
        return task