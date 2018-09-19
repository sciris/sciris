"""
Blobs.py -- classes for Sciris objects which are generally managed
    
Last update: 2018sep19
"""

import six
import hashlib
import sciris as sc

__all__ = ['DSObject', 'Blob', 'User', 'Task']


class DSObject(sc.prettyobj):
    ''' A general Sciris object (base class for all such objects) '''
    
    def __init__(self, objtype=None, uid=None):
        # Handle input arguments
        if objtype is None: objtype = 'object'
        if uid     is None: uid     = sc.uuid()
        
        # Set attributes
        self.objtype  = objtype
        self.uid      = uid
        self.created  = sc.now()
        self.modified = [self.created]
        return None
    
    def update(self):
        ''' When the object is updated, append the current time to the modified list '''
        now = sc.now()
        self.modified.append(now)
        return now
        

class Blob(DSObject):
    ''' Wrapper for any Python object we want to store in the DataStore. '''
    
    def __init__(self, objtype=None, uid=None, data=None):
        ''' Create a new Blob, optionally saving data if provided '''
        DSObject.__init__(self, objtype=objtype, uid=uid)
        self.data = None
        if data is not None:
            self.save(data)
        return None
    
    def save(self, data):
        ''' Save new data to the Blob '''
        self.data = sc.dumpstr(data)
        self.update()
        return None
    
    def load(self):
        ''' Load data from the Blob '''
        output = sc.loadstr(self.data)
        return output


class User(DSObject):
    '''
    A Sciris user.
                    
    Attributes:
        is_authenticated (bool) -- is this user authenticated? (attribute required by Flask-Login)
        is_active (bool)        -- is this user's account active? (attribute required by Flask-Login)
        is_anonymous (bool)     -- is this user considered anonymous?  (attribute required by Flask-Login)
        is_admin (bool)         -- does this user have admin rights?
        username (str)          -- the username used to log in
        displayname (str)       -- the user's name, which gets displayed in the  browser
        email (str)             -- the user's email     
        password (str)          -- the user's SHA224-hashed password
    '''
    
    def  __init__(self, username=None, password=None, displayname=None, email=None, uid=None, is_admin=False):
        # Handle general properties
        if username    is None: username    = 'default'
        if password    is None: password    = 'default'
        if displayname is None: displayname = username
        
        # General initialization
        DSObject.__init__(self, objtype='user', uid=uid) # Set superclass parameters.
        
        # Set user-specific properties
        self.is_authenticated = True # Set the user to be authentic.
        self.is_active        = True # Set the account to be active.
        self.is_anonymous     = False # The user is not anonymous.
        self.is_admin         = is_admin # Set whether this user has admin rights.
        self.username         = username # Set the username.
        self.displayname      = displayname # Set the displayname (what the browser will show).
        self.email            = email  # Set the user's email.
        
        # Handle the password
        if six.PY3: raw_password = password.encode('utf-8')
        else:       raw_password = password # Set the raw password and use SHA224 to get the hashed version in hex form.
        self.password = hashlib.sha224(raw_password).hexdigest()
        return None
    
    def show(self, verbose=False):
        ''' Display the user's properties '''
        if not verbose: # Simply display the username and UID
            print('Username: %s; UID: %s' % (self.username, self.uid))
        else: # Or full information
            print('---------------------')
            print('        Username: %s' % self.username)
            print('    Display name: %s' % self.displayname)
            print(' Hashed password: %s' % self.password)
            print('   Email address: %s' % self.email)
            print('Is authenticated: %s' % self.is_authenticated)
            print('       Is active: %s' % self.is_active)
            print('    Is anonymous: %s' % self.is_anonymous)
            print('        Is admin: %s' % self.is_admin)
            print('---------------------')
        return None
    
    def jsonify(self, verbose=False):
        ''' Return a JSON-friendly representation of a user '''
        output = {'username':    self.username, 
                  'displayname': self.displayname, 
                  'email':       self.email,
                  'uid':         self.uid}
        if verbose:
            output.update({'is_authenticated': self.is_authenticated,
                           'is_active':        self.is_active,
                           'is_anonymous':     self.is_anonymous,
                           'is_admin':         self.is_admin,
                           'created':          self.created,
                           'modified':         self.modified[-1]})
        return output


class Task(DSObject):
    '''
    A Sciris record for an asynchronous task.
    
    Attributes:
        task_id (str)         -- the ID / name for the task that typically is chosen by the client
        status (str)          -- the status of the task:
                                    'unknown': unknown status, for example, just initialized
                                    'error':   task failed with an actual error
        error_text (str)      -- string giving an idea of what error has transpired
        func_name (str)       -- string of the function name for what's called
        args (list)           -- list containing args for the function
        kwargs (dict)         -- dict containing kwargs for the function
        result_id (str)       -- string for the Redis ID of the AsyncResult
        queue_time (datetime) -- the time the task was queued for Celery
        start_time (datetime) -- the time the task was actually started
        stop_time (datetime)  -- the time the task completed
        pending_time (int)    -- the time the process has been waiting to be executed on the server in seconds        
        execution_time (int)  -- the time the process required to complete
    '''
    
    def  __init__(self, task_id):
        # General initialization
        DSObject.__init__(self, objtype='task', uid=task_id) # Set superclass parameters.
        
        # Set other properties
        self.task_id        = task_id # Set the task ID (what the client typically knows as the task).
        self.status         = 'unknown' # Start the status at 'unknown'.
        self.error_text     = None # Start the error_text at None.
        self.func_name      = None # Start the func_name at None.
        self.args           = None # Start the args and kwargs at None.
        self.kwargs         = None
        self.result_id      = None # Start with no result_id.
        self.queue_time     = None # Start the queue, start, and stop times at None.
        self.start_time     = None
        self.stop_time      = None
        self.pending_time   = None # Start the pending and execution times at None.
        self.execution_time = None
        return None
    
    def show(self):
        print('-----------------------------')
        print('        Task ID: %s'   % self.task_id)
        print('         Status: %s'   % self.status)
        print('     Error text: %s'   % self.error_text)
        print('  Function name: %s'   % self.func_name)
        print('  Function args: %s'   % self.args)
        print('Function kwargs: %s'   % self.kwargs)        
        print('      Result ID: %s'   % self.result_id)
        print('     Queue time: %s'   % self.queue_time)        
        print('     Start time: %s'   % self.start_time)
        print('      Stop time: %s'   % self.stop_time)
        print('   Pending time: %s s' % self.pending_time)        
        print(' Execution time: %s s' % self.execution_time)  
        print('-----------------------------')
    
    def jsonify(self):
        output = {'UID':           self.uid,                    
                  'taskId':        self.task_id,
                  'status':        self.status,
                  'errorText':     self.error_text,
                  'funcName':      self.func_name,
                  'funcArgs':      self.args,
                  'funcKwargs':    self.kwargs,
                  'resultId':      self.result_id,
                  'queueTime':     self.queue_time,                
                  'startTime':     self.start_time,
                  'stopTime':      self.stop_time,
                  'pendingTime':   self.pending_time,
                  'executionTime': self.execution_time                
        }
        return output