"""
Blobs.py -- classes for Sciris objects which are generally managed
    
Last update: 2018sep02
"""

import sciris as sc

__all__ = ['DSObject', 'Blob', 'User', 'Task']

class DSObject(sc.prettyobj):
    ''' A general Sciris object (base class for all such objects). '''
    
    def __init__(self, objtype=None, uid=None):
        # Handle input arguments
        if objtype is None: objtype = 'object'
        if uid     is None: uid     = str(sc.uuid())
        
        # Set attributes
        self.objtype  = objtype
        self.uid      = uid
        self.created  = str(sc.now())
        self.modified = [self.created]
        return None
    
    def update(self):
        ''' When the object is updated, append the current time to the modified list '''
        timestr = str(sc.now())
        self.modified.append(timestr)
        return timestr
        


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
    ''' Wrapper for a User '''
    
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
