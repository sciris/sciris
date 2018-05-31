"""
fileio.py -- code for file management in Sciris 
    
Last update: 5/18/18 (gchadder3)
"""

#
# Imports
#

import os
from tempfile import mkdtemp
from shutil import rmtree
import atexit

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
# TODO: I'm not sure if we're going to want to keep and use these globals, or 
# whether we really will want to put this stuff in ScirisApp attributes.

# Directory (FileSaveDirectory object) for saved files.
file_save_dir = None

# Directory (FileSaveDirectory object) for file uploads to be routed to.
uploads_dir = None

# Directory (FileSaveDirectory object) for file downloads to be routed to.
downloads_dir = None

#
# Classes
#
    
class FileSaveDirectory(object):
    """
    An object wrapping a directory where files may get saved by the web 
    application.
    
    Methods:
        __init__(dir_path: str [None], temp_dir: bool [False]): void -- 
            constructor
        cleanup(): void -- clean up after web app is exited
        clear(): void -- erase the contents of the directory
        delete(): void -- delete the entire directory
                    
    Attributes:
        dir_path (str) -- the full path of the directory on disk
        is_temp_dir (bool) -- is the directory to be spawned on startup and 
            erased on exit?
        
    Usage:
        >>> new_dir = FileSaveDirectory(transfer_dir_path, temp_dir=True)
    """
    
    def __init__(self, dir_path=None, temp_dir=False):
        # Set whether we are a temp directory.
        self.is_temp_dir = temp_dir
               
        # If no path is specified, create the temp directory.
        if dir_path is None:
            self.dir_path = mkdtemp()
            
        # Otherwise...
        else:
            # Set the path to what was passed in.
            self.dir_path = dir_path
            
            # If the directory doesn't exist yet, create it.
            if not os.path.exists(dir_path):            
                os.mkdir(dir_path)
            
        # Register the cleanup method to be called on web app exit.
        atexit.register(self.cleanup)
            
    def cleanup(self):
        # If we are a temp directory and the directory still exists, do the cleanup.
        if self.is_temp_dir and os.path.exists(self.dir_path):
            # Show cleanup message.
            print '>> Cleaning up FileSaveDirectory at %s' % self.dir_path
            
            # Delete the entire directory (file contents included).
            self.delete()
            
    def clear(self):
        # Delete the entire directory (file contents included).
        rmtree(self.dir_path)
        
        # Create a fresh direcgtory
        os.mkdir(self.dir_path)
    
    def delete(self):
        # Delete the entire directory (file contents included).
        rmtree(self.dir_path)
        

#
# Pickle / unpickle functions
#

def object_to_string_pickle(obj):
    return pickle.dumps(obj, protocol=-1)

def string_pickle_to_object(string_pickle):
    return pickle.loads(string_pickle)

def object_to_gzip_string_pickle_file(full_file_name, obj, compresslevel=5):
    # Object a Gzip file object to write to and set the compression level 
    # (which the function defaults to 5, since higher is much slower, but 
    # not much more compact).
    with GzipFile(full_file_name, 'wb', compresslevel=compresslevel) as fileobj:
        # Write the string pickle conversion of the object to the file.
        fileobj.write(object_to_string_pickle(obj))

def gzip_string_pickle_file_to_object(full_file_name):
    # Object a Gzip file object to read from.
    with GzipFile(full_file_name, 'rb') as fileobj:
        # Read the string pickle from the file.
        string_pickle = fileobj.read()
        
    # Return the object gotten from the string pickle.    
    return string_pickle_to_object(string_pickle)

def object_to_gzip_string_pickle(obj):
    # Start with a null result.
    result = None
    
    # Open a "fake file."
    with closing(StringIO()) as output:
        # Open a Gzip-compressing way to write to this "file."
        with GzipFile(fileobj=output, mode='wb') as fileobj: 
            # Write the string pickle conversion of the object to the "file."
            fileobj.write(object_to_string_pickle(obj))
            
        # Move the mark to the beginning of the "file."
        output.seek(0)
        
        # Read all of the content into result.
        result = output.read()
        
    # Return the read-in result.
    return result

def gzip_string_pickle_to_object(gzip_string_pickle):
    # Open a "fake file" with the Gzip string pickle in it.
    with closing(StringIO(gzip_string_pickle)) as output:
        # Set a Gzip reader to pull from the "file."
        with GzipFile(fileobj=output, mode='rb') as fileobj: 
            # Read the string pickle from the "file" (applying Gzip 
            # decompression).
            string_pickle = fileobj.read()  
            
            # Extract the object from the string pickle.
            obj = string_pickle_to_object(string_pickle)
            
    # Return the object.
    return obj