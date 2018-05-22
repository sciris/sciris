"""
test_datastore.py -- test module for datastore.py

Usage:
python -m unittest sciris2gc.test.test_datastore
    
Last update: 5/22/18 (gchadder3)
"""

# Imports
import unittest
import sciris2gc.datastore as ds
import os
import redis

class StoreObjectHandleTest(unittest.TestCase):
    def setUp(self):
        # Make a test directory.
        if not os.path.exists('testdatastoredir'):
            os.mkdir('testdatastoredir')
            
    def tearDown(self):
        # Remove the test directory.
        os.rmdir('testdatastoredir')
            
    def test_handle_lifecycle(self):
        # Create a test dict.
        test_dict = { 'optima': 'developer', 'sciris': 'product' }
        
        # Create a new handle which we'll use to store the object.
        new_handle = ds.StoreObjectHandle(type_prefix='testobj')
        
        # Make sure the stored uid is returned by the get_uid function.
        self.assertEqual(new_handle.get_uid(), new_handle.uid)
        
        # Show the new handle.
        print('The new handle:')
        new_handle.show()
        
        # File storage tests.
        
        # Store the object in a file.
        new_handle.file_store('testdatastoredir', test_dict)
        
        # Make sure the file exists and it has the right name.
        correct_file_name = 'testdatastoredir' + os.sep + \
            new_handle.type_prefix + '-' + new_handle.uid.hex + \
            new_handle.file_suffix
        self.assertTrue(os.path.exists(correct_file_name))
        
        # Retrieve the test_dict object and make sure it matches.
        retrieved_test_dict = new_handle.file_retrieve('testdatastoredir')
        self.assertEqual(retrieved_test_dict, test_dict, 
            'Retrieved test object does not match')
        
        # Delete the handle and make sure it's gone.
        new_handle.file_delete('testdatastoredir')
        self.assertFalse(os.path.exists(correct_file_name), 'File not deleted') 
        
        # Redis storage tests.   
        
        # Go to the default Redis database (#0).
        redis_db_URL = 'redis://localhost:6379/0/'
        redis_db = redis.StrictRedis.from_url(redis_db_URL)
        
        # Store the test_dict in the Redis database.
        new_handle.redis_store(test_dict, redis_db)
        
        # Make sure the key exists and retrieval gives a match to test_dict.
        correct_key_name = new_handle.type_prefix + '-' + new_handle.uid.hex
        self.assertEqual(new_handle.redis_retrieve(redis_db), 
            test_dict, 'Retrieved test object does not match')
        
        # Delete the Redis entry and make sure it's gone.
        new_handle.redis_delete(redis_db)
        self.assertEqual(redis_db.get(correct_key_name), None, 'Key not deleted')
        
class DataStoreTest(unittest.TestCase):
    def test_thang1(self):
        pass
    
#    def test_FileSaveDirectory(self):
#        # file_save_dir tests
#        
#        # Create a persistant file_save_dir directory.
#        fileio.file_save_dir = fileio.FileSaveDirectory('testsavedfiles')
#        
#        # Check to make sure the directory is really there.
#        self.assertTrue(os.path.exists('testsavedfiles'), 'Directory was not created')
#        
#        # Do a cleanup of 'testsavedfiles'
#        fileio.file_save_dir.cleanup()
#        
#        # Check to make sure the directory is still there.  (It should be.)
#        self.assertTrue(os.path.exists('testsavedfiles'), 'Directory is missing')
#        
#        # Get rid of the file_save_dir.
#        fileio.file_save_dir.delete()
#        
#        # Make sure the directory is gone.
#        self.assertFalse(os.path.exists('testsavedfiles'), 'Directory was not deleted')        
#        
#        # file_uploads_dir tests
#        
#        # Create a temporary uploads_dir directory.
#        fileio.uploads_dir = fileio.FileSaveDirectory('testuploads', temp_dir=True)
#        
#        # Check to make sure the directory is really there.
#        self.assertTrue(os.path.exists('testuploads'), 'Directory was not created')        
#        
#        # Do a cleanup of 'testuploads'
#        fileio.uploads_dir.cleanup()
#        
#        # Check to make sure the directory is gone.
#        self.assertFalse(os.path.exists('testuploads'), 'Directory was not deleted')        
#        
#        # file_downloads_dir tests        
#        
#        # Create a temporary downloads_dir directory at a temp directory location.
#        fileio.downloads_dir = fileio.FileSaveDirectory(temp_dir=True)  
#        
#        # Check to make sure the directory is really there.
#        self.assertTrue(os.path.exists(fileio.downloads_dir.dir_path), 'Directory was not created')         
#
#        # Do a cleanup of download directory.
#        fileio.downloads_dir.cleanup()
#        
#        # Check to make sure the directory is gone.
#        self.assertFalse(os.path.exists(fileio.downloads_dir.dir_path), 'Directory was not deleted')
#        
#    def test_pickle_functions(self):
#        # Create a test dict.
#        test_dict = { 'optima': 'developer', 'sciris': 'product' }
#        
#        # Pickle it into a string.
#        str_pickle = fileio.object_to_string_pickle(test_dict)
#        
#        # Unpickle the string into an object.
#        unpickle_obj = fileio.string_pickle_to_object(str_pickle)
#        
#        # Check to make sure the objects are the same.
#        self.assertEqual(test_dict, unpickle_obj, 'Pickle to str and back failed to match')
#
#
#        # Pickle test_dict into a gzip string pickle file.
#        fileio.object_to_gzip_string_pickle_file('testgzippickle.obj', test_dict)
#        
#        # Unpickle the gzip string pickle file into an object.
#        unpickle_obj = fileio.gzip_string_pickle_file_to_object('testgzippickle.obj')
#        
#        # Check to make sure the objects are the same.
#        self.assertEqual(test_dict, unpickle_obj, 'Pickle to gzip string file and back failed to match')
#        
#        # Remove the file.
#        os.remove('testgzippickle.obj')
#        
#
#        # Pickle test_dict into a gzip string pickle object.
#        gzip_str_pickle = fileio.object_to_gzip_string_pickle(test_dict)
#        
#        # Unpickle the gzip string pickle object into an object.
#        unpickle_obj = fileio.gzip_string_pickle_to_object(gzip_str_pickle)
#        
#        # Check to make sure the objects are the same.
#        self.assertEqual(test_dict, unpickle_obj, 'Pickle to str and back failed to match')
        
        
if __name__ == '__main__':
    unittest.main()  # simplest run
    
    # More verbose run.
#    suite = unittest.TestLoader().loadTestsFromTestCase(FileioTest)
#    unittest.TextTestRunner(verbosity=2).run(suite)