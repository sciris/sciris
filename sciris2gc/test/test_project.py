"""
test_project.py -- test module for project.py

Usage:
python -m unittest sciris2gc.test.test_project
    
Last update: 5/22/18 (gchadder3)
"""

# Imports
import unittest
import sciris2gc.project as project
import uuid
import os

class ProjectTest(unittest.TestCase):
    def test_thang1(self):
        pass
    
class ProjectCollectionTest(unittest.TestCase):
    def test_thang1(self):
        pass 
       
#class FileioTest(unittest.TestCase):
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