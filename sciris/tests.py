def testfiles(path=None, filelist=None):

    """
    TESTFILES
    
    Run a number of files in series. Usage:
        from sciris.tests import testfiles
        testfiles(path='my-tests-folder', filelist = ['test-file1', 'test-file2']) # Run test-file1.py and test-file2.py
        testfiles() # Runs all .py files in the current folder
    
    It runs everything in the same namespace, but deletes variables that get
    added along the way, so each test runs from a clean slate.
    
    Version: 2018mar27 by cliffk
    """
    
    ## Initialization
    from time import time as tf_TIME # Use caps to distinguish 'global' variables
    from sys import exc_info as tf_EXC_INFO
    from glob import glob as tf_GLOB
    import os as tf_OS
    
    ## Define the path
    tf_PATH = path; del path # Rename variable to avoid naming collisions
    tf_FILELIST = filelist; del filelist
    if tf_PATH is None: 
        tf_PATH = tf_OS.getcwd()
    tf_PATH = tf_OS.path.realpath(tf_PATH)
    if tf_OS.path.isfile(tf_PATH):
        tf_PATH = tf_OS.path.dirname(tf_PATH)
    
    ## Optionally run everything
    if tf_FILELIST is not None: # If using the tf_FILELIST list supplied, just need to append .py ending
        for tf_INDEX in range(len(tf_FILELIST)):
            tf_FILELIST[tf_INDEX] = tf_PATH + tf_OS.sep + tf_FILELIST[tf_INDEX] # Prepend path
            try:    assert tf_FILELIST[tf_INDEX][-3:] == '.py' # Could fail if wrong ending, or if string isn't that long
            except: tf_FILELIST[tf_INDEX] += '.py' 
    else: # We're using everything
        tf_FILELIST = tf_GLOB(tf_PATH + tf_OS.sep + '*.py') # Figure out the path -- adapted from defaults.py
    try:    tf_FILELIST.remove(tf_OS.path.realpath(__file__)) # Simple solution if file is in list
    except: pass
    
    ## Run the tests in a loop
    tf_VARIABLES = []
    tf_VERYSTART = tf_TIME()
    tf_FAILED = []
    tf_SUCCEEDED = []
    for tf_TEST in tf_FILELIST:
        try:
            tf_THISSTART = tf_TIME()
            tf_VARIABLES = locals().keys() # Get the state before the test is run
            print('\n'*10+'#'*200)
            print('NOW RUNNING: %s' % tf_TEST)
            print('#'*200+'\n'*3)
            exec(open(tf_TEST).read()) # Run the test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            tf_SUCCEEDED.append({'test':tf_TEST, 'time':tf_TIME()-tf_THISSTART})
            for KEY in locals().keys(): # Clean up -- delete any new variables added
                if KEY not in tf_VARIABLES:
                    print('       "%s" complete; deleting "%s"' % (tf_TEST, KEY))
                    exec('del '+KEY)
        except:
            tf_FAILED.append({'test':tf_TEST, 'msg':tf_EXC_INFO()})
    
    
    print('\n'*5)
    if len(tf_FAILED):
        print('The following %i/%i tests failed :(' % (len(tf_FAILED), len(tf_FILELIST)))
        for tf_FAIL in tf_FAILED: print('\n'*2 + '  %s:\n%s' % (tf_FAIL['test'], tf_FAIL['msg']))
    else:
        print('All %i tests passed! :)' % len(tf_FILELIST))
        for tf_SUCCESS in tf_SUCCEEDED: print('  %s: %0.1f s' % (tf_SUCCESS['test'], tf_SUCCESS['time']))
    print('Elapsed time: %0.1f s.' % (tf_TIME()-tf_VERYSTART))
    
    return None