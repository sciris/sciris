#!/usr/bin/env python

"""
TESTALL

Run all tests. It runs everything in the same namespace, but deletes variables that get
added along the way.

Version: 2020apr27
"""

## Initialization
from time import time as TIME # Use caps to distinguish 'global' variables
from sys import exc_info as EXC_INFO
from glob import glob as GLOB
import os as OS

doplot = False # Don't run plotting in batch


## Run the tests in a loop
KEY = None
ORIGINALVARIABLES = []
CURRENTVARIABLES = []
VERYSTART = TIME()
FAILED = []
SUCCEEDED = []
MASTER = GLOB(OS.path.dirname(OS.path.realpath(__file__))+OS.sep+'test_*.py') # Figure out the path -- adapted from defaults.py
for TEST in MASTER:
    try:
        THISSTART = TIME()
        ORIGINALVARIABLES = list(locals().keys()) # Get the state before the test is run
        print('\n'*10+'•'*100)
        print('NOW RUNNING: %s' % TEST)
        print('•'*100+'\n'*3)
        exec(open(TEST).read()) # Run the test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUCCEEDED.append({'test':TEST, 'time':TIME()-THISSTART})
        CURRENTVARIABLES = list(locals().keys())
        for KEY in CURRENTVARIABLES: # Clean up -- delete any new variables added
            if KEY not in ORIGINALVARIABLES:
                print('       "%s" complete; deleting "%s"' % (TEST, KEY))
                exec('del '+KEY)
    except:
        FAILED.append({'test':TEST, 'msg':EXC_INFO()[1]})


print('\n'*5)
if len(FAILED):
    print('The following %i/%i tests failed :(' % (len(FAILED), len(MASTER)))
    for FAIL in FAILED: print('  %s: %s' % (FAIL['test'], FAIL['msg']))
else:
    print('All %i tests passed!!!' % len(MASTER))
    for SUCCESS in SUCCEEDED: print('  %s: %0.1f s' % (SUCCESS['test'], SUCCESS['time']))
print('Elapsed time: %0.1f s.' % (TIME()-VERYSTART))