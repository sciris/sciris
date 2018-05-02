#!/usr/bin/env python

'''
Small script to reset the user database -- WARNING, deletes everything!!!
Version: 2018mar17
'''

# Import needed libraries.
import os

# Load the Sciris datastore code.
import sciris.datastore as ds

# Load the config file.
import imp
config = imp.load_source('config', os.pardir + os.sep + 'webapp' + os.sep + 'config.py')

answer = raw_input('Are you sure you want to reset the database for "%s"? (y/[n]): ' % (config.ROOT_DIR))
if answer == 'y':
	ds.theDataStore = ds.DataStore(redisDbURL=config.REDIS_URL)
	ds.theDataStore.deleteAll()
	print('Database reset.')
else:
    print('Database not reset.')