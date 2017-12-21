"""
build_client.py -- script for building the client pointed to by config.py
    
Last update: 12/21/17 (gchadder3)
"""

# Import necessary packages.
import os

# Run the config file.
execfile('../sessionmanager/config.py')

# Navigate to the client directory.
os.chdir('%s%s%s' % (os.pardir, os.sep, CLIENT_DIR))

# Run the npm build.
os.system('npm run build')