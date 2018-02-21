"""
run_client.py -- script for running the client pointed to by config.py
    
Last update: 2/20/18 (gchadder3)
"""

# Import necessary packages.
import os

# Run the config file.
execfile('../sessionmanager/config.py')

# If we have a full path for the client directory, use that directory.
if os.path.isabs(CLIENT_DIR):
    clientDirTarget = CLIENT_DIR
    
# Otherwise (we have a relative path), use it (correcting so it is with 
# respect to the sciris repo directory).
else:
    clientDirTarget = '%s%s%s' % (os.pardir, os.sep, CLIENT_DIR) 

# Navigate to the client directory.
os.chdir(clientDirTarget)

# Run the npm build.
os.system('npm run dev')