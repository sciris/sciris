#!/usr/bin/env python

# Import needed libraries.
import os

# Get the currrent directory (should be a bin directory).
thisdir = os.getcwd()

# Move into the client directory.
os.chdir(os.pardir + os.sep + 'vueinterface')

# Install of the node modules for the client.
# After that, you can build and use the client.
os.system('npm install')

# Go back to this directory.
os.chdir(thisdir)