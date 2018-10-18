#!/usr/bin/env python

# Import needed libraries.
import os

# Get the currrent directory (should be a bin directory).
thisdir = os.getcwd()

# Move into the client directory.
os.chdir(os.pardir + os.sep + 'vueinterface')

# Run an npm build.
os.system('npm run build')

# Go back to this directory.
os.chdir(thisdir)