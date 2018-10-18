#!/usr/bin/env python

# Import needed libraries.
import os

# Load Sciris.
from sciris.api import makeapp

# Load the config file.
import imp
config = imp.load_source('config', os.pardir + os.sep + 'webapp' + os.sep + 'config.py')

# Create the app.
app = makeapp(config=config)

# Run the server.
app.run()