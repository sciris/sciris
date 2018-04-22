#!/usr/bin/env python

# Import needed libraries.
import os

# Load Sciris.
from sciris import server

# Load the config file.
import imp
config = imp.load_source('config', os.pardir + os.sep + 'webapp' + os.sep + 'config.py')

# Run the server.
server.start(config=config)