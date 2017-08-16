# Ensure that the current folder is used, not the global defaults
import sys
import os
scirisfolder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, scirisfolder) 

# Load Optima
from sessionmanager.api import app

# Run the server
app.run()
#_autoreload.main(_twisted_wsgi.run)