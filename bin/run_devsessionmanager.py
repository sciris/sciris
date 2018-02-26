# Ensure that the current folder is used, not the global defaults
import sys
import os

# Load Optima
from sciris.api import app

# Run the server
app.run()
#_autoreload.main(_twisted_wsgi.run)