"""
run_simpleclientapp.py -- Simple ScirisApp use case 
    
Last update: 5/4/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp

# Create the ScirisApp object.
theApp = ScirisApp(clientPath='vueclient')
    
# Run the client page with a Twisted server.
theApp.runServer(withClient=True, withFlask=False)