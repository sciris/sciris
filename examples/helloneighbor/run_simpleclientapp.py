"""
run_simpleclientapp.py -- Simple ScirisApp use case 
    
Last update: 6/8/18 (gchadder3)
"""

# Imports
from sciris.weblib.scirisapp import ScirisApp

# Create the ScirisApp object.  NOTE: app.config will thereafter contain all 
# of the configuration parameters, including for Flask.
app = ScirisApp(__file__, client_dir='vueclient')
    
# Run the client page with a Twisted server.
app.run_server(with_client=True, with_flask=False)