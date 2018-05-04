"""
run_simpleclientapp.py -- Simple ScirisApp use case 
    
Last update: 5/4/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp

# Create the ScirisApp object.
app = ScirisApp(client_path='vueclient')
    
# Run the client page with a Twisted server.
app.run_server(with_client=True, with_flask=False)