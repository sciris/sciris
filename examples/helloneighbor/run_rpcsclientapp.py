"""
run_rpcsclientapp.py -- Simple ScirisApp use case 
    
Last update: 5/12/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp
import model # The actual Python model we want to incorporate
import mpld3 # For plotting
import test_rpcs

# Create the ScirisApp object.
app = ScirisApp(client_path='vueclient_rpcs')

# Register RPCs directly into the app.

@app.register_RPC()
def get_graph():
    new_graph = model.makegraph()
    graph_dict = mpld3.fig_to_dict(new_graph)
    return graph_dict

@app.register_RPC()
def test_func():
    return '<h1>Test Me!</h1>'

# Register the RPCs in the test_rpcs.py module.
app.add_RPC_dict(test_rpcs.RPC_dict)

# Run the client page with Flask and a Twisted server.
app.run_server()