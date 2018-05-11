"""
run_rpcsclientapp.py -- Simple ScirisApp use case 
    
Last update: 5/11/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp, json_sanitize_result
import model # The actual Python model we want to incorporate
import mpld3 # For plotting
import json # For converting to HTML
import test_rpcs

# Create the ScirisApp object.
app = ScirisApp(client_path='vueclient_rpcs')

# Register RPCs directly into the app.

@app.register_RPC()
def get_graph():
    new_graph = model.makegraph()
    graph_dict = mpld3.fig_to_dict(new_graph)
    graph_json = json.dumps(json_sanitize_result(graph_dict)) # Convert to JSON
    return graph_json # Return the JSON representation of the Matplotlib figure

@app.register_RPC()
def test_func():
    return '<h1>Test Me!</h1>'

# Register the RPCs in the test_rpcs.py module.
app.add_RPC_dict(test_rpcs.RPC_dict)

# Run the client page with Flask and a Twisted server.
app.run_server()