"""
run_rpcsclientapp.py -- Simple ScirisApp use case 
    
Last update: 5/8/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp
import model # The actual Python model we want to incorporate
import mpld3 # For plotting
import json # For converting to HTML

# Create the ScirisApp object.
app = ScirisApp(client_path='vueclient_rpcs')

# Create a callback rendering function at /.
@app.define_endpoint_callback('/')
def my_root_page():
    return '<h1>Hello, Flask!</h1>'



# stuff to take out...
    
from sciris2gc.scirisapp import json_sanitize_result
 
## Create a callback rendering function at /.
#@app.define_endpoint_callback('/rpcs', methods=['POST'])
#def rpc():
#    new_graph = model.makegraph()
#    graph_dict = mpld3.fig_to_dict(new_graph)
#    graph_json = json.dumps(json_sanitize_result(graph_dict)) # Convert to JSON
#    return graph_json # Return the JSON representation of the Matplotlib figure

@app.register_RPC('/rpcs', type='normal')
def rpc():
    new_graph = model.makegraph()
    graph_dict = mpld3.fig_to_dict(new_graph)
    graph_json = json.dumps(json_sanitize_result(graph_dict)) # Convert to JSON
    return graph_json # Return the JSON representation of the Matplotlib figure


# Run the client page with Flask and a Twisted server.
app.run_server()