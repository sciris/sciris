"""
run_rpcsclientapp.py -- Simple ScirisApp use case 
    
Last update: 5/16/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp
import model # The actual Python model we want to incorporate
import mpld3 # For plotting
import test_rpcs
import pandas as pd
from pylab import figure
import config

# Create the ScirisApp object.  NOTE: app.config will thereafter contain all 
# of the configuration parameters, including for Flask.
app = ScirisApp(app_config=config, client_dir='vueclient_rpcs')

# Register RPCs directly into the app.

@app.register_RPC()
def get_graph():
#    x = 1 / 0  # uncomment to test exceptions with ZeroDivisionError
#    return {'error': 'get_graph() just does not feel like working.'}  # uncomment to test custom error
    new_graph = model.makegraph()
    graph_dict = mpld3.fig_to_dict(new_graph)
    return graph_dict

@app.register_RPC()
def test_func():
    return '<h1>Test Me!</h1>'

@app.register_RPC(call_type='upload')
def show_csv_file():
    # Extract the data from the .csv file.
    df = pd.read_csv('./graph1.csv')
    
    # Create the figure from the data.
    new_graph = figure()
    ax = new_graph.add_subplot(111)
    ax.scatter(df.x, df.y)
    
    # Convert the figure to a JSON-able dict.
    graph_dict = mpld3.fig_to_dict(new_graph)
    
    # Return the dict.
    return graph_dict

# Register the RPCs in the test_rpcs.py module.
app.add_RPC_dict(test_rpcs.RPC_dict)

# Run the client page with Flask and a Twisted server.
app.run_server()