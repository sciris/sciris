"""
run_rpcsclientapp.py -- Simple ScirisApp use case 
    
Last update: 6/14/18 (gchadder3)
"""

#
# Imports
#

from sciris.weblib.scirisapp import ScirisApp
from sciris.weblib.tasks import RPC_dict
from sciris.corelib import fileio
import model # The actual Python model we want to incorporate
import mpld3 # For plotting
import apptasks
import test_rpcs
import test_rpcs2
import pandas as pd
from pylab import figure
import config
import os   

# Create the ScirisApp object.  NOTE: app.config will thereafter contain all 
# of the configuration parameters, including for Flask.
#app = ScirisApp(app_config=config, client_dir='vueclient_rpcs')
app = ScirisApp(__file__, app_config=config)

# Register RPCs directly into the app.

@app.register_RPC(validation_type='admin user')
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
def show_csv_file(full_file_name):
#    x = 1 / 0  # uncomment to test exceptions with ZeroDivisionError
#    return {'error': 'show_csv_file() just does not feel like working.'}  # uncomment to test custom error

    # Extract the data from the .csv file.
    df = pd.read_csv(full_file_name)
    
    # Create the figure from the data.
    new_graph = figure()
    ax = new_graph.add_subplot(111)
    ax.scatter(df.x, df.y)
    
    # Convert the figure to a JSON-able dict.
    graph_dict = mpld3.fig_to_dict(new_graph)
    
    # Return the dict.
    return graph_dict

@app.register_RPC(call_type='download')
def download_graph_png():
#    x = 1 / 0  # uncomment to test exceptions with ZeroDivisionError
#    return {'error': 'download_graph_png() just does not feel like working.'}  # uncomment to test custom error
    
    # Make a new graph with random data.
    new_graph = model.makegraph()
    
    # Save a .png file of this graph.
    full_file_name = '%s%sgraph.png' % (fileio.downloads_dir.dir_path, os.sep)
    new_graph.savefig(full_file_name)
    
    # Return the full filename.
    return full_file_name

# Register the RPCs in the apptasks.py module.
app.add_RPC_dict(RPC_dict)

# Register the RPCs in the test_rpcs.py module.
app.add_RPC_dict(test_rpcs.RPC_dict)

# Register the RPCs in the test_rpcs2.py module.
app.add_RPC_dict(test_rpcs2.RPC_dict)

if __name__ == '__main__':
    # Run the client page with Flask and a Twisted server.
    app.run_server()