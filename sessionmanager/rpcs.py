"""
rpcs.py -- code for RPC interfaces between client and server
    
Last update: 8/25/17 (gchadder3)
"""

#
# Imports
#

import scirismodel.model as model
import mpld3

#
# Functions
#

def get_graph_from_file(fileStem):
    # Create the matplotlib graph from Python.
    # At the moment, this passes in the 'fileStem' setting to tell model which
    # graph (graph1, graph2, graph3) to load and display.
    graphData = model.plotDataFromFile(fileStem)

    # If we didn't find a match, return a null response.
    if graphData is None:
        return {'error': 'NoSuchFile'}
    
    # Return the dictionary representation of the matplotlib figure.
    return mpld3.fig_to_dict(graphData) 