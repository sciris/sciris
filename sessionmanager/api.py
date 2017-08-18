"""
api.py -- script for setting up a flask server
    
Last update: 8/17/17 (gchadder3)
"""

# Do the imports.
# from pylab import figure, scatter, rand
import scirismodel.model as model
import mpld3
import json
from flask import Flask, request

# Create the app
app = Flask(__name__)

# Define the API
@app.route('/api', methods=['GET', 'POST'])
def showGraph():
    # Initialize the model code.
    model.init()
    
    # Create the matplotlib graph from Python.
    # At the moment, this passes in the 'value' setting to tell model which
    # graph (graph1, graph2, graph3) to load and display.
    graphData = model.makeGraph(request.get_json()['value'])
    
    # If we didn't find a match, return a null response.
    if graphData is None:
        return json.dumps({'error': 'NoSuchFile'})
    
    # Convert the figure to JSON.
    graphjson = json.dumps(mpld3.fig_to_dict(graphData)) 
    
    # Return the JSON representation of the Matplotlib figure.
    return graphjson 
    # Use this for the GET method.
    #return 'I will give you info for: %s ' % request.args['value']

    # Use this for the POST method.
    #return 'I will give you info for: %s ' % request.get_json()['value']
    

if __name__ == "__main__":
    app.run()