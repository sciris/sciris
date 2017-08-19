"""
api.py -- script for setting up a flask server
    
Last update: 8/19/17 (gchadder3)
"""

# Do the imports.
from flask import Flask, request, json, send_from_directory
import scirismodel.model as model
import mpld3
import os

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
    
# Define the /api/download endpoint.
@app.route('/api/download', methods=['POST'])
def downloadFile():
    # Initialize the model code.
    model.init()
    
    # Get the directory and file names for the file we want to download.
    dirName = '%s%sdatafiles' % (os.pardir, os.sep)
    fileName = '%s.csv' % request.get_json()['value']
    
    # Make the response message with the file loaded as an attachment.
    response = send_from_directory(dirName, fileName, as_attachment=True)
    response.status_code = 201
    response.headers['filename'] = fileName
    
    # Return the response message.
    return response

    
if __name__ == "__main__":
    app.run()