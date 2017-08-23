"""
api.py -- script for setting up a flask server
    
Last update: 8/22/17 (gchadder3)
"""

# Do the imports.
from flask import Flask, request, json, send_from_directory
from werkzeug.utils import secure_filename
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
    # dirName = '%s%sdatafiles' % (os.pardir, os.sep)
    dirName = model.datafilesPath   
    fileName = '%s.csv' % request.get_json()['value']
    
    # Make the response message with the file loaded as an attachment.
    response = send_from_directory(dirName, fileName, as_attachment=True)
    response.status_code = 201  # Status 201 = Created
    response.headers['filename'] = fileName
    
    # Return the response message.
    return response

# Define the /api/upload endpoint.
@app.route('/api/upload', methods=['POST'])
def uploadFile():
    # Initialize the model code.
    model.init()
      
    # Grab the formData file that was uploaded.    
    file = request.files['uploadfile']
    
    # Grab the filename of this file, and generate the full upload path / 
    # filename.
    filename = secure_filename(file.filename)
    uploaded_fname = os.path.join(model.uploadsPath, filename)
    
    # Save the file to the uploads directory.
    file.save(uploaded_fname)
    
    # Return success.
    return json.dumps({'result': 'success'})  
 

if __name__ == "__main__":
    app.run()