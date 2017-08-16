"""
api.py -- script for setting up a flask server
    
Last update: 8/16/17 (gchadder3)
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
    # Create the matplotlib graph from Python.
    graphdata = model.makegraph()
    
    # Convert the figure to JSON.
    graphjson = json.dumps(mpld3.fig_to_dict(graphdata)) 
    
    # Return the JSON representation of the Matplotlib figure.
    return graphjson 
    # Use this for the GET method.
    #return 'I will give you info for: %s ' % request.args['value']

    # Use this for the POST method.
    #return 'I will give you info for: %s ' % request.get_json()['value']
    

if __name__ == "__main__":
    app.run()