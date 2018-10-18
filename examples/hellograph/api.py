"""
Simple demonstration Sciris webapp.
"""

# Imports
import model # The actual Python model we want to incorporate
import mpld3 # For plotting
import json # For converting to HTML
from flask import Flask  # The webapp

# Create the app
app = Flask(__name__) 

# Define the API
@app.route('/api', methods=['GET', 'POST'])
def showGraph():
    graphdata = model.makegraph() # Create the graph from Python
    graphdict = mpld3.fig_to_dict(graphdata)  # Convert to dict
    graphjson = json.dumps(graphdict) # Convert to JSON
    return graphjson  # Return the JSON representation of the Matplotlib figure

# Run the webapp
if __name__ == "__main__":
    app.run()