"""
Simple demonstration Flask webapp.
"""

# Imports
import model # The actual Python model we want to incorporate
import mpld3 # For plotting
import json # For converting to HTML
import numpy as np
from flask import Flask, request # The webapp

def json_sanitize_result(theResult):
    """
    This is the main conversion function for Python data-structures into
    JSON-compatible data structures.
    Use this as much as possible to guard against data corruption!
    Args:
        theResult: almost any kind of data structure that is a combination
            of list, numpy.ndarray, etc.
    Returns:
        A converted dict/list/value that should be JSON compatible
    """

    if isinstance(theResult, list) or isinstance(theResult, tuple):
        return [json_sanitize_result(p) for p in list(theResult)]
    
    if isinstance(theResult, np.ndarray):
        if theResult.shape: # Handle most cases, incluing e.g. array([5])
            return [json_sanitize_result(p) for p in list(theResult)]
        else: # Handle the special case of e.g. array(5)
            return [json_sanitize_result(p) for p in list(np.array([theResult]))]

    if isinstance(theResult, dict):
        return {str(k): json_sanitize_result(v) for k, v in theResult.items()}

    if isinstance(theResult, np.bool_):
        return bool(theResult)

    if isinstance(theResult, float):
        if np.isnan(theResult):
            return None

    if isinstance(theResult, np.float64):
        if np.isnan(theResult):
            return None
        else:
            return float(theResult)

    if isinstance(theResult, unicode):
        return theResult
#        return str(theResult)  # original line  (watch to make sure the 
#                                                 new line doesn't break things)
    
    if isinstance(theResult, set):
        return list(theResult)

#    if isinstance(theResult, UUID):
#        return str(theResult)

    return theResult

# Create the app
app = Flask(__name__) 

# Define the API
@app.route('/api', methods=['GET', 'POST'])
def showGraph():
    graphdata = model.makegraph() # Create the graph from Python
    graphdict = mpld3.fig_to_dict(graphdata)  # Convert to dict
    graphjson = json.dumps(json_sanitize_result(graphdict)) # Convert to JSON
    return graphjson # Return the JSON representation of the Matplotlib figure

# Run the webapp
if __name__ == "__main__":
    app.run()