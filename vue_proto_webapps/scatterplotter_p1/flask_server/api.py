"""
api.py -- script for setting up a flask server
    
Last update: 8/3/17 (gchadder3)
"""

import matplotlib.pyplot as plt
import mpld3
import json

from flask import Flask, request
app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def showScatterplot():
#    myFig = plt.figure()
#    plt.plot([3,1,4,1,5], 'ks-', mec='w', mew=5, ms=20)
#    json01 = json.dumps(mpld3.fig_to_dict(myFig))
#    return json01
    # Use this for the GET method.
    return 'I will give you info for: %s ' % request.args['value']

    # Use this for the POST method.
    #return 'I will give you info for: %s ' % request.get_json()['value']

if __name__ == "__main__":
    app.run()