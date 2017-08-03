"""
api.py -- script for setting up a flask server
    
Last update: 8/2/17 (gchadder3)
"""

from flask import Flask, request
app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def showScatterplot():
    return 'I will give you info for: %s ' % request.get_json()['value']

if __name__ == "__main__":
    app.run()