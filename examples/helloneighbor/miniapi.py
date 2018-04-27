"""
Simple demonstration Flask webapp.
"""

# Imports
from flask import Flask

# Create the app
app = Flask(__name__) 

# Define the API
@app.route('/api', methods=['GET', 'POST'])
def myPage():
    return '<h1>Hello, Flask!</h1>'

# Run the webapp
if __name__ == "__main__":
    app.run()