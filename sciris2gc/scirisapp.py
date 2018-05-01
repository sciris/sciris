"""
scirisapp.py -- classes for Sciris (Flask-based) apps 
    
Last update: 5/1/18 (gchadder3)
"""

# Imports
from flask import Flask

#
# Classes
#

class ScirisApp(object):
    """
    An object encapsulating a Sciris webapp, generally.  This app has an 
    associated Flask app that actually runs to listen for and respond to 
    HTTP requests.
    
    Methods:
        __init__(): void -- constructor
        runServer(): void -- run the actual server
                    
    Attributes:
        flaskApp (Flask) -- the actual Flask app
            
    Usage:
        >>> theApp = ScirisApp()                      
    """
    
    def  __init__(self):
        # Open a new Flask app.
        self.flaskApp = Flask(__name__)
        
    def runServer(self):       
        # Run the Flask app.
        self.flaskApp.run()
        
        