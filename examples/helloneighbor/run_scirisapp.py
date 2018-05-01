"""
Simple ScirisApp use case
"""

# Imports
from sciris2gc.scirisapp import ScirisApp

# Run the webapp
if __name__ == "__main__":
    # Create the ScirisApp object.
    theApp = ScirisApp()
    
    appEndpointHandler = theApp.flaskApp.route
    
    @appEndpointHandler('/')
    def myPage():
        return '<h1>Hello, Flask!</h1>'
    
    @appEndpointHandler('/api')
    def myPage2():
        return '<h1>Look at that!  Two flask endpoints!</h1>' 
        
    # Run the Flask server in the app.
    theApp.runServer()