"""
run_scirisapp.py -- Simple ScirisApp use case 
    
Last update: 5/1/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp
import model # The actual Python model we want to incorporate
import mpld3 # For plotting

# Run the webapp
if __name__ == "__main__":
    # Create the ScirisApp object.
    theApp = ScirisApp()
    
    appEndpointHandler = theApp.flaskApp.route
    
    @appEndpointHandler('/')
    def myRootPage():
        return '<h1>Hello, Flask!</h1>'
    
    @appEndpointHandler('/api')
    def myApiPage():
        return '<h1>Look at that!  Two Flask endpoints!</h1>' 
    
    @appEndpointHandler('/graph')
    def myGraphPage():
        graphFig = model.makegraph()
        graphHtml = mpld3.fig_to_html(graphFig)
        return '<h1>My graph</h1>' + graphHtml
        
    # Run the Flask server in the app.
    theApp.runServer()