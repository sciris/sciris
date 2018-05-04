"""
run_scirisapp.py -- Simple ScirisApp use case 
    
Last update: 5/4/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp
import model # The actual Python model we want to incorporate
import mpld3 # For plotting

# Run the webapp
if __name__ == "__main__":
    # Create the ScirisApp object.
    app = ScirisApp()
    
    appEndpointHandler = app.flask_app.route
    
    @appEndpointHandler('/')
    def my_root_page():
        return '<h1>Hello, Flask!</h1>'
    
    @appEndpointHandler('/api')
    def my_api_page():
        return '<h1>Look at that!  Two Flask endpoints!</h1>' 
    
    @appEndpointHandler('/graph')
    def my_graph_page():
        graph_fig = model.makegraph()
        graph_html = mpld3.fig_to_html(graph_fig)
        return '<h1>My graph</h1>' + graph_html
        
    # Run the Flask server in the app.
    app.run_server()  # Twisted + client + server    
#    app.run_server(with_twisted=False)  # Flask app (only) without Twisted  
#    app.run_server(with_client=True, with_flask=False)  # client only
#    app.run_server(with_client=False, with_flask=True)  # Flask only with Twisted
