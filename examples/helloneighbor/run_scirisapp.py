"""
run_scirisapp.py -- Simple ScirisApp use case 
    
Last update: 5/18/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp
import model # The actual Python model we want to incorporate
import mpld3 # For plotting

# Create the ScirisApp object.  NOTE: app.config will thereafter contain all 
# of the configuration parameters, including for Flask.
app = ScirisApp(__file__)

# Create a callback rendering function at /.
@app.define_endpoint_callback('/')
def my_root_page():
    return '<h1>Hello, Flask!</h1>'

# Create a callback rendering function at /api.    
@app.define_endpoint_callback('/api')
def my_api_page():
    return '<h1>Look at that!  Two Flask endpoints!</h1>' 

# Create a callback rendering function at /graph.   
@app.define_endpoint_callback('/graph')
def my_graph_page():
    graph_fig = model.makegraph()
    graph_html = mpld3.fig_to_html(graph_fig)
    return '<h1>My graph</h1>' + graph_html
        
# Run the Flask server in the app.
#app.run_server()  # Twisted + client + server    
#app.run_server(with_twisted=False)  # Flask app (only) without Twisted  
#app.run_server(with_client=True, with_flask=False)  # client only
app.run_server(with_client=False, with_flask=True)  # Flask only with Twisted
