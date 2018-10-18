"""
run_staticapp.py -- Simple ScirisApp use case 
    
Last update: 6/8/18 (gchadder3)
"""

# Imports
from sciris.weblib.scirisapp import ScirisApp
import sciris.weblib.appcomponents as acs
import model # The actual Python model we want to incorporate

# Create the ScirisApp object.  NOTE: app.config will thereafter contain all 
# of the configuration parameters, including for Flask.
app = ScirisApp(__file__)
    
# Define the first static page at /staticapp.   
the_layout = [ \
    acs.H1('Here is a Title'), 
    acs.P('Here is a sample paragraph.'), 
    acs.Graph(model.makegraph())]
app.define_endpoint_layout('/staticapp', the_layout)

# Define the first static page at /staticapp2.  
the_layout2 = [ \
    acs.H1('Here is a Second Title'), 
    acs.P('Here is another sample paragraph.'),
    acs.H1('Another Graph'),
    acs.Graph(model.makegraph())]
app.define_endpoint_layout('/staticapp2', the_layout2)      
 
# Run the Flask server in the app.
#app.run_server()  # Twisted + client + server    
#app.run_server(with_twisted=False)  # Flask app (only) without Twisted  
#app.run_server(with_client=True, with_flask=False)  # client only
app.run_server(with_client=False, with_flask=True, 
    use_twisted_logging=True)  # Flask only with Twisted
