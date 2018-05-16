"""
run_staticapp.py -- Simple ScirisApp use case 
    
Last update: 5/16/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp
import sciris2gc.appcomponents as acs
import model # The actual Python model we want to incorporate

# Create the ScirisApp object.  NOTE: app.config will thereafter contain all 
# of the configuration parameters, including for Flask.
app = ScirisApp()
    
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
app.run_server(with_client=False, with_flask=True)  # Flask only with Twisted
