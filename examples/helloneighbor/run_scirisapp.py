"""
run_scirisapp.py -- Simple ScirisApp use case 
    
Last update: 5/7/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisApp
import model # The actual Python model we want to incorporate
import mpld3 # For plotting

# Create the ScirisApp object.
app = ScirisApp()
    
@app.define_endpoint_callback('/')
def my_root_page():
    return '<h1>Hello, Flask!</h1>'
    
@app.define_endpoint_callback('/api')
def my_api_page():
    return '<h1>Look at that!  Two Flask endpoints!</h1>' 
    
@app.define_endpoint_callback('/graph')
def my_graph_page():
    graph_fig = model.makegraph()
    graph_html = mpld3.fig_to_html(graph_fig)
    return '<h1>My graph</h1>' + graph_html



# Stuff to pull out...
class TitleComponent(object):
    def __init__(self, display_text):
        self.display_text = display_text
        
    def render(self):
        return '<h1>%s</h1>' % self.display_text
        
class ParagraphComponent(object):
    def __init__(self, display_text):
        self.display_text = display_text
        
    def render(self):
        return '<p>%s</p>' % self.display_text 
    
class GraphComponent(object):
    def __init__(self, graph_fig):
        self.graph_fig = graph_fig
        
    def render(self):
        return mpld3.fig_to_html(self.graph_fig) 
    
the_layout = [TitleComponent('Here is a Title'), 
    ParagraphComponent('Here is a sample paragraph.'), 
    GraphComponent(model.makegraph())]
      
@app.define_endpoint_callback('/staticapp')
def my_staticapp_page():
    render_str = '<html>'
    render_str += '<body>'
    for layout_comp in the_layout:
        render_str += layout_comp.render()
    render_str += '</body>'
    render_str += '</html>'
    return render_str


        
# Run the Flask server in the app.
#app.run_server()  # Twisted + client + server    
#app.run_server(with_twisted=False)  # Flask app (only) without Twisted  
#app.run_server(with_client=True, with_flask=False)  # client only
app.run_server(with_client=False, with_flask=True)  # Flask only with Twisted
