"""
scirisapp.py -- classes for Sciris (Flask-based) apps 
    
Last update: 5/9/18 (gchadder3)
"""

# Imports
from flask import Flask, request
import sys
import numpy as np
from twisted.internet import reactor
from twisted.internet.endpoints import serverFromString
from twisted.logger import globalLogBeginner, FileLogObserver, formatEvent
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.static import File
from twisted.web.wsgi import WSGIResource
from twisted.python.threadpool import ThreadPool

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
        run_server(): void -- run the actual server
                    
    Attributes:
        flask_app (Flask) -- the actual Flask app
        client_path (str) -- home path for any client browser-side files
        define_endpoint_callback (func) -- points to the flask_app.route() 
            function so you can use @app.define_endpoint_callback in the calling code
            
    Usage:
        >>> app = ScirisApp()                      
    """
    
    def  __init__(self, client_path=None):
        # Open a new Flask app.
        self.flask_app = Flask(__name__)
        
        # Remember a decorator for adding an endpoint.
        self.define_endpoint_callback = self.flask_app.route
        
        # Create an empty layout dictionary.
        self.endpoint_layout_dict = {}
        
        # Save the client path.
        
        # If nothing was passed in, then assume the path is the current 
        # directory.
        if client_path is None:
            self.client_path = '.'
        else:
            self.client_path = client_path
        
    def run_server(self, with_twisted=True, with_flask=True, with_client=True):
        # If we are not running the app with Twisted, just run the Flask app.
        if not with_twisted:
            self.flask_app.run()

        # Otherwise (with Twisted).
        else:
            if not with_client and not with_flask:
                run_twisted()  # nothing, should return error
            if with_client and not with_flask:
                run_twisted(client_path=self.client_path)   # client page only / no Flask
            elif not with_client and with_flask:
                run_twisted(flask_app=self.flask_app)  # Flask app only, no client
            else:
                run_twisted(flask_app=self.flask_app, client_path=self.client_path)  # Flask + client
                
    def define_endpoint_layout(self, rule, layout):
        # Save the layout in the endpoint layout dictionary.
        self.endpoint_layout_dict[rule] = layout
        
        # Set up the callback, to point to the _layout_render() function.
        self.flask_app.add_url_rule(rule, 'layout_render', self._layout_render)

    def register_RPC(self, rule, **callerkwargs):
        def RPC_decorator(RPC_func):
            def wrapper(*args, **kwargs):        
                RPC_func(*args, **kwargs)
                
            self.flask_app.add_url_rule(rule, 'RPC', RPC_func, methods=['POST'])
            return wrapper

        return RPC_decorator
           
    def _layout_render(self):
        render_str = '<html>'
        render_str += '<body>'
        for layout_comp in self.endpoint_layout_dict[str(request.url_rule)]:
            render_str += layout_comp.render()
        render_str += '</body>'
        render_str += '</html>'
        return render_str
        
        
class ScirisResource(Resource):
    isLeaf = True

    def __init__(self, wsgi):
        self._wsgi = wsgi

    def render(self, request):
#        request.prepath = []
#        request.postpath = ['api'] + request.postpath[:]

        # Get the WSGI render results (i.e. for Flask app).
        r = self._wsgi.render(request)

        # Keep the client browser from caching Flask response, and set 
        # the response as already being "expired."
        request.responseHeaders.setRawHeaders(
            b'Cache-Control', [b'no-cache', b'no-store', b'must-revalidate'])
        request.responseHeaders.setRawHeaders(b'expires', [b'0'])
        
        # Pass back the WSGI render results.
        return r
    

def run_twisted(port=8080, flask_app=None, client_path=None):
    # Give an error if we pass in no Flask server or client path.
    if (flask_app is None) and (client_path is None):
        print 'ERROR: Neither client or server are defined.'
        return None
    
    # Set up logging.
    globalLogBeginner.beginLoggingTo([
        FileLogObserver(sys.stdout, lambda _: formatEvent(_) + "\n")])

    # If there is a client path, set up the base resource.
    if client_path is not None:
        base_resource = File(client_path)
        
    # If we have a flask app...
    if flask_app is not None:
        # Create a thread pool to use with the app.
        thread_pool = ThreadPool(maxthreads=30)
        
        # Create the WSGIResource object for the flask server.
        wsgi_app = WSGIResource(reactor, thread_pool, flask_app)
        
        # If we have no client path, set the WSGI app to be the base resource.
        if client_path is None:
            base_resource = ScirisResource(wsgi_app)
        # Otherwise, make the Flask app a child resource.
        else: 
            base_resource.putChild('api', ScirisResource(wsgi_app))

        # Start the threadpool now, shut it down when we're closing
        thread_pool.start()
        reactor.addSystemEventTrigger('before', 'shutdown', thread_pool.stop)
    
    # Create the site.
    site = Site(base_resource)
    
    # Create the endpoint we want to listen on, and point it to the site.
    endpoint = serverFromString(reactor, "tcp:port=" + str(port))
    endpoint.listen(site)

    # Start the reactor.
    reactor.run()  
    
    
def json_sanitize_result(theResult):
    """
    This is the main conversion function for Python data-structures into
    JSON-compatible data structures.
    Use this as much as possible to guard against data corruption!
    Args:
        theResult: almost any kind of data structure that is a combination
            of list, numpy.ndarray, etc.
    Returns:
        A converted dict/list/value that should be JSON compatible
    """

    if isinstance(theResult, list) or isinstance(theResult, tuple):
        return [json_sanitize_result(p) for p in list(theResult)]
    
    if isinstance(theResult, np.ndarray):
        if theResult.shape: # Handle most cases, incluing e.g. array([5])
            return [json_sanitize_result(p) for p in list(theResult)]
        else: # Handle the special case of e.g. array(5)
            return [json_sanitize_result(p) for p in list(np.array([theResult]))]

    if isinstance(theResult, dict):
        return {str(k): json_sanitize_result(v) for k, v in theResult.items()}

    if isinstance(theResult, np.bool_):
        return bool(theResult)

    if isinstance(theResult, float):
        if np.isnan(theResult):
            return None

    if isinstance(theResult, np.float64):
        if np.isnan(theResult):
            return None
        else:
            return float(theResult)

    if isinstance(theResult, unicode):
        return theResult
#        return str(theResult)  # original line  (watch to make sure the 
#                                                 new line doesn't break things)
    
    if isinstance(theResult, set):
        return list(theResult)

#    if isinstance(theResult, UUID):
#        return str(theResult)

    return theResult