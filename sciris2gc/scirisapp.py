"""
scirisapp.py -- classes for Sciris (Flask-based) apps 
    
Last update: 5/4/18 (gchadder3)
"""

# Imports
from flask import Flask
import sys
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
        runServer(): void -- run the actual server
                    
    Attributes:
        flaskApp (Flask) -- the actual Flask app
            
    Usage:
        >>> theApp = ScirisApp()                      
    """
    
    def  __init__(self, clientPath=None):
        # Open a new Flask app.
        self.flaskApp = Flask(__name__)
        
        # Save the client path.
        
        # If nothing was passed in, then assume the path is the current 
        # directory.
        if clientPath is None:
            self.clientPath = '.'
        else:
            self.clientPath = clientPath
        
    def runServer(self, withTwisted=True, withFlask=True, withClient=True):
        # If we are not running the app with Twisted, just run the Flask app.
        if not withTwisted:
            self.flaskApp.run()

        # Otherwise (with Twisted).
        else:
            if not withClient and not withFlask:
                runTwisted()  # nothing, should return error
            if withClient and not withFlask:
                runTwisted(clientPath=self.clientPath)   # client page only / no Flask
            elif not withClient and withFlask:
                runTwisted(theFlaskApp=self.flaskApp)  # Flask app only, no client
            else:
                runTwisted(theFlaskApp=self.flaskApp, clientPath=self.clientPath)  # Flask + client
        
        
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
    

def runTwisted(thePort=8080, theFlaskApp=None, clientPath=None):
    # Give an error if we pass in no Flask server or client path.
    if (theFlaskApp is None) and (clientPath is None):
        print 'ERROR: Neither client or server are defined.'
        return None
    
    # Set up logging.
    globalLogBeginner.beginLoggingTo([
        FileLogObserver(sys.stdout, lambda _: formatEvent(_) + "\n")])

    # If there is a client path, set up the base resource.
    if clientPath is not None:
        baseResource = File(clientPath)
        
    # If we have a flask app...
    if theFlaskApp is not None:
        # Create a thread pool to use with the app.
        threadPool = ThreadPool(maxthreads=30)
        
        # Create the WSGIResource object for the flask server.
        wsgiApp = WSGIResource(reactor, threadPool, theFlaskApp)
        
        # If we have no client path, set the WSGI app to be the base resource.
        if clientPath is None:
            baseResource = ScirisResource(wsgiApp)
        # Otherwise, make the Flask app a child resource.
        else: 
            baseResource.putChild('api', ScirisResource(wsgiApp))

        # Start the threadpool now, shut it down when we're closing
        threadPool.start()
        reactor.addSystemEventTrigger('before', 'shutdown', threadPool.stop)
    
    # Create the site.
    site = Site(baseResource)
    
    # Create the endpoint we want to listen on, and point it to the site.
    endpoint = serverFromString(reactor, "tcp:port=" + str(thePort))
    endpoint.listen(site)

    # Start the reactor.
    reactor.run()        