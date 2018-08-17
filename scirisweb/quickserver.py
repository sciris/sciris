"""
A simple server used to show mpld3 images -- based on _server in the mpld3 package.

Version: 2015dec29
"""

import threading
import webbrowser
import socket
import itertools
import random
import numpy as np
from collections import OrderedDict
import sciris as sc
try:    import BaseHTTPServer as server # Python 2.x
except: from http import server # Python 3.x

def normalize_obj(obj):
    """
    This is the main conversion function for Python data-structures into
    JSON-compatible data structures.

    Use this as much as possible to guard against data corruption!

    Args:
        obj: almost any kind of data structure that is a combination
            of list, numpy.ndarray, odicts etc

    Returns:
        A converted dict/list/value that should be JSON compatible
    """

    if isinstance(obj, list) or isinstance(obj, tuple):
        return [normalize_obj(p) for p in list(obj)]
    
    if isinstance(obj, np.ndarray):
        if obj.shape: # Handle most cases, incluing e.g. array([5])
            return [normalize_obj(p) for p in list(obj)]
        else: # Handle the special case of e.g. array(5)
            return [normalize_obj(p) for p in list(np.array([obj]))]

    if isinstance(obj, dict):
        return {str(k): normalize_obj(v) for k, v in obj.items()}

    if isinstance(obj, sc.odict):
        result = OrderedDict()
        for k, v in obj.items():
            result[str(k)] = normalize_obj(v)
        return result

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, float):
        if np.isnan(obj):
            return None

    if isinstance(obj, np.float64):
        if np.isnan(obj):
            return None
        else:
            return float(obj)

    if isinstance(obj, unicode):
        try:    string = str(obj) # Try to convert it to ascii
        except: string = obj # Give up and use original
        return string

    if isinstance(obj, set):
        return list(obj)

    return obj

def generate_handler(html, files=None):
    if files is None:
        files = {}

    class MyHandler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            """Respond to a GET request."""
            if self.path == '/':
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())
            elif self.path in files:
                content_type, content = files[self.path]
                self.send_response(200)
                self.send_header("Content-type", content_type)
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self.send_error(404)

    return MyHandler


def find_open_port(ip, port, n=50):
    """Find an open port near the specified port"""
    ports = itertools.chain((port + i for i in range(n)), (port + random.randint(-2 * n, 2 * n)))
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((ip, port))
        s.close()
        if result != 0: return port
    raise ValueError("No open ports found")


def serve(html, ip='127.0.0.1', port=8888, n_retries=50):
    """Start a server serving the given HTML, and open a browser

    Example:
       html = '<b>Hello, world!</b>'
       import optima as op
       op._webserver.serve(html)

    Parameters
    ----------
    html : string
        HTML to serve
    ip : string (default = '127.0.0.1')
        ip address at which the HTML will be served.
    port : int (default = 8888)
        the port at which to serve the HTML
    n_retries : int (default = 50)
        the number of nearby ports to search if the specified port is in use.
    """
    port = find_open_port(ip, port, n_retries)
    Handler = generate_handler(html)

    # Create server
    srvr = server.HTTPServer((ip, port), Handler)

    # Use a thread to open a web browser pointing to the server
    try: browser = webbrowser.get('google-chrome') # CK: Try google Chrome first
    except: browser = webbrowser.get() # If not, just use whatever
    b = lambda: browser.open('http://{0}:{1}'.format(ip, port))
    threading.Thread(target=b).start()

    # CK: don't serve forever, just create it once
    srvr.handle_request()


def browser(figs=None, doserve=True, jquery_url=None, d3_url=None, mpld3_url=None):
    ''' 
    Create an MPLD3 GUI and display in the browser.
    
    Usage example:
        import pylab as pl
        figs = []
        for n in [10, 50]:
            fig = pl.figure()
            pl.plot(pl.rand(n), pl.rand(n))
            figs.append(fig)
        qs.browser(figs)
    
    figs can be a single figure or a list of figures.
    
    With doserve=True, launch a web server. Otherwise, return the HTML representation of the figures.
    
    Version: 2018jul24
    '''
    import mpld3 # Only import this if needed, since might not always be available
    import json
    legacy = False

    ## Specify the div style, and create the HTML template we'll add the data to -- WARNING, library paths are hardcoded!
    divstyle = "float: left"
    if legacy:
        if jquery_url is None: jquery_url = 'https://code.jquery.com/jquery-1.11.3.min.js'
        if d3_url     is None: d3_url     = 'https://mpld3.github.io/js/d3.v3.min.js'
        if mpld3_url  is None: mpld3_url  = 'https://mpld3.github.io/js/mpld3.v0.3git.js'
    else:
        if jquery_url is None: jquery_url = 'https://code.jquery.com/jquery-1.11.3.min.js'
        if d3_url     is None: d3_url     = 'http://thekerrlab.com/tmp/d3.v5.min.js'
        if mpld3_url  is None: mpld3_url  = 'http://thekerrlab.com/tmp/mpld3.v0.4.0.min.js'
    html = '''
    <html>
    <head><script src="%s"></script></head>
    <body>
    !MAKE DIVS!
    <script>
    function mpld3_load_lib(url, callback){var s = document.createElement('script'); s.src = url; s.async = true; s.onreadystatechange = s.onload = callback; s.onerror = function(){console.warn("failed to load library " + url);}; document.getElementsByTagName("head")[0].appendChild(s)} mpld3_load_lib("%s", function(){mpld3_load_lib("%s", function(){
    !DRAW FIGURES!
    })});
    </script>
    </body></html>
    ''' % (jquery_url, d3_url, mpld3_url)
    
    ## Create the figures to plot
    jsons = [] # List for storing the converted JSONs
    figs = sc.promotetolist(figs)
    nfigs = len(figs) # Figure out how many plots there are
    for f in range(nfigs): # Loop over each plot
        jsons.append(str(json.dumps(normalize_obj(mpld3.fig_to_dict(figs[f]))))) # Save to JSON
    
    ## Create div and JSON strings to replace the placeholers above
    divstr = ''
    jsonstr = ''
    for f in range(nfigs):
        divstr += '<div style="%s" id="fig%i" class="fig"></div>\n' % (divstyle, f) # Add div information: key is unique ID for each figure
        jsonstr += 'mpld3.draw_figure("fig%i", %s);\n' % (f, jsons[f]) # Add the JSON representation of each figure -- THIS IS KEY!
    html = html.replace('!MAKE DIVS!',divstr) # Populate div information
    html = html.replace('!DRAW FIGURES!',jsonstr) # Populate figure information
    
    ## Launch a server or return the HTML representation
    if doserve: serve(html)
    else:       return html