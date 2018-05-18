"""
scirisapp.py -- classes for Sciris (Flask-based) apps 
    
Last update: 5/17/18 (gchadder3)
"""

# Imports
from flask import Flask, request, json, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
import sys
import os
import numpy as np
from functools import wraps
import traceback
from twisted.internet import reactor
from twisted.internet.endpoints import serverFromString
from twisted.logger import globalLogBeginner, FileLogObserver, formatEvent
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.static import File
from twisted.web.wsgi import WSGIResource
from twisted.python.threadpool import ThreadPool
import fileio

#
# Classes
#

class ScirisApp(object):
    """
    An object encapsulating a Sciris webapp, generally.  This app has an 
    associated Flask app that actually runs to listen for and respond to 
    HTTP requests.
    
    Methods:
        __init__(client_path: str [None]): void -- constructor
        run_server(with_twisted: bool [True], with_flask: bool [True], 
            with_client: [True]): void -- run the actual server
        define_endpoint_layout(rule: str, layout: list): void -- set up an 
            endpoint with a layout of a static Flask page
        add_RPC(new_RPC: ScirisRPC): void -- add an RPC to the app's dictionary
        add_RPC_dict(new_RPC_dict: dict): void -- add RPCs from another RPC 
            dictionary to the app's dictionary
        register_RPC(**kwargs): func -- decorator factory for adding a function 
            as an RPC
        _layout_render(): void -- render HTML for the layout of the given 
            rule (in the request)
        _do_RPC(): void -- process a request in such a way as to find and 
            dispatch the chosen RPC, doing any validation checking and error 
            logging necessary in the process
                    
    Attributes:
        flask_app (Flask) -- the actual Flask app
        config (flask.config.Config) -- the Flask configuration dictionary
        define_endpoint_callback (func) -- points to the flask_app.route() 
            function so you can use @app.define_endpoint_callback in the calling code
        endpoint_layout_dict (dict) -- dictionary holding static page layouts
        RPC_dict (dict) -- dictionary of site RPCs
            
    Usage:
        >>> app = ScirisApp()                      
    """
    
    def  __init__(self, app_config=None, client_dir=None):
        # Open a new Flask app.
        self.flask_app = Flask(__name__)
        
        # If we have a config module, load it into Flask's config dict.
        if app_config is not None:
            self.flask_app.config.from_object(app_config)
            
        # Set an easier link to the configs dictionary.
        self.config = self.flask_app.config

        # Set up default values for configs that are not already defined.
        self._set_config_defaults(self.config)
        
        # Set an alias for the decorator factory for adding an endpoint.
        self.define_endpoint_callback = self.flask_app.route
        
        # Create an empty layout dictionary.
        self.endpoint_layout_dict = {}
        
        # Create an empty RPC dictionary.
        self.RPC_dict = {} 
        
        # Set config parameters in the configs if they were passed in.
        if client_dir is not None:
            self.config['CLIENT_DIR'] = client_dir
            
        # Set up file paths.
        self._init_file_dirs(self.config)
      
    @staticmethod
    def _set_config_defaults(app_config):
        if 'CLIENT_DIR' not in app_config:
            app_config['CLIENT_DIR'] = '.'
            
        if 'TWISTED_PORT' not in app_config:
            app_config['TWISTED_PORT'] = 8080

    @staticmethod
    def _init_file_dirs(app_config):
        # If the config parameter is not there (or comment out), set the 
        # path to None.
        if 'TRANSFER_DIR' not in app_config:
            transfer_dir_path = None
        else:
            transfer_dir_path = app_config['TRANSFER_DIR']
            
#            # If we have an absolute directory, use it.
#            if os.path.isabs(app_config['TRANSFER_DIR']):
#                transfer_dir_path = app_config['TRANSFER_DIR']
#                
#            # Otherwise (we have a relative path), use it (correcting so it is with 
#            # respect to the sciris repo directory).
#            else:
#                transfer_dir_path = '%s%s%s' % (os.pardir, os.sep, 
#                    app_config['TRANSFER_DIR'])  
        
        # Set the file save root path.
        
        # If the config parameter is not there (or comment out), set the 
        # path to None.
        if 'FILESAVEROOT_DIR' not in app_config:
            file_save_root_path = None
        else:  
            file_save_root_path = app_config['FILESAVEROOT_DIR']
            
#            # If we have an absolute directory, use it.
#            if os.path.isabs(app_config['FILESAVEROOT_DIR']):
#                file_save_root_path = app_config['FILESAVEROOT_DIR']
#                
#            # Otherwise (we have a relative path), use it (correcting so it is with 
#            # respect to the sciris repo directory).
#            else:
#                file_save_root_path = '%s%s%s' % (os.pardir, os.sep, 
#                    app_config['FILESAVEROOT_DIR'])

        # Create a file save directory.
        # Perhaps this is unnecessary for this particular webapp, but I'll leave 
        # it in for now.
        fileio.file_save_dir = fileio.FileSaveDirectory(file_save_root_path, temp_dir=False)
        
        # Create a downloads directory.
        fileio.downloads_dir = fileio.FileSaveDirectory(transfer_dir_path, temp_dir=True)
        
        # Have the uploads directory use the same directory as the downloads 
        # directory.
        fileio.uploads_dir = fileio.downloads_dir
        
        # Show the downloads and uploads directories.
        print '>> File save directory path: %s' % fileio.file_save_dir.dir_path
        print '>> Downloads directory path: %s' % fileio.downloads_dir.dir_path
        print '>> Uploads directory path: %s' % fileio.uploads_dir.dir_path
        
    def run_server(self, with_twisted=True, with_flask=True, with_client=True):
        # If we are not running the app with Twisted, just run the Flask app.
        if not with_twisted:
            self.flask_app.run()

        # Otherwise (with Twisted).
        else:
            if not with_client and not with_flask:
                run_twisted(port=self.config['TWISTED_PORT'])  # nothing, should return error
            if with_client and not with_flask:
                run_twisted(port=self.config['TWISTED_PORT'], 
                    client_dir=self.config['CLIENT_DIR'])   # client page only / no Flask
            elif not with_client and with_flask:
                run_twisted(port=self.config['TWISTED_PORT'], 
                    flask_app=self.flask_app)  # Flask app only, no client
            else:
                run_twisted(port=self.config['TWISTED_PORT'], 
                    flask_app=self.flask_app, 
                    client_dir=self.config['CLIENT_DIR'])  # Flask + client
                
    def define_endpoint_layout(self, rule, layout):
        # Save the layout in the endpoint layout dictionary.
        self.endpoint_layout_dict[rule] = layout
        
        # Set up the callback, to point to the _layout_render() function.
        self.flask_app.add_url_rule(rule, 'layout_render', self._layout_render)
        
    def add_RPC(self, new_RPC):
        # If we are setting up our first RPC, add the actual endpoint.
        if len(self.RPC_dict) == 0:
            self.flask_app.add_url_rule('/rpcs', 'do_RPC', self._do_RPC, methods=['POST'])
          
        # If the function name is in the dictionary...
        if new_RPC.funcname in self.RPC_dict:
            # If we have the power to override the function, give a warning.
            if new_RPC.override:
                print('>> add_RPC(): WARNING: Overriding previous version of %s:' % new_RPC.funcname)
                print('>>   Old: %s.%s' % 
                    (self.RPC_dict[new_RPC.funcname].call_func.__module__, 
                    self.RPC_dict[new_RPC.funcname].call_func.__name__))
                print('>>   New: %s.%s' % (new_RPC.call_func.__module__, 
                    new_RPC.call_func.__name__))
            # Else, give an error, and exit before the RPC is added.
            else:
                print('>> add_RPC(): ERROR: Attempt to override previous version of %s: %s.%s' % \
                      (new_RPC.funcname, self.RPC_dict[new_RPC.funcname].call_func.__module__, self.RPC_dict[new_RPC.funcname].funcname))
                return
        
        # Create the RPC and add it to the dictionary.
        self.RPC_dict[new_RPC.funcname] = new_RPC
    
    def add_RPC_dict(self, new_RPC_dict):
        for RPC_funcname in new_RPC_dict:
            self.add_RPC(new_RPC_dict[RPC_funcname])

    def register_RPC(self, **callerkwargs):
        def RPC_decorator(RPC_func):
            @wraps(RPC_func)
            def wrapper(*args, **kwargs):        
                RPC_func(*args, **kwargs)

            # Create the RPC and try to add it to the dictionary.
            new_RPC = ScirisRPC(RPC_func, **callerkwargs)
            self.add_RPC(new_RPC)
            
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
    
    def _do_RPC(self):
        # Check to see whether the RPC is getting passed in in request.form.
        # If so, we are doing an upload, and we want to download the RPC 
        # request info from the form, not request.data.
        if 'funcname' in request.form:
            # Pull out the function name, args, and kwargs
            fn_name = request.form.get('funcname')
            args = json.loads(request.form.get('args', "[]"))
            kwargs = json.loads(request.form.get('kwargs', "{}"))
            
        # Otherwise, we have a normal or download RPC, which means we pull 
        # the RPC request info from request.data.
        else:
            reqdict = json.loads(request.data)
            fn_name = reqdict['funcname']
            args = reqdict.get('args', [])
            kwargs = reqdict.get('kwargs', {})
        
        # If the function name is not in the RPC dictionary, return an 
        # error.
        if not fn_name in self.RPC_dict:
            return jsonify({'error': 'Could not find requested RPC'})
            
        # Get the RPC we've found.
        found_RPC = self.RPC_dict[fn_name]
        
        # Do any validation checks we need to do and return errors if they 
        # don't pass.
        
        # If we are doing an upload...
        if found_RPC.call_type == 'upload':
            # Grab the formData file that was uploaded.    
            file = request.files['uploadfile']
        
            # Extract a sanitized filename from the one we start with.
            filename = secure_filename(file.filename)
            
            # Generate a full upload path/file name.
            uploaded_fname = os.path.join(fileio.uploads_dir.dir_path, filename)
        
            # Save the file to the uploads directory.
            file.save(uploaded_fname)
        
            # Prepend the file name to the args list.
            args.insert(0, uploaded_fname)
        
        # Show the call of the function.    
        print('>> Calling RPC function "%s.%s"' % 
            (found_RPC.call_func.__module__, found_RPC.funcname))
    
        # Execute the function to get the results, putting it in a try block 
        # in case there are errors in what's being called. 
        try:
            result = found_RPC.call_func(*args, **kwargs)
        except Exception as e:
            # Grab the trackback stack.
            exception = traceback.format_exc()
            
            # Post an error to the Flask logger
            # limiting the exception information to 10000 characters maximum
            # (to prevent monstrous sqlalchemy outputs)
            self.flask_app.logger.error("Exception during request %s: %.10000s" % (request, exception))
            
            # If we have a werkzeug exception, pass it on up to werkzeug to 
            # resolve and reply to.
            # [Do we really want to do this?: GLC 5/15/18]
            if isinstance(e, HTTPException):
                raise
                
            # Send back a response with status 500 that includes the exception 
            # traceback.
            code = 500
            reply = {'exception': exception}
            return make_response(jsonify(reply), code)
        
        # If we are doing a download, prepare the response and send it off.
        if found_RPC.call_type == 'download':
            # If we got None for a result (the full file name), return an error 
            # to the client.
            if result is None:
                return jsonify({'error': 'Could not find requested resource'})
            
            # Else, if the result is not even a string (which means it's not 
            # a file name as expected)...
            elif type(result) is not str:
                # If the result is a dict with an 'error' key, then assume we 
                # have a custom error that we want the RPC to return to the 
                # browser, and do so.
                if type(result) is dict and 'error' in result:
                    return jsonify(result)
                
                # Otherwise, return an error that the download RPC did not 
                # return a filename.
                else:
                    return jsonify({'error': 'Download RPC did not return a filename'})
            
            # Pull out the directory and file names from the full file name.
            dir_name, file_name = os.path.split(result)
         
            # Make the response message with the file loaded as an attachment.
            response = send_from_directory(dir_name, file_name, as_attachment=True)
            response.status_code = 201  # Status 201 = Created
            response.headers['filename'] = file_name
                
            # Unfortunately, we cannot remove the actual file at this point 
            # because it is in use during the actual download, so we rely on 
            # later cleanup to remove download files.
        
            # Return the response message.
            return response
    
        # Otherwise (normal and upload RPCs), 
        else:
            # If we are doing an upload....
            if found_RPC.call_type == 'upload':
                # Erase the physical uploaded file, since it is no longer needed.
                os.remove(uploaded_fname)
        
            # If None was returned by the RPC function, return ''.
            if result is None:
                return ''
        
            # Otherwise, convert the result (probably a dict) to JSON and return it.
            else:
                return jsonify(json_sanitize_result(result))
        
        
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
    
    
class ScirisRPC(object):
    def __init__(self, call_func, call_type='normal', override=False):
        self.call_func = call_func
        self.funcname = call_func.__name__
        self.call_type = call_type
        self.override = override
    
    
def run_twisted(port=8080, flask_app=None, client_dir=None):
    # Give an error if we pass in no Flask server or client path.
    if (flask_app is None) and (client_dir is None):
        print 'ERROR: Neither client or server are defined.'
        return None
    
    # Set up logging.
    globalLogBeginner.beginLoggingTo([
        FileLogObserver(sys.stdout, lambda _: formatEvent(_) + "\n")])

    # If there is a client path, set up the base resource.
    if client_dir is not None:
        base_resource = File(client_dir)
        
    # If we have a flask app...
    if flask_app is not None:
        # Create a thread pool to use with the app.
        thread_pool = ThreadPool(maxthreads=30)
        
        # Create the WSGIResource object for the flask server.
        wsgi_app = WSGIResource(reactor, thread_pool, flask_app)
        
        # If we have no client path, set the WSGI app to be the base resource.
        if client_dir is None:
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


def make_register_RPC(RPC_dict, **callerkwargs):
    def RPC_decorator_factory(**callerkwargs):
        def RPC_decorator(RPC_func):
            @wraps(RPC_func)
            def wrapper(*args, **kwargs):        
                RPC_func(*args, **kwargs)
    
            # Create the RPC and add it to the dictionary.
            RPC_dict[RPC_func.__name__] = ScirisRPC(RPC_func, **callerkwargs)
            
            return wrapper
    
        return RPC_decorator
    
    return RPC_decorator_factory