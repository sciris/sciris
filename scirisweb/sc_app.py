"""
sc_app.py -- classes for Sciris (Flask-based) apps 
    
Last update: 2018sep24
"""

# Imports
import sys
import os
import socket
import logging
import traceback
from functools import wraps
import matplotlib.pyplot as ppl
from collections import OrderedDict
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from flask import Flask, request, abort, json, jsonify as flask_jsonify, send_from_directory, make_response, current_app as flaskapp
from flask_login import LoginManager, current_user
from flask_session import RedisSessionInterface
from twisted.internet import reactor
from twisted.internet.endpoints import serverFromString
from twisted.logger import globalLogBeginner, FileLogObserver, formatEvent
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.static import File
from twisted.web.wsgi import WSGIResource
from twisted.python.threadpool import ThreadPool
import sciris as sc
from . import sc_rpcs as rpcs
from . import sc_datastore as ds
from . import sc_users as users
from . import sc_tasks as tasks


#################################################################
### Classes and functions
#################################################################

__all__ = ['robustjsonify', 'ScirisApp', 'ScirisResource', 'run_twisted', 'flaskapp']

def robustjsonify(response, robustify=True):
    ''' Flask's default jsonifier clobbers dict order; this preserves it -- warning, may cause problems with some payloads, use with caution! '''
    a = flask_jsonify(sc.sanitizejson(response)) # Use standard Flask jsonification
    sc.pr(a)
    sc.pp(a)
    resp_a = a.response
    resp_b = [sc.sanitizejson(response, tostring=True)+'\n']
    print('HIIIIIIIIIIIIIIIIIIIIII')
    print(sc.prepr(resp_a))
    print(repr(resp_b))
    if robustify:
        output = flask_jsonify('placeholder') # Create the container
        output.response = resp_a # Replace there response with a properly formatted JSON
    else:
        output = flask_jsonify(sc.sanitizejson(response)) # Use standard Flask jsonification
    return output


class ScirisApp(object):
    """
    An object encapsulating a Sciris webapp, generally.  This app has an 
    associated Flask app that actually runs to listen for and respond to 
    HTTP requests.
    
    Methods:
        __init__(script_path: str, self.config: config module [None], 
            client_dir: str [None]): void -- constructor
        run(with_twisted: bool [True], with_flask: bool [True], 
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
        >>> app = ScirisApp(__file__)                      
    """
    
    def  __init__(self, filepath=None, config=None, name=None, RPC_dict=None, **kwargs):
        if name is None: name = 'default'
        self.name = name
        self.flask_app = Flask(__name__) # Open a new Flask app.
        if config is not None: # If we have a config module, load it into Flask's config dict.
            self.flask_app.config.from_object(config)
        self.config = self.flask_app.config # Set an easier link to the configs dictionary.
        self.config['ROOT_ABS_DIR'] = os.path.dirname(os.path.abspath(filepath)) # Extract the absolute directory path from the above.
        self._init_logger() # Initialize the Flask logger. 
        self._set_config_defaults() # Set up default values for configs that are not already defined.
        self._update_config_defaults(**kwargs) # If additional command-line arguments are supplied, use them
        self.define_endpoint_callback = self.flask_app.route # Set an alias for the decorator factory for adding an endpoint.
        self.endpoint_layout_dict = {} # Create an empty layout dictionary.
        self.RPC_dict = {}  # Create an empty RPC dictionary.
        
        # Initialize plotting
        try:
            ppl.switch_backend(self.config['MATPLOTLIB_BACKEND'])
            print('Matplotlib backend switched to "%s"' % (self.config['MATPLOTLIB_BACKEND']))
        except Exception as E:
            print('Switching Matplotlib backend to "%s" failed: %s' % (self.config['MATPLOTLIB_BACKEND'], repr(E)))
            
        # Set up file paths.
        self._init_file_dirs()
        
        # Set up RPCs
        if RPC_dict:
            self.add_RPC_dict(RPC_dict)
        
        # If we are including DataStore functionality, initialize it.
        if self.config['USE_DATASTORE']:
            self._init_datastore()
            self.flask_app.datastore = self.datastore
            self.flask_app.session_interface = RedisSessionInterface(self.datastore.redis, 'sess')

        # If we are including DataStore and users functionality, initialize users.
        if self.config['USE_DATASTORE'] and self.config['USE_USERS']:
            self.login_manager = LoginManager() # Create a LoginManager() object.
            
            # This function gets called when authentication gets done by 
            # Flask-Login.  userid is the user ID pulled out of the session 
            # cookie the browser passes in during an HTTP request.  The 
            # function returns the User object that matches this, so that the 
            # user data may be used to authenticate (for example their 
            # rights to have admin access).  The return sets the Flask-Login 
            # current_user value.
            @self.login_manager.user_loader
            def load_user(userid):
                return self.datastore.loaduser(userid) # Return the matching user (if any).
            
            self.login_manager.init_app(self.flask_app) # Configure Flask app for login with the LoginManager.
            self.add_RPC_dict(users.RPC_dict) # Register the RPCs in the users.py module.
            
        # If we are including DataStore and tasks, initialize them.    
        if self.config['USE_DATASTORE'] and self.config['USE_TASKS']:
            self._init_tasks() # Initialize the users.
            self.add_RPC_dict(tasks.RPC_dict) # Register the RPCs in the user.py module.    
                
        return None # End of __init__
            
    def _init_logger(self):
        self.flask_app.logger.setLevel(logging.DEBUG)
        return None
    
    def _set_config_defaults(self):
        if 'CLIENT_DIR'         not in self.config: self.config['CLIENT_DIR']         = '.'
        if 'LOGGING_MODE'       not in self.config: self.config['LOGGING_MODE']       = 'FULL' 
        if 'SERVER_PORT'        not in self.config: self.config['SERVER_PORT']        = 8080
        if 'USE_DATASTORE'      not in self.config: self.config['USE_DATASTORE']      = False
        if 'USE_USERS'          not in self.config: self.config['USE_USERS']          = False
        if 'USE_TASKS'          not in self.config: self.config['USE_TASKS']          = False
        if 'MATPLOTLIB_BACKEND' not in self.config: self.config['MATPLOTLIB_BACKEND'] = 'Agg'
        if 'SLACK'              not in self.config: self.config['SLACK']              = None
        return None
    
    def _update_config_defaults(self, **kwargs):
        ''' Used to update config with command-line arguments '''
        for key,val in kwargs.items():
            KEY = key.upper() # Since they're always uppercase
            if KEY in self.config:
                origval = self.config[KEY]
                self.config[KEY] = val
                print('Resetting configuration option "%s" from "%s" to "%s"' % (KEY, origval, val))
            else:
                warningmsg = '\nWARNING: kwarg "%s":"%.100s" will be ignored since it is not in the list of valid config options:\n' % (KEY, val)
                for validkey in sorted(self.config.keys()):
                    warningmsg += '  %s\n' % validkey
                print(warningmsg)
        return None
                

    def _init_file_dirs(self):
        # Set the absolute client directory path.
        
        # If we do not have an absolute directory, tack what we have onto the 
        # ROOT_ABS_DIR setting.
        if not os.path.isabs(self.config['CLIENT_DIR']):
            self.config['CLIENT_DIR'] = os.path.join(self.config['ROOT_ABS_DIR'], self.config['CLIENT_DIR'])
        
        return None
        
    def _init_datastore(self):
        # Create the DataStore object, setting up Redis.
        self.datastore = ds.DataStore(redis_url=self.config['REDIS_URL'])
                
        if self.config['LOGGING_MODE'] == 'FULL':
            maxkeystoshow = 20
            keys = self.datastore.keys()
            nkeys = len(keys)
            keyinds = range(1,nkeys+1)
            keypairs = zip(keyinds, keys)
            print('>> Loaded DataStore with %s Redis key(s)' % nkeys)
            if nkeys>2*maxkeystoshow:
                print('>> First and last %s keys:' % maxkeystoshow)
                keypairs    = keypairs[:maxkeystoshow] + keypairs[-maxkeystoshow:]
            for k,key in keypairs:
                print('  Key %02i: %s' % (k,key))
        return None
    
    def _init_tasks(self):
        # Have the tasks.py module make the Celery app to connect to the worker, passing in the config parameters.
        print('Making Celery instance...')
        tasks.make_celery(self.config)
        
    def run(self, with_twisted=True, with_flask=True, with_client=True, do_log=False, show_logo=True):
        
        # Display the logo
        appstring = 'ScirisApp "%s" is now running :)' % self.name
        borderstr = '='*len(appstring)
        logostr = '''\
      ___  ___    %s 
     / __|/ __|   %s     
     \__ \ |__    %s     
     |___/\___|   %s     
                  %s''' % (' '*(len(appstring)+4), borderstr, appstring, borderstr, ' '*(len(appstring)+5))
        logocolors = ['gray','bgblue'] # ['gray','bgblue']
        if show_logo:
            print('')
            for linestr in logostr.splitlines(): sc.colorize(logocolors,linestr)
            print('')
        
        # Run the thing
        if not with_twisted: # If we are not running the app with Twisted, just run the Flask app.
            self.flask_app.run()
        else: # Otherwise (with Twisted).
            port       = int(self.config['SERVER_PORT']) # Not sure if casting to int is necessary
            client_dir = self.config['CLIENT_DIR']
            if   not with_client and not with_flask: run_twisted(port=port, do_log=do_log)  # nothing, should return error
            if   with_client     and not with_flask: run_twisted(port=port, do_log=do_log, client_dir=client_dir)   # client page only / no Flask
            elif not with_client and     with_flask: run_twisted(port=port, do_log=do_log, flask_app=self.flask_app)  # Flask app only, no client
            else:                                    run_twisted(port=port, do_log=do_log, flask_app=self.flask_app, client_dir=client_dir)  # Flask + client
        return None      
    
    def define_endpoint_layout(self, rule, layout):
        # Save the layout in the endpoint layout dictionary.
        self.endpoint_layout_dict[rule] = layout
        
        # Set up the callback, to point to the _layout_render() function.
        self.flask_app.add_url_rule(rule, 'layout_render', self._layout_render)
        return None
        
    def slacknotification(self, message=None):
        ''' Send a message on Slack '''
        if self.config.get('SLACK'):
            slack_webhook = self.config['SLACK'].get('webhook')
            slack_to      = self.config['SLACK'].get('to')
            slack_from    = self.config['SLACK'].get('from')
            sc.slacknotification(message=message, webhook=slack_webhook, to=slack_to, fromuser=slack_from, die=False)
        else:
            print('Cannot send Slack message "%s": Slack not enabled in config file' % message)
        return None

    def add_RPC(self, new_RPC):
        # If we are setting up our first RPC, add the actual endpoint.
        if len(self.RPC_dict) == 0:
            self.flask_app.add_url_rule('/rpcs', 'do_RPC', self._do_RPC, methods=['POST'])
          
        # If the function name is in the dictionary...
        if new_RPC.funcname in self.RPC_dict:
            # If we have the power to override the function, give a warning.
            if new_RPC.override:
                if self.config['LOGGING_MODE'] == 'FULL':
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
            new_RPC = rpcs.ScirisRPC(RPC_func, **callerkwargs)
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
        if 'funcname' in request.form: # Pull out the function name, args, and kwargs
            fn_name = request.form.get('funcname')
            args = json.loads(request.form.get('args', "[]"), object_pairs_hook=OrderedDict)
            kwargs = json.loads(request.form.get('kwargs', "{}"), object_pairs_hook=OrderedDict)
        else: # Otherwise, we have a normal or download RPC, which means we pull the RPC request info from request.data.
            reqdict = json.loads(request.data, object_pairs_hook=OrderedDict)
            fn_name = reqdict['funcname']
            args = reqdict.get('args', [])
            kwargs = reqdict.get('kwargs', {})
        
        # If the function name is not in the RPC dictionary, return an error.
        if not fn_name in self.RPC_dict:
            return robustjsonify({'error': 'Could not find requested RPC "%s"' % fn_name})
        
        found_RPC = self.RPC_dict[fn_name] # Get the RPC we've found.
        
        ## Do any validation checks we need to do and return errors if they don't pass.
        
        # If the RPC is disabled, always return a Status 403 (Forbidden)
        if found_RPC.validation == 'disabled':
            abort(403)
                
        # Only do other validation if DataStore and users are included -- NOTE: Any "unknown" validation values are treated like 'none'.
        if self.config['USE_DATASTORE'] and self.config['USE_USERS']:
            if found_RPC.validation == 'any' and not (current_user.is_anonymous or current_user.is_authenticated):
                abort(401) # If the RPC should be executable by any user, including an anonymous one, but there is no authorization or anonymous login, return a Status 401 (Unauthorized)
            elif found_RPC.validation == 'named' and (current_user.is_anonymous or not current_user.is_authenticated):
                abort(401) # Else if the RPC should be executable by any non-anonymous user, but there is no authorization or there is an anonymous login, return a Status 401 (Unauthorized)
            elif found_RPC.validation == 'admin': # Else if the RPC should be executable by any admin user, but there is no admin login or it's an anonymous login...
                if current_user.is_anonymous or not current_user.is_authenticated:
                    abort(401) # If the user is anonymous or no authenticated user is logged in, return Status 401 (Unauthorized).
                elif not current_user.is_admin:
                    abort(403) # Else, if the user is not an admin user, return Status 403 (Forbidden).
                    
        # If we are doing an upload...
        if found_RPC.call_type == 'upload':
            thisfile = request.files['uploadfile'] # Grab the formData file that was uploaded.    
            filename = secure_filename(thisfile.filename) # Extract a sanitized filename from the one we start with.
            uploaded_fname = os.path.join(self.datastore.tempfolder, filename) # Generate a full upload path/file name.
            thisfile.save(uploaded_fname) # Save the file to the uploads directory.
            args.insert(0, uploaded_fname) # Prepend the file name to the args list.
        
        # Show the call of the function.
        callcolor    = ['cyan', 'bgblue']
        successcolor = ['green', 'bgblue']
        failcolor    = ['gray', 'bgred']
        timestr = '[%s]' % sc.now(tostring=True)
        try:    userstr = ' <%s>' % current_user.username
        except: userstr =' <no user>'
        RPCinfo = sc.objdict({'time':timestr, 'user':userstr, 'module':found_RPC.call_func.__module__, 'name':found_RPC.funcname})
        
        if self.config['LOGGING_MODE'] == 'FULL':
            string = '%s%s RPC called: "%s.%s"' % (RPCinfo.time, RPCinfo.user, RPCinfo.module, RPCinfo.name)
            sc.colorize(callcolor, string)
    
        # Execute the function to get the results, putting it in a try block in case there are errors in what's being called. 
        try:
            T = sc.tic()
            result = found_RPC.call_func(*args, **kwargs)
            elapsed = sc.toc(T, output=True)
            if self.config['LOGGING_MODE'] == 'FULL':
                string = '%s%s RPC finished in %0.2f s: "%s.%s"' % (RPCinfo.time, RPCinfo.user, elapsed, RPCinfo.module, RPCinfo.name)
                sc.colorize(successcolor, string)
        except Exception as E:
            shortmsg = str(E)
            exception = traceback.format_exc() # Grab the trackback stack.
            hostname = '|%s| ' % socket.gethostname()
            tracemsg = '%s%s%s Exception during RPC "%s.%s" \nRequest: %s \n%.10000s' % (hostname, RPCinfo.time, RPCinfo.user, RPCinfo.module, RPCinfo.name, request, exception)
            sc.colorize(failcolor, tracemsg) # Post an error to the Flask logger limiting the exception information to 10000 characters maximum (to prevent monstrous sqlalchemy outputs)
            if self.config['SLACK']:
                self.slacknotification(tracemsg)
            if isinstance(E, HTTPException): # If we have a werkzeug exception, pass it on up to werkzeug to resolve and reply to.
                raise E
            code = 500 # Send back a response with status 500 that includes the exception traceback.
            fullmsg = shortmsg + '\n\nException details:\n' + tracemsg
            reply = {'exception':fullmsg} # NB, not sure how to actually access 'traceback' on the FE, but keeping it here for future
            return make_response(robustjsonify(reply), code)
        
        # If we are doing a download, prepare the response and send it off.
        if found_RPC.call_type == 'download':
            if result is None: # If we got None for a result (the full file name), return an error to the client.
                return robustjsonify({'error': 'Could not find resource to download from RPC "%s": result is None' % fn_name})
            elif not sc.isstring(result): # Else, if the result is not even a string (which means it's not a file name as expected)...
                if type(result) is dict and 'error' in result: # If the result is a dict with an 'error' key, then assume we have a custom error that we want the RPC to return to the browser, and do so.
                    return robustjsonify(result)
                else: # Otherwise, return an error that the download RPC did not return a filename.
                    return robustjsonify({'error': 'Download RPC "%s" did not return a filename (result is of type %s)' % (fn_name, type(result))})
            dir_name, file_name = os.path.split(result)  # Pull out the directory and file names from the full file name.
         
            # Make the response message with the file loaded as an attachment.
            response = send_from_directory(dir_name, file_name, as_attachment=True)
            response.status_code = 201  # Status 201 = Created
            response.headers['filename'] = file_name
                
            # Unfortunately, we cannot remove the actual file at this point 
            # because it is in use during the actual download, so we rely on 
            # later cleanup to remove download files.
            return response # Return the response message.
    
        # Otherwise (normal and upload RPCs), 
        else: 
            if found_RPC.call_type == 'upload': # If we are doing an upload....
                os.remove(uploaded_fname) # Erase the physical uploaded file, since it is no longer needed.
            if result is None: # If None was returned by the RPC function, return ''.
                return ''
            else: # Otherwise, convert the result (probably a dict) to JSON and return it.
                output = robustjsonify(result)
                return output
        
        
class ScirisResource(Resource):
    isLeaf = True

    def __init__(self, wsgi):
        self._wsgi = wsgi

    def render(self, request):
        r = self._wsgi.render(request) # Get the WSGI render results (i.e. for Flask app).

        # Keep the client browser from caching Flask response, and set the response as already being "expired."
        request.responseHeaders.setRawHeaders(b'Cache-Control', [b'no-cache', b'no-store', b'must-revalidate'])
        request.responseHeaders.setRawHeaders(b'expires', [b'0'])
        
        # Pass back the WSGI render results.
        return r
    
    
def run_twisted(port=8080, flask_app=None, client_dir=None, do_log=False):
    # Give an error if we pass in no Flask server or client path.
    if (flask_app is None) and (client_dir is None): 
        print('ERROR: Neither client or server are defined.')
        return None
    if do_log: # Set up logging.
        globalLogBeginner.beginLoggingTo([FileLogObserver(sys.stdout, lambda _: formatEvent(_) + "\n")])
    if client_dir is not None: # If there is a client path, set up the base resource.
        base_resource = File(client_dir)
        
    # If we have a flask app...
    if flask_app is not None:
        thread_pool = ThreadPool(maxthreads=30) # Create a thread pool to use with the app.
        wsgi_app = WSGIResource(reactor, thread_pool, flask_app) # Create the WSGIResource object for the flask server.
        if client_dir is None: # If we have no client path, set the WSGI app to be the base resource.
            base_resource = ScirisResource(wsgi_app)
        else:  # Otherwise, make the Flask app a child resource.
            base_resource.putChild('api', ScirisResource(wsgi_app))
        thread_pool.start() # Start the threadpool now, shut it down when we're closing
        reactor.addSystemEventTrigger('before', 'shutdown', thread_pool.stop)

    # Create the site.
    site = Site(base_resource) 
    endpoint = serverFromString(reactor, "tcp:port=" + str(port)) # Create the endpoint we want to listen on, and point it to the site.
    endpoint.listen(site)
    reactor.run() # Start the reactor.
    return None
    
    
    
