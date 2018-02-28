"""
api.py -- script for setting up a flask server
    
Last update: 1/15/18 (gchadder3)
"""

#
# Imports (Block 1)
#

from flask import Flask, request, abort, jsonify, make_response
from flask_login import LoginManager, current_user, login_required
from functools import wraps
import traceback
import sys
import logging
import os

#
# Globals
#

def makeapp():

    # Create the Flask app.
    app = Flask(__name__)
    
    # Create the LoginManager.
    login_manager = LoginManager()
    
    #
    # Script code (Block 1)
    #
    
    # Get the full path for the loaded sciris repo.  (It in sys path at the 
    # beginning because the caller puts it there.)
    scirisRepoFullPath = sys.path[0]
    
    # Try to load the config file.  This may fail, so output a warning message if
    # it does.
    errormsg = 'Could not load Sciris configuration file\n'
    errormsg += 'Please ensure that you have copied server/config_example.py to server/config.py\n'
    errormsg += 'Note that this is NOT done automatically'
    try: # File exists
        app.config.from_pyfile('config.py')
    except: # File doesn't exist
        raise Exception(errormsg)
        
    #
    # Imports (Block 2, dependent on config file)
    #
    
    print('Remove hardcoding for scirismain.py')
    
    # If we have a full path for the webapp directory, load scirismain.py from that.
    if os.path.isabs(app.config['WEBAPP_DIR']):
        scirismainTarget = '%s%s%s' % (app.config['WEBAPP_DIR'], os.sep, 
            'main.py')
        
    # Otherwise (we have a relative path), use it (correcting so it is with 
    # respect to the sciris repo directory).
    else:
        scirismainTarget = '%s%s%s%s%s' % (os.pardir, os.sep, 
            app.config['WEBAPP_DIR'], os.sep, 'main.py')  
        
    # Do the import.
    import imp
    scirismain = imp.load_source('scirismain', scirismainTarget)
    
    #
    # Functions
    #
    
    # This function initializes the Flask app's logger.
    def init_logger():
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]'
        ))
        app.logger.addHandler(stream_handler)
        app.logger.setLevel(logging.DEBUG)
        
    # This function gets called when authentication gets done by Flask-Login.  
    # userid is the user ID pulled out of the session cookie the browser passes in 
    # during an HTTP request.  The function returns the User object that matches 
    # this, so that the user data may be used to authenticate (for example their 
    # rights to have admin access).  The return sets the Flask-Login current_user 
    # value.
    @login_manager.user_loader
    def load_user(userid):
        # Return the matching user (if any).
        return scirismain.user.theUserDict.getUserByUID(userid)
    
    # Decorator function which allows any exceptions made by the RPC calls to be 
    # trapped and return in the response message.
    def report_exception_decorator(api_call):
        @wraps(api_call)
        def _report_exception(*args, **kwargs):
            from werkzeug.exceptions import HTTPException
            try:
                return api_call(*args, **kwargs)
            except Exception as e:
                exception = traceback.format_exc()
                # limiting the exception information to 10000 characters maximum
                # (to prevent monstrous sqlalchemy outputs)
                app.logger.error("Exception during request %s: %.10000s" % (request, exception))
                if isinstance(e, HTTPException):
                    raise
                code = 500
                reply = {'exception': exception}
                return make_response(jsonify(reply), code)
            
        return _report_exception
    
    # Decorator function for checking that the user that is logged in has admin
    # rights.
    def verify_admin_request_decorator(api_call):
        @wraps(api_call)
        def _verify_admin_request(*args, **kwargs):
            if not current_user.is_anonymous and \
                current_user.is_authenticated and current_user.is_admin:
                app.logger.debug("admin_user: %s %s %s" % 
                    (current_user.username, current_user.password, 
                     current_user.email))
                return api_call(*args, **kwargs)
            else:
                abort(403)
    
        return _verify_admin_request
    
    # Responder for (unused) /api root endpoint.
    @app.route('/api', methods=['GET'])
    def root():
        """ API root, nothing interesting here """
        return 'Sciris API v.0.0.0'
    
    # Define the /api/user/login endpont for RPCs for user login.
    @app.route('/api/user/login', methods=['POST'])
    @report_exception_decorator
    def loginRPC():
        return scirismain.doRPC('normal', 'user', request.method)
    
    # Define the /api/user/logout endpont for RPCs for user logout.
    @app.route('/api/user/logout', methods=['POST'])
    @report_exception_decorator
    def logoutRPC():
        return scirismain.doRPC('normal', 'user', request.method)
    
    # Define the /api/user/current endpont for RPCs for returning the current 
    # user's user information.
    @app.route('/api/user/current', methods=['GET'])
    @report_exception_decorator
    @login_required
    def currentUserRPC():
        return scirismain.doRPC('normal', 'user', request.method)
    
    # Define the /api/user/list endpont for RPCs for returning (for admin users 
    # only) a summary of all of the users. 
    @app.route('/api/user/list', methods=['GET'])
    @report_exception_decorator
    @verify_admin_request_decorator
    def userListRPC():
        return scirismain.doRPC('normal', 'user', request.method)
    
    # Define the /api/user/register endpont for RPCs for user registration.
    @app.route('/api/user/register', methods=['POST'])
    @report_exception_decorator
    def registrationRPC():
        return scirismain.doRPC('normal', 'user', request.method)
    
    # Define the /api/user/changeinfo endpont for RPCs for user changing their 
    # (non-password) info.
    @app.route('/api/user/changeinfo', methods=['POST'])
    @report_exception_decorator
    @login_required
    def changeUserInfoRPC():
        return scirismain.doRPC('normal', 'user', request.method)
    
    # Define the /api/user/changepassword endpont for RPCs for user changing 
    # their password.
    @app.route('/api/user/changepassword', methods=['POST'])
    @report_exception_decorator
    @login_required
    def changePasswordRPC():
        return scirismain.doRPC('normal', 'user', request.method)
    
    @app.route('/api/user/<username>', methods=['GET', 'POST'])
    @report_exception_decorator
    @verify_admin_request_decorator
    def specificUserRPC(username):
        return scirismain.doRPC('normal', 'user', request.method, username=username)
    
    # Define the /api/procedure endpoint for normal RPCs.
    @app.route('/api/procedure', methods=['POST'])
    @report_exception_decorator
    @login_required
    def normalRPC():
        return scirismain.doRPC('normal', 'scirismain', request.method)
    
    # Define the /api/procedure endpoint for normal RPCs that don't require a 
    # current valid login.
    @app.route('/api/publicprocedure', methods=['POST'])
    @report_exception_decorator
    def publicNormalRPC():
        return scirismain.doRPC('normal', 'scirismain', request.method)
    
    # Define the /api/download endpoint for RPCs that download a file to the 
    # client.
    @app.route('/api/download', methods=['POST'])
    @report_exception_decorator
    @login_required
    def downloadRPC():
        return scirismain.doRPC('download', 'scirismain', request.method)
    
    # Define the /api/upload endpoint for RPCs that work from uploaded files.
    @app.route('/api/upload', methods=['POST'])
    @report_exception_decorator
    @login_required
    def uploadRPC():
        return scirismain.doRPC('upload', 'scirismain', request.method)  
    
    # 
    # Script code (main)
    #
    
    # Initialize the logger.
    init_logger()
    
    # Configure app for login with the LoginManager.
    login_manager.init_app(app)
    
    # Initialize files, paths, and directories.
    print '>> Initializing files, paths, and directories...'
    scirismain.init_filepaths(app)
    
    # Initialize the DataStore.
    print '>> Initializing the data store...'
    scirismain.init_datastore(app)
    
    # Initialize the users.
    print '>> Initializing the users data...'
    scirismain.init_users(app)
    
    # Initialize the projects.
    print '>> Initializing the projects data...'
    scirismain.init_projects(app)
    
    # Perform other scirismain.py-specific initialization.
    print '>> Doing other scirismain-specific initialization...'
    scirismain.init_main(app)
    
    return app

def runapp():
    app = makeapp()
    app.run(threaded=True, debug=True, use_debugger=False)
    return None
    

# The following code just gets called if we are running this standalone.
if __name__ == "__main__":
    runapp()