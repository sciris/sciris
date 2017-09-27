"""
api.py -- script for setting up a flask server
    
Last update: 9/27/17 (gchadder3)
"""

#
# Imports
#

from flask import Flask, request, abort, json, jsonify, make_response, \
    send_from_directory
from flask_login import LoginManager, current_user, login_required
from werkzeug.utils import secure_filename
from functools import wraps
import traceback
import os
import sys
import logging
import main
import datastore
import user

#
# Globals
#

# Create the Flask app.
app = Flask(__name__)

# Create the LoginManager.
login_manager = LoginManager()
   
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
    return user.theUserDict.getUserByUID(userid)

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
    return doRPC('normal', 'user', request.method)

# Define the /api/user/logout endpont for RPCs for user logout.
@app.route('/api/user/logout', methods=['POST'])
@report_exception_decorator
def logoutRPC():
    return doRPC('normal', 'user', request.method)

# Define the /api/user/current endpont for RPCs for returning the current 
# user's user information.
@app.route('/api/user/current', methods=['GET'])
@report_exception_decorator
@login_required
def currentUserRPC():
    return doRPC('normal', 'user', request.method)

# Define the /api/user/list endpont for RPCs for returning (for admin users 
# only) a summary of all of the users. 
@app.route('/api/user/list', methods=['GET'])
@report_exception_decorator
@verify_admin_request_decorator
def userListRPC():
    return doRPC('normal', 'user', request.method)

# Define the /api/user/register endpont for RPCs for user registration.
@app.route('/api/user/register', methods=['POST'])
@report_exception_decorator
def registrationRPC():
    return doRPC('normal', 'user', request.method)

# Define the /api/user/changeinfo endpont for RPCs for user changing their 
# (non-password) info.
@app.route('/api/user/changeinfo', methods=['POST'])
@report_exception_decorator
@login_required
def changeUserInfoRPC():
    return doRPC('normal', 'user', request.method)

# Define the /api/user/changepassword endpont for RPCs for user changing 
# their password.
@app.route('/api/user/changepassword', methods=['POST'])
@report_exception_decorator
@login_required
def changePasswordRPC():
    return doRPC('normal', 'user', request.method)

@app.route('/api/user/<username>', methods=['GET', 'POST'])
@report_exception_decorator
@verify_admin_request_decorator
def specificUserRPC(username):
    return doRPC('normal', 'user', request.method, username=username)

# Define the /api/procedure endpoint for normal RPCs.
@app.route('/api/procedure', methods=['POST'])
@report_exception_decorator
@login_required
def normalRPC():
    return doRPC('normal', 'main', request.method)

# Define the /api/procedure endpoint for normal RPCs that don't require a 
# current valid login.
@app.route('/api/publicprocedure', methods=['POST'])
@report_exception_decorator
def publicNormalRPC():
    return doRPC('normal', 'main', request.method)

# Define the /api/download endpoint for RPCs that download a file to the 
# client.
@app.route('/api/download', methods=['POST'])
@report_exception_decorator
@login_required
def downloadRPC():
    return doRPC('download', 'main', request.method)

# Define the /api/upload endpoint for RPCs that work from uploaded files.
@app.route('/api/upload', methods=['POST'])
@report_exception_decorator
@login_required
def uploadRPC():
    return doRPC('upload', 'main', request.method)  
  
# Do the meat of the RPC calls, passing args and kwargs to the appropriate 
# function in the appropriate handler location.
def doRPC(rpcType, handlerLocation, requestMethod, username=None):
    # If we are doing an upload, pull the RPC information out of the request 
    # form instead of the request data.
    if rpcType == 'upload':
        # Pull out the function name, args, and kwargs
        fn_name = request.form.get('funcname')
        args = json.loads(request.form.get('args', "[]"))
        kwargs = json.loads(request.form.get('kwargs', "{}"))
    
    # Otherwise (normal and download RPCs), pull the RPC information from the 
    # request data.
    else:
        # Convert the request.data JSON to a Python dict, and pull out the 
        # function name, args, and kwargs.  
        if requestMethod in ['POST', 'PUT']:
            reqdict = json.loads(request.data)
        elif requestMethod == 'GET':
            reqdict = request.args
        fn_name = reqdict['funcname']
        args = reqdict.get('args', [])
        # Insert the username as a the first argument if it is passed in not
        # None.
        if username is not None:
            args.insert(0, username)
        kwargs = reqdict.get('kwargs', {})
        
    # Check to see whether the function to be called exists and get it ready 
    # to call in func if it's found.
    if handlerLocation == 'main':
        funcExists = hasattr(main, fn_name)
        print('>> Checking RPC function "main.%s" -> %s' % (fn_name, funcExists))
        if funcExists:
            func = getattr(main, fn_name)
    elif handlerLocation == 'user':
        funcExists = hasattr(user, fn_name)
        print('>> Checking RPC function "user.%s" -> %s' % (fn_name, funcExists))
        if funcExists:
            func = getattr(user, fn_name)        
    else:
        return jsonify({'error': 
            'Attempted to call RPC function in non-existent handler location \'%s\'' \
                % handlerLocation}) 
    
    # If the function doesn't exist, return an error to the client saying it 
    # doesn't exist.
    if not funcExists:
        return jsonify({'error': 
            'Attempted to call non-existent RPC function \'%s\'' % fn_name}) 
    
    # If we are doing an upload.
    if rpcType == 'upload':
        # Grab the formData file that was uploaded.    
        file = request.files['uploadfile']
        
        # Grab the filename of this file, and generate the full upload path / 
        # filename.
        filename = secure_filename(file.filename)
        uploaded_fname = os.path.join(datastore.uploadsPath, filename)
        
        # Save the file to the uploads directory.
        file.save(uploaded_fname)
        
        # Prepend the file name to the args list.
        args.insert(0, uploaded_fname)
        
    # Show the call of the function.    
    print('>> Calling RPC function "%s.%s"' % (handlerLocation, fn_name))
    
    # Execute the function to get the results.
    result = func(*args, **kwargs)   
     
    # If we are doing a download, prepare the response and send it off.
    if rpcType == 'download':
        # If we got None for a result (the full file name), return an error to 
        # the client.
        if result is None:
            return jsonify({'error': 'Could not find requested resource'})
    
        # Pull out the directory and file names from the full file name.
        dirName, fileName = os.path.split(result)
         
        # Make the response message with the file loaded as an attachment.
        response = send_from_directory(dirName, fileName, as_attachment=True)
        response.status_code = 201  # Status 201 = Created
        response.headers['filename'] = fileName
        
        # Return the response message.
        return response
    
    # Otherwise (normal and upload RPCs), 
    else:
        # If None was returned by the RPC function, return ''.
        if result is None:
            return ''
        
        # Otherwise, convert the result (probably a dict) to JSON and return it.
        else:
            return jsonify(result)

# 
# Script code
#

# Try to load the config file.  This may fail, so output a warning message if
# it does.
errormsg = 'Could not load Sciris configuration file\n'
errormsg += 'Please ensure that you have copied server/config_example.py to server/config.py\n'
errormsg += 'Note that this is NOT done automatically'
try: # File exists
    app.config.from_pyfile('config.py')
except: # File doesn't exist
    raise Exception(errormsg)

# Initialize the logger.
init_logger()

# Configure app for login with the LoginManager.
login_manager.init_app(app)

# Initialize files, paths, and directories.
print '>> Initializing files, paths, and directories...'
main.init_filepaths(app)

# Initialize the DataStore.
print '>> Initializing the data store...'
main.init_datastore(app)

# Initialize the users.
print '>> Initializing the users data...'
main.init_users(app)

# The following code just gets called if we are running this standalone.
if __name__ == "__main__":
    app.run(threaded=True, debug=True, use_debugger=False)