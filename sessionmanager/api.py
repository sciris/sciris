"""
api.py -- script for setting up a flask server
    
Last update: 9/1/17 (gchadder3)
"""

#
# Imports
#

from flask import Flask, request, json, jsonify, current_app, make_response, \
    send_from_directory
from werkzeug.utils import secure_filename
from functools import wraps
import traceback
import os
import rpcs

#
# Globals
#

# Create the app.
app = Flask(__name__)

#
# Functions
#

# Return the request.data JSON converted into a Python dict.
def getRequestDict():
    return json.loads(request.data)

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
            current_app.logger.error("Exception during request %s: %.10000s" % (request, exception))
            if isinstance(e, HTTPException):
                raise
            code = 500
            reply = {'exception': exception}
            return make_response(jsonify(reply), code)
        
    return _report_exception

# Responder for (unused) /api root endpoint.
@app.route('/api', methods=['GET'])
def root():
    """ API root, nothing interesting here """
    return 'Sciris API v.0.0.0'

def commonRPCstart():
    # Initialize the session.
    rpcs.init_session()
    
    # Create an empty dict for the RPC call.
    callDict = {}
    
    # Convert the request.data JSON to a Python dict, and pull out the 
    # function name, args, and kwargs.
    reqdict = getRequestDict()
    fn_name = reqdict['name']
    callDict['args'] = reqdict.get('args', [])
    callDict['kwargs'] = reqdict.get('kwargs', {})
    callDict['func'] = None     # start with no function found
    callDict['result'] = None   # start with no result
    
    # Check to see whether the function to be called exists.
    funcExists = hasattr(rpcs, fn_name)
    print('>> Checking RPC function "rpcs.%s" -> %s' % (fn_name, funcExists))
    
    # If the function doesn't exist, prepare an error for the client saying it 
    # doesn't exist.
    if not funcExists:
        callDict['result'] = {'error': 
            'Attempted to call non-existent RPC function %s' % fn_name}
    
    # Otherwise, extract the rpcs module's function.
    else:
        callDict['func'] = getattr(rpcs, fn_name)
    
    # Return the callDict.
    return callDict

# Define the /api/procedure endpoint for normal RPCs.
@app.route('/api/procedure', methods=['POST'])
@report_exception_decorator
def normalRPC():
    """
    POST-data:
        'name': string name of function in handler
        'args': list of arguments for the function
        'kwargs: dictionary of keyword arguments
    """
    # Perform the starting RPC functionality, including checking for failure
    # to find a matching RPC function.
    callDict = commonRPCstart()
    
    # If we have no result yet, call the function, passing in args and kwargs.
    if callDict['result'] is None:
        callDict['result'] = callDict['func'](*(callDict['args']), 
            **(callDict['kwargs']))

    # If None was returned by the RPC function, return ''.
    if callDict['result'] is None:
        return ''
    
    # Otherwise, convert the result (probably a dict) to JSON and return it.
    else:
        return jsonify(callDict['result'])
   
# Define the /api/download endpoint for normal RPCs.
@app.route('/api/download', methods=['POST'])
@report_exception_decorator
def downloadRPC():
    """
    POST-data:
        'name': string name of function in handler
        'args': list of arguments for the function
        'kwargs: dictionary of keyword arguments
    """
    # Perform the starting RPC functionality, including checking for failure
    # to find a matching RPC function.
    callDict = commonRPCstart()
    
    # If we have no result yet...
    if callDict['result'] is None:
        # Call the function, passing in args and kwargs, and hopefully
        fullFileName = callDict['func'](*(callDict['args']), 
            **(callDict['kwargs']))
        
        # If we got None for a fullFileName, return an error to the client.
        if fullFileName is None:
            return jsonify({'error': 'Could not find requested resource'})
    
        # Pull out the directory and file names from the full file name.
        dirName, fileName = os.path.split(fullFileName)
         
        # Make the response message with the file loaded as an attachment.
        response = send_from_directory(dirName, fileName, as_attachment=True)
        response.status_code = 201  # Status 201 = Created
        response.headers['filename'] = fileName
        
        # Return the response message.
        return response
        
    # Otherwise (we had an error), return the result.
    else:
        return jsonify(callDict['result'])

# Define the /api/upload endpoint.
@app.route('/api/upload', methods=['POST'])
@report_exception_decorator
def uploadRPC():
    """
    POST-data:
        'funcname': string name of function in handler
        'args': list of arguments for the function
        'kwargs: dictionary of keyword arguments
    """   
    # Initialize the session.
    rpcs.init_session()

    # Pull out the function name, args, and kwargs
    fn_name = request.form.get('funcname')
    args = json.loads(request.form.get('args', "[]"))
    kwargs = json.loads(request.form.get('kwargs', "{}"))
    
    # Check to see whether the function to be called exists.
    funcExists = hasattr(rpcs, fn_name)
    print('>> Checking RPC function "rpcs.%s" -> %s' % (fn_name, funcExists))
    
    # If the function doesn't exist, return an error to the client saying it 
    # doesn't exist.
    if not funcExists:
        return jsonify({'error': 
            'Attempted to call non-existent RPC function %s' % fn_name}) 
            
    # Get the function from the rpcs module.        
    func = getattr(rpcs, fn_name)
    
    # Grab the formData file that was uploaded.    
    file = request.files['uploadfile']
    
    # Grab the filename of this file, and generate the full upload path / 
    # filename.
    filename = secure_filename(file.filename)
    uploaded_fname = os.path.join(rpcs.uploadsPath, filename)
    
    # Save the file to the uploads directory.
    file.save(uploaded_fname)
    
    # Prepend the file name to the args list.
    args.insert(0, uploaded_fname)
    
    # Execute the function to get the results.
    result = func(*args, **kwargs)    
    
    # If None was returned by the RPC function, return ''.
    if result is None:
        return ''
    
    # Otherwise, convert the result (probably a dict) to JSON and return it.
    else:
        return jsonify(result)

# 
# Script code
#

if __name__ == "__main__":
    app.run(threaded=True, debug=True, use_debugger=False)