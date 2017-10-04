"""
scirismain.py -- main code for Sciris users to change to create their web apps
    
Last update: 10/3/17 (gchadder3)
"""

#
# Imports
#

import sys
sys.path.append('../sessionmanager')
import model
import scirisobjects as sobj
import datastore as ds
import imp
user = imp.load_source('user', '../sessionmanager/user.py')
from flask import request, current_app, json, jsonify, send_from_directory
from flask_login import current_user
from werkzeug.utils import secure_filename
import mpld3
import os
import re
import uuid

#
# Classes
# NOTE: These probably should end up in other files (if I keep them).
#

# Wraps a Matplotlib figure displayable in the GUI.
class GraphFigure(object):
    def __init__(self, theFigure):
        self.theFigure = theFigure

# Wraps a collection of data.
class DataCollection(object):
    def __init__(self, dataObj):
        self.dataObj = dataObj

#
# Initialization functions 
#
        
def init_filepaths(theApp):
    # Set the Sciris root path to the parent of the current directory, the 
    # latter being the sciris/bin directory since that is where the code is 
    # executed from.
    ds.scirisRootPath = os.path.abspath(os.pardir)
    
    #  Using the current_app set to the passed in app..
    with theApp.app_context():
        # Set the uploads path.
        ds.uploadsPath = '%s%s%s' % (ds.scirisRootPath, os.sep, 
            current_app.config['UPLOADS_DIR'])
        
        # Set the file save root path.
        ds.fileSaveRootPath = '%s%s%s' % (ds.scirisRootPath, os.sep, 
            current_app.config['FILESAVEROOT_DIR'])
         
    # If the datafiles path doesn't exist yet...
    if not os.path.exists(ds.fileSaveRootPath):
        # Create datafiles directory.
        os.mkdir(ds.fileSaveRootPath)
        
        # Create an uploads subdirectory of this.
        os.mkdir(ds.uploadsPath)
        
        # Create the fake data for scatterplots.
        sd = model.ScatterplotData(model.makeUniformRandomData(50))
        fullFileName = '%s%sgraph1.csv' % (ds.fileSaveRootPath, os.sep)
        sd.saveAsCsv(fullFileName)
        
        sd = model.ScatterplotData(model.makeUniformRandomData(50))
        fullFileName = '%s%sgraph2.csv' % (ds.fileSaveRootPath, os.sep)
        sd.saveAsCsv(fullFileName)
        
        sd = model.ScatterplotData(model.makeUniformRandomData(50))
        fullFileName = '%s%sgraph3.csv' % (ds.fileSaveRootPath, os.sep)
        sd.saveAsCsv(fullFileName) 
        
def init_datastore(theApp):
    # Create the DataStore object, setting up Redis.
    with theApp.app_context():
        ds.theDataStore = ds.DataStore(redisDbURL=current_app.config['REDIS_URL'])

    # Load the DataStore state from disk.
    ds.theDataStore.load()
    
    #ds.theDataStore.deleteAll()
    
def init_users(theApp):
    # Create the user dictionary object.
    with theApp.app_context():
        theUserDictUID = uuid.UUID(current_app.config['USERDICT_UUID'])
    user.theUserDict = user.UserDict(theUserDictUID)
    
    #user.theUserDict.deleteFromDataStore()
    # Load (from DataStore) or create (anew) the user dictionary.
    if user.theUserDict.inDataStore():
        print '>> Loading UserDict from the DataStore.'
        user.theUserDict.loadFromDataStore()
    else:
        print '>> Creating a new UserDict.'   
        user.theUserDict.addToDataStore()
        user.theUserDict.add(user.testUser)
        user.theUserDict.add(user.testUser2)     

    # Show all of the handles in theDataStore.
    print '>> List of all DataStore handles...'
    ds.theDataStore.showHandles()
    
    # Show all of the users in theUserDict.
    print '>> List of all users...'
    user.theUserDict.show()
    
#
# Other functions
#

def get_saved_scatterplotdata_file_path(spdName):
    # Set the directory to the user's private directory.
    userFileSavePath = '%s%s%s' % (ds.fileSaveRootPath, os.sep, 
        current_user.username)
    
    # Create a full file name for the file.
    fullFileName = '%s%s%s.csv' % (userFileSavePath, os.sep, spdName)
    
    # If the file is there, return the path name; otherwise, return None.
    if os.path.exists(fullFileName):
        return fullFileName
    else:
        return None
    


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
        
    # Check to see if the function exists here in scirismain.py and get it 
    # ready to call if it is.
    funcExists = hasattr(sys.modules[__name__], fn_name)
    print('>> Checking RPC function "scirismain.%s" -> %s' % (fn_name, funcExists))
    if funcExists:
        func = getattr(sys.modules[__name__], fn_name)
        
    # Otherwise (it's not in scirismain.py), if the handlerLocation is 'user'...
    elif handlerLocation == 'user':
        # Check to see if there is a match.
        funcExists = hasattr(user, fn_name)
        print('>> Checking RPC function "user.%s" -> %s' % (fn_name, funcExists))
        
        # If there is a match, get the function ready.
        if funcExists:
            func = getattr(user, fn_name)
            
        # Otherwise, return an error.
        else:
            return jsonify({'error': 
                'Attempted to call RPC function in non-existent handler location \'%s\'' \
                    % handlerLocation})             
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
        uploaded_fname = os.path.join(ds.uploadsPath, filename)
        
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
# RPC functions
#
        
def list_saved_scatterplotdata_resources():
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'normalRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Set the directory to the user's private directory.
    userFileSavePath = '%s%s%s' % (ds.fileSaveRootPath, os.sep, 
        current_user.username)
    
    # If the directory doesn't exist, return an empty list.
    if not os.path.exists(userFileSavePath):
        return []
    
    # Get the total list of entries in the user's directory.
    allEntries = os.listdir(userFileSavePath)
    
    # Extract just the entries that are .csv files.
    fTypeEntries = [entry for entry in allEntries if re.match('.+\.csv$', entry)]
    
    # Truncate the .csv suffix from each entry.
    truncEntries = [re.sub('\.csv$', '', entry) for entry in fTypeEntries]
    
    # Return the truncated entries.
    return truncEntries

def get_saved_scatterplotdata_graph(spdName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'normalRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Look for a match of the resource, and if we don't find it, return
    # an error.
    fullFileName = get_saved_scatterplotdata_file_path(spdName)
    if fullFileName is None:
        return {'error': 'Cannot find resource \'%s\'' % spdName}
    
    # Create a ScatterplotData object.
    spd = model.ScatterplotData()

    # Load the data for this from the csv file.
    spd.loadFromCsv(fullFileName)
    
    # Generate a matplotib graph for display.
    graphData = spd.plot()
  
    # Return the dictionary representation of the matplotlib figure.
    return mpld3.fig_to_dict(graphData) 

def delete_saved_scatterplotdata_graph(spdName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'normalRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Look for a match of the resource, and if we don't find it, return
    # an error.
    fullFileName = get_saved_scatterplotdata_file_path(spdName)
    if fullFileName is None:
        return {'error': 'Cannot find resource \'%s\'' % spdName}
    
    # Delete the file.
    os.remove(fullFileName)
    
    # Return success.
    return 'success'
    
def download_saved_scatterplotdata(spdName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'downloadRPC':
        return {'error': 'Unauthorized RPC'}
    
    return get_saved_scatterplotdata_file_path(spdName)

def upload_scatterplotdata_from_csv(fullFileName, spdName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'uploadRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Set the directory to the user's private directory.
    userFileSavePath = '%s%s%s' % (ds.fileSaveRootPath, os.sep, 
        current_user.username)
    
    # If the directory doesn't exist yet, make it.
    if not os.path.exists(userFileSavePath):
        os.mkdir(userFileSavePath)
    
    # Pull out the directory and file names from the full file name.
    dirName, fileName = os.path.split(fullFileName)
    
    # Create a new destination for the file in the datafiles directory.
    newFullFileName = '%s%s%s' % (userFileSavePath, os.sep, fileName)
    
    # If the new file already exists, return a failure.
    if os.path.exists(newFullFileName):
        return {'error': 'Resource \'%s\' already on server' % spdName}        
    
    # Move the file into the datafiles directory.
    os.rename(fullFileName, newFullFileName)
    
    # Return the new file name.
    #return newFullFileName

    # Return the resource we added (i.e. stripping off the full filename).
    return re.sub('\.csv$', '', fileName)