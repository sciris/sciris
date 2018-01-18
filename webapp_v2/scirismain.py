"""
scirismain.py -- main code for Sciris users to change to create their web apps
    
Last update: 1/17/18 (gchadder3)
"""

#
# Imports (Block 1)
#

from flask import request, current_app, json, jsonify, send_from_directory
from flask_login import current_user
from werkzeug.utils import secure_filename
import numpy as np
import mpld3
import sys
import os
import re
import copy
import sessionmanager.scirisobjects as sobj
import sessionmanager.datastore as ds
import sessionmanager.user as user
import sessionmanager.project as project

#
# Script code (Block 1)
#

# Get the full path for the loaded sciris repo.  (It in sys path at the 
# beginning because the caller puts it there.)
scirisRepoFullPath = sys.path[0]

# Execute the config.py file to get parameter values we need (directories).
execfile('%s%s%s%s%s' % (scirisRepoFullPath, os.sep, 'sessionmanager', 
    os.sep, 'config.py'))

# If we have a full path for the model directory, load scirismain.py from that.
if os.path.isabs(MODEL_DIR):
    modelDirTarget = MODEL_DIR
    
# Otherwise (we have a relative path), use it (correcting so it is with 
# respect to the sciris repo directory).
else:
    modelDirTarget = '%s%s%s' % (os.pardir, os.sep, MODEL_DIR) 
    
# If we have a full path for the webapp directory, load scirismain.py from that.
if os.path.isabs(WEBAPP_DIR):
    webappDirTarget = WEBAPP_DIR
    
# Otherwise (we have a relative path), use it (correcting so it is with 
# respect to the sciris repo directory).
else:
    webappDirTarget = '%s%s%s' % (os.pardir, os.sep, WEBAPP_DIR) 
    
#
# Imports (Block 2, dependent on config file)
#  
    
# Append the model directory to the path and import needed files.    
sys.path.append(modelDirTarget)
import model

# Append the webapp directory to the path and import needed files.    
sys.path.append(webappDirTarget)
import imp
imp.load_source('ourexceptions', '%s%sexceptions.py' % \
   (webappDirTarget, os.sep))
from ourexceptions import ProjectDoesNotExist, SpreadsheetDoesNotExist

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
        
        # If we have an absolute directory, use it.
        if os.path.isabs(current_app.config['UPLOADS_DIR']):
            ds.uploadsPath = current_app.config['UPLOADS_DIR']
            
        # Otherwise (we have a relative path), use it (correcting so it is with 
        # respect to the sciris repo directory).
        else:
            ds.uploadsPath = '%s%s%s' % (os.pardir, os.sep, 
                current_app.config['UPLOADS_DIR'])  
        
        # Set the file save root path.
        
        # If we have an absolute directory, use it.
        if os.path.isabs(current_app.config['FILESAVEROOT_DIR']):
            ds.fileSaveRootPath = current_app.config['FILESAVEROOT_DIR']
            
        # Otherwise (we have a relative path), use it (correcting so it is with 
        # respect to the sciris repo directory).
        else:
            ds.fileSaveRootPath = '%s%s%s' % (os.pardir, os.sep, 
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
    # Look for an existing users dictionary.
    theUserDictUID = ds.theDataStore.getUIDFromInstance('userdict', 'Users Dictionary')
    
    # Create the user dictionary object.  Note, that if no match was found, 
    # this will be assigned a new UID.
    user.theUserDict = user.UserDict(theUserDictUID)
    
    # If there was a match...
    if theUserDictUID is not None:
        print '>> Loading UserDict from the DataStore.'
        user.theUserDict.loadFromDataStore() 
    
    # Else (no match)...
    else:
        print '>> Creating a new UserDict.'   
        user.theUserDict.addToDataStore()
        user.theUserDict.add(user.testUser)
        user.theUserDict.add(user.testUser2)
        user.theUserDict.add(user.testUser3)

    # Show all of the handles in theDataStore.
    print '>> List of all DataStore handles...'
    ds.theDataStore.showHandles()
    
    # Show all of the users in theUserDict.
    print '>> List of all users...'
    user.theUserDict.show()
    
def init_projects(theApp):
    # Look for an existing ProjectCollection.
    theProjsUID = ds.theDataStore.getUIDFromInstance('projectscoll', 
        'Projects Collection')
    
    # Create the projects collection object.  Note, that if no match was found, 
    # this will be assigned a new UID.    
    project.theProjCollection = project.ProjectCollection(theProjsUID)
    
    # If there was a match...
    if theProjsUID is not None:
        print '>> Loading ProjectCollection from the DataStore.'
        project.theProjCollection.loadFromDataStore() 
    
    # Else (no match)...
    else:
        print '>> Creating a new ProjectCollection.'   
        project.theProjCollection.addToDataStore()
        
        print '>> Starting a demo project.'
        fullFileName = '%s%sgraph1.csv' % (ds.fileSaveRootPath, os.sep)
        theProject = project.Project('Graph 1', user.get_scirisdemo_user(), 
            spreadsheetPath=fullFileName)
        project.theProjCollection.addObject(theProject)
        
        print '>> Starting a second demo project.'
        fullFileName = '%s%sgraph2.csv' % (ds.fileSaveRootPath, os.sep)
        theProject = project.Project('Graph 2', user.get_scirisdemo_user(), 
            spreadsheetPath=fullFileName)
        project.theProjCollection.addObject(theProject)
        
        print '>> Starting a third demo project.'
        fullFileName = '%s%sgraph3.csv' % (ds.fileSaveRootPath, os.sep)
        theProject = project.Project('Graph 3', user.get_scirisdemo_user(), 
            spreadsheetPath=fullFileName)
        project.theProjCollection.addObject(theProject)
        
        print '>> Starting a fourth demo project.'
        theProject = project.Project('Empty graph', user.get_scirisdemo_user(), 
            spreadsheetPath=None)
        project.theProjCollection.addObject(theProject)
        
    # Show what's in the ProjectCollection.    
    project.theProjCollection.show()
    
def init_main(theApp): 
    print '-- Version 2 of the app --'
    
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
            return jsonify(json_sanitize_result(result))

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
        return str(theResult)

    if isinstance(theResult, set):
        return list(theResult)

#    if isinstance(theResult, UUID):
#        return str(theResult)

    return theResult
        
#
# RPC functions
#

def user_change_info(userName, password, displayname, email):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'changeUserInfoRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Make a copy of the current_user.
    theUser = copy.copy(current_user)
    
    # If the password entered doesn't match the current user password, fail.
    if password != theUser.password:
        return 'failure'
       
    # If the username entered by the user is different from the current_user
    # name (meaning they are trying to change the name)...
    if userName != theUser.username:
        # Get any matching user (if any) to the new username we're trying to 
        # switch to.
        matchingUser = user.theUserDict.getUserByUsername(userName)
    
        # If we have a match, fail because we don't want to rename the user to 
        # another existing user.
        if matchingUser is not None:
            return 'failure'
        
        # If a directory exists at the old name, rename it to the new name.
        userOldFileSavePath = '%s%s%s' % (ds.fileSaveRootPath, os.sep, theUser.username)
        userNewFileSavePath = '%s%s%s' % (ds.fileSaveRootPath, os.sep, userName)
        if os.path.exists(userOldFileSavePath):
            os.rename(userOldFileSavePath, userNewFileSavePath)
        
    # Change the user name, display name, email, and instance label.
    theUser.username = userName
    theUser.displayname = displayname
    theUser.email = email
    theUser.instanceLabel = userName
    
    # Update the user in theUserDict.
    user.theUserDict.update(theUser)
    
    # Return success.
    return 'success'

def admin_delete_user(userName):
    # Get the result of doing the normal user.py call.
    callResult = user.admin_delete_user(userName)
    
    # If we had a successful result, do the additional processing.
    if callResult == 'success':
        # Set the directory to the user's private directory.
        userFileSavePath = '%s%s%s' % (ds.fileSaveRootPath, os.sep, userName)
        
        # If the directory exists...
        if os.path.exists(userFileSavePath):
            # Get the total list of entries in the user's directory.
            allEntries = os.listdir(userFileSavePath)
            
            # Remove each of the files.
            for fileName in allEntries:
                # Create a full file name for the file.
                fullFileName = '%s%s%s' % (userFileSavePath, os.sep, fileName)

                # Remove the file.                
                os.remove(fullFileName)
            
            # Remove the directory itself.
            os.rmdir(userFileSavePath)
        
    # Return the callResult.    
    return callResult
    
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
    return {'graph': mpld3.fig_to_dict(graphData)}

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