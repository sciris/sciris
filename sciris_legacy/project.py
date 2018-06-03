"""
project.py -- code related to Sciris project management
    
Last update: 3/27/18 (gchadder3)
"""

#
# Imports
#

from flask import request
from flask_login import current_user
import datetime
import dateutil
import dateutil.tz
import uuid
import scirisobjects as sobj
import datastore as ds
import user
import os
from zipfile import ZipFile
from exceptions import ProjectDoesNotExist, SpreadsheetDoesNotExist

#
# Globals
#

# The ProjectCollection object for all of the app's projects.  Gets 
# initialized by and loaded by init_projects().
theProjCollection = None

#
# Classes
#

# At the moment (3/8/18) I'm unsure of whether this class should exist at all. 
# A Sciris user will need to create a ProjectSO class in webapp/main.py to 
# wrap whatever Project class or project-relevant data is specific to the app.
# The question is whether this class here can be made into a useful collection 
# of reusable code so that it is worth subclassing.  Another question is 
# whether the more complicated or simpler version of the class below should 
# be developed from.

# more complicated, recent version...
class ProjectSOBase(sobj.ScirisObject):
    """
    Base class for a Sciris project record wrapper.
    
    Methods:
        __init__(name: str, ownerUID: UUID, theUID: UUID [None], 
            spreadsheetPath: str [None]): void -- constructor            
        updateName(newName: str): void -- change the project name to newName
        updateSpreadsheet(spreadsheetPath: str): void -- change the 
            spreadsheet the project is using
            
        saveToPrjFile(dirPath: str, saveResults: bool [False]) -- save the 
            project to a .prj file and return the full path
            
        loadFromCopy(otherObject): void -- assuming otherObject is another 
            object of our type, copy its contents to us (calls the 
            ScirisObject superclass version of this method also)             
        show(): void -- print the contents of the object
        getUserFrontEndRepr(): dict -- get a JSON-friendly dictionary 
            representation of the object state the front-end uses for non-
            admin purposes 
            
        saveAsFile(loadDir: str): str -- given a load dictionary, save the 
            project in a file there and return the file name
            
    Attributes:
        uid (UUID) -- the UID of the Project
        ownerUID (UUID) -- the UID of the User that owns the Project        
        name (str) -- the Project's name
        spreadsheetPath (str) -- the full path name for the Excel spreadsheet
        createdTime (datetime.datetime) -- the time that the Project was 
            created
        updatedTime (datetime.datetime) -- the time that the Project was last 
            updated
        dataUploadTime (datetime.datetime) -- the time that the Project's 
            spreadsheet was last updated
        
    Usage:
        >>> theProj = Project('myproject', uuid.UUID('12345678123456781234567812345672'), uuid.UUID('12345678123456781234567812345678'))                    
    """ 
    
    def  __init__(self, name, ownerUID, theUID=None, spreadsheetPath=None):
        # Make sure owner has a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        validOwnerUID = sobj.getValidUUID(ownerUID)
        
        # If we have a valid UUID...
        if validOwnerUID is not None:  
            # Set the owner (User) UID.
            self.ownerUID = validOwnerUID
                       
            # If a UUID was passed in...
            if theUID is not None:
                # Make sure the argument is a valid UUID, converting a hex text to a
                # UUID object, if needed.        
                validUID = sobj.getValidUUID(theUID) 
                
                # If a validUID was found, use it.
                if validUID is not None:
                    self.uid = validUID
                # Otherwise, generate a new random UUID using uuid4().
                else:
                    self.uid = uuid.uuid4()
            # Otherwise, generate a new random UUID using uuid4().
            else:
                self.uid = uuid.uuid4()
                
            # Set the project name.
            self.name = name
            
            # Set the spreadsheetPath.
            self.spreadsheetPath = spreadsheetPath
                                
            # Set the creation time for now.
            self.createdTime = now_utc()
            
            # Set the updating time to None.
            self.updatedTime = None
            
            # Set the spreadsheet upload time to None.
            self.dataUploadTime = None
            
            # If we have passed in a spreadsheet path...
            if self.spreadsheetPath is not None:
                # Set the data spreadsheet upload time for now.
                self.dataUploadTime = now_utc()
                
            # Set the type prefix to 'user'.
            self.typePrefix = 'project'
            
            # Set the file suffix to '.usr'.
            self.fileSuffix = '.prj'
            
            # Set the instance label to the username.
            self.instanceLabel = name   
          
    def updateName(self, newName):
        # Set the project name.
        self.name = newName
        self.instanceLabel = newName
        
        # Set the updating time to now.
        self.updatedTime = now_utc()
        
    def updateSpreadsheet(self, spreadsheetPath):
        # Set the spreadsheetPath from what's passed in.
        self.spreadsheetPath = spreadsheetPath
        
        # Set the data spreadsheet upload time for now.
        self.dataUploadTime = now_utc()
        
        # Set the updating time to now.
        self.updatedTime = now_utc()
        
#    def saveToPrjFile(self, dirPath, saveResults=False):
#        # Create a filename containing the project name followed by a .prj 
#        # suffix.
#        fileName = '%s.prj' % self.name
#        
#        # Generate the full file name with path.
#        fullFileName = '%s%s%s' % (dirPath, os.sep, fileName)
#        
#        # Write the object to a Gzip string pickle file.
#        objectToGzipStringPickleFile(fullFileName, self)
#        
#        # Return the full file name.
#        return fullFileName

    def loadFromCopy(self, otherObject):
        if type(otherObject) == type(self):
            # Do the superclass copying.
            super(Project, self).loadFromCopy(otherObject)
            
            self.ownerUID = otherObject.ownerUID
            self.name = otherObject.name
            self.spreadsheetPath = otherObject.spreadsheetPath
            self.createdTime = otherObject.createdTime
            self.updatedTime = otherObject.updatedTime
            self.dataUploadTime = otherObject.dataUploadTime
            
            # Copy the owner UID.
            self.ownerUID = otherObject.ownerUID
            
    def show(self):
        # Show superclass attributes.
        super(Project, self).show()  
        
        print '---------------------'

        print 'Owner User UID: %s' % self.ownerUID.hex
        print 'Project Name: %s' % self.name
        print 'Spreadsheet Path: %s' % self.spreadsheetPath
        print 'Creation Time: %s' % self.createdTime
        print 'Update Time: %s' % self.updatedTime
        print 'Data Upload Time: %s' % self.dataUploadTime
        
    def getUserFrontEndRepr(self):
        objInfo = {
            'project': {
                'id': self.uid,
                'name': self.name,
                'userId': self.ownerUID,
                'spreadsheetPath': self.spreadsheetPath,
                'creationTime': self.createdTime,
                'updatedTime': self.updatedTime,
                'dataUploadTime': self.dataUploadTime
            }
        }
        return objInfo
    
#    def saveAsFile(self, loadDir):
#        # Save the project in the file.
#        self.theProject.saveToPrjFile(loadDir, saveResults=True)
#        
#        # Return the filename (not the full one).
#        return self.theProject.name + ".prj" 

# simpler version of the class used in nutritiongui...
#class ProjectSOBase(sobj.ScirisObject):
#    """
#    A ScirisObject-wrapped Optima Nutrition Project object.
#    
#    Methods:
#        __init__(theProject: Project, ownerUID: UUID, theUID: UUID [None]): 
#            void -- constructor
#        loadFromCopy(otherObject): void -- assuming otherObject is another 
#            object of our type, copy its contents to us (calls the 
#            ScirisObject superclass version of this method also)   
#        show(): void -- print the contents of the object
#        getUserFrontEndRepr(): dict -- get a JSON-friendly dictionary 
#            representation of the object state the front-end uses for non-
#            admin purposes  
#        saveAsFile(loadDir: str): str -- given a load dictionary, save the 
#            project in a file there and return the file name
#                    
#    Attributes:
#        theProject (Project) -- the actual Project object being wrapped
#        ownerUID (UUID) -- the UID of the User that owns the Project
#        
#    Usage:
#        >>> myProject = ProjectSO(theProject,uuid.UUID('12345678123456781234567812345678'))                      
#    """
#    
#    def  __init__(self, theProject, ownerUID, theUID=None):
#        # NOTE: theUID argument is ignored but kept here to not mess up
#        # inheritance.
#        
#        # Make sure the argument is a valid UUID, converting a hex text to a
#        # UUID object, if needed.        
#        validUID = sobj.getValidUUID(ownerUID)
#        
#        # If we have a valid UUID...
#        if validUID is not None:       
#            # Set superclass parameters (passing in the actual Project's UID).
#            super(ProjectSO, self).__init__(theProject.uid)
#                   
#            # Set the project to the Optima Project that is passed in.
#            self.theProject = theProject
#            
#            # Set the owner (User) UID.
#            self.ownerUID = validUID
#        
#    def loadFromCopy(self, otherObject):
#        if type(otherObject) == type(self):
#            # Do the superclass copying.
#            super(ProjectSO, self).loadFromCopy(otherObject)
#            
#            # Copy the Project object itself.
#            self.theProject = dcp(otherObject.theProject)
#            
#            # Copy the owner UID.
#            self.ownerUID = otherObject.ownerUID
#                
#    def show(self):
#        # Show superclass attributes.
#        super(ProjectSO, self).show()  
#        
#        # Show the Optima defined display text for the project.
#        print '---------------------'
#        print 'Owner User UID: %s' % self.ownerUID.hex
#        print 'Project Name: %s' % self.theProject.name
#        print 'Spreadsheet Path: %s' % self.theProject.spreadsheetPath
#        print 'Creation Time: %s' % self.theProject.createdTime
#        print 'Update Time: %s' % self.theProject.updatedTime
#        print 'Data Upload Time: %s' % self.theProject.dataUploadTime
#            
#    def getUserFrontEndRepr(self):
#        objInfo = {
#          'id': self.uid,
#          'name': self.theProject.name,
#          'userId': self.ownerUID,
#          'spreadsheetPath': self.theProject.spreadsheetPath,
#          'creationTime': self.theProject.createdTime,
#          'updatedTime': self.theProject.updatedTime,
#          'dataUploadTime': self.theProject.dataUploadTime,
#          'simulationOK': (self.theProject.dataUploadTime is not None)
#        }
#        return objInfo
#    
#    def saveAsFile(self, loadDir):
#        # Save the project in the file.
#        self.theProject.saveToPrjFile(loadDir, saveResults=True)
#        
#        # Return the filename (not the full one).
#        return self.theProject.name + ".prj"
    
class ProjectCollection(sobj.ScirisCollection):
    """
    A collection of Projects.
    
    Methods:
        __init__(theUID: UUID [None], theTypePrefix: str ['projectscoll'], 
            theFileSuffix: str ['.pc'], 
            theInstanceLabel: str ['Projects Collection']): void -- constructor  
        getUserFrontEndRepr(ownerUID: UUID): list -- return a list of dicts 
            containing JSON-friendly project contents for each project that 
            is owned by the specified user UID
        getProjectEntriesByUser(ownerUID: UUID): list -- return the ProjectSOs 
            that match the owning User UID in a list
        
    Usage:
        >>> theProjCollection = ProjectCollection(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def __init__(self, theUID, theTypePrefix='projectscoll', theFileSuffix='.pc', 
        theInstanceLabel='Projects Collection'):
        # Set superclass parameters.
        super(ProjectCollection, self).__init__(theUID, theTypePrefix, theFileSuffix, 
             theInstanceLabel)
            
    def getUserFrontEndRepr(self, ownerUID):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        validUID = sobj.getValidUUID(ownerUID)
        
        # If we have a valid UUID...
        if validUID is not None:               
            # Get dictionaries for each Project in the dictionary.
            projectsInfo = [self.theObjectDict[theKey].getUserFrontEndRepr() \
                for theKey in self.theObjectDict \
                if self.theObjectDict[theKey].ownerUID == validUID]
            return projectsInfo
        
        # Otherwise, return an empty list.
        else:
            return []
        
    def getProjectEntriesByUser(self, ownerUID):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        validUID = sobj.getValidUUID(ownerUID)
        
        # If we have a valid UUID...
        if validUID is not None:    
            # Get ProjectSO entries for each Project in the dictionary.
            projectEntries = [self.theObjectDict[theKey] \
                for theKey in self.theObjectDict \
                if self.theObjectDict[theKey].ownerUID == validUID]
            return projectEntries
        
        # Otherwise, return an empty list.
        else:
            return []

#
# Other functions (mostly helpers for the RPCs)
#

def now_utc():
    ''' Get the current time, in UTC time '''
    now = datetime.datetime.now(dateutil.tz.tzutc())
    return now

def load_project_record(project_id, raise_exception=True):
    """
    Return the project DataStore reocord, given a project UID.
    """ 
    
    # Load the matching ProjectSO object from the database.
    project_record = theProjCollection.getObjectByUID(project_id)

    # If we have no match, we may want to throw an exception.
    if project_record is None:
        if raise_exception:
            raise ProjectDoesNotExist(id=project_id)
            
    # Return the Project object for the match (None if none found).
    return project_record

def load_project(project_id, raise_exception=True):
    """
    Return the Nutrition Project object, given a project UID, or None if no 
    ID match is found.
    """ 
    
    # Load the project record matching the ID passed in.
    project_record = load_project_record(project_id, 
        raise_exception=raise_exception)
    
    # If there is no match, raise an exception or return None.
    if project_record is None:
        if raise_exception:
            raise ProjectDoesNotExist(id=project_id)
        else:
            return None
        
    # Return the found project.
    return project_record.theProject

def load_project_summary_from_project_record(project_record):
    """
    Return the project summary, given the DataStore record.
    """ 
    
    # Return the built project summary.
    return project_record.getUserFrontEndRepr()  
          
def get_unique_name(name, other_names=None):
    """
    Given a name and a list of other names, find a replacement to the name 
    that doesn't conflict with the other names, and pass it back.
    """
    
    # If no list of other_names is passed in, load up a list with all of the 
    # names from the project summaries.
    if other_names is None:
        other_names = [p['project']['name'] for p in load_current_user_project_summaries(checkEndpoint=False)['projects']]
      
    # Start with the passed in name.
    i = 0
    unique_name = name
    
    # Try adding an index (i) to the name until we find one that no longer 
    # matches one of the other names in the list.
    while unique_name in other_names:
        i += 1
        unique_name = "%s (%d)" % (name, i)
        
    # Return the found name.
    return unique_name

#
# RPC functions
#

def tester_func_project(project_id):
    theProj = load_project(project_id)
    print theProj.name
    
    return 'success'

def get_scirisdemo_projects():
    """
    Return the projects associated with the Sciris Demo user.
    """
    
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'normalProjectRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Get the user UID for the _ScirisDemo user.
    user_id = user.get_scirisdemo_user()
   
    # Get the ProjectSO entries matching the _ScirisDemo user UID.
    projectEntries = theProjCollection.getProjectEntriesByUser(user_id)

    # Collect the project summaries for that user into a list.
    projectSummaryList = map(load_project_summary_from_project_record, 
        projectEntries)
    
    # Sort the projects by the project name.
    sortedSummaryList = sorted(projectSummaryList, 
        key=lambda proj: proj['project']['name']) # Sorts by project name
    
    # Return a dictionary holding the project summaries.
    output = {'projects': sortedSummaryList}
    return output

def load_project_summary(project_id):
    """
    Return the project summary, given the Project UID.
    """ 
    
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'normalProjectRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Load the project record matching the UID of the project passed in.
    project_entry = load_project_record(project_id)
    
    # Return a project summary from the accessed ProjectSO entry.
    return load_project_summary_from_project_record(project_entry)

def load_current_user_project_summaries(checkEndpoint=True):
    """
    Return project summaries for all projects the user has to the client.
    """ 
    
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if checkEndpoint:
        if request.endpoint != 'normalProjectRPC':
            return {'error': 'Unauthorized RPC'}
    
    # Get the ProjectSO entries matching the user UID.
    projectEntries = theProjCollection.getProjectEntriesByUser(current_user.get_id())
    
    # Grab a list of project summaries from the list of ProjectSO objects we 
    # just got.
    return {'projects': map(load_project_summary_from_project_record, 
        projectEntries)}
                
def load_all_project_summaries():
    """
    Return project summaries for all projects to the client.
    """ 
    
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'normalProjectRPC':
        return {'error': 'Unauthorized RPC'}   
    
    # Get all of the ProjectSO entries.
    projectEntries = theProjCollection.getAllObjects()
    
    # Grab a list of project summaries from the list of ProjectSO objects we 
    # just got.
    return {'projects': map(load_project_summary_from_project_record, 
        projectEntries)}
    
def delete_projects(project_ids):
    """
    Delete all of the projects with the passed in UIDs.
    """ 
    
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'normalProjectRPC':
        return {'error': 'Unauthorized RPC'}   
    
    # Loop over the project UIDs of the projects to be deleted...
    for project_id in project_ids:
        # Load the project record matching the UID of the project passed in.
        record = load_project_record(project_id, raise_exception=True)
        
        # If a matching record is found, delete the object from the 
        # ProjectCollection.
        if record is not None:
            theProjCollection.deleteObjectByUID(project_id)
   
def download_project(project_id):
    """
    For the passed in project UID, get the Project on the server, save it in a 
    file, minus results, and pass the full path of this file back.
    """
    
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'downloadProjectRPC':
        return {'error': 'Unauthorized RPC'}   
    
    # Load the project with the matching UID.
    theProj = load_project(project_id, raise_exception=True)
    
    # Use the downloads directory to put the file in.
    dirname = ds.downloadsDir.dirPath
        
    # Create a filename containing the project name followed by a .prj 
    # suffix.
    fileName = '%s.prj' % theProj.name
        
    # Generate the full file name with path.
    fullFileName = '%s%s%s' % (dirname, os.sep, fileName)
        
    # Write the object to a Gzip string pickle file.
    ds.objectToGzipStringPickleFile(fullFileName, theProj)
    
    # Display the call information.
    print(">> download_project %s" % (fullFileName))
    
    # Return the full filename.
    return fullFileName

def load_zip_of_prj_files(project_ids):
    """
    Given a list of project UIDs, make a .zip file containing all of these 
    projects as .prj files, and return the full path to this file.
    """
    
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'downloadProjectRPC':
        return {'error': 'Unauthorized RPC'}   
    
    # Use the downloads directory to put the file in.
    dirname = ds.downloadsDir.dirPath

    # Build a list of ProjectSO objects for each of the selected projects, 
    # saving each of them in separate .prj files.
    prjs = [load_project_record(id).saveAsFile(dirname) for id in project_ids]
    
    # Make the zip file name and the full server file path version of the same..
    zip_fname = '{}.zip'.format(uuid.uuid4())
    server_zip_fname = os.path.join(dirname, zip_fname)
    
    # Create the zip file, putting all of the .prj files in a projects 
    # directory.
    with ZipFile(server_zip_fname, 'w') as zipfile:
        for prj in prjs:
            zipfile.write(os.path.join(dirname, prj), 'projects/{}'.format(prj))
            
    # Display the call information.
    print(">> load_zip_of_prj_files %s" % (server_zip_fname))

    # Return the server file name.
    return server_zip_fname

#
# Script code
#

