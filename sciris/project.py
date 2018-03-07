"""
project.py -- code related to Sciris project management
    
Last update: 3/6/18 (gchadder3)
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

#
# Globals
#

# The ProjectCollection object for all of the app's projects.  Gets 
# initialized by and loaded by init_projects().
theProjCollection = None

#
# Classes
#

class Project(sobj.ScirisObject):
    """
    A Sciris project.
    
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

def load_project_summary_from_project_record(project_record):
    """
    Return the project summary, given the DataStore record.
    """ 
    
    # Return the built project summary.
    return project_record.getUserFrontEndRepr()  

#
# RPC functions
#

def get_scirisdemo_projects():
    """
    Return the projects associated with the Sciris Demo user.
    """
    
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'normalProjectRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Get the user UID for the _OptimaDemo user.
    user_id = user.get_scirisdemo_user()
   
    # Get the Project entries matching the _OptimaLite user UID.
    projectEntries = theProjCollection.getProjectEntriesByUser(user_id)

    # Get the projects for that user in a project list.
    projectlist = map(load_project_summary_from_project_record, projectEntries)
    
    # Sort the projects by the project name.
    sortedprojectlist = sorted(projectlist, key=lambda proj: proj['project']['name']) # Sorts by project name
    
    # Return a dictionary holding the projects.
    output = {'projects': sortedprojectlist}
    return output

def load_current_user_project_summaries():
    """
    Return project summaries for all projects the user has to the client.
    """ 
    
    # Get the ProjectSO entries matching the user UID.
    projectEntries = theProjCollection.getProjectEntriesByUser(current_user.get_id())
    
    # Grab a list of project summaries from the list of ProjectSO objects we 
    # just got.
    return {'projects': map(load_project_summary_from_project_record, 
        projectEntries)}
                
#def load_project_record(project_id, raise_exception=True):
#    """
#    Return the project DataStore reocord, given a project UID.
#    """ 
#    
#    # Load the matching ProjectSO object from the database.
#    project_record = theProjCollection.getObjectByUID(project_id)
#
#    # If we have no match, we may want to throw an exception.
#    if project_record is None:
#        if raise_exception:
#            raise ProjectDoesNotExist(id=project_id)
#            
#    # Return the ProjectSO object for the match (None if none found).
#    return project_record
#
#def save_project(project, skip_result=False):
#    """
#    Given a Project object, wrap it in a new ProjectSO object and put this 
#    in the project collection (either adding a new object, or updating an 
#    existing one)  skip_result lets you null out saved results in the Project.
#    """ 
#    
#    # Load the project record matching the UID of the project passed in.
#    project_record = load_project_record(project.uid)
#    
#    # Copy the project, only save what we want...
#    new_project = dcp(project)
##    if skip_result:
##        new_project.results = op.odict()
#         
#    # Create the new project entry and enter it into the ProjectCollection.
#    # Note: We don't need to pass in project.uid as a 3rd argument because 
#    # the constructor will automatically use the Project's UID.
#    theProj = ProjectSO(new_project, project_record.ownerUID)
#    theProjCollection.updateObject(theProj)
#    
#def load_project(project_id, raise_exception=True):
#    """
#    Return the Nutrition Project object, given a project UID, or None if no 
#    ID match is found.
#    """ 
#    
#    # Load the project record matching the ID passed in.
#    project_record = load_project_record(project_id, 
#        raise_exception=raise_exception)
#    
#    # If there is no match, raise an exception or return None.
#    if project_record is None:
#        if raise_exception:
#            raise ProjectDoesNotExist(id=project_id)
#        else:
#            return None
#        
#    # Return the found project.
#    return project_record.theProject
#
#def update_project_with_fn(project_id, update_project_fn):
#    """
#    Do an update of a Nutrition Project, given the UID and a function that 
#    does the actual Project updating.
#    """ 
#    
#    # Load the project.
#    project = load_project(project_id)
#    
#    # Execute the passed-in function that modifies the project.
#    update_project_fn(project)
#    
#    # Set the updating time to now.
#    project.updatedTime = guidemo1.today()
#    
#    # Save the changed project.
#    save_project(project)


#def load_project_summary(project_id):
#    """
#    Return the project summary, given the Project UID.
#    """ 
#    
#    # Load the project record matching the UID of the project passed in.
#    project_entry = load_project_record(project_id)
#    
#    # Return a project summary from the accessed ProjectSO entry.
#    return load_project_summary_from_project_record(project_entry)
#

#
#def load_all_project_summaries():
#    """
#    Return project summaries for all projects to the client.
#    """ 
#    
#    # Get all of the ProjectSO entries.
#    projectEntries = theProjCollection.getAllObjects()
#    
#    # Grab a list of project summaries from the list of ProjectSO objects we 
#    # just got.
#    return {'projects': map(load_project_summary_from_project_record, 
#        projectEntries)}
#
#def create_project(user_id, project_summary):
#    """
#    Create a new Nutrition Project for a user from a passed in project 
#    summary.
#    """ 
#    
#    # Create a new Project object with the name passed in from the project 
#    # summary.
#    project = guidemo1.Project(name=project_summary['name'])
#    
#    # Display the call information.
#    print(">> create_project %s" % (project.name))
#    
#    # Save the new project.
#    save_project_as_new(project, user_id)
#    
#    # Return the new project UID.
#    return {'projectId': str(project.uid)}
#
#def delete_projects(project_ids):
#    """
#    Delete all of the projects with the passed in UIDs.
#    """ 
#    
#    # Loop over the project UIDs of the projects to be deleted...
#    for project_id in project_ids:
#        # Load the project record matching the UID of the project passed in.
#        record = load_project_record(project_id, raise_exception=True)
#        
#        # If a matching record is found, delete the object from the 
#        # ProjectCollection.
#        if record is not None:
#            theProjCollection.deleteObjectByUID(project_id)
#                        
#def update_project_from_summary(project_summary, delete_data=False):
#    """
#    Given the passed in project summary, update the underlying project 
#    accordingly.
#    """ 
#    
#    # Load the project corresponding with this summary.
#    project = load_project(project_summary['id'])
#       
#    # If we want to delete the data, delete it in the summary. 
#    if delete_data:
#        pass
#        # parse.clear_project_data(project)
#        
#    # Use the summary to set the actual project.
#    project.updateName(project_summary['name'])
#    #parse.set_project_summary_on_project(project, project_summary)
#    
#    # Save the changed project to the DataStore.
#    save_project(project)
#    
#def download_data_spreadsheet(project_id):
#    """
#    Get the spreadsheet file in the Nutrition Project with the passed in UID.
#    """
#    
#    # Load the project with the matching UID.
#    project = load_project(project_id, raise_exception=True)
#    
#    # Get the file name directly from from the spreadsheet path saved in the 
#    # Project.
#    server_filename = project.spreadsheetPath
#    
#    # Display the call information.
#    print(">> download_data_spreadsheet %s" % (server_filename))
#    
#    # Return the full filename.
#    return server_filename
#
#def save_project_as_new(project, user_id):
#    """
#    Given a Project object and a user UID, wrap the Project in a new ProjectSO 
#    object and put this in the project collection, after getting a fresh UID
#    for this Project.  Then do the actual save.
#    """ 
#    
#    # Set a new project UID, so we aren't replicating the UID passed in.
#    project.uid = uuid.uuid4()
#    
#    # Create the new project entry and enter it into the ProjectCollection.
#    theProj = ProjectSO(project, user_id)
#    theProjCollection.addObject(theProj)  
#
#    # Display the call information.
#    print(">> save_project_as_new '%s'" % project.name)
#
#    # Save the changed Project object to the DataStore.
#    save_project(project)
#    
#    return None
#
#def copy_project(project_id, new_project_name):
#    """
#    Given a project UID and a new project name, creates a copy of the project 
#    with a new UID and returns that UID.
#    """
#    
#    # Display the call information.
#    print(">> copy_project args project_id %s" % project_id)
#    print(">> copy_project args new_project_name %s" % new_project_name)     
#    
#    # Get the Project object for the project to be copied.
#    project_record = load_project_record(project_id, raise_exception=True)
#    project = project_record.theProject
#    
#    # Make a copy of the project loaded in to work with.
#    new_project = dcp(project)
#    
#    # Just change the project name, and we have the new version of the 
#    # Project object to be saved as a copy.
#    new_project.name = new_project_name
#    
#    # Set the user UID for the new projects record to be the current user.
#    user_id = current_user.get_id() 
#    
#    # Save a DataStore projects record for the copy project.
#    save_project_as_new(new_project, user_id)
#    
#    # Remember the new project UID (created in save_project_as_new()).
#    copy_project_id = new_project.uid
#
#    # Return the UID for the new projects record.
#    return { 'projectId': copy_project_id }
#
#def get_unique_name(name, other_names=None):
#    """
#    Given a name and a list of other names, find a replacement to the name 
#    that doesn't conflict with the other names, and pass it back.
#    """
#    
#    # If no list of other_names is passed in, load up a list with all of the 
#    # names from the project summaries.
#    if other_names is None:
#        other_names = [p['name'] for p in load_current_user_project_summaries()]
#      
#    # Start with the passed in name.
#    i = 0
#    unique_name = name
#    
#    # Try adding an index (i) to the name until we find one that no longer 
#    # matches one of the other names in the list.
#    while unique_name in other_names:
#        i += 1
#        unique_name = "%s (%d)" % (name, i)
#        
#    # Return the found name.
#    return unique_name
#
#def create_project_from_prj_file(prj_filename, user_id, other_names):
#    """
#    Given a .prj file name, a user UID, and other other file names, 
#    create a new project from the file with a new UID and return the new UID.
#    """
#    
#    # Display the call information.
#    print(">> create_project_from_prj_file '%s'" % prj_filename)
#    
#    # Try to open the .prj file, and return an error message if this fails.
#    try:
#        project = guidemo1.loadProjectFromPrjFile(prj_filename)
#    except Exception:
#        return { 'projectId': 'BadFileFormatError' }
#    
#    # Reset the project name to a new project name that is unique.
#    project.name = get_unique_name(project.name, other_names)
#    
#    # Save the new project in the DataStore.
#    save_project_as_new(project, user_id)
#    
#    # Return the new project UID in the return message.
#    return { 'projectId': str(project.uid) }
#
#def create_project_from_spreadsheet(xlsx_filename, user_id, other_names):
#    """
#    Given an Excel file name, a user UID, and other other file names, 
#    create a new project from the file with a new UID and return the new UID.
#    """
#    
#    # Display the call information.
#    print(">> create_project_from_spreadsheet '%s'" % xlsx_filename)
#    
#    # Create a new Project with the passed in Excel file.
#    project = guidemo1.Project(name='spreadsheetread', 
#        spreadsheetPath=xlsx_filename)
#    
#    # Get a unique project name if one is lacking.
#    project.name = get_unique_name(project.name, other_names)
#    
#    # Save the new project.
#    save_project_as_new(project, user_id)
#    
#    # Return the UID for the new project.
#    return { 'projectId': str(project.uid) }
#
#def download_project(project_id):
#    """
#    For the passed in project UID, get the Project on the server, save it in a 
#    file, minus results, and pass the full path of this file back.
#    """
#    
#    # Load the project with the matching UID.
#    project = load_project(project_id, raise_exception=True)
#    
#    # Use the uploads directory to put the file in.
#    dirname = ds.uploadsPath
#    
#    # Save the project to the uploads directory file, leaving out results.
#    server_filename = project.saveToPrjFile(dirPath=dirname, saveResults=False)
#    
#    # Display the call information.
#    print(">> download_project %s" % (server_filename))
#    
#    # Return the full filename.
#    return server_filename
#
#def download_project_with_result(project_id):
#    """
#    For the passed in project UID, get the Project on the server, save it in a 
#    file, and pass the full path of this file back.
#    """
#    
#    # Load the project with the matching UID.
#    project = load_project(project_id, raise_exception=True)
#            
#    # Use the uploads directory to put the file in.
#    dirname = ds.uploadsPath
#    
#    # Save the project to the uploads directory file, including results.       
#    server_filename = project.saveToPrjFile(dirPath=dirname, saveResults=True)
#    
#    # Display the call information.
#    print(">> download_project_with_result %s" % (server_filename))
#    
#    # Return the full filename.
#    return server_filename
#
#def update_project_from_uploaded_spreadsheet(spreadsheet_fname, project_id):
#    """
#    Update the spreadsheet in the project pointed to by the project UID.
#    """
#    
#    # Display the call information.
#    print(">> update_project_from_uploaded_spreadsheet %s" % (spreadsheet_fname))
#    
#    # Function to pass into update_project_with_fn()
#    def modify(project):
#        project.updateSpreadsheet(spreadsheet_fname)
#        
#    # Update the project with the above function.    
#    update_project_with_fn(project_id, modify)
#        
#    # Return success.
#    return { 'success': True }
#
#def load_zip_of_prj_files(project_ids):
#    """
#    Given a list of project UIDs, make a .zip file containing all of these 
#    projects as .prj files, and return the full path to this file.
#    """
#    
#    # Use the uploads directory to put the file in.
#    dirname = ds.uploadsPath
#
#    # Build a list of ProjectSO objects for each of the selected projects, 
#    # saving each of them in separate .prj files.
#    prjs = [load_project_record(id).saveAsFile(dirname) for id in project_ids]
#    
#    # Make the zip file name and the full server file path version of the same..
#    zip_fname = '{}.zip'.format(uuid.uuid4())
#    server_zip_fname = os.path.join(dirname, zip_fname)
#    
#    # Create the zip file, putting all of the .prj files in a projects 
#    # directory.
#    with ZipFile(server_zip_fname, 'w') as zipfile:
#        for prj in prjs:
#            zipfile.write(os.path.join(dirname, prj), 'projects/{}'.format(prj))
#            
#    # Display the call information.
#    print(">> load_zip_of_prj_files %s" % (server_zip_fname))
#
#    # Return the server file name.
#    return server_zip_fname

#
# Script code
#

