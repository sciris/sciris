"""
project.py -- code related to Sciris project management
    
Last update: 1/17/18 (gchadder3)
"""

#
# Imports
#

import datetime
import dateutil
import dateutil.tz
import uuid
import scirisobjects as sobj
import datastore as ds

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
                'name': self.theProject.name,
                'userId': self.ownerUID,
                'spreadsheetPath': self.theProject.spreadsheetPath,
                'creationTime': self.theProject.createdTime,
                'updatedTime': self.theProject.updatedTime,
                'dataUploadTime': self.theProject.dataUploadTime
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
# Other functions
#

def now_utc():
    ''' Get the current time, in UTC time '''
    now = datetime.datetime.now(dateutil.tz.tzutc())
    return now
  
#
# RPC functions
#



#
# Script code
#

