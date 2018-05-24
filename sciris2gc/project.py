"""
project.py -- code related to Sciris project management
    
Last update: 5/22/18 (gchadder3)
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
proj_collection = None

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
        __init__(name: str, owner_uid: UUID, uid: UUID [None], 
            spreadsheet_path: str [None]): void -- constructor            
        update_name(new_name: str): void -- change the project name to new_name
        update_spreadsheet(spreadsheet_path: str): void -- change the 
            spreadsheet the project is using
            
        save_to_prj_file(dir_path: str, save_results: bool [False]) -- save the 
            project to a .prj file and return the full path
            
        load_from_copy(other_object): void -- assuming other_object is another 
            object of our type, copy its contents to us (calls the 
            ScirisObject superclass version of this method also)             
        show(): void -- print the contents of the object
        get_user_front_end_repr(): dict -- get a JSON-friendly dictionary 
            representation of the object state the front-end uses for non-
            admin purposes 
            
        save_as_file(loadDir: str): str -- given a load dictionary, save the 
            project in a file there and return the file name
            
    Attributes:
        uid (UUID) -- the UID of the Project
        owner_uid (UUID) -- the UID of the User that owns the Project        
        name (str) -- the Project's name
        spreadsheet_path (str) -- the full path name for the Excel spreadsheet
        created_time (datetime.datetime) -- the time that the Project was 
            created
        updated_time (datetime.datetime) -- the time that the Project was last 
            updated
        data_upload_time (datetime.datetime) -- the time that the Project's 
            spreadsheet was last updated
        
    Usage:
        >>> proj = Project('myproject', uuid.UUID('12345678123456781234567812345672'), uuid.UUID('12345678123456781234567812345678'))                    
    """ 
    
    def  __init__(self, name, owner_uid, uid=None, spreadsheet_path=None):
        # Make sure owner has a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        valid_owner_uid = sobj.get_valid_uuid(owner_uid)
        
        # If we have a valid UUID...
        if valid_owner_uid is not None:  
            # Set the owner (User) UID.
            self.owner_uid = valid_owner_uid
                       
            # If a UUID was passed in...
            if uid is not None:
                # Make sure the argument is a valid UUID, converting a hex text to a
                # UUID object, if needed.        
                valid_uuid = sobj.get_valid_uuid(uid) 
                
                # If a valid_uuid was found, use it.
                if valid_uuid is not None:
                    self.uid = valid_uuid
                # Otherwise, generate a new random UUID using uuid4().
                else:
                    self.uid = uuid.uuid4()
            # Otherwise, generate a new random UUID using uuid4().
            else:
                self.uid = uuid.uuid4()
                
            # Set the project name.
            self.name = name
            
            # Set the spreadsheet_path.
            self.spreadsheet_path = spreadsheet_path
                                
            # Set the creation time for now.
            self.created_time = now_utc()
            
            # Set the updating time to None.
            self.updated_time = None
            
            # Set the spreadsheet upload time to None.
            self.data_upload_time = None
            
            # If we have passed in a spreadsheet path...
            if self.spreadsheet_path is not None:
                # Set the data spreadsheet upload time for now.
                self.data_upload_time = now_utc()
                
            # Set the type prefix to 'user'.
            self.type_prefix = 'project'
            
            # Set the file suffix to '.usr'.
            self.file_suffix = '.prj'
            
            # Set the instance label to the username.
            self.instance_label = name   
          
    def update_name(self, new_name):
        # Set the project name.
        self.name = new_name
        self.instance_label = new_name
        
        # Set the updating time to now.
        self.updated_time = now_utc()
        
    def update_spreadsheet(self, spreadsheet_path):
        # Set the spreadsheet_path from what's passed in.
        self.spreadsheet_path = spreadsheet_path
        
        # Set the data spreadsheet upload time for now.
        self.data_upload_time = now_utc()
        
        # Set the updating time to now.
        self.updated_time = now_utc()
        
#    def save_to_prj_file(self, dir_path, save_results=False):
#        # Create a filename containing the project name followed by a .prj 
#        # suffix.
#        file_name = '%s.prj' % self.name
#        
#        # Generate the full file name with path.
#        full_file_name = '%s%s%s' % (dir_path, os.sep, file_name)
#        
#        # Write the object to a Gzip string pickle file.
#        object_to_gzip_string_pickle_file(full_file_name, self)
#        
#        # Return the full file name.
#        return full_file_name

    def load_from_copy(self, other_object):
        if type(other_object) == type(self):
            # Do the superclass copying.
            super(ProjectSOBase, self).load_from_copy(other_object)
            
            self.owner_uid = other_object.owner_uid
            self.name = other_object.name
            self.spreadsheet_path = other_object.spreadsheet_path
            self.created_time = other_object.created_time
            self.updated_time = other_object.updated_time
            self.data_upload_time = other_object.data_upload_time
            
            # Copy the owner UID.
            self.owner_uid = other_object.owner_uid
            
    def show(self):
        # Show superclass attributes.
        super(ProjectSOBase, self).show()  
        
        print '---------------------'

        print 'Owner User UID: %s' % self.owner_uid.hex
        print 'Project Name: %s' % self.name
        print 'Spreadsheet Path: %s' % self.spreadsheet_path
        print 'Creation Time: %s' % self.created_time
        print 'Update Time: %s' % self.updated_time
        print 'Data Upload Time: %s' % self.data_upload_time
        
    def get_user_front_end_repr(self):
        obj_info = {
            'project': {
                'id': self.uid,
                'name': self.name,
                'userId': self.owner_uid,
                'spreadsheet_path': self.spreadsheet_path,
                'creationTime': self.created_time,
                'updated_time': self.updated_time,
                'data_upload_time': self.data_upload_time
            }
        }
        return obj_info
    
#    def save_as_file(self, loadDir):
#        # Save the project in the file.
#        self.proj.save_to_prj_file(loadDir, save_results=True)
#        
#        # Return the file_name (not the full one).
#        return self.proj.name + ".prj" 
    
class ProjectCollection(sobj.ScirisCollection):
    """
    A collection of Projects.
    
    Methods:
        __init__(uid: UUID [None], type_prefix: str ['projectscoll'], 
            file_suffix: str ['.pc'], 
            instance_label: str ['Projects Collection']): void -- constructor  
        get_user_front_end_repr(owner_uid: UUID): list -- return a list of dicts 
            containing JSON-friendly project contents for each project that 
            is owned by the specified user UID
        get_project_entries_by_user(owner_uid: UUID): list -- return the ProjectSOs 
            that match the owning User UID in a list
        
    Usage:
        >>> proj_collection = ProjectCollection(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def __init__(self, uid, type_prefix='projectscoll', file_suffix='.pc', 
        instance_label='Projects Collection'):
        # Set superclass parameters.
        super(ProjectCollection, self).__init__(uid, type_prefix, file_suffix, 
             instance_label)
            
    def get_user_front_end_repr(self, owner_uid):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        valid_uuid = sobj.get_valid_uuid(owner_uid)
        
        # If we have a valid UUID...
        if valid_uuid is not None:               
            # Get dictionaries for each Project in the dictionary.
            projects_info = [self.obj_dict[key].get_user_front_end_repr() \
                for key in self.obj_dict \
                if self.obj_dict[key].owner_uid == valid_uuid]
            return projects_info
        
        # Otherwise, return an empty list.
        else:
            return []
        
    def get_project_entries_by_user(self, owner_uid):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        valid_uuid = sobj.get_valid_uuid(owner_uid)
        
        # If we have a valid UUID...
        if valid_uuid is not None:    
            # Get ProjectSO entries for each Project in the dictionary.
            project_entries = [self.obj_dict[key] \
                for key in self.obj_dict \
                if self.obj_dict[key].owner_uid == valid_uuid]
            return project_entries
        
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
    project_record = proj_collection.get_object_by_uid(project_id)

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
    return project_record.proj

def load_project_summary_from_project_record(project_record):
    """
    Return the project summary, given the DataStore record.
    """ 
    
    # Return the built project summary.
    return project_record.get_user_front_end_repr()  
          
def get_unique_name(name, other_names=None):
    """
    Given a name and a list of other names, find a replacement to the name 
    that doesn't conflict with the other names, and pass it back.
    """
    
    # If no list of other_names is passed in, load up a list with all of the 
    # names from the project summaries.
    if other_names is None:
        other_names = [p['project']['name'] for p in load_current_user_project_summaries(check_endpoint=False)['projects']]
      
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
    proj = load_project(project_id)
    print proj.name
    
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
    project_entries = proj_collection.get_project_entries_by_user(user_id)

    # Collect the project summaries for that user into a list.
    project_summary_list = map(load_project_summary_from_project_record, 
        project_entries)
    
    # Sort the projects by the project name.
    sorted_summary_list = sorted(project_summary_list, 
        key=lambda proj: proj['project']['name']) # Sorts by project name
    
    # Return a dictionary holding the project summaries.
    output = {'projects': sorted_summary_list}
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

def load_current_user_project_summaries(check_endpoint=True):
    """
    Return project summaries for all projects the user has to the client.
    """ 
    
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if check_endpoint:
        if request.endpoint != 'normalProjectRPC':
            return {'error': 'Unauthorized RPC'}
    
    # Get the ProjectSO entries matching the user UID.
    project_entries = proj_collection.get_project_entries_by_user(current_user.get_id())
    
    # Grab a list of project summaries from the list of ProjectSO objects we 
    # just got.
    return {'projects': map(load_project_summary_from_project_record, 
        project_entries)}
                
def load_all_project_summaries():
    """
    Return project summaries for all projects to the client.
    """ 
    
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'normalProjectRPC':
        return {'error': 'Unauthorized RPC'}   
    
    # Get all of the ProjectSO entries.
    project_entries = proj_collection.get_all_objects()
    
    # Grab a list of project summaries from the list of ProjectSO objects we 
    # just got.
    return {'projects': map(load_project_summary_from_project_record, 
        project_entries)}
    
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
            proj_collection.deleteObjectByUID(project_id)
   
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
    proj = load_project(project_id, raise_exception=True)
    
    # Use the downloads directory to put the file in.
    dirname = ds.downloads_dir.dir_path
        
    # Create a filename containing the project name followed by a .prj 
    # suffix.
    file_name = '%s.prj' % proj.name
        
    # Generate the full file name with path.
    full_file_name = '%s%s%s' % (dirname, os.sep, file_name)
        
    # Write the object to a Gzip string pickle file.
    ds.object_to_gzip_string_pickle_file(full_file_name, proj)
    
    # Display the call information.
    print(">> download_project %s" % (full_file_name))
    
    # Return the full filename.
    return full_file_name

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
    dirname = ds.downloads_dir.dir_path

    # Build a list of ProjectSO objects for each of the selected projects, 
    # saving each of them in separate .prj files.
    prjs = [load_project_record(id).save_as_file(dirname) for id in project_ids]
    
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

