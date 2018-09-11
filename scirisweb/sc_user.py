"""
user.py -- code related to Sciris user management
    
Last update: 2018aug20
"""

from flask import Flask, session, current_app
from flask_login import current_user, login_user, logout_user
from hashlib import sha224
from numpy import argsort
import six
import sciris as sc
from . import sc_rpcs as rpcs #import make_RPC
from . import sc_objects as sobj
from . import sc_datastore as ds # CK: Not sure if this is needed

##############################################################
### Globals
##############################################################

__all__ = ['user_dict'] # 'RPC_dict', 'RPC' not visible

user_dict = None # The UserDict object for all of the app's users.  Gets initialized by and loaded by init_users().
RPC_dict = {} # Dictionary to hold all of the registered RPCs in this module.
RPC = rpcs.makeRPCtag(RPC_dict) # RPC registration decorator factory created using call to make_RPC().


##############################################################
### Classes
##############################################################

__all__ += ['User', 'UserDict']

class User(sobj.Blob):
    """
    A Sciris user.
    
    Methods:
        __init__(username: str, password: str, display_name: str, 
            email: str [''], has_admin_rights: bool [False], 
            uid: UUID [None], hash_the_password: bool [True]): 
            void -- constructor
        load_from_copy(other_object): void -- assuming other_object is another 
            object of our type, copy its contents to us (calls the 
            Blob superclass version of this method also)            
        get_id(): UUID -- get the unique ID of this user (method required by 
            Flask-Login)    
        show(): void -- print the contents of the object
        get_user_front_end_repr(): dict -- get a JSON-friendly dictionary 
            representation of the object state the front-end uses for non-
            admin purposes
        get_admin_front_end_repr(): dict -- get a JSON-friendly dictionary
            representation of the object state the front-end uses for admin
            purposes        
                    
    Attributes:
        is_authenticated (bool) -- is this user authenticated? (attribute 
            required by Flask-Login)
        is_active (bool) -- is this user's account active? (attribute 
            required by Flask-Login)
        is_anonymous (bool) -- is this user considered anonymous?  (attribute
            required by Flask-Login)
        username (str) -- the username used to log in
        password (str) -- the user's SHA224-hashed password
        displayname (str) -- the user's name, which gets displayed in the 
            browser
        email (str) -- the user's email
        is_admin (bool) -- does this user have admin rights?
        
    Usage:
        >>> my_user = User('newguy', 'mesogreen', 'Ozzy Mandibulus',  \
            'tastybats@yahoo.com', uid=uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def  __init__(self, username, password, display_name, email='', 
        has_admin_rights=False, uid=None, hash_the_password=True):
        super(User, self).__init__(uid) # Set superclass parameters.
        self.is_authenticated = True # Set the user to be authentic.
        self.is_active = True # Set the account to be active.
        self.is_anonymous = False # The user is not anonymous.
        self.username = username # Set the username.
        raw_password = password # Set the raw password and use SHA224 to get the hashed version in hex form.
        if hash_the_password:
            if six.PY3: raw_password = raw_password.encode('utf-8')
            self.password = sha224(raw_password).hexdigest() 
        else:                
            self.password = password
        self.displayname = display_name # Set the displayname (what the browser will show).
        self.email = email # Set the user's email.
        self.is_admin = has_admin_rights # Set whether this user has admin rights.
        self.type_prefix = 'user' # Set the type prefix to 'user'.
        self.file_suffix = '.usr' # Set the file suffix to '.usr'.
        self.instance_label = username  # Set the instance label to the username.
        
    def load_from_copy(self, other_object):
        if type(other_object) == type(self):
            # Do the superclass copying.
            super(User, self).load_from_copy(other_object)
            self.is_authenticated = other_object.is_authenticated
            self.is_active = other_object.is_active
            self.is_anonymous = other_object.is_anonymous
            self.username = other_object.username
            self.password = other_object.password
            self.displayname = other_object.displayname
            self.email = other_object.email
            self.is_admin = other_object.is_admin
            
    def get_id(self):
        return self.uid
    
    def show(self, verbose=False):
        # Show superclass attributes.
        if verbose:
            super(User, self).show()
            print('---------------------')
            print('        Username: %s' % self.username)
            print('    Display name: %s' % self.displayname)
            print(' Hashed password: %s' % self.password)
            print('   Email address: %s' % self.email)
            print('Is authenticated: %s' % self.is_authenticated)
            print('       Is active: %s' % self.is_active)
            print('    Is anonymous: %s' % self.is_anonymous)
            print('        Is admin: %s' % self.is_admin)
        else:
            print('Username: %s; UID: %s' % (self.username, self.uid))
        return None
            
    def get_user_front_end_repr(self):
        obj_info = {
            'user': {
                'UID': self.uid.hex,                    
                'instanceLabel': self.instance_label,
                'username': self.username, 
                'displayname': self.displayname, 
                'email': self.email,
                'admin': self.is_admin                
            }
        }
        return obj_info
    
    def get_admin_front_end_repr(self):
        obj_info = {
            'user': {
                'UID': self.uid.hex, 
                'typePrefix': self.type_prefix, 
                'fileSuffix': self.file_suffix, 
                'instanceLabel': self.instance_label,
                'username': self.username, 
                'password': self.password, 
                'displayname': self.displayname, 
                'email': self.email,
                'authenticated': self.is_authenticated,
                'accountactive': self.is_active,
                'anonymous': self.is_anonymous,
                'admin': self.is_admin                
            }
        }
        return obj_info             
            
class UserDict(sobj.BlobDict):
    """
    A dictionary of Sciris users.
    
    Methods:
        __init__(uid: UUID [None], type_prefix: str ['userdict'], 
            file_suffix: str ['.ud'], 
            instance_label: str ['Users Dictionary']): void -- constructor
        load_from_copy(other_object): void -- assuming other_object is another 
            object of our type, copy its contents to us (calls the 
            BlobDict superclass version of this method also)           
        get_user_by_uid(uid: UUID or str): User or None -- returns the User  
            object pointed to by uid
        get_user_by_username(username: str): User or None -- return the User
            object pointed to by the username
        add(the_user: User): void -- add a User to the dictionary and update
            the dictionary's DataStore state
        update(the_user: User): void -- update a User in the dictionary and 
            update the dictionary's DataStore state
        delete_by_uid(uid: UUID or str): void -- delete a User in the dictionary
            selected by the UID, and update the dictionary's DataStore state
        delete_by_username(username: str): void -- delete a User in the 
            dictionary selected by a username, and update the dictionary's 
            DataStore state
        delete_all(): void -- delete the entire contents of the UserDict and 
            update the dictionary's DataStore state            
        get_user_front_end_repr(): dict -- get a JSON-friendly dictionary 
            representation of the collection state the front-end uses for non-
            admin purposes
        get_admin_front_end_repr(): dict -- get a JSON-friendly dictionary
            representation of the collection state the front-end uses for admin
            purposes
                    
    Attributes:
        username_hashes (dict) -- a dict mapping usernames to UIDs, so either
            indexing by UIDs or usernames can be fast
        
    Usage:
        >>> user_dict = UserDict(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def __init__(self, uid, type_prefix='userdict', file_suffix='.ud', instance_label='Users Dictionary'):
        # Set superclass parameters.
        super(UserDict, self).__init__(uid, type_prefix, file_suffix, instance_label)
        
        # Create the Python dict to hold the hashes from usernames to the UIDs.
        self.username_hashes = {}
        
    def load_from_copy(self, other_object):
        if type(other_object) == type(self):
            # Do the superclass copying.
            super(UserDict, self).load_from_copy(other_object)
            
            self.username_hashes = other_object.username_hashes
            
    def get_user_by_uid(self, uid):
        return self.get_object_by_uid(uid)
    
    def get_user_by_username(self, username):
        # Get the user's UID matching the username.
        user_index = self.username_hashes.get(username, None)
        
        # If we found at match, use the UID to try to fetch the user; 
        # otherwise, return None.
        if user_index is not None:
            return self.get_user_by_uid(user_index)
        else:
            return None
        
    def add(self, the_user):
        # Add the object to the hash table, keyed by the UID.
        self.add_object(the_user)
        
        # Add the username hash for this user.
        self.username_hashes[the_user.username] = the_user.get_id()
        
        # Update our DataStore representation if we are there. 
        if self.in_data_store():
            self.update_data_store()
    
    def update(self, the_user):
#        # Get the old username.
#        old_username = self.obj_dict[the_user.get_id()].username
#        
#        # If we new username is different than the old one, delete the old 
#        # usernameHash.
#        if the_user.username != old_username:
#            del self.username_hashes[old_username]
 
        # Add the user to the hash table, keyed by the UID.
        self.add_object(the_user)
        
        # Add the username hash for this user.
        self.username_hashes[the_user.username] = the_user.get_id()
       
        # Update our DataStore representation if we are there. 
        if self.in_data_store():
            self.update_data_store()
    
    def delete_by_uid(self, uid):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        

        obj = self.obj_dict[uid]
        
        # If a match is found...
        if obj is not None:
            # Remove entries from both user_dict and username_hashes 
            # attributes.
            username = obj.username
            del self.obj_dict[uid]
            del self.username_hashes[username]                
            
            # Update our DataStore representation if we are there. 
            if self.in_data_store():
                self.update_data_store()
        
    def delete_by_username(self, username):
        # Get the UID of the user matching username.
        user_index = self.username_hashes.get(username, None)
        
        # If we found a match, call delete_by_uid to complete the deletion.
        if user_index is not None:
            self.delete_by_uid(user_index)
    
    def delete_all(self):
        # Reset the Python dicts to hold the user objects and hashes from 
        # usernames to the UIDs.
        self.obj_dict = {}
        self.username_hashes = {}  
        
        # Update our DataStore representation if we are there. 
        if self.in_data_store():
            self.update_data_store()
            
    def get_user_front_end_repr(self):
        # Get dictionaries for each user in the dictionary.
        users_info = [self.obj_dict[key].get_user_front_end_repr() 
            for key in self.obj_dict]
        return users_info
        
    def get_admin_front_end_repr(self):
        # Get dictionaries for each user in the dictionary.       
        users_info = [self.obj_dict[key].get_admin_front_end_repr() 
            for key in self.obj_dict]
        
        # Extract just the usernames.
        user_names = [the_user['user']['username'] for the_user in users_info]
        
        # Get sorting indices with respect to the usernames.
        sort_order = argsort(user_names)
        
        # Created a list of the sorted users_info list.
        sorted_users_info = [users_info[ind] for ind in sort_order]

        # Return the sorted users info.      
        return sorted_users_info  



##############################################################
### Functions and RPCs
##############################################################

__all__ += ['get_scirisdemo_user', 'user_login', 'user_logout', 'get_current_user_info', 'get_all_users', 'user_register']
__all__ += ['user_change_info', 'user_change_password', 'admin_get_user_info', 'admin_delete_user', 'admin_activate_account']
__all__ += ['admin_deactivate_account', 'admin_grant_admin', 'admin_revoke_admin', 'admin_reset_password', 'make_test_users']

def get_scirisdemo_user(name='_ScirisDemo'):    
    # Get the user object matching (if any)...
    the_user = user_dict.get_user_by_username(name)
    
    # If there is a match, return the UID, otherwise return None.
    if the_user is not None:
        return the_user.get_id()
    else:
        return None

        
@RPC()
def user_login(username, password):  
    # Get the matching user (if any).
    matching_user = user_dict.get_user_by_username(username)
    
    # If we have a match and the password matches, and the account is active,
    # also, log in the user and return success; otherwise, return failure.
    if matching_user is not None and matching_user.password == password and \
        matching_user.is_active:
        # Log the user in.
        login_user(matching_user)
        
        return 'success'
    else:
        return 'failure'
    
@RPC(validation='named')       
def user_logout():
    # Log the user out and set the session to having an anonymous user.
    logout_user()
    
    # Clear the session cookie.
    session.clear()
    
    # Return nothing.
    return None

@RPC(validation='named') 
def get_current_user_info(): 
    return current_user.get_user_front_end_repr()

@RPC(validation='admin')
def get_all_users():
    # Return success.
    return user_dict.get_admin_front_end_repr()

@RPC()
def user_register(username, password, displayname, email): 
    # Get the matching user (if any).
    matching_user = user_dict.get_user_by_username(username)
    
    # If we have a match, fail because we don't want to register an existing 
    # user.
    if matching_user is not None:
        return 'failure'
    
    # Create a new User object with the new information.
    new_user = User(username, password, displayname, email, hash_the_password=False)
    
    # Set the user to be inactive (so an admin user needs to activate it, 
    # even though the account has been created) or active (so that no admin
    # action is necessary), according to the REGISTER_AUTOACTIVATE config 
    # parameter.
    new_user.is_active = current_app.config['REGISTER_AUTOACTIVATE']
    
    # Put the user right into the UserDict.
    user_dict.add(new_user)
    
    # Return success.
    return 'success'

@RPC(validation='named') 
def user_change_info(username, password, displayname, email):
    # Make a copy of the current_user.
    the_user = sc.dcp(current_user)
    
    # If the password entered doesn't match the current user password, fail.
    if password != the_user.password:
        return 'failure'
       
    # If the username entered by the user is different from the current_user
    # name (meaning they are trying to change the name)...
    if username != the_user.username:
        # Get any matching user (if any) to the new username we're trying to 
        # switch to.
        matching_user = user_dict.get_user_by_username(username)
    
        # If we have a match, fail because we don't want to rename the user to 
        # another existing user.
        if matching_user is not None:
            return 'failure'
        
    # Change the user name, display name, email, and instance label.
    the_user.username = username
    the_user.displayname = displayname
    the_user.email = email
    the_user.instance_label = username
    
    # Update the user in user_dict.
    user_dict.update(the_user)
    
    # Return success.
    return 'success'

@RPC(validation='named') 
def user_change_password(oldpassword, newpassword):
    # Make a copy of the current_user.
    the_user = sc.dcp(current_user)
    
    # If the password entered doesn't match the current user password, fail.
    if oldpassword != the_user.password:
        return 'failure' 
    
    # Change just the password.
    the_user.password = newpassword
    
    # Update the user in user_dict.
    user_dict.update(the_user)
    
    # Return success.
    return 'success'

@RPC(validation='admin')
def admin_get_user_info(username):
    # Get the matching user (if any).
    matching_user = user_dict.get_user_by_username(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    return matching_user.get_admin_front_end_repr()

@RPC(validation='admin')
def admin_delete_user(username):
    # Get the matching user (if any).
    matching_user = user_dict.get_user_by_username(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # Delete the user from the dictionary.
    user_dict.delete_by_username(username)

    # Return success.
    return 'success'

@RPC(validation='admin')
def admin_activate_account(username):
    # Get the matching user (if any).
    matching_user = user_dict.get_user_by_username(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # If the account is already activated, fail.
    if matching_user.is_active:
        return 'failure'
    
    # Activate the account.
    matching_user.is_active = True
    user_dict.update(matching_user)
    
    # Return success.
    return 'success'    

@RPC(validation='admin')
def admin_deactivate_account(username):
    # Get the matching user (if any).
    matching_user = user_dict.get_user_by_username(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # If the account is already deactivated, fail.
    if not matching_user.is_active:
        return 'failure'
    
    # Activate the account.
    matching_user.is_active = False
    user_dict.update(matching_user)
    
    # Return success.
    return 'success'

@RPC(validation='admin')
def admin_grant_admin(username):
    # Get the matching user (if any).
    matching_user = user_dict.get_user_by_username(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # If the account already has admin access, fail.
    if matching_user.is_admin:
        return 'failure'
    
    # Grant admin access.
    matching_user.is_admin = True
    user_dict.update(matching_user)
    
    # Return success.
    return 'success'

@RPC(validation='admin')
def admin_revoke_admin(username):
    # Get the matching user (if any).
    matching_user = user_dict.get_user_by_username(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # If the account has no admin access, fail.
    if not matching_user.is_admin:
        return 'failure'
    
    # Revoke admin access.
    matching_user.is_admin = False
    user_dict.update(matching_user)
    
    # Return success.
    return 'success'

@RPC(validation='admin')
def admin_reset_password(username):
    # Get the matching user (if any).
    matching_user = user_dict.get_user_by_username(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # Set the password to the desired raw password for them to use.
    raw_password = 'sciris'
    matching_user.password = sha224(raw_password).hexdigest() 
    user_dict.update(matching_user)
    
    # Return success.
    return 'success'


def make_test_users():
    # Create two test Users that can get added to a new UserDict.
    test_user = User('demo', 'demo', 'Demo', 'demo@demo.com', uid=sc.uuid('12345678123456781234567812345678'))
    test_user2 = User('admin', 'admin', 'Admin', 'admin@scirisuser.net', has_admin_rights=True, uid=sc.uuid('12345678123456781234567812345679'))
    users = [test_user, test_user2]
    return users