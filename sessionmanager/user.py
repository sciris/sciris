"""
user.py -- code related to Sciris user management
    
Last update: 10/3/17 (gchadder3)
"""

#
# Imports
#

from flask import session, request, current_app
from flask_login import current_user, login_user, logout_user
from hashlib import sha224
from numpy import argsort
import uuid
import copy
import scirisobjects as sobj

#
# Globals
#

# The UserDict object for all of the app's users.  Gets initialized by
# and loaded by init_users().
theUserDict = None

#
# Classes
#

class User(sobj.ScirisObject):
    """
    A Sciris user.
    
    Methods:
        __init__(theUsername: str, thePassword: str, theDisplayName: str, 
            theEmail: str [''], hasAdminRights: bool [False], 
            theUID: UUID [None], hashThePassword: bool [True]): 
            void -- constructor
        loadFromCopy(otherObject): void -- assuming otherObject is another 
            object of our type, copy its contents to us (calls the 
            ScirisObject superclass version of this method also)            
        get_id(): UUID -- get the unique ID of this user (method required by 
            Flask-Login)    
        show(): void -- print the contents of the object
        getUserFrontEndRepr(): dict -- get a JSON-friendly dictionary 
            representation of the object state the front-end uses for non-
            admin purposes
        getAdminFrontEndRepr(): dict -- get a JSON-friendly dictionary
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
        >>> myUser = User('newguy', 'mesogreen', 'Ozzy Mandibulus',  \
            'tastybats@yahoo.com', theUID=uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def  __init__(self, theUsername, thePassword, theDisplayName, theEmail='', 
        hasAdminRights=False, theUID=None, hashThePassword=True):
        # Set superclass parameters.
        super(User, self).__init__(theUID)
        
        # Set the user to be authentic.
        self.is_authenticated = True
        
        # Set the account to be active.
        self.is_active = True
        
        # The user is not anonymous.
        self.is_anonymous = False
        
        # Set the username.
        self.username = theUsername
        
        # Set the raw password and use SHA224 to get the hashed version in 
        # hex form.
        rawPassword = thePassword
        if hashThePassword:
            self.password = sha224(rawPassword).hexdigest() 
        else:
            self.password = thePassword
        
        # Set the displayname (what the browser will show).
        self.displayname = theDisplayName
        
        # Set the user's email.
        self.email = theEmail
        
        # Set whether this user has admin rights.
        self.is_admin = hasAdminRights
        
        # Set the type prefix to 'user'.
        self.typePrefix = 'user'
        
        # Set the file suffix to '.usr'.
        self.fileSuffix = '.usr'
        
        # Set the instance label to the username.
        self.instanceLabel = theUsername 
        
    def loadFromCopy(self, otherObject):
        if type(otherObject) == type(self):
            # Do the superclass copying.
            super(User, self).loadFromCopy(otherObject)
            
            self.is_authenticated = otherObject.is_authenticated
            self.is_active = otherObject.is_active
            self.is_anonymous = otherObject.is_anonymous
            self.username = otherObject.username
            self.password = otherObject.password
            self.displayname = otherObject.displayname
            self.email = otherObject.email
            self.is_admin = otherObject.is_admin
            
    def get_id(self):
        return self.uid
    
    def show(self):
        # Show superclass attributes.
        super(User, self).show()  
        
        print '---------------------'

        print 'Username: %s' % self.username
        print 'Displayname: %s' % self.displayname
        print 'Hashed password: %s' % self.password
        print 'Email: %s' % self.email
        if self.is_authenticated:
            print 'Is authenticated?: Yes'
        else:
            print 'Is authenticated?: No'
        if self.is_active:
            print 'Account active?: Yes'
        else:
            print 'Account active?: No'
        if self.is_anonymous:
            print 'Is anonymous?: Yes'
        else:
            print 'Is anonymous?: No'
        if self.is_admin:
            print 'Has admin rights?: Yes'
        else:
            print 'Has admin rights?: No' 
            
    def getUserFrontEndRepr(self):
        objInfo = {
            'user': {
                'instancelabel': self.instanceLabel,
                'username': self.username, 
                'displayname': self.displayname, 
                'email': self.email,
                'admin': self.is_admin                
            }
        }
        return objInfo
    
    def getAdminFrontEndRepr(self):
        objInfo = {
            'user': {
                'UID': self.uid.hex, 
                'typeprefix': self.typePrefix, 
                'filesuffix': self.fileSuffix, 
                'instancelabel': self.instanceLabel,
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
        return objInfo             
            
class UserDict(sobj.ScirisCollection):
    """
    A dictionary of Sciris users.
    
    Methods:
        __init__(theUID: UUID [None], theTypePrefix: str ['userdict'], 
            theFileSuffix: str ['.ud'], 
            theInstanceLabel: str ['Users Dictionary']): void -- constructor
        loadFromCopy(otherObject): void -- assuming otherObject is another 
            object of our type, copy its contents to us (calls the 
            ScirisCollection superclass version of this method also)           
        getUserByUID(theUID: UUID or str): User or None -- returns the User  
            object pointed to by theUID
        getUserByUsername(theUsername: str): User or None -- return the User
            object pointed to by the username
        add(theUser: User): void -- add a User to the dictionary and update
            the dictionary's DataStore state
        update(theUser: User): void -- update a User in the dictionary and 
            update the dictionary's DataStore state
        deleteByUID(theUID: UUID or str): void -- delete a User in the dictionary
            selected by the UID, and update the dictionary's DataStore state
        deleteByUsername(theUsername: str): void -- delete a User in the 
            dictionary selected by a username, and update the dictionary's 
            DataStore state
        deleteAll(): void -- delete the entire contents of the UserDict and 
            update the dictionary's DataStore state            
        getUserFrontEndRepr(): dict -- get a JSON-friendly dictionary 
            representation of the collection state the front-end uses for non-
            admin purposes
        getAdminFrontEndRepr(): dict -- get a JSON-friendly dictionary
            representation of the collection state the front-end uses for admin
            purposes
                    
    Attributes:
        usernameHashes (dict) -- a dict mapping usernames to UIDs, so either
            indexing by UIDs or usernames can be fast
        
    Usage:
        >>> theUserDict = UserDict(uuid.UUID('12345678123456781234567812345678'))                      
    """
    
    def __init__(self, theUID, theTypePrefix='userdict', theFileSuffix='.ud', 
        theInstanceLabel='Users Dictionary'):
        # Set superclass parameters.
        super(UserDict, self).__init__(theUID, theTypePrefix, theFileSuffix, 
             theInstanceLabel)
        
        # Create the Python dict to hold the hashes from usernames to the UIDs.
        self.usernameHashes = {}
        
    def loadFromCopy(self, otherObject):
        if type(otherObject) == type(self):
            # Do the superclass copying.
            super(UserDict, self).loadFromCopy(otherObject)
            
            self.usernameHashes = otherObject.usernameHashes
            
    def getUserByUID(self, theUID):
        return self.getObjectByUID(theUID)
    
    def getUserByUsername(self, theUsername):
        # Get the user's UID matching the username.
        userIndex = self.usernameHashes.get(theUsername, None)
        
        # If we found at match, use the UID to try to fetch the user; 
        # otherwise, return None.
        if userIndex is not None:
            return self.getUserByUID(userIndex)
        else:
            return None
        
    def add(self, theUser):
        # Add the object to the hash table, keyed by the UID.
        self.theObjectDict[theUser.uid] = theUser
        
        # Add the username hash for this user.
        self.usernameHashes[theUser.username] = theUser.get_id()
        
        # Update our DataStore representation if we are there. 
        if self.inDataStore():
            self.updateDataStore()
    
    def update(self, theUser):
        # Get the old username.
        oldUsername = self.theObjectDict[theUser.get_id()].username
        
        # If we new username is different than the old one, delete the old 
        # usernameHash.
        if theUser.username != oldUsername:
            del self.usernameHashes[oldUsername]
 
        # Add the user to the hash table, keyed by the UID.
        self.theObjectDict[theUser.get_id()] = theUser
        
        # Add the username hash for this user.
        self.usernameHashes[theUser.username] = theUser.get_id()
       
        # Update our DataStore representation if we are there. 
        if self.inDataStore():
            self.updateDataStore()
    
    def deleteByUID(self, theUID):
        # Make sure the argument is a valid UUID, converting a hex text to a
        # UUID object, if needed.        
        validUID = sobj.getValidUUID(theUID)
        
        # If we have a valid UUID...
        if validUID is not None:
            # Get the object pointed to by the UID.
            theObject = self.theObjectDict[validUID]
            
            # If a match is found...
            if theObject is not None:
                # Remove entries from both theUserDict and usernameHashes 
                # attributes.
                theUsername = theObject.username
                del self.theObjectDict[validUID]
                del self.usernameHashes[theUsername]                
                
                # Update our DataStore representation if we are there. 
                if self.inDataStore():
                    self.updateDataStore()
        
    def deleteByUsername(self, theUsername):
        # Get the UID of the user matching theUsername.
        userIndex = self.usernameHashes.get(theUsername, None)
        
        # If we found a match, call deleteByUID to complete the deletion.
        if userIndex is not None:
            self.deleteByUID(userIndex)
    
    def deleteAll(self):
        # Reset the Python dicts to hold the user objects and hashes from 
        # usernames to the UIDs.
        self.theObjectDict = {}
        self.usernameHashes = {}  
        
        # Update our DataStore representation if we are there. 
        if self.inDataStore():
            self.updateDataStore()
            
    def getUserFrontEndRepr(self):  
        usersInfo = [self.theObjectDict[theKey].getUserFrontEndRepr() 
            for theKey in self.theObjectDict]
        return usersInfo
        
    def getAdminFrontEndRepr(self):
        usersInfo = [self.theObjectDict[theKey].getAdminFrontEndRepr() 
            for theKey in self.theObjectDict]
        return usersInfo   

# use this to figure out how to sort the entries      
#def get_user_summaries():
#    raw_users = [parse_user_record(q) for q in UserDb.query.all()]
#    user_names = [user['username'] for user in raw_users]
#    sort_order = argsort(user_names)
#    sorted_users = [raw_users[o] for o in sort_order]
#    users_dict = {'users': sorted_users}
#    return users_dict    
    
#
# RPC functions
#

def user_login(userName, password):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'loginRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Get the matching user (if any).
    matchingUser = theUserDict.getUserByUsername(userName)
    
    # If we have a match and the password matches, and the account is active,
    # also, log in the user and return success; otherwise, return failure.
    if matchingUser is not None and matchingUser.password == password and \
        matchingUser.is_active:
        # Log the user in.
        login_user(matchingUser)
        
        return 'success'
    else:
        return 'failure'
    
def user_logout():
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'logoutRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Log the user out and set the session to having an anonymous user.
    logout_user()
    
    # Clear the session cookie.
    session.clear()
    
    # Return nothing.
    return None

def get_current_user_info():
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'currentUserRPC':
        return {'error': 'Unauthorized RPC'}
    
    return current_user.getUserFrontEndRepr()

def user_register(userName, password, displayname, email):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'registrationRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Get the matching user (if any).
    matchingUser = theUserDict.getUserByUsername(userName)
    
    # If we have a match, fail because we don't want to register an existing 
    # user.
    if matchingUser is not None:
        return 'failure'
    
    # Create a new User object with the new information.
    newUser = User(userName, password, displayname, email, hashThePassword=False)
    
    # Set the user to be inactive (so an admin user needs to activate it, 
    # even though the account has been created) or active (so that no admin
    # action is necessary), according to the REGISTER_AUTOACTIVATE config 
    # parameter.
    newUser.is_active = current_app.config['REGISTER_AUTOACTIVATE']
    
    # Put the user right into the UserDict.
    theUserDict.add(newUser)
    
    # Return success.
    return 'success'

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
        matchingUser = theUserDict.getUserByUsername(userName)
    
        # If we have a match, fail because we don't want to rename the user to 
        # another existing user.
        if matchingUser is not None:
            return 'failure'
        
    # Change the user name, display name, email, and instance label.
    theUser.username = userName
    theUser.displayname = displayname
    theUser.email = email
    theUser.instanceLabel = userName
    
    # Update the user in theUserDict.
    theUserDict.update(theUser)
    
    # Return success.
    return 'success'

def user_change_password(oldpassword, newpassword):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'changePasswordRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Make a copy of the current_user.
    theUser = copy.copy(current_user)
    
    # If the password entered doesn't match the current user password, fail.
    if oldpassword != theUser.password:
        return 'failure' 
    
    # Change just the password.
    theUser.password = newpassword
    
    # Update the user in theUserDict.
    theUserDict.update(theUser)
    
    # Return success.
    return 'success'

def get_all_users():
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'userListRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Return success.
    return theUserDict.getAdminFrontEndRepr()

def admin_get_user_info(userName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'specificUserRPC':
        return {'error': 'Unauthorized RPC'}   
    
    # Get the matching user (if any).
    matchingUser = theUserDict.getUserByUsername(userName)
    
    # If we don't have a match, fail.
    if matchingUser is None:
        return 'failure'
    
    return matchingUser.getAdminFrontEndRepr()

def admin_delete_user(userName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'specificUserRPC':
        return {'error': 'Unauthorized RPC'}   
    
    # Get the matching user (if any).
    matchingUser = theUserDict.getUserByUsername(userName)
    
    # If we don't have a match, fail.
    if matchingUser is None:
        return 'failure'
    
    # Delete the user from the dictionary.
    theUserDict.deleteByUsername(userName)
    
    # Delete other resources allocated for the user.

    # Return success.
    return 'success'

def admin_activate_account(userName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'specificUserRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Get the matching user (if any).
    matchingUser = theUserDict.getUserByUsername(userName)
    
    # If we don't have a match, fail.
    if matchingUser is None:
        return 'failure'
    
    # If the account is already activated, fail.
    if matchingUser.is_active:
        return 'failure'
    
    # Activate the account.
    matchingUser.is_active = True
    theUserDict.update(matchingUser)
    
    # Return success.
    return 'success'    

def admin_deactivate_account(userName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'specificUserRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Get the matching user (if any).
    matchingUser = theUserDict.getUserByUsername(userName)
    
    # If we don't have a match, fail.
    if matchingUser is None:
        return 'failure'
    
    # If the account is already deactivated, fail.
    if not matchingUser.is_active:
        return 'failure'
    
    # Activate the account.
    matchingUser.is_active = False
    theUserDict.update(matchingUser)
    
    # Return success.
    return 'success'

def admin_grant_admin(userName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'specificUserRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Get the matching user (if any).
    matchingUser = theUserDict.getUserByUsername(userName)
    
    # If we don't have a match, fail.
    if matchingUser is None:
        return 'failure'
    
    # If the account already has admin access, fail.
    if matchingUser.is_admin:
        return 'failure'
    
    # Grant admin access.
    matchingUser.is_admin = True
    theUserDict.update(matchingUser)
    
    # Return success.
    return 'success'

def admin_revoke_admin(userName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'specificUserRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Get the matching user (if any).
    matchingUser = theUserDict.getUserByUsername(userName)
    
    # If we don't have a match, fail.
    if matchingUser is None:
        return 'failure'
    
    # If the account has no admin access, fail.
    if not matchingUser.is_admin:
        return 'failure'
    
    # Revoke admin access.
    matchingUser.is_admin = False
    theUserDict.update(matchingUser)
    
    # Return success.
    return 'success'

def admin_reset_password(userName):
    # Check (for security purposes) that the function is being called by the 
    # correct endpoint, and if not, fail.
    if request.endpoint != 'specificUserRPC':
        return {'error': 'Unauthorized RPC'}
    
    # Get the matching user (if any).
    matchingUser = theUserDict.getUserByUsername(userName)
    
    # If we don't have a match, fail.
    if matchingUser is None:
        return 'failure'
    
    # Set the password to the desired raw password for them to use.
    rawPassword = 'sciris'
    matchingUser.password = sha224(rawPassword).hexdigest() 
    theUserDict.update(matchingUser)
    
    # Return success.
    return 'success'

#
# Script code
#

# Create two test Users that can get added to a new UserDict.
testUser = User('newguy', 'mesogreen', 'Ozzy Mandibulus', 'tastybats@yahoo.com', \
    theUID=uuid.UUID('12345678123456781234567812345678'))
testUser2 = User('admin', 'mesoawesome', 'Admin Dude', 'admin@scirisuser.net', \
    hasAdminRights=True, theUID=uuid.UUID('12345678123456781234567812345679'))