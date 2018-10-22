"""
user.py -- code related to Sciris user management
    
Last update: 2018sep20
"""

from flask import Flask, session, current_app as app # analysis:ignore
from flask_login import current_user, login_user, logout_user
import six
import sciris as sc
from . import sc_rpcs as rpcs


##############################################################
### Globals
##############################################################

__all__ = ['RPC_dict']

RPC_dict = {}
RPC = rpcs.RPCwrapper(RPC_dict) # RPC registration decorator factory created using call to make_RPC().



##############################################################
### Classes
##############################################################

__all__ += ['User']


class User(sc.prettyobj):
    '''
    A Sciris user.
                    
    Attributes:
        is_authenticated (bool) -- is this user authenticated? (attribute required by Flask-Login)
        is_active (bool)        -- is this user's account active? (attribute required by Flask-Login)
        is_anonymous (bool)     -- is this user considered anonymous?  (attribute required by Flask-Login)
        is_admin (bool)         -- does this user have admin rights?
        username (str)          -- the username used to log in
        displayname (str)       -- the user's name, which gets displayed in the  browser
        email (str)             -- the user's email     
        password (str)          -- the user's SHA224-hashed password
        raw_password (str)      -- the user's unhashed password
    '''
    
    def  __init__(self, username=None, password=None, displayname=None, email=None, uid=None, raw_password=None, is_admin=False):
        # Handle general properties
        if not username:    username    = 'default'
        if not password:    password    = 'default'
        if not displayname: displayname = username
        if not uid:         uid         = sc.uuid()
        
        # Set user-specific properties
        self.username         = username # Set the username.
        self.displayname      = displayname # Set the displayname (what the browser will show).
        self.email            = email  # Set the user's email.
        self.uid              = uid # The user's UID
        self.is_authenticated = True # Set the user to be authentic.
        self.is_active        = True # Set the account to be active.
        self.is_anonymous     = False # The user is not anonymous.
        self.is_admin         = is_admin # Set whether this user has admin rights.
        
        # Handle the password
        if raw_password is not None:
            if six.PY3: raw_password = raw_password.encode('utf-8')
            password = sc.sha(raw_password).hexdigest()
        self.password = password
        return None
    
    
    def get_id(self):
        ''' Method required by Flask-login '''
        return self.username
    
    
    def show(self, verbose=False):
        ''' Display the user's properties '''
        if not verbose: # Simply display the username and UID
            print('Username: %s; UID: %s' % (self.username, self.uid))
        else: # Or full information
            print('---------------------')
            print('        Username: %s' % self.username)
            print('    Display name: %s' % self.displayname)
            print(' Hashed password: %s' % self.password)
            print('   Email address: %s' % self.email)
            print('             UID: %s' % self.uid)
            print('Is authenticated: %s' % self.is_authenticated)
            print('       Is active: %s' % self.is_active)
            print('    Is anonymous: %s' % self.is_anonymous)
            print('        Is admin: %s' % self.is_admin)
            print('---------------------')
        return None
    
    
    def jsonify(self, verbose=False):
        ''' Return a JSON-friendly representation of a user '''
        output = {'user':
                         {'username':    self.username, 
                          'displayname': self.displayname, 
                          'email':       self.email,
                          'uid':         self.uid}
                 }
        if verbose:
            output['user'].update({'is_authenticated': self.is_authenticated,
                                   'is_active':        self.is_active,
                                   'is_anonymous':     self.is_anonymous,
                                   'is_admin':         self.is_admin,
                                   'created':          self.created,
                                   'modified':         self.modified[-1]})
        return output



##############################################################
### Functions and RPCs
##############################################################

__all__ += ['save_user', 'load_user', 'user_login', 'user_logout', 'user_register']
__all__ += ['user_change_info', 'user_change_password', 'admin_delete_user', 'admin_activate_account']
__all__ += ['admin_deactivate_account', 'admin_grant_admin', 'admin_revoke_admin', 'admin_reset_password', 'make_default_users']



def save_user(user):
    ''' Save the specified user to the DataStore '''
    app.datastore.saveuser(user)
    return None


def load_user(username=None):
    ''' Load the currently active user from the DataStore '''
    if username is None:
        try: 
            username = current_user.get_id()
        except Exception as E:
            errormsg = 'No username supplied and could not get current user: %s' % str(E)
            raise Exception(errormsg)
    user = app.datastore.loaduser(username)
    return user


@RPC()
def user_login(username, password, verbose=False):  
    matching_user = app.datastore.loaduser(username, die=False) # Get the matching user (if any).
    
    # If we have a match and the password matches, and the account is active, also, log in the user and return success; otherwise, return failure.
    if matching_user and matching_user.password==password and matching_user.is_active:
        if verbose: print('User %s found, logging in' % username)
        login_user(matching_user) # Log the user in.
        matching_user.show(verbose=verbose)
        return 'success'
    else:
        if   not matching_user:                    errormsg = 'Login failed: user "%s" does not exist.' % username
        elif not matching_user.password==password: errormsg = 'Login failed: password for user "%s" is incorrect.' % username
        elif not matching_user.is_active:          errormsg = 'Login failed: user "%s" is inactive.' % username
        else:                                      errormsg = 'Unknown error'
        if verbose: print('User %s not found, not logging in (%s)' % (username, errormsg))
        return errormsg
    
    
@RPC(validation='named')
def user_logout():
    logout_user() # Log the user out and set the session to having an anonymous user.
    session.clear() # Clear the session cookie.
    return None # Return nothing.


@RPC()
def user_register(username, password, displayname, email): 
    matching_user = app.datastore.loaduser(username, die=False) # Get the matching user (if any).
    if matching_user is not None: # If we have a match, fail because we don't want to register an existing user.
        errormsg = 'User registration failed: user "%s" already exists' % username
        return errormsg
    new_user = User(username=username, password=password, displayname=displayname, email=email) # Create a new User object with the new information.
    new_user.is_active = app.config['REGISTER_AUTOACTIVATE'] # Optionally set the user to be inactive (so an admin user needs to activate it)
    app.datastore.saveuser(new_user) # Save the user
    return 'success' # Return success.


@RPC(validation='named')
def user_change_info(username, password, displayname, email):
    the_user = app.datastore.loaduser(current_user.username) # Reload the current user from the database
    if password != the_user.password: # If the password entered doesn't match the current user password, fail.
        errormsg = 'User info change failed: password for user "%s" is incorrect.' % username
        return errormsg
       
    # If the username entered by the user is different from the current_user name (meaning they are trying to change the name)...
    if username != the_user.username:
        matching_user = app.datastore.loaduser(username) # Get any matching user (if any) to the new username we're trying to switch to.
        if matching_user is not None: # If we have a match, fail because we don't want to rename the user to another existing user.
            errormsg = 'User rename failed: user "%s" already exists.' % username
            return errormsg
        
    # Change the user name, display name, email, and instance label.
    the_user.username    = username
    the_user.displayname = displayname
    the_user.email       = email
    
    # Update the user in user_dict.
    app.datastore.saveuser(the_user, overwrite=True)
    return 'success'


@RPC(validation='named') 
def user_change_password(oldpassword, newpassword):
    the_user = app.datastore.loaduser(current_user.username) # Reload the current user from the database
    
    # If the password entered doesn't match the current user password, fail.
    if oldpassword != the_user.password:
        errormsg = 'Password change failed: password for user "%s" is incorrect.' % the_user.username
        return errormsg
    
    the_user.password = newpassword # Change just the password.
    app.datastore.saveuser(the_user, overwrite=True) # Update the user in user_dict.
    return 'success'


@RPC(validation='admin')
def admin_delete_user(username):
    matching_user = app.datastore.loaduser(username) # Get the matching user (if any).
    if matching_user is None: return 'failure' # If we don't have a match, fail.
    app.datastore.delete(objtype='user', uid=username) # Delete the user from the dictionary.
    return 'success'


@RPC(validation='admin')
def admin_activate_account(username):
    # Get the matching user (if any).
    matching_user = app.datastore.loaduser(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # If the account is already activated, fail.
    if matching_user.is_active:
        return 'failure'
    
    # Activate the account.
    matching_user.is_active = True
    app.datastore.saveuser(matching_user, overwrite=True)
    
    # Return success.
    return 'success'    

@RPC(validation='admin')
def admin_deactivate_account(username):
    # Get the matching user (if any).
    matching_user = app.datastore.loaduser(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # If the account is already deactivated, fail.
    if not matching_user.is_active:
        return 'failure'
    
    # Activate the account.
    matching_user.is_active = False
    app.datastore.saveuser(matching_user, overwrite=True)
    
    # Return success.
    return 'success'

@RPC(validation='admin')
def admin_grant_admin(username):
    # Get the matching user (if any).
    matching_user = app.datastore.loaduser(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # If the account already has admin access, fail.
    if matching_user.is_admin:
        return 'failure'
    
    # Grant admin access.
    matching_user.is_admin = True
    app.datastore.saveuser(matching_user, overwrite=True)
    
    # Return success.
    return 'success'

@RPC(validation='admin')
def admin_revoke_admin(username):
    # Get the matching user (if any).
    matching_user = app.datastore.loaduser(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # If the account has no admin access, fail.
    if not matching_user.is_admin:
        return 'failure'
    
    # Revoke admin access.
    matching_user.is_admin = False
    app.datastore.saveuser(matching_user, overwrite=True)
    
    # Return success.
    return 'success'

@RPC(validation='admin')
def admin_reset_password(username):
    # Get the matching user (if any).
    matching_user = app.datastore.loaduser(username)
    
    # If we don't have a match, fail.
    if matching_user is None:
        return 'failure'
    
    # Set the password to the desired raw password for them to use.
    raw_password = 'sciris'
    matching_user.password = sc.sha(raw_password).hexdigest() 
    app.datastore.saveuser(matching_user, overwrite=True)
    
    # Return success.
    return 'success'


@RPC(validation='named') 
def get_current_user_info():
    return current_user.jsonify()


def make_default_users(app, include_admin=False):
    # Create two test Users that can get added
    test_user = User(username='demo', raw_password='demo', uid='abcdef0123456789')
    users = [test_user]
    if include_admin:
        admin_user = User(username='admin', raw_password='admin', uid='00112233445566778899', is_admin=True)
        users.append(admin_user)
    for user in users:
        app.datastore.saveuser(user, overwrite=False, die=False) # Save the user
    return users