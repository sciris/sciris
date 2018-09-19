"""
user.py -- code related to Sciris user management
    
Last update: 2018sep19
"""

from flask import Flask, session, current_app as app # analysis:ignore
from flask_login import current_user, login_user, logout_user
import hashlib
import sciris as sc
from .sc_objects import User
from . import sc_rpcs as rpcs


##############################################################
### Globals
##############################################################

__all__ = ['RPC_dict']

RPC_dict = {}
RPC = rpcs.makeRPCtag(RPC_dict) # RPC registration decorator factory created using call to make_RPC().


##############################################################
### Functions and RPCs
##############################################################

__all__ += ['user_login', 'user_logout', 'user_register']
__all__ += ['user_change_info', 'user_change_password', 'admin_delete_user', 'admin_activate_account']
__all__ += ['admin_deactivate_account', 'admin_grant_admin', 'admin_revoke_admin', 'admin_reset_password', 'make_test_users']

        
@RPC()
def user_login(username, password):  
    matching_user = app.datastore.loaduser(username) # Get the matching user (if any).
    
    # If we have a match and the password matches, and the account is active, also, log in the user and return success; otherwise, return failure.
    if matching_user and matching_user.password==password and matching_user.is_active:
        login_user(matching_user) # Log the user in.
    else:
        if not matching_user:
            errormsg = 'Could not log in: user "%s" does not exist' % username
        elif not matching_user.password==password: 
            errormsg = 'Could not log in: password for user "%s" is incorrect' % username
        elif not matching_user.is_active:
            errormsg = 'Could not log in: user "%s" is inactive' % username
        raise Exception(errormsg)
    return 'success' 
    
    
@RPC(validation='named')
def user_logout():
    logout_user() # Log the user out and set the session to having an anonymous user.
    session.clear() # Clear the session cookie.
    return None # Return nothing.


@RPC()
def user_register(username, password, displayname, email): 
    matching_user = app.datastore.loaduser(username, die=False) # Get the matching user (if any).
    if matching_user is not None: # If we have a match, fail because we don't want to register an existing user.
        errormsg = 'Could not register user: user "%s" already exists' % username
        raise Exception(errormsg)
    new_user = User(username=username, password=password, displayname=displayname, email=email) # Create a new User object with the new information.
    new_user.is_active = app.config['REGISTER_AUTOACTIVATE'] # Optionally set the user to be inactive (so an admin user needs to activate it)
    app.datastore.saveuser(new_user) # Save the user
    return 'success' # Return success.


@RPC(validation='named')
def user_change_info(username, password, displayname, email):
    the_user = sc.dcp(current_user) # Make a copy of the current_user.
    if password != the_user.password: # If the password entered doesn't match the current user password, fail.
        errormsg = 'Could not change user info: password for user "%s" is incorrect' % username
        raise Exception(errormsg)
       
    # If the username entered by the user is different from the current_user name (meaning they are trying to change the name)...
    if username != the_user.username:
        matching_user = app.datastore.loaduser(username) # Get any matching user (if any) to the new username we're trying to switch to.
        if matching_user is not None: # If we have a match, fail because we don't want to rename the user to another existing user.
            errormsg = 'Could not rename user: user "%s" already exists' % username
            raise Exception(errormsg)
        
    # Change the user name, display name, email, and instance label.
    the_user.username    = username
    the_user.displayname = displayname
    the_user.email       = email
    
    # Update the user in user_dict.
    app.datastore.saveuser(the_user, overwrite=True)
    return 'success'


@RPC(validation='named') 
def user_change_password(oldpassword, newpassword):
    the_user = sc.dcp(current_user) # Make a copy of the current_user.
    
    # If the password entered doesn't match the current user password, fail.
    if oldpassword != the_user.password:
        errormsg = 'Could not change password: password for user "%s" is incorrect' % the_user.username
        raise Exception(errormsg)
    
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
    matching_user.password = hashlib.sha224(raw_password).hexdigest() 
    app.datastore.saveuser(matching_user, overwrite=True)
    
    # Return success.
    return 'success'


def make_test_users(include_admin=False):
    # Create two test Users that can get added
    test_user = User('demo', 'demo', 'Demo', 'demo@sciris.org', uid='abcdef0123456789')
    users = [test_user]
    if include_admin:
        admin_user = User('admin', 'admin', 'Admin', 'admin@sciris.org', is_admin=True, uid='00112233445566778899')
        users.append(admin_user)
    return users