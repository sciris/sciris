"""
user.py -- code related to Sciris user management
    
Last update: 2018aug20
"""

from flask import Flask, session, current_app # analysis:ignore
from flask_login import current_user, login_user, logout_user
import hashlib
import sciris as sc
from .sc_objects import User
from . import sc_rpcs as rpcs


##############################################################
### Globals
##############################################################

__all__ = ['user_dict'] # 'RPC_dict', 'RPC' not visible

user_dict = None # The UserDict object for all of the app's users.  Gets initialized by and loaded by init_users().
RPC_dict = {} # Dictionary to hold all of the registered RPCs in this module.
RPC = rpcs.makeRPCtag(RPC_dict) # RPC registration decorator factory created using call to make_RPC().




##############################################################
### Functions and RPCs
##############################################################

__all__ += ['user_login', 'user_logout', 'user_register']
__all__ += ['user_change_info', 'user_change_password', 'admin_get_user_info', 'admin_delete_user', 'admin_activate_account']
__all__ += ['admin_deactivate_account', 'admin_grant_admin', 'admin_revoke_admin', 'admin_reset_password', 'make_test_users']

        
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
    logout_user() # Log the user out and set the session to having an anonymous user.
    session.clear() # Clear the session cookie.
    return None # Return nothing.


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
    matching_user.password = hashlib.sha224(raw_password).hexdigest() 
    user_dict.update(matching_user)
    
    # Return success.
    return 'success'


def make_test_users():
    # Create two test Users that can get added to a new UserDict.
    test_user = User('demo', 'demo', 'Demo', 'demo@demo.com', uid=sc.uuid('12345678123456781234567812345678'))
    test_user2 = User('admin', 'admin', 'Admin', 'admin@scirisuser.net', has_admin_rights=True, uid=sc.uuid('12345678123456781234567812345679'))
    users = [test_user, test_user2]
    return users