"""
user.py -- code related to Sciris user management
    
Last update: 9/8/17 (gchadder3)
"""

#
# Imports
#

from flask_login import current_user, login_user, logout_user
from hashlib import sha224
import uuid

#
# Classes
#

class User(object):
    def  __init__(self):
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
        self.uid = uuid.UUID('12345678123456781234567812345678')
        
        self.username = 'newguy'
        
        # Store the SHA224 hashed version of the password in hex form.
        rawPassword = 'mesogreen'
        self.password = sha224(rawPassword).hexdigest()
        
    def get_id(self):
        return self.uid

myUser = User()

#
# RPC functions
#

def user_login(userName, password):
    if userName == myUser.username and password == myUser.password:
        login_user(myUser)
        return 'success'
    else:
        return 'failure'
    
def user_logout():
    logout_user()
    return