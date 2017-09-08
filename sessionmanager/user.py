"""
user.py -- code related to Sciris user management
    
Last update: 9/8/17 (gchadder3)
"""

#
# Imports
#

from flask_login import current_user, login_user, logout_user
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
        
    def get_id(self):
        return self.uid

myUser = User()

#
# RPC functions
#

def user_login(userName, password):
    if userName == 'newguy' and password == 'mesogreen':
        login_user(myUser)
        return 'success'
    else:
        return 'failure'
    
def user_logout():
    logout_user()
    return