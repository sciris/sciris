"""
rpcs.py -- code related to Sciris RPCs
    
Last update: 5/23/18 (gchadder3)
"""

# Imports
from functools import wraps

#
# Classes
#

class ScirisRPC(object):
    def __init__(self, call_func, call_type='normal', override=False, 
        validation_type='none'):
        self.call_func = call_func
        self.funcname = call_func.__name__
        self.call_type = call_type
        self.override = override
        self.validation_type = validation_type  
            # 'none' : no validation required
            # 'any user': any login validates
            # 'nonanonymous user': any non-anonymous user validates
            # 'admin user': any admin login validates
            # 'user <name>': being logged in as <name> validates (TODO)
            # 'disabled': presently disabled for clients

#
# Functions
#
        
def make_register_RPC(RPC_dict=None, **callerkwargs):
    def RPC_decorator_factory(**callerkwargs):
        def RPC_decorator(RPC_func):
            @wraps(RPC_func)
            def wrapper(*args, **kwargs):        
                RPC_func(*args, **kwargs)
    
            # Create the RPC and add it to the dictionary.
            RPC_dict[RPC_func.__name__] = ScirisRPC(RPC_func, **callerkwargs)
            
            return wrapper
    
        return RPC_decorator
    
    return RPC_decorator_factory