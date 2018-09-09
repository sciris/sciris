"""
rpcs.py -- code related to Sciris RPCs
    
Last update: 5/23/18 (gchadder3)
"""

# Imports

from functools import wraps

__all__ = ['ScirisRPC', 'makeRPCtag']




class ScirisRPC(object):
    '''
    Validation type:
        'none' : no validation required
        'any':   any login validates
        'named': any non-anonymous user validates
        'admin': any admin login validates
        'user <name>': being logged in as <name> validates (TODO)
        'disabled': presently disabled for clients
    '''
    def __init__(self, call_func, call_type='normal', override=False, validation='none'):
        self.call_func  = call_func
        self.funcname   = call_func.__name__
        self.call_type  = call_type
        self.override   = override
        self.validation = validation 
            

        
def makeRPCtag(RPC_dict=None, **callerkwargs):
    def RPC_decorator_factory(**callerkwargs):
        def RPC_decorator(RPC_func):
            @wraps(RPC_func)
            def wrapper(*args, **kwargs):        
                output = RPC_func(*args, **kwargs)
                return output
            RPC_dict[RPC_func.__name__] = ScirisRPC(RPC_func, **callerkwargs) # Create the RPC and add it to the dictionary.
            return wrapper
        return RPC_decorator
    return RPC_decorator_factory