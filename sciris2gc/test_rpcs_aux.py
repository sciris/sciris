"""
test_rpcs_aux.py -- xxx
    
Last update: 5/17/18 (gchadder3)
"""

# Imports
from sciris2gc.scirisapp import ScirisRPC
from functools import wraps

#
# Globals
#

# Dictionary to hold all of the registered RPCs in this module.
RPC_dict = {}

#
# Miscellaneous functions
#

# Decorator factory for registering RPCs in this module.
def register_RPC(**callerkwargs):
    def RPC_decorator(RPC_func):
        @wraps(RPC_func)
        def wrapper(*args, **kwargs):        
            RPC_func(*args, **kwargs)

        # Create the RPC and add it to the dictionary.
        RPC_dict[RPC_func.__name__] = ScirisRPC(RPC_func, **callerkwargs)
        
        return wrapper

    return RPC_decorator