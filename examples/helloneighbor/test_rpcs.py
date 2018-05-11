"""
test_rpcs.py -- A testing file for installing RPCs outside of the caller of
    scirisapp.py
    
Last update: 5/11/18 (gchadder3)
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
# TODO: If we can, it would be nice to remove this boilerplate by defining 
# this decorator factory somehow in scirisapp.py.
def register_RPC(**callerkwargs):
    def RPC_decorator(RPC_func):
        @wraps(RPC_func)
        def wrapper(*args, **kwargs):        
            RPC_func(*args, **kwargs)

        # Create the RPC and add it to the dictionary.
        RPC_dict[RPC_func.__name__] = ScirisRPC(RPC_func, **callerkwargs)
        
        return wrapper

    return RPC_decorator

#
# RPC functions
#
    
@register_RPC(override=True)  # Override any other test_func() defined before
def test_func():
    return '<h1>Test Me NOW!</h1>'

@register_RPC()
def test_func2():
    return '<h1>Test Me Forever!</h1>'

@register_RPC()
def test_func3():
    return '<h1>Test Me Testily!</h1>'