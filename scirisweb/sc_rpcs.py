"""
rpcs.py -- code related to Sciris RPCs
    
Last update: 5/23/18 (gchadder3)
"""

# Imports
from collections import OrderedDict
from functools import wraps
import numpy as np
import sciris as sc

__all__ = ['sanitize_json', 'ScirisRPC', 'makeRPCtag']



def sanitize_json(obj):
    """
    This is the main conversion function for Python data-structures into
    JSON-compatible data structures.
    Use this as much as possible to guard against data corruption!
    Args:
        obj: almost any kind of data structure that is a combination
            of list, numpy.ndarray, odicts, etc.
    Returns:
        A converted dict/list/value that should be JSON compatible
    """

    if isinstance(obj, list) or isinstance(obj, tuple):
        return [sanitize_json(p) for p in list(obj)]
    
    if isinstance(obj, np.ndarray):
        if obj.shape: return [sanitize_json(p) for p in list(obj)] # Handle most cases, incluing e.g. array([5])
        else:         return [sanitize_json(p) for p in list(np.array([obj]))] # Handle the special case of e.g. array(5)

    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}

    if isinstance(obj, sc.odict):
        result = OrderedDict()
        for k, v in obj.items():
            result[str(k)] = sanitize_json(v)
        return result

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, float):
        if np.isnan(obj): return None
        else:             return obj
        
    if isinstance(obj, np.int64):
        if np.isnan(obj): return None
        else:             return int(obj)
        
    if isinstance(obj, np.float64):
        if np.isnan(obj): return None
        else:             return float(obj)

    if isinstance(obj, unicode):
        try:    string = str(obj) # Try to convert it to ascii
        except: string = obj # Give up and use original
        return string

    if isinstance(obj, set):
        return list(obj)

    return obj


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