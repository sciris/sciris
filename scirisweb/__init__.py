'''
Import the key functions required for the webapp.

Example usage:
    import scirisweb as sw
    app = sw.ScirisApp()

Version: 2018-08-17
'''

from .datastore import DataStore
from .rpcs import ScirisRPC, make_register_RPC
from .scirisapp import ScirisApp, json_sanitize_result
from .scirisobjects import ScirisObject, ScirisCollection
from .user import User, UserDict, user_login
from .quickserver import normalize_obj, serve, browser