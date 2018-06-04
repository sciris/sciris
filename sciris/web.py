'''
Import the key functions required for the webapp.

Example usage:
	import sciris.web as sw
	app = sw.ScirisApp()

Version: 2018-06-03
'''

from .datastore import DataStore
from .rpcs import ScirisRPC, make_register_RPC
from .scirisapp import ScirisApp
from .scirisobjects import ScirisObject, ScirisCollection
from .user import User, UserDict, user_login