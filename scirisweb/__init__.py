'''
Import the key functions required for the webapp.

Example usage:
    import scirisweb as sw
    app = sw.ScirisApp()

Version: 2018-08-20
'''

from .sc_datastore import *
from .sc_rpcs import *
from .sc_app import *
from .sc_objects import *
from .sc_user import *
from .sc_server import *

import sciris as sc
print('Sciris Web %s (%s) -- (c) 2018 by the Optima Consortium (info@sciris.org)' % (sc.__version__, sc.__versiondate__))