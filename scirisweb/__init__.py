'''
Import the key functions required for the webapp.

Example usage:
    import scirisweb as sw
    app = sw.ScirisApp()

Version: 2018-09-08
'''

from .sc_datastore import * # analysis:ignore
from .sc_rpcs      import * # analysis:ignore
from .sc_app       import * # analysis:ignore
from .sc_objects   import * # analysis:ignore
from .sc_user      import * # analysis:ignore
from .sc_tasks     import * # analysis:ignore
from .sc_server    import * # analysis:ignore

import sciris as sc
print('Sciris Web %s (%s) -- (c) Optima Consortium' % (sc.__version__, sc.__versiondate__))