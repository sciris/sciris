'''
Import the key functions required for the webapp.

Example usage:
    import scirisweb as sw
    app = sw.ScirisApp()

Version: 2018sep22
'''

from .sc_rpcs      import * # analysis:ignore
from .sc_users     import * # analysis:ignore
from .sc_tasks     import * # analysis:ignore
from .sc_datastore import * # analysis:ignore
from .sc_app       import * # analysis:ignore
from .sc_server    import * # analysis:ignore

import sciris as sc
print(sc.__license__)