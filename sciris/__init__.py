# Import version information
from .version import version, versiondate

# Import core functions
from . import core

# Import web functions
try:
    from . import web
    webapptext = 'with webapp'
    E, traceback = None, None # So del command later works
except Exception as E:
    import traceback
    webapp_error = traceback.format_exc()
    webapptext = 'without webapp (see sciris.webapp_error for details)'

scirislicense = 'Sciris %s (%s)' % (version, versiondate)
print(scirislicense + ' ' + webapptext)

del scirislicense, webapptext, E, traceback # Remove unneeded variables