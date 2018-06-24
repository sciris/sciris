# Import version information
from .version import version, versiondate

# Import core functions
from . import core

# Import web functions
try:
    from . import web
    webtext = 'with web library'
except Exception as webapp_exception:
    import traceback as _traceback
    web_error = _traceback.format_exc()
    webtext = 'without web library (see sciris.web_error for details)'

scirislicense = 'Sciris %s (%s)' % (version, versiondate)
print(scirislicense + ' ' + webtext)

del scirislicense, webtext # Remove unneeded variables