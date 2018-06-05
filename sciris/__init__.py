# Import core functions
from . import core

# Import web functions
try:
    from . import web
except:
    print('Warning: was not able to import web components of Sciris')