from . import utils # Import utilities -- these will always be required
from . import core
from .utils import odict # Import odict class
from .utils import dataframe # Import dataframe class
import os as _os
import matplotlib as _mpl

if _os.environ.get('DISPLAY','') == '':
    print('Sciris: no display found, using non-interactive Agg backend')
    _mpl.use('Agg')