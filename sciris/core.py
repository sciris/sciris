'''
Import the core Sciris functionality.

Example usage:
	import sciris.core as sc
	my_odict = sc.odict()

Version: 2018-06-03
'''

print('Use of "import sciris.core as sc" is deprecated. Please use "import sciris as sc" instead.')
from .utils import *
from .colortools import *
from .odict import *
from .dataframe import *
from .fileio import loadspreadsheet, export_xlsx # WARNING, make consistent
from .asd import asd