from . import utils # Import utilities -- these will always be required
from .utils import odict # Import odict class
from .utils import dataframe # Import dataframe class

#from . import version
#print('Sciris v%s (%s) loaded' % (version.version, version.versiondate))

## To allow graphs to be created without a DISPLAY variable -- WARNING, maybe a cleaner way of doing this?
#try:
#    import os
#    import matplotlib.pyplot as plt
#    if 'DISPLAY' in os.environ:
#        if not os.environ['DISPLAY']:
#            plt.switch_backend('agg')
#            print(string+' for server use')
#        else:
#            print(string+' for local use (display=%s)' % os.environ['DISPLAY'])
#    else:
#        plt.switch_backend('agg')
#        print(string+'; display not recognized')
#except Exception as E:
#	print(string+'; could not set Matplotlib backend (%s), proceeding anyway...' % repr(E))