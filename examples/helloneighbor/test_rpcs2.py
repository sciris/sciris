"""
test_rpcs2.py -- A testing file for installing RPCs outside of the caller of
    scirisapp.py
    
Last update: 5/17/18 (gchadder3)
"""

# Imports
#from sciris2gc.scirisapp import ScirisRPC
#from functools import wraps
#import pandas as pd
#from pylab import figure
#import mpld3
#import model
#import os
#from sciris2gc import fileio


# Attempts to remove boilerplate...

# This appeared to work at first, but test_rpcs and test_rpcs2 then share the 
# same RPC_dict, which is bad.
#from sciris2gc.test_rpcs_aux import RPC_dict, register_RPC

# This works only on my machine and uses execfile() which is probably not great.
execfile('C:\\GitRepos\\OptimaRepos\\sciris\\sciris2gc\\test_rpcs_aux.py')


##
## Globals
##
#
## Dictionary to hold all of the registered RPCs in this module.
#RPC_dict = {}
#
##
## Miscellaneous functions
##
#
## Decorator factory for registering RPCs in this module.
## TODO: If we can, it would be nice to remove this boilerplate by defining 
## this decorator factory somehow in scirisapp.py.
#def register_RPC(**callerkwargs):
#    def RPC_decorator(RPC_func):
#        @wraps(RPC_func)
#        def wrapper(*args, **kwargs):        
#            RPC_func(*args, **kwargs)
#
#        # Create the RPC and add it to the dictionary.
#        RPC_dict[RPC_func.__name__] = ScirisRPC(RPC_func, **callerkwargs)
#        
#        return wrapper
#
#    return RPC_decorator

#
# RPC functions
#
    
@register_RPC()
def test_func4():
    return '<h1>Test Me Always!</h1>'

@register_RPC()
def test_func5():
    return '<h1>Test Me Never!</h1>'

@register_RPC()
def test_func6():
    return '<h1>Test Me Warily!</h1>'

#@register_RPC(call_type='upload')
#def show_csv_file(full_file_name):
##    x = 1 / 0  # uncomment to test exceptions with ZeroDivisionError
##    return {'error': 'show_csv_file() just does not feel like working.'}  # uncomment to test custom error
#
#    # Extract the data from the .csv file.
#    df = pd.read_csv(full_file_name)
#    
#    # Create the figure from the data.
#    new_graph = figure()
#    ax = new_graph.add_subplot(111)
#    ax.scatter(df.x, df.y)
#    
#    # Convert the figure to a JSON-able dict.
#    graph_dict = mpld3.fig_to_dict(new_graph)
#    
#    # Return the dict.
#    return graph_dict
#
#@register_RPC(call_type='download')
#def download_graph_png():
##    x = 1 / 0  # uncomment to test exceptions with ZeroDivisionError
##    return {'error': 'download_graph_png() just does not feel like working.'}  # uncomment to test custom error
#    
#    # Make a new graph with random data.
#    new_graph = model.makegraph()
#    
#    # Save a .png file of this graph.
#    full_file_name = '%s%sgraph.png' % (fileio.downloads_dir.dir_path, os.sep)
#    new_graph.savefig(full_file_name)
#    
#    # Return the full filename.
#    return full_file_name