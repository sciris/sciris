"""
rpcs.py -- code for RPC interfaces between client and server
    
Last update: 8/30/17 (gchadder3)
"""

#
# Imports
#

import scirismodel.model as model
import mpld3
import os

#
# Package globals
#

# NOTE: These should probably end up in another file since they are session-
# related.
datafilesPath = '%s%sdatafiles' % (os.pardir, os.sep)
uploadsPath = '%s%sdatafiles' % (os.pardir, os.sep)
# uploadsPath = '%s%suploads' % (datafilesPath, os.sep)

#
# Session functions 
# NOTE: These should probably end up in another file.
#

# Put code here that initializes a user session.
# NOTE: This functionality should probably not go here.
def init_session():
    # If the datafiles path doesn't exist yet...
    if not os.path.exists(datafilesPath):
        # Create datafiles directory.
        os.mkdir(datafilesPath)
        
        # Create an uploads subdirectory of this.
        # os.mkdir(uploadsPath)
        
        # Create the fake data for scatterplots.
        sd = model.ScatterplotData(model.makeUniformRandomData(50))
        fullFileName = '%s%sgraph1.csv' % (datafilesPath, os.sep)
        sd.saveAsCsv(fullFileName)
        
        sd = model.ScatterplotData(model.makeUniformRandomData(50))
        fullFileName = '%s%sgraph2.csv' % (datafilesPath, os.sep)
        sd.saveAsCsv(fullFileName)
        
        sd = model.ScatterplotData(model.makeUniformRandomData(50))
        fullFileName = '%s%sgraph3.csv' % (datafilesPath, os.sep)
        sd.saveAsCsv(fullFileName)
        
#
# RPC functions
#

def get_resource_file_path(resourceName):
    # Create a full file name for the file.
    fullFileName = '%s%s%s.csv' % (datafilesPath, os.sep, resourceName)
    
    # If the file is there, return the path name; otherwise, return None.
    if os.path.exists(fullFileName):
        return fullFileName
    else:
        return None

def get_resource_graph(resourceName):
    # Look for a match of the resource, and if we don't find it, return
    # an error.
    fullFileName = get_resource_file_path(resourceName)
    if fullFileName is None:
        return {'error': 'Cannot find resource \'%s\'' % resourceName}
    
    # Create a ScatterplotData object.
    spd = model.ScatterplotData()

    # Load the data for this from the csv file.
    spd.loadFromCsv(fullFileName)
    
    # Generate a matplotib graph for display.
    graphData = spd.plot()
    
    # Return the dictionary representation of the matplotlib figure.
    return mpld3.fig_to_dict(graphData) 