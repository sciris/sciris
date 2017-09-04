"""
rpcs.py -- code for RPC interfaces between client and server
    
Last update: 9/4/17 (gchadder3)
"""

#
# Imports
#

import scirismodel.model as model
import mpld3
import os
import re

#
# Package globals
#

# NOTE: These should probably end up in another file since they are session-
# related.
datafilesPath = '%s%sdatafiles' % (os.pardir, os.sep)
uploadsPath = '%s%suploads' % (datafilesPath, os.sep)

#
# Classes
# NOTE: These probably should end up in other files (if I keep them).
#

# Wraps a Matplotlib figure displayable in the GUI.
class GraphFigure(object):
    def __init__(self, theFigure):
        self.theFigure = theFigure

# Wraps a collection of data.
class DataCollection(object):
    def __init__(self, dataObj):
        self.dataObj = dataObj

# Wraps a directory storage place for the session manager.
class DirectoryStore(object):
    def __init__(self, dirPath):
        self.dirPath = dirPath

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
        os.mkdir(uploadsPath)
        
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
     
def get_saved_scatterplotdata_file_path(spdName):
    # Create a full file name for the file.
    fullFileName = '%s%s%s.csv' % (datafilesPath, os.sep, spdName)
    
    # If the file is there, return the path name; otherwise, return None.
    if os.path.exists(fullFileName):
        return fullFileName
    else:
        return None
    
#
# RPC functions
#

def list_saved_scatterplotdata_resources():
    # Get the total list of entries in the datafiles directory.
    allEntries = os.listdir(datafilesPath)
    
    # Extract just the entries that are .csv files.
    fTypeEntries = [entry for entry in allEntries if re.match('.+\.csv$', entry)]
    
    # Truncate the .csv suffix from each entry.
    truncEntries = [re.sub('\.csv$', '', entry) for entry in fTypeEntries]
    
    # Return the truncated entries.
    return truncEntries

def get_saved_scatterplotdata_graph(spdName):
    # Look for a match of the resource, and if we don't find it, return
    # an error.
    fullFileName = get_saved_scatterplotdata_file_path(spdName)
    if fullFileName is None:
        return {'error': 'Cannot find resource \'%s\'' % spdName}
    
    # Create a ScatterplotData object.
    spd = model.ScatterplotData()

    # Load the data for this from the csv file.
    spd.loadFromCsv(fullFileName)
    
    # Generate a matplotib graph for display.
    graphData = spd.plot()
    
    # Return the dictionary representation of the matplotlib figure.
    return mpld3.fig_to_dict(graphData) 

def download_saved_scatterplotdata(spdName):
    return get_saved_scatterplotdata_file_path(spdName)

def upload_scatterplotdata_from_csv(fullFileName, spdName):
    # Pull out the directory and file names from the full file name.
    dirName, fileName = os.path.split(fullFileName)
    
    # Create a new destination for the file in the datafiles directory.
    newFullFileName = '%s%s%s' % (datafilesPath, os.sep, fileName)
    
    # Move the file into the datafiles directory.
    os.rename(fullFileName, newFullFileName)
    
    # Return the new file name.
    return newFullFileName
