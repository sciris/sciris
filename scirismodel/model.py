'''
model.py -- a tiny Python model.

Usage: To run from the command line:
    python model.py 
    
Last update: 8/22/17 (gchadder3)
'''

from pylab import figure, rand, show
import os
import pandas as pd

# Set datafilesPath and uploadsPath.
datafilesPath = '%s%sdatafiles' % (os.pardir, os.sep)
uploadsPath = '%s%sdatafiles' % (os.pardir, os.sep)
# uploadsPath = '%s%suploads' % (datafilesPath, os.sep)

# Perform any setup that needs to happen to use the model code.
def init():
    # Detect whether we're calling from the main (a standalone run).
    calledFromMain = (__name__ == '__main__')
      
    # If the datafiles path doesn't exist yet...
    if not os.path.exists(datafilesPath):
        # Create datafiles directory.
        os.mkdir(datafilesPath)
        
        # Create an uploads subdirectory of this.
        # os.mkdir(uploadsPath)
        
        # Create the fake data for scatterplots.
        df = pd.DataFrame({'x': rand(50), 'y': rand(50)})
        df.to_csv('%s/graph1.csv' % datafilesPath)
        df = pd.DataFrame({'x': rand(50), 'y': rand(50)})
        df.to_csv('%s/graph2.csv' % datafilesPath)    
        df = pd.DataFrame({'x': rand(50), 'y': rand(50)})
        df.to_csv('%s/graph3.csv' % datafilesPath)

# Make a random scatterplot.
def makeRandomGraph(n=50):
    # Create a figure and an axes object for it.
    fig = figure()
    ax = fig.add_subplot(111)
    
    # Create random data for all x/y points.
    xdata = rand(n)
    ydata = rand(n)
    
    # Generate the scatterplot.
    ax.scatter(xdata, ydata)
    
    # Return the figure.
    return fig

# Make a scatterplot for a chosen csv file.
def makeGraph(fileName):
    # Create a full file name for the file.
    fullFileName = '%s/%s.csv' % (datafilesPath, fileName)
    
    # If the file is missing, return None.
    if not os.path.exists(fullFileName):
        return None
    
    # Create a figure and an axes object for it.
    fig = figure()
    ax = fig.add_subplot(111)
    
    # Load the csv file into a pandas DataFrame.
    df = pd.read_csv(fullFileName)
    
    # Generate the scatterplot of the data.
    ax.scatter(df.x, df.y)
    
    # Return the figure.
    return fig
    
# Run from command line
if __name__ == "__main__":
    init()
    makeGraph('graph1')
    show()