'''
model.py -- a tiny Python model.

Usage: To run from the command line:
    python model.py 
    
Last update: 8/23/17 (gchadder3)
'''

#
# Load relevant packages and functions from packages.
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#
# Package globals
#

# Set datafilesPath and uploadsPath.
datafilesPath = '%s%sdatafiles' % (os.pardir, os.sep)
uploadsPath = '%s%sdatafiles' % (os.pardir, os.sep)
# uploadsPath = '%s%suploads' % (datafilesPath, os.sep)

#
# Classes
#

class Gaussian2D(object):
    """
    A 2D Gaussian model (currently only with non-diagonal covariance matrix).
    
    Methods:
        __init__(x_mean: float, y_mean: float, x_std: float, y_std: float, 
            r_xy: float): void -- constructor 
        generateData(nPoints: int): DataFrame -- given a number of points, 
            create x and y data according to the Gaussian distribution, and 
            put this into a pandas DataFrame to be returned
                    
    Attributes:
        x_mean -- the x mean of the distribution
        y_mean -- the y mean of the distribution
        x_std -- the x standard deviation of the distribution
        y_std -- the y standard deviation of the distribution
        r_xy -- the correlation value between the x and y values in the data
        
    Usage:
        >>> g1 = Gaussian2D()  # a normal 2D Gaussian (zero mean, unit std.)                      
    """
    
    def __init__(self, x_mean=0.0, y_mean=0.0, x_std=1.0, y_std=1.0, r_xy=0.0):
        # Right now, disallow non-diagonal covariance matrices.
        if r_xy != 0.0:
            print "Sorry, the code doesn't handle diagonal covariances yet, setting r_xy to 0.0."
            r_xy = 0.0
            
        # Set all of the parameters as passed in.
        self.x_mean = x_mean
        self.y_mean = y_mean 
        self.x_std = x_std 
        self.y_std = y_std 
        self.r_xy = r_xy 
        
    def generateData(self, nPoints=50):
        # Draw Gaussian x and y values.
        xs = np.random.normal(self.x_mean, self.x_std, nPoints)
        ys = np.random.normal(self.y_mean, self.y_std, nPoints)
        
        # Create and return a DataFrame holding the data.
        df = pd.DataFrame({'x': xs, 'y': ys})
        return df

#
# Functions
#

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
        df = makeUniformRandomData(50)
        df.to_csv('%s/graph1.csv' % datafilesPath)
        df = makeUniformRandomData(50)
        df.to_csv('%s/graph2.csv' % datafilesPath)
        df = makeUniformRandomData(50)
        df.to_csv('%s/graph3.csv' % datafilesPath)
        
# Make a (uniform distribution) random scatterplot.
def makeUniformRandomData(nPoints=50):
    # Create (uniform between 0 and 1) random data for all x/y points.
    xs = np.random.uniform(0.0, 1.0, nPoints)
    ys = np.random.uniform(0.0, 1.0, nPoints)
    
    # Create and return a DataFrame holding the data.
    df = pd.DataFrame({'x': xs, 'y': ys})
    return df

# Plot DataFrame data.
def plotData(df):
    # Create a figure and an axes object for it.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Generate the scatterplot of the data.
    ax.scatter(df.x, df.y)
    
    # Add title and axis labels.
    plt.title('Scatterplot y vs. x')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Return the figure.
    return fig

# Make a scatterplot for a chosen csv file.
def plotDataFromFile(fileName):
    # Create a full file name for the file.
    fullFileName = '%s/%s.csv' % (datafilesPath, fileName)
    
    # If the file is missing, return None.
    if not os.path.exists(fullFileName):
        return None
    
    # Load the csv file into a pandas DataFrame.
    df = pd.read_csv(fullFileName)
    
    # Create a scatterplot figure for the data and return it.
    return plotData(df)
    
    
# Run from command line
if __name__ == "__main__":
    init()
    plotDataFromFile('graph1')
    plt.show()