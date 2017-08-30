'''
model.py -- a small Python model.

Usage: To run from the command line:
    python model.py 
    
Last update: 8/30/17 (gchadder3)
'''

#
# Imports
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        
    def generateData(self, nPoints=50, xLabel='x', yLabel='y'):
        # Draw Gaussian x and y values.
        xs = np.random.normal(self.x_mean, self.x_std, nPoints)
        ys = np.random.normal(self.y_mean, self.y_std, nPoints)
        
        # Create and return a DataFrame holding the data.
        df = pd.DataFrame({xLabel: xs, yLabel: ys})
        df = df.ix[:, [xLabel, yLabel]]  # make sure column order correct
        return df
    
class ScatterplotData(object):
    """
    A collection of 2D datapoints for making scatterplots.
    
    Methods:
        __init__(df: DataFrame): void -- constructor 
        plot(): matplotlib figure -- plots the DataFrame in this object and 
            returns the figure object
        loadFromCsv(fullFileName: str): void -- overwrite the DataFrame with 
            data from a .csv file
        saveAsCsv(fullFileName: str): void -- save the DataFrame content to a
            .csv file
                    
    Attributes:
        df -- the pandas DataFrame holding the data
        
    Usage:
        >>> spd = ScatterplotData()
        >>> spd.loadFromCsv('mydatafile.csv')                     
    """
    
    def __init__(self, df=None):
        # Set the DataFrame.
        self.df = df
        
    def plot(self):
        # Create a figure and an axes object for it.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # Grab the x and y labels, assuming column 0 is x, and 1 y.
        theXlabel = self.df.columns[0]
        theYlabel = self.df.columns[1]  
        
        # Generate the scatterplot of the data.
        ax.scatter(self.df[theXlabel], self.df[theYlabel])
        
        # Add title and axis labels.
        plt.title('Scatterplot %s vs. %s' % (theYlabel, theXlabel))
        plt.xlabel(theXlabel)
        plt.ylabel(theYlabel)
        
        # Return the figure.
        return fig
    
    def loadFromCsv(self, fullFileName):
        # Load the csv file into a pandas DataFrame.
        self.df = pd.read_csv(fullFileName, index_col=0)
    
    def saveAsCsv(self, fullFileName):
        # Save the pandas data to a csv file.
        self.df.to_csv(fullFileName)
        
#
# Functions
#
        
# Make a (uniform distribution) random scatterplot.
def makeUniformRandomData(nPoints=50, xLabel='x', yLabel='y'):
    # Create (uniform between 0 and 1) random data for all x/y points.
    xs = np.random.uniform(0.0, 1.0, nPoints)
    ys = np.random.uniform(0.0, 1.0, nPoints)
    
    # Create and return a DataFrame holding the data.
    df = pd.DataFrame({xLabel: xs, yLabel: ys})
    df = df.ix[:, [xLabel, yLabel]]  # make sure column order correct
    return df

#
# Script code
#

if __name__ == "__main__":
    # Create a Gaussian 2D model.
    g1 = Gaussian2D()
    
    # Create ScatterplotData for it.
    spd = ScatterplotData(g1.generateData())
    
    # Create a second ScatterplotData object for a uniform distribution.
    spd2 = ScatterplotData(makeUniformRandomData(50))
    
    # Plot the Gaussian scatterplot.
    spd.plot()
    
    # Plot the uniform scatterplot.
    spd2.plot()
    
    plt.show()