"""
model.py -- A tiny Python model. 

To run from the command line:
    python model.py
    
Last update: 5/11/18 (gchadder3)
"""

from pylab import figure, rand, show

# Define the graph
def makegraph(n=50):
	fig = figure()
	ax = fig.add_subplot(111)
	xdata = rand(n)
	ydata = rand(n)
	ax.scatter(xdata, ydata)
	return fig

# Run from command line
if __name__ == "__main__":
    makegraph()
    show()