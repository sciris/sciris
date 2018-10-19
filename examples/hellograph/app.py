# Imports
import scirisweb as sw
from pylab import figure, rand

# Create the app
app = sw.ScirisApp(__name__, name="HelloGraph")

# Define the graph
def makegraph(n=50):
	fig = figure()
	ax = fig.add_subplot(111)
	xdata = rand(n)
	ydata = rand(n)
	ax.scatter(xdata, ydata)
	return fig

# Define the API
@app.route('/showgraph')
def showgraph():
    graphdata = makegraph() # Create the graph from Python
    graphjson = sw.mpld3ify(graphdata)  # Convert to dict
    return graphjson  # Return the JSON representation of the Matplotlib figure

# Run the server
if __name__ == "__main__":
    app.run()