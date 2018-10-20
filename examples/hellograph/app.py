# Imports
import pylab as pl
import sciris as sc
import scirisweb as sw

runserver = True # Choose to run in the frontend or backend

# Create the app
app = sw.ScirisApp(__name__, name="HelloGraph")

# Define the API
@app.route('/showgraph')
def showgraph(n=1000):
    
    # Make graph
    fig = pl.figure()
    ax = fig.add_subplot(111)
    xdata = pl.randn(n)
    ydata = pl.randn(n)
    colors = sc.vectocolor(pl.sqrt(xdata**2+ydata**2))
    ax.scatter(xdata, ydata, c=colors)
    
    # Convert to FE
    graphjson = sw.mpld3ify(fig)  # Convert to dict
    return graphjson  # Return the JSON representation of the Matplotlib figure

# Run the server
if __name__ == "__main__" and runserver:
    app.run()
else:
    showgraph()