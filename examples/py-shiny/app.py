# Imports
import pylab as pl
import sciris as sc
import scirisweb as sw
import datetime

runserver = False # Choose to run in the frontend or backend

# Create the app
app = sw.ScirisApp(__name__, name="PyShiny")

# Get the data
dataurl = 'https://raw.githubusercontent.com/rstudio/shiny-examples/master/120-goog-index/data/trend_data.csv'
rawdata = sc.wget(dataurl).splitlines()
data = []
for r,rawline in enumerate(rawdata):
    line = rawline.split(',')
    if r==0: # Read header
        cols = line
    else: # Read data
        tag = line[0]
        date = datetime.datetime.strptime(line[1], '%Y-%m-%dT%I:%M:%fZ')
        yearnum = date.year + (date.month-1)/12 + (date.day-1)/365
        value = float(line[2]) if r>0 else line[2]
        data.append([tag, yearnum, value])
df = sc.dataframe(cols=cols, data=data)


# Define the API
@app.route('/plotdata')
def plotdata(datatype=None, startdate=None, enddate=None):
    
    # Make graph
    fig = pl.figure()
    ax = fig.add_subplot(111)
    thesedata = df.findrows(key=datatype, col='type')
    years = thesedata['date']
    vals = thesedata['close']
    validinds = sc.findinds(years>=float(startdate) and years<=float(enddate))
    ax.plot(years[validinds], vals[validinds])
    
    # Convert to FE
    graphjson = sw.mpld3ify(fig)  # Convert to dict
    return graphjson  # Return the JSON representation of the Matplotlib figure

# Run the server
if __name__ == "__main__" and runserver:
    app.run()
else:
    showgraph()