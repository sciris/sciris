# Imports
import pylab as pl
import sciris as sc
import scirisweb as sw
from datetime import datetime as dt

runserver = True # Choose to run in the frontend or backend

# Convert dates to years
def convertdate(datestr, fmt):
    date = dt.strptime(datestr, fmt)
    yearnum = date.year + (date.month-1)/12. + (date.day-1)/365.
    return yearnum

# Get the data
def loaddata():
    print('Loading data...')
    dataurl = 'https://raw.githubusercontent.com/rstudio/shiny-examples/master/120-goog-index/data/trend_data.csv'
    rawdata = sc.wget(dataurl).splitlines()
    data = []
    for r,rawline in enumerate(rawdata):
        line = rawline.split(',')
        if r==0: # Read header
            cols = line
        else: # Read data
            tag = line[0]
            yearnum = convertdate(line[1], '%Y-%m-%dT%I:%M:%fZ')
            value = float(line[2]) if r>0 else line[2]
            data.append([tag, yearnum, value])
    df = sc.dataframe(cols=cols, data=data)
    return df

# Create the app
app = sw.ScirisApp(__name__, name="PyShiny")
df = loaddata()

# Define the API
@app.route('/getoptions')
def getoptions(tojson=True):
    options = sc.odict([
        ('Advertising',    'advert'),
        ('Education',      'educat'),
        ('Small business', 'smallbiz'),
        ('Travel',         'travel'),
        ('Unemployment',   'unempl'),
        ])
    if tojson:
        output = sc.sanitizejson(options.keys(), tostring=True)
    else:
        output = options
    return output

@app.route('/plotdata/<trendselection>/<startdate>/<enddate>/<trendline>')
def plotdata(trendselection=None, startdate='2000-01-01', enddate='2018-01-01', trendline='false'):
    
    # Handle inputs
    startyear = convertdate(startdate, '%Y-%m-%d')
    endyear   = convertdate(enddate,   '%Y-%m-%d')
    trendoptions = getoptions(tojson=False)
    if trendselection is None: trendselection  = trendoptions.keys()[0]
    datatype = trendoptions[trendselection]

    # Make graph
    fig = pl.figure()
    fig.add_subplot(111)
    thesedata = df.findrows(key=datatype, col='type')
    years = thesedata['date']
    vals = thesedata['close']
    validinds = sc.findinds(pl.logical_and(years>=startyear, years<=endyear))
    x = years[validinds]
    y = vals[validinds]
    pl.plot(x, y)
    pl.xlabel('Date')
    pl.ylabel('Trend index')
    
    # Add optional trendline
    if trendline == 'true':
        newy = sc.smoothinterp(x, x, y, smoothness=200)
        pl.plot(x, newy, lw=3)
    
    # Convert to FE
    graphjson = sw.mpld3ify(fig)  # Convert to dict
    return graphjson  # Return the JSON representation of the Matplotlib figure

# Run the server
if __name__ == "__main__" and runserver:
    app.run()
else:
    plotdata()