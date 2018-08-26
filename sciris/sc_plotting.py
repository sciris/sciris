##############################################################################
### IMPORTS
##############################################################################

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib import ticker
import numpy as np
from numpy.linalg import norm
import pylab as pl
from . import sc_utils as ut
from . import sc_fileio as fio
from .sc_odict import odict
from numpy import linspace, meshgrid, array, transpose, inf, zeros, argmax, minimum
from pylab import randn, show, convolve, array, seed, linspace, meshgrid, xlabel, ylabel, figure, pcolor
from matplotlib.colors import LinearSegmentedColormap as makecolormap



##############################################################################
### COLOR FUNCTIONS
##############################################################################

__all__ = ['processcolors', 'shifthue', 'gridcolors', 'alpinecolormap', 'vectocolor', 'bicolormap', 'hex2rgb'] # apinecolortest and bicolormaptest not included


def processcolors(colors=None, asarray=False, reverse=False):
    ''' 
    Small helper function to do common transformations on the colors, once generated.
    Expects colors to be an array. If asarray is True and reverse are False, returns 
    that array. Otherwise, does the required permutations.    
    '''
    if asarray:
        output = colors
        if reverse: output = output[::-1] # Reverse the array
    else:
        output = []
        for c in colors: # Gather output
            output.append(tuple(c))
        if reverse: # Reverse the list
            output.reverse()
    return output


def shifthue(colors=None, hueshift=0.0):
    '''
    Shift the hue of the colors being fed in.
    
    Example:
        colors = shifthue(colors=[(1,0,0),(0,1,0)], hueshift=0.5)
    '''
    colors = ut.dcp(colors) # So we don't overwrite the original
    origndim = np.ndim(colors) # Original dimensionality
    if origndim==1: colors = [colors] # Wrap it in another list
    colors = np.array(colors) # Just convert it to an array
    for c,color in enumerate(colors):
        hsvcolor = rgb_to_hsv(color)
        hsvcolor[0] = (hsvcolor[0]+hueshift) % 1.0 # Calculate new hue and return the modulus
        rgbcolor = hsv_to_rgb(hsvcolor)
        colors[c] = rgbcolor
    if origndim==1: colors = colors[0] # Pull it out again
    return colors


def gridcolors(ncolors=10, limits=None, nsteps=10, asarray=False, reverse=False, doplot=False, hueshift=0):
    """
    GRIDCOLORS

    Create a qualitative "color map" by assigning points according to the maximum pairwise distance in the
    color cube. Basically, the algorithm generates n points that are maximally uniformly spaced in the
    [R, G, B] color cube.
    
    Arguments:
        ncolors: the number of colors to create
        limits: how close to the edges of the cube to make colors (to avoid white and black)
        nsteps: the discretization of the color cube (e.g. 10 = 10 units per side = 1000 points total)
        asarray: whether to return the colors as an array instead of as a list of tuples
        doplot: whether or not to plot the color cube itself

    Usage example:
        from pylab import *
        from colortools import gridcolors
        ncolors = 10
        piedata = rand(ncolors)
        colors = gridcolors(ncolors)
        figure()
        pie(piedata, colors=colors)
        gridcolors(ncolors, doplot=True)
        show()

    Version: 1.2 (2015dec29) 
    """

    # Steal colorbrewer colors for small numbers of colors
    colorbrewercolors = array([
    [ 55, 126, 184], # [27,  158, 119], # Old color
    [228,  26,  28], # [217, 95,  2],
    [ 77, 175,  74], # [117, 112, 179],
    [162,  78, 153], # [231, 41,  138],
    [255, 127,   0],
    [200, 200,  51], # Was too bright yellow
    [166,  86,  40],
    [247, 129, 191],
    [153, 153, 153],
    ])/255.
    
    if ncolors<=len(colorbrewercolors):
        colors = colorbrewercolors[:ncolors]
        
    else: # Too many colors, calculate instead
        ## Calculate sliding limits if none provided
        if limits is None:
            colorrange = 1-1/float(ncolors**0.5)
            limits = [0.5-colorrange/2, 0.5+colorrange/2]
        
        ## Calculate primitives and dot locations
        primitive = np.linspace(limits[0], limits[1], nsteps) # Define primitive color vector
        x, y, z = meshgrid(primitive, primitive, primitive) # Create grid of all possible points
        dots = transpose(array([x.flatten(), y.flatten(), z.flatten()])) # Flatten into an array of dots
        ndots = nsteps**3 # Calculate the number of dots
        indices = [0] # Initialize the array
        
        ## Calculate the distances
        for pt in range(ncolors-1): # Loop over each point
            totaldistances = inf+zeros(ndots) # Initialize distances
            for ind in indices: # Loop over each existing point
                rgbdistances = dots - dots[ind] # Calculate the distance in RGB space
                totaldistances = minimum(totaldistances, norm(rgbdistances,axis=1)) # Calculate the minimum Euclidean distance
            maxindex = argmax(totaldistances) # Find the point that maximizes the minimum distance
            indices.append(maxindex) # Append this index
        
        colors = dots[indices,:]
    
    ## Wrap up -- turn color array into a list, or reverse
    if hueshift: colors = shifthue(colors, hueshift=hueshift) # Shift hue if requested
    output = processcolors(colors=colors, asarray=asarray, reverse=reverse)
    
    ## For plotting -- optional
    if doplot:
        from mpl_toolkits.mplot3d import Axes3D # analysis:ignore
        if doplot=='new':
            fig = pl.figure(facecolor='w')
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = pl.gca()
        ax.scatter(colors[:,0], colors[:,1], colors[:,2], c=output, s=200, depthshade=False, lw=0)
        ax.set_xlabel('Red', fontweight='bold')
        ax.set_ylabel('Green', fontweight='bold')
        ax.set_zlabel('Blue', fontweight='bold')
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_zlim((0,1))
    
    return output
    
    


## Create colormap
def alpinecolormap(gap=0.1, mingreen=0.2, redbluemix=0.5, epsilon=0.01, test=False):
   """
   ALPINECOLORMAP

   This program generates a map based on ascending height. Based on data from
   Kazakhstan.

   Test case:
   import alpinecolormap
   alpinecolormap.testcolormap()

   Usage example:
   from alpinecolormap import alpinecolormap
   from pylab import randn, imshow, show
   imshow(randn(20,20),interpolation='none',cmap=alpinecolormap())
   show()

   Version: 2014aug06 
   """
   water = array([3,18,59])/256.
   desert = array([194,175,160*0.6])/256.
   forest1 = array([61,86,46])/256.
   forest2 = array([61,86,46])/256.*1.2
   rock = array([119,111,109])/256.*1.3
   snow = array([243,239,238])/256.
   breaks = [0.0,0.5,0.7,0.8,0.9,1.0]
   
   cdict = {'red': ((breaks[0], water[0], water[0]),
                    (breaks[1], desert[0], desert[0]),
                    (breaks[2], forest1[0], forest1[0]),
                    (breaks[3], forest2[0], forest2[0]),
                    (breaks[4], rock[0], rock[0]),
                    (breaks[5], snow[0], snow[0])),

         'green':  ((breaks[0], water[1], water[1]),
                    (breaks[1], desert[1], desert[1]),
                    (breaks[2], forest1[1], forest1[1]),
                    (breaks[3], forest2[1], forest2[1]),
                    (breaks[4], rock[1], rock[1]),
                    (breaks[5], snow[1], snow[1])),

         'blue':   ((breaks[0], water[2], water[2]),
                    (breaks[1], desert[2], desert[2]),
                    (breaks[2], forest1[2], forest1[2]),
                    (breaks[3], forest2[2], forest2[2]),
                    (breaks[4], rock[2], rock[2]),
                    (breaks[5], snow[2], snow[2]))}
   
   cmap = makecolormap('alpinecolormap',cdict,256)
   
   
   if test:
       from mpl_toolkits.mplot3d import Axes3D # analysis:ignore
    
        maxheight = 3
        horizontalsize = 4;
        seed(8)
        n = 100
        smoothing = 40;
        kernel = array([0.25,0.5,0.25])
        data = randn(n,n)
        for s in range(smoothing): # Quick-and-dirty-and-slow smoothing
            for i in range(n): data[:,i] = convolve(data[:,i],kernel,mode='same')
            for i in range(n): data[i,:] = convolve(data[i,:],kernel,mode='same')
        data -= data.min()
        data /= data.max()
        data *= maxheight
        
        fig = figure(figsize=(18,8))
        ax = fig.gca(projection='3d')
        ax.view_init(elev=45, azim=30)
        X = np.linspace(0,horizontalsize,n)
        X, Y = meshgrid(X, X)
        surf = ax.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=alpinecolormap(), linewidth=0, antialiased=False)
        cb = fig.colorbar(surf)
        cb.set_label('Height (km)',horizontalalignment='right', labelpad=50)
        xlabel('Position (km)')
        ylabel('Position (km)')
        show()
    
        fig = figure(figsize=(8,6))
        ax = fig.gca()
        X = np.linspace(0,horizontalsize,n)
        pcl = pcolor(X, X, data, cmap=alpinecolormap(), linewidth=0, antialiased=False)
        cb2 = fig.colorbar(pcl)
        cb2.set_label('Height (km)',horizontalalignment='right', labelpad=50)
        xlabel('Position (km)')
        ylabel('Position (km)')
        show()
   
   
   
   return cmap






def vectocolor(vector, cmap=None, asarray=True, reverse=False):
   """
   VECTOCOLOR
   This function converts a vector of N values into an Nx3 matrix of color
   values according to the current colormap. It automatically scales the 
   vector to provide maximum dynamic range for the color map.

   Usage:
   colors = vectocolor(vector,cmap=None)

   where:
   colors is an Nx4 list of RGB-alpha color values
   vector is the input vector (or list, it's converted to an array)
   cmap is the colormap (default: jet)

   Example:
   n = 1000
   x = randn(n,1);
   y = randn(n,1);
   c = vectocolor(y);
   scatter(x,y,20,c)

   Version: 2016sep28 
   """
   from numpy import array, zeros
   from pylab import cm

   if cmap==None:
      cmap = cm.jet
   elif type(cmap)==str:
      try: cmap = getattr(cm,cmap)
      except: raise Exception('%s is not a valid color map; choices are:\n%s' % (cmap, '\n'.join(sorted(cm.datad.keys()))))

   # The vector has elements
   if len(vector):
      vector = array(vector) # Just to be sure
      vector = vector-vector.min() # Subtract minimum
      vector = vector/float(vector.max()) # Divide by maximum
      nelements = len(vector) # Count number of elements
      colors=zeros((nelements,4))
      for i in range(nelements):
         colors[i,:]=array(cmap(vector[i]))

   # It doesn't; just return black
   else: colors=(0,0,0,1)
   
   # Process output
   output = processcolors(colors=colors, asarray=asarray, reverse=reverse)

   return output






## Create colormap
def bicolormap(gap=0.1,mingreen=0.2,redbluemix=0.5,epsilon=0.01):
   """
   BICOLORMAP

   This program generators a two-color map, blue for negative, red for
   positive changes, with grey in the middle. The input argument is how much
   of a color gap there is between the red scale and the blue one.

   The function has four parameters:
     gap: sets how big of a gap between red and blue color scales there is (0=no gap; 1=pure red and pure blue)
     mingreen: how much green to include at the extremes of the red-blue color scale
     redbluemix: how much red to mix with the blue and vice versa at the extremes of the scale
     epsilon: what fraction of the colormap to make gray in the middle

   Examples:
     bicolormap(gap=0,mingreen=0,redbluemix=1,epsilon=0) # From pure red to pure blue with white in the middle
     bicolormap(gap=0,mingreen=0,redbluemix=0,epsilon=0.1) # Red -> yellow -> gray -> turquoise -> blue
     bicolormap(gap=0.3,mingreen=0.2,redbluemix=0,epsilon=0.01) # Red and blue with a sharp distinction between

   Version: 2013sep13 
   """
   from matplotlib.colors import LinearSegmentedColormap as makecolormap
   
   mng=mingreen; # Minimum amount of green to add into the colors
   mix=redbluemix; # How much red to mix with the blue an vice versa
   eps=epsilon; # How much of the center of the colormap to make gray
   omg=1-gap # omg = one minus gap
   
   cdict = {'red': ((0.00000, 0.0, 0.0),
                    (0.5-eps, mix, omg),
                    (0.50000, omg, omg),
                    (0.5+eps, omg, 1.0),
                    (1.00000, 1.0, 1.0)),

         'green':  ((0.00000, mng, mng),
                    (0.5-eps, omg, omg),
                    (0.50000, omg, omg),
                    (0.5+eps, omg, omg),
                    (1.00000, mng, mng)),

         'blue':   ((0.00000, 1.0, 1.0),
                    (0.5-eps, 1.0, omg),
                    (0.50000, omg, omg),
                    (0.5+eps, omg, mix),
                    (1.00000, 0.0, 0.0))}
   cmap = makecolormap('bicolormap',cdict,256)

   return cmap

## Test
def bicolormaptest():
    from pylab import figure, subplot, imshow, colorbar, rand, show
    
    maps=[]
    maps.append(bicolormap()) # Default ,should work for most things
    maps.append(bicolormap(gap=0,mingreen=0,redbluemix=1,epsilon=0)) # From pure red to pure blue with white in the middle
    maps.append(bicolormap(gap=0,mingreen=0,redbluemix=0,epsilon=0.1)) # Red -> yellow -> gray -> turquoise -> blue
    maps.append(bicolormap(gap=0.3,mingreen=0.2,redbluemix=0,epsilon=0.01)) # Red and blue with a sharp distinction between
    nexamples=len(maps)
    
    figure(figsize=(5*nexamples,4))    
    for m in range(nexamples):
        subplot(1,nexamples,m+1)
        imshow(rand(20,20),cmap=maps[m],interpolation='none');
        colorbar()
    show()



def hex2rgb(string):
    ''' A little helper function to convert e.g. '86bc25' to a pleasing shade of green. '''
    from numpy import array
    from struct import unpack
    rgb = array(unpack('BBB',string.decode('hex')),dtype=float)/255.
    return rgb




##############################################################################
### PLOTTING FUNCTIONS
##############################################################################

__all__ += ['boxoff', 'setylim', 'commaticks', 'SItickformatter', 'SIticks']


def boxoff(ax=None, removeticks=True, flipticks=True):
    '''
    I don't know why there isn't already a Matplotlib command for this.
    
    Removes the top and right borders of a plot. Also optionally removes
    the tick marks, and flips the remaining ones outside.

    Version: 2017may22    
    '''
    from pylab import gca
    if ax is None: ax = gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if removeticks:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    if flipticks:
        ax.tick_params(direction='out', pad=5)
    return ax
    
    

def setylim(data=None, ax=None):
    '''
    A small script to determine how the y limits should be set. Looks
    at all data (a list of arrays) and computes the lower limit to
    use, e.g.
    
        setylim([array([-3,4]), array([6,4,6])], ax)
    
    will keep Matplotlib's lower limit, since at least one data value
    is below 0.
    
    Note, if you just want to set the lower limit, you can do that 
    with this function via:
        setylim(0, ax)
    '''
    # Get current limits
    currlower, currupper = ax.get_ylim()
    
    # Calculate the lower limit based on all the data
    lowerlim = 0
    upperlim = 0
    data = ut.promotetolist(data) # Make sure it'siterable
    for ydata in data:
        lowerlim = min(lowerlim, ut.promotetoarray(ydata).min())
        upperlim = max(upperlim, ut.promotetoarray(ydata).max())
    
    # Set the new y limits
    if lowerlim<0: lowerlim = currlower # If and only if the data lower limit is negative, use the plotting lower limit
    upperlim = max(upperlim, currupper) # Shouldn't be an issue, but just in case...
    
    # Specify the new limits and return
    ax.set_ylim((lowerlim, upperlim))
    return lowerlim,upperlim



def commaticks(fig=None, ax=None, axis='y'):
    ''' Use commas in formatting the y axis of a figure -- see http://stackoverflow.com/questions/25973581/how-to-format-axis-number-format-to-thousands-with-a-comma-in-matplotlib '''
    from matplotlib import ticker
    if   ax  is not None: axlist = ut.promotetolist(ax)
    elif fig is not None: axlist = fig.axes
    else: raise Exception('Must supply either figure or axes')
    for ax in axlist:
        if   axis=='x': thisaxis = ax.xaxis
        elif axis=='y': thisaxis = ax.yaxis
        elif axis=='z': thisaxis = ax.zaxis
        else: raise Exception('Axis must be x, y, or z')
        thisaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    return None
    
    

def SItickformatter(x, pos=None, sigfigs=2, SI=True, *args, **kwargs):  # formatter function takes tick label and tick position
    ''' Formats axis ticks so that e.g. 34,243 becomes 34K '''
    output = ut.sigfig(x, sigfigs=sigfigs, SI=SI) # Pretty simple since ut.sigfig() does all the work
    return output



def SIticks(fig=None, ax=None, axis='y', fixed=False):
    ''' Apply SI tick formatting to one axis of a figure '''
    if  fig is not None: axlist = fig.axes
    elif ax is not None: axlist = ut.promotetolist(ax)
    else: raise Exception('Must supply either figure or axes')
    for ax in axlist:
        if   axis=='x': thisaxis = ax.xaxis
        elif axis=='y': thisaxis = ax.yaxis
        elif axis=='z': thisaxis = ax.zaxis
        else: raise Exception('Axis must be x, y, or z')
        if fixed:
            ticklocs = thisaxis.get_ticklocs()
            ticklabels = []
            for tickloc in ticklocs:
                ticklabels.append(SItickformatter(tickloc))
            thisaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
        else:
            thisaxis.set_major_formatter(ticker.FuncFormatter(SItickformatter))
    return None



##############################################################################
### FIGURE SAVING
##############################################################################

__all__ += ['savefigs', 'loadfig']


def savefigs(plots=None, filetype=None, filename=None, folder=None, savefigargs=None, index=None, verbose=2, **kwargs):
    '''
    Save the requested plots to disk.
    
    Arguments:
        plots -- the figure objects to save
        filetype -- the file type; can be 'fig', 'singlepdf' (default), or anything supported by savefig()
        folder -- the folder to save the file(s) in
        filename -- the file to save to (only uses path if multiple files)
        savefigargs -- dictionary of arguments passed to savefig()
        index -- optional argument to only save the specified plot index
        kwargs -- passed to makeplots()
    
    Example usages:
        import pylab as pl
        import sciris as sc
        fig1 = pl.figure(); pl.plot(pl.rand(10))
        fig2 = pl.figure(); pl.plot(pl.rand(10))
        sc.savefigs([fig1, fig2]) # Save everything to one PDF file
        sc.savefigs(fig2, 'png', filename='myfig.png', savefigargs={'dpi':200})
        sc.savefigs([fig1, fig2], filepath='/home/me', filetype='svg')
        sc.savefigs(fig1, position=[0.3,0.3,0.5,0.5])
    
    If saved as 'fig', then can load and display the plot using sc.loadfig().
    
    Version: 2018aug26 
    '''
    
    # Preliminaries
    wasinteractive = pl.isinteractive() # You might think you can get rid of this...you can't!
    if wasinteractive: pl.ioff()
    if filetype is None: filetype = 'singlepdf' # This ensures that only one file is created
    
    # Either take supplied plots, or generate them
    plots = odict.promote(plots)
    nplots = len(plots)
    
    # Handle file types
    filenames = []
    if filetype=='singlepdf': # See http://matplotlib.org/examples/pylab_examples/multipage_pdf.html
        from matplotlib.backends.backend_pdf import PdfPages
        defaultname = 'figures.pdf'
        fullpath = fio.makefilepath(filename=filename, folder=folder, default=defaultname, ext='pdf')
        pdf = PdfPages(fullpath)
        filenames.append(fullpath)
        ut.printv('PDF saved to %s' % fullpath, 2, verbose)
    for p,item in enumerate(plots.items()):
        key,plt = item
        if index is None or index==p:
            # Handle filename
            if filename and nplots==1: # Single plot, filename supplied -- use it
                fullpath = fio.makefilepath(filename=filename, folder=folder, default='optima-figure', ext=filetype) # NB, this filename not used for singlepdf filetype, so it's OK
            else: # Any other case, generate a filename
                keyforfilename = filter(str.isalnum, str(key)) # Strip out non-alphanumeric stuff for key
                defaultname = keyforfilename
                fullpath = fio.makefilepath(filename=filename, folder=folder, default=defaultname, ext=filetype)
            
            # Do the saving
            if savefigargs is None: savefigargs = {}
            defaultsavefigargs = {'dpi':200, 'bbox_inches':'tight'} # Specify a higher default DPI and save the figure tightly
            defaultsavefigargs.update(savefigargs) # Update the default arguments with the user-supplied arguments
            if filetype == 'fig':
                fio.saveobj(fullpath, plt)
                filenames.append(fullpath)
                ut.printv('Figure object saved to %s' % fullpath, 2, verbose)
            else:
                reanimateplots(plt)
                if filetype=='singlepdf':
                    pdf.savefig(figure=plt, **defaultsavefigargs) # It's confusing, but defaultsavefigargs is correct, since we updated it from the user version
                else:
                    plt.savefig(fullpath, **defaultsavefigargs)
                    filenames.append(fullpath)
                    ut.printv('%s plot saved to %s' % (filetype.upper(),fullpath), 2, verbose)
                pl.close(plt)

    if filetype=='singlepdf': pdf.close()
    if wasinteractive: pl.ion()
    return filenames


def loadfig(filename=None):
    '''
    Load a plot from a file and reanimate it.
    
    Example usage:
        import pylab as pl
        import sciris as sc
        fig = pl.figure(); pl.plot(pl.rand(10))
        op.saveplots(fig, filetype='fig', filename='example.fig')
    Later:
        example = op.loadplot('example.fig')
    '''
    pl.ion() # Without this, it doesn't show up
    fig = fio.loadobj(filename)
    reanimateplots(fig)
    return fig


def reanimateplots(plots=None):
    ''' Reconnect plots (actually figures) to the Matplotlib backend. plots must be an odict of figure objects. '''
    try:
        from matplotlib.backends.backend_agg import new_figure_manager_given_figure as nfmgf # Warning -- assumes user has agg on their system, but should be ok. Use agg since doesn't require an X server
    except Exception as E:
        errormsg = 'To reanimate plots requires the "agg" backend, which could not be imported: %s' % repr(E)
        raise Exception(errormsg)
    if len(pl.get_fignums()): fignum = pl.gcf().number # This is the number of the current active figure, if it exists
    else: fignum = 1
    plots = odict.promote(plots) # Convert to an odict
    for plot in plots.values(): nfmgf(fignum, plot) # Make sure each figure object is associated with the figure manager -- WARNING, is it correct to associate the plot with an existing figure?
    return None