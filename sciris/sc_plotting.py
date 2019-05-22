##############################################################################
### IMPORTS
##############################################################################

import six
from struct import unpack
import pylab as pl
import numpy as np
from numpy.linalg import norm
from matplotlib import ticker
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib.colors import LinearSegmentedColormap as makecolormap
from .sc_odict import odict
from . import sc_utils as ut
from . import sc_fileio as fio


##############################################################################
### COLOR FUNCTIONS
##############################################################################

__all__ = ['shifthue', 'gridcolors', 'alpinecolormap', 'vectocolor', 'bicolormap', 'hex2rgb', 'rgb2hex', 'rgb2hsv', 'hsv2rgb'] # apinecolortest and bicolormaptest not included


def _listify_colors(colors, origndim=None):
    ''' Do standard transformation on colors -- internal helpfer function '''
    if not origndim:
        colors = ut.dcp(colors) # So we don't overwrite the original
        origndim = np.ndim(colors) # Original dimensionality
        if origndim==1: 
            colors = [colors] # Wrap it in another list if needed
        colors = np.array(colors) # Just convert it to an array
        return colors, origndim
    else: # Reverse the transformation
        if origndim==1: 
            colors = colors[0] # Pull it out again
        return colors


def _processcolors(colors=None, asarray=False, ashex=False, reverse=False):
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
        if ashex:
            for c,color in enumerate(output):
                output[c] = rgb2hex(color)
    return output


def shifthue(colors=None, hueshift=0.0):
    '''
    Shift the hue of the colors being fed in.
    
    Example:
        colors = shifthue(colors=[(1,0,0),(0,1,0)], hueshift=0.5)
    '''
    colors, origndim = _listify_colors(colors)
    for c,color in enumerate(colors):
        hsvcolor = rgb_to_hsv(color)
        hsvcolor[0] = (hsvcolor[0]+hueshift) % 1.0 # Calculate new hue and return the modulus
        rgbcolor = hsv_to_rgb(hsvcolor)
        colors[c] = rgbcolor
    colors = _listify_colors(colors, origndim)
    return colors


def gridcolors(ncolors=10, limits=None, nsteps=20, asarray=False, ashex=False, reverse=False, hueshift=0, basis='default', doplot=False):
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
        basis: what basis to use -- options are 'colorbrewer', 'kelly', 'default', or 'none'

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

    Version: 2.0 (2018oct30) 
    """
    
    # Choose default colors
    if basis == 'default':
        if ncolors<=9: basis = 'colorbrewer' # Use these cos they're nicer
        else:          basis = 'kelly' # Use these cos there are more of them

    # Steal colorbrewer colors for small numbers of colors
    colorbrewercolors = np.array([
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
    
    # Steal Kelly's colors from https://gist.github.com/ollieglass/f6ddd781eeae1d24e391265432297538, removing
    # black: '222222', off-white: 'F2F3F4', mid-grey: '848482',
    kellycolors = ['F3C300', '875692', 'F38400', 'A1CAF1', 'BE0032', 'C2B280', '008856', 'E68FAC', '0067A5', 'F99379', '604E97', 'F6A600', 'B3446C', 'DCD300', '882D17', '8DB600', '654522', 'E25822', '2B3D26']
    for c,color in enumerate(kellycolors):
        kellycolors[c] = list(hex2rgb(color))
    kellycolors = np.array(kellycolors)
    
    if basis == 'colorbrewer' and ncolors<=len(colorbrewercolors):
        colors = colorbrewercolors[:ncolors]
    elif basis == 'kelly' and ncolors<=len(kellycolors):
        colors = kellycolors[:ncolors]
    else: # Too many colors, calculate instead
        ## Calculate sliding limits if none provided
        if limits is None:
            colorrange = 1-1/float(ncolors**0.5)
            limits = [0.5-colorrange/2, 0.5+colorrange/2]
        
        ## Calculate primitives and dot locations
        primitive = np.linspace(limits[0], limits[1], nsteps) # Define primitive color vector
        x, y, z = np.meshgrid(primitive, primitive, primitive) # Create grid of all possible points
        dots = np.transpose(np.array([x.flatten(), y.flatten(), z.flatten()])) # Flatten into an array of dots
        ndots = nsteps**3 # Calculate the number of dots
        
        ## Start from the colorbrewer colors
        if basis=='colorbrewer' or basis=='kelly':
            indices = [] # Initialize the array
            if   basis == 'colorbrewer': basiscolors = colorbrewercolors
            elif basis == 'kelly':       basiscolors = kellycolors
            for color in basiscolors:
                rgbdistances = dots - color # Calculate the distance in RGB space
                totaldistances = norm(rgbdistances,axis=1)
                closest = np.argmin(totaldistances)
                indices.append(closest)
        else:
            indices = [0]
        
        ## Calculate the distances
        for pt in range(ncolors-len(indices)): # Loop over each point
            totaldistances = np.inf+np.zeros(ndots) # Initialize distances
            for ind in indices: # Loop over each existing point
                rgbdistances = dots - dots[ind] # Calculate the distance in RGB space
                totaldistances = np.minimum(totaldistances, norm(rgbdistances,axis=1)) # Calculate the minimum Euclidean distance
            maxindex = np.argmax(totaldistances) # Find the point that maximizes the minimum distance
            indices.append(maxindex) # Append this index
        
        colors = dots[indices,:]
    
    ## Wrap up -- turn color array into a list, or reverse
    if hueshift: colors = shifthue(colors, hueshift=hueshift) # Shift hue if requested
    output = _processcolors(colors=colors, asarray=asarray, ashex=ashex, reverse=reverse)
    
    ## For plotting -- optional
    if doplot:
        from mpl_toolkits.mplot3d import Axes3D # analysis:ignore
        fig = pl.figure(facecolor='w')
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(colors[:,0], colors[:,1], colors[:,2], c=output, s=200, depthshade=False, lw=0)
        ax.set_xlabel('Red', fontweight='bold')
        ax.set_ylabel('Green', fontweight='bold')
        ax.set_zlabel('Blue', fontweight='bold')
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_zlim((0,1))
    
    return output
    
    


## Create colormap
def alpinecolormap(gap=0.1, mingreen=0.2, redbluemix=0.5, epsilon=0.01, demo=False):
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
    water = np.array([3,18,59])/256.
    desert = np.array([194,175,160*0.6])/256.
    forest1 = np.array([61,86,46])/256.
    forest2 = np.array([61,86,46])/256.*1.2
    rock = np.array([119,111,109])/256.*1.3
    snow = np.array([243,239,238])/256.
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
   
   
    def demoplot():
        maxheight = 3
        horizontalsize = 4;
        pl.seed(8)
        n = 100
        smoothing = 40;
        kernel = np.array([0.25,0.5,0.25])
        data = pl.randn(n,n)
        for s in range(smoothing): # Quick-and-dirty-and-slow smoothing
            for i in range(n): data[:,i] = np.convolve(data[:,i],kernel,mode='same')
            for i in range(n): data[i,:] = np.convolve(data[i,:],kernel,mode='same')
        data -= data.min()
        data /= data.max()
        data *= maxheight
        
        fig,ax = ax3d(returnfig=True, figsize=(18,8))
        ax.view_init(elev=45, azim=30)
        X = np.linspace(0,horizontalsize,n)
        X, Y = np.meshgrid(X, X)
        surf = ax.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=alpinecolormap(), linewidth=0, antialiased=False)
        cb = fig.colorbar(surf)
        cb.set_label('Height (km)',horizontalalignment='right', labelpad=50)
        pl.xlabel('Position (km)')
        pl.ylabel('Position (km)')
        pl.show()
    
        fig = pl.figure(figsize=(8,6))
        ax = fig.gca()
        X = np.linspace(0,horizontalsize,n)
        pcl = pl.pcolor(X, X, data, cmap=alpinecolormap(), linewidth=0, antialiased=False)
        cb2 = fig.colorbar(pcl)
        cb2.set_label('Height (km)',horizontalalignment='right', labelpad=50)
        pl.xlabel('Position (km)')
        pl.ylabel('Position (km)')
        pl.show()
        
    if demo: demoplot()

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

   Version: 2018sep25
   """
   from numpy import array, zeros
   from pylab import cm
   
   if cmap==None:
      cmap = cm.jet
   elif type(cmap)==str:
      try: cmap = getattr(cm,cmap)
      except: raise Exception('%s is not a valid color map; choices are:\n%s' % (cmap, '\n'.join(sorted(cm.datad.keys()))))

    # If a scalar is supplied, convert it to a vector instead
   if ut.isnumber(vector):
        vector = np.arange(vector)

   # The vector has elements
   if len(vector):
      vector = np.array(vector) # Just to be sure
      vector = vector-vector.min() # Subtract minimum
      vector = vector/float(vector.max()) # Divide by maximum
      nelements = len(vector) # Count number of elements
      colors=zeros((nelements,4))
      for i in range(nelements):
         colors[i,:]=array(cmap(vector[i]))

   # It doesn't; just return black
   else:
       colors=(0,0,0,1)
   
   # Process output
   output = _processcolors(colors=colors, asarray=asarray, reverse=reverse)

   return output



## Create colormap
def bicolormap(gap=0.1, mingreen=0.2, redbluemix=0.5, epsilon=0.01, demo=False):
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
    
    def demoplot():
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
    
    if demo: demoplot()

    return cmap



def hex2rgb(string):
    ''' A little helper function to convert e.g. '86bc25' to a pleasing shade of green. '''
    if string[0] == '#':
        string = string[1:] # Trim leading #, if it exists
    if len(string)==3:
        string = string[0]*2+string[1]*2+string[2]*2 # Convert e.g. '8b2' to '88bb22'
    if len(string)!=6:
        errormsg = 'Cannot convert "%s" to an RGB color:must be 3 or 6 characters long' % string
        raise Exception(errormsg)
    if six.PY3: hexstring = bytes.fromhex(string) # Ensure it's the right type
    else:       hexstring = string.decode('hex')
    rgb = np.array(unpack('BBB',hexstring),dtype=float)/255.
    return rgb



def rgb2hex(arr):
    ''' And going back the other way '''
    arr = np.array(arr)
    if len(arr) != 3:
        errormsg = 'Cannot convert "%s" to hex: wrong length' % arr
        raise Exception(errormsg)
    if all(arr<=1): arr *= 255. # Convert from 0-1 to 0-255
    hexstr = '#%02x%02x%02x' % (int(arr[0]), int(arr[1]), int(arr[2]))
    return hexstr



def rgb2hsv(colors=None):
    ''' Shortcut to Matplotlib's rgb_to_hsv method, accepts a color triplet or a list/array of color triplets '''
    colors, origndim = _listify_colors(colors)
    for c,color in enumerate(colors):
        hsvcolor = rgb_to_hsv(color)
        colors[c] = hsvcolor
    colors = _listify_colors(colors, origndim)
    return colors



def hsv2rgb(colors=None):
    ''' Shortcut to Matplotlib's hsv_to_rgb method, accepts a color triplet or a list/array of color triplets '''
    colors, origndim = _listify_colors(colors)
    for c,color in enumerate(colors):
        hsvcolor = hsv_to_rgb(color)
        colors[c] = hsvcolor
    colors = _listify_colors(colors, origndim)
    return colors


##############################################################################
### PLOTTING FUNCTIONS
##############################################################################

__all__ += ['ax3d', 'scatter3d', 'surf3d', 'bar3d', 'boxoff', 'setaxislim', 'setxlim', 'setylim', 'commaticks', 'SItickformatter', 'SIticks']


def ax3d(fig=None, returnfig=False, silent=False, **kwargs):
    ''' Create a 3D axis to plot in -- all arguments are passed to figure() '''
    from mpl_toolkits.mplot3d import Axes3D # analysis:ignore
    if fig is None: 
        fig = pl.figure(**kwargs)
    else:
        silent = False # Never close an already open figure
    ax = fig.gca(projection='3d')
    if silent:
        pl.close(fig)
    if returnfig:
        return fig,ax
    else:
        return ax


def scatter3d(x, y, z, c=None, fig=None, returnfig=False, plotkwargs=None, **kwargs):
    ''' Plot 3D data as a scatter '''
    # Set default arguments
    if plotkwargs is None: plotkwargs = {}
    settings = {'s':200, 'depthshade':False, 'linewidth':0}
    settings.update(plotkwargs)
    
    # Create figure
    fig,ax = ax3d(returnfig=True, fig=fig, **kwargs)
    ax.view_init(elev=45, azim=30)

    ax.scatter(x, y, z, c=c, **settings)

    if returnfig:
        return fig,ax
    else:
        return ax


def surf3d(data, fig=None, returnfig=False, plotkwargs=None, colorbar=True, **kwargs):
    ''' Plot 2D data as a 3D surface '''
    
    # Set default arguments
    if plotkwargs is None: plotkwargs = {}
    settings = {'rstride':1, 'cstride':1, 'linewidth':0, 'antialiased':False, 'cmap':'viridis'}
    settings.update(plotkwargs)
    
    # Create figure
    fig,ax = ax3d(returnfig=True, fig=fig, **kwargs)
    ax.view_init(elev=45, azim=30)
    ny,nx = pl.array(data).shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, data, **settings)
    if colorbar:
        fig.colorbar(surf)
    
    if returnfig:
        return fig,ax
    else:
        return ax



def bar3d(data, fig=None, returnfig=False, plotkwargs=None, **kwargs):
    ''' Plot 2D data as 3D bars '''
    
    # Set default arguments
    if plotkwargs is None: plotkwargs = {}
    settings = {'width':0.8, 'depth':0.8, 'shade':True, 'cmap':'viridis'}
    settings.update(plotkwargs)
    
    # Create figure
    fig,ax = ax3d(returnfig=True, fig=fig, **kwargs)
    
    x, y, z = [], [], []
    dx, dy, dz = [], [], []
    color = vectocolor(data.flatten(), cmap=settings['cmap'])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x.append(i)
            y.append(j)
            z.append(0)
            dx.append(settings['width'])
            dy.append(settings['depth'])
            dz.append(data[i,j])
    ax.bar3d(x=x, y=y, z=z, dx=settings['width'], dy=settings['depth'], dz=dz, color=color, shade=settings['shade'])
    
    if returnfig:
        return fig,ax
    else:
        return ax



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
    
    

def setaxislim(which=None, ax=None, data=None):
    '''
    A small script to determine how the y limits should be set. Looks
    at all data (a list of arrays) and computes the lower limit to
    use, e.g.
    
        setaxislim([np.array([-3,4]), np.array([6,4,6])], ax)
    
    will keep Matplotlib's lower limit, since at least one data value
    is below 0.
    
    Note, if you just want to set the lower limit, you can do that 
    with this function via:
        setaxislim()
    '''

    # Handle which axis
    if which is None:
        which = 'both'
    if which not in ['x','y','both']:
        errormsg = 'Setting axis limit for axis %s is not supported' % which
        raise Exception(errormsg)
    if which == 'both':
        setaxislim(which='x', ax=ax, data=data)
        setaxislim(which='y', ax=ax, data=data)
        return None

    # Ensure axis exists
    if ax is None:
        ax = pl.gca()

    # Get current limits
    if   which == 'x': currlower, currupper = ax.get_xlim()
    elif which == 'y': currlower, currupper = ax.get_ylim()
    
    # Calculate the lower limit based on all the data
    lowerlim = 0
    upperlim = 0
    if ut.checktype(data, 'arraylike'): # Ensure it's numeric data (probably just None)
        flatdata = ut.promotetoarray(data).flatten() # Make sure it's iterable
        lowerlim = min(lowerlim, flatdata.min())
        upperlim = max(upperlim, flatdata.max())
    
    # Set the new y limits
    if lowerlim<0: lowerlim = currlower # If and only if the data lower limit is negative, use the plotting lower limit
    upperlim = max(upperlim, currupper) # Shouldn't be an issue, but just in case...
    
    # Specify the new limits and return
    if   which == 'x': ax.set_xlim((lowerlim, upperlim))
    elif which == 'y': ax.set_ylim((lowerlim, upperlim))
    return lowerlim,upperlim


def setxlim(data=None, ax=None):
    ''' See setaxislim '''
    return setaxislim(data=data, ax=ax, which='x')

def setylim(data=None, ax=None):
    ''' See setaxislim '''
    return setaxislim(data=data, ax=ax, which='y')


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

__all__ += ['savefigs', 'loadfig', 'emptyfig', 'separatelegend']



def savefigs(figs=None, filetype=None, filename=None, folder=None, savefigargs=None, aslist=False, verbose=False, **kwargs):
    '''
    Save the requested plots to disk.
    
    Arguments:
        figs        -- the figure objects to save
        filetype    -- the file type; can be 'fig', 'singlepdf' (default), or anything supported by savefig()
        filename    -- the file to save to (only uses path if multiple files)
        folder      -- the folder to save the file(s) in
        savefigargs -- dictionary of arguments passed to savefig()
        aslist      -- whether or not return a list even for a single file
    
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
    figs = odict.promote(figs)
    nfigs = len(figs)
    
    # Handle file types
    filenames = []
    if filetype=='singlepdf': # See http://matplotlib.org/examples/pylab_examples/multipage_pdf.html
        from matplotlib.backends.backend_pdf import PdfPages
        defaultname = 'figures.pdf'
        fullpath = fio.makefilepath(filename=filename, folder=folder, default=defaultname, ext='pdf')
        pdf = PdfPages(fullpath)
        filenames.append(fullpath)
        if verbose: print('PDF saved to %s' % fullpath)
    for p,item in enumerate(figs.items()):
        key,plt = item
        # Handle filename
        if filename and nfigs==1: # Single plot, filename supplied -- use it
            fullpath = fio.makefilepath(filename=filename, folder=folder, default='Figure', ext=filetype) # NB, this filename not used for singlepdf filetype, so it's OK
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
            if verbose: print('Figure object saved to %s' % fullpath)
        else:
            reanimateplots(plt)
            if filetype=='singlepdf':
                pdf.savefig(figure=plt, **defaultsavefigargs) # It's confusing, but defaultsavefigargs is correct, since we updated it from the user version
            else:
                plt.savefig(fullpath, **defaultsavefigargs)
                filenames.append(fullpath)
                if verbose: print('%s plot saved to %s' % (filetype.upper(),fullpath))
            pl.close(plt)
    
    # Do final tidying
    if filetype=='singlepdf': pdf.close()
    if wasinteractive: pl.ion()
    if aslist or len(filenames)>1:
        return filenames
    else:
        return filenames[0]


def loadfig(filename=None):
    '''
    Load a plot from a file and reanimate it.
    
    Example usage:
        import pylab as pl
        import sciris as sc
        fig = pl.figure(); pl.plot(pl.rand(10))
        sc.savefigs(fig, filetype='fig', filename='example.fig')
    Later:
        example = sc.loadfig('example.fig')
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


def emptyfig():
    ''' The emptiest figure possible '''
    fig = pl.Figure(facecolor='None')
    return fig


def separatelegend(ax=None, handles=None, labels=None, reverse=False, figsettings=None, legendsettings=None):
    ''' Allows the legend of a figure to be rendered in a separate window instead '''
    
    # Handle settings
    if figsettings    is None: figsettings = {}
    if legendsettings is None: legendsettings = {}
    f_settings = {'figsize':(4.0,4.8)} # (6.4,4.8) is the default, so make it a bit narrower
    l_settings = {'loc': 'center', 'bbox_to_anchor': None, 'frameon': False}
    f_settings.update(figsettings)
    l_settings.update(legendsettings)

    # Construct handle and label list, from either
    # - A list of handles and a list of labels
    # - A list of handles, where each handle contains the label
    # - An axis object, containing the objects that should appear in the legend
    # - A figure object, from which the first axis will be used
    if handles is None:
        if ax is None:
            ax = pl.gca()
        else:
            if isinstance(ax, pl.Figure): ax = ax.axes[0]  # Allows an argument of a figure instead of an axes
        handles, labels = ax.get_legend_handles_labels()
    else:
        if labels is None:
            labels = [h.get_label() for h in handles]
        else:
            assert len(handles) == len(labels), "Number of handles (%d) and labels (%d) must match" % (len(handles),len(labels))

    # Set up new plot
    fig = pl.figure(**f_settings)
    ax = fig.add_subplot(111)
    ax.set_position([-0.05,-0.05,1.1,1.1]) # This cuts off the axis labels, ha-ha
    ax.set_axis_off()  # Hide axis lines

    # A legend renders the line/patch based on the object handle. However, an object
    # can only appear in one figure. Thus, if the legend is in a different figure, the
    # object cannot be shown in both the original figure and in the legend. Thus we need
    # to copy the handles, and use the copies to render the legend
    handles2 = []
    for h in handles:
        h2 = ut.cp(h)
        h2.axes = None
        h2.figure = None
        handles2.append(h2)

    # Reverse order, e.g. for stacked plots
    if reverse:
        handles2 = handles2[::-1]
        labels   = labels[::-1]
    
    # Plot the new legend
    ax.legend(handles=handles2, labels=labels, **l_settings)

    return fig