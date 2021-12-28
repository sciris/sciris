'''
Handle colors and colormaps.

Highlights:
    - Adds colormaps including 'turbo', 'parula', and 'orangeblue'
    - ``sc.hex2grb()/sc.rgb2hex()``: convert between different color conventions
    - ``sc.vectocolor()``: map a list of sequential values onto a list of colors
    - ``sc.gridcolors()``: map a list of qualitative categories onto a list of colors
'''

##############################################################################
#%% Imports
##############################################################################

import struct
import pylab as pl
import numpy as np
import matplotlib.colors as mplc
from . import sc_utils as scu


##############################################################################
#%% Color functions
##############################################################################

__all__ = ['shifthue', 'rgb2hex', 'hex2rgb', 'rgb2hsv', 'hsv2rgb']


def _listify_colors(colors, origndim=None):
    ''' Do standard transformation on colors -- internal helpfer function '''
    if not origndim:
        colors = scu.dcp(colors) # So we don't overwrite the original
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

    **Example**::

        colors = sc.shifthue(colors=[(1,0,0),(0,1,0)], hueshift=0.5)
    '''
    colors, origndim = _listify_colors(colors)
    for c,color in enumerate(colors):
        hsvcolor = mplc.rgb_to_hsv(color)
        hsvcolor[0] = (hsvcolor[0]+hueshift) % 1.0 # Calculate new hue and return the modulus
        rgbcolor = mplc.hsv_to_rgb(hsvcolor)
        colors[c] = rgbcolor
    colors = _listify_colors(colors, origndim)
    return colors



def rgb2hex(arr):
    '''
    A little helper function to convert e.g. [0.53, 0.74, 0.15] to a pleasing shade of green.

    **Example**::

        hx = sc.rgb2hex([0.53, 0.74, 0.15]) # Returns '#87bc26'
    '''
    arr = np.array(arr)
    if len(arr) != 3: # pragma: no cover
        errormsg = f'Cannot convert "{arr}" to hex: wrong length'
        raise ValueError(errormsg)
    if all(arr<=1): arr *= 255. # Convert from 0-1 to 0-255
    hexstr = '#%02x%02x%02x' % (int(arr[0]), int(arr[1]), int(arr[2]))
    return hexstr


def hex2rgb(string):
    '''
    A little helper function to convert e.g. '87bc26' to a pleasing shade of green.

    **Example**::

        rgb = sc.hex2rgb('#87bc26') # Returns array([0.52941176, 0.7372549 , 0.14901961])
    '''
    if string[0] == '#':
        string = string[1:] # Trim leading #, if it exists
    if len(string)==3:
        string = string[0]*2+string[1]*2+string[2]*2 # Convert e.g. '8b2' to '88bb22'
    if len(string)!=6: # pragma: no cover
        errormsg = f'Cannot convert "{string}" to an RGB color: must be 3 or 6 characters long'
        raise ValueError(errormsg)
    hexstring = bytes.fromhex(string) # Ensure it's the right type
    rgb = np.array(struct.unpack('BBB',hexstring),dtype=float)/255.
    return rgb


def rgb2hsv(colors=None):
    '''
    Shortcut to Matplotlib's rgb_to_hsv method, accepts a color triplet or a list/array of color triplets.

    **Example**::

        hsv = sc.rgb2hsv([0.53, 0.74, 0.15]) # Returns array([0.2259887, 0.7972973, 0.74     ])
    '''
    colors, origndim = _listify_colors(colors)
    for c,color in enumerate(colors):
        hsvcolor = mplc.rgb_to_hsv(color)
        colors[c] = hsvcolor
    colors = _listify_colors(colors, origndim)
    return colors



def hsv2rgb(colors=None):
    '''
    Shortcut to Matplotlib's hsv_to_rgb method, accepts a color triplet or a list/array of color triplets.

    **Example**::

        rgb = sc.hsv2rgb([0.23, 0.80, 0.74]) # Returns array([0.51504, 0.74   , 0.148  ])
    '''
    colors, origndim = _listify_colors(colors)
    for c,color in enumerate(colors):
        hsvcolor = mplc.hsv_to_rgb(color)
        colors[c] = hsvcolor
    colors = _listify_colors(colors, origndim)
    return colors



##############################################################################
#%% Colormap-related functions
##############################################################################

__all__ += ['vectocolor', 'arraycolors', 'gridcolors', 'midpointnorm', 'colormapdemo']


def vectocolor(vector, cmap=None, asarray=True, reverse=False, minval=None, maxval=None, midpoint=None, norm=None):
    """
    This function converts a vector (i.e., 1D array) of N values into an Nx3 matrix
    of color values according to the current colormap. It automatically scales the
    vector to provide maximum dynamic range for the color map.

    Note: see sc.arraycolors() for multidimensional input.

    Args:
        vector (array): Input vector (or list, it's converted to an array)
        cmap (str): is the colormap (default: current)
        asarray (bool): whether to return as an array (otherwise, a list of tuples)
        reverse (bool): whether to reverse the list of colors
        minval (float): the minimum value to use
        maxval (float): the maximum value to use
        midpoint (float): the midpoint value to use

    Returns:
        colors (array): Nx4 array of RGB-alpha color values

    **Example**::

        n = 1000
        x = randn(n,1);
        y = randn(n,1);
        c = sc.vectocolor(y);
        pl.scatter(x, y, c=c, s=50)

    New in version 1.2.0: midpoint argument.
    """

    from numpy import array, zeros
    from pylab import cm

    if cmap is None:
        cmap = pl.get_cmap() # Get current colormap
    elif type(cmap) == str:
        try:
            cmap = cm.get_cmap(cmap)
        except: # pragma: no cover
            choices = "\n".join(sorted(cm.datad.keys()))
            raise ValueError(f'{cmap} is not a valid color map; choices are:\n{choices}')

    # If a scalar is supplied, convert it to a vector instead
    if scu.isnumber(vector):
        vector = np.linspace(0, 1, vector)

    # Usual situation -- the vector has elements
    vector = scu.dcp(vector) # To avoid in-place modification
    vector = np.array(vector) # Just to be sure
    if len(vector):
        if minval is None:
            minval = vector.min()
        if maxval is None:
            maxval = vector.max()

        vector = vector-minval # Subtract minimum
        vector = vector/float(maxval-minval) # Divide by maximum
        if midpoint is not None:
            norm = midpointnorm(vcenter=midpoint, vmin=0, vmax=1)
            vector = np.array(norm(vector))
        nelements = len(vector) # Count number of elements
        colors = zeros((nelements,4))
        for i in range(nelements):
            colors[i,:] = array(cmap(vector[i]))

    # It doesn't; just return black
    else:
        colors=(0,0,0,1)

    # Process output
    output = _processcolors(colors=colors, asarray=asarray, reverse=reverse)

    return output



def arraycolors(arr, **kwargs):
    """
    Map an N-dimensional array of values onto the current colormap. An extension
    of vectocolor() for multidimensional arrays; see that function for additional
    arguments.

    **Example**::

        n = 1000
        ncols = 5
        arr = pl.rand(n,ncols)
        for c in range(ncols):
            arr[:,c] += c
        x = pl.rand(n)
        y = pl.rand(n)
        colors = sc.arraycolors(arr)
        pl.figure(figsize=(20,16))
        for c in range(ncols):
            pl.scatter(x+c, y, s=50, c=colors[:,c])

    Version: 2020mar07
    """
    arr = scu.dcp(arr) # Avoid modifications
    new_shape = arr.shape + (4,) # RGBα
    colors = np.zeros(new_shape)
    colorvec = vectocolor(vector=arr.reshape(-1), **kwargs)
    colors = colorvec.reshape(new_shape)
    return colors



def gridcolors(ncolors=10, limits=None, nsteps=20, asarray=False, ashex=False, reverse=False, hueshift=0, basis='default', demo=False):
    """
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

    **Example**::

        from pylab import *
        from sciris import gridcolors
        ncolors = 10
        piedata = rand(ncolors)
        colors = gridcolors(ncolors)
        figure()
        pie(piedata, colors=colors)
        gridcolors(ncolors, demo=True)
        show()

    Version: 2018oct30
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
                totaldistances = np.linalg.norm(rgbdistances,axis=1)
                closest = np.argmin(totaldistances)
                indices.append(closest)
        else:
            indices = [0]

        ## Calculate the distances
        for pt in range(ncolors-len(indices)): # Loop over each point
            totaldistances = np.inf+np.zeros(ndots) # Initialize distances
            for ind in indices: # Loop over each existing point
                rgbdistances = dots - dots[ind] # Calculate the distance in RGB space
                totaldistances = np.minimum(totaldistances, np.linalg.norm(rgbdistances,axis=1)) # Calculate the minimum Euclidean distance
            maxindex = np.argmax(totaldistances) # Find the point that maximizes the minimum distance
            indices.append(maxindex) # Append this index

        colors = dots[indices,:]

    ## Wrap up -- turn color array into a list, or reverse
    if hueshift: colors = shifthue(colors, hueshift=hueshift) # Shift hue if requested
    output = _processcolors(colors=colors, asarray=asarray, ashex=ashex, reverse=reverse)

    ## For plotting -- optional
    if demo:
        from . import sc_plotting as scp # To avoid circular import
        ax = scp.scatter3d(colors[:,0], colors[:,1], colors[:,2], c=output, s=200, depthshade=False, lw=0, figkwargs={'facecolor':'w'})
        ax.set_xlabel('Red', fontweight='bold')
        ax.set_ylabel('Green', fontweight='bold')
        ax.set_zlabel('Blue', fontweight='bold')
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_zlim((0,1))

    return output



def midpointnorm(vcenter=0, vmin=None, vmax=None):
    '''
    Alias to Matplotlib's TwoSlopeNorm. Used to place the center of the colormap
    somewhere other than the center of the data.

    Args:
        vcenter (float): the center of the colormap (0 by default)
        vmin (float): the minimum of the colormap
        vmax (float): the maximum of the colormap

    **Example**::

        data = pl.rand(10,10) - 0.2
        pl.pcolor(data, cmap='bi', norm=sc.midpointnorm())

    New in version 1.2.0.
    '''
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    return norm



def colormapdemo(cmap=None, n=None, smoothing=None, randseed=None, doshow=True):
    '''
    Demonstrate a color map using simulated elevation data, shown in both 2D and
    3D. The argument can be either a colormap itself or a string describing a
    colormap.

    **Examples**::

        sc.colormapdemo('inferno') # Use a registered Matplotlib colormap
        sc.colormapdemo('parula') # Use a registered Sciris colormap
        sc.colormapdemo(sc.alpinecolormap(), n=200, smoothing=20, randseed=2942) # Use a colormap object

    Version: 2019aug22
    '''
    from . import sc_plotting as scp # To avoid circular import

    # Set data
    if n         is None: n         = 100
    if smoothing is None: smoothing = 40
    if randseed  is None: randseed  = 8
    if cmap is None: cmap = 'parula' # For no particular reason
    maxheight = 1
    horizontalsize = 4
    pl.seed(randseed)
    kernel = np.array([0.25,0.5,0.25])
    data = pl.randn(n,n)
    for s in range(smoothing): # Quick-and-dirty-and-slow smoothing
        for i in range(n): data[:,i] = np.convolve(data[:,i],kernel,mode='same')
        for i in range(n): data[i,:] = np.convolve(data[i,:],kernel,mode='same')
    data -= data.min()
    data /= data.max()
    data *= maxheight

    # Plot in 2D
    fig1 = pl.figure(figsize=(10,8))
    X = np.linspace(0,horizontalsize,n)
    pcl = pl.pcolor(X, X, data, cmap=cmap, linewidth=0, antialiased=False, shading='auto')
    cb2 = fig1.colorbar(pcl)
    cb2.set_label('Height (km)',horizontalalignment='right', labelpad=50)
    pl.xlabel('Position (km)')
    pl.ylabel('Position (km)')
    if doshow:
        pl.show()

    # Plot in 3D
    fig2,ax2 = scp.fig3d(returnax=True, figsize=(18,8))
    ax2.view_init(elev=45, azim=30)
    X = np.linspace(0,horizontalsize,n)
    X, Y = np.meshgrid(X, X)
    surf = ax2.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=cmap, linewidth=0, antialiased=False)
    cb = fig2.colorbar(surf)
    cb.set_label('Height (km)', horizontalalignment='right', labelpad=50)
    pl.xlabel('Position (km)')
    pl.ylabel('Position (km)')
    if doshow:
        pl.show()

    return {'2d':fig1, '3d':fig2}



##############################################################################
#%% Colormaps
##############################################################################

__all__ += ['alpinecolormap', 'bicolormap', 'parulacolormap', 'turbocolormap', 'bandedcolormap', 'orangebluecolormap']

def alpinecolormap(apply=False):
    """
    This function generates a map based on ascending height. Based on data from
    Kazakhstan.

    **Test case**::

        sc.colormapdemo('alpine')

    **Usage example**::

        import sciris as sc
        import pylab as pl
        pl.imshow(pl.randn(20,20), interpolation='none', cmap=sc.alpinecolormap())

    Version: 2014aug06
    """

    # Set parameters
    water = np.array([3,18,59])/256.
    desert = np.array([194,175,160*0.6])/256.
    forest1 = np.array([61,86,46])/256.
    forest2 = np.array([61,86,46])/256.*1.2
    rock = np.array([119,111,109])/256.*1.3
    snow = np.array([243,239,238])/256.
    breaks = [0.0,0.5,0.7,0.8,0.9,1.0]

    # Create dictionary
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

    # Make map
    cmap = mplc.LinearSegmentedColormap('alpine', cdict, 256)
    if apply:
        pl.set_cmap(cmap)
    return cmap



def bicolormap(gap=0.1, mingreen=0.2, redbluemix=0.5, epsilon=0.01, demo=False, apply=False):
    """
    This function generators a two-color map, blue for negative, red for
    positive changes, with grey in the middle. The input argument is how much
    of a color gap there is between the red scale and the blue one.

    Args:
      gap: sets how big of a gap between red and blue color scales there is (0=no gap; 1=pure red and pure blue)
      mingreen: how much green to include at the extremes of the red-blue color scale
      redbluemix: how much red to mix with the blue and vice versa at the extremes of the scale
      epsilon: what fraction of the colormap to make gray in the middle

    **Examples**::

        bicolormap(gap=0,mingreen=0,redbluemix=1,epsilon=0) # From pure red to pure blue with white in the middle
        bicolormap(gap=0,mingreen=0,redbluemix=0,epsilon=0.1) # Red -> yellow -> gray -> turquoise -> blue
        bicolormap(gap=0.3,mingreen=0.2,redbluemix=0,epsilon=0.01) # Red and blue with a sharp distinction between

    Version: 2013sep13
    """
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

    cmap = mplc.LinearSegmentedColormap('bi',cdict,256)
    if apply:
        pl.set_cmap(cmap)

    def demoplot(): # pragma: no cover
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


def parulacolormap(apply=False):
    '''
    Create a map similar to Viridis, but brighter. Set apply=True to use
    immediately.

    **Demo and example**::

        cmap = sc.parulacolormap()
        sc.colormapdemo(cmap=cmap)

    Version: 2019aug22
    '''
    data = [[0.2422,0.1504,0.6603], [0.2444,0.1534,0.6728], [0.2464,0.1569,0.6847], [0.2484,0.1607,0.6961], [0.2503,0.1648,0.7071], [0.2522,0.1689,0.7179], [0.2540,0.1732,0.7286], [0.2558,0.1773,0.7393],
            [0.2576,0.1814,0.7501], [0.2594,0.1854,0.7610], [0.2611,0.1893,0.7719], [0.2628,0.1932,0.7828], [0.2645,0.1972,0.7937], [0.2661,0.2011,0.8043], [0.2676,0.2052,0.8148], [0.2691,0.2094,0.8249],
            [0.2704,0.2138,0.8346], [0.2717,0.2184,0.8439], [0.2729,0.2231,0.8528], [0.2740,0.2280,0.8612], [0.2749,0.2330,0.8692], [0.2758,0.2382,0.8767], [0.2766,0.2435,0.8840], [0.2774,0.2489,0.8908],
            [0.2781,0.2543,0.8973], [0.2788,0.2598,0.9035], [0.2794,0.2653,0.9094], [0.2798,0.2708,0.9150], [0.2802,0.2764,0.9204], [0.2806,0.2819,0.9255], [0.2809,0.2875,0.9305], [0.2811,0.2930,0.9352],
            [0.2813,0.2985,0.9397], [0.2814,0.3040,0.9441], [0.2814,0.3095,0.9483], [0.2813,0.3150,0.9524], [0.2811,0.3204,0.9563], [0.2809,0.3259,0.9600], [0.2807,0.3313,0.9636], [0.2803,0.3367,0.9670],
            [0.2798,0.3421,0.9702], [0.2791,0.3475,0.9733], [0.2784,0.3529,0.9763], [0.2776,0.3583,0.9791], [0.2766,0.3638,0.9817], [0.2754,0.3693,0.9840], [0.2741,0.3748,0.9862], [0.2726,0.3804,0.9881],
            [0.2710,0.3860,0.9898], [0.2691,0.3916,0.9912], [0.2670,0.3973,0.9924], [0.2647,0.4030,0.9935], [0.2621,0.4088,0.9946], [0.2591,0.4145,0.9955], [0.2556,0.4203,0.9965], [0.2517,0.4261,0.9974],
            [0.2473,0.4319,0.9983], [0.2424,0.4378,0.9991], [0.2369,0.4437,0.9996], [0.2311,0.4497,0.9995], [0.2250,0.4559,0.9985], [0.2189,0.4620,0.9968], [0.2128,0.4682,0.9948], [0.2066,0.4743,0.9926],
            [0.2006,0.4803,0.9906], [0.1950,0.4861,0.9887], [0.1903,0.4919,0.9867], [0.1869,0.4975,0.9844], [0.1847,0.5030,0.9819], [0.1831,0.5084,0.9793], [0.1818,0.5138,0.9766], [0.1806,0.5191,0.9738],
            [0.1795,0.5244,0.9709], [0.1785,0.5296,0.9677], [0.1778,0.5349,0.9641], [0.1773,0.5401,0.9602], [0.1768,0.5452,0.9560], [0.1764,0.5504,0.9516], [0.1755,0.5554,0.9473], [0.1740,0.5605,0.9432],
            [0.1716,0.5655,0.9393], [0.1686,0.5705,0.9357], [0.1649,0.5755,0.9323], [0.1610,0.5805,0.9289], [0.1573,0.5854,0.9254], [0.1540,0.5902,0.9218], [0.1513,0.5950,0.9182], [0.1492,0.5997,0.9147],
            [0.1475,0.6043,0.9113], [0.1461,0.6089,0.9080], [0.1446,0.6135,0.9050], [0.1429,0.6180,0.9022], [0.1408,0.6226,0.8998], [0.1383,0.6272,0.8975], [0.1354,0.6317,0.8953], [0.1321,0.6363,0.8932],
            [0.1288,0.6408,0.8910], [0.1253,0.6453,0.8887], [0.1219,0.6497,0.8862], [0.1185,0.6541,0.8834], [0.1152,0.6584,0.8804], [0.1119,0.6627,0.8770], [0.1085,0.6669,0.8734], [0.1048,0.6710,0.8695],
            [0.1009,0.6750,0.8653], [0.0964,0.6789,0.8609], [0.0914,0.6828,0.8562], [0.0855,0.6865,0.8513], [0.0789,0.6902,0.8462], [0.0713,0.6938,0.8409], [0.0628,0.6972,0.8355], [0.0535,0.7006,0.8299],
            [0.0433,0.7039,0.8242], [0.0328,0.7071,0.8183], [0.0234,0.7103,0.8124], [0.0155,0.7133,0.8064], [0.0091,0.7163,0.8003], [0.0046,0.7192,0.7941], [0.0019,0.7220,0.7878], [0.0009,0.7248,0.7815],
            [0.0018,0.7275,0.7752], [0.0046,0.7301,0.7688], [0.0094,0.7327,0.7623], [0.0162,0.7352,0.7558], [0.0253,0.7376,0.7492], [0.0369,0.7400,0.7426], [0.0504,0.7423,0.7359], [0.0638,0.7446,0.7292],
            [0.0770,0.7468,0.7224], [0.0899,0.7489,0.7156], [0.1023,0.7510,0.7088], [0.1141,0.7531,0.7019], [0.1252,0.7552,0.6950], [0.1354,0.7572,0.6881], [0.1448,0.7593,0.6812], [0.1532,0.7614,0.6741],
            [0.1609,0.7635,0.6671], [0.1678,0.7656,0.6599], [0.1741,0.7678,0.6527], [0.1799,0.7699,0.6454], [0.1853,0.7721,0.6379], [0.1905,0.7743,0.6303], [0.1954,0.7765,0.6225], [0.2003,0.7787,0.6146],
            [0.2061,0.7808,0.6065], [0.2118,0.7828,0.5983], [0.2178,0.7849,0.5899], [0.2244,0.7869,0.5813], [0.2318,0.7887,0.5725], [0.2401,0.7905,0.5636], [0.2491,0.7922,0.5546], [0.2589,0.7937,0.5454],
            [0.2695,0.7951,0.5360], [0.2809,0.7964,0.5266], [0.2929,0.7975,0.5170], [0.3052,0.7985,0.5074], [0.3176,0.7994,0.4975], [0.3301,0.8002,0.4876], [0.3424,0.8009,0.4774], [0.3548,0.8016,0.4669],
            [0.3671,0.8021,0.4563], [0.3795,0.8026,0.4454], [0.3921,0.8029,0.4344], [0.4050,0.8031,0.4233], [0.4184,0.8030,0.4122], [0.4322,0.8028,0.4013], [0.4463,0.8024,0.3904], [0.4608,0.8018,0.3797],
            [0.4753,0.8011,0.3691], [0.4899,0.8002,0.3586], [0.5044,0.7993,0.3480], [0.5187,0.7982,0.3374], [0.5329,0.7970,0.3267], [0.5470,0.7957,0.3159], [0.5609,0.7943,0.3050], [0.5748,0.7929,0.2941],
            [0.5886,0.7913,0.2833], [0.6024,0.7896,0.2726], [0.6161,0.7878,0.2622], [0.6297,0.7859,0.2521], [0.6433,0.7839,0.2423], [0.6567,0.7818,0.2329], [0.6701,0.7796,0.2239], [0.6833,0.7773,0.2155],
            [0.6963,0.7750,0.2075], [0.7091,0.7727,0.1998], [0.7218,0.7703,0.1924], [0.7344,0.7679,0.1852], [0.7468,0.7654,0.1782], [0.7590,0.7629,0.1717], [0.7710,0.7604,0.1658], [0.7829,0.7579,0.1608],
            [0.7945,0.7554,0.1570], [0.8060,0.7529,0.1546], [0.8172,0.7505,0.1535], [0.8281,0.7481,0.1536], [0.8389,0.7457,0.1546], [0.8495,0.7435,0.1564], [0.8600,0.7413,0.1587], [0.8703,0.7392,0.1615],
            [0.8804,0.7372,0.1650], [0.8903,0.7353,0.1695], [0.9000,0.7336,0.1749], [0.9093,0.7321,0.1815], [0.9184,0.7308,0.1890], [0.9272,0.7298,0.1973], [0.9357,0.7290,0.2061], [0.9440,0.7285,0.2151],
            [0.9523,0.7284,0.2237], [0.9606,0.7285,0.2312], [0.9689,0.7292,0.2373], [0.9770,0.7304,0.2418], [0.9842,0.7330,0.2446], [0.9900,0.7365,0.2429], [0.9946,0.7407,0.2394], [0.9966,0.7458,0.2351],
            [0.9971,0.7513,0.2309], [0.9972,0.7569,0.2267], [0.9971,0.7626,0.2224], [0.9969,0.7683,0.2181], [0.9966,0.7740,0.2138], [0.9962,0.7798,0.2095], [0.9957,0.7856,0.2053], [0.9949,0.7915,0.2012],
            [0.9938,0.7974,0.1974], [0.9923,0.8034,0.1939], [0.9906,0.8095,0.1906], [0.9885,0.8156,0.1875], [0.9861,0.8218,0.1846], [0.9835,0.8280,0.1817], [0.9807,0.8342,0.1787], [0.9778,0.8404,0.1757],
            [0.9748,0.8467,0.1726], [0.9720,0.8529,0.1695], [0.9694,0.8591,0.1665], [0.9671,0.8654,0.1636], [0.9651,0.8716,0.1608], [0.9634,0.8778,0.1582], [0.9619,0.8840,0.1557], [0.9608,0.8902,0.1532],
            [0.9601,0.8963,0.1507], [0.9596,0.9023,0.1480], [0.9595,0.9084,0.1450], [0.9597,0.9143,0.1418], [0.9601,0.9203,0.1382], [0.9608,0.9262,0.1344], [0.9618,0.9320,0.1304], [0.9629,0.9379,0.1261],
            [0.9642,0.9437,0.1216], [0.9657,0.9494,0.1168], [0.9674,0.9552,0.1116], [0.9692,0.9609,0.1061], [0.9711,0.9667,0.1001], [0.9730,0.9724,0.0938], [0.9749,0.9782,0.0872], [0.9769,0.9839,0.0805]]

    cmap = mplc.LinearSegmentedColormap.from_list('parula', data)
    if apply:
        pl.set_cmap(cmap)
    return cmap


def turbocolormap(apply=False):
    '''
    NOTE: as of Matplotlib 3.4.0, this colormap is included by default, and will
    soon be removed from Sciris.

    Copyright 2019 Google LLC.

    SPDX-License-Identifier: Apache-2.0

    Author: Anton Mikhailov

    Borrowed directly from https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f, with thanks!

    Create a map similar to Jet, but better. Set apply=True to use
    immediately.

    **Demo and example**::

        cmap = sc.turbocolormap()
        sc.colormapdemo(cmap=cmap)

    Version: 2020mar20
    '''
    data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],
            [0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],
            [0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],
            [0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],
            [0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],
            [0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],
            [0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],
            [0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],
            [0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],
            [0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],
            [0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],
            [0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],
            [0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],
            [0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],
            [0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],
            [0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],
            [0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],
            [0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],
            [0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],
            [0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],
            [0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],
            [0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],
            [0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],
            [0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],
            [0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],
            [0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],
            [0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],
            [0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],
            [0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],
            [0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],
            [0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],
            [0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]

    cmap = mplc.LinearSegmentedColormap.from_list('turbo', data)
    if apply:
        pl.set_cmap(cmap)
    return cmap



def bandedcolormap(minvalue=None, minsaturation=None, hueshift=None, saturationscale=None, npts=None, apply=False):
    '''
    Map colors onto bands of hue and saturation, with lightness mapped linearly.
    Unlike most colormaps, this colormap does not aim to be percentually uniform,
    but rather aims to make it easy to relate colors to as-exact-as-possible numbers
    (while still maintaining a semblance of overall trend from low to high).

    **Demo and example**::

        cmap = sc.bandedcolormap(minvalue=0, minsaturation=0)
        sc.colormapdemo(cmap=cmap)

    Version: 2019aug22
    '''
    # Set parameters
    if minvalue        is None: minvalue        = 0.1
    if hueshift        is None: hueshift        = 0.8
    if minsaturation   is None: minsaturation   = 0.5
    if saturationscale is None: saturationscale = 4.3
    if npts            is None: npts            = 256

    # Calculate
    x = pl.linspace(0, pl.pi, npts)
    hsv = pl.zeros((npts, 3))
    hsv[:,2] = pl.sqrt(pl.linspace(minvalue,1,npts)) # Value: map linearly
    hsv[:,0] = pl.sin(x+hueshift)**2 # Hue: a big sine wave
    hsv[:,1] = minsaturation+(1-minsaturation)*pl.sin(saturationscale*x)**2 # Saturation: a small sine wave
    data = hsv2rgb(hsv)

    # Create and use
    cmap = mplc.LinearSegmentedColormap.from_list('banded', data)
    if apply:
        pl.set_cmap(cmap)
    return cmap


def orangebluecolormap(apply=False):
    '''
    Create an orange-blue colormap; most like RdYlBu but more pleasing. Created
    by Prashanth Selvaraj.

    **Demo and example**::

        cmap = sc.orangebluecolormap()
        sc.colormapdemo(cmap=cmap)

    New in version 1.0.0.
    '''
    bottom = pl.cm.get_cmap('Oranges', 128)
    top = pl.cm.get_cmap('Blues_r', 128)
    x = np.linspace(0, 1, 128)
    data = np.vstack((top(x), bottom(x)))

    # Create and use
    cmap = mplc.LinearSegmentedColormap.from_list('orangeblue', data)
    if apply:
        pl.set_cmap(cmap)
    return cmap


# Register colormaps
pl.cm.register_cmap('alpine',     alpinecolormap())
pl.cm.register_cmap('parula',     parulacolormap())
pl.cm.register_cmap('banded',     bandedcolormap())
pl.cm.register_cmap('bi',         bicolormap())
pl.cm.register_cmap('orangeblue', orangebluecolormap())
try:
    pl.cm.register_cmap('turbo',      turbocolormap())
except: # Included since Matplotlib 3.4.0
    pass

