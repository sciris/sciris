
"""
Extensions to Matplotlib, including 3D plotting and plot customization.

Highlights:
    - :func:`sc.plot3d() <plot3d>`: easy way to render 3D plots
    - :func:`sc.boxoff() <boxoff>`: turn off top and right parts of the axes box
    - :func:`sc.commaticks() <commaticks>`: convert labels from "10000" and "1e6" to "10,000" and "1,000,0000"
    - :func:`sc.SIticks() <SIticks>`: convert labels from "10000" and "1e6" to "10k" and "1m"
    - :func:`sc.maximize() <maximize>`: make the figure fill the whole screen
    - :func:`sc.savemovie() <savemovie>`: save a sequence of figures as an MP4 or other movie
    - :func:`sc.fonts() <fonts>`: list available fonts or add new ones
"""

##############################################################################
#%% Imports
##############################################################################

import os
import tempfile
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import sciris as sc


##############################################################################
#%% 3D plotting functions
##############################################################################

__all__ = ['fig3d', 'ax3d', 'plot3d', 'scatter3d', 'surf3d', 'bar3d']


def fig3d(num=None, nrows=1, ncols=1, index=1, returnax=False, figkwargs=None, axkwargs=None, **kwargs):
    """
    Shortcut for creating a figure with 3D axes.

    Usually not invoked directly; kwargs are passed to :func:`plt.figure() <matplotlib.pyplot.figure>`
    """
    figkwargs = sc.mergedicts(figkwargs, kwargs, num=num)
    axkwargs = sc.mergedicts(axkwargs)

    fig = plt.figure(**figkwargs)
    ax = ax3d(nrows=nrows, ncols=ncols, index=index, returnfig=False, figkwargs=figkwargs, **axkwargs)
    if returnax: # pragma: no cover
        return fig,ax
    else:
        return fig


def ax3d(nrows=None, ncols=None, index=None, fig=None, ax=None, returnfig=False,
         elev=None, azim=None, figkwargs=None, **kwargs):
    """
    Create a 3D axis to plot in.

    Usually not invoked directly; kwargs are passed to ``fig.add_subplot()``
    
    Args:
        nrows (int): number of rows of axes in plot
        ncols (int): number of columns of axes in plot
        index (int): index of current plot
        fig (Figure): if provided, use existing figure
        ax (Axes): if provided, validate and use these axes
        returnfig (bool): whether to return the figure (else just the axes)
        elev (float): the elevation of the 3D viewpoint
        azim (float): the azimuth of the 3D viewpoint
        figkwargs (dict): passed to :func:`plt.figure() <matplotlib.pyplot.figure>`
        kwargs (dict): passed to :func:`plt.axes() <matplotlib.pyplot.axes>`
        
    | *New in version 3.0.0:* nrows, ncols, and index arguments first
    | *New in version 3.1.0:* improved validation; 'silent' and 'axkwargs' argument removed
    """
    from mpl_toolkits.mplot3d import Axes3D

    figkwargs = sc.mergedicts(figkwargs)
    axkwargs = sc.mergedicts(dict(nrows=nrows, ncols=ncols, index=index), kwargs)
    nrows = axkwargs.pop('nrows', nrows) # Since fig.add_subplot() can't handle kwargs...
    ncols = axkwargs.pop('ncols', ncols)
    index = axkwargs.pop('index', index)
    
    # Handle the "111" format of subplots
    try:
        if ncols is None and index is None:
            nrows, ncols, index = map(int, str(nrows))
    except:
        pass # This is fine, just a different format

    # Handle the figure
    if not isinstance(fig, plt.Figure):
        if (fig in [True, False] or (fig is None and figkwargs)) and not ax: # Confusingly, any of these things indicate that we want a new figure
            fig = plt.figure(**figkwargs)
        else:
            if ax is None:
                if not plt.get_fignums():
                    fig = plt.figure(**figkwargs)
                else:
                    fig = plt.gcf()
            else:
                fig = ax.figure

    # Handle the figure
    if isinstance(fig, plt.Figure):
        pass
    elif ax is not None:
        fig = ax.figure
    elif fig == False:
        fig = plt.gcf()
    else:
        fig = plt.figure(**figkwargs)

    # Create and initialize the axis
    if ax is None:
        if fig.axes and index is None:
            ax = plt.gca()
        else:
            if nrows is None: nrows = 1
            if ncols is None: ncols = 1
            if index is None: index = 1
            ax = fig.add_subplot(nrows, ncols, index, projection='3d', **axkwargs)
    
    # Validate the axes
    if not isinstance(ax, Axes3D): # pragma: no cover
        errormsg = f'''Cannot create 3D plot into axes {ax}: ensure "projection='3d'" was used when making it'''
        raise ValueError(errormsg)
    
    # Change the view if requested
    if elev is not None or azim is not None: # pragma: no cover
        ax.view_init(elev=elev, azim=azim)
        
    # Tidy up
    if returnfig:
        return fig,ax
    else: # pragma: no cover
        return ax



def _process_2d_data(x, y, z, c, flatten=False):
    """ Helper function to handle data transformations -- not for the user """
    
    # Swap variables so z always exists
    if z is None and x is not None:
        z,x = x,z
    z = np.array(z)
    
    if z.ndim == 2:
        ny,nx = z.shape
        x = np.arange(nx) if x is None else np.array(x)
        y = np.arange(ny) if y is None else np.array(y)
        assert x.ndim == y.ndim, 'Cannot handle x and y axes with different array shapes'
        if x.ndim == 1 or y.ndim == 1:
            x,y = np.meshgrid(x, y)
        if flatten:
            x,y,z = x.flatten(), y.flatten(), z.flatten() # Flatten everything to 1D
            if sc.isarray(c) and c.shape == (ny,nx): # Flatten colors, too, if 2D and the same size as Z
                c = c.flatten()
    elif flatten == False: # pragma: no cover
        raise ValueError('Must provide z values as a 2D array')
    
    if x is None or y is None: # pragma: no cover
        raise ValueError('Must provide x and y values if z is 1D')
        
    # Handle automatic color scaling
    if isinstance(c, str):
        if c == 'z':
            c = z
        elif c == 'index':
            c = np.arange(len(z))
    
    return x, y, z, c


def _process_colors(c, z, cmap=None, to2d=False):
    """ Helper function to get color data in the right format -- not for the user """
    
    from . import sc_colors as scc # To avoid circular import
    
    # Handle colors
    if c.ndim == 1: # Used by scatter3d and bar3d
        assert len(c) == len(z), 'Number of colors does not match length of data'
        c = scc.vectocolor(c, cmap=cmap)
    elif c.ndim == 2: # Used by surf3d
        assert c.shape == z.shape, 'Shape of colors does not match shape of data'
        c = scc.arraycolors(c, cmap=cmap)
    
    # Used by bar3d -- flatten from 3D to 2D
    if to2d and c.ndim == 3:
        c = c.reshape((-1, c.shape[2]))
    
    return c
    


def plot3d(x, y, z, c='index', fig=True, ax=None, returnfig=False, figkwargs=None, 
           axkwargs=None, **kwargs):
    """
    Plot 3D data as a line

    Args:
        x (arr): x coordinate data
        y (arr): y coordinate data
        z (arr): z coordinate data
        c (str/tuple): color, can be an array or any of the types accepted by :func:`plt.plot() <matplotlib.pyplot.plot>`; if 'index' (default), color by index
        fig (fig): an existing figure to draw the plot in (or set to True to create a new figure)
        ax (axes): an existing axes to draw the plot in
        returnfig (bool): whether to return the figure, or just the axes
        figkwargs (dict): :func:`plt.figure() <matplotlib.pyplot.figure>`
        axkwargs (dict): :func:`plt.axes() <matplotlib.pyplot.axes>`
        kwargs (dict): passed to :func:`plt.plot() <matplotlib.pyplot.plot>`

    **Examples**::

        x,y,z = np.random.rand(3,10)
        sc.plot3d(x, y, z)
        
        fig = plt.figure()
        n = 100
        x = np.array(sorted(np.random.rand(n)))
        y = x + np.random.randn(n)
        z = np.random.randn(n)
        c = np.arange(n)
        sc.plot3d(x, y, z, c=c, fig=fig)
    
    *New in version 3.1.0:* Allow multi-colored line; removed "plotkwargs" argument; "fig" defaults to True
    """
    # Set default arguments
    plotkwargs = sc.mergedicts({'lw':2}, kwargs)
    axkwargs = sc.mergedicts(axkwargs)
    
    # Do input checking
    assert len(x) == len(y) == len(z), 'All inputs must have the same length'
    n = len(z)

    # Create axis
    fig,ax = ax3d(returnfig=True, fig=fig, ax=ax, figkwargs=figkwargs, **axkwargs)
    
    # Handle different-colored line segments
    if c == 'index':
        c = np.arange(n) # Assign automatically based on index
    if sc.isarray(c) and len(c) in [n, n-1]: # Technically don't use the last color if the color has the same length as the data # pragma: no cover
        if c.ndim == 1:
            c = _process_colors(c, z=z)
        for i in range(n-1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], c=c[i], **plotkwargs)
            
    # Standard case: single color
    else:
        ax.plot(x, y, z, c=c, **plotkwargs)

    if returnfig: # pragma: no cover
        return fig,ax
    else:
        return ax


def scatter3d(x=None, y=None, z=None, c='z', fig=True, ax=None, returnfig=False, 
              figkwargs=None, axkwargs=None, **kwargs):
    """
    Plot 3D data as a scatter
    
    Typically, ``x``, ``y``, and ``z``, are all vectors. However, if a single 2D
    array is provided, then this will be treated as ``z`` values and ``x`` and ``y``
    will be inferred on a grid (or they can be provided explicitly).

    Args:
        x (arr): 1D or 2D x coordinate data (or z-coordinate data if 2D and ``z`` is ``None``)
        y (arr): 1D or 2D y coordinate data
        z (arr): 1D or 2D z coordinate data
        c (arr): color data; defaults to match z; to use default colors, explicitly pass ``c=None``; to use index, use c='index'
        fig (fig): an existing figure to draw the plot in (or set to True to create a new figure)
        ax (axes): an existing axes to draw the plot in
        returnfig (bool): whether to return the figure, or just the axes
        figkwargs (dict): passed to :func:`plt.figure() <matplotlib.pyplot.figure>`
        axkwargs (dict): passed to :func:`plt.axes() <matplotlib.pyplot.axes>`
        kwargs (dict): passed to :func:`plt.scatter() <matplotlib.pyplot.scatter>`

    **Examples**::

        # Implicit coordinates, color by height (z-value)
        data = np.random.randn(10, 10)
        sc.scatter3d(data)
        
        # Explicit coordinates, color by index (i.e. ordering)
        x,y,z = np.random.rand(3,50)
        sc.scatter3d(x, y, z, c='index')
    
    | *New in version 3.0.0:* Allow 2D input
    | *New in version 3.1.0:* Allow "index" color argument; removed "plotkwargs" argument; "fig" defaults to True
    """
    # Set default arguments
    plotkwargs = sc.mergedicts({'s':200, 'depthshade':False, 'lw':0}, kwargs)
    axkwargs = sc.mergedicts(axkwargs)

    # Create figure
    fig,ax = ax3d(returnfig=True, fig=fig, ax=ax, figkwargs=figkwargs, **axkwargs)
    
    # Process data
    x, y, z, c = _process_2d_data(x, y, z, c, flatten=True)

    # Actually plot
    ax.scatter(x, y, z, c=c, **plotkwargs)

    if returnfig: # pragma: no cover
        return fig,ax
    else:
        return ax


def surf3d(x=None, y=None, z=None, c=None, fig=True, ax=None, returnfig=False, colorbar=None, 
           figkwargs=None, axkwargs=None, **kwargs):
    """
    Plot 2D or 3D data as a 3D surface
    
    Typically, ``x``, ``y``, and ``z``, are all 2D arrays of the same size. However, 
    if a single 2D array is provided, then this will be treated as ``z`` values and 
    ``x`` and ``y`` will be inferred on a grid (or they can be provided explicitly,
    either as vectors or 2D arrays).

    Args:
        x (arr): 1D or 2D array of x coordinates (or z-coordinate data if 2D and ``z`` is ``None``)
        y (arr): 1D or 2D array of y coordinates (optional)
        z (arr): 2D array of z coordinates
        c (arr): color data; defaults to match z
        fig (fig): an existing figure to draw the plot in (or set to True to create a new figure)
        ax (axes): an existing axes to draw the plot in
        returnfig (bool): whether to return the figure, or just the axes
        colorbar (bool): whether to plot a colorbar (true by default unless color data is provided)
        figkwargs (dict): passed to :func:`plt.figure() <matplotlib.pyplot.figure>`
        axkwargs (dict): passed to :func:`plt.axes() <matplotlib.pyplot.axes>`
        kwargs (dict): passed to :func:`ax.plot_surface() <mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface>`

    **Examples**::

        # Simple example
        data = sc.smooth(np.random.rand(30,50))
        sc.surf3d(data)
        
        # Use non-default axes and colors
        nx = 20
        ny = 50
        x = 10*np.arange(nx)
        y = np.arange(ny) + 100
        z = sc.smooth(np.random.randn(ny,nx))
        c = z**2
        sc.surf3d(x=x, y=y, z=z, c=c, cmap='orangeblue')
    
    *New in 3.1.0:* updated arguments from "data" to x, y, z, c; removed "plotkwargs" argument; "fig" defaults to True
    """

    # Set default arguments
    plotkwargs = sc.mergedicts({'cmap':plt.get_cmap()}, kwargs)
    axkwargs = sc.mergedicts(axkwargs)
    if colorbar is None:
        colorbar = False if sc.isarray(c) else True # Use a colorbar unless colors provided
    
    # Create figure
    fig,ax = ax3d(returnfig=True, fig=fig, ax=ax, figkwargs=figkwargs, **axkwargs)
    
    # Process data
    x, y, z, c = _process_2d_data(x, y, z, c, flatten=False)
    
    # Handle colors
    if sc.isarray(c):
        c = _process_colors(c, z=z, cmap=plotkwargs.get('cmap'))
        plotkwargs['facecolors'] = c
    
    # Actually plot
    surf = ax.plot_surface(x, y, z, **plotkwargs)
    if colorbar:
        fig.colorbar(surf)

    if returnfig: # pragma: no cover
        return fig,ax
    else:
        return ax



def bar3d(x=None, y=None, z=None, c='z', dx=0.8, dy=0.8, dz=None, fig=True, ax=None, 
          returnfig=False, figkwargs=None, axkwargs=None, **kwargs):
    """
    Plot 2D data as 3D bars

    Args:
        x (arr): 1D or 2D array of x coordinates (or z-coordinate data if 2D and ``z`` is ``None``)
        y (arr): 1D or 2D array of y coordinates (optional)
        z (arr): 2D array of z coordinates; interpreted as the heights of the bars unless ``dz`` is also provided
        c (arr): color data; defaults to match z
        dx (float/arr): width of the bars
        dy (float/arr): depth of the bars
        dz (float/arr): height of the bars, in which case ``z`` is interpreted as the base of the bars
        fig (fig): an existing figure to draw the plot in (or set to True to create a new figure)
        ax (axes): an existing axes to draw the plot in
        returnfig (bool): whether to return the figure, or just the axes
        colorbar (bool): whether to plot a colorbar (true by default unless color data is provided)
        figkwargs (dict): passed to :func:`plt.figure() <matplotlib.pyplot.figure>`
        axkwargs (dict): passed to :func:`plt.axes() <matplotlib.pyplot.axes>`
        kwargs (dict): passed to :func:`ax.bar3d() <mpl_toolkits.mplot3d.axes3d.Axes3D.bar3d>`

    **Examples**::

        # Simple example
        data = np.random.rand(5,4)
        sc.bar3d(data)
        
        # Use non-default axes and colors (note: this one is pretty!)
        nx = 5
        ny = 6
        x = 10*np.arange(nx)
        y = np.arange(ny) + 10
        z = -np.random.rand(ny,nx)
        dz = -2*z
        c = z**2
        sc.bar3d(x=x, y=y, z=z, dx=0.5, dy=0.5, dz=dz, c=c, cmap='orangeblue')
    
    *New in 3.1.0:* updated arguments from "data" to x, y, z, c; removed "plotkwargs" argument; "fig" defaults to True
    """

    # Set default arguments
    plotkwargs = sc.mergedicts(dict(shade=True), kwargs)
    axkwargs = sc.mergedicts(axkwargs)

    # Create figure
    fig,ax = ax3d(returnfig=True, fig=fig, ax=ax, figkwargs=figkwargs, **axkwargs)
    
    # Process data
    z_base = None # Assume no base is provided, and ...
    z_height = z # ... height was provided
    if z is not None and dz is not None: # Handle dz and z separately if both provided
        z_base = z.flatten() # In case z is provided as 2D
        z_height = dz
    elif z is None and dz is not None: # Swap order if dz is provided instead of z
        z_height = dz
        
    x, y, z_height, c = _process_2d_data(x=x, y=y, z=z_height, c=c, flatten=True)
    
    # Ensure the bottom of the bars is provided
    if z_base is None:
        z_base = np.zeros_like(z)
    
    # Process colors
    c = _process_colors(c, z_height, cmap=kwargs.get('cmap'), to2d=True)
    
    # Plot
    ax.bar3d(x=x, y=y, z=z_base, dx=dx, dy=dy, dz=z_height, color=c, **plotkwargs)

    if returnfig: # pragma: no cover
        return fig,ax
    else:
        return ax



##############################################################################
#%% Other plotting functions
##############################################################################

__all__ += ['stackedbar', 'boxoff', 'setaxislim', 'setxlim', 'setylim', 'commaticks', 'SIticks',
            'getrowscols', 'get_rows_cols', 'figlayout', 'maximize', 'fonts']


def stackedbar(x=None, values=None, colors=None, labels=None, transpose=False, 
               flipud=False, is_cum=False, barh=False, **kwargs):
    """
    Create a stacked bar chart.
    
    Args:
        x         (array)    : the x coordinates of the values
        values    (array)    : the 2D array of values to plot as stacked bars
        colors    (list/arr) : the color of each set of bars
        labels    (list)     : the label for each set of bars
        transpose (bool)     : whether to transpose the array prior to plotting
        flipud    (bool)     : whether to flip the array upside down prior to plotting
        is_cum    (bool)     : whether the array is already a cumulative sum
        barh      (bool)     : whether to plot as a horizontal instead of vertical bar
        kwargs    (dict)     : passed to :func:`plt.bar() <matplotlib.pyplot.bar>`
    
    **Example**::
        
        values = np.random.rand(3,5)
        sc.stackedbar(values, labels=['bottom','middle','top'])
        plt.legend()
    
    *New in version 2.0.4.*
    """
    from . import sc_colors as scc # To avoid circular import
    
    # Handle inputs
    if x is not None and values is None:
        values = x
        x = None
    
    if values is None: # pragma: no cover
        errormsg = 'Must supply values to plot, typically as a 2D array'
        raise ValueError(errormsg)
    
    values = sc.toarray(values)
    if values.ndim == 1: # pragma: no cover
        values = values[None,:] # Convert to a 2D array
    
    if transpose: # pragma: no cover
        values = values.T
    
    if flipud: # pragma: no cover
        values = values[::-1,:]
    
    nstack = values.shape[0]
    npts = values.shape[1]
    
    if x is None:
        x = np.arange(npts)
        
    if is_cum:
        values = np.diff(values, prepend=0, axis=0)

    # Handle labels and colors
    if labels is not None: # pragma: no cover
        nlabels = len(labels)
        if nlabels != nstack:
            errormsg = f'Expected {nstack} labels, got {nlabels}'
            raise ValueError(errormsg)
    
    if colors is not None: # pragma: no cover
        ncolors = len(colors)
        if ncolors != nstack:
            errormsg = f'Expected {nstack} colors, got {ncolors}'
            raise ValueError(errormsg)
    else:
        colors = scc.gridcolors(nstack)
        
    # Actually plot
    artists = []
    for i in range(nstack):
        if labels is not None:
            label = labels[i]
        else: # pragma: no cover
            label = None
        
        h = values[i,:]
        b = values[:i,:].sum(axis=0)
        kw = dict(facecolor=colors[i], label=label, **kwargs)
        if not barh:
            artist = plt.bar(x=x, height=h, bottom=b, **kw)
        else:
            artist = plt.barh(y=x, width=h, left=b, **kw)
        artists.append(artist)
    
    return artists


def boxoff(ax=None, which=None, removeticks=True):
    """
    Removes the top and right borders ("spines") of a plot.

    Also optionally removes the tick marks, and flips the remaining ones outside.
    Can be used as an alias to ``plt.axis('off')`` if ``which='all'``.

    Args:
        ax (Axes): the axes to remove the spines from (if None, use current)
        which (str/list): a list or comma-separated string of spines: 'top', 'bottom', 'left', 'right', or 'all' (default top & right)
        removeticks (bool): whether to also remove the ticks from these spines
        flipticks (bool): whether to flip remaining ticks out

    **Examples**::

        plt.figure()
        plt.plot([2,5,3])
        sc.boxoff()

        fig, ax = plt.subplots()
        plt.plot([1,4,1,4])
        sc.boxoff(ax=ax, which='all')

        fig = plt.figure()
        plt.scatter(np.arange(100), np.random.rand(100))
        sc.boxoff('top, bottom')

    *New in version 1.3.3:* ability to turn off multiple spines; removed "flipticks" arguments
    """
    # Handle axes
    if isinstance(ax, (str, list)): # Swap input arguments # pragma: no cover
        ax,which = which,ax
    if ax is None: ax = plt.gca()

    # Handle which
    if not isinstance(which, list):
        if which is None:
            which = 'top, right'
        if which == 'all': # pragma: no cover
            which = 'top, bottom, right, left'
        if isinstance(which, str):
            which = which.split(',')
            which = [w.rstrip().lstrip() for w in which]

    for spine in which: # E.g. ['top', 'right']
        ax.spines[spine].set_visible(False)
        if removeticks:
            ax.tick_params(**{spine:False, f'label{spine}':False})

    return ax



def setaxislim(which=None, ax=None, data=None):
    """
    A small script to determine how the y limits should be set. Looks
    at all data (a list of arrays) and computes the lower limit to
    use, e.g.::

        sc.setaxislim([np.array([-3,4]), np.array([6,4,6])], ax)

    will keep Matplotlib's lower limit, since at least one data value
    is below 0.

    Note, if you just want to set the lower limit, you can do that
    with this function via::

        sc.setaxislim()
    """

    # Handle which axis
    if which is None: # pragma: no cover
        which = 'both'
    if which not in ['x','y','both']: # pragma: no cover
        errormsg = f'Setting axis limit for axis {which} is not supported'
        raise ValueError(errormsg)
    if which == 'both': # pragma: no cover
        setaxislim(which='x', ax=ax, data=data)
        setaxislim(which='y', ax=ax, data=data)
        return

    # Ensure axis exists
    if ax is None:
        ax = plt.gca()

    # Get current limits
    if   which == 'x': currlower, currupper = ax.get_xlim()
    elif which == 'y': currlower, currupper = ax.get_ylim()

    # Calculate the lower limit based on all the data
    lowerlim = 0
    upperlim = 0
    if sc.checktype(data, 'arraylike'): # Ensure it's numeric data (probably just None) # pragma: no cover
        flatdata = sc.toarray(data).flatten() # Make sure it's iterable
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
    """ Alias for :func:`sc.setaxislim(which='x') <setaxislim>` """
    return setaxislim(data=data, ax=ax, which='x')


def setylim(data=None, ax=None):
    """
    Alias for :func:`sc.setaxislim(which='y') <setaxislim>`.

    **Example**::

        plt.plot([124,146,127])
        sc.setylim() # Equivalent to plt.ylim(bottom=0)
    """
    return setaxislim(data=data, ax=ax, which='y')


def _get_axlist(ax): # pragma: no cover
    """ Helper function to turn either a figure, an axes, or a list of axes into a list of axes """

    if ax is None: # If not supplied, get current axes
        axlist = [plt.gca()]
    elif isinstance(ax, plt.Axes): # If it's an axes, turn to a list
        axlist = [ax]
    elif isinstance(ax, plt.Figure): # If it's a figure, pull all axes
        axlist = ax.axes
    elif isinstance(ax, list): # If it's a list, use directly
        axlist = ax
    else:
        errormsg = f'Could not recognize object {type(ax)}: must be None, Axes, Figure, or list of axes'
        raise ValueError(errormsg)

    return axlist


def commaticks(ax=None, axis='y', precision=2, cursor_precision=0):
    """
    Use commas in formatting the y axis of a figure (e.g., 34,000 instead of 34000).

    To use something other than a comma, set the default separator via e.g. :class:`sc.options(sep='.') <sciris.sc_settings.ScirisOptions>`.

    Args:
        ax (any): axes to modify; if None, use current; else can be a single axes object, a figure, or a list of axes
        axis (str/list): which axis to change (default 'y'; can accept a list)
        precision (int): shift how many decimal places to show for small numbers (+ve = more, -ve = fewer)
        cursor_precision (int): ditto, for cursor

    **Example**::

        data = np.random.rand(10)*1e4
        plt.plot(data)
        sc.commaticks()

    See http://stackoverflow.com/questions/25973581/how-to-format-axis-number-format-to-thousands-with-a-comma-in-matplotlib

    | *New in version 1.3.0:* ability to use non-comma thousands separator
    | *New in version 1.3.1:* added "precision" argument
    | *New in version 2.0.0:* ability to set x and y axes simultaneously
    """
    def commaformatter(x, pos=None): # pragma: no cover
        interval = thisaxis.get_view_interval()
        prec = precision + cursor_precision if pos is None else precision # Use higher precision for cursor
        decimals = int(max(0, prec-np.floor(np.log10(np.ptp(interval)))))
        string = f'{x:0,.{decimals}f}' # Do the formatting
        if pos is not None and '.' in string: # Remove trailing decimal zeros from axis labels
            string = string.rstrip('0')
            if string[-1] == '.': # If we trimmed 0.0 to 0., trim the remaining period
                string = string[:-1]
        if sep != ',': # Use custom separator if desired
            string = string.replace(',', sep)
        return string

    sep = sc.options.sep
    axlist = _get_axlist(ax)
    axislist = sc.tolist(axis)
    for ax in axlist:
        for axis in axislist:
            if   axis=='x': thisaxis = ax.xaxis
            elif axis=='y': thisaxis = ax.yaxis
            elif axis=='z': thisaxis = ax.zaxis # pragma: no cover
            else: raise ValueError('Axis must be x, y, or z') # pragma: no cover
            thisaxis.set_major_formatter(mpl.ticker.FuncFormatter(commaformatter))
    return



def SIticks(ax=None, axis='y', fixed=False):
    """
    Apply SI tick formatting to one axis of a figure  (e.g., 34k instead of 34000)

    Args:
        ax (any): axes to modify; if None, use current; else can be a single axes object, a figure, or a list of axes
        axis (str): which axes to change (default 'y')
        fixed (bool): use fixed-location tick labels (by default, update them dynamically)

    **Example**::

        data = np.random.rand(10)*1e4
        plt.plot(data)
        sc.SIticks()
    """
    def SItickformatter(x, pos=None, sigfigs=2, SI=True, *args, **kwargs):  # formatter function takes tick label and tick position # pragma: no cover
        """ Formats axis ticks so that e.g. 34000 becomes 34k -- usually not invoked directly """
        output = sc.sigfig(x, sigfigs=sigfigs, SI=SI) # Pretty simple since sc.sigfig() does all the work
        return output

    axlist = _get_axlist(ax)
    for ax in axlist:
        if   axis=='x': thisaxis = ax.xaxis
        elif axis=='y': thisaxis = ax.yaxis
        elif axis=='z': thisaxis = ax.zaxis # pragma: no cover
        else: raise ValueError('Axis must be x, y, or z') # pragma: no cover
        if fixed: # pragma: no cover
            ticklocs = thisaxis.get_ticklocs()
            ticklabels = []
            for tickloc in ticklocs:
                ticklabels.append(SItickformatter(tickloc))
            thisaxis.set_major_formatter(mpl.ticker.FixedFormatter(ticklabels))
        else:
            thisaxis.set_major_formatter(mpl.ticker.FuncFormatter(SItickformatter))
    return


def getrowscols(n, nrows=None, ncols=None, ratio=1, make=False, tight=True, remove_extra=True, **kwargs):
    """
    Get the number of rows and columns needed to plot N figures.

    If you have 37 plots, then how many rows and columns of axes do you know? This
    function convert a number (i.e. of plots) to a number of required rows and columns.
    If nrows or ncols is provided, the other will be calculated. Ties are broken
    in favor of more rows (i.e. 7x6 is preferred to 6x7). It can also generate
    the plots, if ``make=True``.

    Note: :func:`sc.getrowscols() <getrowscols>` and :func:`sc.get_rows_cols() <get_rows_cols>` are aliases.

    Args:
        n (int): the number (of plots) to accommodate
        nrows (int): if supplied, keep this fixed and calculate the columns
        ncols (int): if supplied, keep this fixed and calculate the rows
        ratio (float): sets the number of rows relative to the number of columns (i.e. for 100 plots, 1 will give 10x10, 4 will give 20x5, etc.).
        make (bool): if True, generate subplots
        tight (bool): if True and make is True, then apply tight layout
        remove_extra (bool): if True and make is True, then remove extra subplots
        kwargs (dict): passed to plt.subplots()

    Returns:
        A tuple of ints for the number of rows and the number of columns (which, of course, you can reverse)

    **Examples**::

        nrows,ncols = sc.get_rows_cols(36) # Returns 6,6
        nrows,ncols = sc.get_rows_cols(37) # Returns 7,6
        nrows,ncols = sc.get_rows_cols(100, ratio=2) # Returns 15,7
        nrows,ncols = sc.get_rows_cols(100, ratio=0.5) # Returns 8,13 since rows are prioritized
        fig,axs     = sc.getrowscols(37, make=True) # Create 7x6 subplots, using the alias

    | *New in version 1.0.0.*
    | *New in version 1.2.0:* "make", "tight", and "remove_extra" arguments
    | *New in version 1.3.0:* alias without underscores
    """

    # Simple cases -- calculate the one missing
    if nrows is not None: # pragma: no cover
        ncols = int(np.ceil(n/nrows))
    elif ncols is not None: # pragma: no cover
        nrows = int(np.ceil(n/ncols))

    # Standard case -- calculate both
    else:
        guess = np.sqrt(n)
        nrows = int(np.ceil(guess*np.sqrt(ratio)))
        ncols = int(np.ceil(n/nrows)) # Could also call recursively!
    
    # If asked, make subplots
    if make: # pragma: no cover
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
        if remove_extra:
            flat = axs.flat[n:] if isinstance(axs, np.ndarray) else []
            for ax in flat:
                ax.set_visible(False) # to remove last plot
        if tight:
            figlayout(fig, tight=True)
        return fig,axs
    else: # Otherwise, just return rows and columns
        return nrows,ncols

get_rows_cols = getrowscols  # Alias


def figlayout(fig=None, tight=True, keep=None, **kwargs):
    """
    Alias to both :meth:`fig.set_layout_engine() <matplotlib.figure.Figure.set_layout_engine>`
    and :meth:`fig.subplots_adjust() <matplotlib.figure.Figure.subplots_adjust>`.

    Args:
        fig (Figure): the figure (by default, use current)
        tight (bool): passed to :meth:`fig.set_layout_engine() <matplotlib.figure.Figure.set_layout_engine>`; default True
        keep (bool): if True, then leave tight layout on; else, turn it back off to allow additional layout updates (which requires a render, so can be slow)
        kwargs (dict): passed to :meth:`fig.subplots_adjust() <matplotlib.figure.Figure.subplots_adjust>`

    **Example**::

        fig,axs = sc.get_rows_cols(37, make=True, tight=False) # Create 7x6 subplots, squished together
        sc.figlayout(bottom=0.3)

    | *New in version 1.2.0.*
    | *New in version 3.1.1:* ``keep`` defaults to ``True`` to avoid the need to refresh
    """
    if isinstance(fig, bool): # pragma: no cover
        fig = None
        tight = fig # To allow e.g. sc.figlayout(False)
    if fig is None:
        fig = plt.gcf()
    if keep is None:
        keep = False if len(kwargs) else True
    layout = ['none', 'tight'][tight]
    try: # Matplotlib >=3.6
        fig.set_layout_engine(layout)
    except: # Earlier versions # pragma: no cover
        fig.set_tight_layout(tight)
    if not keep: # pragma: no cover
        if not plt.get_backend() == 'agg':
            plt.pause(0.01) # Force refresh if using an interactive backend
        try:
            fig.set_layout_engine('none')
        except: # pragma: no cover
            fig.set_tight_layout(False)
    if len(kwargs): # pragma: no cover
        fig.subplots_adjust(**kwargs)
    return


def maximize(fig=None, die=False):  # pragma: no cover
    """
    Maximize the current (or supplied) figure. Note: not guaranteed to work for
    all Matplotlib backends (e.g., agg).

    Args:
        fig (Figure): the figure object; if not supplied, use the current active figure
        die (bool): whether to propagate an exception if encountered (default no)

    **Example**::

        plt.plot([2,3,5])
        sc.maximize()

    *New in version 1.0.0.*
    """
    backend = plt.get_backend().lower()
    if fig is not None:
        plt.figure(fig.number) # Set the current figure
    try:
        mgr = plt.get_current_fig_manager()
        if   'qt'  in backend: mgr.window.showMaximized()
        elif 'gtk' in backend: mgr.window.maximize()
        elif 'wx'  in backend: mgr.frame.Maximize(True)
        elif 'tk'  in backend: mgr.resize(*mgr.window.maxsize())
        else:
            errormsg = f'The maximize() function is not supported for the backend "{backend}"; use Qt5Agg if possible'
            raise NotImplementedError(errormsg)
    except Exception as E:
        errormsg = f'Warning: maximizing the figure failed because: "{str(E)}"'
        if die:
            raise RuntimeError(errormsg) from E
        else:
            print(errormsg)
    return


def fonts(add=None, use=False, output='name', dryrun=False, rebuild=False, verbose=False, die=False, **kwargs):
    """
    List available fonts, or add new ones. Alias to Matplotlib's font manager.

    Note: if the font is not available after adding it, set rebuild=True. However,
    note that this can be very slow.

    Args:
        add (str/list): path of the fonts or folders to add; if none, list available fonts
        use (bool): set the last-added font as the default font
        output (str): what to display the listed fonts as: options are 'name' (list of names, default), 'path' (dict of name:path), or 'font' (dict of name:font object)
        dryrun (bool): list fonts to be added rather than adding them
        rebuild (bool): whether to rebuild Matplotlib's font cache (slow)
        verbose (bool): print out information on errors
        die (bool): whether to raise an exception if fonts can't be added
        kwargs (dict): passed to :func:`matplotlib.font_manager.findSystemFonts()`

    **Examples**::

        sc.fonts() # List available font names
        sc.fonts(fullfont=True) # List available font objects
        sc.fonts('myfont.ttf', use=True) # Add this font and immediately set to default
        sc.fonts(['/folder1', '/folder2']) # Add all fonts in both folders
        sc.fonts(rebuild=True) # Run this if added fonts aren't appearing
    """
    fm = mpl.font_manager # Shorten

    # List available fonts
    if add is None and not rebuild:

        # Find fonts
        f = sc.objdict() # Create a dictionary for holding the results
        keys = ['names', 'paths', 'objs']
        for key in keys:
            f[key] = sc.autolist()
        for fontpath in fm.findSystemFonts(**kwargs):
            try:
                fontobj = fm.get_font(fontpath)
                fontname = fontobj.family_name
                if fontname not in f.names: # Don't allow duplicates
                    f.names += fontname
                    f.paths += fontpath
                    f.objs  += fontobj
            except Exception as E:
                if verbose: # pragma: no cover
                    print(f'Could not load {fontpath}: {str(E)}')

        # Handle output
        order = np.argsort(f.names) # Order by name
        for key in keys:
            f[key] = [f[key][o] for o in order]
        if 'name' in output:
            out = f.names
        elif 'path' in output: # pragma: no cover
            out = dict(zip(f.names, f.paths))
        elif 'font' in output: # pragma: no cover
            out = dict(zip(f.names, f.objs))
        else: # pragma: no cover
            errormsg = f'Output type not recognized: must be "name", "path", or "font", not "{output}"'
            raise ValueError(errormsg)
        return out

    # Or, add new fonts
    else:

        # Try, but by default don't crash if they can't be added
        try:
            fontname = None
            fontpaths = []
            paths = sc.tolist(add)
            for path in paths:
                path = str(path)
                if os.path.isdir(path): # pragma: no cover
                    fps = fm.findSystemFonts(path, **kwargs)
                    fontpaths.extend(fps)
                else:
                    fontpaths.append(sc.makefilepath(path))

            if dryrun: # pragma: no cover
                print(fontpaths)
            else:
                for path in fontpaths:
                    fm.fontManager.addfont(path)
                    fontname = fm.get_font(path).family_name
                    if verbose:
                        print(f'Added "{fontname}"')
                if verbose and fontname is None: # pragma: no cover
                    print('Warning: no fonts were added')
                if use and fontname: # Set as default font
                    plt.rc('font', family=fontname)

            if rebuild: # pragma: no cover
                print('Rebuilding font cache, please be patient...')
                try:
                    fm._load_fontmanager(try_read_cache=False) # This used to be fm._rebuild(), but this was removed; this works as of Matplotlib 3.4.3
                    print(f'Font cache rebuilt in folder: {mpl.get_cachedir()}')
                except Exception as E:
                    exc = type(E)
                    errormsg = f'Rebuilding font cache failed:\n{str(E)}'
                    raise exc(errormsg) from E

            if verbose:
                print(f'Done: added {len(fontpaths)} fonts.')

        # Exception was encountered, quietly print
        except Exception as E: # pragma: no cover
            exc = type(E)
            errormsg = f'Warning, could not install some fonts:\n{str(E)}'
            if die:
                raise exc(errormsg) from E
            else:
                print(errormsg)

    return


##############################################################################
#%% Date plotting
##############################################################################

__all__ += ['ScirisDateFormatter', 'dateformatter', 'datenumformatter']


class ScirisDateFormatter(mpl.dates.ConciseDateFormatter):
    """
    An adaptation of Matplotlib's ConciseDateFormatter with a slightly different
    approach to formatting dates. Specifically:

        - Years are shown below dates, rather than on the RHS
        - The day and month are always shown.
        - The cursor shows only the date, not the time

    This formatter is not intended to be called directly -- use :func:`sc.dateformatter() <dateformatter>`
    instead. It is also optimized for plotting dates, rather than times -- for those,
    ConciseDateFormatter is better.

    See :func:`sc.dateformatter() <dateformatter>` for explanation of arguments.

    *New in version 1.3.0.*
    """

    def __init__(self, locator, formats=None, zero_formats=None, show_offset=False, show_year=True, **kwargs):

        # Handle inputs
        self.show_year = show_year
        if formats is None:
            formats = [
                '%Y',    # ticks are mostly years
                '%b',    # ticks are mostly months
                '%b-%d', # ticks are mostly days
                '%H:%M', # hrs
                '%H:%M', # min
                '%S.%f', # secs
                ]
        if zero_formats is None:
            zero_formats = [
                '%Y',    # ticks are mostly years -- no zeros
                '%b',    # ticks are mostly months
                '%b-%d', # ticks are mostly days
                '%b-%d', # hrs
                '%H:%M', # min
                '%H:%M', # secs
                ]

        # Initialize the ConciseDateFormatter with the corrected input
        super().__init__(locator, formats=formats, zero_formats=zero_formats, show_offset=show_offset, **kwargs)

        return

    def format_data_short(self, value): # pragma: no cover
        """
        Show year-month-day, not with hours and seconds
        """
        return mpl.dates.num2date(value, tz=self._tz).strftime('%Y-%b-%d')

    def format_ticks(self, values): # pragma: no cover
        """
        Append the year to the tick label for the first label, or if the year changes.
        This avoids the need to use offset_text, which is difficult to control.
        """

        def addyear(label, year):
            """ Add the year to the label if it's not already present """
            yearstr = str(year)
            if yearstr not in label: # Be agnostic about where in the label the year string might be present
                label += f'\n{yearstr}'
            return label

        # Get the default labels and years
        labels = super().format_ticks(values)
        years = [mpl.dates.num2date(v).year for v in values]

        # Add year information to any labels that require it
        if self.show_year:
            for i,label in enumerate(labels):
                year = years[i]
                if i == 0 or (year != years[i-1]):
                    labels[i] = addyear(label, year)

        return labels


def dateformatter(ax=None, style='sciris', dateformat=None, start=None, end=None,
                  rotation=None, locator=None, axis='x', **kwargs):
    """
    Format the x-axis to use a given date formatter.

    By default, this will apply the Sciris date formatter to the current x-axis.
    This formatter is a combination of Matplotlib's Concise date formatter, and
    Plotly's date formatter.

    See also :func:`sc.datenumformatter() <datenumformatter>` to convert a numeric axis to date labels.

    Args:
        ax         (axes)     : if supplied, use these axes instead of the current one
        style      (str)      : the style to use if the axis already uses dates; options are "sciris", "auto", "concise", or a Formatter object
        dateformat (str)      : the date format (default ``'%Y-%b-%d'``; not needed if x-axis already uses dates)
        start      (str/int)  : if supplied, the lower limit of the axis
        end        (str/int)  : if supplied, the upper limit of the axis
        rotation   (float)    : rotation of the labels, in degrees
        locator    (Locator)  : if supplied, use this instead of the default ``AutoDateLocator`` locator
        axis       (str)      : which axis to apply to the formatter to (default 'x')
        kwargs     (dict)     : passed to the date formatter (e.g., :class:`ScirisDateFormatter`)

    **Examples**::

        # Reformat date data
        plt.figure()
        x = sc.daterange('2021-04-04', '2022-05-05', asdate=True)
        y = sc.smooth(np.random.rand(len(x)))
        plt.plot(x, y)
        sc.dateformatter()

        # Configure with Matplotlib's Concise formatter
        fig,ax = plt.subplots()
        plt.plot(sc.date(np.arange(365), start_date='2022-01-01'), np.random.randn(365))
        sc.dateformatter(ax=ax, style='concise')

    | *New in version 1.2.0.*
    | *New in version 1.2.2:* "rotation" argument; renamed "start_day" to "start_date"
    | *New in version 1.3.0:* refactored to use built-in Matplotlib date formatting
    | *New in version 1.3.2:* "axis" argument
    | *New in version 1.3.3:* split ``sc.dateformatter()`` from ``sc.datenumformatter()``
    """

    # Handle deprecation
    style = kwargs.pop('dateformatter', style) # Allow this as an alias

    # Handle axis
    if isinstance(ax, str): # Swap style and axes -- allows sc.dateformatter(ax) or sc.dateformatter('auto') # pragma: no cover
        style = ax
        ax = None
    if ax is None:
        ax = plt.gca()

    # Handle dateformat, if provided
    if dateformat is not None: # pragma: no cover
        if isinstance(dateformat, str):
            kwargs['formats'] = [dateformat]*6
        elif isinstance(dateformat, list):
            kwargs['formats'] = dateformat
        else:
            errormsg = f'Could not recognize date format {type(dateformat)}: expecting string or list'
            raise ValueError(errormsg)
        kwargs['zero_formats'] = kwargs['formats']

    # Handle locator and styles
    if locator is None:
        locator = mpl.dates.AutoDateLocator(minticks=3)
    style = str(style).lower()
    if style in ['none', 'sciris', 'house', 'default']:
        formatter = ScirisDateFormatter(locator, **kwargs)
    elif style in ['auto', 'matplotlib']:
        formatter = mpl.dates.AutoDateFormatter(locator, **kwargs)
    elif style in ['concise', 'brief']:
        formatter = mpl.dates.ConciseDateFormatter(locator, **kwargs)
    elif isinstance(style, mpl.ticker.Formatter): # If a formatter is provided, use directly # pragma: no cover
        formatter = style
    else: # pragma: no cover
        errormsg = f'Style "{style}" not recognized; must be one of "sciris", "auto", or "concise"'
        raise ValueError(errormsg)

    # Handle axis and set the locator and formatter
    if axis == 'x':
        axis = ax.xaxis
    elif axis == 'y': # If it's not x or y (!), assume it's an axis object # pragma: no cover
        axis = ax.yaxis
    axis.set_major_locator(locator)
    axis.set_major_formatter(formatter)

    # Handle limits
    xmin, xmax = ax.get_xlim()
    if start: xmin = sc.date(start)
    if end:   xmax = sc.date(end)
    ax.set_xlim((xmin, xmax))

    # Set the rotation
    if rotation:
        ax.tick_params(axis='x', labelrotation=rotation)

    # Set the formatter
    ax.xaxis.set_major_formatter(formatter)

    return formatter


def datenumformatter(ax=None, start_date=None, dateformat=None, interval=None, start=None,
                     end=None, rotation=None):
    """
    Format a numeric x-axis to use dates.

    See also :func:`sc.dateformatter() <dateformatter>`, which is intended for use when the axis already
    has date data.

    Args:
        ax         (axes)     : if supplied, use these axes instead of the current one
        start_date (str/date) : the start day, either as a string or date object (not needed if x-axis already uses dates)
        dateformat (str)      : the date format (default ``'%Y-%b-%d'``; not needed if x-axis already uses dates)
        interval   (int)      : if supplied, the interval between ticks (not needed if x-axis already uses dates)
        start      (str/int)  : if supplied, the lower limit of the axis
        end        (str/int)  : if supplied, the upper limit of the axis
        rotation   (float)    : rotation of the labels, in degrees

    **Examples**::

        # Automatically configure a non-date axis with default options
        plt.plot(np.arange(365), np.random.rand(365))
        sc.datenumformatter(start_date='2021-01-01')

        # Manually configure
        fig,ax = plt.subplots()
        ax.plot(np.arange(60), np.random.random(60))
        formatter = sc.datenumformatter(start_date='2020-04-04', interval=7, start='2020-05-01', end=50, dateformat='%m-%d', ax=ax)

    | *New in version 1.2.0.*
    | *New in version 1.2.2:* "rotation" argument; renamed "start_day" to "start_date"
    | *New in version 1.3.3:* renamed from ``sc.dateformatter()`` to  ``sc.datenumformatter()``
    """

    # Handle axis
    if isinstance(ax, str): # Swap inputs # pragma: no cover
        ax, start_date = start_date, ax
    if ax is None:
        ax = plt.gca()

    # Set the default format -- "2021-01-01"
    if dateformat is None:
        dateformat = '%Y-%b-%d'

    # Convert to a date object
    if start_date is None: # pragma: no cover
        start_date = plt.num2date(ax.dataLim.x0)
    start_date = sc.date(start_date)

    @mpl.ticker.FuncFormatter
    def formatter(x, pos): # pragma: no cover
        return (start_date + dt.timedelta(days=int(x))).strftime(dateformat)

    # Handle limits
    xmin, xmax = ax.get_xlim()
    if start: xmin = sc.day(start, start_date=start_date)
    if end:   xmax = sc.day(end,   start_date=start_date)
    ax.set_xlim((xmin, xmax))

    # Set the x-axis intervals
    if interval: # pragma: no cover
        ax.set_xticks(np.arange(xmin, xmax+1, interval))

    # Set the rotation
    if rotation: # pragma: no cover
        ax.tick_params(axis='x', labelrotation=rotation)

    # Set the formatter
    ax.xaxis.set_major_formatter(formatter)

    return formatter


##############################################################################
#%% Figure saving
##############################################################################

__all__ += ['savefig', 'savefigs', 'loadfig', 'emptyfig', 'separatelegend', 'orderlegend']



def _get_dpi(dpi=None, min_dpi=200):
    """ Helper function to choose DPI for saving figures """
    if dpi is None:
        mpl_dpi = plt.rcParams['savefig.dpi']
        if mpl_dpi == 'figure':
            mpl_dpi = plt.rcParams['figure.dpi']
        dpi = max(mpl_dpi, min_dpi) # Don't downgrade DPI
    return dpi


def savefig(filename, fig=None, dpi=None, comments=None, pipfreeze=False, relframe=0, 
            folder=None, makedirs=True, die=True, verbose=True, **kwargs):
    """
    Save a figure, including metadata

    Wrapper for Matplotlib's :func:`plt.savefig() <matplotlib.pyplot.savefig>` function which automatically stores
    metadata in the figure. By default, it saves (git) information from the calling
    function. Additional comments can be added to the saved file as well. These
    can be retrieved via :func:`sc.loadmetadata() <sciris.sc_versioning.loadmetadata>`.

    Metadata can be stored and retrieved for PNG or SVG. Metadata
    can be stored for PDF, but cannot be automatically retrieved.

    Args:
        filename  (str/Path) : name of the file to save to
        fig       (Figure)   : the figure to save (if None, use current)
        dpi       (int)      : resolution of the figure to save (default 200 or current default, whichever is higher)
        comments  (str)      : additional metadata to save to the figure
        pipfreeze (bool)     : whether to store the contents of ``pip freeze`` in the metadata
        relframe  (int)      : which calling file to try to store information from (default 0, the file calling :func:`sc.savefig() <savefig>`)
        folder    (str/Path) : optional folder to save to (can also be provided as part of the filename)
        makedirs  (bool)     : whether to create folders if they don't already exist
        die       (bool)     : whether to raise an exception if metadata can't be saved
        verbose   (bool)     : if die is False, print a warning if metadata can't be saved
        kwargs    (dict)     : passed to ``fig.save()``

    **Examples**::

        plt.plot([1,3,7])

        sc.savefig('example1.png')
        print(sc.loadmetadata('example1.png'))

        sc.savefig('example2.png', comments='My figure', freeze=True)
        sc.pp(sc.loadmetadata('example2.png'))
    
    | *New in version 1.3.3.*
    | *New in version 3.0.0:* "freeze" renamed "pipfreeze"; "frame" replaced with "relframe"; replaced metadata with ``sc.metadata()``
    """
    # Handle deprecation
    orig_metadata = kwargs.pop('metadata', {}) # In case metadata is supplied, as it can be for fig.save()
    pipfreeze     = kwargs.pop('freeze', pipfreeze) 
    frame         = kwargs.pop('frame', None)
    if frame is not None: # pragma: no cover
        relframe = frame - 2
    
    # Handle figure
    if fig is None:
        fig = plt.gcf()

    # Handle DPI
    dpi = _get_dpi(dpi)

    # Get caller and git info
    jsonstr = sc.metadata(relframe=relframe+1, pipfreeze=pipfreeze, comments=comments, tostring=True, **orig_metadata)

    # Handle different formats
    filename = str(filename)
    lcfn = filename.lower() # Lowercase filename
    metadataflag = sc.sc_versioning._metadataflag
    if lcfn.endswith('png'):
        metadata = {metadataflag:jsonstr}
    elif lcfn.endswith('svg') or lcfn.endswith('pdf'): # pragma: no cover
        metadata = dict(Keywords=f'{metadataflag}={jsonstr}')
    else:
        errormsg = f'Warning: filename "{filename}" has unsupported type for metadata: must be PNG, SVG, or PDF. For JPG, use the separate exif library. To silence this message, set die=False and verbose=False.'
        if die:
            raise ValueError(errormsg)
        else:
            metadata = None
            if verbose:
                print(errormsg)

    # Save the figure
    if metadata is not None:
        kwargs['metadata'] = metadata # To avoid warnings for unsupported filenames

    # Allow savefig to make directories
    filepath = sc.makefilepath(filename=filename, folder=folder, makedirs=makedirs)
    fig.savefig(filepath, dpi=dpi, **kwargs)
    return filename


def savefigs(figs=None, filetype=None, filename=None, folder=None, savefigargs=None, aslist=False, verbose=False, **kwargs):
    """
    Save the requested plots to disk.

    Args:
        figs        (list) : the figure objects to save
        filetype    (str)  : the file type; can be 'fig', 'singlepdf' (default), or anything supported by savefig()
        filename    (str)  : the file to save to (only uses path if multiple files)
        folder      (str)  : the folder to save the file(s) in
        savefigargs (dict) : arguments passed to savefig()
        aslist      (bool) : whether or not return a list even for a single file
        varbose     (bool) : whether to print progress

    **Examples**::

        import matplotlib.pyplot as plt
        import sciris as sc
        fig1 = plt.figure(); plt.plot(np.random.rand(10))
        fig2 = plt.figure(); plt.plot(np.random.rand(10))
        sc.savefigs([fig1, fig2]) # Save everything to one PDF file
        sc.savefigs(fig2, 'png', filename='myfig.png', savefigargs={'dpi':200})
        sc.savefigs([fig1, fig2], filepath='/home/me', filetype='svg')
        sc.savefigs(fig1, position=[0.3,0.3,0.5,0.5])

    If saved as 'fig', then can load and display the plot using sc.loadfig().

    Version: 2018aug26
    """

    # Preliminaries
    wasinteractive = plt.isinteractive() # You might think you can get rid of this...you can't!
    if wasinteractive: plt.ioff()
    if filetype is None: filetype = 'singlepdf' # This ensures that only one file is created

    # Either take supplied plots, or generate them
    figs = sc.odict.promote(figs)
    nfigs = len(figs)

    # Handle file types
    filenames = []
    if filetype=='singlepdf': # See http://matplotlib.org/examples/pylab_examples/multipage_pdf.html  # pragma: no cover
        from matplotlib.backends.backend_pdf import PdfPages
        defaultname = 'figures.pdf'
        fullpath = sc.makefilepath(filename=filename, folder=folder, default=defaultname, ext='pdf', makedirs=True)
        pdf = PdfPages(fullpath)
        filenames.append(fullpath)
        if verbose: print(f'PDF saved to {fullpath}')
    for p,item in enumerate(figs.items()):
        key,plot = item
        # Handle filename
        if filename and nfigs==1: # Single plot, filename supplied -- use it
            fullpath = sc.makefilepath(filename=filename, folder=folder, default='Figure', ext=filetype, makedirs=True) # NB, this filename not used for singlepdf filetype, so it's OK
        else: # Any other case, generate a filename # pragma: no cover
            keyforfilename = filter(str.isalnum, str(key)) # Strip out non-alphanumeric stuff for key
            defaultname = keyforfilename
            fullpath = sc.makefilepath(filename=filename, folder=folder, default=defaultname, ext=filetype, makedirs=True)

        # Do the saving
        if savefigargs is None: savefigargs = {}
        defaultsavefigargs = {'dpi':200, 'bbox_inches':'tight'} # Specify a higher default DPI and save the figure tightly
        defaultsavefigargs.update(savefigargs) # Update the default arguments with the user-supplied arguments
        if filetype == 'fig':
            sc.save(fullpath, plot)
            filenames.append(fullpath)
            if verbose: print(f'Figure object saved to {fullpath}')
        else: # pragma: no cover
            reanimateplots(plot)
            if filetype=='singlepdf':
                pdf.savefig(figure=plot, **defaultsavefigargs) # It's confusing, but defaultsavefigargs is correct, since we updated it from the user version
            else:
                plt.savefig(fullpath, **defaultsavefigargs)
                filenames.append(fullpath)
                if verbose: print(f'{filetype.upper()} plot saved to {fullpath}')
            plt.close(plot)

    # Do final tidying
    if filetype=='singlepdf': pdf.close()
    if wasinteractive: plt.ion()
    if aslist or len(filenames)>1: # pragma: no cover
        return filenames
    else:
        return filenames[0]


def loadfig(filename=None):
    """
    Load a plot from a file and reanimate it.

    **Example usage**::

        import matplotlib.pyplot as plt
        import sciris as sc
        fig = plt.figure(); plt.plot(np.random.rand(10))
        sc.savefigs(fig, filetype='fig', filename='example.fig')

    **Later**::

        example = sc.loadfig('example.fig')
    """
    plt.ion() # Without this, it doesn't show up
    try:
        fig = sc.loadobj(filename)
    except Exception as E: # pragma: no cover
        errormsg = f'Unable to open file "{filename}": are you sure it was saved as a .fig file (not an image)?'
        raise type(E)(errormsg) from E
    
    reanimateplots(fig)
    
    return fig


def reanimateplots(plots=None):
    """ Reconnect plots (actually figures) to the Matplotlib backend. Plots must be an odict of figure objects. """
    try:
        from matplotlib.backends.backend_agg import new_figure_manager_given_figure as nfmgf # Warning -- assumes user has agg on their system, but should be ok. Use agg since doesn't require an X server
    except Exception as E: # pragma: no cover
        errormsg = f'To reanimate plots requires the "agg" backend, which could not be imported: {repr(E)}'
        raise ImportError(errormsg) from E
    
    if len(plt.get_fignums()):
        fignum = plt.gcf().number # This is the number of the current active figure, if it exists
    else: # pragma: no cover
        fignum = 1
        
    plots = sc.mergelists(plots) # Convert to an odict
    for plot in plots:
        nfmgf(fignum, plot) # Make sure each figure object is associated with the figure manager -- WARNING, is it correct to associate the plot with an existing figure?
    
    return


def emptyfig(*args, **kwargs):
    """ The emptiest figure possible """
    fig = plt.Figure(facecolor='None', *args, **kwargs)
    return fig


def _get_legend_handles(ax, handles, labels):
    """
    Construct handle and label list, from one of:

         - A list of handles and a list of labels
         - A list of handles, where each handle contains the label
         - An axis object, containing the objects that should appear in the legend
         - A figure object, from which the first axis will be used
    """
    if handles is None:
        if ax is None:
            ax = plt.gca()
        elif isinstance(ax, plt.Figure): # Allows an argument of a figure instead of an axes # pragma: no cover
            ax = ax.axes[-1]
        handles, labels = ax.get_legend_handles_labels()
    else: # pragma: no cover
        if labels is None:
            labels = [h.get_label() for h in handles]
        else:
            assert len(handles) == len(labels), f"Number of handles ({len(handles)}) and labels ({len(labels)}) must match"
    return ax, handles, labels


def separatelegend(ax=None, handles=None, labels=None, reverse=False, figsettings=None, legendsettings=None):
    """ Allows the legend of a figure to be rendered in a separate window instead """

    # Handle settings
    f_settings = sc.mergedicts({'figsize':(4.0,4.8)}, figsettings) # (6.4,4.8) is the default, so make it a bit narrower
    l_settings = sc.mergedicts({'loc': 'center', 'bbox_to_anchor': None, 'frameon': False}, legendsettings)

    # Get handles and labels
    _, handles, labels = _get_legend_handles(ax, handles, labels)

    # Set up new plot
    fig = plt.figure(**f_settings)
    ax = fig.add_subplot(111)
    ax.set_position([-0.05,-0.05,1.1,1.1]) # This cuts off the axis labels, ha-ha
    ax.set_axis_off()  # Hide axis lines

    # A legend renders the line/patch based on the object handle. However, an object
    # can only appear in one figure. Thus, if the legend is in a different figure, the
    # object cannot be shown in both the original figure and in the legend. Thus we need
    # to copy the handles, and use the copies to render the legend
    handles2 = []
    for h in handles:
        h2 = sc.cp(h)
        h2.axes = None
        h2.figure = None
        handles2.append(h2)

    # Reverse order, e.g. for stacked plots
    if reverse: # pragma: no cover
        handles2 = handles2[::-1]
        labels   = labels[::-1]

    # Plot the new legend
    ax.legend(handles=handles2, labels=labels, **l_settings)

    return fig


def orderlegend(order=None, ax=None, handles=None, labels=None, reverse=None, **kwargs):
    """
    Create a legend with a specified order, or change the order of an existing legend.
    Can either specify an order, or use the reverse argument to simply reverse the order.
    Note: you do not need to create the legend before calling this function; if you do,
    you will need to pass any additional keyword arguments to this function since it will
    override existing settings.

    Args:
        order (list or array): the new order of the legend, as from e.g. np.argsort()
        ax (axes): the axes object; if omitted, defaults to current axes
        handles (list): the legend handles; can be used instead of ax
        labels (list): the legend labels; can be used instead of ax
        reverse (bool): if supplied, simply reverse the legend order
        kwargs (dict): passed to ax.legend()

    **Examples**::

        plt.plot([1,4,3], label='A')
        plt.plot([5,7,8], label='B')
        plt.plot([2,5,2], label='C')
        sc.orderlegend(reverse=True) # Legend order C, B, A
        sc.orderlegend([1,0,2], frameon=False) # Legend order B, A, C with no frame
        plt.legend() # Restore original legend order A, B, C
    """

    # Get handles and labels
    ax, handles, labels = _get_legend_handles(ax, handles, labels)
    if order:
        handles = [handles[o] for o in order]
        labels = [labels[o] for o in order]
    if reverse:
        handles = handles[::-1]
        labels = labels[::-1]

    ax.legend(handles, labels, **kwargs)

    return


#%% Animation

__all__ += ['animation', 'savemovie']


class animation(sc.prettyobj):
    """
    A class for storing and saving a Matplotlib animation.

    See also :func:`sc.savemovie() <savemovie>`, which works directly with Matplotlib artists rather
    than an entire figure. Depending on your use case, one is likely easier to use
    than the other. Use :func:`sc.animation() <animation>` if you want to animate a complex figure
    including non-artist objects (e.g., titles and legends); use :func:`sc.savemovie() <savemovie>`
    if you just want to animate a set of artists (e.g., lines).

    This class works by saving snapshots of the figure to disk as image files, then
    reloading them either via ``ffmpeg`` or as a Matplotlib animation. While (slightly)
    slower than working with artists directly, it means that anything that can be
    rendered to a figure can be animated.

    Note: the terms "animation" and "movie" are used interchangeably here.

    Args:
        fig          (fig):  the Matplotlib figure to animate (if none, use current)
        filename     (str):  the name of the output animation (default: animation.mp4)
        dpi          (int):  the resolution to save the animation at
        fps          (int):  frames per second for the animation
        imageformat  (str):  file type for temporary image files, e.g. 'jpg'
        basename     (str):  name for temporary image files, e.g. 'myanimation'
        nametemplate (str):  as an alternative to imageformat and basename, specify the full name template, e.g. 'myanimation%004d.jpg'
        imagefolder  (str):  location to store temporary image files; default current folder, or use 'tempfile' to create a temporary folder
        anim_args    (dict): passed to :obj:`matplotlib.animation.ArtistAnimation` or ``ffmpeg.input()``
        save_args    (dict): passed to :meth:`animation.save() <matplotlib.animation.Animation.save>` or ``ffmpeg.run()``
        tidy         (bool): whether to delete temporary files
        verbose      (bool): whether to print progress
        kwargs       (dict): also passed to :meth:`animation.save() <matplotlib.animation.Animation.save>`

    **Example**::

        anim = sc.animation()

        plt.figure()
        repeats = 21
        colors = sc.vectocolor(repeats, cmap='turbo')
        for i in range(repeats):
            scale = 1/np.sqrt(i+1)
            x = scale*np.random.randn(10)
            y = scale*np.random.randn(10)
            label = str(i) if not(i%5) else None
            plt.scatter(x, y, c=[colors[i]], label=label)
            plt.title(f'Scale = 1/√{i}')
            plt.legend()
            sc.boxoff('all')
            anim.addframe()

        anim.save('dots.mp4')

    | *New in version 1.3.3.*
    | *New in version 2.0.0:* ``ffmpeg`` option.
    """
    def __init__(self, fig=None, filename=None, dpi=200, fps=10, imageformat='png', basename='animation', nametemplate=None,
                 imagefolder=None, anim_args=None, save_args=None, frames=None, tidy=True, verbose=True, **kwargs):
        self.fig          = fig
        self.filename     = filename
        self.dpi          = dpi
        self.fps          = fps
        self.imageformat  = imageformat
        self.basename     = basename
        self.nametemplate = nametemplate
        self.imagefolder  = imagefolder
        self.anim_args    = anim_args
        self.save_args    = sc.mergedicts(save_args, kwargs)
        self.tidy         = tidy
        self.verbose      = verbose
        self.filenames    = sc.autolist()
        self.frames       = frames if frames else []
        self.fig_size     = None
        self.fig_dpi      = None
        self.anim         = None
        self.initialize()
        return


    def initialize(self):
        """ Handle additional initialization of variables """

        # Handle folder
        if self.imagefolder == 'tempfile': # pragma: no cover
            self.imagefolder = tempfile.gettempdir()
        if self.imagefolder is None:
            self.imagefolder = os.getcwd()
        self.imagefolder = sc.path(self.imagefolder)

        # Handle name template
        if self.nametemplate is None:
            self.nametemplate = f'{self.basename}_%04d.{self.imageformat}' # ADD name template

        # Handle dpi
        self.dpi = _get_dpi(self.dpi)

        return


    def _getfig(self, fig=None):
        """ Get the Matplotlib figure to save the animation from """
        if fig is None:
            if self.fig is not None:
                fig = self.fig
            else:
                try:    fig = self.frames[0][0].get_figure()
                except: fig = plt.gcf()
        return fig


    def _getfilename(self, path=True):
        """ Generate a filename for the next image file to save """
        try:
            name = self.nametemplate % self.n_files
        except TypeError as E: # pragma: no cover
            errormsg = f'Name template "{self.nametemplate}" does not seem valid for inserting current file number {self.n_files} into: should contain the string "%04d" or similar'
            raise TypeError(errormsg) from E
        if path:
            name = self.imagefolder / name
        return name


    def __add__(self, *args, **kwargs): # pragma: no cover
        """ Allow anim += fig """
        self.addframe(*args, **kwargs)
        return self

    def __radd__(self, *args, **kwargs): # pragma: no cover
        """ Allow anim += fig """
        self.addframe(self, *args, **kwargs)
        return self

    @property
    def n_files(self):
        return len(self.filenames)

    @property
    def n_frames(self):
        return len(self.frames)

    def __len__(self): # pragma: no cover
        """ Since we can have either files or frames, need to check both  """
        return max(self.n_files, self.n_frames)


    def addframe(self, fig=None, *args, **kwargs):
        """ Add a frame to the animation -- typically a figure object, but can also be an artist or list of artists """

        # If a figure is supplied but it's not a figure, add it to the frames directly
        if fig is not None and isinstance(fig, (list, mpl.artist.Artist)): # pragma: no cover
            self.frames.append(fig)

        # Typical case: add a figure
        else:
            if self.verbose and self.n_files == 0: # First frame
                print('Adding frames...')

            # Get the figure, name, and save
            fig = self._getfig(fig)
            filename = self._getfilename()
            fig.savefig(filename, dpi=self.dpi)
            self.filenames += filename

            # Check figure properties
            fig_size = fig.get_size_inches()
            fig_dpi = fig.get_dpi()

            if self.fig_size is None:
                self.fig_size = fig_size
            else:
                if not np.allclose(self.fig_size, fig_size): # pragma: no cover
                    warnmsg = f'Note: current figure size {fig_size} does not match saved {self.fig_size}, unexpected results may occur!'
                    print(warnmsg)

            if self.fig_dpi is None:
                self.fig_dpi = fig_dpi
            else:
                if self.fig_dpi != fig_dpi: # pragma: no cover
                    warnmsg = f'Note: current figure DPI {fig_dpi} does not match saved {self.fig_dpi}, unexpected results may occur!'
                    print(warnmsg)

            if self.verbose:
                print(f'  Added frame {self.n_files}: {self._getfilename(path=False)}')

        return


    def loadframes(self): # pragma: no cover
        """ Load saved images as artists """
        animfig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        ax = animfig.add_axes([0,0,1,1])
        if self.verbose:
            print('Preprocessing frames...')
        for f,filename in enumerate(self.filenames):
            if self.verbose:
                sc.progressbar(f+1, self.filenames)
            im = plt.imread(filename)
            self.frames.append(ax.imshow(im))
        plt.close(animfig)
        return


    def __enter__(self, *args, **kwargs): # pragma: no cover
        """ To allow with...as """
        return self


    def __exit__(self, *args, **kwargs): # pragma: no cover
        """ Save on exist from a with...as block """
        return self.save()


    def rmfiles(self):
        """ Remove temporary image files """
        succeeded = 0
        failed = sc.autolist()
        for filename in self.filenames:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    succeeded += 1
                except Exception as E: # pragma: no cover
                    failed += f'{filename} ({E})'
        if self.verbose:
            if succeeded:
                print(f'Removed {succeeded} temporary files')
        if failed: # pragma: no cover
            print(f'Failed to remove the following temporary files:\n{sc.newlinejoin(failed)}')


    def save(self, filename=None, fps=None, dpi=None, engine='ffmpeg', anim_args=None,
             save_args=None, frames=None, tidy=None, verbose=True, **kwargs):
        """ Save the animation -- arguments the same as :func:`sc.animation() <animation>` and :func:`sc.savemovie() <savemovie>`, and are described there """

        # Handle engine
        if engine == 'ffmpeg':
            try:
                import ffmpeg
            except:
                print('Warning: engine ffmpeg not available; falling back to Matplotlib. Run "pip install ffmpeg-python" to use in future.')
                engine = 'matplotlib'
        engines = ['ffmpeg', 'matplotlib']
        if engine not in engines: # pragma: no cover
            errormsg = f'Could not understand engine "{engine}": must be one of {sc.strjoin(engines)}'
            raise ValueError(errormsg)

        # Handle dictionary args
        anim_args = sc.mergedicts(self.anim_args, anim_args)
        save_args = sc.mergedicts(self.save_args, save_args)

        # Handle filename
        if filename is None:
            if self.filename is None: # pragma: no cover
                self.filename = f'{self.basename}.mp4'
            filename = self.filename

        # Set parameters
        if fps  is None: fps  = save_args.pop('fps', self.fps)
        if dpi  is None: dpi  = save_args.pop('dpi', self.dpi)
        if tidy is None: tidy = self.tidy

        # Start timing
        T = sc.timer()

        if engine == 'ffmpeg': # pragma: no cover
            save_args = sc.mergedicts(dict(overwrite_output=True, quiet=True), save_args)
            stream = ffmpeg.input(self.nametemplate, framerate=fps, **anim_args)
            stream = stream.output(filename)
            stream.run(**save_args, **kwargs)

        elif engine == 'matplotlib': # pragma: no cover
            import matplotlib.animation as mpl_anim

            # Load and sanitize frames
            if frames is None:
                if not self.n_frames:
                    self.loadframes()
                if self.n_files and (self.n_frames != self.n_files):
                    errormsg = f'Number of files ({self.n_files}) does not match number of frames ({self.n_frames}): please do not mix and match adding figures and adding artists as frames!'
                    raise RuntimeError(errormsg)
                frames = self.frames

            for f in range(len(frames)):
                if not sc.isiterable(frames[f]):
                    frames[f] = (frames[f],) # This must be either a tuple or a list to work with ArtistAnimation

            # Try to get the figure from the frames, else use the current one
            fig = self._getfig()

            # Optionally print progress
            if verbose:
                print(f'Saving {len(frames)} frames at {fps} fps and {dpi} dpi to "{filename}"...')
                callback = lambda i,n: sc.progressbar(i+1, len(frames)) # Default callback
                callback = save_args.pop('progress_callback', callback) # if provided as an argument
            else:
                callback = None

            # Actually create the animation -- warning, no way to not actually have it render!
            anim = mpl_anim.ArtistAnimation(fig, frames, **anim_args)
            anim.save(filename, fps=fps, dpi=dpi, progress_callback=callback, **save_args, **kwargs)

        if tidy:
            self.rmfiles()

        if verbose:
            print(f'Done; movie saved to "{filename}"')
            try: # Not essential, so don't try too hard if this doesn't work
                filesize = os.path.getsize(filename)
                if filesize<1e6: print(f'File size: {filesize/1e3:0.0f} KB')
                else:            print(f'File size: {filesize/1e6:0.1f} MB') # pragma: no cover
            except: # pragma: no cover
                pass
            T.toc(label='Time saving movie')

        return


def savemovie(frames, filename=None, fps=None, quality=None, dpi=None, writer=None, bitrate=None,
              interval=None, repeat=False, repeat_delay=None, blit=False, verbose=True, **kwargs):
    """
    Save a set of Matplotlib artists as a movie.

    Note: in most cases, it is preferable to use :func:`sc.animation() <animation>`.

    Args:
        frames (list): The list of frames to animate
        filename (str): The name (or full path) of the file; expected to end with mp4 or gif (default movie.mp4)
        fps (int): The number of frames per second (default 10)
        quality (string): The quality of the movie, in terms of dpi (default "high" = 300 dpi)
        dpi (int): Instead of using quality, set an exact dpi
        writer (str or object): Specify the writer to be passed to :meth:`animation.save() <matplotlib.animation.Animation.save>` (default "ffmpeg")
        bitrate (int): The bitrate. Note, may be ignored; best to specify in a writer and to pass in the writer as an argument
        interval (int): The interval between frames; alternative to using fps
        repeat (bool): Whether or not to loop the animation (default False)
        repeat_delay (bool): Delay between repeats, if repeat=True (default None)
        blit (bool): Whether or not to "blit" the frames (default False, since otherwise does not detect changes )
        verbose (bool): Whether to print statistics on finishing.
        kwargs (dict): Passed to :meth:`animation.save() <matplotlib.animation.Animation.save>`

    Returns:
        A Matplotlib animation object

    **Examples**::

        import matplotlib.pyplot as plt
        import sciris as sc

        # Simple example (takes ~5 s)
        plt.figure()
        frames = [pl.plot(np.cumsum(np.random.randn(100))) for i in range(20)] # Create frames
        sc.savemovie(frames, 'dancing_lines.gif') # Save movie as medium-quality gif

        # Complicated example (takes ~15 s)
        plt.figure()
        nframes = 100 # Set the number of frames
        ndots = 100 # Set the number of dots
        axislim = 5*pl.sqrt(nframes) # Pick axis limits
        dots = plt.zeros((ndots, 2)) # Initialize the dots
        frames = [] # Initialize the frames
        old_dots = sc.dcp(dots) # Copy the dots we just made
        fig = plt.figure(figsize=(10,8)) # Create a new figure
        for i in range(nframes): # Loop over the frames
            dots += np.random.randn(ndots, 2) # Move the dots randomly
            color = plt.norm(dots, axis=1) # Set the dot color
            old = plt.array(old_dots) # Turn into an array
            plot1 = plt.scatter(old[:,0], old[:,1], c='k') # Plot old dots in black
            plot2 = plt.scatter(dots[:,0], dots[:,1], c=color) # Note: Frames will be separate in the animation
            plt.xlim((-axislim, axislim)) # Set x-axis limits
            plt.ylim((-axislim, axislim)) # Set y-axis limits
            kwargs = {'transform':pl.gca().transAxes, 'horizontalalignment':'center'} # Set the "title" properties
            title = plt.text(0.5, 1.05, f'Iteration {i+1}/{nframes}', **kwargs) # Unfortunately plt.title() can't be dynamically updated
            plt.xlabel('Latitude') # But static labels are fine
            plt.ylabel('Longitude') # Ditto
            frames.append((plot1, plot2, title)) # Store updated artists
            old_dots = plt.vstack([old_dots, dots]) # Store the new dots as old dots
        sc.savemovie(frames, 'fleeing_dots.mp4', fps=20, quality='high') # Save movie as a high-quality mp4

    Version: 2019aug21
    """
    from matplotlib import animation as mpl_anim # Place here since specific only to this function

    if not isinstance(frames, list): # pragma: no cover
        errormsg = f'sc.savemovie(): argument "frames" must be a list, not "{type(frames)}"'
        raise TypeError(errormsg)
    for f in range(len(frames)):
        if not sc.isiterable(frames[f]): # pragma: no cover
            frames[f] = (frames[f],) # This must be either a tuple or a list to work with ArtistAnimation

    # Try to get the figure from the frames, else use the current one
    try:    fig = frames[0][0].get_figure()
    except: fig = plt.gcf() # pragma: no cover

    # Set parameters
    if filename is None: # pragma: no cover
        filename = 'movie.mp4'
    if writer is None:
        if   filename.endswith('mp4'): writer = 'ffmpeg'
        elif filename.endswith('gif'): writer = 'imagemagick'
        else: # pragma: no cover
            errormsg = f'sc.savemovie(): unknown movie extension for file {filename}'
            raise ValueError(errormsg)
    if fps is None:
        fps = 10
    if interval is None:
        interval = 1000./fps
        fps = 1000./interval # To ensure it's correct

    # Handle dpi/quality
    if dpi is None and quality is None:
        quality = 'medium' # Make it medium quailty by default
    if isinstance(dpi, str): # pragma: no cover
        quality = dpi # Interpret dpi arg as a quality command
        dpi = None
    if dpi is not None and quality is not None: # pragma: no cover
        print(f'sc.savemovie() warning: quality is simply a shortcut for dpi; please specify one or the other, not both (dpi={dpi}, quality={quality})')
    if quality is not None:
        if   quality == 'low':    dpi =  50
        elif quality == 'medium': dpi = 150
        elif quality == 'high':   dpi = 300 # pragma: no cover
        else: # pragma: no cover
            errormsg = f'Quality must be high, medium, or low, not "{quality}"'
            raise ValueError(errormsg)

    # Optionally print progress
    if verbose:
        start = sc.tic()
        print(f'Saving {len(frames)} frames at {fps} fps and {dpi} dpi to "{filename}" using {writer}...')

    # Actually create the animation -- warning, no way to not actually have it render!
    anim = mpl_anim.ArtistAnimation(fig, frames, interval=interval, repeat_delay=repeat_delay, repeat=repeat, blit=blit)
    anim.save(filename, writer=writer, fps=fps, dpi=dpi, bitrate=bitrate, **kwargs)

    if verbose:
        print(f'Done; movie saved to "{filename}"')
        try: # Not essential, so don't try too hard if this doesn't work
            filesize = os.path.getsize(filename)
            if filesize<1e6: print(f'File size: {filesize/1e3:0.0f} KB')
            else:            print(f'File size: {filesize/1e6:0.2f} MB') # pragma: no cover
        except: # pragma: no cover
            pass
        sc.toc(start)

    return anim