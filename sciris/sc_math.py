'''
Extensions to Numpy, including finding array elements and smoothing data.

Highlights:
    - ``sc.findinds()``: find indices of an array matching a condition
    - ``sc.findnearest()``: find nearest matching value
    - ``sc.rolling()``: calculate rolling average
    - ``sc.smooth()``: simple smoothing of 1D or 2D arrays
'''

import numpy as np
import warnings
from . import sc_utils as scu


##############################################################################
#%% Find and approximation functions
##############################################################################

__all__ = ['approx', 'safedivide', 'findinds', 'findfirst', 'findlast', 'findnearest',
           'dataindex', 'getvalidinds', 'sanitize', 'getvaliddata', 'isprime']


def approx(val1=None, val2=None, eps=None, **kwargs):
    '''
    Determine whether two scalars (or an array and a scalar) approximately match.
    Alias for np.isclose() and may be removed in future versions.

    Args:
        val1 (number or array): the first value
        val2 (number): the second value
        eps (float): absolute tolerance
        kwargs (dict): passed to np.isclose()

    **Examples**::

        sc.approx(2*6, 11.9999999, eps=1e-6) # Returns True
        sc.approx([3,12,11.9], 12) # Returns array([False, True, False], dtype=bool)
    '''
    if eps is not None:
        kwargs['atol'] = eps # Rename kwarg to match np.isclose()
    output = np.isclose(a=val1, b=val2, **kwargs)
    return output


def safedivide(numerator=None, denominator=None, default=None, eps=None, warn=False):
    '''
    Handle divide-by-zero and divide-by-nan elegantly.

    **Examples**::

        sc.safedivide(numerator=0, denominator=0, default=1, eps=0) # Returns 1
        sc.safedivide(numerator=5, denominator=2.0, default=1, eps=1e-3) # Returns 2.5
        sc.safedivide(3, np.array([1,3,0]), -1, warn=True) # Returns array([ 3,  1, -1])
    '''
    # Set some defaults
    if numerator   is None: numerator   = 1.0
    if denominator is None: denominator = 1.0
    if default     is None: default     = 0.0

    # Handle types
    if isinstance(numerator,   list): numerator   = np.array(numerator)
    if isinstance(denominator, list): denominator = np.array(denominator)

    # Handle the logic
    invalid = approx(denominator, 0.0, eps=eps)
    if scu.isnumber(denominator): # The denominator is a scalar
        if invalid:
            output = default
        else:
            output = numerator/denominator
    elif scu.checktype(denominator, 'array'):
        if not warn:
            denominator[invalid] = 1.0 # Replace invalid values with 1
        output = numerator/denominator
        output[invalid] = default
    else: # pragma: no cover # Unclear input, raise exception
        errormsg = f'Input type {type(denominator)} not understood: must be number or array'
        raise TypeError(errormsg)

    return output


def findinds(arr=None, val=None, eps=1e-6, first=False, last=False, die=True, **kwargs):
    '''
    Little function to find matches even if two things aren't eactly equal (eg.
    due to floats vs. ints). If one argument, find nonzero values. With two arguments,
    check for equality using eps. Returns a tuple of arrays if val1 is multidimensional,
    else returns an array. Similar to calling np.nonzero(np.isclose(arr, val))[0].

    Args:
        arr (array): the array to find values in
        val (float): if provided, the value to match
        eps (float): the precision for matching (default 1e-6, equivalent to np.isclose's atol)
        first (bool): whether to return the first matching value
        last (bool): whether to return the last matching value
        die (bool): whether to raise an exception if first or last is true and no matches were found
        kwargs (dict): passed to np.isclose()

    **Examples**::

        sc.findinds(rand(10)<0.5) # returns e.g. array([2, 4, 5, 9])
        sc.findinds([2,3,6,3], 3) # returs array([1,3])
        sc.findinds([2,3,6,3], 3, first=True) # returns 1

    New in version 1.2.3: "die" argument
    '''

    # Handle first or last
    if first and last: raise ValueError('Can use first or last but not both')
    elif first: ind = 0
    elif last:  ind = -1
    else:       ind = None

    # Handle kwargs
    atol = kwargs.pop('atol', eps) # Ensure atol isn't specified twice
    if 'val1' in kwargs or 'val2' in kwargs: # pragma: no cover
        arr = kwargs.pop('val1', arr)
        val = kwargs.pop('val2', val)
        warnmsg = 'sc.findinds() arguments "val1" and "val2" have been deprecated as of v1.0.0; use "arr" and "val" instead'
        warnings.warn(warnmsg, category=DeprecationWarning, stacklevel=2)

    # Calculate matches
    arr = scu.promotetoarray(arr)
    if val is None: # Check for equality
        output = np.nonzero(arr) # If not, just check the truth condition
    else:
        if scu.isstring(val):
            output = np.nonzero(arr==val)
        try: # Standard usage, use nonzero
            output = np.nonzero(np.isclose(a=arr, b=val, atol=atol, **kwargs)) # If absolute difference between the two values is less than a certain amount
        except Exception as E: # pragma: no cover # As a fallback, try simpler comparison
            output = np.nonzero(abs(arr-val) < atol)
            if kwargs: # Raise a warning if and only if special settings were passed
                warnmsg = f'{str(E)}\nsc.findinds(): np.isclose() encountered an exception (above), falling back to direct comparison'
                warnings.warn(warnmsg, category=RuntimeWarning, stacklevel=2)

    # Process output
    try:
        if arr.ndim == 1: # Uni-dimensional
            output = output[0] # Return an array rather than a tuple of arrays if one-dimensional
            if ind is not None:
                output = output[ind] # And get the first element
        else:
            if ind is not None:
                output = [output[i][ind] for i in range(arr.ndim)]
    except IndexError as E:
        if die:
            errormsg = 'No matching values found; use die=False to return None instead of raising an exception'
            raise IndexError(errormsg) from E
        else:
            output = None

    return output


def findfirst(*args, **kwargs):
    ''' Alias for findinds(..., first=True). New in version 1.0.0. '''
    return findinds(*args, **kwargs, first=True)


def findlast(*args, **kwargs):
    ''' Alias for findinds(..., last=True). New in version 1.0.0. '''
    return findinds(*args, **kwargs, last=True)


def findnearest(series=None, value=None):
    '''
    Return the index of the nearest match in series to value -- like findinds, but
    always returns an object with the same type as value (i.e. findnearest with
    a number returns a number, findnearest with an array returns an array).

    **Examples**::

        findnearest(rand(10), 0.5) # returns whichever index is closest to 0.5
        findnearest([2,3,6,3], 6) # returns 2
        findnearest([2,3,6,3], 6) # returns 2
        findnearest([0,2,4,6,8,10], [3, 4, 5]) # returns array([1, 2, 2])

    Version: 2017jan07
    '''
    series = scu.promotetoarray(series)
    if scu.isnumber(value):
        output = np.argmin(abs(series-value))
    else:
        output = []
        for val in value: output.append(findnearest(series, val))
        output = scu.promotetoarray(output)
    return output


def dataindex(dataarray, index): # pragma: no cover
    '''
    Take an array of data and return either the first or last (or some other) non-NaN entry.

    This function is deprecated.
    '''

    nrows = np.shape(dataarray)[0] # See how many rows need to be filled (either npops, nprogs, or 1).
    output = np.zeros(nrows)       # Create structure
    for r in range(nrows):
        output[r] = sanitize(dataarray[r])[index] # Return the specified index -- usually either the first [0] or last [-1]

    return output


def getvalidinds(data=None, filterdata=None): # pragma: no cover
    '''
    Return the indices that are valid based on the validity of the input data from an arbitrary number
    of 1-D vector inputs. Warning, closely related to getvaliddata().

    This function is deprecated.

    **Example**::

        getvalidinds([3,5,8,13], [2000, nan, nan, 2004]) # Returns array([0,3])
    '''
    data = scu.promotetoarray(data)
    if filterdata is None: filterdata = data # So it can work on a single input -- more or less replicates sanitize() then
    filterdata = scu.promotetoarray(filterdata)
    if filterdata.dtype=='bool': filterindices = filterdata # It's already boolean, so leave it as is
    else:                        filterindices = findinds(~np.isnan(filterdata)) # Else, assume it's nans that need to be removed
    dataindices = findinds(~np.isnan(data)) # Also check validity of data
    validindices = np.intersect1d(dataindices, filterindices)
    return validindices # Only return indices -- WARNING, not consistent with sanitize()


def getvaliddata(data=None, filterdata=None, defaultind=0): # pragma: no cover
    '''
    Return the data value indices that are valid based on the validity of the input data.

    This function is deprecated.

    **Example**::

        getvaliddata(array([3,5,8,13]), array([2000, nan, nan, 2004])) # Returns array([3,13])
    '''
    data = np.array(data)
    if filterdata is None: filterdata = data # So it can work on a single input -- more or less replicates sanitize() then
    filterdata = np.array(filterdata)
    if filterdata.dtype=='bool': validindices = filterdata # It's already boolean, so leave it as is
    else:                        validindices = ~np.isnan(filterdata) # Else, assume it's nans that need to be removed
    if validindices.any(): # There's at least one data point entered
        if len(data)==len(validindices): # They're the same length: use for logical indexing
            validdata = np.array(np.array(data)[validindices]) # Store each year
        elif len(validindices)==1: # They're different lengths and it has length 1: it's an assumption
            validdata = np.array([np.array(data)[defaultind]]) # Use the default index; usually either 0 (start) or -1 (end)
        else:
            errormsg = f'Array sizes are mismatched: {len(data)} vs. {len(validindices)}'
            raise ValueError(errormsg)
    else:
        validdata = np.array([]) # No valid data, return an empty array
    return validdata


def sanitize(data=None, returninds=False, replacenans=None, die=True, defaultval=None, label=None, verbose=True):
        '''
        Sanitize input to remove NaNs. Warning, does not work on multidimensional data!!

        **Examples**::

            sanitized,inds = sanitize(array([3,4,nan,8,2,nan,nan,nan,8]), returninds=True)
            sanitized = sanitize(array([3,4,nan,8,2,nan,nan,nan,8]), replacenans=True)
            sanitized = sanitize(array([3,4,nan,8,2,nan,nan,nan,8]), replacenans=0)
        '''
        try:
            data = np.array(data,dtype=float) # Make sure it's an array of float type
            inds = np.nonzero(~np.isnan(data))[0] # WARNING, nonzero returns tuple :(
            sanitized = data[inds] # Trim data
            if replacenans is not None:
                newx = range(len(data)) # Create a new x array the size of the original array
                if replacenans==True: replacenans = 'nearest'
                if replacenans in ['nearest','linear']:
                    sanitized = smoothinterp(newx, inds, sanitized, method=replacenans, smoothness=0) # Replace nans with interpolated values
                else:
                    naninds = inds = np.nonzero(np.isnan(data))[0]
                    sanitized = scu.dcp(data)
                    sanitized[naninds] = replacenans
            if len(sanitized)==0:
                if defaultval is not None:
                    sanitized = defaultval
                else:
                    sanitized = 0.0
                    if verbose: # pragma: no cover
                        if label is None: label = 'this parameter'
                        print(f'sc.sanitize(): no data entered for {label}, assuming 0')
        except Exception as E: # pragma: no cover
            if die:
                errormsg = f'Sanitization failed on array: "{repr(E)}":\n{data}'
                raise RuntimeError(errormsg)
            else:
                sanitized = data # Give up and just return an empty array
                inds = []
        if returninds: return sanitized, inds
        else:          return sanitized


def isprime(n, verbose=False):
    '''
    Determine if a number is prime.

    From https://stackoverflow.com/questions/15285534/isprime-function-for-python-language

    **Example**::

        for i in range(100): print(i) if sc.isprime(i) else None
    '''
    if n < 2:
        if verbose: print('Not prime: n<2')
        return False
    if n == 2:
        if verbose: print('Is prime: n=2')
        return True
    if n == 3:
        if verbose: print('Is prime: n=3')
        return True
    if n%2 == 0:
        if verbose: print('Not prime: divisible by 2')
        return False
    if n%3 == 0:
        if verbose: print('Not prime: divisible by 3')
        return False
    if n < 9:
        if verbose: print('Is prime: <9 and not divisible by 2')
        return True
    r = int(n**0.5)
    f = 5
    while f <= r:
        if n%f == 0:
            if verbose: print(f'Not prime: divisible by {f}')
            return False
        if n%(f+2) == 0:
            if verbose: print(f'Not prime: divisible by {f+2}')
            return False
        f +=6
    if verbose: print('Is prime!')
    return True



##############################################################################
#%% Other functions
##############################################################################

__all__ += ['perturb', 'normsum', 'normalize', 'inclusiverange', 'randround', 'cat']


def perturb(n=1, span=0.5, randseed=None, normal=False):
    '''
    Define an array of numbers uniformly perturbed with a mean of 1.

    Args:
        n (int): number of points
        span (float): width of distribution on either side of 1
        normal (bool):  whether to use a normal distribution instead of uniform

    **Example**::

        sc.perturb(5, 0.3) # Returns e.g. array([0.73852362, 0.7088094 , 0.93713658, 1.13150755, 0.87183371])
    '''
    if randseed is not None:
        np.random.seed(int(randseed)) # Optionally reset random seed
    if normal:
        output = 1.0 + span*np.random.randn(n)
    else:
        output = 1.0 + 2*span*(np.random.rand(n)-0.5)
    return output


def normsum(arr, total=None):
    '''
    Multiply a list or array by some normalizing factor so that its sum is equal
    to the total. Formerly called ``sc.scaleratio()``.

    Args:
        arr (array): array (or list) to normalize
        total (float): amount to sum to (default 1)

    **Example**::

        normarr = sc.normsum([2,5,3,10], 100) # Scale so sum equals 100; returns [10.0, 25.0, 15.0, 50.0]

    Renamed in version 1.0.0.
    '''
    if total is None: total = 1.0
    origtotal = float(sum(arr))
    ratio = float(total)/origtotal
    out = np.array(arr)*ratio
    if isinstance(arr, list): out = out.tolist() # Preserve type
    return out


def normalize(arr, minval=0.0, maxval=1.0):
    '''
    Rescale an array between a minimum value and a maximum value.

    Args:
        arr (array): array to normalize
        minval (float): minimum value in rescaled array
        maxval (float): maximum value in rescaled array

    **Example**::

        normarr = sc.normalize([2,3,7,27]) # Returns array([0.  , 0.04, 0.2 , 1.  ])
    '''
    out = np.array(arr, dtype=float) # Ensure it's a float so divide works
    out -= out.min()
    out /= out.max()
    out *= (maxval - minval)
    out += minval
    if isinstance(arr, list): out = out.tolist() # Preserve type
    return out


def inclusiverange(*args, **kwargs):
    '''
    Like ``np.arange()``/``np.linspace()``, but includes the start and stop points.
    Accepts 0-3 args, or the kwargs start, stop, step.

    In most cases, equivalent to ``np.linspace(start, stop, int((stop-start)/step)+1)``.

    Args:
        start (float): value to start at
        stop (float): value to stop at
        step (float): step size
        kwargs (dict): passed to ``np.linspace()``

    **Examples**::

        x = sc.inclusiverange(10)        # Like np.arange(11)
        x = sc.inclusiverange(3,5,0.2)   # Like np.linspace(3, 5, int((5-3)/0.2+1))
        x = sc.inclusiverange(stop=5)    # Like np.arange(6)
        x = sc.inclusiverange(6, step=2) # Like np.arange(0, 7, 2)
    '''
    # Handle args
    if len(args)==0:
        start, stop, step = None, None, None
    elif len(args)==1:
        stop = args[0]
        start, step = None, None
    elif len(args)==2:
        start = args[0]
        stop  = args[1]
        step =  None
    elif len(args)==3:
        start = args[0]
        stop  = args[1]
        step  = args[2]
    else: # pragma: no cover
        raise ValueError('Too many arguments supplied: inclusiverange() accepts 0-3 arguments')

    # Handle kwargs
    start = kwargs.pop('start', start)
    stop  = kwargs.pop('stop',  stop)
    step  = kwargs.pop('step',  step)

    # Finalize defaults
    if start is None: start = 0
    if stop  is None: stop  = 1
    if step  is None: step  = 1

    # OK, actually generate -- can't use arange since handles floating point arithmetic badly, e.g. compare arange(2000, 2020, 0.2) with arange(2000, 2020.2, 0.2)
    x = np.linspace(start, stop, int(round((stop-start)/float(step))+1), **kwargs)
    return x


def randround(x):
    '''
    Round a float, list, or array probabilistically to the nearest integer. Works
    for both positive and negative values.

    Adapted from:
        https://stackoverflow.com/questions/19045971/random-rounding-to-integer-in-python

    Args:
        x (int, list, arr): the floating point numbers to probabilistically convert to the nearest integer

    Returns:
        Array of integers

    **Example**::

        sc.randround(np.random.randn(8)) # Returns e.g. array([-1,  0,  1, -2,  2,  0,  0,  0])

    New in version 1.0.0.
    '''
    if isinstance(x, np.ndarray):
        output = np.array(np.floor(x+np.random.random(x.size)), dtype=int)
    elif isinstance(x, list):
        output = [randround(i) for i in x]
    else:
        output = int(np.floor(x+np.random.random()))
    return output


def cat(*args, axis=None, copy=False, **kwargs):
    '''
    Like np.concatenate(), but takes anything and returns an array. Useful for
    e.g. appending a single number onto the beginning or end of an array.

    Args:
        args   (any):  items to concatenate into an array
        axis   (int):  axis along which to concatenate
        copy   (bool): whether or not to deepcopy the result
        kwargs (dict): passed to ``np.array()``

    **Examples**::

        arr = sc.cat(4, np.ones(3))
        arr = sc.cat(np.array([1,2,3]), [4,5], 6)
        arr = sc.cat(np.random.rand(2,4), np.random.rand(2,6), axis=1)

    | New in version 1.0.0.
    | New in version 1.1.0: "copy" and keyword arguments.
    '''
    if not len(args):
        return np.array([])
    output = scu.promotetoarray(args[0])
    for arg in args[1:]:
        arg = scu.promotetoarray(arg)
        output = np.concatenate((output, arg), axis=axis)
    output = np.array(output, **kwargs)
    if copy:
        output = scu.dcp(output)
    return output



##############################################################################
#%% Smoothing functions
##############################################################################


__all__ += ['rolling', 'convolve', 'smooth', 'smoothinterp', 'gauss1d', 'gauss2d']


def rolling(data, window=7, operation='mean', **kwargs):
    '''
    Alias to Pandas' rolling() (window) method to smooth a series.

    Args:
        data (list/arr): the 1D or 2D data to be smoothed
        window (int): the length of the window
        operation (str): the operation to perform: 'mean' (default), 'median', 'sum', or 'none'
        kwargs (dict): passed to pd.Series.rolling()

    **Example**::

        data = [5,5,5,0,0,0,0,7,7,7,7,0,0,3,3,3]
        rolled = sc.rolling(data)
    '''
    import pandas as pd # Optional import

    # Handle the data
    data = np.array(data)
    data = pd.Series(data) if data.ndim == 1 else pd.DataFrame(data)

    # Perform the roll
    roll = data.rolling(window=window, **kwargs)

    # Handle output
    if   operation in [None, 'none']: output = roll
    elif operation == 'mean':         output = roll.mean().values
    elif operation == 'median':       output = roll.median().values
    elif operation == 'sum':          output = roll.sum().values
    else:
        errormsg = f'Operation "{operation}" not recognized; must be mean, median, sum, or none'
        raise ValueError(errormsg)

    return output


def convolve(a, v):
    '''
    Like ``np.convolve()``, but always returns an array the size of the first array
    (equivalent to mode='same'), and solves the boundary problem present in ``np.convolve()``
    by adjusting the edges by the weight of the convolution kernel.

    Args:
        a (arr): the input array
        v (arr): the convolution kernel

    **Example**::

        a = np.ones(5)
        v = np.array([0.3, 0.5, 0.2])
        c1 = np.convolve(a, v, mode='same') # Returns array([0.8, 1.  , 1.  , 1.  , 0.7])
        c2 = sc.convolve(a, v)              # Returns array([1., 1., 1., 1., 1.])

    | New in version 1.3.0.
    | New in version 1.3.1: handling the case where len(a) < len(v)
    '''

    # Handle types
    a = np.array(a)
    v = np.array(v)

    # Perform standard Numpy convolution
    out = np.convolve(a, v, mode='same')

    # Handle edge weights
    len_a = len(a) # Length of input array
    len_v = len(v) # Length of kernel
    minlen = min(len_a, len_v)
    vtot = v.sum() # Total kernel weight
    len_lhs = minlen // 2 # Number of points to re-weight on LHS
    len_rhs = (minlen-1) // 2 # Ditto
    if len_lhs:
        w_lhs = np.cumsum(v)/vtot # Cumulative sum of kernel weights on the LHS, divided by total weight
        w_lhs = w_lhs[-1-len_lhs:-1] # Trim to the correct length
        out[:len_lhs] = out[:len_lhs]/w_lhs # Re-weight
    if len_rhs:
        w_rhs = (np.cumsum(v[::-1])[::-1]/vtot) # Ditto, reversed for RHS
        w_rhs = w_rhs[1:len_rhs+1] # Ditto
        out[-len_rhs:] = out[-len_rhs:]/w_rhs # Ditto

    # Handle the case where len(v) > len(a)
    len_diff = max(0, len_v - len_a)
    if len_diff:
        lhs_trim = len_diff // 2
        out = out[lhs_trim:lhs_trim+len_a]

    return out


def smooth(data, repeats=None, kernel=None, legacy=False):
    '''
    Very simple function to smooth a 1D or 2D array.

    See also ``sc.gauss1d()`` for simple Gaussian smoothing.

    Args:
        data (arr): 1D or 2D array to smooth
        repeats (int): number of times to apply smoothing (by default, scale to be 1/5th the length of data)
        kernel (arr): the smoothing kernel to use (default: ``[0.25, 0.5, 0.25]``)
        legacy (bool): if True, use the old (pre-1.3.0) method of calculation that doesn't correct for edge effects

    **Example**::

        data = pl.randn(5,5)
        smoothdata = sc.smooth(data)

    New in version 1.3.0: Fix edge effects.
    '''
    if repeats is None:
        repeats = int(np.floor(len(data)/5))
    if kernel is None:
        kernel = [0.25,0.5,0.25]
    kernel = np.array(kernel)
    output = scu.dcp(np.array(data))

    # Only convolve the kernel with itself -- equivalent to doing the full convolution multiple times
    v = scu.dcp(kernel)
    for r in range(repeats-1):
        v = np.convolve(v, kernel, mode='full')

    # Support legacy method of not correcting for edge effects
    if legacy:
        conv = np.convolve
        kw = {'mode':'same'}
    else:
        conv = convolve
        kw = {}

    # Perform the convolution
    if output.ndim == 1:
        output = conv(output, v, **kw)
    elif output.ndim == 2:
        for i in range(output.shape[0]): output[i,:] = conv(output[i,:], v, **kw)
        for j in range(output.shape[1]): output[:,j] = conv(output[:,j], v, **kw)
    else: # pragma: no cover
        errormsg = 'Simple smoothing only implemented for 1D and 2D arrays'
        raise ValueError(errormsg)

    return output


def smoothinterp(newx=None, origx=None, origy=None, smoothness=None, growth=None, ensurefinite=False, keepends=True, method='linear'):
    '''
    Smoothly interpolate over values and keep end points. Same format as numpy.interp().

    Args:
        newx (arr): the points at which to interpolate
        origx (arr): the original x coordinates
        origy (arr): the original y coordinates
        smoothness (float): how much to smooth
        growth (float): the growth rate to apply past the ends of the data [deprecated]
        ensurefinite (bool):  ensure all values are finite
        keepends (bool): whether to keep the ends [deprecated]
        method (str): the type of interpolation to use (options are 'linear' or 'nearest')

    Returns:
        newy (arr): the new y coordinates

    **Example**::

        origy = np.array([0,0.1,0.3,0.8,0.7,0.9,0.95,1])
        origx = np.linspace(0,1,len(origy))
        newx = np.linspace(0,1,5*len(origy))
        newy = sc.smoothinterp(newx, origx, origy, smoothness=5)
        pl.plot(newx,newy)
        pl.scatter(origx,origy)

    Version: 2018jan24
    '''
    # Ensure arrays and remove NaNs
    if scu.isnumber(newx):  newx = [newx] # Make sure it has dimension
    if scu.isnumber(origx): origx = [origx] # Make sure it has dimension
    if scu.isnumber(origy): origy = [origy] # Make sure it has dimension
    newx  = np.array(newx, dtype=float)
    origx = np.array(origx, dtype=float)
    origy = np.array(origy, dtype=float)

    # If only a single element, just return it, without checking everything else
    if len(origy)==1:
        newy = np.zeros(newx.shape)+origy[0]
        return newy

    if not(newx.shape):  raise ValueError('To interpolate, must have at least one new x value to interpolate to') # pragma: no cover
    if not(origx.shape): raise ValueError('To interpolate, must have at least one original x value to interpolate to') # pragma: no cover
    if not(origy.shape): raise ValueError('To interpolate, must have at least one original y value to interpolate to') # pragma: no cover
    if not(origx.shape==origy.shape): # pragma: no cover
        errormsg = f'To interpolate, original x and y vectors must be same length (x={len(origx)}, y={len(origy)})'
        raise ValueError(errormsg)

    # Make sure it's in the correct order
    correctorder = np.argsort(origx)
    origx = origx[correctorder]
    origy = origy[correctorder]
    neworder = np.argsort(newx)
    newx = newx[neworder] # And sort newx just in case

    # Only keep finite elements
    finitey = np.isfinite(origy) # Boolean for whether it's finite
    if finitey.any() and not finitey.all(): # If some but not all is finite, pull out indices that are
        finiteorigy = origy[finitey]
        finiteorigx = origx[finitey]
    else: # Otherwise, just copy the original
        finiteorigy = origy.copy()
        finiteorigx = origx.copy()

    # Perform actual interpolation
    if method=='linear':
        newy = np.interp(newx, finiteorigx, finiteorigy) # Perform standard interpolation without infinities
    elif method=='nearest':
        newy = np.zeros(newx.shape) # Create the new array of the right size
        for i,x in enumerate(newx): # Iterate over each point
            xind = np.argmin(abs(finiteorigx-x)) # Find the nearest neighbor
            newy[i] = finiteorigy[xind] # Copy it
    else: # pragma: no cover
        errormsg = f'Method "{method}" not found; methods are "linear" or "nearest"'
        raise ValueError(errormsg)

    # Perform smoothing
    if smoothness is None: smoothness = np.ceil(len(newx)/len(origx)) # Calculate smoothness: this is consistent smoothing regardless of the size of the arrays
    smoothness = int(smoothness) # Make sure it's an appropriate number

    if smoothness:
        kernel = np.exp(-np.linspace(-2,2,2*smoothness+1)**2)
        kernel /= kernel.sum()
        validinds = findinds(~np.isnan(newy)) # Remove nans since these don't exactly smooth well
        if len(validinds): # No point doing these steps if no non-nan values
            validy = newy[validinds]
            prepend = validy[0]*np.ones(smoothness)
            postpend = validy[-1]*np.ones(smoothness)
            if not keepends: # pragma: no cover
                try: # Try to compute slope, but use original prepend if it doesn't work
                    dyinitial = (validy[0]-validy[1])
                    prepend = validy[0]*np.ones(smoothness) + dyinitial*np.arange(smoothness,0,-1)
                except:
                    pass
                try: # Try to compute slope, but use original postpend if it doesn't work
                    dyfinal = (validy[-1]-validy[-2])
                    postpend = validy[-1]*np.ones(smoothness) + dyfinal*np.arange(1,smoothness+1,1)
                except:
                    pass
            validy = np.concatenate([prepend, validy, postpend])
            validy = np.convolve(validy, kernel, 'valid') # Smooth it out a bit
            newy[validinds] = validy # Copy back into full vector

    # Apply growth if required
    if growth is not None: # pragma: no cover
        pastindices = findinds(newx<origx[0])
        futureindices = findinds(newx>origx[-1])
        if len(pastindices): # If there are past data points
            firstpoint = pastindices[-1]+1
            newy[pastindices] = newy[firstpoint] * np.exp((newx[pastindices]-newx[firstpoint])*growth) # Get last 'good' data point and apply inverse growth
        if len(futureindices): # If there are past data points
            lastpoint = futureindices[0]-1
            newy[futureindices] = newy[lastpoint] * np.exp((newx[futureindices]-newx[lastpoint])*growth) # Get last 'good' data point and apply growth

    # Add infinities back in, if they exist
    if any(~np.isfinite(origy)): # pragma: no cover # Infinities exist, need to add them back in manually since interp can only handle nan
        if not ensurefinite: # If not ensuring all entries are finite, put nonfinite entries back in
            orignan      = np.zeros(len(origy)) # Start from scratch
            origplusinf  = np.zeros(len(origy)) # Start from scratch
            origminusinf = np.zeros(len(origy)) # Start from scratch
            orignan[np.isnan(origy)]     = np.nan  # Replace nan entries with nan
            origplusinf[origy==np.inf]   = np.nan  # Replace plus infinite entries with nan
            origminusinf[origy==-np.inf] = np.nan  # Replace minus infinite entries with nan
            newnan      = np.interp(newx, origx, orignan) # Interpolate the nans
            newplusinf  = np.interp(newx, origx, origplusinf) # ...again, for positive
            newminusinf = np.interp(newx, origx, origminusinf) # ...and again, for negative
            newy[np.isnan(newminusinf)] = -np.inf # Add minus infinity back in first
            newy[np.isnan(newplusinf)]  = np.inf # Then, plus infinity
            newy[np.isnan(newnan)]  = np.nan # Finally, the nans

    # Restore original sort order for newy
    restoredorder = np.argsort(neworder)
    newy = newy[restoredorder]

    return newy


# For Gaussian functions -- doubles the speed to convert to 32 bit, functions faster than lambdas
def _arr32(arr): return np.array(arr, dtype=np.float32)
def _f32(x):     return np.float32(x)


def gauss1d(x=None, y=None, xi=None, scale=None, use32=True):
    '''
    Gaussian 1D smoothing kernel.

    Create smooth interpolation of input points at interpolated points. If no points
    are supplied, use the same as the input points.

    Args:
        x (arr): 1D list of x coordinates
        y (arr): 1D list of y values at each of the x coordinates
        xi (arr): 1D list of points to calculate the interpolated y
        scale (float): how much smoothing to apply (by default, width of 5 data points)
        use32 (bool): convert arrays to 32-bit floats (doubles speed for large arrays)

    **Examples**::

        # Setup
        import pylab as pl
        x = pl.rand(40)
        y = (x-0.3)**2 + 0.2*pl.rand(40)

        # Smooth
        yi = sc.gauss1d(x, y)
        yi2 = sc.gauss1d(x, y, scale=0.3)
        xi3 = pl.linspace(0,1)
        yi3 = sc.gauss1d(x, y, xi)

        # Plot oiginal and interpolated versions
        pl.scatter(x, y,     label='Original')
        pl.scatter(x, yi,    label='Default smoothing')
        pl.scatter(x, yi2,   label='More smoothing')
        pl.scatter(xi3, yi3, label='Uniform spacing')
        pl.show()

        # Simple usage
        sc.gauss1d(y)

    New in version 1.3.0.
    '''

    # Swap inputs if x is provided but not y
    if y is None and x is not None:
        y,x = x,y
    if x is None:
        x = np.arange(len(y))
    if xi is None:
        xi = x

    # Convert to arrays
    try:
        orig_dtype = y.dtype
    except:
        orig_dtype = np.float64
    if use32:
        x, y, xi, = _arr32(x), _arr32(y), _arr32(xi)

    # Handle scale
    if scale is None:
        minmax = np.ptp(x) # Calculate the range of x
        npts = len(x)
        scale = 5*minmax/npts
    scale  = _f32(scale)

    def calc(xi):
        ''' Calculate the calculation '''
        dist = (x - xi)/scale
        weights = np.exp(-dist**2)
        weights = weights/np.sum(weights)
        val = np.sum(weights*y)
        return val

    # Actual computation
    n = len(xi)
    yi = np.zeros(n)
    for i in range(n):
        yi[i] = calc(xi[i])

    yi = np.array(yi, dtype=orig_dtype) # Convert back

    return yi


def gauss2d(x=None, y=None, z=None, xi=None, yi=None, scale=1.0, xscale=1.0, yscale=1.0, grid=False, use32=True):
    '''
    Gaussian 2D smoothing kernel.

    Create smooth interpolation of input points at interpolated points. Can handle
    either 1D or 2D inputs.

    Args:
        x (arr): 1D or 2D array of x coordinates (if None, take from z)
        y (arr): ditto, for y
        z (arr): 1D or 2D array of z values at each of the (x,y) points
        xi (arr): 1D or 2D array of points to calculate the interpolated Z; if None, same as x
        yi (arr): ditto, for y
        scale (float): overall scale factor
        xscale (float): ditto, just for x
        yscale (float): ditto, just for y
        grid (bool): if True, then return Z at a grid of (xi,yi) rather than at points
        use32 (bool): convert arrays to 32-bit floats (doubles speed for large arrays)

    **Examples**::

        # Setup
        import pylab as pl
        x = pl.rand(40)
        y = pl.rand(40)
        z = 1-(x-0.5)**2 + (y-0.5)**2 # Make a saddle

        # Simple usage -- only works if z is 2D
        zi0 = sc.gauss2d(pl.rand(10,10))
        sc.surf3d(zi0)

        # Method 1 -- form grid
        xi = pl.linspace(0,1,20)
        yi = pl.linspace(0,1,20)
        zi = sc.gauss2d(x, y, z, xi, yi, scale=0.1, grid=True)

        # Method 2 -- use points directly
        xi2 = pl.rand(400)
        yi2 = pl.rand(400)
        zi2 = sc.gauss2d(x, y, z, xi2, yi2, scale=0.1)

        # Plot oiginal and interpolated versions
        sc.scatter3d(x, y, z, c=z)
        sc.surf3d(zi)
        sc.scatter3d(xi2, yi2, zi2, c=zi2)
        pl.show()

    | New in version 1.3.0.
    | New in version 1.3.1: default arguments; support for 2D inputs
    '''
    # Swap variables if needed
    if z is None and x is not None:
        z,x = x,z
    if x is None or y is None:
        if z.ndim != 2:
            errormsg = f'If the x and y axes are not provided, then z must be 2D, not {z.ndim}D'
            raise ValueError(errormsg)
        else:
            x = np.arange(z.shape[1]) # It's counterintuitive, but x is the 1st dimension, not the 0th
            y = np.arange(z.shape[0])

    # Handle shapes
    if z.ndim == 2 and (x.ndim == 1) and (y.ndim == 1):
        if (z.shape[0] != z.shape[1]) and (len(x) == z.shape[0]) and (len(y) == z.shape[1]):
            print(f'sc.gauss2d() warning: the length of x (={len(x)}) and y (={len(y)}) match the wrong dimensions of z; transposing')
            z = z.transpose()
        if len(x) != z.shape[1] or len(y) != z.shape[0]:
            errormsg = f'Shape mismatch: (y, x) = {(len(y), len(x))}, but z = {z.shape}'
            raise ValueError(errormsg)

        # If checks pass, convert to full arrays
        x, y = np.meshgrid(x, y)

    # Handle data types
    orig_dtype = z.dtype
    if xi is None: xi = scu.dcp(x)
    if yi is None: yi = scu.dcp(y)
    if use32:
        x, y, z, xi, yi = _arr32(x), _arr32(y), _arr32(z), _arr32(xi), _arr32(yi)
        scale, xscale, yscale = _f32(scale), _f32(xscale), _f32(yscale)
    xsc = xscale*scale
    ysc = yscale*scale

    # Now that we have xi and yi, handle more 1D vs 2D logic
    if xi.ndim == 1 and yi.ndim == 1 and grid:
        xi, yi = np.meshgrid(xi, yi)
    if xi.shape != yi.shape:
        errormsg = f'Output arrays must have same shape, but xi = {xi.shape} and yi = {yi.shape}'
        raise ValueError(errormsg)

    # Flatten everything and check sizes
    orig_shape = xi.shape
    x  = x.flatten()
    y  = y.flatten()
    z  = z.flatten()
    xi = xi.flatten()
    yi = yi.flatten()
    ni = len(xi)
    if len(x) != len(y) != len(z):
        errormsg = f'Input arrays do not have the same number of elements: x = {len(x)}, y = {len(y)}, z = {len(z)}'
        raise ValueError(errormsg)
    if len(xi) != len(yi):
        errormsg = f'Output arrays do not have the same number of elements: xi = {len(xi)}, yi = {len(yi)}'
        raise ValueError(errormsg)

    def calc(xi, yi):
        ''' Calculate the calculation '''
        dist = ((x - xi)/xsc)**2 + ((y - yi)/ysc)**2
        weights = np.exp(-dist)
        weights = weights/np.sum(weights)
        val = np.sum(weights*z)
        return val

    # Actual computation
    zi = np.zeros(ni)
    if use32: zi = _arr32(zi)
    for i in range(ni):
        zi[i] = calc(xi[i], yi[i])

    # Convert back to original size and dtype
    zi = np.array(zi.reshape(orig_shape), dtype=orig_dtype)

    return zi