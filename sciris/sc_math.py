"""
Extensions to Numpy, including finding array elements and smoothing data.

Highlights:
    - :func:`sc.findinds() <findinds>`: find indices of an array matching a condition
    - :func:`sc.findnearest() <findnearest>`: find nearest matching value
    - :func:`sc.rolling() <rolling>`: calculate rolling average
    - :func:`sc.smooth() <smooth>`: simple smoothing of 1D or 2D arrays
"""

import numpy as np
import pandas as pd
import warnings
import sciris as sc


##############################################################################
#%% Find and approximation functions
##############################################################################

__all__ = ['approx', 'safedivide', 'findinds', 'findfirst', 'findlast', 'findnearest', 'count',
           'dataindex', 'getvalidinds', 'getvaliddata', 'sanitize', 'rmnans','fillnans', 
           'findnans', 'nanequal', 'isprime', 'numdigits']


def approx(val1=None, val2=None, eps=None, **kwargs):
    """
    Determine whether two scalars (or an array and a scalar) approximately match.
    Alias for :func:`np.isclose() <numpy.isclose>` and may be removed in future versions.

    Args:
        val1 (number or array): the first value
        val2 (number): the second value
        eps (float): absolute tolerance
        kwargs (dict): passed to np.isclose()

    **Examples**::

        sc.approx(2*6, 11.9999999, eps=1e-6) # Returns True
        sc.approx([3,12,11.9], 12) # Returns array([False, True, False], dtype=bool)
    """
    if eps is not None:
        kwargs['atol'] = eps # Rename kwarg to match np.isclose()
    output = np.isclose(a=val1, b=val2, **kwargs)
    return output


def safedivide(numerator=None, denominator=None, default=None, eps=None, warn=False):
    """
    Handle divide-by-zero and divide-by-nan elegantly.

    **Examples**::

        sc.safedivide(numerator=0, denominator=0, default=1, eps=0) # Returns 1
        sc.safedivide(numerator=5, denominator=2.0, default=1, eps=1e-3) # Returns 2.5
        sc.safedivide(3, np.array([1,3,0]), -1, warn=True) # Returns array([ 3,  1, -1])
    """
    # Set some defaults
    if numerator   is None: numerator   = 1.0
    if denominator is None: denominator = 1.0
    if default     is None: default     = 0.0

    # Handle types
    if isinstance(numerator,   list): numerator   = np.array(numerator)
    if isinstance(denominator, list): denominator = np.array(denominator)

    # Handle the logic
    invalid = approx(denominator, 0.0, eps=eps)
    if sc.isnumber(denominator): # The denominator is a scalar
        if invalid:
            output = default
        else: # pragma: no cover
            output = numerator/denominator
    elif sc.checktype(denominator, 'array'):
        if not warn:
            denominator[invalid] = 1.0 # Replace invalid values with 1
        output = numerator/denominator
        output[invalid] = default
    else: # pragma: no cover # Unclear input, raise exception
        errormsg = f'Input type {type(denominator)} not understood: must be number or array'
        raise TypeError(errormsg)

    return output


def findinds(arr=None, val=None, *args, eps=1e-6, first=False, last=False, ind=None, die=True, **kwargs):
    """
    Find matches even if two things aren't eactly equal (e.g. floats vs. ints).

    If one argument, find nonzero values. With two arguments, check for equality
    using eps (by default 1e-6, to handle single-precision floating point). Returns 
    a tuple of arrays if val1 is multidimensional, else returns an array. Similar 
    to calling :func:`np.nonzero(np.isclose(arr, val))[0] <numpy.nonzero>`.

    Args:
        arr    (array): the array to find values in
        val    (float): if provided, the value to match
        args   (list):  if provided, additional boolean arrays
        eps    (float): the precision for matching (default 1e-6, equivalent to :func:`numpy.isclose`'s atol)
        first  (bool):  whether to return the first matching value (equivalent to ind=0)
        last   (bool):  whether to return the last matching value (equivalent to ind=-1)
        ind    (int):   index of match to retrieve
        die    (bool):  whether to raise an exception if first or last is true and no matches were found
        kwargs (dict):  passed to :func:`numpy.isclose()`

    **Examples**::

        data = np.random.rand(10)
        sc.findinds(data<0.5) # Standard usage; returns e.g. array([2, 4, 5, 9])
        sc.findinds(data>0.1, data<0.5) # Multiple arguments

        sc.findinds([2,3,6,3], 3) # Returs array([1,3])
        sc.findinds([2,3,6,3], 3, first=True) # Returns 1

    | *New in version 1.2.3:* "die" argument
    | *New in version 2.0.0:* fix string matching; allow multiple arguments
    | *New in version 3.0.0:* multidimensional arrays now return a list of tuples
    """

    # Handle first or last
    if first and last: raise ValueError('Can use first or last but not both')
    elif first: ind = 0
    elif last:  ind = -1

    # Handle kwargs
    atol = kwargs.pop('atol', eps) # Ensure atol isn't specified twice
    if 'val1' in kwargs or 'val2' in kwargs: # pragma: no cover
        arr = kwargs.pop('val1', arr)
        val = kwargs.pop('val2', val)
        warnmsg = 'sc.findinds() arguments "val1" and "val2" have been deprecated as of v1.0.0; use "arr" and "val" instead'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)

    # Calculate matches
    arr = sc.toarray(arr)
    arglist = list(args)
    if val is None: # Check for equality
        boolarr = arr # If not, just check the truth condition
    else:
        if sc.isstring(val): # A string only matches itself
            boolarr = (arr == val)
        else:
            if sc.isnumber(val): # Standard usage, use nonzero
                boolarr = np.isclose(a=arr, b=val, atol=atol, **kwargs) # If absolute difference between the two values is less than a certain amount
            elif sc.checktype(val, 'arraylike'): # It's not actually a value, it's another array
                boolarr = arr
                arglist.append(sc.toarray(val))
            else: # pragma: no cover
                errormsg = f'Cannot understand input {type(val)}: must be number or array-like'
                raise TypeError(errormsg)

    # Handle any additional inputs
    for arg in arglist:
        if arg.shape != boolarr.shape: # pragma: no cover
            errormsg = f'Could not handle inputs with shapes {boolarr.shape} vs {arg.shape}'
            raise ValueError(errormsg)
        boolarr *= arg

    # Actually find indices
    output = np.nonzero(boolarr)

    # Process output
    try:
        if arr.ndim == 1: # Uni-dimensional
            output = output[0] # Return an array rather than a tuple of arrays if one-dimensional
            if ind is not None:
                output = output[ind] # And get the first element
        else:
            if ind is not None:
                output = tuple([output[i][ind] for i in range(arr.ndim)])
    except IndexError as E:
        if die:
            errormsg = 'No matching values found; use die=False to return None instead of raising an exception'
            raise IndexError(errormsg) from E
        else:
            output = None

    return output


def findfirst(*args, **kwargs):
    """ Alias for :func:`sc.findinds(..., first=True) <findinds>`. *New in version 1.0.0.* """
    return findinds(*args, **kwargs, first=True)


def findlast(*args, **kwargs):
    """ Alias for :func:`sc.findinds(..., last=True) <findinds>`. *New in version 1.0.0.* """
    return findinds(*args, **kwargs, last=True)


def findnearest(series=None, value=None):
    """
    Return the index of the nearest match in series to value -- like :func:`sc.findinds() <findinds>`, but
    always returns an object with the same type as value (i.e. findnearest with
    a number returns a number, findnearest with an array returns an array).
    
    Args:
        series (array): the array of numbers to look for nearest matches in
        value (scalar or array): the number or numbers to compare against

    **Examples**::

        sc.findnearest(rand(10), 0.5) # returns whichever index is closest to 0.5
        sc.findnearest([2,3,6,3], 6) # returns 2
        sc.findnearest([2,3,6,3], 6) # returns 2
        sc.findnearest([0,2,4,6,8,10], [3, 4, 5]) # returns array([1, 2, 2])
    """
    series = sc.toarray(series)
    if sc.isnumber(value):
        output = np.argmin(abs(series-value))
    else:
        output = []
        for val in value: output.append(findnearest(series, val))
        output = sc.toarray(output)
    return output


def count(arr=None, val=None, eps=1e-6, **kwargs):
    """
    Count the number of matching elements.

    Similar to :func:`numpy.count_nonzero()`, but allows for slight mismatches (e.g.,
    floats vs. ints). Equivalent to ``len(sc.findinds())``.

    Args:
        arr (array): the array to find values in
        val (float): if provided, the value to match
        eps (float): the precision for matching (default 1e-6, equivalent to :func:`numpy.isclose`'s atol)
        kwargs (dict): passed to :func:`numpy.isclose()` 

    **Examples**::

        sc.count(rand(10)<0.5) # returns e.g. 4
        sc.count([2,3,6,3], 3) # returns 2

    *New in version 2.0.0.*
    """
    output = len(findinds(arr=arr, val=val, eps=eps, **kwargs))
    return output


def dataindex(dataarray, index): # pragma: no cover
    """
    Take an array of data and return either the first or last (or some other) non-NaN entry.

    This function is deprecated.
    """
    nrows = np.shape(dataarray)[0] # See how many rows need to be filled (either npops, nprogs, or 1).
    output = np.zeros(nrows)       # Create structure
    for r in range(nrows):
        output[r] = sanitize(dataarray[r])[index] # Return the specified index -- usually either the first [0] or last [-1]

    return output


def getvalidinds(data=None, filterdata=None): # pragma: no cover
    """
    Return the indices that are valid based on the validity of the input data from an arbitrary number
    of 1-D vector inputs. Note, closely related to :func:`sc.getvaliddata() <getvaliddata>`.

    This function is deprecated.

    **Example**::

        sc.getvalidinds([3,5,8,13], [2000, nan, nan, 2004]) # Returns array([0,3])
    """
    data = sc.toarray(data)
    if filterdata is None: filterdata = data # So it can work on a single input -- more or less replicates sanitize() then
    filterdata = sc.toarray(filterdata)
    if filterdata.dtype=='bool': filterindices = filterdata # It's already boolean, so leave it as is
    else:                        filterindices = findinds(~np.isnan(filterdata)) # Else, assume it's nans that need to be removed
    dataindices = findinds(~np.isnan(data)) # Also check validity of data
    validindices = np.intersect1d(dataindices, filterindices)
    return validindices # Only return indices -- WARNING, not consistent with sanitize()


def getvaliddata(data=None, filterdata=None, defaultind=0): # pragma: no cover
    """
    Return the data value indices that are valid based on the validity of the input data.

    This function is deprecated; see :func:`sc.sanitize() <sanitize>` instead.

    **Example**::

        sc.getvaliddata(array([3,5,8,13]), array([2000, nan, nan, 2004])) # Returns array([3,13])
    """
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


def sanitize(data=None, returninds=False, replacenans=None, defaultval=None, die=True, verbose=False, label=None):
        """
        Sanitize input to remove NaNs. (NB: :func:`sc.sanitize() <sanitize>` and :func:`sc.rmnans() <rmnans>` are aliases.)

        Returns an array with the sanitized data. If ``replacenans=True``, the sanitized
        array is of the same length/size as data. If ``replacenans=False``, the sanitized
        array may be shorter than data.

        Args:
            data        (arr/list)   : array or list with numbers to be sanitized
            returninds  (bool)       : whether to return indices of non-nan/valid elements, indices are with respect the shape of data
            replacenans (float/str)  : whether to replace the NaNs with the specified value, or if ``True`` or a string, using interpolation
            defaultval  (float)      : value to return if the sanitized array is empty
            die         (bool)       : whether to raise an exception if the sanitization failed (otherwise return an empty array)
            verbose     (bool)       : whether to print out a warning if no valid values are found
            label       (str)        : human readable label for data (for use with verbose mode only)

        **Examples**::

            data = [3, 4, np.nan, 8, 2, np.nan, np.nan, 8]
            sanitized1, inds = sc.sanitize(data, returninds=True) # Remove NaNs
            sanitized2 = sc.sanitize(data, replacenans=True) # Replace NaNs using nearest neighbor interpolation
            sanitized3 = sc.sanitize(data, replacenans='nearest') # Eequivalent to replacenans=True
            sanitized4 = sc.sanitize(data, replacenans='linear') # Replace NaNs using linear interpolation
            sanitized5 = sc.sanitize(data, replacenans=0) # Replace NaNs with 0

        | *New in version 2.0.0:* handle multidimensional arrays
        | *New in version 3.0.0:* return zero-length arrays if all NaN
        """
        try:
            data = sc.toarray(data) # Make sure it's an array
            is_multidim = data.ndim > 1
            if is_multidim:
                if not replacenans:
                    errormsg = 'For multidimensional data, NaNs cannot be removed. Set replacenans=<value>, or flatten data before use.'
                    raise ValueError(errormsg)
            inds = np.nonzero(~pd.isna(data))
            if not is_multidim:
                inds = inds[0] # Since np.nonzero() returns a tuple
                sanitized = data[inds] # Trim data

            if replacenans is not None:
                if replacenans is True:
                    replacenans = 'nearest'
                if sc.isstring(replacenans):
                    if replacenans in ['nearest','linear']:
                        if is_multidim:
                            errormsg = 'Cannot perform interpolation on multidimensional data; use replacenans=<value> instead'
                            raise NotImplementedError(errormsg)
                        newx = range(len(data)) # Create a new x array the size of the original array
                        sanitized = smoothinterp(newx, inds, sanitized, method=replacenans, smoothness=0) # Replace nans with interpolated values
                    else: # pragma: no cover
                        errormsg = f'Interpolation method "{replacenans}" not found: must be "nearest" or "linear"'
                        raise ValueError(errormsg)
                else:
                    naninds = np.nonzero(pd.isna(data))
                    sanitized = data.copy() # To avoid overwriting original array
                    sanitized[naninds] = replacenans

            if len(sanitized)==0:
                if defaultval is not None:
                    sanitized = defaultval
                else:
                    inds = []

                    if verbose: # pragma: no cover
                        if label is None: label = 'these input data'
                        print(f'sc.sanitize(): no valid values found for {label}. Returning 0.')
        except Exception as E: # pragma: no cover
            if die:
                raise E
            else:
                sanitized = data # Give up and just return original array
                inds = []
        if returninds: return sanitized, inds
        else:          return sanitized


# Define as an alias
rmnans = sanitize

def fillnans(data=None, replacenans=True, **kwargs):
    """
    Alias for :func:`sc.sanitize(..., replacenans=True) <sanitize>` with nearest interpolation
    (or a specified value).

    *New in version 2.0.0.*
    """
    return sanitize(data=data, replacenans=replacenans, **kwargs)


def findnans(data=None, **kwargs):
    """
    Alias for :func:`sc.findinds(np.isnan(data)) <findinds>`.
    
    **Examples**::
        
        data = [0, 1, 2, np.nan, 4, np.nan, 6, np.nan, np.nan, np.nan, 10]
        sc.findnans(data) # Returns array([3, 5, 7, 8, 9])

    | *New in version 3.0.0.*
    | *New in version 3.1.0:* replaced ``np.isnan`` with ``pd.isna`` for robustness
    """
    isnan = pd.isna(data)
    inds  = findinds(arr=isnan, **kwargs)
    return inds


_nan_fill = -528876923.87569493 # Define a random value that would never be encountered otherwise
def nanequal(arr, *args, scalar=False, equal_nan=True):
    """
    Compare two or more arrays for equality element-wise, treating NaN values as equal.
    
    Unnlike :func:`numpy.array_equal`, this function works even if the arrays cannot
    be cast to float.
    
    Args:
        arr (array): the array to use as the base for the comparison
        args (list): one or more arrays to compare to
        scalar (bool): whether to return a true/false value (else return the array)
    
    **Examples**::
        
        arr1 = np.array([1, 2, np.nan])
        arr2 = [1, 2, np.nan]
        sc.nanequal(arr1, arr2) # Returns array([ True,  True,  True])
        
        arr3 = [3, np.nan, 'foo']
        sc.nanequal(arr3, arr3, arr3, scalar=True) # Returns True
    
    *New in version 3.1.0.*
    """
    
    if not len(args): # pragma: no cover
        errormsg = 'Only one array provided; requires 2 or more'
        raise ValueError(errormsg)
        
    others = [sc.toarray(arg) for arg in args] # Convert everything to an array
    
    # Remove Nans from base array
    if equal_nan:
        isnan = pd.isna(arr)
        arr = sc.toarray(arr).copy()
        arr[isnan] = _nan_fill # Fill in NaN values
    
    eqarr = None
    
    for other in others:
        if other.shape != arr.shape: # Two arrays with different shapes are always false
            if scalar:
                return False
            else:
                return np.array([False])
        else:
            if equal_nan:
                isnan = pd.isna(other)
                other = other.copy()
                other[isnan] = _nan_fill # Fill in NaN values
            eq = (arr == other) # Do the comparison
            if eqarr is None:
                eqarr = eq
            else:
                eqarr *= eq
    
    if scalar:
        return np.all(eqarr)
    else:
        return eqarr
    



def isprime(n, verbose=False):
    """
    Determine if a number is prime.

    From https://stackoverflow.com/questions/15285534/isprime-function-for-python-language

    **Example**::

        for i in range(100): print(i) if sc.isprime(i) else None
    """
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


def numdigits(n, *args, count_minus=False, count_decimal=False):
    """
    Count the number of digits in a number (or list of numbers).

    Useful for e.g. knowing how long a string needs to be to fit a given number.

    If a number is less than 1, return the number of digits until the decimal
    place.

    Reference: https://stackoverflow.com/questions/22656345/how-to-count-the-number-of-digits-in-python

    Args:
        n (int/float/list/array): number or list of numbers
        args (list): additional numbers
        count_minus (bool): whether to count the minus sign as a digit
        count_decimal (bool): whether to count the decimal point as a digit

    **Examples**::

        sc.numdigits(12345) # Returns 5
        sc.numdigits(12345.5) # Returns 5
        sc.numdigits(0) # Returns 1
        sc.numdigits(-12345) # Returns 5
        sc.numdigits(-12345, count_minus=True) # Returns 6
        sc.numdigits(12, 123, 12345) # Returns [2, 3, 5]
        sc.numdigits(0.01) # Returns -2
        sc.numdigits(0.01, count_decimal=True) # Returns -4

    *New in version 2.0.0.*
    """
    is_scalar = True if sc.isnumber(n) and len(args) == 0 else False

    vals = cat(n, *args)

    output = []
    for n in vals:
        abs_n = abs(n)
        is_decimal = 0 < abs_n < 1
        n_digits = 1
        if n < 0 and count_minus: # pragma: no cover
            n_digits += 1
        if is_decimal:
            if count_decimal:
                n_digits += 1
            else:
                n_digits -= 1

        if abs_n > 0:
            if is_decimal:
                n_digits = -n_digits
            n_digits += int(np.floor(np.log10(abs_n)))
        output.append(n_digits)
    output = np.array(output)
    if is_scalar:
        output = output[0]

    return output



##############################################################################
#%% Other functions
##############################################################################

__all__ += ['perturb', 'normsum', 'normalize', 'inclusiverange', 'randround', 
            'cat', 'linregress', 'sem']


def perturb(n=1, span=0.5, randseed=None, normal=False):
    """
    Define an array of numbers uniformly perturbed with a mean of 1.

    Args:
        n (int): number of points
        span (float): width of distribution on either side of 1
        randseed (int): seed passed to the reseed numpy's legacy MT19937 BitGenerator
        normal (bool):  whether to use a normal distribution instead of uniform

    **Example**::

        sc.perturb(5, 0.3) # Returns e.g. array([0.73852362, 0.7088094 , 0.93713658, 1.13150755, 0.87183371])
    
    *New in version 3.0.0:* Uses a separate random number stream
    """
    rng = np.random.default_rng(randseed)
    if normal:
        output = 1.0 + rng.normal(0, span, size=n)
    else:
        output = 1.0 + rng.uniform(-span, span, size=n)
    return output


def normsum(arr, total=None):
    """
    Multiply a list or array by some normalizing factor so that its sum is equal
    to the total. Formerly called "``scaleratio``".

    Args:
        arr (array): array (or list) to normalize
        total (float): amount to sum to (default 1)

    **Example**::

        normarr = sc.normsum([2,5,3,10], 100) # Scale so sum equals 100; returns [10.0, 25.0, 15.0, 50.0]

    Renamed in version 1.0.0.
    """
    if total is None: total = 1.0
    origtotal = float(sum(arr))
    ratio = float(total)/origtotal
    out = np.array(arr)*ratio
    if isinstance(arr, list): out = out.tolist() # Preserve type
    return out


def normalize(arr, minval=0.0, maxval=1.0):
    """
    Rescale an array between a minimum value and a maximum value.

    Args:
        arr (array): array to normalize
        minval (float): minimum value in rescaled array
        maxval (float): maximum value in rescaled array

    **Example**::

        normarr = sc.normalize([2,3,7,27]) # Returns array([0.  , 0.04, 0.2 , 1.  ])
    """
    out = np.array(arr, dtype=float) # Ensure it's a float so divide works
    out -= out.min()
    out /= out.max()
    out *= (maxval - minval)
    out += minval
    if isinstance(arr, list): out = out.tolist() # Preserve type
    return out


def inclusiverange(*args, stretch=False, **kwargs):
    """
    Like :func:`numpy.arange`/`numpy.linspace`, but includes the start and stop points.
    Accepts 0-3 args, or the kwargs start, stop, step.

    In most cases, equivalent to ``np.linspace(start, stop, int((stop-start)/step)+1)``.

    Args:
        start (float): value to start at
        stop (float): value to stop at
        step (float): step size
        stretch (bool): if True, adjust the step size to end exactly at stop if needed
        kwargs (dict): passed to :func:`numpy.linspace`

    **Examples**::

        x = sc.inclusiverange(10)        # Like np.arange(11)
        x = sc.inclusiverange(3,5,0.2)   # Like np.linspace(3, 5, int((5-3)/0.2+1))
        x = sc.inclusiverange(stop=5)    # Like np.arange(6)
        x = sc.inclusiverange(6, step=2) # Like np.arange(0, 7, 2)
        x = sc.inclusiverange(0, 10, 3) # Like np.arange(0, 10, 3)
        x = sc.inclusiverange(0, 10, 3, stretch=True) # Like np.linspace(0,10,int(10/3)+1)
        
    | *New in version 3.2.0*: "stretch" argument
    """
    # Handle args
    if len(args) == 0:
        start, stop, step = None, None, None
    elif len(args) == 1:
        stop = args[0]
        start, step = None, None
    elif len(args) == 2:
        start = args[0]
        stop  = args[1]
        step =  None
    elif len(args) == 3:
        start = args[0]
        stop  = args[1]
        step  = args[2]
    else: # pragma: no cover
        errormsg = f'Too many arguments supplied ({len(args)}): sc.inclusiverange() accepts 1-3 arguments'
        raise ValueError(errormsg)

    # Handle kwargs
    start = kwargs.pop('start', start)
    stop  = kwargs.pop('stop',  stop)
    step  = kwargs.pop('step',  step)

    # Finalize defaults
    if start is None: start = 0
    if stop  is None: stop  = 1
    if step  is None: step  = 1
    
    # Handle case with a non-integer number of steps
    nsteps = (stop-start)/step
    int_steps = int(nsteps)
    if not nsteps.is_integer() and not stretch:
        stop = start + step*int_steps # Create a new stop based on the step

    # Actually generate -- can't use arange since handles floating point arithmetic badly, e.g. compare arange(2000, 2020, 0.2) with arange(2000, 2020.2, 0.2)
    num = int_steps + 1 # +1 since include both endpoints
    x = np.linspace(start, stop, num, **kwargs)
    return x


def randround(x):
    """
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

    | *New in version 1.0.0.*
    | *New in version 3.0.0:* allow arrays of arbitrary shape
    """
    if isinstance(x, np.ndarray):
        output = np.array(np.floor(x+np.random.random(x.shape)), dtype=int)
    elif isinstance(x, list):
        output = [randround(i) for i in x]
    else:
        output = int(np.floor(x+np.random.random()))
    return output


def cat(*args, copy=False, **kwargs):
    """
    Like :func:`numpy.concatenate`, but takes anything and returns an array. Useful for
    e.g. appending a single number onto the beginning or end of an array.

    Args:
        args   (any):  items to concatenate into an array
        kwargs (dict): passed to :func:`numpy.concatenate`

    **Examples**::

        arr = sc.cat(4, np.ones(3))
        arr = sc.cat(np.array([1,2,3]), [4,5], 6)
        arr = sc.cat(np.random.rand(2,4), np.random.rand(2,6), axis=1)

    | *New in version 1.0.0.*
    | *New in version 1.1.0:* "copy" and keyword arguments.
    | *New in version 2.0.2:* removed "copy" argument; changed default axis of 0; arguments passed to ``np.concatenate()``
    """
    
    if not len(args):
        return np.array([])
    arrs = [sc.toarray(arg) for arg in args] # Key step: convert everything to an array
    if arrs[0].ndim == 2: # Convert to 2D if first array is
        arrs = [np.atleast_2d(arr) for arr in arrs]
    output = np.concatenate(arrs, **kwargs)
    return output


def linregress(x, y, full=False, **kwargs):
    """
    Simple linear regression returning the line of best fit and R value. Similar
    to :func:`scipy.stats.linregress`` but simpler.
    
    Args:
        x (array): the x coordinates
        y (array): the y coordinates
        full (bool): whether to return a full data structure
        kwargs (dict): passed to :func:`numpy.polyfit`
    
    **Examples**::
        
        x = range(10)
        y = sorted(2*np.random.rand(10) + 1)
        m,b = sc.linregress(x, y) # Simple usage
        out = sc.linregress(x, y, full=True) # Has out.m, out.b, out.x, out.y, out.corr, etc.
        plt.scatter(x, y)
        plt.plot(x, m*x+b)
        plt.bar(x, out.residuals)
        plt.title(f'R² = {out.r2}')
    """
    x = sc.toarray(x)
    y = sc.toarray(y)
    fit = np.polyfit(x, y, deg=1, **kwargs) # Do the fit
    if not full: # pragma: no cover
        return fit
    else:
        out = sc.objdict()
        out.m = fit[0] # Slope
        out.b = fit[-1] # Intercept
        out.coeffs = fit
        out.corr = np.corrcoef(x, y)[0,1]
        out.r2 = out.corr**2
        out.x = x
        out.y = out.m*x + out.b
        out.residuals = y - out.y
        return out


def sem(a, axis=None, *args, **kwargs):
    """
    Calculate the standard error of the mean (SEM).
    
    Shortcut (for a 1D array) to ``array.std()/np.sqrt(len(array))``.

    Args:
        a (arr): array to calculate the SEM of
        axis (int): axis to calculate the SEM along
        kwargs (dict): passed to :func:`numpy.std`

    **Example**::

        data = np.random.randn(100)
        sem = sc.sem(data) # Roughly 0.1
        
    | *New in version 3.2.0.*
    """
    a = sc.toarray(a)
    std = a.std(axis=axis)
    if axis is None:
        n = a.size
    elif sc.isnumber(axis):
        n = a.shape[axis]
    else:
        n = np.prod([a.shape[s] for s in axis])
    out = std/np.sqrt(n)
    return out


##############################################################################
#%% Smoothing functions
##############################################################################


__all__ += ['rolling', 'convolve', 'smooth', 'smoothinterp', 'gauss1d', 'gauss2d']


def rolling(data, window=7, operation='mean', replacenans=None, **kwargs):
    """
    Alias to :meth:`pandas.Series.rolling()` (window) method to smooth a series.

    Args:
        data (list/arr): the 1D or 2D data to be smoothed
        window (int): the length of the window
        operation (str): the operation to perform: 'mean' (default), 'median', 'sum', or 'none'
        replacenans (bool/float): if None, leave NaNs; if False, remove them; if a value, replace with that value; if the string 'nearest' or 'linear', do interpolation (see :func:`sc.rmnans() <rmnans>` for details)
        kwargs (dict): passed to :meth:`pandas.Series.rolling()`

    **Example**::

        data = [5,5,5,0,0,0,0,7,7,7,7,0,0,3,3,3]
        rolled = sc.rolling(data, replacenans='nearest')
    """
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
    
    if replacenans is None:
        pass
    else:
        output = sanitize(data=output, replacenans=replacenans)

    return output


def convolve(a, v):
    """
    Like :func:`numpy.convolve`, but always returns an array the size of the first array
    (equivalent to mode='same'), and solves the boundary problem present in :func:`numpy.convolve`
    by adjusting the edges by the weight of the convolution kernel.

    Args:
        a (arr): the input array
        v (arr): the convolution kernel

    **Example**::

        a = np.ones(5)
        v = np.array([0.3, 0.5, 0.2])
        c1 = np.convolve(a, v, mode='same') # Returns array([0.8, 1.  , 1.  , 1.  , 0.7])
        c2 = sc.convolve(a, v)              # Returns array([1., 1., 1., 1., 1.])

    | *New in version 1.3.0.*
    | *New in version 1.3.1:* handling the case where len(a) < len(v)
    """

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
    if len_diff: # pragma: no cover
        lhs_trim = len_diff // 2
        out = out[lhs_trim:lhs_trim+len_a]

    return out


def smooth(data, repeats=None, kernel=None, legacy=False):
    """
    Very simple function to smooth a 1D or 2D array.

    See also :func:`sc.gauss1d() <gauss1d>` for simple Gaussian smoothing.

    Args:
        data (arr): 1D or 2D array to smooth
        repeats (int): number of times to apply smoothing (by default, scale to be 1/5th the length of data)
        kernel (arr): the smoothing kernel to use (default: ``[0.25, 0.5, 0.25]``)
        legacy (bool): if True, use the old (pre-1.3.0) method of calculation that doesn't correct for edge effects

    **Example**::

        data = np.random.randn(5,5)
        smoothdata = sc.smooth(data)

    *New in version 1.3.0:* Fix edge effects.
    """
    if repeats is None:
        repeats = int(np.floor(len(data)/5))
    if kernel is None:
        kernel = [0.25,0.5,0.25]
    kernel = np.array(kernel)
    output = np.array(data).copy()

    # Only convolve the kernel with itself -- equivalent to doing the full convolution multiple times
    v = kernel.copy()
    for r in range(repeats-1):
        v = np.convolve(v, kernel, mode='full')

    # Support legacy method of not correcting for edge effects
    if legacy: # pragma: no cover
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


def smoothinterp(newx=None, origx=None, origy=None, smoothness=None, growth=None, 
                 ensurefinite=True, keepends=True, method='linear'):
    """
    Smoothly interpolate over values
    
    Unlike :func:`np.interp() <numpy.interp>`, this function does exactly pass 
    through each data point: 

    Args:
        newx (arr): the points at which to interpolate
        origx (arr): the original x coordinates
        origy (arr): the original y coordinates
        smoothness (float): how much to smooth
        growth (float): the growth rate to apply past the ends of the data
        ensurefinite (bool):  ensure all values are finite (including skipping NaNs)
        method (str): the type of interpolation to use (options are 'linear' or 'nearest')

    Returns:
        newy (arr): the new y coordinates

    **Example**::

        import sciris as sc
        import numpy as np
        from scipy import interpolate
        
        origy = np.array([0,0.2,0.1,0.9,0.7,0.8,0.95,1])
        origx = np.linspace(0,1,len(origy))
        newx = np.linspace(0,1,5*len(origy))
        sc_y = sc.smoothinterp(newx, origx, origy, smoothness=5)
        np_y = np.interp(newx, origx, origy)
        si_y = interpolate.interp1d(origx, origy, 'cubic')(newx)
        kw = dict(lw=3, alpha=0.7)
        plt.plot(newx, np_y, '--', label='NumPy', **kw)
        plt.plot(newx, si_y, ':',  label='SciPy', **kw)
        plt.plot(newx, sc_y, '-',  label='Sciris', **kw)
        plt.scatter(origx, origy, s=50, c='k', label='Data')
        plt.legend()
        plt.show()

    | *New in verison 3.0.0:* "ensurefinite" now defaults to True; removed "skipnans" argument
    """
    # Ensure arrays and remove NaNs
    if sc.isnumber(newx):  newx = [newx] # Make sure it has dimension
    if sc.isnumber(origx): origx = [origx] # Make sure it has dimension
    if sc.isnumber(origy): origy = [origy] # Make sure it has dimension
    newx  = np.array(newx, dtype=float)
    origx = np.array(origx, dtype=float)
    origy = np.array(origy, dtype=float)

    # If only a single element, just return it, without checking everything else
    if len(origy)==1: # pragma: no cover
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
    """
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
        import numpy as np
        import matplotlib.pyplot as plt
        import sciris as sc

        x = np.random.rand(40)
        y = (x-0.3)**2 + 0.2*np.random.rand(40)

        # Smooth
        yi = sc.gauss1d(x, y)
        yi2 = sc.gauss1d(x, y, scale=0.3)
        xi3 = np.linspace(0,1)
        yi3 = sc.gauss1d(x, y, xi)

        # Plot original and interpolated versions
        plt.scatter(x, y,     label='Original')
        plt.scatter(x, yi,    label='Default smoothing')
        plt.scatter(x, yi2,   label='More smoothing')
        plt.scatter(xi3, yi3, label='Uniform spacing')
        plt.show()

        # Simple usage
        sc.gauss1d(y)

    *New in version 1.3.0.*
    """

    # Swap inputs if x is provided but not y
    if y is None and x is not None: # pragma: no cover
        y,x = x,y
    if x is None: # pragma: no cover
        x = np.arange(len(y))
    if xi is None:
        xi = x

    # Convert to arrays
    try:
        orig_dtype = y.dtype
    except: # pragma: no cover
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
        """ Calculate the calculation """
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
    """
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
        import numpy as np
        import matplotlib.pyplot as plt
        
        x = np.random.rand(40)
        y = np.random.rand(40)
        z = 1-(x-0.5)**2 + (y-0.5)**2 # Make a saddle

        # Simple usage -- only works if z is 2D
        zi0 = sc.gauss2d(np.random.rand(10,10))
        sc.surf3d(zi0)

        # Method 1 -- form grid
        xi = np.linspace(0,1,20)
        yi = np.linspace(0,1,20)
        zi = sc.gauss2d(x, y, z, xi, yi, scale=0.1, grid=True)

        # Method 2 -- use points directly
        xi2 = np.random.rand(400)
        yi2 = np.random.rand(400)
        zi2 = sc.gauss2d(x, y, z, xi2, yi2, scale=0.1)

        # Plot oiginal and interpolated versions
        sc.scatter3d(x, y, z, c=z)
        sc.surf3d(zi)
        sc.scatter3d(xi2, yi2, zi2, c=zi2)
        plt.show()

    | *New in version 1.3.0.*
    | *New in version 1.3.1:* default arguments; support for 2D inputs
    """
    # Swap variables if needed
    if z is None and x is not None: # pragma: no cover
        z,x = x,z
    if x is None or y is None: # pragma: no cover
        if z.ndim != 2:
            errormsg = f'If the x and y axes are not provided, then z must be 2D, not {z.ndim}D'
            raise ValueError(errormsg)
        else:
            x = np.arange(z.shape[1]) # It's counterintuitive, but x is the 1st dimension, not the 0th
            y = np.arange(z.shape[0])

    # Handle shapes
    if z.ndim == 2 and (x.ndim == 1) and (y.ndim == 1): # pragma: no cover
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
    if xi is None: xi = sc.dcp(x)
    if yi is None: yi = sc.dcp(y)
    if use32:
        x, y, z, xi, yi = _arr32(x), _arr32(y), _arr32(z), _arr32(xi), _arr32(yi)
        scale, xscale, yscale = _f32(scale), _f32(xscale), _f32(yscale)
    xsc = xscale*scale
    ysc = yscale*scale

    # Now that we have xi and yi, handle more 1D vs 2D logic
    if xi.ndim == 1 and yi.ndim == 1 and grid:
        xi, yi = np.meshgrid(xi, yi)
    if xi.shape != yi.shape: # pragma: no cover
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
    if len(x) != len(y) != len(z): # pragma: no cover
        errormsg = f'Input arrays do not have the same number of elements: x = {len(x)}, y = {len(y)}, z = {len(z)}'
        raise ValueError(errormsg)
    if len(xi) != len(yi): # pragma: no cover
        errormsg = f'Output arrays do not have the same number of elements: xi = {len(xi)}, yi = {len(yi)}'
        raise ValueError(errormsg)

    def calc(xi, yi):
        """ Calculate the calculation """
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