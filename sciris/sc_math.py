"""
Version:
"""



import numpy as np
from . import sc_utils as ut

##############################################################################
### FIND AND APPROXIMATION FUNCTIONS
##############################################################################

__all__ = ['approx', 'safedivide', 'findinds', 'findnearest', 'dataindex', 'getvalidinds', 'sanitize', 'getvaliddata', 'isprime']


def approx(val1=None, val2=None, eps=None):
    '''
    Determine whether two scalars approximately match. Example:
        sc.approx(2*6, 11.9999999, eps=1e-6) # Returns True
        sc.approx([3,12,11.9], 12) # Returns array([False, True, False], dtype=bool)
    '''
    if val2 is None: val2 = 0.0
    if eps  is None: eps = 1e-9
    if isinstance(val1, list): val1 = np.array(val1) # If it's a list, convert to an array first
    output = abs(val1-val2)<=eps
    return output



def safedivide(numerator=None, denominator=None, default=None, eps=None, warn=False):
    '''
    Handle divide-by-zero and divide-by-nan elegantly. Examples:
        sc.safedivide(numerator=0, denominator=0, default=1, eps=0) # Returns 1
        sc.safedivide(numerator=5, denominator=2.0, default=1, eps=1e-3) # Returns 2.5
        sc.safedivide(3,array([1,3,0]),-1, warn=True) # Returns array([ 3,  1, -1])
    '''
    # Set some defaults
    if numerator   is None: numerator   = 1.0
    if denominator is None: denominator = 1.0
    if default     is None: default     = 0.0
    
    # Handle the logic
    invalid = approx(denominator, 0.0, eps=eps)
    if ut.isnumber(denominator): # The denominator is a scalar
        if invalid:
            output = default
        else:
            output = numerator/denominator
    elif ut.checktype(denominator, 'array'):
        if not warn:
            denominator[invalid] = 1.0 # Replace invalid values with 1 
        output = numerator/denominator
        output[invalid] = default
    else: # Unclear input, raise exception
        errormsg = 'Input type %s not understood: must be number or array' % type(denominator)
        raise Exception(errormsg)
        
    return output    
    


def findinds(val1, val2=None, eps=1e-6):
    '''
    Little function to find matches even if two things aren't eactly equal (eg. 
    due to floats vs. ints). If one argument, find nonzero values. With two arguments,
    check for equality using eps. Returns a tuple of arrays if val1 is multidimensional,
    else returns an array.
    
    Examples:
        findinds(rand(10)<0.5) # e.g. array([2, 4, 5, 9])
        findinds([2,3,6,3], 6) # e.g. array([2])
    
    Version: 2016jun06 
    '''
    if val2==None: # Check for equality
        output = np.nonzero(val1) # If not, just check the truth condition
    else:
        if ut.isstring(val2):
            output = np.nonzero(np.array(val1)==val2)
        else:
            output = np.nonzero(abs(np.array(val1)-val2)<eps) # If absolute difference between the two values is less than a certain amount
    if np.ndim(val1)==1: # Uni-dimensional
        output = output[0] # Return an array rather than a tuple of arrays if one-dimensional
    return output



def findnearest(series=None, value=None):
    '''
    Return the index of the nearest match in series to value -- like findinds, but
    always returns an object with the same type as value (i.e. findnearest with
    a number returns a number, findnearest with an array returns an array).
    
    Examples:
        findnearest(rand(10), 0.5) # returns whichever index is closest to 0.5
        findnearest([2,3,6,3], 6) # returns 2
        findnearest([2,3,6,3], 6) # returns 2
        findnearest([0,2,4,6,8,10], [3, 4, 5]) # returns array([1, 2, 2])
    
    Version: 2017jan07
    '''
    series = ut.promotetoarray(series)
    if ut.isnumber(value):
        output = np.argmin(abs(series-value))
    else:
        output = []
        for val in value: output.append(findnearest(series, val))
        output = ut.promotetoarray(output)
    return output
    
    
def dataindex(dataarray, index):        
    ''' Take an array of data and return either the first or last (or some other) non-NaN entry. '''
    
    nrows = np.shape(dataarray)[0] # See how many rows need to be filled (either npops, nprogs, or 1).
    output = np.zeros(nrows)       # Create structure
    for r in range(nrows): 
        output[r] = sanitize(dataarray[r])[index] # Return the specified index -- usually either the first [0] or last [-1]
    
    return output


def getvalidinds(data=None, filterdata=None):
    '''
    Return the years that are valid based on the validity of the input data from an arbitrary number
    of 1-D vector inputs. Warning, closely related to getvaliddata()!
    
    Example:
        getvalidinds([3,5,8,13], [2000, nan, nan, 2004]) # Returns array([0,3])
    '''
    data = ut.promotetoarray(data)
    if filterdata is None: filterdata = data # So it can work on a single input -- more or less replicates sanitize() then
    filterdata = ut.promotetoarray(filterdata)
    if filterdata.dtype=='bool': filterindices = filterdata # It's already boolean, so leave it as is
    else:                        filterindices = findinds(~np.isnan(filterdata)) # Else, assume it's nans that need to be removed
    dataindices = findinds(~np.isnan(data)) # Also check validity of data
    validindices = np.intersect1d(dataindices, filterindices)
    return validindices # Only return indices -- WARNING, not consistent with sanitize()



def sanitize(data=None, returninds=False, replacenans=None, die=True, defaultval=None, label=None, verbose=True):
        '''
        Sanitize input to remove NaNs. Warning, does not work on multidimensional data!!
        
        Examples:
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
                    sanitized = ut.dcp(data)
                    sanitized[naninds] = replacenans
            if len(sanitized)==0:
                if defaultval is not None:
                    sanitized = defaultval
                else:
                    sanitized = 0.0
                    if verbose:
                        if label is None: label = 'this parameter'
                        print('sanitize(): no data entered for %s, assuming 0' % label)
        except Exception as E:
            if die: 
                raise Exception('Sanitization failed on array: "%s":\n %s' % (repr(E), data))
            else:
                sanitized = data # Give up and just return an empty array
                inds = []
        if returninds: return sanitized, inds
        else:          return sanitized


def getvaliddata(data=None, filterdata=None, defaultind=0):
    '''
    Return the years that are valid based on the validity of the input data.
    
    Example:
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
            raise Exception('Array sizes are mismatched: %i vs. %i' % (len(data), len(validindices)))    
    else: 
        validdata = np.array([]) # No valid data, return an empty array
    return validdata


def isprime(n, verbose=False):
    ''' From https://stackoverflow.com/questions/15285534/isprime-function-for-python-language '''
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
            if verbose: print('Not prime: divisible by %s' % f)
            return False
        if n%(f+2) == 0:
            if verbose: print('Not prime: divisible by %s' % (f+2))
            return False
        f +=6
    if verbose: print('Is prime!')
    return True 



##############################################################################
### OTHER FUNCTIONS
##############################################################################

__all__ += ['quantile', 'perturb', 'scaleratio', 'normalize', 'inclusiverange', 'smooth', 'smoothinterp']


def quantile(data, quantiles=[0.5, 0.25, 0.75]):
    '''
    Custom function for calculating quantiles most efficiently for a given dataset.
        data = a list of arrays, or an array where he first dimension is to be sorted
        quantiles = a list of floats >=0 and <=1
    
    Version: 2014nov23
    '''
    nsamples = len(data) # Number of samples in the dataset
    indices = (np.array(quantiles)*(nsamples-1)).round().astype(int) # Calculate the indices to pull out
    output = np.array(data)
    output.sort(axis=0) # Do the actual sorting along the 
    output = output[indices] # Trim down to the desired quantiles
    
    return output



def perturb(n=1, span=0.5, randseed=None):
    ''' Define an array of numbers uniformly perturbed with a mean of 1. n = number of points; span = width of distribution on either side of 1.'''
    if randseed is not None: np.random.seed(int(randseed)) # Optionally reset random seed
    output = 1. + 2*span*(np.random.rand(n)-0.5)
    return output
    
    
    
def scaleratio(inarray, total=None):
    ''' Multiply a list or array by some factor so that its sum is equal to the total. '''
    if total is None: total = 1.0
    origtotal = float(sum(inarray))
    ratio = total/origtotal
    outarray = np.array(inarray)*ratio
    if type(inarray)==list: outarray = outarray.tolist() # Preserve type
    return outarray



def normalize(inarray, minval=0.0, maxval=1.0):
    ''' Rescale an array between a minimum value and a maximum value '''
    outarray = np.array(inarray)
    outarray -= outarray.min()
    outarray /= outarray.max()
    outarray *= (maxval - minval)
    outarray += minval
    return outarray



def inclusiverange(*args, **kwargs):
    '''
    Like arange/linspace, but includes the start and stop points. 
    Accepts 0-3 args, or the kwargs start, stop, step. Examples:
    
    x = inclusiverange(3,5,0.2)
    x = inclusiverange(stop=5)
    x = inclusiverange(6, step=2)
    '''
    
    # Handle args
    if len(args)==0:
        start, stop, step = None, None, None
    elif len(args)==1:
        stop = args[0]
        start, step = None
    elif len(args)==2:
        start = args[0]
        stop   = args[1]
        step = None
    elif len(args)==3:
        start = args[0]
        stop = args[1]
        step = args[2]
    else:
        raise Exception('Too many arguments supplied: inclusiverange() accepts 0-3 arguments')
    
    # Handle kwargs
    start = kwargs.get('start', start)
    stop  = kwargs.get('stop',  stop)
    step  = kwargs.get('step',  step)
    
    # Finalize defaults
    if start is None: start = 0
    if stop  is None: stop  = 1
    if step  is None: step  = 1
    
    # OK, actually generate
    x = np.linspace(start, stop, int(round((stop-start)/float(step))+1)) # Can't use arange since handles floating point arithmetic badly, e.g. compare arange(2000, 2020, 0.2) with arange(2000, 2020.2, 0.2)
    
    return x



def smooth(data, repeats=None):
    ''' Very crude function to smooth a 2D array -- very slow but simple and easy to use '''
    if repeats is None:
        repeats = int(np.floor(len(data)/5))
    output = np.array(data)
    kernel = np.array([0.25,0.5,0.25])
    for r in range(repeats):
        if output.ndim == 1:
            output = np.convolve(output, kernel, mode='same')
        elif output.ndim == 2:
            for i in range(output.shape[0]): output[i,:] = np.convolve(output[i,:], kernel, mode='same')
            for j in range(output.shape[1]): output[:,j] = np.convolve(output[:,j], kernel, mode='same')
        else:
            errormsg = 'Simple smooting only implemented for 1D and 2D arrays'
            raise Exception(errormsg)
    return output



def smoothinterp(newx=None, origx=None, origy=None, smoothness=None, growth=None, ensurefinite=False, keepends=True, method='linear'):
    '''
    Smoothly interpolate over values and keep end points. Same format as numpy.interp.
    
    Example:
        from utils import smoothinterp
        origy = array([0,0.1,0.3,0.8,0.7,0.9,0.95,1])
        origx = linspace(0,1,len(origy))
        newx = linspace(0,1,5*len(origy))
        newy = smoothinterp(newx, origx, origy, smoothness=5)
        plot(newx,newy)
        hold(True)
        scatter(origx,origy)
    
    Version: 2018jan24
    '''
    # Ensure arrays and remove NaNs
    if ut.isnumber(newx):  newx = [newx] # Make sure it has dimension
    if ut.isnumber(origx): origx = [origx] # Make sure it has dimension
    if ut.isnumber(origy): origy = [origy] # Make sure it has dimension
    newx  = np.array(newx, dtype=float)
    origx = np.array(origx, dtype=float)
    origy = np.array(origy, dtype=float)
    
    # If only a single element, just return it, without checking everything else
    if len(origy)==1: 
        newy = np.zeros(newx.shape)+origy[0]
        return newy
    
    if not(newx.shape): raise Exception('To interpolate, must have at least one new x value to interpolate to')
    if not(origx.shape): raise Exception('To interpolate, must have at least one original x value to interpolate to')
    if not(origy.shape): raise Exception('To interpolate, must have at least one original y value to interpolate to')
    if not(origx.shape==origy.shape): 
        errormsg = 'To interpolate, original x and y vectors must be same length (x=%i, y=%i)' % (len(origx), len(origy))
        raise Exception(errormsg)
    
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
    else:
        raise Exception('Method "%s" not found; methods are "linear" or "nearest"' % method)

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
            if not keepends:
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
    if growth is not None:
        pastindices = findinds(newx<origx[0])
        futureindices = findinds(newx>origx[-1])
        if len(pastindices): # If there are past data points
            firstpoint = pastindices[-1]+1
            newy[pastindices] = newy[firstpoint] * np.exp((newx[pastindices]-newx[firstpoint])*growth) # Get last 'good' data point and apply inverse growth
        if len(futureindices): # If there are past data points
            lastpoint = futureindices[0]-1
            newy[futureindices] = newy[lastpoint] * np.exp((newx[futureindices]-newx[lastpoint])*growth) # Get last 'good' data point and apply growth
    
    # Add infinities back in, if they exist
    if any(~np.isfinite(origy)): # Infinities exist, need to add them back in manually since interp can only handle nan
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