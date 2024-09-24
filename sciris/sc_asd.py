"""
Adaptive stochastic descent optimization algorithm, building on :mod:`scipy.optimize`.

This algorithm is published as:

  Kerr CC, Dura-Bernal S, Smolinski TG, Chadderdon GL, Wilson DP (2018).
  **Optimization by Adaptive Stochastic Descent**. *PLoS ONE* 13(3): e0192944. 
  https://doi.org/10.1371/journal.pone.0192944
"""

import time
import numpy as np
import sciris as sc

__all__ = ['asd']


def asd(function, x, args=None, stepsize=0.1, sinc=2, sdec=2, pinc=2, pdec=2,
    pinitial=None, sinitial=None, xmin=None, xmax=None, maxiters=None, maxtime=None,
    abstol=1e-6, reltol=1e-3, stalliters=None, stoppingfunc=None, randseed=None,
    label=None, verbose=1, minval=0, **kwargs):
    """
    Optimization using adaptive stochastic descent (ASD). Can be used as a faster
    and more powerful alternative to e.g. :func:`scipy.optimize.minimize()`.

    ASD starts at ``x0`` and attempts to find a local minimizer ``x`` of the function ``func()``.
    ``func()`` accepts input ``x`` and returns a scalar function value evaluated at ``x``.
    ``x0`` can be a scalar, list, or Numpy array of any size.

    Args:
        function     (func):  The function to minimize
        x            (arr):   The vector of initial parameters
        args         (any):   List, tuple, or dictionary of additional parameters to be passed to the function
        kwargs       (dict):  Additional keywords passed to the function
        stepsize     (0.1):   Initial step size as a fraction of each parameter
        sinc         (2):     Step size learning rate (increase)
        sdec         (2):     Step size learning rate (decrease)
        pinc         (2):     Parameter selection learning rate (increase)
        pdec         (2):     Parameter selection learning rate (decrease)
        pinitial     (None):  Set initial parameter selection probabilities
        sinitial     (None):  Set initial step sizes; if empty, calculated from stepsize instead
        xmin         (None):  Min value allowed for each parameter
        xmax         (None):  Max value allowed for each parameter
        maxiters     (1000):  Maximum number of iterations (1 iteration = 1 function evaluation)
        maxtime      (3600):  Maximum time allowed, in seconds
        abstol       (1e-6):  Minimum absolute change in objective function
        reltol       (1e-3):  Minimum relative change in objective function
        stalliters   (10*n):  Number of iterations over which to calculate TolFun (n = number of parameters)
        stoppingfunc (None):  External method that can be used to stop the calculation from the outside.
        randseed     (None):  The random seed to use
        label        (None):  A label to use to annotate the output
        verbose      (1):     How much information to print during the run (max 3); less than one will print out once every 1/verbose steps
        minval       (0):     Minimum value the objective function can take

    Returns:
        objdict (see below)

    The returned object is an ``objdict``, which can be accessed by index, key,
    or attribute. Its keys/attributes are:

        - ``x``          -- The parameter set that minimizes the objective function
        - ``fval``       -- The value of the objective function at the final iteration
        - ``exitreason`` -- Why the algorithm terminated;
        - ``details``    -- See below

    The ``details`` key consists of:

        - ``fvals``         -- The value of the objective function at each iteration
        - ``xvals``         -- The parameter values at each iteration;
        - ``probabilities`` -- The probability of each step; and
        - ``stepsizes``     -- The size of each step for each parameter.

    **Examples**::

        # Basic usage
        import numpy as np
        import sciris as sc
        result = sc.asd(np.linalg.norm, [1, 2, 3])
        print(result.x)

        # With arguments
        def my_func(x, scale=1.0, weight=1.0): # Example function with keywords
            return abs((x[0] - 1)) + abs(x[1] + 2)*scale + abs(x[2] + 3)*weight

        result = sc.asd(my_func, x=[0, 0, 1], args=[0.5, 0.1]) # Option 1 for passing arguments
        result = sc.asd(my_func, x=[0, 0, 1], args=dict(scale=0.5, weight=0.1)) # Option 1 for passing arguments
        result = sc.asd(my_func, x=[0, 0, 1], scale=0.5, weight=0.1) # Option 2 for passing arguments

    Please use the following citation for this method:

        CC Kerr, S Dura-Bernal, TG Smolinski, GL Chadderdon, DP Wilson (2018).
        Optimization by adaptive stochastic descent.
        PLOS ONE 13 (3), e0192944.

    | *New in version 3.0.0:* Uses its own random number stream
    """
    rng = np.random.default_rng(randseed)
    if verbose >= 2: print(f'ASD: Launching with random seed {randseed}')

    def consistentshape(userinput, origshape=False):
        """
        Make sure inputs have the right shape and data type.
        """
        output = np.reshape(np.array(userinput, dtype='float'), -1)
        if origshape: return output, np.shape(userinput)
        else:         return output

    # Handle inputs and set defaults
    if maxtime  is None: maxtime  = 3600
    if maxiters is None: maxiters = 1000
    maxrangeiters = 100 # Number of times to try generating a new parameter
    x, origshape = consistentshape(x, origshape=True) # Turn it into a vector but keep the original shape (not necessarily class, though)
    nparams = len(x) # Number of parameters
    if not nparams: # pragma: no cover
        errormsg = 'ASD: The length of the input vector cannot be zero'
        raise ValueError(errormsg)
    if sinc<1: # pragma: no cover
        print('ASD: sinc cannot be less than 1; resetting to 2')
        sinc = 2
    if sdec<1: # pragma: no cover
        print('ASD: sdec cannot be less than 1; resetting to 2')
        sdec = 2
    if pinc<1: # pragma: no cover
        print('ASD: pinc cannot be less than 1; resetting to 2')
        pinc = 2
    if pdec<1: # pragma: no cover
        print('ASD: pdec cannot be less than 1; resetting to 2')
        pdec = 2

    # Set initial parameter selection probabilities -- uniform by default
    if pinitial is None: probabilities = np.ones(2 * nparams)
    else:                probabilities = consistentshape(pinitial)
    if not sum(probabilities): # pragma: no cover
        errormsg = 'ASD: The sum of input probabilities cannot be zero'
        raise ValueError(errormsg)

    # Handle step sizes
    if sinitial is None:
        stepsizes = abs(stepsize * x)
        stepsizes = np.concatenate((stepsizes, stepsizes)) # need to duplicate since two for each parameter
    else:
        stepsizes = consistentshape(sinitial)

    # Handle x limits
    xmin = np.zeros(nparams) - np.inf if xmin is None else consistentshape(xmin)
    xmax = np.zeros(nparams) + np.inf if xmax is None else consistentshape(xmax)

    # Final input checking
    if sum(np.isnan(x)): # pragma: no cover
        errormsg = f'ASD: At least one value in the vector of starting points is NaN:\n{x}'
        raise ValueError(errormsg)
    if label is None: label = ''
    if stalliters is None: stalliters = 10 * nparams # By default, try 10 times per parameter on average
    stalliters = int(stalliters)
    maxiters = int(maxiters)

    # Initialization
    if all(stepsizes == 0): stepsizes += stepsize # Handle the case where all step sizes are 0
    if any(stepsizes == 0): stepsizes[stepsizes == 0] = np.mean(stepsizes[stepsizes != 0]) # Replace step sizes of zeros with the mean of non-zero entries
    if args is None: # Reset if no function arguments supplied
        args = []
    elif isinstance(args, dict): # It's actually kwargs supplied
        kwargs = sc.mergedicts(args, kwargs)
        args = []
    fval = function(x, *args, **kwargs) # Calculate initial value of the objective function
    fvalorig = fval # Store the original value of the objective function, since fval is overwritten on each step
    xorig = x.copy() # Keep the original x, just in case

    # Initialize history
    abserrorhistory = np.zeros(stalliters) # Store previous error changes
    relerrorhistory = np.zeros(stalliters) # Store previous error changes
    fvals = np.zeros(maxiters + 1) # Store all objective function values
    allsteps = np.zeros((maxiters + 1, nparams)) # Store all parameters
    fvals[0] = fvalorig # Store initial function output
    allsteps[0, :] = xorig # Store initial input vector

    # Prepare for the loop
    count = 0 # Keep track of how many iterations have occurred
    start = time.time() # Keep track of when we begin looping
    offset = ' ' * 4 # Offset the print statements
    exitreason = 'Unknown exit reason' # Catch everything else
    
    # Loop
    while True:
        
        # Handle initialization cases
        if fvalorig == minval:
            exitreason = f'Objective function already at minimum value ({fvalorig}), skipping optimization'
            break
        elif fvalorig < 0 and verbose:
            print(f'ASD: Warning, negative objective function starting value ({fvalorig:n}) could lead to unexpected behavior')

        # Begin the loop
        count += 1 # Increment the count
        if verbose >= 3: print(f'\n\n Count={count} \n x={x} \n probabilities={probabilities} \n stepsizes={stepsizes}')

        # Calculate next parameters
        probabilities = probabilities / sum(probabilities) # Normalize probabilities
        cumprobs = np.cumsum(probabilities) # Calculate the cumulative distribution
        inrange = False
        for r in range(maxrangeiters): # Try to find parameters within range
            choice = np.flatnonzero(cumprobs > rng.random())[0] # Choose a parameter and upper/lower at random
            par = np.mod(choice, nparams) # Which parameter was chosen
            pm = np.floor((choice) / nparams) # Plus or minus
            newval = x[par] + ((-1)**pm) * stepsizes[choice] # Calculate the new vector
            if newval<xmin[par]: newval = xmin[par] # Reset to the lower limit
            if newval>xmax[par]: newval = xmax[par] # Reset to the upper limit
            inrange = (newval != x[par])
            if verbose >= 3: print(offset*2 + f'count={count} r={r}, choice={choice}, par={par}, x[par]={x[par]}, pm={(-1)**pm}, step={stepsizes[choice]}, newval={newval}, xmin={xmin[par]}, xmax={xmax[par]}, inrange={inrange}')
            if inrange: # Proceed as long as they're not equal
                break
        if not inrange: # Treat it as a failure if a value in range can't be found
            probabilities[choice] = probabilities[choice] / pdec
            stepsizes[choice] = stepsizes[choice] / sdec

        # Calculate the new value
        xnew = x.copy() # Initialize the new parameter set
        xnew[par] = newval # Update the new parameter set
        fvalnew = function(xnew, *args, **kwargs) # Calculate the objective function for the new parameter set
        eps = 1e-12 # Small value to avoid divide-by-zero errors
        if abs(fvalnew)<eps and abs(fval)<eps: ratio = 1 # They're both zero: set the ratio to 1
        elif abs(fvalnew)<eps:                 ratio = 1.0/eps # Only the denominator is zero: reset to the maximum ratio
        else:                                  ratio = fval/float(fvalnew) # The normal situation: calculate the real ratio
        abserrorhistory[np.mod(count, stalliters)] = max(0, fval-fvalnew) # Keep track of improvements in the error
        relerrorhistory[np.mod(count, stalliters)] = max(0, ratio-1.0) # Keep track of improvements in the error
        if verbose >= 2: print(offset + f'step={count} choice={choice}, par={par}, pm={pm}, origval={x[par]}, newval={xnew[par]}')
        if newval < 0 and verbose:
            print(f'ASD: Warning, negative objective function ({newval:n}) on step {count} could lead to unexpected behavior')

        # Check if this step was an improvement
        fvalold = fval # Store old fval
        if fvalnew < fvalold: # New parameter set is better than previous one
            probabilities[choice] = probabilities[choice] * pinc # Increase probability of picking this parameter again
            stepsizes[choice] = stepsizes[choice] * sinc # Increase size of step for next time
            x = xnew # Reset current parameters
            fval = fvalnew # Reset current error
            flag = '++' # Marks an improvement
        else: # New parameter set is the same or worse than the previous one
            probabilities[choice] = probabilities[choice] / pdec # Decrease probability of picking this parameter again
            stepsizes[choice] = stepsizes[choice] / sdec # Decrease size of step for next time
            flag = '--' # Marks no change
            if np.isnan(fvalnew):
                if verbose >= 1: print('ASD: Warning, objective function returned NaN')
        if verbose > 0 and not (count % max(1, int(1.0/verbose))): # Print out every 1/verbose steps
            orig, best, new, diff = sc.sigfig([fvalorig, fvalold, fvalnew, fvalnew-fvalold])
            print(offset + label + f' step {count} ({time.time()-start:0.1f} s) {flag} (orig:{orig} | best:{best} | new:{new} | diff:{diff})')

        # Store output information
        fvals[count] = fval # Store objective function evaluations
        allsteps[count, :] = x # Store parameters

        # Stopping criteria
        if count >= maxiters: # Stop if the iteration limit is exceeded # pragma: no cover
            exitreason = 'Maximum iterations reached'
            break
        if (time.time() - start) > maxtime: # pragma: no cover
            strtime, strmax = sc.sigfig([(time.time()-start), maxtime])
            exitreason = f'Time limit reached ({strtime} > {strmax})'
            break
        if (count > stalliters) and (abs(np.mean(abserrorhistory)) < abstol): # Stop if improvement is too small
            strabs, strtol = sc.sigfig([np.mean(abserrorhistory), abstol])
            exitreason = f'Absolute improvement too small ({strabs} < {strtol})'
            break
        if (count > stalliters) and (sum(relerrorhistory) < reltol): # Stop if improvement is too small
            strrel, strtol = sc.sigfig([np.mean(relerrorhistory), reltol])
            exitreason = f'Relative improvement too small ({strrel} < {strtol})'
            break
        if stoppingfunc and stoppingfunc(): # pragma: no cover
            exitreason = 'Stopping function called'
            break
        if fval == minval:
            exitreason = f'Minimum objective function value reached ({fval})'
            break

    # Return
    if verbose > 0:
        orig, best = sc.sigfig([fvals[0], fvals[count]])
        eps = 1e-12 # Small value to avoid divide-by-zero errors
        if abs(fvals[count])<eps and abs(fvals[0])<eps:
            ratio = 1 # They're both zero: set the ratio to 1
        elif abs(fvals[0])<eps:
            ratio = 1.0/eps # Only the denominator is zero: reset to the maximum ratio
        else:
            ratio = fvals[count]/float(fvals[0]) # The normal situation: calculate the real ratio

        print(f'=== {label} {exitreason} ({count} steps, orig: {orig} | best: {best} | ratio: {ratio}) ===')
    
    output = sc.objdict()
    output['x'] = np.reshape(x, origshape) # Parameters
    output['fval'] = fvals[count]
    output['exitreason'] = exitreason
    output['details'] = sc.objdict()
    output['details']['fvals'] = fvals[:count+1] # Function evaluations
    output['details']['xvals'] = allsteps[:count+1, :]
    output['details']['probabilities'] = probabilities
    output['details']['stepsizes'] = stepsizes
    
    return output # Return parameter vector as well as details about run
