"""
Adaptive stochastic descent optimization algorithm, building on :mod:`scipy.optimize`.

This algorithm is published as:

  Kerr CC, Dura-Bernal S, Smolinski TG, Chadderdon GL, Wilson DP (2018).
  **Optimization by Adaptive Stochastic Descent**. *PLoS ONE* 13(3): e0192944.
  https://doi.org/10.1371/journal.pone.0192944
"""

import time
import warnings
import numpy as np
import sciris as sc

__all__ = ['asd']


def _consistent_shape(userinput, origshape=False):
    """ Ensure inputs have the right shape and data type. """
    output = np.reshape(np.array(userinput, dtype='float'), -1)
    if origshape:
        return output, np.shape(userinput)
    return output


def _improvement_ratio(fval_old, fval_new, eps=1e-12):
    """ Compute improvement ratio with divide-by-zero guard
    
    Returns fval_old/fval_new, or 1 when both near zero, or a large value when only fval_new is near zero.
    """
    if abs(fval_new) < eps and abs(fval_old) < eps:
        return 1.0
    if abs(fval_new) < eps:
        return 1.0 / eps
    return fval_old / float(fval_new)


def _validate_fval(fval, die=True):
    """ Validate that the objective function returns a scalar """
    if not sc.isnumber(fval):
        if isinstance(fval, np.ndarray) and fval.size == 1: # Automatically convert size-1 arrays to scalars
            fval = fval[0]
        else:
            errormsg = f'ASD: The objective function should return a scalar, not: {fval} (type: {type(fval)})'
            raise ValueError(errormsg)
    return fval


def asd(function, x, args=None, stepsize=0.1, sinc=2, sdec=2, pinc=2, pdec=2,
    pinitial=None, sinitial=None, xmin=None, xmax=None, maxiters=None, maxtime=None,
    abstol=1e-6, reltol=1e-3, stalliters=None, stoppingfunc=None, randseed=None,
    label=None, verbose=1, minval=0, die=True, **kwargs):
    """
    Optimization using adaptive stochastic descent (ASD). Can be used as a faster
    and more powerful alternative to e.g. :func:`scipy.optimize.minimize()`.

    ASD starts at ``x`` and attempts to find a local minimizer of the function ``function()``.
    ``function()`` accepts input ``x`` and returns a scalar function value evaluated at ``x``.
    ``x`` can be a scalar, list, or Numpy array of any size.

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
        die          (True):  If True, raise when the objective function raises; if False, treat that trial as np.inf and continue

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

        # With arguments: positional via args, or dict of keywords, or keyword arguments
        def my_func(x, scale=1.0, weight=1.0):  # Example function with keywords
            return abs((x[0] - 1)) + abs(x[1] + 2)*scale + abs(x[2] + 3)*weight

        result = sc.asd(my_func, x=[0, 0, 1], args=[0.5, 0.1]) # Option 1 for passing arguments
        result = sc.asd(my_func, x=[0, 0, 1], args=dict(scale=0.5, weight=0.1)) # Option 2 for passing arguments
        result = sc.asd(my_func, x=[0, 0, 1], scale=0.5, weight=0.1) # Option 3 for passing arguments

    Please use the following citation for this method:

        CC Kerr, S Dura-Bernal, TG Smolinski, GL Chadderdon, DP Wilson (2018).
        Optimization by adaptive stochastic descent.
        PLOS ONE 13 (3), e0192944.

    | *New in version 3.0.0:* Uses its own random number stream
    """
    rng = np.random.default_rng(randseed)
    if verbose >= 2: print(f'ASD: Launching with random seed {randseed}')

    # Reshape initial point and get parameter count
    x, origshape = _consistent_shape(x, origshape=True)
    nparams = len(x)
    maxrangeiters = 100  # Number of times to try generating a new parameter

    # Set defaults in one place
    if maxtime is None:
        maxtime = 3600
    if maxiters is None:
        maxiters = 1000
    if label is None:
        label = ''
    if stalliters is None:
        stalliters = 10 * nparams
    stalliters = int(stalliters)
    maxiters = int(maxiters)
    eps = 1e-12

    # Validate input vector length
    if not nparams:
        errormsg = 'ASD: The length of the input vector cannot be zero'
        raise ValueError(errormsg)

    # Validate learning rates (must be >= 1)
    if sinc < 1:
        errormsg = 'ASD: sinc cannot be less than 1; resetting to 2'
        if die:
            raise ValueError(errormsg)
        warnings.warn(errormsg)
        sinc = 2
    if sdec < 1:
        errormsg = 'ASD: sdec cannot be less than 1; resetting to 2'
        if die:
            raise ValueError(errormsg)
        warnings.warn(errormsg)
        sdec = 2
    if pinc < 1:
        errormsg = 'ASD: pinc cannot be less than 1; resetting to 2'
        if die:
            raise ValueError(errormsg)
        warnings.warn(errormsg)
        pinc = 2
    if pdec < 1:
        errormsg = 'ASD: pdec cannot be less than 1; resetting to 2'
        if die:
            raise ValueError(errormsg)
        warnings.warn(errormsg)
        pdec = 2

    # Set initial parameter selection probabilities (uniform by default)
    if pinitial is None:
        probabilities = np.ones(2 * nparams)
    else:
        probabilities = _consistent_shape(pinitial)
    if not sum(probabilities):
        errormsg = 'ASD: The sum of input probabilities cannot be zero'
        raise ValueError(errormsg)

    # Step sizes
    if sinitial is None:
        stepsizes = abs(stepsize * x)
        stepsizes = np.concatenate((stepsizes, stepsizes))  # Two entries per parameter (up/down)
    else:
        stepsizes = _consistent_shape(sinitial)

    # Parameter bounds
    xmin = np.zeros(nparams) - np.inf if xmin is None else _consistent_shape(xmin)
    xmax = np.zeros(nparams) + np.inf if xmax is None else _consistent_shape(xmax)

    # Reject NaN in starting point
    if sum(np.isnan(x)):
        errormsg = f'ASD: At least one value in the vector of starting points is NaN:\n{x}'
        raise ValueError(errormsg)

    # Initialization
    if all(stepsizes == 0):
        stepsizes += stepsize
    if any(stepsizes == 0):
        stepsizes[stepsizes == 0] = np.mean(stepsizes[stepsizes != 0])
    if args is None:
        args = []
    elif isinstance(args, dict):
        kwargs = sc.mergedicts(args, kwargs)
        args = []

    # Get initial function value
    fval = function(x, *args, **kwargs)
    fval = _validate_fval(fval, die=die)
    fvalorig = fval
    xorig = x.copy()

    # Allocate history arrays
    abserrorhistory = np.zeros(stalliters)
    relerrorhistory = np.zeros(stalliters)
    fvals = np.zeros(maxiters + 1)
    allsteps = np.zeros((maxiters + 1, nparams))
    fvals[0] = fvalorig
    allsteps[0, :] = xorig

    # Prepare for the loop
    count = 0 # Keep track of how many iterations have occurred
    start = time.time() # Keep track of when we begin looping
    offset = ' ' * 4 # Offset the print statements
    exitreason = 'Unknown exit reason' # Catch everything else

    # Main optimization loop
    while True:

        # Skip if already at minimum
        if fvalorig == minval:
            exitreason = f'Objective function already at minimum value ({fvalorig}), skipping optimization'
            break
        if fvalorig < 0 and verbose:
            print(f'ASD: Warning, negative objective function starting value ({fvalorig:n}) could lead to unexpected behavior')

        count += 1
        if verbose >= 3:
            print(f'\n\n Count={count} \n x={x} \n probabilities={probabilities} \n stepsizes={stepsizes}')

        # Normalize probabilities and sample a parameter/direction
        probabilities = probabilities / sum(probabilities)
        cumprobs = np.cumsum(probabilities)
        inrange = False
        for r in range(maxrangeiters): # Try to find parameters within range
            choice = np.flatnonzero(cumprobs > rng.random())[0] # Choose a parameter and upper/lower at random
            par = np.mod(choice, nparams) # Which parameter was chosen
            pm = np.floor((choice) / nparams) # Plus or minus
            newval = x[par] + ((-1)**pm) * stepsizes[choice] # Calculate the new vector
            if newval<xmin[par]: newval = xmin[par] # Reset to the lower limit
            if newval>xmax[par]: newval = xmax[par] # Reset to the upper limit
            inrange = (newval != x[par])
            if verbose >= 3:
                print(offset*2 + f'count={count} r={r}, choice={choice}, par={par}, x[par]={x[par]}, pm={(-1)**pm}, step={stepsizes[choice]}, newval={newval}, xmin={xmin[par]}, xmax={xmax[par]}, inrange={inrange}')
            if inrange:
                break
        if not inrange: # Treat it as a failure if a value in range can't be found
            probabilities[choice] = probabilities[choice] / pdec
            stepsizes[choice] = stepsizes[choice] / sdec

        # Evaluate objective at proposed point
        xnew = x.copy()
        xnew[par] = newval
        try:
            fvalnew = function(xnew, *args, **kwargs)
            fvalnew = _validate_fval(fvalnew, die=die)
        except Exception as e:
            if die:
                raise
            warnings.warn(f'ASD: Objective function raised on step {count}; treating trial as np.inf. Error: {e}')
            fvalnew = np.inf
        ratio = _improvement_ratio(fval, fvalnew, eps)

        # Update improvement history
        abserrorhistory[count % stalliters] = max(0, fval - fvalnew)
        relerrorhistory[count % stalliters] = max(0, ratio - 1.0)
        if verbose >= 2:
            print(offset + f'step={count} choice={choice}, par={par}, pm={pm}, origval={x[par]}, newval={xnew[par]}')
        if newval < 0 and verbose:
            print(f'ASD: Warning, negative objective function ({newval:n}) on step {count} could lead to unexpected behavior')

        # Accept or reject step and update learning state
        fvalold = fval
        if fvalnew < fvalold:
            probabilities[choice] = probabilities[choice] * pinc
            stepsizes[choice] = stepsizes[choice] * sinc
            x = xnew
            fval = fvalnew
            flag = '++'
        else:
            probabilities[choice] = probabilities[choice] / pdec
            stepsizes[choice] = stepsizes[choice] / sdec
            flag = '--'
            if np.isnan(fvalnew) and verbose >= 1:
                print('ASD: Warning, objective function returned NaN')
        if verbose > 0 and not (count % max(1, int(1.0/verbose))):
            orig, best, new, diff = sc.sigfig([fvalorig, fvalold, fvalnew, fvalnew - fvalold])
            print(offset + label + f' step {count} ({time.time()-start:0.1f} s) {flag} (orig:{orig} | best:{best} | new:{new} | diff:{diff})')

        # Record history for this iteration
        fvals[count] = fval
        allsteps[count, :] = x

        # Stopping criteria
        if count >= maxiters: # Stop if the iteration limit is exceeded # pragma: no cover
            exitreason = 'Maximum iterations reached'
            break
        if (time.time() - start) > maxtime: # Stop if the time limit is exceeded # pragma: no cover
            strtime, strmax = sc.sigfig([(time.time()-start), maxtime])
            exitreason = f'Time limit reached ({strtime} > {strmax})'
            break
        if (count > stalliters) and (abs(np.mean(abserrorhistory)) < abstol): # Stop if absolute improvement is too small
            strabs, strtol = sc.sigfig([np.mean(abserrorhistory), abstol])
            exitreason = f'Absolute improvement too small ({strabs} < {strtol})'
            break
        if (count > stalliters) and (sum(relerrorhistory) < reltol): # Stop if relativeimprovement is too small
            strrel, strtol = sc.sigfig([np.mean(relerrorhistory), reltol])
            exitreason = f'Relative improvement too small ({strrel} < {strtol})'
            break
        if stoppingfunc and stoppingfunc(): # Stop if explicitly requested # pragma: no cover
            exitreason = 'Stopping function called'
            break
        if fval == minval: # Stop if the objective function value is at the minimum
            exitreason = f'Minimum objective function value reached ({fval})'
            break

    # Build and return result
    if verbose > 0:
        orig, best = sc.sigfig([fvals[0], fvals[count]])
        ratio = _improvement_ratio(fvals[0], fvals[count], eps)
        print(f'=== {label} {exitreason} ({count} steps, orig: {orig} | best: {best} | ratio: {ratio}) ===')

    output = sc.objdict(
        x = np.reshape(x, origshape), # Parameters
        fval = fvals[count],
        exitreason = exitreason,
        details = sc.objdict(
            fvals = fvals[:count+1], # Function evaluations
            xvals = allsteps[:count+1, :],
            probabilities = probabilities,
            stepsizes = stepsizes
        )
    )
    return output
