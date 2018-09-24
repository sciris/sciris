def asd(function, x, args=None, stepsize=0.1, sinc=2, sdec=2, pinc=2, pdec=2,
    pinitial=None, sinitial=None, xmin=None, xmax=None, maxiters=None, maxtime=None, 
    abstol=1e-6, reltol=1e-3, stalliters=None, stoppingfunc=None, randseed=None, 
    label=None, fulloutput=True, verbose=2, **kwargs):
    """
    Optimization using adaptive stochastic descent (ASD).
    
    output = asd(func,x0) starts at x0 and attempts to find a 
    local minimizer x of the function func. func accepts input x and returns a scalar 
    function value evaluated  at x. x0 can be a scalar, list, or Numpy array of 
    any size. 
    
    If fulloutput is False, then asd() returns x only. If it is true, then it returns
    a tuple with the following items:
        x       -- The parameter set that minimizes the objective function
        fval    -- The value of the objective function at each iteration (use fval[-1] for final)
        details -- A dictionary with additional output: exitreason, why the algorithm terminated;
                   probabilities, the probability of each step; and stepsizes, the size of each
                   step for each parameter
    
    asd() has the following options that can be set using keyword arguments. Their
    names and default values are as follows:
    
      stepsize       0.1     Initial step size as a fraction of each parameter
      sinc           2       Step size learning rate (increase)
      sdec           2       Step size learning rate (decrease)
      pinc           2       Parameter selection learning rate (increase)
      pdec           2       Parameter selection learning rate (decrease)
      pinitial       None    Set initial parameter selection probabilities
      sinitial       None    Set initial step sizes; if empty, calculated from stepsize instead
      xmin           None    Min value allowed for each parameter  
      xmax           None    Max value allowed for each parameter 
      maxiters       1000    Maximum number of iterations (1 iteration = 1 function evaluation)
      maxtime        3600    Maximum time allowed, in seconds
      abstol         1e-6    Minimum absolute change in objective function
      reltol         1e-3    Minimum relative change in objective function
      stalliters     10*n    Number of iterations over which to calculate TolFun (n = number of parameters)
      stoppingfunc   None    External method that can be used to stop the calculation from the outside.
      randseed       None    The random seed to use
      fulloutput     True    Whether or not to return the full output
      verbose        2       How much information to print during the run
      label          None    A label to use to annotate the output
  
    Example:
        from sciris import asd
        from numpy.linalg import norm
        x, fval, details = asd(norm, [1, 2, 3])
    
    Version: 2018sep24
    """
    from numpy import array, shape, reshape, ones, zeros, mean, cumsum, mod, concatenate, floor, flatnonzero, isnan, inf
    from numpy.random import random, seed
    from time import time
    from .sc_utils import dcp, sigfig
    if randseed is not None:
        seed(int(randseed)) # Don't reset it if not supplied
        if verbose >= 3: print('ASD: Launching with random seed is %i; sample: %f' % (randseed, random()))

    def consistentshape(userinput, origshape=False):
        """
        Make sure inputs have the right shape and data type.
        """
        output = reshape(array(userinput, dtype='float'), -1)
        if origshape: return output, shape(userinput)
        else:         return output

    # Handle inputs and set defaults
    if maxtime  is None: maxtime  = 3600
    if maxiters is None: maxiters = 1000
    maxrangeiters = 100 # Number of times to try generating a new parameter
    x, origshape = consistentshape(x, origshape=True) # Turn it into a vector but keep the original shape (not necessarily class, though)
    nparams = len(x) # Number of parameters
    if not nparams:
        errormsg = 'ASD: The length of the input vector cannot be zero'
        raise Exception(errormsg)
    if sinc<1:
        print('ASD: sinc cannot be less than 1; resetting to 2'); sinc = 2
    if sdec<1:
        print('ASD: sdec cannot be less than 1; resetting to 2'); sdec = 2
    if pinc<1:
        print('ASD: pinc cannot be less than 1; resetting to 2')
        pinc = 2
    if pdec<1:
        print('ASD: pdec cannot be less than 1; resetting to 2')
        pdec = 2

    # Set initial parameter selection probabilities -- uniform by default
    if pinitial is None: probabilities = ones(2 * nparams)
    else:                probabilities = consistentshape(pinitial)
    if not sum(probabilities):
        errormsg = 'ASD: The sum of input probabilities cannot be zero'
        raise Exception(errormsg)

    # Handle step sizes
    if sinitial is None:
        stepsizes = abs(stepsize * x)
        stepsizes = concatenate((stepsizes, stepsizes)) # need to duplicate since two for each parameter
    else:
        stepsizes = consistentshape(sinitial)

    # Handle x limits
    xmin = zeros(nparams) - inf if xmin is None else consistentshape(xmin)
    xmax = zeros(nparams) + inf if xmax is None else consistentshape(xmax)

    # Final input checking
    if sum(isnan(x)):
        errormsg = 'ASD: At least one value in the vector of starting points is NaN:\n%s' % x
        raise Exception(errormsg)
    if label is None: label = ''
    if stalliters is None: stalliters = 10 * nparams # By default, try 10 times per parameter on average
    stalliters = int(stalliters)
    maxiters = int(maxiters)

    # Initialization
    if all(stepsizes == 0): stepsizes += stepsize # Handle the case where all step sizes are 0
    if any(stepsizes == 0): stepsizes[stepsizes == 0] = mean(stepsizes[stepsizes != 0]) # Replace step sizes of zeros with the mean of non-zero entries
    if args is None: args = {} # Reset if no function arguments supplied
    fval = function(x, **args) # Calculate initial value of the objective function
    fvalorig = fval # Store the original value of the objective function, since fval is overwritten on each step
    xorig = dcp(x) # Keep the original x, just in case

    # Initialize history
    abserrorhistory = zeros(stalliters) # Store previous error changes
    relerrorhistory = zeros(stalliters) # Store previous error changes
    fvals = zeros(maxiters + 1) # Store all objective function values
    allsteps = zeros((maxiters + 1, nparams)) # Store all parameters
    fvals[0] = fvalorig # Store initial function output
    allsteps[0, :] = xorig # Store initial input vector

    # Loop
    count = 0 # Keep track of how many iterations have occurred
    start = time() # Keep track of when we begin looping
    offset = ' ' * 4 # Offset the print statements
    exitreason = 'Unknown exit reason' # Catch everything else
    while True:
        count += 1 # Increment the count
        if verbose == 1: print(offset + label + 'Iteration %i; elapsed %0.1f s; objective: %0.3e' % (count, time() - start, fval)) # For more verbose, use other print statement below
        if verbose >= 4: print('\n\n Count=%i \n x=%s \n probabilities=%s \n stepsizes=%s' % (count, x, probabilities, stepsizes))
        
        # Calculate next parameters
        probabilities = probabilities / sum(probabilities) # Normalize probabilities
        cumprobs = cumsum(probabilities) # Calculate the cumulative distribution
        inrange = False
        for r in range(maxrangeiters): # Try to find parameters within range
            choice = flatnonzero(cumprobs > random())[0] # Choose a parameter and upper/lower at random
            par = mod(choice, nparams) # Which parameter was chosen
            pm = floor((choice) / nparams) # Plus or minus
            newval = x[par] + ((-1)**pm) * stepsizes[choice] # Calculate the new vector
            if newval<xmin[par]: newval = xmin[par] # Reset to the lower limit
            if newval>xmax[par]: newval = xmax[par] # Reset to the upper limit
            inrange = (newval != x[par])
            if verbose >= 4: print(offset*2 + 'count=%i r=%s, choice=%s, par=%s, x[par]=%s, pm=%s, step=%s, newval=%s, xmin=%s, xmax=%s, inrange=%s' % (count, r, choice, par, x[par], (-1)**pm, stepsizes[choice], newval, xmin[par], xmax[par], inrange))
            if inrange: # Proceed as long as they're not equal
                break
        if not inrange: # Treat it as a failure if a value in range can't be found
            probabilities[choice] = probabilities[choice] / pdec
            stepsizes[choice] = stepsizes[choice] / sdec

        # Calculate the new value
        
        xnew = dcp(x) # Initialize the new parameter set
        xnew[par] = newval # Update the new parameter set
        fvalnew = function(xnew, **args) # Calculate the objective function for the new parameter set
        eps = 1e-12 # Small value to avoid divide-by-zero errors
        if abs(fvalnew)<eps and abs(fval)<eps: ratio = 1 # They're both zero: set the ratio to 1
        elif abs(fvalnew)<eps:                 ratio = 1.0/eps # Only the denominator is zero: reset to the maximum ratio
        else:                                  ratio = fval/float(fvalnew) # The normal situation: calculate the real ratio
        abserrorhistory[mod(count, stalliters)] = max(0, fval-fvalnew) # Keep track of improvements in the error
        relerrorhistory[mod(count, stalliters)] = max(0, ratio-1.0) # Keep track of improvements in the error
        if verbose >= 3: print(offset + 'step=%i choice=%s, par=%s, pm=%s, origval=%s, newval=%s' % (count, choice, par, pm, x[par], xnew[par]))

        # Check if this step was an improvement
        fvalold = fval # Store old fval
        if fvalnew < fvalold: # New parameter set is better than previous one
            probabilities[choice] = probabilities[choice] * pinc # Increase probability of picking this parameter again
            stepsizes[choice] = stepsizes[choice] * sinc # Increase size of step for next time
            x = xnew # Reset current parameters
            fval = fvalnew # Reset current error
            flag = '++' # Marks an improvement
        elif fvalnew >= fvalold: # New parameter set is the same or worse than the previous one
            probabilities[choice] = probabilities[choice] / pdec # Decrease probability of picking this parameter again
            stepsizes[choice] = stepsizes[choice] / sdec # Decrease size of step for next time
            flag = '--' # Marks no change
        else:
            exitreason = 'Objective function returned NaN'
            break
        if verbose >= 2: print(offset + label + ' step %i (%0.1f s) %s (orig: %s | best:%s | new:%s | diff:%s)' % ((count, time() - start, flag) + sigfig([fvalorig, fvalold, fvalnew, fvalnew - fvalold])))

        # Store output information
        fvals[count] = fval # Store objective function evaluations
        allsteps[count, :] = x # Store parameters

        # Stopping criteria
        if count >= maxiters: # Stop if the iteration limit is exceeded
            exitreason = 'Maximum iterations reached'
            break
        if (time() - start) > maxtime:
            exitreason = 'Time limit reached (%s > %s)' % sigfig([(time()-start), maxtime])
            break
        if (count > stalliters) and (abs(mean(abserrorhistory)) < abstol): # Stop if improvement is too small
            exitreason = 'Absolute improvement too small (%s < %s)' % sigfig([mean(abserrorhistory), abstol])
            break
        if (count > stalliters) and (sum(relerrorhistory) < reltol): # Stop if improvement is too small
            exitreason = 'Relative improvement too small (%s < %s)' % sigfig([mean(relerrorhistory), reltol])
            break
        if stoppingfunc and stoppingfunc():
            exitreason = 'Stopping function called'
            break

    # Return
    x = reshape(x, origshape) # Parameters
    fvals = fvals[:count + 1] # Function evaluations
    if verbose >= 2: print('=== %s %s (%i steps, orig: %s | best: %s | ratio: %s) ===' % ((label, exitreason, count) + sigfig([fvals[0], fvals[-1], fvals[-1] / fvals[0]])))
    if fulloutput:
        details = dict()
        details['exitreason'] = exitreason
        details['probabilities'] = probabilities
        details['stepsizes'] = stepsizes
        details['allsteps'] = allsteps[:count + 1, :]
        return (x, fvals, details)
    else:
        return (x, None, None) # so that it has the same shape regardless of fulloutput
