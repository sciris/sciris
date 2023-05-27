"""
Walker's alias method

Implementation references:
    https://en.wikipedia.org/wiki/Alias_method (main reference)
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/ (Python)
    https://twitter.com/jeremyphoward/status/955136770806444032
    https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087 (Python)
    https://www.keithschwarz.com/darts-dice-coins/
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

Bibliographic refs:
R. A. Kronmal and A. V. Peterson. On the alias method for generating random variables from a discrete distribution.
The American Statistician, 33(4):214-218, 1979.

D. Knuth. The Art of Computer Programming, Vol 2: Seminumerical Algorithms, section 3.4.1. (p120-121,p127)
"""
from numba import jit
import numpy.random as npr
import numpy as np
import functools
from . import sc_utils as scu
from . import sc_math as scm
from . import sc_parallel as scp

__all__ = ['alias_sampler', 'AliasSample']


def alias_sample_one(x=None, num_indices=None, probs_table=None, alias_table=None):
    """
    Perform singleton sampling from a discrete probability distribution using the Alias method.

    Args:
        x (float)            : a uniform random variate between 0 and 1.
        num_indices (int)    : the number of indices in the discrete distribution.
        probs_table (ndarray): the probability table used for sampling.
        alias_table (ndarray): the alias table used for sampling.

    Returns:
        res (int): A single sample drawn from the distribution.
    """
    ii = np.floor(x * num_indices).astype(np.int32)  # Get a random integer between 0 and num_indices-1
    yy = num_indices * x - ii
    # If y < prob_i then return i.This is the biased coin flip.
    if yy < probs_table[ii]:
        res = ii
    else:  # Otherwise, return Ki.
        res = alias_table[ii]
    return res


def alias_sample_vec(x=None, num_indices=None, probs_table=None, alias_table=None):
    """
    Perform vectorised sampling from a discrete probability distribution using the alias method.

    Args:
        x (ndarray)          : an array of uniform random variates between 0 and 1.
        num_indices (int)    : the number of indices in the distribution.
        probs_table (ndarray): the probability table used for sampling.
        alias_table (ndarray): the alias table used for sampling.

    Returns:
        ndarray: An array of samples drawn from the distribution.
    """
    # Allocate space for results
    # TODO: maybe initialise with inf or nan, to identify potential problems with the tables and/or sampling
    res = np.zeros_like(x, dtype=np.int32)
    # Create indices into the probs table
    ii = np.floor(x * num_indices).astype(np.int32)
    yy = num_indices * x - ii
    smaller = yy < probs_table[ii]
    res[smaller] = ii[smaller]
    res[~smaller] = alias_table[ii[~smaller]]
    return res


class alias_sampler:
    """
    Walker's Alias method for sampling from a discrete probability distribution

    If this class is instantiated with one argument (probs) then the draw methods
    returns `indices` into the discrete distribution. If the class is instantiated
    with probs and vals then draw methods return randomly sampled 'values' from
    the distribution.

    Args:
    probs (int/array/list/tuple): a sequence of probabilities for each element in the discrete distribution.
    vals (int/array/list/tuple) : a sequence with the values to which the probability entries correspond.
    randseed                    : a seed to initialize the BitGenerator. Usually an int, though could be any type
                                 accepted by numpy.random.default_rng(). If None, then fresh, unpredictable entropy
                                 will be pulled from the OS.
    parallel(bool)              : whether to parallelise samplping or not(default)

    """

    def __init__(self, probs, vals=None, randseed=None, verbose=True, parallel=False):
        self.probs = scu.toarray(probs)
        self.vals = scu.toarray(vals)  # If vals is None, then self.vals is an empty array
        self.rng = np.random.default_rng(randseed)  # Construct a new Generator
        self.num_buckets = n_buckets = len(probs)
        self.parallel = parallel

        # Do some checks if both probs and vals are given
        if vals is not None:
            # Check lengths of prob and vals are equal
            if not (self.probs.size == self.vals.size):
                errormsg = f" 'probs' and 'vals' must be of the same length/size"
                raise ValueError(errormsg)

        # Run checks on probs
        self._check_probs(verbose)

        # Initialise tables
        self.probs_table = probs_table = n_buckets * self.probs
        self.alias_table = alias_table = np.zeros((n_buckets,), dtype=np.int32)

        # Divide the table entries into three categories,
        overfull = probs_table > 1.0
        underfull = probs_table < 1.0
        exactly_full = probs_table == 1.0

        # Alias table K, initialise exactly full entries
        alias_table[exactly_full] = scm.findinds(exactly_full)

        # As long as not all prob table entries are exactly full, repeat the following steps:
        while any(underfull) and any(overfull):
            # Find True indices
            uf = scm.findinds(underfull)
            of = scm.findinds(overfull)
            # Arbitrarily choose an overfull entry (i) and an underfull entry (j)
            uf_j = self.rng.choice(uf, 1)
            of_i = self.rng.choice(of, 1)

            # Allocate the unused space in entry j to outcome i, by setting Kj = i.
            alias_table[uf_j] = of_i

            # Remove the allocated space from entry i by changing
            probs_table[of_i] = probs_table[of_i] - (1.0 - probs_table[uf_j])

            # Entry j is now exactly full, so update the corresponding category arrays
            underfull[uf_j] = False
            exactly_full[uf_j] = True

            # Assign entry i to the appropriate category based on the new value of Ui.
            overfull[of_i] = False
            if probs_table[of_i] > 1.0:
                overfull[of_i] = True
            elif probs_table[of_i] < 1.0:
                underfull[of_i] = True
            else:
                exactly_full[of_i] = True

        # Update tables
        self.probs_table = probs_table
        self.alias_table = alias_table

        # Create a generator
        self.rng = np.random.default_rng(randseed)

        # TODO: check some potential rounding error problems

    def _check_probs(self, verbose):
        """
        Run some basic checks on the probs array
        """
        # Check for really edge cases
        where_conds = [0.0, np.inf, np.nan]
        for cond in where_conds:
            if self.probs.all(where=cond):
                errormsg = f" All probabilities are {cond}"
                ValueError(errormsg)
        # Check for some bad values in probs
        where_conds = [np.inf, np.nan]
        for cond in where_conds:
            if self.probs.any(where=cond):
                errormsg = f"There are some {cond} in probs. "
                ValueError(errormsg)
        # Check for negative probabilities
        if any(self.probs < 0):
            errormsg = f"There are negative values in probs. "
            raise ValueError(errormsg)
        # Check it adds up to 1
        prob_sum = self.probs.sum()
        if not (prob_sum == 1.0):
            self.probs /= prob_sum
            if verbose:
                errormsg = f"Warning! Probabilities didn't add up to 1. Normalising by prob.sum(): {prob_sum}"
                print(errormsg)

    def draw(self, n_samples):
        """
        An interface function to draw samples
        """
        # Generate a uniform random variate 0 ≤ x < 1,independently if we run in parallel or not
        uniform_x = self.rng.random(n_samples)
        n_buckets = self.num_buckets
        if self.parallel:
            # NOTE: not sure it makes a lot of sense to parallelise sampling for n_samples < 1e6 samples maybe?
            # TODO: handle output of parallelise so it is consistent with the output of vectorised sampling
            sample_func = functools.partial(alias_sample_one, num_indices=n_buckets, probs_table=self.probs_table,
                                            alias_table=self.alias_table)
            res = scp.parallelize(sample_func, iterkwargs={'x': uniform_x})
        else:
            sample_func = functools.partial(alias_sample_vec, num_indices=n_buckets, probs_table=self.probs_table,
                                            alias_table=self.alias_table)
            res = sample_func(x=uniform_x)

        # Check whether we need to return indices or values
        if len(self.vals) > 1:
            return self.vals[res]
        return res


@jit(nopython=True)
def sample(n, q, J, r1, r2):
    res = np.zeros(n, dtype=np.int32)
    lj = len(J)
    for i in range(n):
        kk = int(np.floor(r1[i] * lj))
        if r2[i] < q[kk]:
            res[i] = kk
        else:
            res[i] = J[kk]
    return res


class AliasSample():
    def __init__(self, probs):
        self.K = K = len(probs)
        self.q = q = np.zeros(K)
        self.J = J = np.zeros(K, dtype=np.int)

        smaller, larger = [], []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small, large = smaller.pop(), larger.pop()
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def draw_one(self):
        K, q, J = self.K, self.q, self.J
        kk = int(np.floor(npr.rand() * len(J)))
        if npr.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def draw_n(self, n):
        r1, r2 = npr.rand(n), npr.rand(n)
        return sample(n, self.q, self.J, r1, r2)
