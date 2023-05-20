"""
Walker's alias method

https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
Implementation references:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    https://twitter.com/jeremyphoward/status/955136770806444032
    https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087
    https://en.wikipedia.org/wiki/Alias_method
    https://www.keithschwarz.com/darts-dice-coins/

Bibliographic refs:
R. A. Kronmal and A. V. Peterson. On the alias method for generating random variables from a discrete distribution.
The American Statistician, 33(4):214-218, 1979.

D. Knuth. The Art of Computer Programming, Vol 2: Seminumerical Algorithms, section 3.4.1.
"""

import numpy.random as npr
import numpy as np
from . import sc_utils as scu

__all__ = ['alias_sampler']


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


class alias_sampler:
    """
    Walker's Alias method for sampling from a discrete probability distribution

    If this class is instantiated with one argument (probs) then the draw methods
    returns `indices` into the discrete distribution. If the class is instantiated
    with probs and vals then draw methods return randomly sampled 'values' from
    the distribution.

    Args:
    probs(int/array/list/tuple): a sequence of probabilities for each element in the discrete distribution.
    vals(int/array/list/tuple) : a sequence with the values to which the probability entries correspond.
    randseed                   : a seed to initialize the BitGenerator. Usually an int, though could be any type
                                 accepted by numpy.random.default_rng(). If None, then fresh, unpredictable entropy
                                 will be pulled from the OS.


    """

    def __init__(self, probs, vals=None, randseed=None, verbose=False):
        self.probs = scu.toarray(probs)
        self.vals = scu.toarray(vals)  # If vals is None, then self.vals is an empty array
        self.rng = np.random.default_rng(randseed)  # Construct a new Generator
        self.K = K = len(probs)
        self.q = q = np.zeros(K)
        self.J = J = np.zeros(K, dtype=np.int32)

        # Do some checks if both probs and vals are given
        if vals is not None:
            # Check lengths of prob and vals are equal
            if not (self.probs.size == self.vals.size):
                errormsg = f" 'probs' and 'vals' must be of the same length/size"
                raise ValueError(errormsg)

        # Run checks on probs
        self._check_probs(verbose)

        smaller, larger = [], []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:  # underfull group
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
            ValueError(errormsg)
        # Check it adds up to 1
        prob_sum = self.probs.sum()
        if not prob_sum == 1.0:
            self.probs /= prob_sum
            if verbose:
                errormsg = f"Warning! Probabilities didn't add up to 1. Normalising by prob.sum(): {prob_sum}"
                print(errormsg)

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
