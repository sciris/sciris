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
from . import sc_math as scm
from . import sc_parallel as scp

__all__ = ['alias_sampler']


def sample(r1, r2, probs_table=None, alias_table=None):
    res = scu.toarray(0, dtype=np.int32)
    lj = len(alias_table)
    kk = int(np.floor(r1 * lj))
    if r2 < probs_table[kk]:
        res[0] = kk
    else:
        res[0] = alias_table[kk]
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

    def __init__(self, probs, vals=None, randseed=None, verbose=False, parallel=True):
        self.probs = scu.toarray(probs)
        self.vals = scu.toarray(vals)  # If vals is None, then self.vals is an empty array
        self.rng = np.random.default_rng(randseed)  # Construct a new Generator
        self.n = n = len(probs)
        # Initialise tables
        self.probs_table = probs_table = n * self.probs
        self.alias_table = alias_table = np.empty((n,), dtype=np.int32)
        self.parallel = parallel

        # Do some checks if both probs and vals are given
        if vals is not None:
            # Check lengths of prob and vals are equal
            if not (self.probs.size == self.vals.size):
                errormsg = f" 'probs' and 'vals' must be of the same length/size"
                raise ValueError(errormsg)

        # Run checks on probs
        self._check_probs(verbose)

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
            uf_j = uf(self.rng.choice(uf, 1))
            of_i = of(self.rng.choice(of, 1))

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

        # Create a generato
        self.rng = np.random.default_rng(randseed)

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

    def draw(self, n):
        r1, r2 = self.rng.rand(n), self.rng.rand(n)
        if self.parallel:
            res = scp.parallelize(sample, iterarg=[r1, r2],
                                          kwargs={'probs_table': self.probs_table,
                                                  'alias_table': self.alias_table})
        else:
            res = np.zeros(n)
            for i in range(n):
                res[i] = sample(r1[i], r2[i], probs_table=self.probs_table, alias_table=self.alias_table)
        return res
