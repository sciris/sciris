'''
Example sim for the "Other tools" tutorial
'''

import sciris as sc
import numpy as np
import pylab as pl

class Sim(sc.prettyobj):

    def __init__(self, days=30, trials=5):
        self.days = days
        self.trials = trials

    def run(self):
        self.x = np.arange(self.days)
        self.y = np.cumsum(np.random.randn(self.days, self.trials)**3, axis=0)
        return self

    def plot(self):
        with pl.style.context('sciris.fancy'):
            pl.plot(self.x, self.y, alpha=0.6)