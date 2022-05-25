'''
Collection of regression tests.
Needs pytests-regression
Stores yaml and npz files in test_regressions/
Test math functions that are widely used in covasim

'''

import sciris as sc
import numpy as np


def test_default_settings(data_regression):
    options, _ = sc._set_default_options()

    #NOTE: Make it a regular dictionary for the moment, yaml does not recognise sciris object
    # yaml.dumps fails
    options_dict = dict()
    for key in options.keys():
        options_dict[key] = options[key]
    data_regression.check(options_dict)


def test_regression_smooth(ndarrays_regression):

    # For reproducibility as long as numpy's default PRNG remains the same
    np.random.seed(42)

    data1d = np.random.rand(200, 5)
    smoothdata1d = sc.smooth(data1d, 10)

    data2d = np.random.rand(200)
    smoothdata2d = sc.smooth(data2d, 10)

    data_dict = {'data1d': data1d, 'smoothdata1d': smoothdata1d,
                 'data2d': data2d, 'smoothdata2d': smoothdata2d
                 }
    ndarrays_regression.check(data_dict)


def test_regression_smoothinterp(ndarrays_regression):
    # use documentation example
    origy = np.array([0, 0.1, 0.3, 0.8, 0.7, 0.9, 0.95, 1])
    origx = np.linspace(0, 1, len(origy))
    newx = np.linspace(0, 1, 5 * len(origy))

    new5 = sc.smoothinterp(newx, origx, origy, smoothness=5)
    new2 = sc.smoothinterp(newx, origx, origy, smoothness=2)

    data_dict = {'new5': new5, 'new2': new2}
    ndarrays_regression.check(data_dict)


def test_regression_rolling(ndarrays_regression):
    # use documentation example

    data = [5, 5, 5, 0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 3, 3, 3]
    rolled = sc.rolling(data)

    data_dict = {'data': data, 'rolled': rolled}
    ndarrays_regression.check(data_dict)


def test_regression_convolve(ndarrays_regression):
    # use documentation example
    a = np.ones(15)
    v = np.array([0.3, 0.5, 0.2])

    c1a = sc.convolve(a, v)
    c1b = sc.convolve(v, a)

    data_dict = {'a': a, 'v': v,
                 'c1a': c1a, 'c1b': c1b
                 }

    ndarrays_regression.check(data_dict)


def test_regression_sanitize(ndarrays_regression):
    # use documentation example
    sanitized1, inds1 = sc.sanitize(np.array([3, 4, np.nan, 8, 2, np.nan, np.nan, np.nan, 8]), returninds=True)
    sanitized2 = sc.sanitize(np.array([3, 4, np.nan, 8, 2, np.nan, np.nan, np.nan, 8]), replacenans=True)
    sanitized3 = sc.sanitize(np.array([3, 4, np.nan, 8, 2, np.nan, np.nan, np.nan, 8]), replacenans=0)

    data_dict = {'sanitized1': sanitized1, 'inds1': inds1,
                 'sanitized2': sanitized2,
                 'sanitized3': sanitized3,
                      }
    ndarrays_regression.check(data_dict)


def test_regression_findinds(ndarrays_regression):
    np.random.seed(42)
    inds1 = sc.findinds(np.random.rand(10) < 0.5)
    inds2 = sc.findinds([2, 3, 6, 3], 3)
    inds3 = sc.findinds([2, 3, 6, 3], 3, first=True)

    data_dict = {'inds1': inds1, 'inds2': inds2, 'inds3': inds3}
    ndarrays_regression.check(data_dict)

###########################################################
##             odict regression tests                    ##
###########################################################

def myfunc(mylist):
    return [i ** 2 for i in mylist]


def test_map(data_regression):

    cat = sc.odict({'a':[1, 2], 'b':[3, 4]})
    dog = cat.map(myfunc) # Returns sc.odict({'a':[1, 4], 'b':[9, 16]})
    data_dict = {'cat': cat, 'dog': dog}
    data_regression.check(data_dict)


