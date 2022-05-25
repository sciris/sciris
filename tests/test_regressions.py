'''
Collection of regression tests.
Needs pytests-regression
Stores yaml files in test_regressions/
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
                 'data2d': data2d, 'smoothdata2d': smoothdata2d}
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
