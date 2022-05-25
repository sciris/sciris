'''
Collection of regression tests.
Needs pytests-regression
Stores yaml files in test_regressions/
'''

import sciris as sc


def test_default_settings(data_regression):
    options, _ = sc._set_default_options()

    #NOTE: Make it a regular dictionary for the moment, yaml does not recognise sciris object
    # yaml.dumps fails
    options_dict = dict()
    for key in options.keys():
        options_dict[key] = options[key]
    data_regression.check(options_dict)

