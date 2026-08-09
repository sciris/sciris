================
Regression tests
================

This folder contains some specialized tests for checking for regressions.


Testing older packages
----------------------

To test Sciris on older package versions, run ``setup_envs`` followed by ``run_regression``.


Testing downstream packages
---------------------------

To run the test suites of key packages that rely on Sciris, run ``run_downstream``. (Note: requires local installation of each of these packages, not via PyPI.)