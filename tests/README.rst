=====
Tests
=====

This folder contains the core tests for Sciris.

Installation
------------

To install test dependencies, use ``pip install -r requirements_test.txt``. (Note: ``pytest-parallel`` is used instead of ``pytest-xdist`` since it has much less overhead for running fast tests.)

Usage
-----

Recommended usage is ``./check_coverage`` or ``./run_tests``. You can also use ``pytest`` to run all the tests in the folder.