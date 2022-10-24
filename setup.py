#!/usr/bin/env python

'''
Sciris is a flexible open source framework for building scientific web
applications using Python and JavaScript. This library provides the underlying
functions and data structures that support the webapp features, as well as
being generally useful for scientific computing.
'''

from setuptools import setup, find_packages
import os
import sys
import runpy

# Get the current folder
cwd = os.path.abspath(os.path.dirname(__file__))

# Define the requirements for core functionality
requirements = [
        'matplotlib',   # Plotting
        'numpy',        # Numerical functions
        'pandas',       # Dataframes and spreadsheet input
        'openpyxl',     # To read Excel files; removed as a dependency of pandas as of version 1.3
        'xlsxwriter',   # Spreadsheet output
        'psutil',       # Load monitoring
        'dill',         # For pickling more complex object types
        'multiprocess', # More flexible version of multiprocessing
        'jsonpickle',   # For converting arbitrary objects to JSON
        'pyyaml',       # For loading/saving YAML
        'packaging',    # For parsing versions
        'gitpython',    # Git version information
        'jellyfish',    # For fuzzy string matching
        'line_profiler ;   platform_system == "Linux"',   # For the line profiler -- only install on Linux
        'memory_profiler ; platform_system == "Linux"',   # For the memory profiler -- only install on Linux
        'colorama ;        platform_system == "Windows"', # For colored text output -- only install on Windows
        ]

# Get version
versionpath = os.path.join(cwd, 'sciris', 'sc_version.py')
version = runpy.run_path(versionpath)['__version__']

# Get the documentation
with open(os.path.join(cwd, 'README.rst'), "r") as fh:
    long_description = fh.read()

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
]

setup(
    name='sciris',
    version=version,
    author='Sciris Development Team',
    author_email='info@sciris.org',
    description='Scientific tools for Python',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='http://github.com/sciris/sciris',
    keywords=['scientific', 'webapp', 'framework'],
    platforms=['OS Independent'],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements
)
