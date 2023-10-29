#!/usr/bin/env python

'''
Sciris is a flexible open source framework for building scientific web
applications using Python and JavaScript. This library provides the underlying
functions and data structures that support the webapp features, as well as
being generally useful for scientific computing.
'''

import os
import sys
import runpy
from setuptools import setup, find_packages

# Get the current folder
cwd = os.path.abspath(os.path.dirname(__file__))

# Load requirements from txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

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
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
]

setup(
    name='sciris',
    version=version,
    author='Sciris Development Team',
    author_email='info@sciris.org',
    description='Fast, flexible tools to simplify scientific Python',
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
