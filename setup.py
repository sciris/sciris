#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
from setuptools import setup, find_packages

# Optionally define extras
if 'minimal' in sys.argv:
    sys.argv.remove('minimal')
    extras = []
else:
    extras = [
        'dill',              # File I/O
        'json',              # More file I/O
        'gitpython',         # Version information
        'openpyxl>=2.5',     # Spreadsheet functions
        'pandas',            # Spreadsheet input
        'psutil',            # Load monitoring
        'xlrd',              # Spreadsheet input
        'xlsxwriter',        # Spreadsheet output
        ]

# Get the version
with open("sciris/sc_version.py", "r") as f:
    version_file = {}
    exec(f.read(), version_file)
    version = version_file["version"]

CLASSIFIERS = [
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GPLv3',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 2.7',
]

setup(
    name='sciris',
    version=version,
    author='Cliff Kerr, George Chadderdon',
    author_email='info@sciris.org',
    description='Scientific tools for Python',
    url='http://github.com/optimamodel/sciris',
    keywords=['scientific','webapp', 'framework'],
    platforms=['OS Independent'],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib>=1.4.2', # Plotting
        'numpy>=1.10.1',     # Numerical functions
    ] + extras,              # Optional dependencies
)
