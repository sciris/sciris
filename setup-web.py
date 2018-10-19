#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from setuptools import setup, find_packages

with open("sciris/sc_version.py", "r") as f:
    version_file = {}
    exec(f.read(), version_file)
    version = version_file["__version__"]

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
    name='scirisweb',
    version=version,
    author='ScirisOrg',
    author_email='info@sciris.org',
    description='Scientific webapps for Python',
    url='http://github.com/optimamodel/sciris',
    keywords=['scientific','webapp', 'framework'],
    platforms=['OS Independent'],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'sciris', # Basic tools -- NB, this includes numpy, scipy, pandas, and matplotlib as dependencies
        'decorator>=4.1.2', # For API calls
        'redis>=2.10.6', # Database
        'mpld3',    # Rendering plots in the browser
        'werkzeug', # HTTP tools
        'flask>=1.0.0', # Creating the webapp
        'flask-login>=0.4.1', # Handling users
        'flask-session>=0.3.1', # use redis for sessions
        'celery>=4.2', # Task manager
        'twisted>=18.4.0', # Server
        'service_identity', # Identity manager for Celery (not installed with Celery though)
        'pyasn1', # Required for service_identity (but not listed as a dependency!)
        'pyparsing', # Also for processing requests
    ],
)
