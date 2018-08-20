#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from setuptools import setup, find_packages

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
    name='scirisweb',
    version=version,
    author='Cliff Kerr, George Chadderdon',
    author_email='info@sciris.org',
    description='Scientific webapps for Python',
    url='http://github.com/optimamodel/sciris',
    keywords=['scientific','webapp', 'framework'],
    platforms=['OS Independent'],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'sciris', # Basic tools
        'redis', # Database
        'mpld3', # Rendering plots in the browser
        'flask', # Creating the webapp
        'flask-login', # Handling users
        'werkzeug', # HTTP tools
        'twisted', # Server
        'celery', # Task manager
    ],
)
