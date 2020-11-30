# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import os
import sys
import matplotlib

matplotlib.use("agg")
sys.path.insert(0, os.path.abspath("../"))  # Source code dir relative to this file

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# -- Project information -----------------------------------------------------

project = 'Sciris'
copyright = '2020, Sciris.org.'
author = 'Sciris.org'

# The short X.Y version
version = '1.0.0'
# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '3.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "sphinx_markdown_tables",
    "recommonmark",
]

napoleon_google_docstring = True

# Configure autosummary
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_member_order = 'bysource' # Keep original ordering
add_module_names = False 

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", 
                    "_autosummary/_autosummary", # CK: Not sure why this gets created, but exclude it here
                    ]


# -- Options for HTML output -------------------------------------------------

# Use RTD
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]




import subprocess

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if sys.platform in ["linux", "darwin"]:
    subprocess.check_output(["make", "generate-api"], cwd=os.path.dirname(os.path.abspath(__file__)))
else:
    subprocess.check_output(["make.bat", "generate-api"], cwd=os.path.dirname(os.path.abspath(__file__)))

# Rename "covasim package" to "API reference"
filename = os.path.join('_autosummary/', 'sciris.rst') # This must match the Makefile
with open(filename) as f: # Read exitsting file
    lines = f.readlines()
lines[0] = "This is a test\n" # Blast away the existing heading and replace with this
lines[1] = "==============\n" # Ensure the heading is the right length
with open(filename, "w") as f: # Write new file
    f.writelines(lines)