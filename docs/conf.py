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

matplotlib.use("agg") # To ensure figures don't render
sys.path.insert(0, os.path.abspath("../"))  # Source code dir relative to this file

import sciris as sc

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# -- Project information -----------------------------------------------------

project = 'Sciris'
copyright = f'2020 by the Sciris team (version {sc.__version__}).'
author = 'Sciris.org'

# The short X.Y version
version = sc.__version__
# The full version, including alpha/beta/rc tags
release = sc.__version__


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '3.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "recommonmark",
]

napoleon_google_docstring = True

# Configure autosummary
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_member_order = 'bysource' # Keep original ordering
add_module_names = False  # NB, does not work
autodoc_inherit_docstrings = False # Stops sublcasses from including docs from parent classes

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# Use RTD
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = "sciris-logo-small.png"
html_favicon = "favicon.ico"
html_static_path = ['_static']
html_context = {
    'css_files': [
        '_static/theme_overrides.css'
    ]
}

html_show_sourcelink = True
html_show_sphinx = False
htmlhelp_basename = 'Sciris'