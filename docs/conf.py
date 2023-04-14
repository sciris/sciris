# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
import sciris as sc

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# -- Project information -----------------------------------------------------

project = 'Sciris'
copyright = f'2014–2023 by the Sciris Development Team (version {sc.__version__})'
author = 'Sciris Development Team'

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
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc -- causes warnings with Napoleon however
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
]

# Use Google docstrings
napoleon_google_docstring = True

# Configure autosummary
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autoclass_content = "init"  # Add __init__ doc (ie. params) to class summaries
# html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_member_order = 'bysource' # Keep original ordering
add_module_names = False  # NB, does not work
# autodoc_inherit_docstrings = False # Stops subclasses from including docs from parent classes

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Syntax highlighting style
pygments_style = "sphinx"
modindex_common_prefix = ["sciris."]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "ansicolors.py"]

# Suppress certain warnings
suppress_warnings = ['autosectionlabel.*']

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 2,
    "show_prev_next": False,
    "icon_links": [
        {"name": "Web", "url": "https://sciris.org", "icon": "fas fa-home"},
        {
            "name": "GitHub",
            "url": "https://github.com/sciris/sciris",
            "icon": "fab fa-github-square",
        },
    ],
    # "external_links": [{"name": "Guides", "url": "https://networkx.org/nx-guides/"}],
    # "navbar_end": ["theme-switcher", "navbar-icon-links", "version"],
    "secondary_sidebar_items": ["search-field", "page-toc", "edit-this-page"],
    "header_links_before_dropdown": 7,
}
html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
    "index": [],
    "install": [],
    "tutorial": [],
    "auto_examples/index": [],
}
html_logo = "sciris-logo-small.png"
html_favicon = "favicon.ico"
html_static_path = ['_static']
html_baseurl = "https://sciris.readthedocs.io/en/latest/"
html_context = {
    'rtd_url': 'https://sciris.readthedocs.io/en/latest/',
    'theme_vcs_pageview_mode': 'edit'
}

html_last_updated_fmt = '%Y-%b-%d'
html_show_sourcelink = True
html_show_sphinx = False
html_copy_source = False
htmlhelp_basename = 'Sciris'
