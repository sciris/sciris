Welcome to Sciris
=================

.. image:: https://badgen.net/pypi/v/sciris/
 :target: https://pypi.com/project/sciris

.. image:: https://img.shields.io/pypi/v/sciris?style=for-the-badge
   :alt: PyPI

.. image:: https://github.com/sciris/sciris/actions/workflows/tests.yaml/badge.svg
 :target: https://github.com/sciris/sciris/actions/workflows/tests.yaml?query=workflow

.. image:: https://static.pepy.tech/personalized-badge/sciris?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads
 :target: https://pepy.tech/project/sciris

.. image:: https://img.shields.io/pypi/dm/sciris?style=for-the-badge
   :alt: PyPI - Downloads

.. image:: https://img.shields.io/github/contributors/sciris/sciris
   :alt: GitHub contributors


What is Sciris?
---------------

Glad you asked! **Sciris** (http://sciris.org) is a library of tools that can help make writing scientific Python code easier and more pleasant. Built on top of `NumPy <https://numpy.org/>`__ and `Matplotlib <https://matplotlib.org/>`__, Sciris provides functions covering a wide range of common math, file I/O, and plotting operations. This means you can get more done with less code, and spend less time looking things up on StackOverflow. It was originally written to help epidemiologists and neuroscientists focus on doing science, rather than on writing coding, but Sciris is applicable across scientific domains.

Sciris is available on `PyPi <https://pypi.org/project/sciris/>`__ (``pip install sciris``) and `GitHub <https://github.com/sciris/sciris>`__. Full documentation is available at http://docs.sciris.org. If you have questions, feature suggestions, or would like some help getting started, please reach out to us at info@sciris.org.


Highlights
~~~~~~~~~~
Some highlights of Sciris (``import sciris as sc``):

- **Powerful containers** – The ``sc.odict`` class is what ``OrderedDict`` (almost) could have been, allowing reference by position or key, casting to a NumPy array, sorting and enumeration functions, etc.
- **Array operations** – Want to find the indices of an array that match a certain value or condition? ``sc.findinds()`` will do that. How about just the nearest value, regardless of exact match? ``sc.findnearest()``. What about the last matching value? ``sc.findlast()``. Yes, you could do ``np.nonzero()[0][-1]`` instead, but ``sc.findlast()`` is easier to read, type, and remember, and handles edge cases more elegantly.
- **File I/O** – One-liner functions for saving and loading text, JSON, spreadsheets, or even arbitrary Python objects.
- **Plotting recipes** – Simple functions for mapping sequential or qualitative data onto colors, manipulating color data, and updating axis limits and tick labels, plus several new colormaps.


I'm not convinced.
~~~~~~~~~~~~~~~~~~
That's OK. Perhaps you'd be interested in seeing what a script that performs tasks like parallelization, saving and loading files, and 3D plotting looks like when written in "`vanilla Python <https://github.com/sciris/sciris/blob/main/tests/showcase_vanilla.py>`_" (left) compared to `using Sciris <https://github.com/sciris/sciris/blob/main/tests/showcase_sciris.py>`_ (right):

|Sciris showcase|

Both of these do the same thing, but the plain Python version (left) requires 50% more lines of code to produce the same graph as Sciris (right):

|Sciris output|


Where did Sciris come from?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Development of Sciris began in 2014 to support development of the `Optima <http://optimamodel.com>`_ suite of models. We kept encountering the same issues and inconveniences over and over while building scientific webapps, and began collecting the tools we used to overcome these issues into a shared library. This library evolved into Sciris. (Note: while "Sciris" doesn't mean anything, "iris" means "rainbow" in Greek, and the name was loosely inspired by the wide spectrum of scientific computing features included in Sciris.)

To give a based-on-a-true-story example, let's say you have a dictionary of results for multiple runs of your model, called ``results``. The output of each model run is itself a dictionary, with keys such as ``name`` and ``data``. Now let's say you want to access the data from the first model run. Using plain Python dictionaries, this would be ``results[list(results.keys())[0]]['data']``. Using a Sciris ``objdict``, this is ``results[0].data`` – almost 3x shorter.


Where has Sciris been used?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sciris is currently used by a number of scientific computing libraries, including `Atomica <http://atomica.tools>`_ and `Covasim <http://covasim.org>`__. `ScirisWeb <http://github.com/sciris/scirisweb>`_, a lightweight web framework built on top of Sciris, provides the backend for webapps such as the `Cascade Analysis Tool <http://cascade.tools>`_, `HIPtool <http://hiptool.org>`_, and `Covasim <http://app.covasim.org>`_.


Features
-------------------

Here are a few more of the most commonly used features.

Containers
~~~~~~~~~~
-  ``sc.odict()``: flexible container representing the best-of-all-worlds across lists, dicts, and arrays
-  ``sc.objdict()``: like an odict, but allows get/set via e.g. ``foo.bar`` instead of ``foo['bar']``

Array operations
~~~~~~~~~~~~~~~~
-  ``sc.findinds()``: find indices of an array matching a value or condition
-  ``sc.findnearest()``: find nearest matching value
-  ``sc.smooth()``: simple smoothing of 1D or 2D arrays
-  ``sc.isnumber()``: checks if something is any number type
-  ``sc.promotetolist()``: converts any object to a list, for easy iteration
-  ``sc.promotetoarray()``: tries to convert any object to an array, for easy use with NumPy

File I/O
~~~~~~~~
-  ``sc.save()/sc.load()``: efficiently save/load any Python object (via pickling)
-  ``sc.savejson()/sc.loadjson()``: likewise, for JSONs
-  ``sc.thisdir()``: get current folder
-  ``sc.getfilelist()``: easy way to access glob

Plotting
~~~~~~~~
-  ``sc.hex2rgb()/sc.rgb2hex()``: convert between different color conventions
-  ``sc.vectocolor()``: map a list of sequential values onto a list of colors
-  ``sc.gridcolors()``: map a list of qualitative categories onto a list of colors
-  ``sc.plot3d()/sc.surf3d()``: easy way to render 3D plots
-  ``sc.boxoff()``: turn off top and right parts of the axes box
-  ``sc.commaticks()``: convert labels from "10000" and "1e6" to "10,000" and "1,000,0000"
-  ``sc.SIticks()``: convert labels from "10000" and "1e6" to "10k" and "1m"
-  ``sc.maximize()``: make the figure fill the whole screen
-  ``sc.savemovie()``: save a sequence of figures as an MP4 or other movie

Parallelization
~~~~~~~~~~~~~~~
-  ``sc.parallelize()``: as-easy-as-possible parallelization
-  ``sc.loadbalancer()``: very basic load balancer

Other utilities
~~~~~~~~~~~~~~~
-  ``sc.readdate()``: convert strings to dates using common formats
-  ``sc.tic()/sc.toc()``: simple method for timing durations
-  ``sc.runcommand()``: simple way of executing shell commands (shortcut to ``subprocess.Popen()``)
-  ``sc.dcp()``: simple way of copying objects (shortcut to ``copy.deepcopy()``)
-  ``sc.pr()``: print full representation of an object, including methods and each attribute
-  ``sc.heading()``: print text as a 'large' heading
-  ``sc.colorize()``: print text in a certain color
-  ``sc.sigfigs()``: truncate a number to a certain number of significant figures


Installation and run instructions
---------------------------------

1. Install Sciris: ``pip install sciris``

2. Use Sciris: ``import sciris as sc``

3. Do science (left as an exercise to the reader).


.. |Sciris showcase| image:: https://github.com/sciris/sciris/raw/main/docs/sciris-showcase-code.png
.. |Sciris output| image:: https://github.com/sciris/sciris/raw/main/docs/sciris-showcase-output.png