Overview
========

.. currentmodule:: sciris

.. image:: https://badgen.net/pypi/v/sciris/?color=blue
 :target: https://pypi.com/project/sciris

.. image:: https://static.pepy.tech/personalized-badge/sciris?period=total&units=international_system&left_color=grey&right_color=yellow&left_text=Downloads
 :target: https://pepy.tech/project/sciris

.. image:: https://img.shields.io/pypi/l/sciris.svg
 :target: https://github.com/sciris/sciris/blob/main/LICENSE

.. image:: https://github.com/sciris/sciris/actions/workflows/tests.yaml/badge.svg
 :target: https://github.com/sciris/sciris/actions/workflows/tests.yaml?query=workflow


What is Sciris?
---------------

Glad you asked! **Sciris** (http://sciris.org) is a library of tools that can help make writing scientific Python code easier and more pleasant. Built on top of `NumPy <https://numpy.org/>`__ and `Matplotlib <https://matplotlib.org/>`__, Sciris provides functions covering a wide range of common math, file I/O, and plotting operations. This means you can get more done with less code, and spend less time looking things up on StackOverflow. It was originally written to help epidemiologists and neuroscientists focus on doing science, rather than on writing coding, but Sciris is applicable across scientific domains.

Sciris is available on `PyPI <https://pypi.org/project/sciris/>`__ (``pip install sciris``) and `GitHub <https://github.com/sciris/sciris>`__. Full documentation is available at `here <http://docs.sciris.org>`__. The paper describing Sciris is available `here <http://paper.sciris.org>`__. If you have questions, feature suggestions, or would like some help getting started, please reach out to us at info@sciris.org.


Highlights
~~~~~~~~~~
Some highlights of Sciris (``import sciris as sc``):

- **Powerful containers** – The :class:`sc.odict <sc_odict.odict>` class is what ``OrderedDict`` (almost) could have been, allowing reference by position or key, casting to a NumPy array, sorting and enumeration functions, etc.
- **Array operations** – Want to find the indices of an array that match a certain value or condition? :func:`sc.findinds() <sc_math.findinds>` will do that. How about just the nearest value, regardless of exact match? :func:`sc.findnearest() <sc_math.findnearest>`. What about the last matching value? :func:`sc.findlast() <sc_math.findlast>`. Yes, you could do :func:`np.nonzero()[0][-1] <numpy.nonzero>` instead, but :func:`sc.findlast() <sc_math.findlast>` is easier to read, type, and remember, and handles edge cases more elegantly.
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

To give a based-on-a-true-story example, let's say you have a dictionary of results for multiple runs of your model, called ``results``. The output of each model run is itself a dictionary, with keys such as ``name`` and ``data``. Now let's say you want to access the data from the first model run. Using plain Python dictionaries, this would be ``results[list(results.keys())[0]]['data']``. Using a Sciris :class:`sc.objdict <sc_odict.objdict>`, this is ``results[0].data`` – almost 3x shorter.


Where has Sciris been used?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sciris is currently used by a number of scientific computing libraries, including `Atomica <http://atomica.tools>`_ and `Covasim <http://covasim.org>`__. `ScirisWeb <http://github.com/sciris/scirisweb>`_, a lightweight web framework built on top of Sciris, provides the backend for webapps such as the `Cascade Analysis Tool <http://cascade.tools>`_, `HIPtool <http://hiptool.org>`_, and `Covasim <http://app.covasim.org>`_.


Features
-------------------

Here are a few more of the most commonly used features.

Containers
~~~~~~~~~~
-  :class:`sc.odict() <sc_odict.odict>`: flexible container representing the best-of-all-worlds across lists, dicts, and arrays
-  :class:`sc.objdict() <sc_odict.objdict>`: like an odict, but allows get/set via e.g. ``foo.bar`` instead of ``foo['bar']``

Array operations
~~~~~~~~~~~~~~~~
-  :func:`sc.findinds() <sc_math.findinds>`: find indices of an array matching a value or condition
-  :func:`sc.findnearest() <sc_math.findnearest>`: find nearest matching value
-  :func:`sc.smooth() <sc_math.smooth>`: simple smoothing of 1D or 2D arrays
-  :func:`sc.isnumber() <sc_utils.isnumber>`: checks if something is any number type
-  :func:`sc.tolist() <sc_utils.promotetolist>`: converts any object to a list, for easy iteration
-  :func:`sc.toarray() <sc_utils.toarray>`: tries to convert any object to an array, for easy use with NumPy

File I/O
~~~~~~~~
-  :func:`sc.save() <sc_fileio.save>`/:func:`sc.load() <sc_fileio.load>`: efficiently save/load any Python object (via pickling)
-  :func:`sc.savejson() <sc_fileio.savejson>`/:func:`sc.loadjson() <sc_fileio.loadjson>`: likewise, for JSONs
-  :func:`sc.thisdir() <sc_fileio.thisdir>`: get current folder
-  :func:`sc.getfilelist() <sc_fileio.getfilelist>`: easy way to access glob

Plotting
~~~~~~~~
-  :func:`sc.hex2rgb() <sc_colors.hex2rgb>`/:func:`sc.rgb2hex() <sc_colors.rgb2hex>`: convert between different color conventions
-  :func:`sc.vectocolor() <sc_colors.vectocolor>`: map a list of sequential values onto a list of colors
-  :func:`sc.gridcolors() <sc_colors.gridcolors>`: map a list of qualitative categories onto a list of colors
-  :func:`sc.plot3d() <sc_plotting.plot3d>`/:func:`sc.surf3d() <sc_plotting.surf3d>`: easy way to render 3D plots
-  :func:`sc.boxoff() <sc_plotting.boxoff>`: turn off top and right parts of the axes box
-  :func:`sc.commaticks() <sc_plotting.commaticks>`: convert labels from "10000" and "1e6" to "10,000" and "1,000,0000"
-  :func:`sc.SIticks() <sc_plotting.SIticks>`: convert labels from "10000" and "1e6" to "10k" and "1m"
-  :func:`sc.maximize() <sc_plotting.maximize>`: make the figure fill the whole screen
-  :func:`sc.savemovie() <sc_plotting.savemovie>`: save a sequence of figures as an MP4 or other movie

Parallelization
~~~~~~~~~~~~~~~
-  :func:`sc.parallelize() <sc_parallel.parallelize>`: as-easy-as-possible parallelization
-  :func:`sc.loadbalancer() <sc_profiling.loadbalancer>`: very basic load balancer

Other utilities
~~~~~~~~~~~~~~~
-  :func:`sc.readdate() <sc_datetime.readdate>`: convert strings to dates using common formats
-  :func:`sc.tic() <sc_datetime.tic>`/:func:`sc.toc() <sc_datetime.toc>`: simple method for timing durations
-  :func:`sc.runcommand() <sc_utils.runcommand>`: simple way of executing shell commands (shortcut to ``subprocess.Popen()``)
-  :func:`sc.dcp() <sc_utils.dcp>`: simple way of copying objects (shortcut to :func:`copy.deepcopy()`)
-  :func:`sc.pr() <sc_printing.pr>`: print full representation of an object, including methods and each attribute
-  :func:`sc.heading() <sc_printing.heading>`: print text as a 'large' heading
-  :func:`sc.colorize() <sc_printing.colorize>`: print text in a certain color
-  :func:`sc.sigfig() <sc_printing.sigfig>`: truncate a number to a certain number of significant figures
-  :func:`sc.search() <sc_nested.search>`: search for a key, attribute, or value in a complex object
-  :func:`sc.equal() <sc_nested.equal>`: check whether two or more complex objects are equal


Installation and run instructions
---------------------------------

1. Install Sciris: ``pip install sciris`` (or ``conda install -c conda-forge sciris``)

2. Use Sciris: ``import sciris as sc``

3. Do science (left as an exercise to the reader).


Citation
--------

To cite Sciris, cite the `paper <http://paper.sciris.org>`_:

  Kerr CC, Sanz-Leon P, Abeysuriya RG, Chadderdon GL, Harbuz VS, Saidi P, Quiroga M, Martin-Hughes R, Kelly SL, Cohen JA, Stuart RM, Nachesa AN. **Sciris: Simplifying scientific software in Python.** *Journal of Open Source Software* 2023 **8** (88):5076. DOI: https://doi.org/10.21105/joss.05076

The citation is also available in `BibTeX format`_.


.. |Sciris showcase| image:: https://github.com/sciris/sciris/raw/main/docs/sciris-showcase-code.png
.. |Sciris output| image:: https://github.com/sciris/sciris/raw/main/docs/sciris-showcase-output.png
.. _BibTeX format: https://github.com/sciris/sciris/raw/main/docs/sciris-citation.bib
