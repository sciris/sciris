=========
Tutorials
=========

These tutorials illustrate the main features of Sciris. They roughly make the most sense in the order listed, but can be done in any order. Each one should take about 5 minutes to skim through, or 20-30 minutes to go through in detail. Each tutorial contains a link to an interactive version running on `Binder <http://mybinder.org>`_.

These tutorials assume you are already familiar with `Numpy <https://numpy.org/doc/stable/user/index.html>`__, `Matplotlib <https://matplotlib.org/stable/index.html>`__, and (recommended but not essential) `pandas <https://pandas.pydata.org/docs/>`__. If you aren't, you'll be much happier if you learn those first, since Sciris builds off them.

Questions? Did you complete all the tutorials and want to claim your free "I Am A Scirientist" t-shirt?† Reach out to us at info@sciris.org. 


.. grid:: 2
    :gutter: 4

    .. grid-item-card:: 1 Whirlwind tour
        :link: tut_intro
        :link-type: doc
        :img-top: img-intro.png

        A quick high-level overview of all of Sciris' most important features.


    .. grid-item-card:: 2 Array tools
        :link: tut_arrays
        :link-type: doc
        :img-top: img-arrays.png

        Arrays are at the core of scientific computing, and this tutorial will go through some tools that will make it easier to work with them.


    .. grid-item-card:: 3 Dictionaries and dataframes
        :link: tut_dicts
        :link-type: doc
        :img-top: img-dicts.png

        Data are only as good as the container they're stored in, so this tutorial goes through Sciris' two main containers (ordered dictionaries and dataframes).


    .. grid-item-card:: 4 Files and versioning
        :link: tut_files
        :link-type: doc
        :img-top: img-files.png
        
        All that data has to come from and go somewhere: this tutorial covers saving and loading files, including with version metadata.


    .. grid-item-card:: 5 Printing
        :link: tut_printing
        :link-type: doc
        :img-top: img-printing.png

        This tutorial covers tools for quickly printing out information about objects and data. (As you can see, you won't have an opportunity to make any cool plots in this tutorial, but it's worth doing anyway, promise!)


    .. grid-item-card:: 6 Plotting
        :link: tut_plotting
        :link-type: doc
        :img-top: img-plotting.png

        If you don't plot it, did it even happen? This tutorial covers Sciris' plotting tools and shortcuts that extend :mod:`Matplotlib <matplotlib>`.


    .. grid-item-card:: 7 Parallelization and profiling
        :link: tut_parallel
        :link-type: doc
        :img-top: img-parallel.png

        When you need answers faster: this tutorial covers running your code in parallel, as well as how to profile it to help find performance improvements.


    .. grid-item-card:: 8 Dates and times
        :link: tut_dates
        :link-type: doc
        :img-top: img-dates.png
        
        Compared to nice simple numbers that behave in nice simple ways, dates can be a pain to work with. This tutorial goes through some ways to manipulate them more easily, as well as how to time parts of your code.


    .. grid-item-card:: 9 Miscellaneous utilities
        :link: tut_utils
        :link-type: doc
        :img-top: img-utils.png
        
        Sometimes (probably often), you just need to do some random tedious task. Is there a shortcut for in Sciris? Find out here.


    .. grid-item-card:: 10 Advanced features
        :link: tut_advanced
        :link-type: doc
        :img-top: img-advanced.gif
        
        You probably won't often need these tools, but we think they're cool, and they're here waiting for you just in case.


Full contents
-------------

.. toctree::
   :maxdepth: 3

   tut_intro
   tut_arrays
   tut_dicts
   tut_files
   tut_printing
   tut_plotting
   tut_parallel
   tut_dates
   tut_utils
   tut_advanced

† There is no free t-shirt, sorry. But you made it to the end, yay!