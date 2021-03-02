What's new
==========

All notable changes to this project will be documented in this file.

By import convention, components of the Sciris library are listed beginning with ``sc.``, e.g. ``sc.odict()``.


Version 1.0.1 (2021-03-01)
---------------------------
1. Fixed bug with Matplotlib 3.4.0 also defining colormap ``'turbo'``, which caused Sciris to fail to load.
2. Added a new function, ``sc.orderlegend()``, that lets you specify the order you want the legend items to appear.
3. Fixed bug with paths returned by ``sc.getfilelist(nopath=True)``.
4. Fixed bug with ``sc.loadjson()`` only reading from a string if ``fromfile=False``.
5. Fixed recursion issue with printing ``sc.Failed`` objects.
6. Changed ``sc.approx()`` to be an alias to ``np.isclose()``; this function may be removed in future versions.
7. Changed ``sc.findinds()`` to call ``np.isclose()``, allowing for greater flexibility.
8. Changed the ``repr`` for ``sc.objdict()`` to differ from ``sc.odict()``.
9. Improved ``sc.maximize()`` to work on more platforms (but still not inline or on Macs).
10. Improved the flexiblity of ``sc.htmlify()`` to handle tabs and other kinds of newlines.
11. Added additional checks to ``sc.prepr()`` to avoid failing on recursive objects.
12. Updated ``sc.mergedicts()`` to return the same type as the first dict supplied.
13. Updated ``sc.readdate()`` and ``sc.date()`` to support timestamps as well as strings.
14. Updated ``sc.gitinfo()`` to try each piece independently, so if it fails on one (e.g., extracting the date) it will still return the other pieces (e.g., the hash).
15. Pinned ``xlrd`` to 1.2.0 since later versions fail to read xlsx files.



Version 1.0.0 (2020-11-30)
--------------------------
This major update (and official release!) includes many new utilities adopted from the `Covasim <http://covasim.org>`__ and `Atomica <http://atomica.tools>`__ libraries, as well as important improvements and bugfixes for parallel processing, object representation, and file I/O.

New functions
~~~~~~~~~~~~~

Math functions
^^^^^^^^^^^^^^
1. ``sc.findfirst()`` and ``sc.findlast()`` return the first and last indices, respectively, of what ``sc.findinds()`` would return. These keywords (``first`` and ``last``) can also be passed directly to ``sc.findinds()``.
2. ``sc.randround()`` probabilistically rounds numbers to the nearest integer; e.g. 1.2 will round down 80% of the time.
3. ``sc.cat()`` is a generalization of ``np.append()``/``np.concatenate()`` that handles arbitrary types and numbers of inputs.
4. ``sc.isarray()`` checks if the object is a Numpy array.

Plotting functions
^^^^^^^^^^^^^^^^^^
1. A new diverging colormap, ``'orangeblue'``, has been added (courtesy Prashanth Selvaraj). It is rather pretty; you should try it out.
2. ``sc.get_rows_cols()`` solves the small but annoying issue of trying to figure out how many rows and columns you need to plot *N* axes. It is similar to ``np.unravel_index()``, but allows the desired aspect ratio to be varied.
3. ``sc.maximize()`` maximizes the current figure window.

Date functions
^^^^^^^^^^^^^^
1. ``sc.date()`` will convert practically anything to a date.
2. ``sc.day()`` will convert practically anything to an integer number of days from a starting point; for example, ``sc.day(sc.now())`` returns the number of days since Jan. 1st.
3. ``sc.daydiff()`` computes the number of days between two or more start and end dates.
4. ``sc.daterange()`` returns a list of date strings or date objects between the start and end dates.
5. ``sc.datetoyear()`` converts a date to a decimal year (from Romesh Abeysuriya via Atomica).

Other functions
^^^^^^^^^^^^^^^
1. The "flagship" functions ``sc.loadobj()``/``sc.saveobj()`` now have shorter aliases: ``sc.load()``/``sc.save()``. These functions can be used interchangeably.
2. A convenience function, ``sc.toctic()``, has been added that does ``sc.toc(); sc.tic()``, i.e. for sequentially timing multiple blocks of code.
3. ``sc.checkram()`` reports the current process' RAM usage at the current moment in time; useful for debugging memory leaks.
4. ``sc.getcaller()`` returns the name and line number of the calling function; useful for logging and version control purposes.
5. ``sc.nestedloop()`` iterates over lists in the specified order (from Romesh Abeysuriya via Atomica).
6. ``sc.parallel_progress()`` runs a function in parallel whilst displaying a single progress bar across all processes (from Romesh Abeysuriya via Atomica).
7. An experimental function, ``sc.asobj()``, has been added that lets any dictionary-like object be used with attributes instead (i.e. ``foo.bar`` instead of ``foo['bar']``).

Bugfixes and other improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. ``sc.parallelize()`` now uses the ``multiprocess`` library instead of ``multiprocessing``. This update fixes bugs with trying to run parallel processing in certain environments (e.g., in Jupyter notebooks). This function also returns a more helpful error message when running in the wrong context on Windows.
2. ``sc.prepr()`` has been updated to use a simpler method of parsing objects for display; this should be faster and more robust. A default 3 second time limit has also been added.
3. ``sc.savejson()`` now uses an indent of 2 by default, leading to much more human-readable JSON files.
4. ``sc.gitinfo()`` has been updated to use the code from Atomica's ``fast_gitinfo()`` instead (courtesy Romesh Abeysuriya).
5. ``sc.thisdir()`` now no longer requires the ``__file__`` argument to be supplied to get the current folder.
6. ``sc.readdate()`` can now handle a list of dates.
7. ``sc.getfilelist()`` now has more options, such as to return the absolute path or no path, as well as handling file matching patterns more flexibly.
8. ``sc.Failed`` and ``sc.Empty``, which may be encountered when loading a corrupted pickle file, are now exposed to the user (before they could only be accessed via ``sc.sc_fileio.Failed``).
9. ``sc.perturb()`` can now use either uniform or normal perturbations via the ``normal`` argument.

Renamed/removed functions
~~~~~~~~~~~~~~~~~~~~~~~~~
1. The function ``sc.quantile()`` has been removed. Please use ``np.quantile()`` instead (though admittedly, it is extremely unlikely you were using it to begin with).
2. The function ``sc.scaleratio()`` has been renamed ``sc.normsum()``, since it normalizes an array by the sum.

Other updates
~~~~~~~~~~~~~
1. Module imports were moved to inside functions, improving Sciris loading time by roughly 30%.
2. All tests were refactored to be in consistent format, increasing test coverage by roughly 50%.
3. Continuous integration testing was updated to use GitHub Actions instead of Travis/Tox.


Version 0.17.4 (2020-08-11)
---------------------------
1. ``sc.profile()`` and ``sc.mprofile()`` now return the line profiler instance for later use (e.g., to extract additional statistics).
2. ``sc.prepr()`` (also used in ``sc.prettyobj()``) can now support objects with slots instead of dicts.


Version 0.17.3 (2020-07-21)
---------------------------
1. ``sc.parallelize()`` now explicitly deep-copies objects, since on some platforms this copying does not take place as part of the parallelization process.


Version 0.17.2 (2020-07-13)
---------------------------
1. ``sc.search()`` is a new function to find nested attributes/keys within objects or dictionaries.


Version 0.17.1 (2020-07-07)
---------------------------
1. ``sc.Blobject`` has been modified to allow more flexibility with saving (e.g., ``Path`` objects).


Version 0.17.0 (2020-04-27)
---------------------------
1. ``sc.mprofile()`` has been added, which does memory profiling just like ``sc.profile()``.
2. ``sc.progressbar()`` has been added, which prints a progress bar.
3. ``sc.jsonpickle()`` and ``sc.jsonunpickle()`` have been added, wrapping the module of the same name, to convert arbitrary objects to JSON.
4. ``sc.jsonify()`` checks objects for a ``to_json()`` method, handling e.g Pandas dataframes, and falls back to ``sc.jsonpickle()`` instead of raising an exception for unknown object types.
5. ``sc.suggest()`` now uses ``jellyfish`` instead of ``python-levenshtein`` for fuzzy string matching.
6. ``sc.saveobj()`` now uses protocol 4 instead of the latest by default, to avoid backwards incompatibility issues caused by using protocol 5 (only compatible with Python 3.8).
7. ``sc.odict()`` and related classes now raise ``sc.KeyNotFoundError`` exceptions. These are derived from ``KeyError``, but fix a `bug in the string representation <https://stackoverflow.com/questions/34051333/strange-error-message-printed-out-for-keyerror>`__ to allow multi-line error messages.
8. Rewrote all tests to be pytest-compatible.


Version 0.16.8 (2020-04-11)
---------------------------
1. ``sc.makefilepath()`` now has a ``checkexists`` flag, which will optionally raise an exception if the file does (or doesn't) exist.
2. ``sc.sanitizejson()`` now handles ``datetime.date`` and ``datetime.time``.
3. ``sc.uuid()`` and ``sc.fast_uuid()`` now work with non-integer inputs, e.g., ``sc.uuid(n=10e3)``.
4. ``sc.thisdir()`` now accepts additional arguments, so can be used to form a full path, e.g. ``sc.thisdir(__file__, 'myfile.txt')``.
5. ``sc.checkmem()`` has better parsing of objects.
6. ``sc.prepr()`` now lists properties of objects, and has some aesthetic improvements.
