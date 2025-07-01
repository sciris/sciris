What's new
==========

.. currentmodule:: sciris

All major updates to Sciris are documented here.

By import convention, components of the Sciris library are listed beginning with ``sc.``, e.g. ``sc.odict()``.


Version 3.2.2 (2025-07-01)
--------------------------
#. :class:`sc.profile() <sc_profiling.profile>` has been converted from a function to a class, with many new features, including improved display, exporting to dataframe, plotting, etc.
#. :class:`sc.listfuncs() <sc_profiling.listfuncs>` will list all functions across supplied modules, classes, and (of course) functions.
#. :class:`sc.timer() <sc_datetime.timer>` can now be used as a function decorator.
#. :func:`sc.movelegend() <sc_plotting.movelegend>` will move a legend from one axes to another.
#. :func:`sc.getrowscols() <sc_plotting.getrowscols>` now deletes rather than hides extra axes.
#. Added ``llms.txt`` since our robot overlords told us to.

Version 3.2.1 (2025-04-19)
--------------------------
#. Replaced :func:`sc.datetoyear(reverse=True) <sc_datetime.datetoyear>` with :func:`sc.yeartodate() <sc_datetime.yeartodate>`.
#. Added positional and keyword arguments to :func:`sc.date() <sc_datetime.date>`, e.g. ``sc.date(year=2002, month=4, day=4)`` or ``sc.date(2002, 04, 04)``.
#. Renamed ``doprint`` to ``verbose`` in :func:`sc.toc() <sc_datetime.toc>`, for consistency with other functions.
#. :func:`sc.load() <sc_fileio.load>` can now load gzipped plain text files (rather than just pickles); use ``sc.load(filename, method='string')`` or ``sc.load(filename, method='bytestr')`` to avoid trying to load as a pickle first.
#. :func:`sc.getfilelist() <sc_fileio.getfilelist>` now skips blank entries.
#. :func:`sc.jsonify() <sc_fileio.jsonify>` avoids recursion in cases where an object has a ``to_json()`` method that itself calls ``sc.jsonify()``.
#. :func:`sc.findnearest() <sc_math.findnearest>` now works for arbitrary scalar objects, not just numbers.
#. :func:`sc.perturb() <sc_math.perturb>` now works with 0 or 1 arguments, and can modify an input array.
#. :func:`sc.iterobj() <sc_nested.iterobj>` can now optionally descend into tuples; use ``sc.iterobj(atomic='default-tuple')`` to use this behavior. :class:`sc.IterObj() <sc_nested.IterObj>` now has a ``disp()`` method, and a bug regarding unintentional skipping of Python built-ins was fixed. ``sc.IterObj.to_df()`` also now skips the object root by default, listing only the subcomponents of the object.
#. :func:`sc.setnested() <sc_nested.setnested>` can now be used to set a single key, e.g. ``sc.setnested(mydict, 'a', 4)``.
#. The `ansicolors <https://pypi.org/project/ansicolors/>`_ module is now available as ``sc.ansi``, e.g. ``print(sc.ansi.green('this is green'))``.
#. :class:`sc.tracecalls() <sc_profiling.tracecalls>` now works with a default trace, provides more control of what gets traced, and has a ``check_expected()`` method that compares actual calls vs. expected calls.
#. :func:`sc.parse_env() <sc_settings.parse_env>` now accepts type inputs (e.g. ``sc.parse_env('MY_VAR', default=3.5, which=float)``.
#. NumPy 2.0 changed the default repr to show types, e.g. ``np.float64(3.5)`` instead of ``3.5``. By default, Sciris now reverses this behavior; use ``sc.options(show_type=True)`` or set ``SCIRIS_SHOW_TYPE=1`` to revert to NumPy's default behavior.
#. :func:`sc.ismodule() <sc_utils.ismodule>` has been added as a shortcut for checking whether an object is a module.
#. :func:`sc.isfunc() <sc_utils.isfunc>` now correctly catches built-in functions and methods.
#. :func:`sc.importbypath() <sc_utils.importbypath>` now has an ``overwrite`` argument to specify whether to overwrite an existing module of the same name.
#. Switched build from ``setup.py`` to ``pyproject.toml``.


Version 3.2.0 (2024-09-24)
--------------------------

New features
~~~~~~~~~~~~
#. :func:`sc.loadany() <sc_fileio.loadany>` will try to load a file using any of the known formats (pickle, JSON, YAML, Excel, CSV, zip, or plain text).
#. :func:`sc.sem() <sc_math.sem>` calculates the standard error of the mean.
#. :func:`sc.sigfiground() <sc_printing.sigfiground>` rounds an array to a specified number of significant figures.
#. :class:`sc.tracecalls() <sc_profiling.tracecalls>` traces every function call within a context block (alias to ``sys.setprofile``).
#. :func:`sc.isfunc() <sc_utils.isfunc>` checks whether something is a function (or a class method).
#. :class:`sc.dataframe() <sc_dataframe.dataframe>` now has an :func:`addcol() <sc_dataframe.dataframe.addcol>` method, which adds one or more columns to the dataframe.
#. :class:`sc.dataframe() <sc_dataframe.dataframe>` now has an :func:`enumrows() <sc_dataframe.dataframe.enumrows>` method, which is similar to :func:`pd.iterrows() <pandas.iterrows>`, but up to 50x faster.

Bugfixes
~~~~~~~~
#. Previously, :func:`sc.importbypath() <sc_utils.importbypath>` would sometimes fail to import a module correctly if a module with the same name was already imported. This has been fixed.
#. Previously, :func:`sc.inclusiverange() <sc_math.inclusiverange>` would stretch steps in order to exactly match ``start`` and ``stop`` (e.g., ``sc.inclusiverange(0,10,3)`` would stretch the step to ``3.333``). It now defaults to not stretching the step. Previous behavior can be restored via ``sc.inclusiverange(..., stretch=True)```.
#. Previously, in :func:`sc.parallel() <sc_parallel.parallel>`, specifying ``interval`` without also specifying ``maxcpu`` had no effect. Now it will still schedule the jobs on intervals, but with ``maxcpu=1.0`` by default.

Other changes
~~~~~~~~~~~~~
#. :func:`sc.makenested() <sc_nested.makenested>` and :func:`sc.setnested() <sc_nested.setnested>` are now more flexible and can operate on objects. (Thanks to `Kelvin Burke <https://github.com/kelvinburke>`_ for this feature.)
#. :func:`sc.search() <sc_nested.search>` has been completely rewritten, and can now be used to e.g. find objects of a certain type in another object.
#. :func:`sc.datedelta() <sc_datetime.datedelta>` can now handle fractional years.
#. :func:`sc.datetoyear() <sc_datetime.datetoyear>` now has a ``reverse`` argument for converting years to dates.
#. :func:`sc.profile() <sc_profiling.profile>` now allows modules and classes to be followed, not just functions.
#. :func:`sc.uniquename() <sc_utils.uniquename>` now has ``human`` (more verbose) and ``suffix`` (positioned after the counter) arguments.
#. :func:`sc.objatt() <sc_printing.objatt>`, :func:`sc.objmeth() <sc_printing.objmeth>`, and :func:`sc.objprop() <sc_printing.objprop>` all now have a ``return_keys`` argument.
#. :func:`sc.sha() <sc_utils.sha>` now has an ``asint`` argument for converting the digest to an integer.
#. Imports have been changed: Sciris internally uses absolute rather than relative imports, and ``pylab`` has been replaced with ``matplotlib.pyplot``. These should not impact the user, but improves load time.


Version 3.1.7 (2024-07-10)
--------------------------
#. Updated :func:`sc.asd() <sc_asd.asd>` to handle negative values in the objective function. (Thanks to `Eloisa Perez-Bennetts <https://github.com/epbennetts>`_ for this feature.)
#. Updated :class:`sc.cprofile() <sc_profiling.cprofile>` with new display options, including a ``maxitems`` argument.
#. Improved :func:`sc.jsonify() <sc_fileio.jsonify>`, including custom and recursive object parsing.
#. Fixed a bug preventing :class:`sc.odict.copy() <sc_odict.odict>` from being sorted by a specified order.
#. Fixed a bug in :func:`sc.getrowscols() <sc_plotting.getrowscols>` that raised an exception when called with ``n=1``.
#. :meth:`sc.options.use_style() <sc_settings.ScirisOptions.use_style>` now resets the style before applying a new one.


Version 3.1.6 (2024-03-31)
--------------------------
#. Added a new profiler, :class:`sc.cprofile() <sc_profiling.cprofile>`, as an interface to Python's built-in `cProfile <https://docs.python.org/3/library/profile.html>`_.
#. Updated :func:`sc.iterobj() <sc_nested.iterobj>` to include several new arguments: ``skip`` will skip objects to avoid iterating over; ``depthfirst`` switches between depth-first (default) and breadth-first (new) iteration options; ``flatten`` returns object traces as strings rather than tuples; and ``to_df`` converts the output to a dataframe.
#. Pretty-repr functions and classes (e.g. :func:`sc.pr() <sc_printing.pr>`, :class:`sc.prettyobj() <sc_printing.prettyobj>`) now include protections against infinite recursion. ``sc.prettyobj()`` was linked back to ``sc.sc_utils`` to prevent unpickling errors (partially reversing the change in version 3.1.4).
#. :class:`sc.dictobj.copy() <sc_odict.dictobj>` now returns another ``dictobj`` (previously it returned a ``dict``).
#. :func:`sc.require() <sc_versioning.require>` has been reimplemented to be faster and avoid ``pkg_resources`` deprecations.


Version 3.1.5 (2024-03-18)
--------------------------
#. Added a new :class:`sc.quickobj() <sc_printing.quickobj>` class, which is like :class:`sc.prettyobj() <sc_printing.prettyobj>` except it only prints attribute names, not values. This is useful for large objects that can be slow to print. 
#. :func:`sc.pr() <sc_printing.pr>` has a new ``vals=False`` argument that skips printing attribute values. The default column width was also increased (from 18 to 22 chars).
#. A new function :func:`sc.ifelse() <sc_utils.ifelse>` was added, which is a shortcut to finding the first non-``None`` (or non-``False``) value in a list.
#. Updated :func:`sc.iterobj() <sc_nested.iterobj>` to prevent recursion, and to handle atomic classes (i.e. objects that are not descended into) more flexibly.


Version 3.1.4 (2024-03-11)
--------------------------
#. Fixed failures of pretty-repr (e.g. :func:`sc.pr() <sc_printing.pr>` and :class:`sc.prettyobj() <sc_printing.prettyobj>`) for objects with invalid properties (e.g., properties that rely on missing/invalid attributes). ``sc.prettyobj()`` was also moved from ``sc_utils`` to ``sc_printing``.
#. Added additional flexibility for loading zip files (:func:`sc.loadzip() <sc_fileio.loadzip>`); saving zip files (:func:`sc.savezip() <sc_fileio.savezip>`) now saves text as plain text even with ``tobytes=True``.
#. :func:`sc.dcp(die=False) <sc_utils.dcp>` now passes ``die=False`` to :func:`sc.cp() <sc_utils.cp>`, and will return the original object if it cannot be copied.
#. :func:`sc.urlopen() <sc_utils.urlopen>` now has additional response options, including ``'json'`` and ``'full'``.


Version 3.1.3 (2024-02-07)
--------------------------
#. :func:`sc.equal() <sc_nested.equal>` now parses the structure of all objects (not just the first), with missing keys/attributes listed in the output table. It also now allows for a ``detailed=2`` argument, which prints the value of each key/attribute in each object. (Thanks to `Kelvin Burke <https://github.com/kelvinburke>`_ for this and other features.)
#. Fixed incorrect keyword arguments (``iterkwargs``) in :func:`sc.parallelize() <sc_parallel.parallelize>` when using ``thread`` or another non-copying parallelizer, when the ``iterkwargs`` are *not* consistent between iterations.
#. Fixed incorrect printout on the final iteration of :func:`sc.asd() <sc_asd.asd>` in verbose mode.
#. Fixed incorrect plotting of non-cumulative data in :func:`sc.stackedbar() <sc_plotting.stackedbar>`.
#. :func:`sc.download(..., save=False) <sc_utils.download>` now returns an :class:`sc.objdict <sc_odict.objdict>` (instead of an :class:`sc.odict <sc_odict.odict>`).
#. :func:`sc.checktype(obj, 'arraylike') <sc_utils.checktype>` is now more robust to handling non-array-like objects (e.g., a ragged list of lists now returns ``False`` instead of raising an exception).
#. :func:`sc.require() <sc_versioning.require>` now takes an optional ``message`` argument, allowing for a custom message if the requirement(s) aren't met.
#. Removed ``object`` from the list of classes shown by :func:`sc.prepr() <sc_printing.prepr>` (since all objects derive from ``object``).


Version 3.1.2 (2023-11-01)
--------------------------
#. Updated logic for :func:`sc.iterobj() <sc_nested.iterobj>` and added a new :class:`sc.IterObj() <sc_nested.IterObj>` class, allowing greater customization of how objects are iterated over.
#. Fixed a bug in which 3D plotting functions (e.g. :func:`sc.bar3d() <sc_plotting.bar3d>`) would create a new figure even if an existing axes instance was passed.


Version 3.1.1 (2023-10-29)
--------------------------
#. :class:`sc.odict <sc_odict.odict>` now supports steps in slice-based indexing, e.g.: ``myodict['foo':'bar':5]`` will select every 5th item from ``'foo'`` to ``'bar'`` inclusive.
#. :meth:`sc.odict.copy() <sc_odict.odict.copy>` now behaves the same as ``dict.copy()``; the previous behavior (which copied an item) is deprecated. Instead of ``mydict.copy(oldkey, newkey)``, use ``mydict[newkey] = sc.dcp(mydict[oldkey])`` instead.
#. :func:`sc.download() <sc_utils.download>` now defaults to expecting ``filename:URL`` pairs rather than ``URL:filename`` pairs (e.g. ``sc.download({'wikipedia.html':'http://wikipedia.org/index.html'})``, though it can accept either as long as ``http`` appears in one.
#. :func:`sc.parallelize() <sc_parallel.parallelize>` has more robust error handling (previously, certain types of exceptions, such as HTTP errors, were not caught even if ``die=False``).
#. :func:`sc.load() <sc_fileio.load>` has improved support for loading old pickles, including a new ``NoneObj`` class that is used when the user explicitly remaps an old class/function to ``None``.
#. :func:`sc.sanitizefilename() <sc_fileio.sanitizefilename>` now excludes newlines and tabs even when ``strict=False``.
#. :func:`sc.runcommand() <sc_utils.runcommand>` now prints out terminal output in real time if ``wait=False``.
#. Added support for Python 3.12. Note: ``line_profiler`` is not compatible with Python 3.12 at the time of writing, so :func:`sc.profile() <sc_profiling.profile>` is not available on Python 3.12.


Version 3.1.0 (2023-08-13)
--------------------------

New features
~~~~~~~~~~~~
#. :func:`sc.equal() <sc_nested.equal>` compares two (or more) arbitrarily complex objects. It can handle arrays, dataframes, custom objects with no ``__eq__`` method defined, etc. It can also print a detailed comparison of the objects.
#. :func:`sc.nanequal() <sc_math.nanequal>` is an extension of :func:`np.array_equal() <numpy.array_equal>` to handle a broader range of types (e.g., mixed-type ``object`` arrays that cannot be cast to float). Other ``NaN``-related methods have also been updated to be more robust.
#. :func:`sc.manualcolorbar() <sc_colors.manualcolorbar>` allows highly customized colorbars to be added to plots, including to plots with no "mappable" data (e.g., scatterplots).
#. Added :meth:`sc.options.reset() <sc_settings.ScirisOptions.reset>` as an alias to ``sc.options.set('defaults')``.

Bugfixes
~~~~~~~~
#. Sciris is now compatible with a broader range of dependencies (e.g., Python, NumPy, pandas, and Matplotlib); in most cases, the latest version of Sciris is now backwards-compatible with all dependency versions since January 2021.
#. Updated :func:`sc.pr() <sc_printing.pr>` to include class attributes (as well as instance attributes), and added a new function :func:`sc.classatt() <sc_printing.classatt>` to list them.
#. :func:`sc.readdate() <sc_datetime.readdate>` now returns ``datetime`` objects unchanged, rather than raising an exception.
#. Fixed ``repr`` for empty :class:`sc.objdict() <sc_odict.objdict>`.
#. Fixed transposed ordering for :func:`sc.bar3d() <sc_plotting.bar3d>`.

Other changes
~~~~~~~~~~~~~
#. :func:`sc.load() <sc_fileio.load>` has been significantly refactored to be simpler and more robust. Pandas' :func:`pd.read_pickle() <pandas.read_pickle>` is now included as one of the default unpickling options. Unsuccessful unpickling now always produces a :class:`Failed <sc_fileio.Failed>` object, with as much data retained as possible.
#. :func:`sc.jsonpickle() <sc_fileio.jsonpickle>` and :func:`sc.jsonunpickle() <sc_fileio.jsonunpickle>` can now save to/read from files directly.
#. Updated :func:`sc.toarray() <sc_utils.toarray>` to use ``dtype=object`` instead of ``dtype=str`` by default; otherwise, all elements in mixed-type arrays (e.g. ``[1,'a']``) are cast to string.
#. :class:`sc.dataframe <sc_dataframe.dataframe>` has a new ``equal`` class method (e.g. ``sc.dataframe.equal(df1, df2)``), and revised ``equals()`` and ``==`` behavior to match pandas.
#. Improved robustness of :func:`sc.parallelize() <sc_parallel.parallelize>`, especially when using custom parallelizers, including more options for customizing the global dictionary.
#. :class:`sc.timer() <sc_datetime.timer>` objects can now be added, which will concatenate all the times.
#. Added an option to run :func:`sc.benchmark() <sc_profiling.benchmark>` in parallel (to test the full capacity of the machine rather than a single core).
#. :func:`sc.iterobj() <sc_nested.iterobj>` now provides more options for controlling how the object is iterated, and no longer (by default) descends into NumPy arrays, pandas DataFrames, etc. :func:`sc.search() <sc_nested.search>` also has additional options.
#. Updated 3D plotting functions (:func:`sc.plot3d() <sc_plotting.plot3d>`, :func:`sc.surf3d() <sc_plotting.surf3d>`, etc.) to have more flexibility of data input, consistency, and robustness.


Version 3.0.0 (2023-04-20)
--------------------------

This version's major changes include:

#. **New Parallel class**: A new :class:`sc.Parallel() <sc_parallel.Parallel>` class allows finer-grained managing of parallel processes, including automatic progress bars, better exception handling, and asynchronous running.
#. **Better versioning**: New functions :func:`sc.metadata() <sc_versioning.metadata>`, :func:`sc.savearchive() <sc_versioning.savearchive>`, and :func:`sc.loadarchive() <sc_versioning.loadarchive>` make it easier to store and save metadata along with objects.
#. **Faster data structures**: :class:`sc.odict() <sc_odict.odict>` and :class:`sc.dataframe() <sc_dataframe.dataframe>` have both been reimplemented for better performance and with additional methods.
#. **Easier imports**: :func:`sc.importbypath() <sc_utils.importbypath>` lets you load a module into Python by providing the folder or filename (useful for loading one-off scripts, or two versions of the same library).
#. **Better documentation**: A comprehensive set of tutorials has been added to the documentation, and the documentation has been rewritten in a new style.


Improvements and new features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Parallelization
^^^^^^^^^^^^^^^^^^
#. There is a new :class:`sc.Parallel() <sc_parallel.Parallel>` class, which is used to implement the (more or less unchanged) :func:`sc.parallelize() <sc_parallel.parallelize>` function.
#. :func:`sc.parallelize() <sc_parallel.parallelize>` now has a ``progress`` argument that will show a progress bar; the ``returnpool`` argument has been removed (use :class:`sc.Parallel() <sc_parallel.Parallel>` instead).


2. Dataframe
^^^^^^^^^^^^
#. Better implementation of underlying logic, leading to significant performance increases in some cases (e.g., iteratively appending rows).
#. Numerous methods have been renamed, modified, or added, specifically: ``append``, ``col_index``, ``col_name``, ``findind``, ``findinds``, ``merge``, ``popcols``, ``poprow``, ``poprows``, and ``sort``.
#. Keyword arguments are now interpreted as columns, e.g. ``df = sc.dataframe(a=[1,2], b=[3,4])``.
#. Better handling of (and preservation) of ``dtypes`` for dataframe columns, including a new :meth:`df.set_dtypes() <sc_dataframe.dataframe.set_dtypes>` method.
#. Dataframes now support equality checks.


3. Time/date
^^^^^^^^^^^^
#. Support for ``pandas`` and ``Numpy`` datetime objects.
#. New :class:`sc.timer <sc_datetime.timer>` attributes and methods: :obj:`sc.rawtimings <sc_datetime.timer.rawtimings>`, :meth:`sc.sum() <sc_datetime.timer.sum>`, :meth:`sc.min() <sc_datetime.timer.min>`, :meth:`sc.max() <sc_datetime.timer.max>`, :meth:`sc.mean() <sc_datetime.timer.mean>`, :meth:`sc.std() <sc_datetime.timer.std>`.
#. :class:`sc.timer <sc_datetime.timer>` now displays time in human-appropriate units (e.g., 3.4 μs instead of 0.0000034 s) by default, or accepts a ``unit`` argument.
#. New :func:`sc.time() <sc_datetime.time>` alias for :func:`time.time()`.
#. :func:`sc.datedelta() <sc_datetime.datedelta>` can now operate on a list of dates.
#. :func:`sc.randsleep() <sc_datetime.randsleep>` now accepts a ``seed`` argument.
#. More accurate computation of self-time in :func:`sc.timedsleep() <sc_datetime.timedsleep>`.


4. Files
^^^^^^^^
#. A new function :func:`sc.unzip() <sc_fileio.unzip>` extracts zip files to disk, while :func:`sc.loadzip() <sc_fileio.loadzip>` now defaults to loading the zip file contents to memory. :func:`sc.savezip() <sc_fileio.savezip>` can now save both data and files, and its ``filelist`` argument has been renamed ``files``.
#. If a saved file can't be unpickled, :func:`sc.load() <sc_fileio.load>` now defaults to using ``dill``, and has more robust error handling (see also "versioning" updates below).
#. :func:`sc.makefilepath() <sc_fileio.makefilepath>` now defaults to ``makedirs=False``.
#. File save functions now make new subfolders by default
#. :func:`sc.save() <sc_fileio.save>` now has an ``allow_empty`` argument (instead of ``die='never'``).
#. :func:`sc.glob() <sc_fileio.glob>` is a new alias for :func:`sc.getfilelist() <sc_fileio.getfilelist>`.
#. :func:`sc.thisdir() <sc_fileio.thisdir>` now gives a correct answer when running in a Jupyter notebook.


5. Printing
^^^^^^^^^^^
#. :func:`sc.progressbar() <sc_printing.progressbar>` can now be used to wrap an iterable, in which case it acts as an alias to ``tqdm.tqdm()``.
#. The new :func:`sc.progressbars() <sc_printing.progressbars>` class will create and manage multiple progress bars, which can be useful for monitoring multiple parallel long-running jobs.
#. New functions :func:`sc.arraymean() <sc_printing.arraymean>` and :func:`sc.arraymedian() <sc_printing.arraymedian>` can be used to quickly summarize an array. To print rather than return a string, use :func:`sc.printmean() <sc_printing.printmean>` and :func:`sc.printmedian() <sc_printing.printmedian>`.
#. The new function :func:`sc.humanize_bytes() <sc_printing.humanize_bytes>` will convert a number of bytes into a human-readable number (e.g. ``32975281`` to ``32.975 MB``).
#. The new function :func:`sc.readjson() <sc_fileio.readjson>`  will read a JSON from a string (alias to :func:`sc.loadjson(string=...) <sc_fileio.loadjson>`); likewise :func:`sc.readyaml() <sc_fileio.readyaml>`. :func:`sc.printjson() <sc_fileio.printjson>` and print an object as if it was a JSON. 
#. :func:`sc.printarr() <sc_printing.printarr>` now has configurable decimal places (``decimals`` argument) and can return a string instead of printing (``doprint=False``).
#. :func:`sc.pp() <sc_utils.pp>` no longer casts objects to JSON first (see :func:`sc.printjson() <sc_fileio.printjson>` for that).
#. :func:`sc.sigfigs() <sc_printing.sigfigs>` is a new alias of :func:`sc.sigfig() <sc_printing.sigfig>`.


6. Profiling
^^^^^^^^^^^^
#. The new :func:`sc.benchmark() <sc_profiling.benchmark>` function runs tests on both regular Python and Numpy operations and reports the performance of the current machine.
#. :func:`sc.checkmem() <sc_profiling.checkmem>` now returns a dataframe, can descend multiple levels through an object, reports subtotals, and has an ``order`` argument instead of ``alphabetical``.


7. Versioning
^^^^^^^^^^^^^
#. A new versioning module has been added.
#. A new function :func:`sc.metadata() <sc_versioning.metadata>` gathers all relevant metadata and returns a dict that can be used for versioning.
#. A pair of new functions :func:`sc.savearchive() <sc_versioning.savearchive>` and :func:`sc.loadarchive() <sc_versioning.loadarchive>`, provide a way to automatically save metadata along with an object for better versioning.
#. Known regressions from older library versions are now automatically handled by :func:`sc.load() <sc_fileio.load>` (e.g., ``pandas`` v2.0 dataframes cannot be loaded in v1.5, and vice versa).
#. :func:`sc.require() <sc_versioning.require>` now has the option to raise a warning instead of an error if a module is not found.


8. Math
^^^^^^^
#. :func:`sc.findnans() <sc_math.findnans>` is a new alias for ``sc.findinds(np.isnan(data))``. :func:`sc.rmnans() <sc_math.rmnans>` is a new alias for :func:`sc.sanitize() <sc_math.sanitize>`.
#. :func:`sc.randround() <sc_math.randround>` now works with multidimensional arrays. (Thanks to `Jamie Cohen <https://github.com/jamiecohen>`_ for the suggestion.)
#. :func:`sc.smoothinterp() <sc_math.smoothinterp>` now defaults to ``ensurefinite=True``.
#. :func:`sc.asd() <sc_asd.asd>` now uses its own random number stream.
#. :func:`sc.cat() <sc_math.cat>` now works on 2D arrays.


9. Dictionaries
^^^^^^^^^^^^^^^
#. :class:`sc.odict() <sc_odict.odict>` now inherits from :class:`dict` rather than :class:`OrderedDict <collections.OrderedDict>`. This makes initialization and some other operations nearly four times faster.
#. :class:`sc.odict() <sc_odict.odict>` can now be initialized with integer keys.
#. There is a new :meth:`sc.dictobj.to_json() <sc_odict.dictobj.to_json>` method. :meth:`sc.dictobj.fromkeys() <sc_odict.dictobj.fromkeys>` is now a static method.


10. Nested objects
^^^^^^^^^^^^^^^^^^
#. Nested "dictionary" operations can now act on other types of object, including lists and regular objects.
#. :func:`sc.iterobj() <sc_nested.iterobj>` applies a function iteratively to an object.
#. :func:`sc.search() <sc_nested.search>` now works on values as well as keys/attributes.


11. System and platform
^^^^^^^^^^^^^^^^^^^^^^^
#. The new function :func:`sc.importbypath() <sc_utils.importbypath>` will import a module by path, as an alternative to standard ``import``. :func:`sc.importbyname() <sc_utils.importbyname>` also now accepts a ``path`` argument.
#. The new function :func:`sc.getuser() <sc_utils.getuser>` will return the current username (as an alias to ``getpass.getuser()``).
#. The new function :func:`sc.isjupyter() <sc_utils.isjupyter>` determines whether or not the code is running in a Jupyter notebook. Default Jupyter plotting has been updated from ``widget`` to ``retina``.


12. Plotting
^^^^^^^^^^^^
#. The two Sciris plotting styles, ``sciris.simple`` and ``sciris.fancy``, are now available through standard Matplotlib (e.g. ``pl.style.use('sciris.simple')``.
#. 3D plots (e.g. :func:`sc.plot3d() <sc_plotting.plot3d>`) will now render into existing figures and axes where possible, rather than always creating a new figure.
#. The ``freeze`` argument of :func:`sc.savefig() <sc_plotting.savefig>` has been renamed ``pipfreeze``, and ``frame`` has been replaced with ``relframe``.


13. Other
^^^^^^^^^
#. A new environment variable, ``SCIRIS_NUM_THREADS``, will set the number of threads Numpy uses (if Sciris is imported first). In some cases, more threads results in *slower* processing (and of course uses way more CPU time).
#. The new function :func:`sc.sanitizestr() <sc_printing.sanitizestr>` will sanitize an input string to e.g. ASCII-only or a valid variable name.
#. :func:`sc.download() <sc_utils.download>` now handles exceptions gracefully with ``die=False``.
#. :func:`sc.isiterable() <sc_utils.isiterable>` now has optional ``exclude`` and ``minlen`` arguments.
#. :func:`sc.flexstr() <sc_utils.flexstr>` now has more options for converting arbitrary or multiple objects to a string.
#. :func:`sc.transposelist() <sc_utils.transposelist>` has a new ``fix_uneven`` argument (previously, elements longer than the shortest sublist were silently removed).
#. :func:`sc.tryexcept() <sc_utils.tryexcept>` now has ``to_df()`` and ``disp()`` methods.


Bugfixes
~~~~~~~~
#. Fixed ``<=`` comparison in :func:`sc.compareversions() <sc_versioning.compareversions>` not handling equality.
#. Fixed the implementation of the ``midpoint`` argument in :func:`sc.vectocolor() <sc_colors.vectocolor>`.
#. Fixed corner cases where some :class:`sc.dataframe <sc_dataframe.dataframe>` methods returned ``pd.DataFrame`` objects instead.
#. Fixed corner cases where some :class:`sc.objdict <sc_odict.objdict>` methods returned :class:`sc.odict <sc_odict.odict>` objects instead.
#. :func:`sc.findinds() <sc_math.findinds>` now returns a tuple for multidimensional arrays, allowing it to be used directly for indexing.
#. :func:`sc.rmnans() <sc_math.rmnans>` now returns a zero-length array if all input is NaNs.
#. :meth:`sc.options.with_style(style) <sc_settings.ScirisOptions.with_style>` now correctly applies the style.
#. Fixed :func:`sc.daydiff() <sc_datetime.daydiff>` with one argument computing the number of days from Jan. 1st of the *current* year (instead of Jan. 1st of the provided year).
#. ``keepends`` and ``skipnans`` arguments were removed from :func:`sc.smoothinterp() <sc_math.smoothinterp>`.


Regression information
~~~~~~~~~~~~~~~~~~~~~~
#. ``tqdm`` is now a required dependency.
#. Calls to :func:`sc.makepath() <sc_fileio.makepath>` and :func:`sc.makefilepath() <sc_fileio.makefilepath>` now need to specify ``makedirs=True``.
#. :class:`sc.odict() <sc_odict.odict>` is no longer an instance of :class:`OrderedDict <collections.OrderedDict>`.
#. The ``returnpool`` argument of :func:`sc.parallelize() <sc_parallel.parallelize>` has been removed.
#. For :func:`sc.savefig() <sc_plotting.savefig>`, ``freeze`` should be renamed ``pipfreeze``, and ``frame`` should be replaced with ``relframe`` with an offset of 2 (e.g. ``frame=2 → relframe=0``).
#. :func:`sc.checkmem(..., alphabetical=True) <sc_profiling.checkmem>` has been replaced with :func:`sc.checkmem(..., order='alphabetical') <sc_profiling.checkmem>`
#. The ``Options`` class has been renamed class :class:`sc.ScirisOptions() <sc_settings.ScirisOptions>`.
#. ``sc.parallel_progress()`` has been moved to ``sc.sc_legacy``. Please use :func:`sc.parallelize(..., progress=True) <sc_parallel.parallelize>` instead.
#. ``sc.parallelcmd()`` has been moved to ``sc.sc_legacy``. Please do not use this function :)



Version 2.1.0 (2022-12-23)
--------------------------

New features
~~~~~~~~~~~~
#. ``sc.save()``/``sc.load()`` now allow files to be saved/loaded in `zstandard <https://github.com/indygreg/python-zstandard>`_ (instead of ``gzip``) format, since the former is usually faster for the same level of compression. ``sc.save()`` still uses ``gzip`` by default; the equivalent ``sc.zsave()`` uses ``zstandard`` by default. ``sc.save()`` also now has the option of not using any compression via ``sc.save(..., compression='none')``. (Thanks to `Fabio Mazza <https://github.com/fabmazz>`_ for the suggestion.)
#. Functions that returned paths as strings by default -- ``sc.thisdir()``, ``sc.getfilelist()``, ``sc.makefilepath()``, ``sc.sanitizefilename()`` -- now all have aliases that return ``Path`` objects by default: ``sc.thispath()``, ``sc.getfilepaths()``, ``sc.makepath()``, and ``sc.sanitizepath()``.
#. ``sc.thisfile()`` gets the path of the current file.
#. ``sc.sanitizecolor()`` will convert any form of color specification (e.g. ``'g'``, ``'crimson'``) into an RGB tuple.
#. ``sc.tryexcept()`` silences all (or some) exceptions in a ``with`` block.

Bugfixes
~~~~~~~~
#. Fixed bug where ``sc.save(filename=None)`` would incorrectly result in creation of a file on disk in addition to returning a ``io.BytesIO`` stream.
#. Fixed bug where ``sc.checkmem()`` would sometimes raise an exception when saving a ``None`` object to check its size.
#. Fixed bug where ``sc.loadbalancer()`` would sometimes fail if ``interval`` was 0 (it is now required to be at least 1 ms).

Other changes
~~~~~~~~~~~~~
#. ``sc.vectocolor()`` now has a ``nancolor`` argument to handle NaN values; NaNs are also now handled correctly.
#. ``sc.timer()`` now has a more compact default string representation; use ``timer.disp()`` to display the full object. In addition, ``timer.total`` is now a property instead of a function.
#. ``sc.thisdir()`` now takes a ``frame`` argument, in case the folder of a file *other* than the calling script is desired.
#. ``sc.getfilelist()`` now has a ``fnmatch`` argument, which allows for Unix-style file matching via the `fnmatch <https://docs.python.org/3/library/fnmatch.html>`_ module.
#. ``sc.importbyname()`` now has a ``verbose`` argument.
#. ``sc.promotetolist()`` and ``sc.promotetoarray()`` are now aliases of ``sc.tolist()`` and ``sc.toarray()``, rather than vice versa.


Version 2.0.4 (2022-10-25)
--------------------------
#. ``sc.stackedbar()`` will automatically plot a 2D array as a stacked bar chart.
#. ``sc.parallelize()`` now uses ``multiprocess`` again by default (due to issues with ``concurrent.futures``).
#. Added a ``die`` argument to ``sc.save()``.
#. Added a ``prefix`` argument to ``sc.urlopen()``, allowing e.g. ``http://`` to be omitted from the URL.


Version 2.0.3 (2022-10-24)
--------------------------
#. Added ``sc.linregress()`` as a simple way to perform linear regression (fit a line of best fit).
#. Improved ``sc.printarr()`` formatting.
#. Reverted incompatibility with older Matplotlib versions introduced in version 2.0.2.


Version 2.0.2 (2022-10-22)
--------------------------

Parallelization
~~~~~~~~~~~~~~~
#. The default parallelizer has been changed from ``multiprocess`` to ``concurrent.futures``. The latter is faster, but less robust (e.g., it can't parallelize lambda functions). If an error is encountered, it will automatically fall back to the former.
#. For debugging, instead of ``sc.parallelize(..., serial=True)``, you can also now use ``sc.parallelize(..., parallelizer='serial')``.
#. Arguments to ``sc.parallelize()`` are now no longer usually deepcopied, since usually they are automatically during the pickling/unpickling process. However, deepcopying has been retained for ``serial`` and ``thread`` parallelizers; to *not* deepcopy, use e.g. ``parallelizer='thread-nocopy'``.

Bugfixes
~~~~~~~~
#. ``sc.autolist()`` now correctly handles input arguments, and can be added on to other objects. (Previously, if an object was added to an ``sc.autolist``, it would itself become an ``sc.autolist``.)
#. ``sc.cat()`` now has the same default behavior as ``np.concatenate()`` for 2D arrays (i.e., concatenating rows). Use ``sc.cat(.., axis=None)`` for the previous behavior.
#. ``sc.dataframe.from_dict()`` and ``sc.dataframe.from_records()`` now return an ``sc.dataframe`` object (previously they returned a ``pd.DataFrame`` object).

Other changes
~~~~~~~~~~~~~
#. ``sc.dataframe.cat()`` will concatenate multiple objects (dataframes, arrays, etc.) into a single dataframe.
#. ``sc.dataframe().concat()`` now by default does *not* modify in-place.
#. Colormaps are now also available with a ``sciris-`` prefix, e.g. ``sciris-alpine``, as well as their original names (to avoid possible name collisions).
#. Added ``packaging`` as a dependency and removed the (deprecated) ``minimal`` install option.


Version 2.0.1 (2022-10-21)
--------------------------

New features
~~~~~~~~~~~~
#. ``sc.asciify()`` converts a Unicode input string to the closest ASCII equivalent.
#. ``sc.dataframe().disp()`` flexibly prints a dataframe (by default, all rows/columns).

Improvements
~~~~~~~~~~~~
#. ``sc.findinds()`` now allows a wider variety of numeric-but-non-array inputs.
#. ``sc.sanitizefilename()`` now handles more characters, including Unicode, and has many new options.
#. ``sc.odict()`` now allows you to delete by index instead of key.
#. ``sc.download()`` now creates folders if they do not already exist.
#. ``sc.checktype(obj, 'arraylike')`` now returns ``True`` for pandas ``Series`` objects.
#. ``sc.promotetoarray()`` now converts pandas ``Series`` or ``DataFrame`` objects into arrays.
#. ``sc.savetext()`` can now save arrays (like ``np.savetxt()``).

Bugfixes
~~~~~~~~
#. Fixed a bug with addition (concatenation) for ``sc.autolist()``.
#. Fixed a bug with the ``_copy`` argument for ``sc.mergedicts()`` being ignored.
#. ``sc.checkmem()`` no longer uses compression, giving more accurate estimates.
#. Fixed a bug with ``sc.options()`` setting the plot style automatically; a ``'default'`` style was also added that restores Matplotlib defaults (which is now the Sciris default as well; use ``'sciris'`` or ``'simple'`` for the Sciris style).
#. Fixed a bug with ``packaging.version`` not being found on some systems.
#. Fixed an issue with colormaps attempting to be re-registered, which caused warnings.


Version 2.0.0 (2022-08-18)
--------------------------

This version contains a number of major improvements, including:

#. **New functions**: new functions for downloading (``sc.download()``), paths (``sc.rmpath()``), and data handling (``sc.loadyaml()``) have been added.
#. **Better parallelization**: ``sc.parallel()`` now allows more flexibility in choosing the pool, including ``concurrent.futures``. There's a new ``sc.resourcemonitor()`` for monitoring or limiting resources during big runs.
#. **Improved dataframe**: ``sc.dataframe()`` is now implemented as an extension of a pandas DataFrame.

New features
~~~~~~~~~~~~
#. ``sc.resourcemonitor()`` provides memory or CPU limits, as well as monitors running processes.
#. ``sc.download()`` downloads multiple files in parallel.
#. ``sc.rmpath()`` removes both files and folders, with an optional interactive mode.
#. ``sc.ispath()`` is an alias for ``isinstance(obj, pathlib.Path)``.
#. ``sc.loadyaml()`` and ``sc.saveyaml()`` load and save YAML files, respectively.
#. ``sc.loadzip()`` extracts (or reads data from) zip files.
#. ``sc.count()`` counts the number of matching elements in an array (similar to ``np.count_nonzero()``, but more flexible with e.g. float vs. int mismatches).
#. ``sc.rmnans()`` and ``sc.fillnans()`` have been added as aliases of ``sc.sanitize()`` with default options.
#. ``sc.strsplit()`` will automatically split common types of delimited strings (e.g. ``sc.strsplit('a b c')``).
#. ``sc.parse_env()`` parses environment variables into common types (e.g., will interpret ``'False'`` as ``False``).
#. ``sc.LazyModule()`` handles lazily loaded modules (see ``sc.importbyname()`` for usage).
#. ``sc.randsleep()`` sleeps for a nondeterministic period of time.

Bugfixes
~~~~~~~~
#. ``sc.mergedicts()`` now handles keyword arguments (previously they were silently ignored). Non-dict inputs also now raise an error by default rather than being silently ignored (except for ``None``).
#. ``sc.savespreadsheet()`` now allows NaNs to be saved.
#. ``sc.loadspreadsheet()`` has been updated to match current ``pd.read_excel()`` syntax.
#. ``Spreadsheet`` objects no longer pickle the binary spreadsheet (in some cases reducing size by 50%).
#. File-saving functions now have a ``sanitizepath`` argument (previously, some used file path sanitization and others didn't). They also now return the full path of the saved file.

Improvements
~~~~~~~~~~~~

Major
^^^^^
#. If a copy/deepcopy is not possible, ``sc.cp()``/``sc.dcp()`` now raise an exception by default (previously, they silenced it).
#. ``sc.dataframe()`` has been completely revamped, and is now a backwards-compatible extension of ``pd.DataFrame()``.
#. ``sc.parallelize()`` now supports additional parallelization options, e.g. ``concurrent.futures``, and new ``maxcpu``/``maxmem`` arguments.

Time/date
^^^^^^^^^
#. ``sc.timer()`` now has ``plot()`` and ``total()`` methods, as well as ``indivtimings`` and ``cumtimings`` properties. It also has new methods ``tocout()`` and ``ttout()``, which return output by default (rather than print a string).
#. ``sc.daterange()`` now accepts ``datedelta`` arguments, e.g. ``sc.daterange('2022-02-22', weeks=2)``.
#. ``sc.date()`` can now read ``np.datetime64`` objects.

Plotting
^^^^^^^^
#. ``sc.animation()`` now defaults to ``ffmpeg`` for saving.
#. ``sc.commaticks()`` can now set both ``x`` and ``y`` axes in a single call.
#. ``sc.savefig()`` by default now creates folders if they don't exist.
#. ``sc.loadmetadata()`` can now read metadata from JPG files.

Math
^^^^
#. ``sc.findinds()`` can now handle multiple inputs, e.g. ``sc.findinds(data>0.1, data<0.5)``.
#. ``sc.checktype()`` now includes boolean arrays as being ``arraylike``, and has a new ``'bool'`` option.
#. ``sc.sanitize()`` can now handle multidimensional arrays.

Files
^^^^^
#. ``sc.urlopen()`` can now save to files.
#. ``sc.savezip()`` can now save data to zip files (instead of just compressing files).
#. ``sc.path()`` is more flexible, including handling ``None`` inputs.
#. ``sc.Spreadsheet()`` now has a ``new()`` method that creates a blank workbook.

Other
^^^^^
#. Added ``dict_keys()``, ``dict_values()``, and ``dict_items()`` methods for ``sc.odict()``.
#. ``sc.checkmem()`` now returns a dictionary of sizes rather than prints to screen.
#. ``sc.importbyname()`` can now load multiple modules, and load them lazily.
#. ``sc.prettyobj()`` and ``sc.dictobj()`` now both take either positional or keyword arguments, e.g. ``sc.prettyobj(a=3)`` or ``sc.dictobj({'a':3})``.

Housekeeping
~~~~~~~~~~~~
#. ``pyyaml`` has been added as a dependency.
#. Profiling and load balancing functions have beem moved from ``sc.sc_utils`` and ``sc.sc_parallel`` to a new submodule, ``sc.sc_profiling``.
#. Most instances of ``DeprecationWarning`` have been changed to ``FutureWarning``.
#. Python 2 compatibility functions (e.g. ``sc.loadobj2or3()``) have been moved to a separate module, ``sc.sc_legacy``, which is no longer imported by default.
#. Added style and contributing guides.
#. Added official support for Python 3.7-3.10.
#. ``sc.wget()`` was renamed ``sc.urlopen()``.
#. Sciris now has a "lazy loading" option, which does not import submodules, meaning loading is effectively instant. To use, set the environment variable ``SCIRIS_LAZY=1``, then load submodules via e.g. ``from sciris import sc_odict as sco``.

Regression information
~~~~~~~~~~~~~~~~~~~~~~
#. The default for ``sc.cp()`` and ``sc.dcp()`` changed from ``die=False`` to ``die=True``, which may cause previously caught exceptions to be uncaught. For previous behavior, use ``sc.dcp(..., die=False)``.
#. The argument ``maxload`` (in ``sc.loadbalancer()``, ``sc.parallelize()``, etc.) has been renamed ``maxcpu`` (for consistency with the new ``maxmem`` argument).
#. Previously ``sc.loadbalancer(maxload=None)`` was interpreted as a default load limit (0.8); ``None`` is now interpreted as no limit.
#. Legacy load functions have been moved to a separate module and must be used from there, e.g. ``sc.sc_legacy.loadobj2or3()``.


Version 1.3.3 (2022-01-16)
--------------------------

Plotting
~~~~~~~~
#. Added ``sc.savefig()``, which is like ``pl.savefig()`` but stores additional metadata in the figure -- the file that created the figure, git hash, even the entire contents of ``pip freeze`` if desired. Useful for making figures more reproducible.
#. Likewise, ``sc.loadmetadata()`` will load the metadata from a PNG/SVG file saved with ``sc.savefig()``.
#. Added ``sc.animation()`` as a more flexible alternative to ``sc.savemovie()``. While ``sc.savemovie()`` works directly with Matplotlib artists, ``sc.animation()`` works with entire figure objects so if you can plot it, you can animate it.
#. Split ``sc.dateformatter()`` into two: ``sc.dateformatter()`` reformats axes that already use dates (e.g. ``pl.plot(sc.daterange('2022-01-01', '2022-01-31'), pl.rand(31))``), while ``sc.datenumformatter()`` reformats axes that use numbers (e.g. ``pl.plot(np.arange(31), pl.rand(31))``).
#. Added flexibility for ``sc.boxoff()`` to turn off any sides of the box.

Other changes
~~~~~~~~~~~~~
#. Added ``sc.capture()``, which will redirect ``stdout`` to a string, e.g. ``with sc.capture() as txt: print('This will be stored in "txt"')``. This is very useful for writing tests against text that is supposed to be printed out.
#. Added quick aliases for ``sc.colorize()``, e.g. ``sc.printgreen('This is like print(), but green')``. Colors available are red, green, blue, cyan, yellow, magenta.
#. Keyword arguments are now allowed for ``sc.mergedicts()``, e.g. ``sc.mergedicts({'a':1}, b=2)``. Existing keywords have been renamed to start with an underscore, e.g. ``_strict``.
#. Added an ``every`` argument to ``sc.progressbar()``, to not update on every step.
#. Fixed labeling bugs in several corner cases for ``sc.timer()``.
#. Added an explicit ``start`` argument to ``sc.timedsleep()``.
#. Added additional flexibility to ``sc.getcaller()``, including storing the code of the calling line.


Version 1.3.2 (2022-01-13)
--------------------------
#. Additional flexibility in ``sc.timer()``: it now stores a list of times (``timer.timings``), allows auto-generated labels (``sc.timer(auto=True)``, and has a new method ``timer.tt()`` (short for ``toctic``) that will restart the timer (i.e. time diff rather than cumulative time).
#. Fixed a bug preventing the label from being passed in ``timer.toc()``.
#. Fixed a bug blocking ``style=None`` in ``sc.dateformatter()``, and added an argument to allow using the ``y`` axis.


Version 1.3.1 (2022-01-11)
--------------------------

Changes to odict and objdict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#. Major improvements to ``sc.odict()`` performance: key lookup (e.g. ``my_odict['key']``) is ~30% faster, nearly identical to native ``dict()``; integer lookup (``my_odict[3]``) is now 10-100x faster. This was achieved by caching the keys rather than looking them up each time.
#. Allow dicts with integer keys to be converted to odicts via the ``makefrom()`` method, e.g. ``sc.odict.makefrom({0:'foo', 1:'bar'})``. If an odict has integer keys, then these take precedence.
#. Added ``force`` option to ``objdict.setattribute()`` to allow attributes to be set even if they already exist. Added ``objdict.delattribute()`` to delete attributes.
#. Removed the ``to_OD()`` method (since dicts preserve order, ``dict(my_odict)`` is now much more common).
#. Made ``sc.dictobj()`` a subclass of ``dict``, so ``isinstance(my_dictobj, dict)`` is now ``True``.
#. Added ``sc.ddict()`` as an alias to ``collections.defaultdict()``.

Plotting
~~~~~~~~
#. Updated ``sc.commaticks()`` to use a more thoughtful number of significant figures.

Printing
~~~~~~~~
#. Fixed a bug in ``sc.heading()`` that printed an extraneous ``None``. Also allows more flexibility in spaces before/after the heading.
#. Fixed a bug in ``sc.fonts()`` that prevented using a ``Path`` object. Also added a ``rebuild`` argument that rebuilds the Matplotlib font cache (useful when added fonts don't show up).
#. Updated ``sc.colorize()`` to wrap the ``ansicolors`` module, allowing more flexible inputs such as ``sc.colorize('cat', fg='orange')``.
#. Added ``output`` argument to ``sc.pp()`` which acts as an alias to ``pprint.pformat()``.

Other changes
~~~~~~~~~~~~~
#. Removed the ``pkg_resources`` import, which roughly halves Sciris import time (from 0.3 s to 0.15 s, assuming ``matplotlib.pyplot`` is already imported).
#. Added option to search the source code in ``sc.help()``.
#. Improved the implementations of ``sc.smooth()``, ``sc.gauss1d()``, and ``sc.gauss2d()`` to handle different object types and edge cases.
#. Fixed requirements for ``minimal`` install option.
#. Removed the ``openpyexcel`` dependency (falling back to the nearly identical ``openpyxl``).


Version 1.3.0 (2021-12-30)
--------------------------

This version contains a number of major improvements, including:

#. **Better date plotting**: ``sc.dateformatter()`` has been revamped to provide compact and intuitive date plotting.
#. **Better smoothing**: The new functions ``sc.convolve()``/``sc.gauss1d()``/``sc.gauss2d()``, and the updated ``sc.smooth()``, provide new options for smoothing data.
#. **Simpler fonts**: ``sc.fonts()`` can both list fonts and add new ones.
#. **Simpler options**: Need a bigger font? Just do ``sc.options(fontsize=18)``.

New functions and methods
~~~~~~~~~~~~~~~~~~~~~~~~~
#. Added a settings module to quickly set both Sciris and Matplotlib options; e.g. ``sc.options(dpi=150)`` is a shortcut for ``pl.rc('figure', dpi=150)``, while e.g. ``sc.options(aspath=True)`` will globally set Sciris functions to return ``Path`` objects instead of strings.
#. Added ``sc.timer()`` as a simpler and more flexible way of accessing ``sc.tic()``/``sc.toc()`` and ``sc.Timer()``.
#. Added ``sc.convolve()``, a simple fix to ``np.convolve()`` that avoids edge effects (see update to ``sc.smooth()`` below).
#. Added ``sc.gauss1d()`` and ``sc.gauss2d()`` as additional (high-performance) smoothing functions.
#. Added ``sc.fonts()``, to easily list or add fonts for use in plotting.
#. Added ``sc.dictobj()``, the inverse of ``sc.objdict()`` -- an object that acts like a dictionary (instead of a dictionary that acts like an object). Compared to ``sc.objdict()``, ``sc.dictobj()`` is lighter-weight and slightly faster but less powerful.
#. Added ``sc.swapdict()``, a shortcut for swapping the keys and values of a dictionary.
#. Added ``sc.loadobj2or3()``, for legacy support for loading Python 2 pickles. (Support had been removed in version 1.1.1.)
#. Added ``sc.help()``, to quickly allow searching of Sciris' docstrings.

Bugfixes
~~~~~~~~
#. Fixed edge effects when using ``sc.smooth()`` by using ``sc.convolve()`` instead of ``np.convolve()``.
#. Fixed a bug with checking types when saving files via ``sc.save()``. (Thanks to Rowan Martin-Hughes.)
#. Fixed a bug with ``output=True`` not being passed correctly for ``sc.heading()``.

Improvements
~~~~~~~~~~~~
#. ``sc.dateformatter()`` is now an interface to a new formatter for plotting dates (``ScirisDateFormatter``). This formatter is optimized for aesthetics, combining the best aspects of Matplotlib's and Plotly's date formatters. (Thanks to Daniel Klein.)
#. ``sc.daterange()`` now accepts an ``interval`` argument.
#. ``sc.datedelta()`` can now return the actual delta rather than just the date.
#. ``sc.toc()`` has more flexible printing options.
#. ``sc.Spreadsheet()`` now keeps a copy of the opened workbook, so there is no need to reopen it for every operation.
#. ``sc.commaticks()`` can now use non-comma separators. 
#. Many other functions had small usability improvements, e.g. input arguments are more consistent and more flexible.

Housekeeping
~~~~~~~~~~~~
#. ``xlrd`` has been removed as a dependency; ``openpyexcel`` is used instead, with simple spreadsheet loading now done by ``pandas``.
#. Source files were refactored and split into smaller pieces (e.g. ``sc_utils.py`` was split into ``sc_utils.py``, ``sc_printing.py``, ``sc_datetime.py``, ``sc_nested.py``).

Regression information
~~~~~~~~~~~~~~~~~~~~~~
#. To restore previous spreadsheet loading behavior, use ``sc.loadspreadsheet(..., method='xlrd')``.
#. To use previous smoothing (with edge effects), use ``sc.smooth(..., legacy=True)``


Version 1.2.3 (2021-08-27)
--------------------------
#. Fixed a bug with ``sc.asd()`` failing for ``verbose > 1``. (Thanks to Nick Scott and Romesh Abeysuriya.)
#. Added ``sc.rolling()`` as a shortcut to pandas' rolling average function.
#. Added a ``die`` argument to ``sc.findfirst()`` and ``sc.findlast()``, to allow returning no indices without error.


Version 1.2.2 (2021-08-21)
--------------------------

New functions and methods
~~~~~~~~~~~~~~~~~~~~~~~~~
#. A new class, ``sc.autolist()``, is available to simplify appending to lists, e.g. ``ls = sc.autolist(); ls += 'not a list'``.
#. Added ``sc.freeze()`` as a programmatic equivalent of ``pip freeze``.
#. Added ``sc.require()`` as a flexible way of checking (or asserting) environment requirements, e.g. ``sc.require('numpy')``.
#. Added ``sc.path()`` as an alias to ``pathlib.Path()``.

Improvements
~~~~~~~~~~~~
#. Added an even more robust unpickler, that should be able to recover data even if exceptions are raised when unpickling.
#. Updated ``sc.loadobj()`` to allow loading standard (not gzipped) pickles and from ``dill``.
#. Updated ``sc.saveobj()`` to automatically swap arguments if the object is supplied first, then the filename.
#. Updated ``sc.asd()`` to allow more flexible argument passing to the optimized function; also updated ``verbose`` to allow skipping iterations.
#. Added a ``path`` argument to ``sc.thisdir()`` to more easily allow subfolders/files.
#. Instead of being separate function definitions, ``sc.load()``, ``sc.save()``, and ``sc.jsonify()`` are now identical to their aliases (e.g. ``sc.loadobj()``).
#. ``sc.dateformatter()`` now allows a ``rotation`` argument, since date labels often collide.
#. ``sc.readdate()`` and ``sc.date()`` can now read additional numeric dates, e.g. ``sc.readdate(16166, dateformat='ordinal')``.

Backwards-incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#. ``sc.promotetolist()`` now converts (rather than wraps) ranges and dict_keys objects to lists. To restore the previous behavior, use the argument ``coerce='none'``.
#. The ``start_day`` argument has been renamed ``start_date`` for ``sc.day()`` and ``sc.dateformatter()``.
#. The ``dateformat`` argument for ``sc.date()`` has been renamed ``outformat``, to differentiate from ``readformat``.


Version 1.2.1 (2021-07-07)
--------------------------
#. Added ``openpyxl`` as a Sciris dependency, since it was `removed from pandas <https://pandas.pydata.org/pandas-docs/stable/whatsnew/v1.3.0.html>`__.
#. Added ``sc.datedelta()``, a function that wraps ``datetime.timedelta`` to easily do date operations on strings, e.g. ``sc.datedelta('2021-07-07', days=-3)`` returns ``'2021-07-04'``.
#. Added additional supported date formats to ``sc.readdate()``, along with new ``'dmy'`` and ``'mdy'`` options to ``dateformat``, to read common day-month-year and month-day-year formats.
#. Added the ability for ``sc.compareversions()`` to handle ``'<'``, ``'>='``, etc.
#. Errors loading pickles from ``sc.load()`` are now more informative.


Version 1.2.0 (2021-07-05)
--------------------------

New functions and methods
~~~~~~~~~~~~~~~~~~~~~~~~~
#. Added ``sc.figlayout()`` as an alias to both ``fig.set_tight_layout(True)`` and ``fig.subplots_adjust()``.
#. Added ``sc.midpointnorm()`` as an alias to Matplotlib's ``TwoSlopeNorm``; it can also be used in e.g. ``sc.vectocolor()``.
#. Added ``sc.dateformatter()``, which will (semi-)automatically format the x-axis using dates.
#. Added ``sc.getplatform()``, ``sc.iswindows()``, ``sc.islinux()``, and ``sc.ismac()``. These are all shortcuts for checking ``sys.platform`` output directly.
#. Added ``sc.cpu_count()`` as a simple alias for ``multiprocessing.cpu_count()``.

Bugfixes
~~~~~~~~
#. Fixed ``sc.checkmem()`` from failing when an attribute was ``None``.
#. Fixed a file handle that was being left open by ``sc.gitinfo()``.

``odict`` updates
~~~~~~~~~~~~~~~~~
#. Defined ``+`` for ``sc.odict`` and derived classes; adding two dictionaries is the same as calling ``sc.mergedicts()`` on them. 
#. Updated nested dictionary functions, and added them as methods to ``sc.odict()`` and derived classes (like ``sc.objdict()``); for example, you can now do ``nestedobj = sc.objdict(); nestedobj.setnested(['a','b','c'], 4)``.
#. Added ``sc.odict.enumvalues()`` as an alias to ``sc.odict.enumvals()``.

Plotting updates
~~~~~~~~~~~~~~~~
#. Updated ``sc.commaticks()`` to use better formatting.
#. Removed the ``fig`` argument from ``sc.commaticks()`` and ``sc.SIticks()``; now, the first argument can be an ``Axes`` object, a ``Figure`` object, or a list of axes.
#. Updated ``sc.get_rows_cols()`` to optionally create subplots, rather than just return the number of rows/columns.
#. Removed ``sc.SItickformatter``; use ``sc.SIticks()`` instead.

Other updates
~~~~~~~~~~~~~
#. Updated ``sc.heading()`` to handle arguments the same way as ``print()``, e.g. ``sc.heading([1,2,3], 'is a list')``.
#. Allowed more flexibility with the ``ncpus`` argument of ``sc.parallelize()``: it can now be a fraction, representing a fraction of available CPUs. Also, it will now never exceed the number of tasks to be run.
#. Updated ``sc.suggest()`` to modify the threshold to be based on the length of the input word.



Version 1.1.1 (2021-03-17)
--------------------------
1. The implementations of ``sc.odict()`` and ``sc.objdict()`` have been updated, to allow for more flexible use of the ``defaultdict`` argument, including better nesting and subclassing.
2. A new ``serial`` argument has been added to ``sc.parallelize()`` to allow for quick debugging.
3. Legacy support for Python 2 has been removed from ``sc.loadobj()`` and ``sc.saveobj()``.
4. A fallback method for ``sc.gitinfo()`` (based on ``gitpython``) has been added, in case reading from the filesystem fails.


Version 1.1.0 (2021-03-12)
--------------------------

New functions
~~~~~~~~~~~~~
1. ``sc.mergelists()`` is similar to ``sc.mergedicts()``: it will take a sequence of inputs and attempt to merge them into a list.
2. ``sc.transposelist()`` will perform a transposition on a list of lists: for example, a list of 10 lists (or tuples) each of length 3 will be transformed into a list of 3 lists each of length 10.
3. ``sc.strjoin()`` and ``sc.newlinejoin()`` are shortcuts to ``', '.join(items)`` and ``'\n'.join(items)``, respectively. The latter is especially useful inside f-strings since you cannot use the ``\n`` character.

Bugfixes
~~~~~~~~
1. ``sc.day()`` now returns a numeric array when an array of datetime objects is passed to it; a bug which was introduced in version 1.0.2 which meant it returned an object array instead.
2. Slices with numeric start and stop indices have been fixed for ``sc.odict()``.
3. ``sc.objatt()`` now correctly handles objects with slots instead of a dict.

Improvements
~~~~~~~~~~~~
1. ``sc.loadobj()`` now accepts a ``remapping`` argument, which lets the user load old pickle files even if the modules no longer exist.
2. Most file functions (e.g. ``sc.makefilepath``, ``sc.getfilelist()`` now accept an ``aspath`` argument, which, if ``True``, will return a ``pathlib.Path`` object instead of a string.
3. Most array-returning functions, such as ``sc.promotetoarray()`` and ``sc.cat()``, now accept a ``copy`` argument and other keywords; these keywords are passed to ``np.array()``, allowing e.g. the ``dtype`` to be set.
4. A fallback option for ``sc.findinds()`` has been implemented, allowing it to work even if the input array isn't numeric.
5. ``sc.odict()`` now has a ``defaultdict`` argument, which lets you use it like a defaultdict as well as an ordered dict.
6. ``sc.odict()`` has a ``transpose`` argument for methods like ``items()`` and ``enumvalues()``, which will return a tuple of lists instead of a list of tuples.
7. ``sc.objdict()`` now prints out differently, to distinguish it from an ``sc.odict``.
8. ``sc.promotetolist()`` has a new ``coerce`` argument, which will convert that data type into a list (instead of wrapping it).

Renamed/removed functions
~~~~~~~~~~~~~~~~~~~~~~~~~
1. The functions ``sc.tolist()`` and ``sc.toarray()`` have been added as aliases of ``sc.promotetolist()`` and ``sc.promotetoarray()``, respectively. You may use whichever you prefer.
2. The ``skipnone`` keyword has been removed from ``sc.promotetoarray()`` and replaced with ``keepnone`` (which does something slightly different).

Other updates
~~~~~~~~~~~~~
1. Exceptions have been made more specific (e.g. ``TypeError`` instead of ``Exception``).
2. Test code coverage has been increased significantly (from 63% to 84%).


Version 1.0.2 (2021-03-10)
--------------------------
1. Fixed bug (introduced in version 1.0.1) with ``sc.readdate()`` returning only the first element of a list of a dates.
2. Fixed bug (introduced in version 1.0.1) with ``sc.date()`` treating an integer as a timestamp rather than an integer number of days when a start day is supplied.
3. Updated ``sc.readdate()``, ``sc.date()``, and ``sc.day()`` to always return consistent output types (e.g. if an array is supplied as an input, an array is supplied as an output).


Version 1.0.1 (2021-03-01)
--------------------------
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
