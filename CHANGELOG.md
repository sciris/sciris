# Changelog

All notable changes to this project will be documented in this file.

By import convention, components of the Sciris library are listed beginning with `sc.`, e.g. `sc.odict()`.


## Version 0.17.4 (2020-08-11)
1. `sc.profile()` and `sc.mprofile()` now return the line profiler instance for later use (e.g., to extract additional statistics).
1. `sc.prepr()` (also used in `sc.prettyobj()`) can now support objects with slots instead of dicts.

## Version 0.17.3 (2020-07-21)
1. `sc.parallelize()` now explicitly deep-copies objects, since on some platforms this copying does not take place as part of the parallelization process.

## Version 0.17.2 (2020-07-13)
1. `sc.search()` is a new function to find nested attributes/keys within objects or dictionaries.

## Version 0.17.1 (2020-07-07)
1. `sc.Blobject` has been modified to allow more flexibility with saving (e.g., `Path` objects).

## Version 0.17.0 (2020-04-27)
1. `sc.mprofile()` has been added, which does memory profiling just like `sc.profile()`.
1. `sc.progressbar()` has been added, which prints a progress bar.
1. `sc.jsonpickle()` and `sc.jsonunpickle()` have been added, wrapping the module of the same name, to convert arbitrary objects to JSON.
1. `sc.jsonify()` checks objects for a `to_json()` method, handling e.g Pandas dataframes, and falls back to `sc.jsonpickle()` instead of raising an exception for unknown object types.
1. `sc.suggest()` now uses `jellyfish` instead of `python-levenshtein` for fuzzy string matching.
1. `sc.saveobj()` now uses protocol 4 instead of the latest by default, to avoid backwards incompatibility issues caused by using protocol 5 (only compatible with Python 3.8).
1.  `sc.odict()` and related classes now raise `sc.KeyNotFoundError` exceptions. These are derived from `KeyError`, but fix a bug in the string representation (https://stackoverflow.com/questions/34051333/strange-error-message-printed-out-for-keyerror) to allow multi-line error messages.
1. Rewrote all tests to be pytest-compatible.

## Version 0.16.8 (2020-04-11)
1. Added a [Code of Conduct](CODE_OF_CONDUCT.md).
1. `sc.makefilepath()` now has a `checkexists` flag, which will optionally raise an exception if the file does (or doesn't) exist.
1. `sc.sanitizejson()` now handles `datetime.date` and `datetime.time`.
1. `sc.uuid()` and `sc.fast_uuid()` now work with non-integer inputs, e.g., `sc.uuid(n=10e3)`.
1. `sc.thisdir()` now accepts additional arguments, so can be used to form a full path, e.g. `sc.thisdir(__file__, 'myfile.txt')`.
1. `sc.checkmem()` has better parsing of objects.
1. `sc.prepr()` now lists properties of objects, and has some aesthetic improvements.
