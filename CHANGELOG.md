# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

By import convention, components of the Sciris library are listed beginning with `sc.`, e.g. `sc.odict`.

## Version 0.16.8 (2020-04-11)
- Added a [Code of Conduct](CODE_OF_CONDUCT.md).
- `sc.makefilepath()` now has a `checkexists` flag, which will optionally raise an exception if the file does (or doesn't) exist.
- `sc.sanitizejson()` now handles `datetime.date` and `datetime.time`.
- `sc.uuid()` and `sc.fast_uuid()` now work with non-integer inputs, e.g., `sc.uuid(n=10e3)`.
- `sc.thisdir()` now accepts additional arguments, so can be used to form a full path, e.g. `sc.thisdir(__file__, 'myfile.txt')`.
- `sc.checkmem()` has better parsing of objects.
- `sc.prepr()` now lists properties of objects, and has some aesthetic improvements.
