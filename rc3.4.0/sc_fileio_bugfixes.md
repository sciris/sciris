# `sc_fileio.py` bug audit

Audit of `sciris/sc_fileio.py` for genuine defects: wrong results, documented arguments that don't work, silent data corruption, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, and performance. This is the largest module in Sciris (2711 lines), so it was covered by two parallel auditors working in halves: one over `load`/`save`/`zsave`/`loadstr`/`dumpstr`, text and zip functions, path functions, `makefilepath`, `sanitizefilename`, `rmpath`, and `loadany` (lines 53-1219), and the other over the JSON/YAML functions, `Blobject`/`Spreadsheet`, spreadsheet load/save, and the robust-unpickler machinery (lines 1223-2711).

**Method**: line-by-line reading of each function, followed by executed hypothesis tests against the editable install (Sciris 3.3.0, commit `2d69aad`, pandas/numpy/openpyxl/xlsxwriter as installed in the dev environment), with each finding reproduced a second time independently before being recorded here.

**Re-verification**: this document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`; pandas 3.0.5, numpy 2.4.6, openpyxl 3.1.5, Python 3.13), re-running every finding. Of the original 35 findings, 25 were confirmed (several with corrected severity or a caveat on the fix), 2 were rewritten because their proposed fix was wrong or incomplete (4, 7), and 8 were rejected as not a bug or not worth fixing (listed under "Rejected on review"); 8 new bugs found during re-verification were added as findings 36-43. The document now lists 35 findings: 4 High, 14 Medium and 17 Low. Findings are listed by severity, so numbers are not contiguous within a section; the original numbers are kept stable for cross-referencing.

**Nothing in this document has been applied.** All fixes are described, not made.

This file contains one of the most consequential defect combinations found in the audit: a failed save can destroy the file it was overwriting (finding 1), and the corresponding load can then return an empty value instead of raising (finding 2), so the loss is silent.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 2 | High | `sc.load()` | Returns the file's raw text (or `''`) instead of raising, even with `die=True` | 108 |
| 3 | High | `sc.unzip()` | Returns paths that do not exist whenever the zip contains subfolders | 565 |
| 4 | High | `sc.Spreadsheet.readcells()` | Passes `sheetname` where `sheetnum` belongs, so `sheetnum=` silently reads the wrong sheet | 1993 |
| 5 | High | `sc.loadspreadsheet()` | Reads the second row as the header by default, mislabelling every column and dropping the first data row | 2095 |
| 1 | Medium | `sc.save()` | A failed save truncates and destroys the pre-existing file at that path, for every compression mode | 329 |
| 6 | Medium | `sc.loadstr()`, `sc.load()` | Cannot read a non-gzip file object, although the same bytes load fine from disk | 110 |
| 7 | Medium | `sc.save()`, `sc.dumpstr()` | `filename=None` raises `ValueError: I/O operation on closed file` for every compression except gzip | 334 |
| 9 | Medium | `sc.unzip(members=...)` | Returns the full member list, not the extracted subset | 562 |
| 12 | Medium | `sc.makefilepath()` | Silently discards the directory component of `filename` when `folder` is supplied | 983 |
| 15 | Medium | `sc.jsonify()` | `custom=...` raises `KeyError` for any subclass of a registered class, including `sc.odict` | 1287 |
| 16 | Medium | `sc.jsonify()` | Raises `TypeError` on `Decimal` and `complex` instead of honouring `die=False` | 1296 |
| 18 | Medium | `sc.jsonify()`, `sc.sanitizejson()`, `sc.savejson()`, `sc.saveyaml()`, `sc.printjson()` | Raises `AttributeError` in any thread other than the one that imported Sciris | 1327 |
| 19 | Medium | `sc.savejson()` | Truncates the output file before it knows the object can be serialised, destroying the previous contents | 1488 |
| 22 | Medium | `sc.loadspreadsheet()` | Argument-alias loop ignores its loop variable, so `sheetnum=` and `sheet_name=` raise `TypeError` | 2124 |
| 36 | Medium | `sc.load()` | Rejects ordinary file objects (e.g. `open(f, 'rb')`), although documented to accept them | 87 |
| 37 | Medium | `sc.load(remapping=..., die=True)` | `die=True` is ignored: the whole object silently becomes a `NamedFailed` instead of raising | 2618 |
| 38 | Medium | `sc.jsonify()`, `sc.savejson()`, `sc.saveyaml()` | Output of `to_dict()`/`to_json()` is returned unsanitized, so `savejson()` crashes and truncates the file | 1333 |
| 39 | Medium | `sc.jsonify()` | Silently discards the imaginary part of numpy complex values | 1303 |
| 13 | Low | `sc.rmpath()` | Deletes a path it has just reported as unremovable, using the previous iteration's removal function | 1088 |
| 21 | Low | `sc.loadspreadsheet()` | `asdataframe` is documented but has no effect for the default `method='pandas'` | 2116 |
| 23 | Low | `sc.savespreadsheet()` | `formats=...` without `formatdata` dies with `UnboundLocalError` | 2327 |
| 24 | Low | `sc.load(remapping=...)` | A remapping value naming a dotted module (no class) is mis-split into `(module, class)`, and can collapse the whole load | 2556 |
| 25 | Low | `sc.load()` | Cannot load a pickle whose top-level payload is `None` | 2682 |
| 26 | Low | `sc.load()` | A `# pragma: no cover` comment was pasted inside an error-message string literal | 130 |
| 27 | Low | `sc.thisdir()` | Silently drops a `..` passed as the first argument, and anchors the result to the cwd | 748 |
| 28 | Low | `sc.getfilepaths()`, `sc.sanitizepath()`, `sc.makepath()` | `aspath` is silently ignored | 845 |
| 30 | Low | `sc.makefilepath()` | `ext=...` produces `f..obj` for a dotted extension and skips the extension for a suffix-less match | 989 |
| 31 | Low | `sc.rmpath()` | Cannot remove a symlink to a directory and raises by default | 1086 |
| 32 | Low | `sc.Blobject()`, `sc.Spreadsheet()` | A `Path` source loses the `name` attribute, and `blob=` plus `filename=` is rejected with a misleading message | 1752 |
| 33 | Low | `sc.savespreadsheet()` | The formatting docstring example raises `TypeError` as written | 2269 |
| 34 | Low | module import / `sc.load(auto_remap=True)` | `except (NameError or AttributeError)` catches only `NameError`, leaving the compatibility-map fallback dead (and broken) | 2371 |
| 40 | Low-Medium | `sc.Blobject.save()` | With no filename, ignores `self.filename`, writes to `./default`, and sets `self.filename=None` | 1819 |
| 41 | Low | `sc.loadspreadsheet()` | `fileobj=` is ignored for `method='openpyxl'` | 2135 |
| 42 | Low | `sc.save()` | `verbose=None` raises `TypeError` | 276 |
| 43 | Low | `sc.rmpath()` | `sc.rmpath(sc.getfilelist(folder))` deletes everything and then raises `FileNotFoundError` | 1076 |

## Recurring patterns

**Truncate-then-serialise: the destination file is opened before the data is serialised.** This is the dominant pattern and spans both halves of the audit. `sc.save()` opens the destination for writing (`gz.GzipFile(filename=filename, mode='wb')` at line 329, or `open(filename, 'wb')` at line 336) before it attempts to serialize the object, for every compression mode and both `die` settings, so a serialization failure destroys whatever was previously at that path; the invalid-`compression` path (rejected finding 8) truncates before even validating its own arguments. `sc.savejson()` has the identical shape at line 1488: it opens (and truncates) the target, and only then runs `jsonify()` and the JSON encode, so a `RecursionError` or `UnicodeEncodeError` mid-encode leaves a zero-length or half-written file where a valid one used to be. `sc.saveyaml()`, by contrast, jsonifies first (line 1616) and only opens the file afterward (line 1621), and survives the identical test — confirming this is an ordering slip in `save()`/`savejson()` rather than a deliberate choice. `Spreadsheet`/`savespreadsheet` are also safe, because xlsxwriter and openpyxl both build in memory and only touch the target file at `close()`/`save()`. Write-to-temp-then-rename appears nowhere in the module. Fixing this pattern means serializing fully in memory first and only then opening the target, at every site above (preferred over temp-file + `os.replace()`, which would replace symlinks and reset permissions; see finding 1).

**A forgiving fallback chain converts a hard error into a plausible-looking value.** `_load_filestr()`'s four-deep `try` cascade and `_unpickler()`'s method list both end in a step that cannot fail (`open(..., 'r').read()`, `bytes.decode()`), so `sc.load()` has no way to report "this is not a Sciris object" and instead returns a plausible-looking `str` — including `''` for a file that a failed save just destroyed, with `die=True` making no difference. `loadzip()` has the same shape with a bare `try: ... except: pass`, so zip members holding uncompressed or zstd pickles are silently returned as raw bytes/text instead of objects.

**Copy-paste and aliasing errors in argument handling.** `Spreadsheet.readcells()` has `sheetnum=kwargs.get('sheetname')` (both arguments read the same key), and `loadspreadsheet()`'s alias loop does `sheet = kwargs.pop('sheetname', sheet)` inside `for key in ['sheetname', 'sheetnum', 'sheet_name']` without ever reading `key` — the same copy-paste slip a few hundred lines apart, both degrading silently rather than failing loudly. `makefilepath(folder=...)` takes `os.path.basename(filename)` unconditionally and only restores the directory `if folder is None`, so `folder` *replaces* rather than prepends to any directory component of `filename`; `savezip(basename=True)` reduces arcnames the same way. In both cases distinct inputs are mapped onto one output path, which is how `unzip()` ends up returning paths that do not exist (finding 3) and how `savezip(basename=True)` flattens names (rejected finding 10; `zipfile` does warn about the duplicates). Related: `jsonify(custom=...)` matches with `isinstance` but dispatches with exact `obj.__class__`, so the two must agree or subclasses raise `KeyError`; and `getfilepaths()`/`sanitizepath()`/`makepath()` each hardcode `aspath=True` in their delegated call instead of forwarding the caller's value, while `thispath()` gets it right.

## High severity

### 2. `sc.load()` returns the file's raw text (or `''`) instead of raising, even with `die=True` — `sc_fileio.py:108`, `135`

`_load_filestr()`'s fallback chain ends by reading any non-gzip, non-zstd file as plain bytes/text (lines 118-125), and `_unpickler()`'s default method list ends with `bytestr = lambda s: s.decode()` (`sc_fileio.py:2647`, reached via the `method=None` list at 2656 regardless of `die`). The combination means `sc.load()` on a file that is not a pickle at all quietly returns a `str`, and `die=True` does not change this -- contradicting the documented meaning of `die` ("whether to raise an exception if errors are encountered").

```python
import sciris as sc
open('e.obj','wb').close()          # empty file
open('t.obj','w').write('hello\n')  # plain text
sc.load('e.obj', die=True)          # actual: ''         expected: an exception
sc.load('t.obj', die=True)          # actual: 'hello\n'  expected: an exception
```

Actual output: `empty: '' | text: 'hello\n'`. This is what turns finding 1 from "a crash" into silent data loss: after a failed save the destroyed file loads as `''` with no warning, so a downstream `for k in obj:` or `obj['results']` fails somewhere far away from the real cause, and a naive truthiness check (`if sc.load(fn):`) reports "no data" rather than "corrupted". It also means `sc.load()` cannot be used to detect that a file is not a Sciris object.

By contrast the genuinely-corrupt-pickle path is handled properly: a gzip file with a flipped byte gives `NamedFailed` plus a printed diagnostic under `die=False`, and raises `UnpicklingError` under `die=True`.

It also defeats `sc.loadany()`, whose comment expects `die=True` to "die rather than forging ahead loading junk": a JSON file named `.obj` comes back from `loadany()` as a raw string.

**Fix**: drop `bytestr` (and the plain-text fallback in `_load_filestr`) from the chain when `die=True`; at minimum, warn when the object was produced by `string`/`bytestr` rather than by an actual unpickler, and raise on a zero-length payload.

### 3. `sc.unzip()` returns paths that do not exist whenever the zip contains subfolders — `sc_fileio.py:565`

The documented return value ("list of the names of the unzipped files") is built with `makefilepath(filename=name, folder=outfolder, ...)`, and `makefilepath()` discards the directory part of `filename` when `folder` is given (see finding 12). So every member stored with a folder prefix is reported as `outfolder/<basename>` while the file is actually extracted to `outfolder/<full member path>`. None of the returned paths exist.

```python
import sciris as sc, os
os.makedirs('s/a', exist_ok=True); open('s/a/f.txt','w').write('hi'); open('s/g.txt','w').write('yo')
sc.savezip('z.zip', ['s/a/f.txt','s/g.txt'], verbose=False)
out = sc.unzip('z.zip', outfolder='o')
print([os.path.relpath(p) for p in out], [os.path.exists(p) for p in out])
print(sorted(os.path.relpath(p) for p in sc.getfilelist('o','**',filesonly=True)))
```

```
actual   returned: ['o/f.txt', 'o/g.txt']   exists: [False, False]
actual   on disk : ['o/s/a/f.txt', 'o/s/g.txt']
expected returned: ['o/s/a/f.txt', 'o/s/g.txt']
```

`sc.savezip()` stores full relative paths by default (`basename=False`), so a `savezip` -> `unzip` round trip of anything but a flat file list produces a wrong list; two members from different folders with the same basename are additionally reported as the *same* path twice. `tests/test_fileio.py:125-127` only round-trips a `basename=True` (flat) zip, so this is untested. A caller doing the natural `for f in sc.unzip(...): process(f)` gets `FileNotFoundError`, or worse silently skips files if it guards with `os.path.exists`.

As a side effect the comprehension passes `makedirs=True`, so `unzip()` also creates directories based on those flattened names.

**Fix**: build the list as `[os.path.join(outfolder, name) for name in names]` (normalized), or use the names `zf.extractall` actually wrote; do not route member names through `makefilepath(folder=...)`, and drop `makedirs=True`.

### 4. `Spreadsheet.readcells()` passes `sheetname` where `sheetnum` belongs, so `sheetnum=` silently reads the wrong sheet — `sc_fileio.py:1993`

Copy-paste error: `ws = self._getsheet(sheetname=kwargs.get('sheetname'), sheetnum=kwargs.get('sheetname'))`. Both arguments read `'sheetname'`, so `sheetnum` is never picked up out of `kwargs`; `_getsheet` then falls through to `self.wb.active` and returns data from the first/active sheet with no error.

```python
import sciris as sc
S = sc.Spreadsheet(); wb = S.openpyxl()
wb.active['A1'] = 'SHEET0'
wb.create_sheet('two')['A1'] = 'SHEET1'
wb.save(S.freshbytes()); S.load()
print(S.readcells(sheetnum=1, cells=[(0,0)]))
print(S.readcells(sheetname='two', cells=[(0,0)]))
```

Actual: `['SHEET0']` then `['SHEET1']`. Expected: `['SHEET1']` in both cases. The wrong-sheet result is returned silently, so a caller that indexes sheets by number gets plausible-looking values from the wrong sheet.

Blast radius: `_getsheet()` (line 1968) and `writecells()` (line 2016) both support `sheetnum` properly, so only `readcells()` is broken; `sheetname=` works, which is why `tests/test_fileio.py:56` does not catch this. Note also that `sc.loadspreadsheet(..., sheetnum=1)` fails outright (finding 22), so there is no working way to select a sheet by number through either reader.

**Fix**: `sheetnum=kwargs.get('sheetnum')` in the openpyxl branch (keep using `.get()`, not `.pop()`), and fix finding 22 so that the forwarded `sheetnum` is also honoured by the pandas branch. (Corrected on review: the original fix also said to pop both keys out of `kwargs` so they are not forwarded to `loadspreadsheet` in the pandas branch. That would break the pandas branch: `readcells(method='pandas', sheetname='two')` currently works *because* `sheetname` is forwarded and `loadspreadsheet()` pops it into `sheet`, returning `['SHEET1']`. Popping the keys would require passing `sheet=` explicitly instead.)

### 5. `loadspreadsheet()` reads the second row as the header by default, mislabelling every column and dropping the first data row — `sc_fileio.py:2095`

The default is `header=1`, which pandas interprets as "the column names are in row index 1", so an ordinary spreadsheet whose first row is the header loads with the first data row promoted to the header and that row lost. The docstring says the opposite: `header (bool): whether the 0-th row is to be read as the header` (line 2108) -- reading row 0 as the header is pandas' `header=0`. The `header` argument is also documented as a bool, but `header=True` (the literal reading of the docstring) is rejected by modern pandas.

```python
import pandas as pd, sciris as sc
pd.DataFrame(dict(name=['a','b','c'], value=[1,2,3])).to_excel('v.xlsx', index=False)
print(sc.loadspreadsheet('v.xlsx'))          # default
print(sc.loadspreadsheet('v.xlsx', header=0))
```

Actual:

```
   a  1
0  b  2
1  c  3
   name  value
0     a      1
1     b      2
2     c      3
```

Expected: the default should give the second block (columns `name`, `value`, three rows). `sc.loadspreadsheet('v.xlsx', header=True)` raises `TypeError: Passing a bool to header is invalid. Use header=None for no header or header=int or list-like of ints...`, so the documented bool spelling does not work at all.

Corroboration from code reading (xlrd is not installed here, so this half is not executed): the `method='xlrd'` branch treats the same argument as a genuine bool -- `for rownum in range(ws.nrows-header)`, `if header: attr = ws.cell_value(0,colnum)`, `val = ws.cell_value(rownum+header,colnum)` (lines 2159-2165) -- i.e. with the default `header=1` it takes row 0 as the header and rows 1..n as data. The two methods therefore disagree by one row for identical arguments, which shows `header=1` was meant as the boolean `True`, not as pandas' row index.

Git history confirms this was a regression: commit 316fc6a (2022) changed `header=True` + `np.arange(header)` (i.e. row 0 as the header) to `header=1` passed straight to pandas, which moved the header to row 1.

Blast radius: `Spreadsheet.readcells()` calls `loadspreadsheet` with an explicit `header=None` (line 1984), so it is unaffected; every direct `sc.loadspreadsheet()` call is affected. Most sheets in the repository's own fixture `tests/files/exampledata.xlsx` have their header in row 0, so they load mislabelled.

**Fix**: make the pandas branch translate the flag (`header = 0 if header else None`, accepting `True`/`1`/`0`/`False`/`None`), or change the default to `header=0` and document it as a row index rather than a bool. Either way the pandas and xlrd branches should agree, and the `header (bool)` doc line needs to match.

## Medium severity

### 1. A failed `sc.save()` destroys the pre-existing file at that path — `sc_fileio.py:329`, `336`

`save()` opens the destination for writing (`gz.GzipFile(filename=filename, mode='wb')` at line 329, or `open(filename, 'wb')` at line 336) *before* it attempts to serialize the object, so the file is truncated to zero the instant the call starts. If serialization then fails, the user's previous save is already gone and is replaced by a 0-29 byte stub. There is no write-to-temp-then-rename and no attempt to restore the old bytes.

```python
import os, sciris as sc
class Bad:
    def __reduce__(self): raise RuntimeError('nope')

sc.save('data.obj', dict(results='10 years of output'))   # the good save
print(os.path.getsize('data.obj'))                        # 71
try: sc.save('data.obj', Bad(), die=True)                 # the failed save
except RuntimeError: pass
print(os.path.getsize('data.obj'))                        # 29  <- previous save destroyed
sc.load('data.obj')                                       # ''
```

Actual: `71 -> 29`, and the reload silently yields `''` (see finding 2). Expected: the exception propagates and `data.obj` still contains the earlier object.

This is not limited to `die=True` or to gzip; every compression mode behaves the same, measured on the same object:

```
gzip die=True    : size 15389 -> 26,    reload -> ''
gzip die=False   : size 15389 -> 26,    reload -> ''
zstd die=False   : size 15446 -> 9,     reload -> ''
none die=False   : size 80201 -> 0,     reload -> ''
```

(`die=False` does not help: pickle failure falls through to dill at line 286, and if dill also fails the exception escapes anyway, by which point the file is already truncated. `die=False` *does* rescue objects that only pickle rejects, e.g. a `threading.Lock`, which dill can serialize.)

The same truncate-first pattern makes a *partial* write possible: saving a dict whose late values fail leaves whatever the compressor had already flushed. Blast radius: everything that writes through `save()` -- `sc.saveobj()`, `sc.zsave()`, `sc.savearchive()` (`sc_versioning.py:670`), `sc.savezip(data=...)` via `dumpstr()` (`sc_fileio.py:633`), `sc.Blobject`. The canonical use is overwriting the results file of a long run, which is exactly when losing the previous version hurts most.

**Severity (re-verified)**: downgraded from High to Medium. The truncation reproduces for gzip/zstd/none (61->29, 45->9 and 36->0 bytes after a failed save), but it is the same behaviour as a plain `open(f, 'wb')` + `pickle.dump`, so it is a missed safeguard rather than a Sciris-specific corruption. It becomes silent only in combination with finding 2.

**Fix**: serialize fully in memory first, and only then open the target and write. `_savepickle`/`_savedill` already call `pickle.dumps`/`dill.dumps` in memory, so produce the serialized bytes (and compress them in memory) before opening the destination; a serialization failure then never touches the existing file. This also removes the invalid-`compression` truncation (rejected finding 8). The originally proposed temp-file + `os.replace()` approach also works, but it silently changes semantics: it replaces a symlink instead of writing through it, resets permissions/ownership, and breaks hardlinks, so the in-memory ordering fix is the lower-risk option.

### 6. `sc.loadstr()`/`sc.load()` cannot read a non-gzip file object, although the same bytes load fine from disk — `sc_fileio.py:110`, `118`, `123`

When `sc.load()` is given a file object, `_load_filestr()` sets `argtype='fileobj'` and reads it as gzip; but all three fallbacks (zstd at 110, binary at 118, text at 123) are hardcoded to `open(filename, ...)`, and `filename` is the `BytesIO` in that case. So they all fail with `TypeError: expected str...`, and an uncompressed or zstd-compressed bytestring is rejected.

```python
import sciris as sc, pickle
sc.loadstr(pickle.dumps(dict(a=1)))   # UnpicklingError: Unable to load <_io.BytesIO ...>
open('raw.obj','wb').write(pickle.dumps(dict(a=1)))
sc.load('raw.obj')                    # {'a': 1}  -- identical bytes, works from disk
```

Actual: `UnpicklingError ... as either a gzipped, zstandard or regular pickle file. Ensure that it is actually a pickle file.` with "Additional errors encountered: expected str". Expected: `{'a': 1}`, since `sc.load()`'s docstring advertises both file objects and non-gzipped pickles ("*New in version 1.2.2:* ability to load non-gzipped pickles"), and the error message itself claims zstandard and regular pickles were attempted.

Blast radius: `sc.loadzip()` calls `loadstr(val)` inside a bare `try/except: pass` (line 528-529), so zip members holding uncompressed or zstd pickles are silently returned as raw bytes/text instead of objects; `sc.loadarchive()` (`sc_versioning.py:766`) can therefore only read gzip-compressed archives. Note the fallbacks also never `seek(0)` the stream after the failed gzip read, so simply substituting `filename`/`fileobj` is not sufficient.

**Fix**: read the bytes once (`filestr = fileobj.read()` for a file object, `open(...).read()` for a path) and then dispatch on the magic bytes (`\x1f\x8b` gzip, `\x28\xb5\x2f\xfd` zstd), decompressing in memory; for the file-object case `seek(0)` before each retry.

### 7. `sc.save(filename=None)` raises `ValueError: I/O operation on closed file` for every compression except gzip — `sc_fileio.py:334`, `358`

In the non-gzip branch the bytestream is wrapped in `filecontext = closing(bytestream)` (line 334), so leaving the `with` block *closes* the `BytesIO`; line 358 then calls `bytestream.seek(0)` on it. The documented behaviour "if None, return an io.BytesIO filestream instead of saving to disk" therefore only works with the default `compression='gzip'`.

```python
import sciris as sc
sc.save(filename=None, obj=1, compression='gzip')   # OK: BytesIO
sc.save(filename=None, obj=1, compression='zstd')   # ValueError: I/O operation on closed file.
sc.save(filename=None, obj=1, compression='none')   # ValueError: I/O operation on closed file.
sc.dumpstr(sc.objdict(a=1), compression='zstd')     # same, raised from sc_fileio.py:358
```

Traceback tail (actual):

```
File "/home/cliffk/sc/sciris/sciris/sc_fileio.py", line 415, in dumpstr
    bytesobj = save(filename=None, obj=obj, **kwargs)
File "/home/cliffk/sc/sciris/sciris/sc_fileio.py", line 358, in save
    bytestream.seek(0)
ValueError: I/O operation on closed file.
```

Blast radius: `sc.dumpstr()` (which documents `kwargs` as "passed to `sc.save()`"), and hence `sc.savearchive(..., dumpargs=dict(compression=...))` at `sc_versioning.py:670` and `sc.savezip(data=..., compression=...)` at `sc_fileio.py:633`. The branch is marked `# pragma: no cover` at line 333, which is why the breakage is invisible to the test suite.

**Fix**: replace `closing(bytestream)` with `contextlib.nullcontext(bytestream)` in the `tobytes` case **and** pass `closefd=False` to zstd's `stream_writer()`; alternatively, capture `bytestream.getvalue()` before the `with` exits. (Corrected on review: the original fix proposed `nullcontext` alone, which only fixes `compression='none'`. zstd's `stream_writer()` closes its inner stream on exit by default, so the `BytesIO` is still closed:)

```python
b = io.BytesIO()
with contextlib.nullcontext(b) as fh:
    with zstd.ZstdCompressor().stream_writer(fh) as w: w.write(b'abc')
b.closed   # True; with stream_writer(fh, closefd=False) it is False
```

### 9. `sc.unzip(members=...)` returns the full member list, not the extracted subset — `sc_fileio.py:562`

`names = zf.namelist()` is taken before `zf.extractall(outfolder, members=members)` and `members` is never consulted when building the return value, so the documented "list of the names of the unzipped files" lists files that were deliberately not extracted.

```python
import sciris as sc
sc.savezip('z.zip', ['s/a/f.txt','s/g.txt'], verbose=False)
out = sc.unzip('z.zip', outfolder='o2', members=['s/g.txt'])
print(len(out), 'paths returned;', len(sc.getfilelist('o2','**',filesonly=True)), 'file(s) extracted')
```

Actual: `2 paths returned; 1 file(s) extracted`. Expected: one path. Combined with finding 3, none of the returned paths exist either.

**Fix**: `names = members if members is not None else zf.namelist()`.

### 12. `makefilepath()` silently discards the directory component of `filename` when `folder` is supplied — `sc_fileio.py:983`

`basename = os.path.basename(filename)` is taken unconditionally at line 982, and the filename's own directory is only recovered `if folder is None` (line 983). So `folder` does not "prepend" as documented -- it *replaces* any path in `filename`, including an absolute one. The docstring says the opposite for both arguments: `folder` is "the name of the folder to be prepended to the filename", and `filename` may be "the filename, or full file path, to save to -- in which case this utility does nothing".

```python
sc.makefilepath('sub/f.txt',      folder='/tmp/xx')   # actual: /tmp/xx/f.txt   expected: /tmp/xx/sub/f.txt
sc.makefilepath('/abs/dir/f.txt', folder='/tmp/xx')   # actual: /tmp/xx/f.txt   expected: /abs/dir/f.txt
```

Reaching the user-facing level:

```python
sc.save('sub/myfile.obj', dict(a=1), folder='out')
# actual   -> .../out/myfile.obj
# expected -> .../out/sub/myfile.obj
```

Because `sc.load()` applies the same rule, a save/load pair stays self-consistent, which is why this has gone unnoticed; the damage is that files land in the wrong place (flattened into the top of `folder`, colliding across subfolders) and that `folder`-relative structure the caller built is lost. Blast radius: every `folder=`-accepting entry point, i.e. `sc.save()`/`sc.load()` (316, 86), `sc.loadtext()` (447), `sc.loadzip()` (518), `sc.unzip()` (560, 565 -- where it produces the wrong-path bug in finding 3), `sc.savezip()` (599), `sc.savejson()`/`sc.loadjson()` (1443, 1481, 1554), `sc.savefig()` (`sc_plotting.py:1446`, 1495-1507), `sc.savearchive()`/`sc.loadarchive()` (`sc_versioning.py:655`, 719), `sc.loadany()` (1169).

**Fix**: join rather than replace, e.g. `folder = os.path.join(folder, os.path.dirname(filename))` when both are given and `filename` is relative; leave `filename` alone when it is absolute (as the docstring promises). This changes where files land for anyone currently passing both `filename` with a directory and `folder`, so it needs a changelog note.

### 15. `jsonify(custom=...)` raises `KeyError` for any subclass of a registered class, including `sc.odict` — `sc_fileio.py:1287`

The match is by `isinstance` but the lookup is by exact type: `if isinstance(obj, custom_classes): return custom[obj.__class__](obj)`. Anything that is an instance of a registered class without being exactly that class passes the test and then fails the lookup. The line is above the `try/except`, so `die=False` does not apply.

```python
import numpy as np, sciris as sc
sc.jsonify(sc.odict(a=1), custom={dict: lambda x: 'DICT'}, die=False)
sc.jsonify(np.array([1,2]).view(np.matrix), custom={np.ndarray: lambda x: 'ARR'}, die=False)
```

Actual: `KeyError: <class 'sciris.sc_odict.odict'>` and `KeyError: <class 'numpy.matrix'>`. Expected: the registered handler is called (`'DICT'`, `'ARR'`), since `isinstance` already decided the object matches. Registering `dict` is a natural way to say "handle all mappings", and `sc.odict` is Sciris's own primary container.

**Fix**: resolve the handler the same way the match is made, e.g. `for cls, func in custom.items(): if isinstance(obj, cls): return func(obj)`, preferring the most derived match.

### 16. `jsonify()` raises `TypeError` on `Decimal` and `complex` instead of honouring `die=False` — `sc_fileio.py:1296`

`sc.isnumber()` is true for anything registered with `numbers.Number`, which includes `decimal.Decimal`, `complex` and `fractions.Fraction`, but the branch then calls `np.isnan(obj)` (line 1297) and `float(obj)` (line 1303), neither of which accepts those types. The branch sits above the `try/except` that implements the documented fallback ("`die` (bool): whether or not to raise an exception if conversion failed (otherwise, return a string)"), so `die=False` cannot help.

```python
import decimal, sciris as sc
sc.jsonify(decimal.Decimal('1.5'), die=False)
sc.jsonify(complex(1,2), die=False)
```

Actual: `TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''` and `TypeError: float() argument must be a string or a real number, not 'complex'`. Expected (per the docstring): a string representation, or a warning, rather than an exception.

Blast radius: `Decimal` is what `openpyxl`/`sqlite`/`json(parse_float=Decimal)` hand back, and a single `Decimal` anywhere in a nested structure takes down the whole `sc.savejson()`/`sc.saveyaml()` call.

Re-verification found two extensions of the same branch: `fractions.Fraction` raises the same way, and Python ints >= 2**64 crash too (`sc.jsonify(2**64, die=False)` raises `TypeError` from `np.isnan`). numpy complex values do *not* raise but are silently truncated to their real part (finding 39).

**Fix**: guard the numeric branch, e.g. use `math.isnan` only for `float`/`np.floating`, and wrap the `float(obj)`/`int(obj)` conversion in the same `try/except` that the rest of the function uses so `die=False` applies. Fix together with finding 39 (explicit complex handling, and `np.isnan` restricted to real floats).

### 18. `jsonify()` raises `AttributeError` in any thread other than the one that imported Sciris — `sc_fileio.py:1327`

The recursion memo is `jsonify_memo = threading.local()` with `jsonify_memo.ids = set()` executed once at import (lines 1227-1228). A `threading.local` attribute set in the importing thread does not exist in any other thread, so `if not obj_id in jsonify_memo.ids:` (line 1327) raises `AttributeError` there. The line is reached for any object exposing `to_json`/`tojson`/`toJSON`/`to_dict`/`todict` -- which includes every pandas `DataFrame` and `Series`. The failure is outside the surrounding `try`, so `die=False` does not soften it.

```python
import threading, pandas as pd, sciris as sc
out = {}
def work():
    try:    out['r'] = sc.jsonify(pd.DataFrame(dict(a=[1])))
    except Exception as E: out['r'] = f'{type(E).__name__}: {E}'
t = threading.Thread(target=work); t.start(); t.join()
print(out['r'])
print(sc.jsonify(pd.DataFrame(dict(a=[1]))))  # main thread, for contrast
```

Actual: `AttributeError: '_thread._local' object has no attribute 'ids'` in the worker; `{"a":{"0":1}}` in the main thread. Expected: the same result in both.

Blast radius: `sanitizejson` is the same function, `savejson`/`saveyaml`/`printjson` all route through it, and Sciris's own web tooling (`sc_app.py`) jsonifies inside request handlers, which are worker threads in every WSGI server.

**Fix**: initialise lazily inside `jsonify()`, e.g. `ids = getattr(jsonify_memo, 'ids', None); if ids is None: jsonify_memo.ids = ids = set()`, or use `jsonify_memo.__dict__.setdefault('ids', set())`.

### 19. `savejson()` truncates the output file before it knows the object can be serialised, destroying the previous contents — `sc_fileio.py:1488`

`with open(filename, 'w', encoding=encoding) as f: json.dump(jsonify(obj), f, ...)` opens (and therefore truncates) the target first, and only then runs `jsonify()` and the encode. Any failure in either leaves a zero-length or half-written file where a valid one used to be. `saveyaml()` does it the right way round (it jsonifies at line 1616, before opening at line 1621) and survives the same test, which confirms this is an ordering slip rather than a deliberate choice.

```python
import os, sciris as sc
sc.savejson('precious.json', dict(important='data'))
rec = {}; rec['self'] = rec              # jsonify() has no cycle guard for plain containers
try:    sc.savejson('precious.json', rec)
except RecursionError: pass
print(os.path.getsize('precious.json'), repr(open('precious.json').read()))
```

Actual: `0 ''` -- the previous file is gone. A mid-encode failure is just as destructive: `sc.savejson('precious.json', dict(a='ok', emoji='☃'), encoding='latin-1', ensure_ascii=False)` raises `UnicodeEncodeError` and leaves the file containing exactly `'{\n  "a": "ok",\n  "emoji": '`, i.e. invalid JSON. The same `saveyaml()` case leaves `precious.yaml` intact.

Note in passing that the module-level comment "Prevent recursive calls by storing a list of seen objects" (line 1226) only applies to the `to_json`/`to_dict` path; self-referencing dicts and lists still hit `RecursionError`.

A realistic trigger (beyond the contrived self-referencing dict) is finding 38: an object whose `to_dict()` returns numpy arrays makes `savejson('keep.json', dict(a=1, m=M()))` raise and leaves `keep.json` holding the partial text `'{\n  "a": 1,\n  "m": {\n    "x": '`.

**Fix**: build the string first (`text = json.dumps(jsonify(obj), indent=indent, **kwargs)`) and only then open and write. This is preferred over writing to a temporary file and `os.replace()`-ing it into place, which would silently replace symlinks, reset permissions/ownership and break hardlinks (see finding 1).

### 22. `loadspreadsheet()`'s argument-alias loop ignores its loop variable, so `sheetnum=` and `sheet_name=` raise `TypeError` — `sc_fileio.py:2124`

```python
for key in ['sheetname', 'sheetnum', 'sheet_name']:
    sheet = kwargs.pop('sheetname', sheet)
```

`key` is never used: the loop pops `'sheetname'` three times and leaves `'sheetnum'` and `'sheet_name'` in `kwargs`, which are then forwarded to `pd.read_excel()`.

```python
import sciris as sc
sc.loadspreadsheet('v.xlsx', sheetnum=0)
sc.loadspreadsheet('v.xlsx', sheet_name=0)
```

Actual: `TypeError: read_excel() got an unexpected keyword argument 'sheetnum'. Did you mean 'sheet_name'?` and, for `sheet_name` (pandas' own argument name, which the loop clearly intends to accept), a slightly different `TypeError`: `read_excel() got multiple values for keyword argument 'sheet_name'`. Expected: all three aliases map onto `sheet`, per the "renamed sheetname and sheetnum arguments to sheet" note in the docstring.

**Fix**: `sheet = kwargs.pop(key, sheet)`.

### 36. `sc.load()` rejects ordinary file objects, despite "Accepts either a filename ... or a file object" — `sc_fileio.py:87`

`_load_filestr()` only accepts `io.BytesIO`, so a normal binary file handle (the most common kind of file object) is rejected before any loading is attempted.

```python
import sciris as sc
sc.save('a.obj', dict(a=1))
with open('a.obj', 'rb') as f:
    sc.load(f)
```

Actual: `TypeError: First argument to sc.load() must be a string or file object, not <class '_io.BufferedReader'>; see also sc.loadstr()`. Expected: `{'a': 1}`.

**Fix**: accept any object with a `.read()` method (e.g. `hasattr(filename, 'read')`), read its bytes once, and dispatch from memory. This combines naturally with the fix for finding 6.

### 37. `die=True` is ignored whenever `remapping` is supplied: the whole object silently becomes a `NamedFailed` — `sc_fileio.py:2611`, `2618-2626`

With `remapping` given, the `'robust'` method is tried first. On failure, `find_class()` raises `UnpicklingError` because `self.die` is set, but `_RobustUnpickler.load()` wraps `super().load()` in `except Exception` and turns *any* error, including that deliberate one, into a top-level `Failed` object. That counts as a successful result, so `_unpickler` returns it. With `die=True` the user therefore gets back a `NamedFailed` in place of the *entire* object (sibling data lost) instead of an exception, while without `die` the same file loads with only the inner object failed.

```python
# nested.obj = dict(name='n', vals=[1,2], inner=pkg.oldmod.Gone(42)); pkg/oldmod.py then deleted
sc.load('nested.obj', die=True)                                           # UnpicklingError (correct)
sc.load('nested.obj', die=True, remapping={'pkg.oldmod.Gone':'nonexist.Mod'})
sc.load('nested.obj')['inner']                                            # NamedFailed only for 'inner'
```

Actual: the second call returns a `NamedFailed` for the whole top-level object with no exception (only a printed notice), so `die=False` recovers more data than `die=True`. Expected: `UnpicklingError`.

**Fix**: in `_RobustUnpickler.load()`, re-raise when `self.die` is true (or at least re-raise `UnpicklingError`), so `_unpickler` falls through to the next method or raises.

### 38. `jsonify()` returns the raw output of `to_dict()`/`to_json()` without sanitizing it, so `savejson()` crashes and truncates the file — `sc_fileio.py:1333`

`return obj_meth()` hands back whatever the object's `to_json`/`tojson`/`toJSON`/`to_dict`/`todict` method produced, unconverted. Any `to_dict()` that contains numpy scalars/arrays, dates or nested custom objects therefore yields non-JSON output.

```python
import numpy as np, sciris as sc
class M:
    def to_dict(self): return dict(x=np.arange(3))
sc.jsonify(M())                       # {'x': array([0, 1, 2])}  -- not JSON-compatible
sc.savejson('keep.json', dict(a=1, m=M()))
sc.saveyaml(obj=dict(m=M()))
```

Actual: `jsonify` returns `{'x': array([0, 1, 2])}`; `savejson` raises `TypeError: Object of type ndarray is not JSON serializable` and leaves `keep.json` holding `'{\n  "a": 1,\n  "m": {\n    "x": '` (finding 19); `saveyaml` emits `!!python/object/apply:numpy...` tags instead of plain YAML. Expected: `{'x': [0, 1, 2]}` in all three.

**Fix**: `return jsonify(obj_meth(), **kw)` inside the existing `try/finally` memo guard. The memo still prevents infinite recursion when `to_dict()` itself calls `sc.jsonify(self)`, and string results (e.g. pandas `to_json`) pass through unchanged.

### 39. `jsonify()` silently discards the imaginary part of numpy complex values — `sc_fileio.py:1303`

Silent data corruption. `np.complex128` subclasses Python `complex`, but `np.isnan()` accepts it, so it reaches `float(obj)`, where numpy truncates to the real part with only a `ComplexWarning`.

```python
import numpy as np, sciris as sc
sc.jsonify(np.complex128(1+2j))                  # 1.0
sc.jsonify(dict(z=np.array([1+2j, 3-4j])))       # {'z': [1.0, 3.0]}
```

Actual: `1.0` and `{'z': [1.0, 3.0]}`. Expected: the value preserved (e.g. `[re, im]`, `{'real':..., 'imag':...}`, or `str`), or an error. Plain Python `complex` raises `TypeError` instead (finding 16), so the two are also inconsistent.

**Fix**: add an explicit `isinstance(obj, (complex, np.complexfloating))` branch before the float conversion, as part of the finding 16 fix. The same fix should restrict `np.isnan` to real floats, which also covers Python ints >= 2**64.

## Low severity

### 13. `rmpath()` deletes a path it has just reported as unremovable, using the previous iteration's removal function — `sc_fileio.py:1088`

The `else` branch for "exists but is neither a file nor a folder" prints/raises but has no `continue`, so control falls through to `rm_func(path)` at line 1110. `rm_func` is a loop-local left over from the previous iteration (or unbound on the first), so the path is removed anyway with whatever function the last path happened to need.

```python
import sciris as sc, os
os.mkfifo('fifo'); open('doomed.txt','w').close()
sc.rmpath(['doomed.txt', 'fifo'], die=False)
print('fifo gone:', not os.path.exists('fifo'))
```

Actual:

```
Removed "doomed.txt"
Path "fifo" exists, but is neither a file nor a folder: unable to remove
Removed "fifo"
fifo gone: True
```

Expected: the fifo is skipped, as the message says. If the unremovable path comes first the fallthrough instead produces a confusing internal error:

```python
os.mkfifo('fifo2'); sc.rmpath('fifo2', die=False)
# actual:  Path "fifo2" exists, but is neither a file nor a folder: unable to remove
#          Could not remove "fifo2": cannot access local variable 'rm_func' where it is not associated with a value
```

If the previous path was a directory, `rm_func` is `shutil.rmtree`, which is then applied to a path the function decided it could not handle. `sc.rmpath()` is the library's delete primitive and is used throughout the test suite (`tests/test_fileio.py:92`, `133`, `285`, `307`), so a stale-function delete is worth closing even though the triggering path types are unusual.

**Severity (re-verified)**: downgraded from Medium to Low, since it requires a FIFO/socket (or similar) plus `die=False`; it is nonetheless a clear logic error with a one-line fix.

**Fix**: add `continue` after the error message in the `else` branch at 1088 (and set `rm_func = None` at the top of each iteration so a fallthrough cannot silently reuse the last one).

### 21. `asdataframe` is documented but has no effect for the default `method='pandas'` — `sc_fileio.py:2116`

`asdataframe (bool): whether to return as a pandas/Sciris dataframe (default True)` is only read inside the `method='xlrd'` branch (line 2145); the pandas branch returns `pd.read_excel(...)` unconditionally, so `asdataframe=False` is silently ignored.

```python
import sciris as sc
print(type(sc.loadspreadsheet('v.xlsx', header=0, asdataframe=False)).__name__)
```

Actual: `DataFrame`. Expected: a list of lists / list of odicts, as the xlrd branch returns, or an explicit error saying the option is unsupported for this method. (The docstring's "pandas/Sciris dataframe" is also inaccurate: the pandas branch always returns a plain `pandas.DataFrame`, never an `sc.dataframe`.)

**Severity (re-verified)**: downgraded from Medium to Low.

**Fix**: honour the flag in the pandas branch (`return data.values.tolist()` or similar when `asdataframe is False`), or raise if it is set with an unsupported method.

### 23. `savespreadsheet(formats=...)` without `formatdata` dies with `UnboundLocalError` — `sc_fileio.py:2327`

`hasformats` requires *both* `formats` and `formatdata` (line 2283), but the fallback `thisformat = workbook.add_format({})` is only executed in the `else` branch of `if formats is not None:` (lines 2327-2328). Supplying `formats` alone therefore leaves `thisformat` undefined by the time `worksheet.write(r, c, cell_data, thisformat)` runs.

```python
import numpy as np, sciris as sc
sc.savespreadsheet(filename='fmt.xlsx', data=np.array([[1,2]]), formats={'plain':{}})
```

Actual: `UnboundLocalError: cannot access local variable 'thisformat' where it is not associated with a value`. Expected: the data written with default formatting, or a clear message that `formats` requires `formatdata`.

**Severity (re-verified)**: downgraded from Medium to Low.

**Fix**: define `thisformat = workbook.add_format({})` unconditionally before the write loop (the `hasformats` branch overwrites it per cell anyway).

### 24. A remapping value naming a dotted module (no class) is mis-split into `(module, class)`, and can collapse the whole load — `sc_fileio.py:2556`

`_remap_module()` unconditionally does `remapped = tuple(remapped.rsplit('.', 1))` for any string value, so `{'oldmod': 'newpkg.newmod'}` -- remapping a whole module to a new dotted location, which is the natural way to handle a package reorganisation -- is read as module `newpkg`, class `newmod`. The in-line comment on that line even claims the wrong result (`# e.g. 'foo.bar.Cat' -> ('foo.bar',)`). Two outcomes, both bad: if `newpkg` has no attribute `newmod` the remap fails and you get a `Failed` placeholder; if it does (because the submodule has been imported, which is normal), `getattr` returns the *module object*, `find_class` hands a module to the unpickler and the entire top-level load dies, losing everything else in the file.

```python
# pkg/livemod3.py contains: class Gone: ...   ; nested.obj holds dict(name=..., vals=..., inner=oldmod.Gone(42), ...)
import sciris as sc
o = sc.load('nested.obj', remapping={'mymod': 'pkg.livemod3'})
print(type(o['inner']), o['inner']._failure.error)

import pkg.livemod3                              # now getattr(pkg, 'livemod3') resolves
o = sc.load('nested.obj', remapping={'mymod': 'pkg.livemod3'})
print(type(o), o.isempty(), o._failure.error)
```

Actual: first, `<class 'sciris.sc_fileio.NamedFailed'> module 'pkg' has no attribute 'livemod3'`; second, `<class 'sciris.sc_fileio.NamedFailed'> True NEWOBJ class argument must be a type, not module` -- the outer dict, including the three keys that had nothing wrong with them, is gone. Expected: the module-level remap resolves to `pkg.livemap3.Gone`, as it does when spelled `remapping={'mymod': ('pkg.livemod3',)}`, which works and returns `<class 'pkg.livemod3.Gone'>` with `{'x': 42}`.

Blast radius: `sc.load()`'s own docstring advertises only the class-level forms, so this is a partially-documented path, but module-level keys are the third supported key form (line 2537) and the tuple workaround is undocumented.

**Severity (re-verified)**: downgraded from Medium to Low. Module-only *keys* are a supported key form (line 2537), but module-only string *values* are not documented, and the tuple form `('pkg.livemod3',)` already works.

**Fix**: only split the string when the tail looks like a class (or, better, resolve the whole dotted string with `importlib.import_module` first and fall back to splitting), and if the resolved target is a module rather than a class, keep the original class name and `getattr` it from that module. Also check that the resolved object is a `type` before returning it, so a module can never reach the unpickler.

### 25. `sc.load()` cannot load a pickle whose top-level payload is `None` — `sc_fileio.py:2682`

`_unpickler()` uses `obj is None` as its "every method failed" sentinel, so a successfully-unpickled `None` is reported as a total failure. `sc.save()` explicitly supports this payload via `die='never'`.

```python
import sciris as sc
sc.save('none.obj', None, die='never')
sc.load('none.obj')
```

Actual: `sciris.sc_fileio.UnpicklingError` (the chained message is "All available unpickling methods failed: ..." followed by the generic "Unable to load file ... as a gzipped pickle file" advice). Expected: `None`.

**Severity (re-verified)**: downgraded from Medium to Low.

**Fix**: use a dedicated sentinel (`obj = _notset` initially, `if obj is _notset:` for the failure check) instead of overloading `None`.

### 26. A `# pragma: no cover` comment was pasted inside an error-message string literal — `sc_fileio.py:130`

The pragma landed inside the quotes instead of at the end of the line, so users see it in the exception text, and the branch is not actually excluded from coverage.

```python
sc.save('full.obj', dict(a=np.arange(500)))
d = open('full.obj','rb').read(); open('tr.obj','wb').write(d[:len(d)//2])   # truncated gzip
sc.load('tr.obj')
```

Actual: `EOFError: sc.load(): Could not open the # pragma: no cover file string for an unknown reason; see error above for details`. Expected: `sc.load(): Could not open the file string for an unknown reason; ...`.

Two smaller problems on the same line: the message says "for an unknown reason" even though the original exception is known and informative (`EOFError: Compressed file ended before the end-of-stream marker was reached`), and `raise exc(errormsg)` reconstructs the original exception *type* with a single string argument, which breaks for any exception class whose constructor needs more than one argument (e.g. `UnicodeDecodeError`). The original is chained with `from E`, so it is at least visible in the traceback.

This is cosmetic and trivial to fix.

**Fix**: move the pragma outside the string, include `str(E)` in the message, and prefer `raise` (bare re-raise) or a plain `RuntimeError` over `exc(errormsg)`.

### 27. `sc.thisdir()` silently drops a `..` passed as the first argument, and anchors the result to the cwd — `sc_fileio.py:748`, `756`

The first positional argument is `file`, and `folder = os.path.abspath(os.path.dirname(file))`. For `file='..'`, `os.path.dirname('..')` is `''`, so the parent component vanishes and `abspath('')` resolves to the *current working directory* rather than anything related to the calling script. The docstring example claims otherwise.

Run from a script in `.../fileio1/scriptdir/` with cwd `.../fileio1/`:

```
sc.thisdir()                     -> .../fileio1/scriptdir            (correct)
sc.thisdir('.')                  -> .../fileio1                      (cwd, not the script's folder)
sc.thisdir('..')                 -> .../fileio1                      ('..' dropped entirely)
sc.thisdir('..','tests','m.py')  -> .../fileio1/tests/m.py           (docstring: "Merge parent folder with subfolders and a file")
```

The documented `sc.thisdir('..', 'tests', 'mytests.py')` therefore never goes to a parent folder, and its result moves with the cwd -- the opposite of what `sc.thisdir()` exists for. The `sc.thisdir('.')` example is hedged with "Ditto (usually)", but "usually" is exactly "when the cwd happens to equal the script's folder".

`sc.thisfile()`/`sc.thisdir()`/`sc.thispath()` frame resolution itself is correct (see Verified clean); only the relative-`file` handling is wrong.

**Fix** (correcting the docstring is sufficient): treat a `file` argument that is a directory (or `.`/`..`) as a path component rather than a file, and anchor it to `thisfile(frame=...)`'s folder; or document that a relative `file` is cwd-relative and correct the `'..'` example to `sc.thisdir(path='../tests/mytests.py')`.

### 28. `aspath` is silently ignored by `getfilepaths()`, `sanitizepath()`, and `makepath()` — `sc_fileio.py:845`, `909`, `1045`

All three aliases accept `aspath` in their signature (or via `**kwargs`) and then hardcode `aspath=True` in the delegated call, so the argument cannot be turned off. `sc.thispath()` (line 770) forwards it correctly, which shows the intent.

```python
type(sc.getfilepaths('.', aspath=False)[0]).__name__   # actual: 'PosixPath'  expected: 'str'
type(sc.sanitizepath('f', aspath=False)).__name__      # actual: 'PosixPath'  expected: 'str'
type(sc.makepath('f', aspath=False)).__name__          # actual: 'PosixPath'  expected: 'str'
type(sc.thispath(aspath=False)).__name__               # 'str' -- correct
```

`getfilepaths` and `sanitizepath` capture `aspath` as a keyword-only parameter and drop it; `makepath` forwards `**kwargs` but appends `aspath=True`, which would be a `TypeError` for a duplicate keyword were `aspath` not already bound by the signature default.

**Fix**: pass `aspath=aspath` in all three, matching `thispath()`.

### 30. `makefilepath(ext=...)` produces `f..obj` for a dotted extension and skips the extension for a suffix-less match — `sc_fileio.py:989`

The guard is `not basename.endswith(ext)` and the append is `basename += '.' + ext`, so neither form of `ext` is handled robustly: with a leading dot the separator is doubled, and without one the test matches any name merely *ending* in those letters.

```python
sc.makefilepath('f', ext='.obj', abspath=False)          # actual: 'f..obj'      expected: 'f.obj'
sc.makefilepath('myfileobj', ext='obj', abspath=False)   # actual: 'myfileobj'   expected: 'myfileobj.obj'
sc.makefilepath('f.txt', ext='obj', abspath=False)       # actual: 'f.txt.obj'   (reasonable)
```

Blast radius: the internal callers all pass a dotless extension (`sc.savefig()` forwards the caller's `filetype` as `ext` at `sc_plotting.py:1503`/`1507`, and `sc.Spreadsheet.save()` uses `ext='xlsx'` at `sc_fileio.py:2090`), so the double-dot form is reachable only through direct `sc.makefilepath()`/`sc.makepath()` calls -- `sc.savefig('fig', filetype='.png')` is rejected earlier by savefig's own metadata check (`ValueError: ... unsupported type for metadata`). The `endswith` trap, by contrast, applies to every caller. The branch is marked `# pragma: no cover` but is reached by all of the above.

**Fix**: normalize with `ext = ext.lstrip('.')` and test `not basename.endswith('.' + ext)`.

### 31. `sc.rmpath()` cannot remove a symlink to a directory and raises by default — `sc_fileio.py:1086`

`os.path.isdir()` follows symlinks, so a symlink pointing at a directory is dispatched to `shutil.rmtree()`, which refuses symlinks. With the default `die=True` this aborts the whole call.

```python
import sciris as sc, os
os.makedirs('rd', exist_ok=True); os.symlink('rd', 'ld')
sc.rmpath('ld')     # actual: OSError: Cannot call rmtree on a symbolic link
os.remove('ld')     # works fine
```

Actual: `OSError: [Errno None] None: 'ld'` (`die=False`) / `OSError('Cannot call rmtree on a symbolic link')` (`die=True`); the symlink survives. Expected: the link is removed and the target left alone. `sc.rmpath(sc.getfilelist(...))` -- the documented usage -- will therefore abort on any tree containing a symlinked directory, after having already deleted the earlier entries in the list.

**Fix**: test `os.path.islink(path)` before `os.path.isdir(path)` and use `os.remove`/`os.unlink` for links.

### 32. `Blobject`: a `Path` source loses the `name` attribute, and `blob=` plus `filename=` is rejected with a misleading message — `sc_fileio.py:1752`

`__init__` derives `filename`/`name` with `sc.isstring(source)` (line 1755), but the `Path`-to-`str` conversion happens later, in `load()` (line 1795). A `Path` source therefore leaves `name=None` even though `load()` goes on to set `self.filename`, contradicting the code's own intent ("If not supplied, use the filename"). Separately, `filename` is promoted to `source` at line 1754 before the mutual-exclusion check at line 1757, so passing `blob=` together with the documented `filename=` ("used as default for load/save") raises `ValueError: Can initialize from either source or blob, but not both` -- when the caller supplied no source at all.

```python
import sciris as sc
from pathlib import Path
print(sc.Blobject('raw.bin').name, sc.Blobject(Path('raw.bin')).name)
sc.Blobject(blob=b'abc', filename='raw.bin')
```

Actual: `raw.bin None`, then `ValueError: Can initialize from either source or blob, but not both`. Expected: `raw.bin raw.bin`, and a `Blobject` holding `b'abc'` with `filename='raw.bin'`.

**Fix**: normalise `Path` to `str` at the top of `__init__`, and only promote `filename` to `source` when `blob is None`.

### 33. The `savespreadsheet()` formatting docstring example raises `TypeError` as written — `sc_fileio.py:2269`

The fifth example builds `testdata5` as an object array whose first row holds the strings `'A','B','C'`, then writes `formatdata[testdata5>0.7] = 'big'`, which compares those strings against a float.

```python
import numpy as np, sciris as sc
nrows = 15; ncols = 3
formats = {'header':{'bold':True, 'bg_color':'#3c7d3e', 'color':'#ffffff'}, 'plain': {}, 'big': {'bg_color':'#ffcccc'}}
testdata5  = np.zeros((nrows+1, ncols), dtype=object)
formatdata = np.zeros((nrows+1, ncols), dtype=object)
testdata5[0,:] = ['A', 'B', 'C']
testdata5[1:,:] = np.random.rand(nrows,ncols)
formatdata[1:,:] = 'plain'
formatdata[testdata5>0.7] = 'big'
```

Actual: `TypeError: '>' not supported between instances of 'str' and 'float'` on the last line. Expected: the example runs. Sciris's own test suite already works around it -- `tests/test_fileio.py:43` uses `formatdata[1:,:][testdata[1:,:]>0.7] = 'big'`, i.e. it excludes the header row from the comparison.

**Fix**: change the docstring line to `formatdata[1:,:][testdata5[1:,:]>0.7] = 'big'`, matching the test.

### 34. `except (NameError or AttributeError)` catches only `NameError`, leaving the compatibility-map fallback dead (and broken) — `sc_fileio.py:2371`

`(NameError or AttributeError)` evaluates to `NameError`, so the guard around `known_fixes = pd.compat.pickle_compat._class_locations_map` does not catch the error that a moved or removed pandas private API would actually raise, and `import sciris` would fail outright. The fallback dict is also unusable as written: its values are dicts (`{('pandas.core.indexes.numeric','Int64Index'): {'pandas...Int64Index': 'pandas...Index'}}`), and `_remap_module()` has no dict case, so it falls through to "assume the user supplied the object directly" and returns the dict itself as the class.

```python
print((NameError or AttributeError) is NameError)
class Fake: pass
try:    Fake.pickle_compat._class_locations_map
except (NameError or AttributeError): print('caught')
except AttributeError as E: print('escaped:', E)

from sciris import sc_fileio as scf
fallback = {('pandas.core.indexes.numeric','Int64Index'): {'pandas.core.indexes.numeric.Int64Index':'pandas.core.indexes.api.Index'}}
print(scf._remap_module(fallback, 'pandas.core.indexes.numeric', 'Int64Index'))
```

Actual: `True`, `escaped: type object 'Fake' has no attribute 'pickle_compat'`, `{'pandas.core.indexes.numeric.Int64Index': 'pandas.core.indexes.api.Index'}` (a dict where a class is required). Expected: `except (NameError, AttributeError):` and fallback values in the `('module', 'name')` or `'module.Name'` form the remapper understands.

This is latent: pandas 3.0.5 still provides `_class_locations_map`, so the fallback is not reached today; but if a future pandas removes it, `import sciris` will fail.

**Fix**: use a tuple, not `or`, in the `except`, and change the fallback values to `('pandas.core.indexes.base', 'Index')`.

### 40. `Blobject.save()` with no filename ignores `self.filename`, writes to `./default`, and then sets `self.filename=None` — `sc_fileio.py:1819-1824`

The class docs say `filename` is "used as default for load/save". `load()` honours it; `save()` does not: it passes `filename=None` straight to `makefilepath()` and then overwrites `self.filename` with that `None`.

```python
import sciris as sc
B = sc.Blobject('raw.bin')
B.save()          # "Object saved to .../default."  (file named 'default', no extension)
B.filename        # None  -- original filename lost
```

Actual: the data is written to `./default` and `B.filename` becomes `None`. Expected: the data is written back to `raw.bin` and `B.filename` is unchanged.

**Fix**: `if filename is None: filename = self.filename` before `makefilepath()`. (`Spreadsheet.save()` has an explicit `'spreadsheet.xlsx'` default in its signature, which the tests rely on, so that one is intentional.)

### 41. `loadspreadsheet(fileobj=..., method='openpyxl')` ignores `fileobj` — `sc_fileio.py:2135`

The docstring says it can "Read from either a filename or a file object", but the openpyxl branch does `Spreadsheet(fullpath)`, and with no filename `fullpath` is `makefilepath(None)`, i.e. `./default`.

```python
import io, sciris as sc
sc.loadspreadsheet(fileobj=io.BytesIO(open('v.xlsx','rb').read()), method='openpyxl')
```

Actual: `FileNotFoundError: [Errno 2] No such file or directory: '.../default'`. Expected: the sheet contents, as with the default pandas method.

**Fix**: `spread = Spreadsheet(fileobj if fileobj is not None else fullpath)`.

### 42. `sc.save(..., verbose=None)` crashes — `sc_fileio.py:276`, `283`

`verbose>=2` raises `TypeError` for `None`, which is the Sciris-wide convention for "default verbosity" and the default of `sc.load()`. The first comparison is inside the `try`, so its error is swallowed but sends control into the `except`, where the second `verbose>=2` raises out of the function.

```python
import sciris as sc
sc.save('v.obj', 1, verbose=None)
```

Actual: `TypeError: '>=' not supported between instances of 'NoneType' and 'int'`. Expected: the object is saved with default verbosity.

**Fix**: `verbose = verbose or 0` at the top of `save()`.

### 43. `sc.rmpath(sc.getfilelist(folder))` deletes everything and then raises `FileNotFoundError` — `sc_fileio.py:1076-1079`

`getfilelist(folder)` (default pattern `'**'`, recursive) returns the folder itself first, followed by its contents. `rmtree` on the first entry removes everything, and the next entry then trips the "does not exist" check, which raises by default (`die=True`).

```python
# rmme/a/f.txt, rmme/g.txt
import sciris as sc
fl = sc.getfilelist('rmme')      # ['rmme/', 'rmme/a', 'rmme/a/f.txt', 'rmme/g.txt']
sc.rmpath(fl)
```

Actual: `FileNotFoundError: Path "rmme/a" does not exist`, although all files are already gone. Expected: the tree is removed with no error, since `sc.rmpath(sc.getfilelist(...))` is the documented example pattern.

**Fix**: before removing, drop entries that lie inside another directory in the list (or skip nonexistent paths whose ancestor was removed earlier in the same call, without raising).

## Cross-checked between halves

Two things bound how much to trust the rest of the module, because they were tested from two directions (one auditor per half, each independently probing the boundary the other half's code depends on).

**The robust unpickler itself came out well.** A class deleted from a surviving module, a class whose whole module has been deleted, and a class that gained or lost attributes all load: the outer container is fully recovered (verified key by key on a five-key dict), the surviving sibling objects keep their real classes, and the failed object becomes a `NamedFailed` carrying the recovered `__dict__`, `_module`, `_name` and `_failure` (error, exception, traceback) with placeholders that carry the original error and warn rather than fail silently. All seven `remapping=` forms were verified: a dotted string key, a tuple key, a bare module key, a class object as the value, a dotted `'module.Class'` string value, a `('module','Class')` tuple value, a `('module',)` one-tuple value, and `None` (-> `NoneObj`); a remapping whose target does not exist falls back to a `NamedFailed` with the remap error recorded rather than crashing. The one real gap is finding 24: a module-only dotted remapping target (`'newpkg.newmod'` with no class) is mis-split into `(module, class)` and can collapse the whole load.

**Zip-Slip is neutralised, but by CPython's `extractall`, not by any Sciris guard.** `savezip(basename=False)` can store a `..` component in an arcname (CPython's `ZipInfo.from_file` strips leading separators and drives but not `..`), so a crafted archive can contain a traversal member; extraction is nonetheless safe here because `unzip()` delegates to `ZipFile.extractall`, whose `_extract_member` drops `''`, `.` and `..` components -- a member named `../escaped.txt` landed at `outfolder/escaped.txt` and nothing was written outside `outfolder`. Sciris adds no guard of its own, so the safety is inherited from the standard library rather than enforced; the class of issue to watch is that `unzip()`'s own path arithmetic at line 565 uses the *raw*, unsanitized member name (it is only saved by `makefilepath()` reducing it to a basename, which is itself finding 3's bug). `savezip` will still happily store a `..` arcname with no warning.

## Misplaced `# pragma: no cover`

Both halves found pragmas sitting on reachable paths, several on paths with confirmed bugs -- hiding exactly the code that most needs testing. All rows were confirmed executed (by direct call, or by tracing line events with `sys.settrace`).

| Line | Branch | Reachable via |
|------|--------|---------------|
| 104 | `except Exception as E:` in `_load_filestr` | any non-gzip file: `sc.load('notes.txt')`, `sc.load()` on a zstd or uncompressed pickle |
| 130 | the "unknown reason" re-raise | `sc.load()` on a truncated gzip file (finding 26) |
| 279, 286 | pickle failure -> dill fallback in `serialize()` | `sc.save('f.obj', threading.Lock())` (pickle fails, dill succeeds) |
| 319 | `if obj is None:` | `sc.save('f.obj')` with no object -- the documented `allow_empty`/`die='never'` path |
| 333 | `if tobytes: filecontext = closing(bytestream)` | `sc.save(filename=None, compression='zstd')`, `sc.dumpstr(obj, compression='none')` (finding 7) |
| 348 | invalid-compression `else` | `sc.save('f.obj', 1, compression='bogus')` (rejected finding 8) |
| 352, 357 | `verbose` print and the bytestream return | `sc.save('f.obj', 1, verbose=1)`; `sc.dumpstr(obj)` (every archive save) |
| 479 | `savetext()` coercion of a non-string, non-list | `sc.savetext('d.txt', {'a':1})` -- the documented "can also save an arbitrary object" |
| 617 | `if orig.is_dir():` in `savezip()` | `sc.savezip('z.zip', ['somefolder'])` -- documented ("file(s) and/or folder(s) to compress") |
| 963 | `folder` supplied as a list | `sc.makefilepath('f.txt', folder=['/tmp','a','b'])` -- documented ("if a list, fed to `os.path.join()`") |
| 974 | resolution of `filename` from `default` | `sc.makefilepath(None, default='d.obj')`; used internally by `sc.savefig()` (`sc_plotting.py:1495-1507`) |
| 989 | `ext` append | `sc.makefilepath('f', ext='obj')`; used internally by `sc.savefig()` and `sc.Spreadsheet.save()` (finding 30) |
| 1006 | `makedirs` failure handler | `sc.makefilepath('/proc/x/f.txt', makedirs=True, die=False)` |
| 1016 | `checkexists` block | `sc.makefilepath('f.txt', checkexists=True)` -- the entire point of the argument |
| 1028, 1030 | `verbose` print, `split` return | `sc.makefilepath('f.txt', verbose=True)`; `sc.makefilepath('sub/f.txt', split=True)` (the docstring's own "complex example" uses `split=True`) |
| 1076 | `if not os.path.exists(path):` in `rmpath()` | `sc.rmpath('nothere', die=False)` |
| 1086 | `elif os.path.isdir(path):` | `sc.rmpath('somefolder')` -- half of what the function is for |
| 1088 | "neither a file nor a folder" | `sc.rmpath(fifo)` (finding 13) |
| 1113 | `except` around `rm_func(path)` | `sc.rmpath(symlink_to_dir, die=False)` (finding 31) |
| 1297 | `if np.isnan(obj):` | `sc.jsonify(np.nan)` -> `None` |
| 1314 | 0-D array branch | `sc.jsonify(np.array(5))` -> `[5]` |
| 1322 | `if isinstance(obj, (dt.time, dt.date, dt.datetime, uuid.UUID)):` | `sc.jsonify(dt.date(2020,1,1))` -> `'2020-01-01'` |
| 1327 | `if not obj_id in jsonify_memo.ids:` | `sc.jsonify(pd.DataFrame(dict(a=[1])))` (also the site of finding 18) |
| 1483, 1607 | `if obj is None and not keepnone:` | evaluated on every `sc.savejson()` / `sc.saveyaml()` call |
| 1548 | `if string is not None or not fromfile:` in `loadyaml()` | `sc.readyaml('{"a":1}')` -- the documented `sc.loadyaml(string=...)` path |
| 1625 | `else:` (return the YAML as a string) | `sc.saveyaml(obj=dict(a=1))` -- the second docstring example |
| 1906, 1958 | `def openpyxl()`, `def update()` | called by `writecells()`, whose docstring examples are the documented API |
| 2045, 2058 | `if len(cells) != len(vals):`, `if isinstance(val, tuple):` | evaluated by every `writecells(cells=...)` call |
| 2134 | `elif method == 'openpyxl':` in `loadspreadsheet()` | `sc.loadspreadsheet('p3.xlsx', method='openpyxl')` -- the second docstring example |
| 2286, 2291 | dict-input branches of `savespreadsheet()` | `sc.savespreadsheet(filename=..., data=sc.odict(...))` -- the fourth docstring example |
| 2327 | `else: thisformat = workbook.add_format({})` | every `savespreadsheet()` call without `formats` (finding 23) |
| 2474, 2478 | `if verbose:` / `if tostring:` in `Failed.showfailure()` | `repr(failed_obj)` -- reached whenever a failed object is printed |
| 2572 | `obj = getattr(module, name)` | every successful `remapping=` -- the core line of the remapping feature |
| 2609 | `if self.verbose is not None:` in `find_class()` | reached on every unpickling failure; `self.verbose` is coerced to `False` in `__init__` (line 2590), so the condition is in fact always true and the `die`/warn behaviour it guards is unconditional |
| 2634 | `if kwargs:` in `_unpickler()` | evaluated on every `sc.load()` |

## Verified clean

Recorded so the same ground isn't re-covered. All of the following were hypothesised, tested by execution, and found correct.

**Pickle round-trip fidelity.** Round-tripped through `sc.save()`/`sc.load()` and checked for exact equality (not just `==`): nested dicts with tuples; numpy arrays across `int64`, `float64`, `float32`, `|S2`, C-order, explicit Fortran order (`np.asfortranarray`), and an empty `(0,3)` array, verifying `dtype`, `shape`, both contiguity flags, and `tobytes()` byte-for-byte; a big-endian `>i4` array (dtype is normalized to native `int32`, but the *values* are correct and plain `pickle` does exactly the same, so this is numpy behaviour and not a Sciris defect); `pandas.DataFrame` with a named string index, with a two-level `MultiIndex` (names preserved), and with a `DatetimeIndex` (all `.equals()`-identical including the index); `sc.dataframe` (class preserved); `sc.odict` and `sc.objdict` (class, key order, and attribute access preserved); a custom class defined at module level in `__main__` (correctly reconstructed as `__main__.Custom`); `datetime.datetime` with microseconds, `datetime.date`, `datetime.timedelta`, `sc.date`, `set`, `frozenset`, and `complex`. No silent mutation of the caller's object: the MD5 of `pickle.dumps(obj)` for a nested `dict`/`sc.odict`/ndarray/list/tuple structure was identical before the save, after the save, and after the subsequent load.

**Compression.** Every documented value round-trips losslessly and writes the codec it claims, verified by magic bytes: `'gzip'` and `'gz'` -> `\x1f\x8b`; `'zstd'`, `'zst'`, `'zstandard'` -> `\x28\xb5\x2f\xfd`; `'none'` -> a bare pickle (`\x80\x04...`); `sc.zsave()` -> zstd. `sc.load()` auto-detects all three from disk without being told. `compresslevel` genuinely reaches the codec and is not ignored: gzip 0/1/5/9 gave 1700341/1519033/1509329/1509329 bytes and zstd -5/1/5/15/22 gave 1600375/1502109/1502110/1502003/1502793 bytes on the same 1.7 MB object; `compresslevel` with `compression='none'` is accepted and ignored without error, as expected.

**Error paths and file descriptors.** A nonexistent file raises `FileNotFoundError` under both `die=True` and `die=False` (correctly re-raised directly at line 107 rather than being swallowed by the fallback chain). A gzip file with a flipped interior byte behaves as documented: `die=False` returns a `NamedFailed` placeholder *and* prints the full `_unpicklingerror()` diagnostic, `die=True` raises `UnpicklingError`; the original error is preserved in the chained traceback in both cases. Unpickling a class that no longer exists exercises the same `Failed`/`NamedFailed` machinery and emits an `UnpicklingWarning` naming the missing module and class. A truncated gzip raises under both `die` values (message wording aside). No descriptor leak was found on any of these paths: 20 repetitions each of loading a truncated gzip, a text file, an empty file, a missing file, `sc.save()` of an unserializable object under gzip/zstd/none, and `sc.loadzip()` of a missing zip all gave a net delta of 0 open descriptors and no `ResourceWarning` under `-W error::ResourceWarning`; the only leak found was the invalid-`compression` path (rejected finding 8). Twenty consecutive failed `loadjson`/`loadyaml` calls (bad syntax and missing file) also leak no file descriptors and raise no `ResourceWarning`.

**Path functions.** `makefilepath()`'s matrix: `makedirs=True` creates exactly the intended directory chain and nothing else (verified by diffing the directory listing: `'a/b/c/f.txt'` created only `a`, `a/b`, `a/b/c`); `makedirs=False` creates nothing. `checkexists` reports correctly in all four combinations (`True`+existing -> pass, `True`+missing -> `FileNotFoundError`, `False`+existing -> `FileExistsError`, `False`+missing -> pass) and downgrades to a printed message under `die=False`. The v3.2.9 changelog claim "allowed `sc.makefilepath()` to create folders when no filename is specified" holds as stated: `makefilepath(None, folder='folder_only', makedirs=True)` creates and returns the folder with no `default` filename appended, and the trailing-slash form `makefilepath('newfolder/', makedirs=True)` is handled by the same directory-only branch. `default` works as a scalar and as a list with leading `None`s, and falls back to the literal name `'default'` when everything is `None` and no folder is given. `abspath=False` leaves relative paths and `~` untouched; `abspath=True` expands `~`. `split=True` returns `(folder, basename)` correctly (`(dir, '')` in directory-only mode). `ext` does not double-append when the extension is already present in the plain (dotless) form. `thisfile`/`thisdir`/`thispath` frame resolution: all three resolve to the correct file with the default `frame=1` from module level, from inside a function, and from inside a nested (closure) function, both in the main script and in an imported helper module; `frame=2` correctly reports the *calling* script from all three. No off-by-one was found, including through the `frame+1` hops at lines 752 and 770.

**Zip.** `savezip`/`loadzip` preserve member *names* exactly with the default `basename=False`, including nested folders (`src/a/deep/nested.txt`) and same-basename files from different folders, and `loadzip` returns the contents byte-exactly with the full member paths as dict keys. `savezip` of a folder recurses correctly and stores explicit directory entries. `members=` is honoured for the extraction itself (only the requested file appears on disk); only the return value ignores it (finding 9). `loadzip` on a non-zip file raises `BadZipFile` with no descriptor leak.

**JSON/YAML.** `jsonify()` type fidelity confirmed correct and non-lossy for: `np.int64`/`np.uint8` -> `int`, `np.float32` -> `float`, `np.bool_` -> `bool`, 1-D and 2-D arrays -> nested lists with the shape preserved (`np.arange(6).reshape(2,3)` -> `[[0,1,2],[3,4,5]]`), 0-D arrays -> a one-element list, `set`/`tuple` -> list (the tuple->list flattening is not documented but is unavoidable in JSON and survives `saveyaml(jsonify=False)` as `!!python/tuple`), `datetime`/`date`/`UUID` -> `str`, `Path` and `timedelta` -> the jsonpickle `py/reduce` form, `bytes` -> its `repr`, and a custom object -> `{'python_class': ..., **vars}`. `custom=` works for exact-class matches including the docstring example. `tostring=True` with `indent=` and the `sc.printjson()`/`sc.readjson()`/`sc.readyaml()` docstring examples all produce the documented output. `strkeys=False` correctly preserves int, bool and `None` keys. `savejson`/`loadjson` and `saveyaml`/`loadyaml` agree with each other on every type listed above, including the `nan` -> `None` and `inf` -> `inf` results (so the two formats are at least mutually consistent; see rejected finding 17 on `Infinity`). `loadjson(string=...)`, `loadjson(filename, fromfile=False)` (documented argument swap) and `loadjson(string=..., filename=...)` (string wins) all behave as documented; `loadjson()` with no arguments raises a clear `FileNotFoundError` naming the `string`/`fromfile` alternatives. `loadyaml(safe=True)` correctly refuses a `!!python/tuple` document that the default unsafe loader accepts, `loader=yaml.loader.UnsafeLoader` overrides `safe=True` as documented, `sort_keys=True/False` works and cannot be overridden by a colliding `kwargs` entry, multi-document YAML returns a list and a single document is unwrapped, and `saveyaml(jsonify=False)` preserves tuples. `encoding=` is respected on both read and write, but is a no-op for ordinary use because `json.dump` defaults to `ensure_ascii=True` and `yaml.dump` to `allow_unicode=False`, so non-ASCII content is escaped to ASCII before the encoder sees it (`'αβγ'` is written as `αβγ` and a latin-1 file is readable as UTF-8); the docstring claim that text defaults to UTF-8 holds.

**The unpickler.** `method='robust'`, `die=True` (raises `UnpicklingError`), and `verbose=True` all behave as documented. The global `unpickling_errors` dict is correctly cleared between loads in every sequence tested, including a `die=True` failure followed by a clean load (no stale warnings, no stale `unpickling_errors` attribute on the next object). `repr()` of a failed object works and appends the failure summary; `showfailure()` and `disp()` work. `auto_remap` was checked for wrong-class risk: `known_fixes` comes from `pd.compat.pickle_compat._class_locations_map`, whose six entries are all `('module','name')` tuple keys mapping to `('module','name')` tuples, and `_remap_module()` only matches a bare name via the *module*-only key form, so there is no bare-class-name matching across modules and no path by which auto-remapping silently substitutes a same-named class from a different module.

**Spreadsheets.** `savespreadsheet()` puts the right data under the right sheet name for a list of arrays plus `sheetnames`, and for `dict`/`sc.odict` input keyed by sheet name -- verified by reading each sheet back and comparing values. `close=False` returns the live `xlsxwriter.Workbook` and writes nothing until the caller closes it; `close=True` returns the path. The first four docstring examples (plain array, array with a header row, list plus `sheetnames`, `sc.odict` input) all run as written, as do all three `writecells()` examples and `Spreadsheet.save()`; only the fifth `savespreadsheet` example is broken (finding 33). One documented-but-silent behaviour that is arguably intentional: with `dict` data *and* `sheetnames` supplied, the lengths are checked but the supplied names are then ignored in favour of the dict keys (line 2296, with the comment "keep original sheet names"). Twenty consecutive mid-write failures (a `formatdata` entry naming an unknown format) leak no file descriptors and no temp files, and leave a pre-existing `.xlsx` byte-identical, so `savespreadsheet` is safe against the truncation problem `savejson` has. `loadspreadsheet()` selects sheets correctly by name and by index, handles an empty sheet (empty DataFrame, no exception), handles a `sheet=[...]` list (dict of DataFrames keyed by name), and handles blank leading rows in the way `header=` dictates. `Spreadsheet.writecells()` was verified for all three documented forms (`'A1'` labels, row/column pairs, and `startrow`/`startcol` with a 2-D array), and `_getsheet(sheetname=...)` selects the right sheet. `Blobject`: bytes round-trip exactly (all 256 byte values) through `blob=`, a filename, a `Path`, a `io.BytesIO` source, `save()`/`load()` and `tofile()`. `load()` flushes and seeks to 0 before reading, so no unrewound-stream truncation was observed; `tofile()` returns a fresh `BytesIO` positioned at 0 on each call (reading twice from the *same* returned handle gives `b''` on the second read, which is ordinary file semantics, not a bug); `tofile(output=False)` followed by `load()` consumes `self.bytes` and resets it to `None`, and a second bare `load()` prints "Nothing to load" and leaves `blob` intact rather than clearing it. Fifty loads from file leak no file descriptors, and `save()` leaves no temp files. `jsonpickle()`/`jsonunpickle()`: the docstring example round-trips values correctly both via an object and via a file; the class is downgraded from `sc.dataframe` to `pd.DataFrame` and a pandas `StringDtype` column comes back as `object`, but the docstring explicitly warns that exact restoration is not guaranteed and calls out mixed-dtype frames, so this is documented rather than defective. `jsonunpickle()` correctly distinguishes a JSON string (leading `[`/`{`) from a filename and raises on a missing file or on both arguments being supplied.

**Other file functions.** `savetext`/`loadtext` round-trip exactly for a multi-line string, a list of strings (joined with `\n`), non-ASCII text (`café 日本語`, via the new UTF-8 default), the empty string, and a string with no trailing newline; `splitlines=True` and the numpy-array shortcut (`fmt='%s', delimiter=', '`) behave as documented, and an arbitrary object is stringified as promised. `sc.path()` correctly drops `None` entries and flattens list arguments. `sc.loadany()` picks the right loader by extension for `.obj`/`.json`/`.txt`/`.csv`, recovers when the extension lies (a pickle named `.json` still loads as a dict), and falls back to text for an unknown extension. `getfilelist()` handles `nopath`, `fnmatch` (applied after `nopath`, so it matches the basename), `filesonly`, `foldersonly`, `recursive`, and the blank-entry filter added in 3.2.1; `filesonly=True, foldersonly=True` silently prefers `filesonly` (an `elif`), which is a benign ambiguity rather than a defect. `rmpath()` removes files and folders correctly, reports missing paths under `die=False` without raising, and returns `None` harmlessly for `rmpath(None)`.

## Rejected on review

- **8.** An invalid `compression` value truncates the target file and leaks the file handle: NOT WORTH FIXING. It only happens with an invalid argument that raises `ValueError` immediately, the single leaked descriptor is closed by refcounting, and the truncation disappears with the fix for finding 1.
- **10.** `savezip(basename=True)` silently writes duplicate zip entries: NOT A BUG. `basename=True` explicitly asks for flattened names, and `zipfile` already warns `Duplicate name: 'dup.txt'`, so the loss is not silent.
- **11.** `sanitizefilename(asciify=True)` maps distinct non-Latin names onto the same output: NOT WORTH FIXING. `asciify=True` is documented and does what it says, `asciify=False` exists for non-Latin names, and collisions are inherent to any sanitizer.
- **14.** `strkeys=True` merges an integer key and its string form: NOT WORTH FIXING. A dict containing both `1` and `'1'` is a contrived input that JSON cannot represent anyway.
- **17.** NaN becomes `null` and +/-inf are written as `Infinity`: NOT A BUG. NaN -> `None` is a deliberate, commented design choice, `Infinity` is the stdlib `json` default, and `allow_nan=False` can already be passed through `savejson(**kwargs)`/`jsonify(tostring=True, **kwargs)`.
- **20.** `writecells()` is 1-based but `readcells()` is 0-based: NOT WORTH FIXING. Each method is self-consistent (`readcells` is 0-based like numpy/pandas, which `tests/test_fileio.py:57-62` relies on), and changing either convention would silently shift every existing caller's reads; at most, document it.
- **29.** `sanitizefilename()` passes through `..`, `.`, reserved Windows names and empty strings: NOT WORTH FIXING. These are corner-case inputs, and the behaviour is not exploitable via `makefilepath()`, which takes the basename first.
- **35.** `Failed.isempty()` almost never returns True, and `Failed[...]` raises: NOT WORTH FIXING. `Failed` is documented as "Not for use by the user", and `isempty()` is not called anywhere in the package.

## Suggested order of work

1. **Findings 1 and 2** -- fix the truncate-before-serialise pattern in `sc.save()` (serialize in memory, then open and write) and drop `sc.load()`'s forgiving fallback chain (or at least make it respect `die=True`). Together these two turn a crash into silent, unrecoverable data loss, and both are contained, well-understood fixes.
2. **Findings 19, 38, 16, 39, 18, 15** -- apply the same serialize-first fix to `savejson()` (19), sanitize `to_dict()`/`to_json()` output (38), then close the `Decimal`/complex/big-int handling (16, 39), the threading bug (18) and the `custom=` subclass lookup (15); all sit in `jsonify()`/`savejson()` and are cheap to fix together.
3. **Findings 3, 4, 5, 9, 12, 22** -- the path-flattening/aliasing family: `makefilepath(folder=...)` (12) is the root cause of `unzip()`'s wrong paths (3) and should be fixed first; then the `sheetnum`/`sheetname` copy-paste bugs (4 and 22, which should be fixed together), the header default (5), and `unzip(members=...)` (9).
4. **Findings 6, 7, 36, 37** -- `loadstr`/`load` on file objects (6, 36, which share one fix), `save(filename=None)` with zstd/none (7), and `die=True` being ignored with `remapping` (37).
5. **Findings 13, 21, 23-28, 30-34, 40-43** -- lower-frequency correctness issues, docstring/pragma cleanup, and the `Blobject`/`aspath`/`thisdir`/`rmpath` edge cases.

Findings that silently return or persist *wrong* data, rather than raising, deserve the most attention regardless of severity tier: the truncations in findings 1 and 19 (destroy the file), finding 2 (`load()` returning `''`/raw text), finding 5 (`loadspreadsheet()`'s consumed header row), finding 4 (`readcells()`'s ignored `sheetnum`), finding 39 (complex values truncated to their real part), finding 37 (`die=True` returning a placeholder for the whole object), finding 40 (`Blobject.save()` writing to `./default`), and finding 3 (`unzip()`'s nonexistent returned paths). By contrast, findings 6, 7, 16, 22, 23, 25, 36 and 42 fail loudly (an exception), which is far easier for a caller to notice and guard against even though it is still a bug to fix.
