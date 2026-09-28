# `sc_versioning.py` bug audit

Audit of `sciris/sc_versioning.py` (all 791 lines, all nine exported functions plus the private helpers `pkg_require()` and `_md_to_objdict()`) for genuine defects: wrong results, documented arguments that do nothing, silently wrong provenance data, dead error-recovery paths, and resource leaks. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps as such, and performance.

**Method**: line-by-line reading, then one executed hypothesis test per suspicion. `compareversions()` was checked against `packaging.version`/`packaging.specifiers` as an oracle over a 729-case grid (9 operators x 9 versions, including `dev`, `rc`, `post` and differing segment counts). `gitinfo()` was checked against real throwaway repos (nested repos, standalone repo, detached HEAD, dirty tree, packed and loose objects). Every docstring example in the module was run verbatim. Reachability of `# pragma: no cover` lines was established with a `sys.settrace` line tracer against the real module file. Every "actual" value below was produced by running the editable install (Sciris 3.3.0, packaging 26.3, pandas 3.0.5, gitpython 3.1.61, Python 3.13.9, commit `2d69aad`), and **every finding was re-run a second time from a minimal snippet in a fresh interpreter** before being recorded.

**Independent re-verification.** This document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`): every original finding was re-run from a minimal snippet with `SCIRIS_BACKEND=agg`. Of the original 16 findings, 7 were confirmed (1, 2, 3, 5, 6, 8, 9; some with corrected severity, side claims or fixes, noted inline), and 9 were rejected as not a bug or not worth fixing (4, 7, 10, 11, 12, 13, 14, 15, 16; see [Rejected on review](#rejected-on-review)). Two missed bugs were added as findings 17 and 18. Line numbers below are for `d91898a`.

**Nothing in this document has been applied.** All fixes are described, not made.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `loadarchive()` | The remapping-recovery path references the non-existent `sc.known_fixes`, so every failed load raises `AttributeError` instead of the real unpickling error type | 771 |
| 2 | High | `metadata()` | Reads the wrong stack frame: `calling_info` is `N/A` for a direct call, and `git_info` then silently reports whatever repo the *working directory* is in | 451, 471 |
| 3 | High | `gitinfo()` | `os.path.dirname()` skips the folder it was given, so a directory argument (including the default `sc.gitinfo()`) returns the *parent* repo's branch and hash | 208 |
| 5 | Medium | `compareversions()` | `~=` is mapped to "not equal", inverting PEP 440's compatible-release operator | 312 |
| 6 | Medium | `savearchive()` | The `user` argument is never forwarded to `metadata()`, so `user=False` still records the username | 664 |
| 17 | Medium | `loadmetadata()` | SVG metadata written by `sc.savefig()` can never be read back: always raises `JSONDecodeError` | 576-584 |
| 8 | Low | `savearchive()` | Passes `frame=3` to `metadata()`, which has no such parameter, so every archive since v3.0.0 carries a bogus `frame: 3` metadata field | 665 |
| 9 | Low | `loadmetadata()` | Returns a plain `dict` for PNG but an `objdict` for JSON and ZIP, so `md.versions.sciris` raises for the primary documented use case | 544 |
| 18 | Low | `loadmetadata()` | Uses `PIL.Image` after only `import PIL`; works only because matplotlib happened to import `PIL.Image` first | 531-535 |

Totals: 3 High, 3 Medium, 3 Low (9 findings). 9 original findings rejected on review.

## Recurring patterns

**Stack-frame arithmetic in `metadata()` is off by one.** `getcaller()` deliberately defaults to `frame=2`, meaning "the caller of the function that calls `getcaller()`", which is exactly what `metadata()` needs. But `metadata()` calls `getcaller(relframe=relframe+1)`, and `sc.savefig()` adds another `sc.metadata(relframe=relframe+1)`. The net effect (finding 2, and the identical `sc_plotting_bugfixes.md` #17) is that the headline provenance feature records nothing for the ordinary case of a top-level script, and `gitinfo('N/A')` then turns that "nothing" into confidently wrong data. The fix belongs in `metadata()` (use `relframe=relframe`) and `sc.savefig()` (drop the `+1`); `getcaller()`'s default must **not** be changed, since that would silently break every existing user of `getcaller()`.

**`savearchive()` calls `metadata()` against a signature that no longer exists.** `sc_plotting.py:1405` records that in v3.0.0 "'frame' [was] replaced with 'relframe'", but `savearchive()` still passes `frame=3` (finding 8), and it never forwarded `user` at all (finding 6). Because `metadata()` accepts `**kwargs` as arbitrary metadata to store, both mistakes are silent: one becomes a junk field, the other a privacy leak. Any keyword `savearchive()` claims to forward should be forwarded explicitly and `**kwargs` should not be the landing pad for typos.

**Error handlers that destroy the error they are handling.** `loadarchive()` computes `exc = type(E)` from the *innermost* exception and re-raises with a generic message, so the exception type the caller sees is the internal `AttributeError` rather than the real `sc.UnpicklingError` (finding 1). This is the highest-value area to fix, because these are the paths users hit precisely when they most need diagnostic information.

**The `loadmetadata()` readers disagree with the `savefig()` writer.** PNG results are not converted to `objdict` (finding 9), the SVG reader assumes a single-line payload that `savefig()` never writes (finding 17), and the bitmap branch relies on an implicit `PIL.Image` import (finding 18). None of the non-JSON/ZIP branches are covered by tests.

## High severity

### 1. `loadarchive()`'s recovery path raises `AttributeError: module 'sciris' has no attribute 'known_fixes'` — `sc_versioning.py:771`

`sc.known_fixes` does not exist. `known_fixes` is a module-level name in `sc_fileio.py:2370` but is never added to any `__all__`, so it is not re-exported into the `sciris` namespace. The consequence is that the entire second-chance "load with all remappings" block — the documented reason `savearchive()`/`loadarchive()` exist ("While there may still be issues opening the pickle, the metadata... should give enough information to figure out how to reconstruct the original environment") — dies on its first statement, and the `exc = type(E)` on line 774 then uses the *internal* `AttributeError` as the raised exception type.

```python
import sciris as sc, zipfile
sc.savearchive('good.zip', dict(a=1))
with zipfile.ZipFile('good.zip') as z: md = z.read('sciris_metadata.json')
with zipfile.ZipFile('corrupt.zip','w') as z:          # a valid archive with a damaged payload
    z.writestr('sciris_metadata.json', md); z.writestr('sciris_data.obj', b'not a pickle')
sc.loadarchive('corrupt.zip')
```

Actual:

```
AttributeError: Could not unpickle the object: to debug using metadata, set die=False
  __cause__: AttributeError("module 'sciris' has no attribute 'known_fixes'")
>>> hasattr(sc, 'known_fixes')
False
```

Expected: the retry actually runs, and if it still fails the raised exception is the real unpickling error (an `sc.UnpicklingError` describing the bad pickle), not an `AttributeError` about a missing Sciris attribute.

Correction on review: the original unpickling error is not entirely discarded — the printed traceback still shows the original `sciris.sc_fileio.UnpicklingError` as implicit context ("During handling of the above exception, another exception occurred"). What is actually lost is the exception *type*: callers that `except sc.UnpicklingError:` get an `AttributeError` instead and are not caught.

Blast radius: the only caller of this block is `loadarchive()` itself, but `loadarchive()` is the documented long-term-storage loader and is reached from `sc.loadmetadata()` for `.zip` inputs. This is a regression introduced by commit `46a3890` ("updated from relative to absolute imports", 2024-09-23, first released in v3.2.0), which rewrote a working `scf.known_fixes` (i.e. `sc_fileio.known_fixes`) as `sc.known_fixes`; every release from v3.2.0 onward is affected. `tests/test_versioning.py::test_regressions` loads `tests/files/archive_2023-04-19.zip`, which succeeds on the first attempt, so the branch is never exercised (it is marked `# pragma: no cover` at 769).

**Fix**: `sc.sc_fileio.known_fixes` would work, but it is redundant: `sc.loadstr()` already applies `known_fixes` by default (`_RobustUnpickler(..., auto_remap=True)`, `sc_fileio.py:2587`), so the retry's only real extra is `method=None` (try every method). The simplest correct fix is therefore to drop the `sc.mergedicts(remapping, sc.known_fixes)` line, make the handler `except Exception as E1:` rather than a bare `except:`, and on final failure re-raise chained from the *first* exception (`E1`) using its type. At the same time, move the `require(reqs=reqs, die=False, warn=True)` call out of the `try` block (after the object has loaded), so a warnings-as-errors failure in the requirements check cannot trigger a spurious retry (this was original finding 7; it costs nothing to fold in here). While in the area, note that `sc_fileio.py:2371` reads `except (NameError or AttributeError):`, which evaluates to `except NameError:` and would not catch a missing pandas compat map; that belongs to `sc_fileio`.

### 2. `sc.metadata()` records no caller and the wrong repository — `sc_versioning.py:451`, `471`

This is the same bug as `sc_plotting_bugfixes.md` #17 (seen there through `sc.savefig()`); fix them together.

`calling_info = dict_fn(getcaller(relframe=relframe+1, tostring=False))` lands one frame too deep. For a direct call the stack is `getcaller`=0, `metadata`=1, user=2; `getcaller()`'s default `frame=2` already targets the user, but `relframe+1` makes it ask for 2+0+1=3. At module level frame 3 does not exist, so `getcaller()` swallows the `IndexError` and returns `N/A`; line 471 then calls `gitinfo('N/A')`, which resolves the nonexistent relative path `'N/A'` against the current working directory and returns whatever repository *that* happens to be in. The `relframe` docstring is explicit that `0` is the value for direct use.

```python
# script.py, run as `python script.py`
import sciris as sc
print(sc.metadata(pipfreeze=False).calling_info)
print(sc.metadata(pipfreeze=False, relframe=-1).calling_info)   # what relframe=0 should give
```

Actual vs expected:

```
{'filename': 'N/A', 'lineno': 'N/A'}                                  <- actual for relframe=0
{'filename': '/.../script.py', 'lineno': 3}                           <- what relframe=-1 gives
```

Inside a function the wrong frame is partly masked (you get the caller's caller: the same file, but the line number of the call *to the wrapper*, not to `sc.metadata()`), which is why `tests/test_versioning.py::test_metadata` does not notice: it calls `sc.metadata()` from inside a test function, so frame 3 exists.

The user-visible damage is the git provenance. Running one unchanged script from two different working directories, inside two different repositories:

```python
# /tmp/gitdemo/outer/inner/figscript.py   (inner/ is its own repo, branch "innerbranch")
import matplotlib; matplotlib.use('agg'); import matplotlib.pyplot as plt, sciris as sc
plt.plot([1,3,7]); sc.savefig('fig_meta.png')
md = sc.loadmetadata('fig_meta.png')
print(md['calling_info'], md['git_info'], sc.gitinfo(__file__))
```

```
$ cd /tmp/gitdemo/outer/inner && python figscript.py
calling_info {'filename': 'N/A', 'lineno': 'N/A'}
git_info     {'branch': 'innerbranch', 'hash': '845ab06', ...}
gitinfo(__file__) {'branch': 'innerbranch', 'hash': '845ab06', ...}

$ cd /tmp/gitdemo/outer && python inner/figscript.py
calling_info {'filename': 'N/A', 'lineno': 'N/A'}
git_info     {'branch': 'outerbranch', 'hash': '9027e97', ...}      <- WRONG repo and commit
gitinfo(__file__) {'branch': 'innerbranch', 'hash': '845ab06', ...}
```

The recorded commit hash depends on the shell's `cd`, not on the code that made the figure, and there is no indication anything went wrong. Expected: `git_info` equal to `sc.gitinfo(__file__)` in both runs.

Blast radius: `sc.savefig()` (`sc_plotting.py:1422`) adds a second `+1` (`sc.metadata(relframe=relframe+1, ...)`), asking for frame 2+1+1=4 when the user is at 3, so it is off by one for exactly the same reason and its docstring promise "relframe (int): ... default 0, the file calling `sc.savefig()`" is not met. `sc.savearchive()` asks for frame 3, which is correct only by accident: its stray `frame=3` (finding 8) is ignored and its extra stack level happens to cancel the error out. `sc.metadata()` is also the public entry point advertised in the module docstring and in `CLAUDE.md`.

**Fix**: change only `metadata()`: `getcaller(relframe=relframe, tostring=False)` (with `getcaller()`'s default left at 2, frame 2 is the direct caller of `metadata()`). Correspondingly drop the `+1` in `sc.savefig()` (`sc_plotting.py:1422`), and change `savearchive()` to pass `relframe=1` instead of `frame=3` (see finding 8). **Do not change `getcaller()`'s default** to `frame=1`, as the original version of this audit suggested: "get caller" deliberately means the caller of the function that calls `getcaller()`, and changing the default would silently break every existing user of `getcaller()`. Optionally, as belt-and-braces, `metadata()` can skip the `gitinfo()` call when `calling_info['filename'] == 'N/A'`.

### 3. `gitinfo(folder)` looks in the folder's parent, so nested repos report the wrong one — `sc_versioning.py:208`

`curpath = os.path.dirname(os.path.abspath(path))` unconditionally strips one path component. That is right for the `sc.gitinfo(my_package.__file__)` form, but for the directory form the `.git` inside `path` is never examined — contradicting the docstring's "path (str): A folder either containing a .git directory, or with a parent that contains a .git directory" and the example `info = sc.gitinfo()` with the default `path = os.getcwd()`.

```python
# /tmp/gitdemo/outer      is a repo on branch "outerbranch"
# /tmp/gitdemo/outer/inner is a separate repo on branch "innerbranch"
sc.gitinfo('/tmp/gitdemo/outer/inner')
```

Actual vs expected:

```
actual:   {'branch': 'outerbranch', 'hash': '9027e97', 'date': '2026-09-08 23:46:57 UTC'}
expected: {'branch': 'innerbranch', 'hash': '845ab06', ...}     ($ git -C .../inner rev-parse --abbrev-ref HEAD  ->  innerbranch)
sc.gitinfo('/tmp/gitdemo/outer/inner/somefile.py')  ->  {'branch': 'innerbranch', ...}   # the file form is correct
```

The most important real-world trigger is the default call: `sc.gitinfo()` run from a repo root whose parent is also a repo (for example, a project checked out under a dotfiles repo in `$HOME`). Re-verified: with the cwd set to `outer/inner`, `sc.gitinfo()` returns `{'branch': 'outerbranch', ...}`.

Correction on review: the original audit claimed that a trailing separator (`sc.gitinfo(path + os.sep)`) accidentally works. That is false: `sc.gitinfo('.../outer/inner/')` still returns `outerbranch`, because `os.path.abspath()` strips the trailing slash before `dirname()` runs.

When the directory *is* the repo root and no ancestor is a repo, the direct read fails outright and only the gitpython fallback saves it:

```python
# gitpython import blocked, /tmp/solo/repo is a repo whose parents are not
sc.gitinfo('/tmp/solo/repo', verbose=True)
```

```
Could not extract git info; please check paths:
  Method 1 (direct read) error: Could not find .git directory
  Method 2 (gitpython) error:   blocked for test
{'branch': 'Branch N/A', 'hash': 'Hash N/A', 'date': 'Date N/A'}
```

This is also why the default `sc.gitinfo()` call takes a different code path depending on where you run it: from `/home/cliffk/sc/sciris` (the repo root, whose parent `/home/cliffk/sc` has no `.git`) it falls through to gitpython; from `/home/cliffk/sc/sciris/tests` the direct read succeeds. The two paths format the date differently (`... UTC` versus ISO 8601 with offset; see rejected item 13).

Blast radius: `metadata()` at line 471 always passes a filename, so it is unaffected by this particular defect; the exposure is the public `sc.gitinfo()` and `sc.gitinfo(some_dir)`. `tests/test_versioning.py:36` calls `sc.gitinfo()` from `tests/`, where the bug is invisible.

**Fix**: only strip a component when the path is not a directory, e.g. `p = os.path.abspath(path); curpath = p if os.path.isdir(p) else os.path.dirname(p)`. This does not change the file-path form or the behaviour of `gitinfo('N/A')`.

## Medium severity

### 5. `compareversions()` treats `~=` as "not equal" — `sc_versioning.py:312`

Line 312 (`elif v2.startswith('~='): valid = [-1,1]`) assigns `~=` exactly the same `valid` list as `!=` on the next line. `~=` is PEP 440's compatible-release operator (`~=1.2.3` means `>=1.2.3, ==1.2.*`), so the result is close to the logical inverse of the intended meaning, and in particular equal versions are reported as *not* satisfying the constraint.

```python
sc.compareversions('1.2.3', '~=1.2.3')   # actual: False   packaging: True
sc.compareversions('2.0.0', '~=1.2.3')   # actual: True    packaging: False
sc.compareversions('1.2.3', '~=1.2.3') == sc.compareversions('1.2.3', '!=1.2.3')   # True: identical to !=
```

A full 9x9 grid against `packaging.specifiers.SpecifierSet(..., prereleases=True)` gives 63 mismatches, all 63 of them `~=` cases. Every other operator form (`>`, `>=`, `<`, `<=`, `==`, `!=`, `=`, `!`) matched the oracle except for two PEP 440 spec subtleties that are not bugs in an ordering comparator (see Verified clean).

Blast radius: `sc.compareversions()` is used with `~=` nowhere inside Sciris (grep: the only `~=` occurrences are this branch and `tests/test_versioning.py:51`), but it is a public, widely used API in downstream packages. Note that `tests/test_versioning.py:51` asserts `sc.compareversions('1.2.3', '~=1.2.9')` is `True`, i.e. the test currently locks in the wrong behaviour and would need updating.

**Fix**: either implement compatible-release properly (delegate to `pkgs.SpecifierSet(v2, prereleases=True).contains(v1)` when `v2` starts with `~=`), or, if only the -1/0/1 machinery is wanted, raise the same "not supported" `ValueError` that bare `~` raises rather than silently aliasing `!=`. Either way, fix the bare-`~` error message at line 319 at the same time: it has a typo ("for not" should be "for now") and recommends `~=` (original finding 15, folded in here).

### 6. `savearchive(user=False)` still records the username — `sc_versioning.py:664`

`savearchive()` accepts `user=True` in its signature and its docstring documents it (as the first of the two duplicated `caller (bool)` entries: "store information on the current user in the metadata"), but the `metadata()` call on line 664 passes `caller`, `git`, `pipfreeze`, `comments` and `require` and never passes `user`, so `metadata()`'s own default `user=True` always wins.

```python
sc.savearchive('u1.zip', dict(a=1), user=False)
sc.loadmetadata('u1.zip').get('user')
```

Actual vs expected:

```
actual:   'cliffk'          # the real username, despite user=False
expected: None              # sc.metadata(user=False).user is None -- the control works
```

Blast radius: `savearchive()` only; but this is the argument someone reaches for before sharing an archive, so the failure mode is a privacy leak into a file that is meant to be distributed. Untested.

**Fix**: add `user=user` to the `metadata()` call, and fix the duplicated `caller (bool)` docstring line to document `user` (and the undocumented `folder`).

### 17. `loadmetadata()` cannot read SVG metadata written by `sc.savefig()`: always raises `JSONDecodeError` — `sc_versioning.py:576-584`

`sc.savefig('x.svg')` stores `metadata(..., tostring=True)`, which is `sc.jsonify(md, tostring=True, indent=2)` (line 485), so the JSON spans many lines inside `<rdf:li>sciris_metadata={ ... }</rdf:li>`. `loadmetadata()` scans the SVG line by line and slices only the single line containing the flag: `line[line.find(flag)+len(flag):line.find(end)]`. That line is just `sciris_metadata={`, and `find('</')` returns `-1`, so the slice is the empty string. The SVG branch, which the docstring advertises ("currently only PNG and SVG are supported"), therefore never works. It is marked `# pragma: no cover` and there is no SVG test.

```python
import sciris as sc, matplotlib.pyplot as plt
plt.plot([1,2]); sc.savefig('plain.svg')
sc.loadmetadata('plain.svg')
```

Actual vs expected:

```
actual:   JSONDecodeError('Expecting value: line 1 column 1 (char 0)')    # even for a figure saved with no comments
expected: the metadata dict, as for PNG
```

A second, independent problem: matplotlib XML-escapes the payload (`<`, `&`, `"` in `comments`), so even a correctly sliced string needs `html.unescape()` before JSON parsing, or the comments come back as `&lt;b&gt;...`.

Blast radius: `loadmetadata()` on any `.svg` file; `sc.savefig()` itself writes the metadata correctly.

**Fix**: search the whole text rather than one line, and unescape before parsing (verified by the reviewer to round-trip both a plain figure and one with `comments='hello <b>&"x"</b>'`):

```python
import html
txt = sc.loadtext(filename)
flag = _metadataflag + '='
start = txt.find(flag)
if start >= 0:
    start += len(flag)
    md = sc.loadjson(string=html.unescape(txt[start:txt.find('</', start)]))
else:
    ... # existing "Can't find metadata" branch
```

Ideally wrap the result in `_md_to_objdict()` for consistency with finding 9.

## Low severity

### 8. `savearchive()` passes `frame=3` to a function that has no `frame` parameter — `sc_versioning.py:665`

Severity lowered from Medium to Low on review: the stored field is harmless junk, though the call is plainly stale.

`metadata()`'s frame argument was renamed in v3.0.0 (`sc_plotting.py:1405`: "'frame' replaced with 'relframe'"), but `savearchive()` was not updated. Because `metadata()` ends with `**kwargs` meaning "any additional data to store", the stale argument is not an error: it is silently written into the metadata as a data field.

```python
sc.savearchive('u1.zip', dict(a=1))
list(sc.loadmetadata('u1.zip').keys())
```

Actual vs expected:

```
actual:   [..., 'comments', 'frame', 'method']       and md['frame'] == 3
expected: [..., 'comments', 'method']
```

This is not new: `tests/files/archive_2023-04-19.zip`, saved with Sciris 3.0.0, already contains `'frame': 3`, so every archive written by every 3.x release carries the junk field. The intended argument was presumably `relframe=1`; as noted in finding 2 the caller information happens to come out right anyway, so fixing this on its own must not change the effective frame.

**Fix**: delete `frame=3` from the `metadata()` call. When `metadata()`'s `relframe` handling is corrected per finding 2, `savearchive()` needs `relframe=1` in its place to keep pointing at the user's file.

### 9. `loadmetadata()` returns a plain `dict` for PNG but an `objdict` for JSON and ZIP — `sc_versioning.py:544`

Severity lowered from Medium to Low on review.

The JSON branch (596) and the ZIP branch (600, via `loadarchive()`) both run `_md_to_objdict()`, but the PNG branch returns `sc.loadjson(string=jsonstr)` unconverted. PNG is the primary documented use of this function ("Read metadata from a saved image"; the docstring example is `sc.savefig('example.png'); sc.loadmetadata('example.png')`), and Sciris' own test uses attribute access on the other two branches (`tests/test_versioning.py:82`: `md2.system.platform == md3.system.platform`).

```python
for fn in ['fig.png', 'md.json', 'arch.zip']:
    md = sc.loadmetadata(fn)
    print(type(md).__name__, md.versions.sciris)
```

Actual vs expected:

```
fig.png    -> dict      md.versions.sciris raises AttributeError: 'dict' object has no attribute 'versions'
md.json    -> objdict   md.versions.sciris = 3.3.0
arch.zip   -> objdict   md.versions.sciris = 3.3.0
expected: objdict in all three cases
```

Blast radius: `loadmetadata()` only; the PNG branch is the one `sc_plotting.savefig()`'s docstring points users to.

**Fix**: wrap the PNG (and JPG) results in `_md_to_objdict()` as the JSON branch does (and the SVG result, per finding 17).

### 18. `loadmetadata()` uses `PIL.Image` after only `import PIL` — `sc_versioning.py:531-535`

`import PIL` does not import the `Image` submodule: `python -c "import PIL; print(hasattr(PIL,'Image'))"` prints `False`. Line 535 (`im = PIL.Image.open(filename)`) only works today because the normal `import sciris` eagerly imports matplotlib, which imports `PIL.Image` as a side effect. The bug is latent in the default import path.

```python
# SCIRIS_LAZY=1
from sciris import sc_versioning
sc_versioning.loadmetadata('fig.png')
```

Actual vs expected:

```
actual:   AttributeError: module 'PIL' has no attribute 'Image'
expected: the metadata dict
```

Blast radius: `loadmetadata()` on PNG/JPG files, only when matplotlib has not already been imported (e.g. lazy import).

**Fix**: `import PIL.Image` instead of `import PIL` (keeping the existing `ImportError` handler). This also removes the dependency on matplotlib's import side effects.

## Rejected on review

These original findings were removed from the list above after independent re-verification on 2026-09-25. Numbers are kept so cross-references remain valid.

- **4. `gitinfo()` does not validate the path, so a bogus path returns the cwd's repo** — NOT A BUG. A nonexistent relative path is still a location inside the cwd, so the cwd's repo is the right answer (and REPL callers `<stdin>`/`<string>` rely on it); the proposed `FileNotFoundError` would regress them, and the only harmful case (`gitinfo('N/A')`) disappears once #2 is fixed.
- **7. `loadarchive()`'s bare `except:` also catches the `require()` warning** — NOT WORTH FIXING as a separate item. `require()` catches every exception internally, so this only triggers under warnings-as-errors with a stored `require` field; moving `require()` out of the `try` is free and has been folded into the #1 fix.
- **10. `loadmetadata()` never closes the image it opens** — NOT WORTH FIXING. Only `im.info` escapes the function, so under CPython the handle is closed at function return; the claimed "indeterminate time" Windows lock is overstated.
- **11. `getcaller()`'s default frame is off by one relative to its own docstring** — NOT A BUG. `getcaller()` deliberately returns the caller of the function that calls it (the documented "Frame 2 is the default assuming it is being called directly"); changing the default to `frame=1` would break every existing user, and `metadata()` (#2) is what misuses it.
- **12. `freeze(lower=True)` returns unsorted output** — NOT WORTH FIXING. Only dict key order is affected; lookups are unchanged and nothing depends on the order.
- **13. `gitinfo()` returns two different date formats for the same commit** — NOT WORTH FIXING. Both are correct representations of the same instant; this is a consistency nicety.
- **14. `require()` computes `count` and never uses it** — NOT WORTH FIXING. Unused variable (style only).
- **15. `compareversions()`'s `~` error message has a typo and points at the broken operator** — NOT WORTH FIXING separately. Error-message wording only; fold it into the #5 fix.
- **16. `loadmetadata()`'s unsupported-format message omits the formats it supports** — NOT WORTH FIXING. Error-message and docstring wording only, with no wrong result.

## Misplaced `# pragma: no cover`

All rows below were demonstrated by running a `sys.settrace` line tracer against `/home/cliffk/sc/sciris/sciris/sc_versioning.py` and recording executed line numbers; each "executed lines" entry is a line that ran during ordinary calls.

| Line | Marked construct | Executed lines observed | Why it is reachable |
|---|---|---|---|
| 55 | `if lower: # pragma: no cover` in `freeze()` | 55, 56 | Line 55 runs on *every* `sc.freeze()` call, and 56 runs whenever the documented `lower=True` is used |
| 213 | `else:` (walk to parent directory) in `gitinfo()` | 214, 215, 216, 218 | Runs for any path whose `.git` is more than one level up, e.g. `sc.gitinfo('/repo/a/b/c/file.py')` |
| 219 | `else:` on the `while` (raise "Could not find .git directory") | 220 | Runs for `sc.gitinfo(repo_root_dir)` and for `sc.gitinfo()` executed from a repo root (finding 3) |
| 246 | `except Exception as E:` (whole gitpython fallback) in `gitinfo()` | 246-263 | Runs for the plain default `sc.gitinfo()` whenever the cwd is a repo root: verified from `/home/cliffk/sc/sciris` |
| 318 | `elif v2.startswith('~'):` in `compareversions()` | 318, 319, 320 | Already covered by `tests/test_versioning.py:53-54` (`pytest.raises(ValueError)`) |
| 381 | `if includelineno:` in `getcaller()` | 381, 382 | Documented argument; runs on `sc.getcaller(1, True, True)` |
| 385 | `if includeline:` in `getcaller()` | 385-390 | Documented argument used in the function's own fourth docstring example |
| 658 | `if not allow_nonzip:` in `savearchive()` | 658, 659 | `allow_nonzip=False` is the default, so both lines run on every ordinary `sc.savearchive()` call |

## Verified clean

**`compareversions()`** — Checked against `packaging.specifiers.SpecifierSet(..., prereleases=True)` over all 729 combinations of 9 operator forms (`<=`, `>=`, `==`, `~=`, `!=`, `<`, `>`, `=`, `!`) and 9 version strings (`1.2.3`, `1.2`, `1.2.0`, `2.0`, `1.2.3.dev0`, `1.2.3rc1`, `1.2.4`, `1.3.0`, `1.2.3.post1`). Apart from the 63 `~=` cases (finding 5), the only differences were two deliberate PEP 440 specifier subtleties that are *correct* for an ordering comparator and are therefore not bugs: `'1.2.3.dev0' < '1.2.3'` and `'1.2.3.post1' > '1.2.3'` are true as orderings, whereas the `<V`/`>V` specifiers exclude pre-/post-releases of `V` itself. Differing segment counts are handled correctly (`1.2` == `1.2.0`, `2` == `2.0.0.0`, both matching `packaging.version` ordering); prerelease ordering (`dev` < `rc` < release < `post`) matches the oracle; all four numeric docstring examples return exactly the claimed `-1`/`0`/`1`/`True`; the two-character operator prefixes are tested before the one-character ones so `>=` is never mis-parsed as `>`; the module-alias form (`sc.compareversions(np, '>=1.0')`) works, and `int`/`float` version1 values (`sc.compareversions(2, '2')`) work. `v2.lstrip('<>=!~')` was probed for over-stripping — no realistic version string starts with those characters, and whitespace after the operator (`'>= 1.2.3'`) is tolerated by `packaging`. Non-numeric segments raise `packaging.version.InvalidVersion`, which is out of contract. `version2` is *not* module-coerced the way `version1` is, so `sc.compareversions(np, np)` raises `InvalidVersion`; this asymmetry is noted but the call is far-fetched enough that it was not written up.

**`require()` and `pkg_require()`** — The v3.3.0 changelog claim was tested as stated by building a synthetic `fakestarsim` 3.2.3.dev0 dist-info on `PYTHONPATH`: `sc.require('fakestarsim>3.0.0')` returns `True`, as does `>=3.0.0` and the dict form `require(fakestarsim='3.0.0')`, and `>4.0.0` correctly returns `False` — the claim holds. Prerelease handling is PEP 440-correct in the cases that look surprising too (`2.0.0rc1` does not satisfy `>=2.0.0`; `3.0.0.dev0` does not satisfy `>=3.0.0`; `4.0.0rc1` does satisfy `>3.0.0`). All five docstring examples run and return the documented outcome. Multiple requirements: the list, `*args`, dict and `**kwargs` forms all merge (dict entries override same-named kwargs), the error message enumerates every failure rather than only the first, `die=True` chains `from` the last error, `die=False, warn=True` warns with `stacklevel=2`, `die=False, warn=False, verbose=True` prints, `die=False, warn=False, verbose=False` is silent, `detailed=True` returns `(data, errs)` with a `False` entry per failed requirement, `exact=True` produces `==` rather than `>=`, a value already starting with a comparator is not double-prefixed, and an empty version string means "any version". `SpecifierSet.__len__` is what makes the `if allowed` guard work for the bare-package case, so `sc.require('numpy')` correctly checks presence only. Distribution-name normalisation is handled by `importlib.metadata`, so `require('PyYAML')`, `require('pyyaml')`, `require('python-dateutil')`, `require('python_dateutil')` and `require('Numpy')` all pass. No global state (warnings filters, `os.environ`, `sc.options`) is left mutated.

**`gitinfo()`** — Detached HEAD is detected and reported correctly (`{'branch': 'Detached head (no branch)', 'hash': 'b727435', ...}`) via the direct-read path, including when the working tree is dirty; a dirty tree simply reports the HEAD commit, which is a design limitation rather than a defect (no dirty/`-dirty` flag is offered or promised). `hashlen` is applied to both the direct-read and gitpython hashes, and the `'N/A' not in githash` guard correctly prevents `"Hash N/A"` from being truncated to `"Hash N"`. `die=True` raises `RuntimeError` chained from the direct-read error; `die=False, verbose=False` is silent. Outside any repository the result is the documented all-`N/A` dict. `.git` files (worktrees/submodules) and packed refs fail the direct read and are picked up by the gitpython fallback.

**`metadata()`** — No global state is aliased or mutated: two successive calls return fully independent objects (mutating `a.versions` does not affect `b`), `freeze()` builds a fresh dict each call, and passing `outfile=` or `tostring=True` does not mutate or downgrade the returned object (`asdict=False` still yields nested `objdict`s afterwards). `asdict=True` yields plain dicts throughout and still works for the `git_info` lookup. `outfile` with a nested, nonexistent directory is created by `sc.makepath(makedirs=True)`. Both docstring examples run. The `require` and `comments` arguments are stored by reference rather than copied, so mutating the *returned* metadata mutates the caller's dict — noted but not reported, since a normal call never mutates the input. `git_info` is still computed when `caller=False`, which is intentional and useful.

**`savearchive()`/`loadarchive()`** — Round trips are faithful: an `objdict` containing an `int64` numpy array (values and dtype preserved), a pandas DataFrame (`.equals()` true), a string, `None` and a float all came back identical, with `method='dill'` (default) and with `method='pickle'`; `remapping=` passes through to `sc.load()`; `files=` adds extra members to the zip without disturbing the two reserved names; `folder=` works for both; `loadmetadata=True` returns `dict(metadata=..., obj=...)`, and the `loadobj=False, loadmetadata=False` combination raises the documented `ValueError`. Error paths leave nothing behind: an object that raises during pickling propagates the real error and creates no partial or temp files (only the target directory, from the documented `makedirs=True`), and the non-`.zip` guard raises before anything is written. The `ZipFile` is opened with `with zf:` so it is closed on every exception path inside `loadarchive()`. Backwards compatibility with an older writer was checked against the shipped `tests/files/archive_2023-04-19.zip` (Sciris 3.0.0, dill): `sc.loadmetadata()` and `sc.loadarchive()` both succeed, and the missing `require` key is handled by `md.get('require', None)` (`sc.objdict.get` honours the default). Dropping `method=` in the retry at line 772 is harmless, because `sc.load(method=None)` means "try all methods".

**`loadmetadata()`** — The JSON and ZIP branches return correctly nested `objdict`s two levels deep via `_md_to_objdict()`; the PNG round trip through `sc.savefig()` recovers the exact metadata payload (the SVG round trip does not; see finding 17); `die=True` on a PNG with no Sciris metadata raises the documented `ValueError`. Extension matching is case-insensitive via `lcfn`.

**`freeze()`** — The default (`lower=False`) output is correctly sorted, and the docstring example (`'numpy' in sc.freeze()`) holds.

**`getcaller()`** — `frame=1` returns the correct file, line number and (with `includeline=True`) the exact source line; `tostring=True` versus `False` return the documented string and dict; `relframe` is additive with `frame` as documented; failures are swallowed and reported as the documented `'N/A'` sentinels rather than propagating, and `die=True` re-raises. The default `frame=2` (the caller of the function calling `getcaller()`) is by design; see rejected item 11. The `includeline` file read uses a `with` block and cannot leak a handle.
