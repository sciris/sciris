# `_extras/legacy.py` bug audit

Audit of `sciris/_extras/legacy.py` (all 571 lines) for genuine defects: dead references to Sciris APIs that no longer exist, unpickling helpers that return the wrong object or a partially-populated one, unclosed streams and hung processes, silent data corruption, and documented workflows that cannot work. Deliberately **out of scope**: style, naming, missing type hints, unhelpful errors on deliberately wrong types, and the fact that the module is old (the file is explicitly "the graveyard for old Sciris functions... preserved for backwards compatibility"). Severity is calibrated to that purpose: the practical job of this file is loading pickles and files written by earlier Sciris versions, so a loader that raises on every call, or that silently recovers only part of an object, is the most serious thing that can be wrong here.

**Method**: line-by-line reading, then an executed test per hypothesis. The Python 2 paths cannot be exercised with a pickle written by Python 3 (`pickle.dumps(..., protocol=2)` still emits `BINUNICODE` for `str`, so the two-pass latin1/bytes unpickler never sees a py2 `str`), so I hand-assembled genuine py2-style protocol-2 pickle streams from raw opcodes (`SHORT_BINSTRING` keys, `GLOBAL` + `TUPLE1` + `REDUCE` datetimes, `NEWOBJ` + `BUILD` instances) and gzipped them, which does drive every branch of `_loadobj2to3()`. Where a finding is masked by an earlier one, the earlier one was shimmed out (a `scf.Empty`/`scf.makefailed` restored from `git show a346ae8^:sciris/sc_fileio.py`) so the later branch could be reached and observed; those cases are labelled. Every "actual" value below was produced by running the code against the editable install (Sciris 3.3.0, numpy 2.4.6, pandas 3.0.5, CPython 3.13.9, multiprocess 0.70.19, dill 0.4.1, commit `2d69aad`), and every finding was reproduced a second time in a fresh interpreter from a minimal snippet before being recorded here. There is no `tests/test_legacy.py`; the module is not imported by `sciris/__init__.py` and nothing in the package imports it (only `_extras/ansicolors.py` is imported, by `sc_printing.py`), so nothing in CI touches any line of it.

**Re-verification**: this document was independently re-verified on 2026-09-25 against commit `d91898a` (CPython 3.13.9, `SCIRIS_BACKEND=agg`), with every repro re-run. Of the original 12 findings, 6 were confirmed as written (3, 4, 5, 6, 7, 9, with severity lowered for 4 and 5), 2 were rewritten as inaccurate (1: the bug is real but is not reached from the public `loadobj2or3()`; 2: the bug is real but the original fix was wrong), and 4 were rejected as not worth fixing (8, 10, 11, 12; see "Rejected on review"). Three new findings were added (13, 14, 15). The most important correction: the public `loadobj2or3()` essentially never reaches the private `_loadobj2to3()` for a readable py2 file, because its fast path `sc.load()` tries `pickle`, `pandas`, `latin`, `dill`, `bytestr` and then (with its default `die=False`) the `robust` unpickler; the `latin` method loads py2 pickles successfully and `robust` swallows top-level errors, so `sc.load()` only raises for things like a missing file. Verified: `loadobj2or3()` on a py2 dict-with-datetime file and a py2 list-of-datetimes file returns correct datetimes. Findings in the private two-pass path (1, 4, 5, 14, 15) are therefore real but only reached by calling `_loadobj2to3()` directly, and are rated accordingly; the fast path itself has its own defect (finding 13).

Import path check: `from sciris._extras import legacy` works and the module imports cleanly (no import-time error, though see finding 2 for what the import does to the process).

**Nothing in this document has been applied.** All fixes are described, not made.

## Summary

11 findings: 2 High, 5 Medium, 4 Low (plus 4 rejected on review).

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 2 | High | `_pickleMethod()` | Importing the module registers a process-global `copyreg` handler that silently corrupts every subsequently written pickle containing a classmethod | 212, 233 |
| 3 | High | `parallelcmd()` | Broken on Python 3.13 (PEP 667): `exec`-assigned loop variables are invisible to the next `exec`, the worker dies, and the parent blocks in `outputqueue.get()` forever | 466, 551, 560, 570 |
| 1 | Medium | `_loadobj2to3()` | References `scf.Empty` and `scf.makefailed`, both removed from `sc_fileio`, so the private two-pass loader raises `AttributeError` inside its own error handler (not reached from the public `loadobj2or3()` for readable files) | 74, 81, 148, 158 |
| 6 | Medium | `_parallelcmd_task()` | Calls `scp.loadbalancer()`, which lives in `sc_parallel`, not `sc_profiling`: any `maxcpu`/`maxmem`/`maxload` call kills the worker (then hangs, per finding 3) | 546 |
| 7 | Medium | `parallel_progress()` | Worker exceptions are swallowed by `apply_async` with no `error_callback`; a failed task returns `None` and looks successful | 523, 526 |
| 9 | Medium | `legacy_dataframe` | The documented `remapping` recovery workflow never fires, because `sciris.sc_dataframe.dataframe` still resolves — to the pandas class, whose `__setstate__` rejects the old state (fix belongs in `sc_fileio`) | 254 |
| 13 | Medium | `loadobj2or3()` | The `sc.load()` fast path succeeds on py2 pickles, so py2 `Blobject`/`Spreadsheet` blobs come back as latin1-decoded `str` and are unusable; the bytes-aware loader is never used | 45-48 |
| 4 | Low | `_loadobj2to3()` | `recursionlimit` counts nodes visited, not recursion depth, so wide objects are only partly substituted: 1000 of 1200 datetimes recovered at an actual depth of 2 (private path only) | 123, 136 |
| 5 | Low | `_loadobj2to3()` | The dict branch calls `setattr()` on the dict instead of assigning an item, so any py2 dict holding a datetime raises `AttributeError` (private path only) | 116 |
| 14 | Low | `_loadobj2to3()` | `recursive_substitute()` never walks lists/tuples, so datetimes inside them stay as `Empty` placeholders (private path only) | 113-138 |
| 15 | Low | `_loadobj2to3()` | An unresolvable class with any nested object attribute aborts the load with `AttributeError` on the `Empty` placeholder (private path only) | 136 |

## Recurring patterns

**Three references to Sciris APIs that were moved, renamed, or deleted (findings 1 and 6).** `scf.Empty` was removed and `scf.makefailed` renamed to `_makefailed` with a different signature (`exc=` instead of `error=`/`exception=`) in commit `a346ae8`; `loadbalancer()` lives in `sc_parallel`, not `sc_profiling`. Each is on the *only* code path that matters for the function containing it, so the function is dead, not merely degraded. Nothing catches this because the module has no test file and is not imported by the package, so neither pytest nor an import-time smoke test ever evaluates these attribute lookups. A single test that calls each public function once — even expecting failure — would have caught all three, as would a grep of `scf.`/`scp.` attribute names against the current namespace:

```python
from sciris import sc_fileio as scf, sc_profiling as scp
print(hasattr(scf,'Empty'), hasattr(scf,'makefailed'), hasattr(scp,'loadbalancer'))  # actual: False False False
```

**An exception in a worker that never reaches the parent (findings 3, 6, 7).** In `parallelcmd()` the result is delivered by `_outputqueue.put()` as the last statement of the task, so *any* exception — including the deliberate `raise Exception` on the `die=True` path — means the parent's `outputqueue.get()` never returns and the program hangs rather than failing. In `parallel_progress()` the opposite mistake: `pool.apply_async()` is called with no `error_callback`, so an exception is discarded and the slot keeps its `None` placeholder, indistinguishable from a task that legitimately returned `None`.

**The two-pass py2 loader is effectively unreachable (findings 1, 4, 5, 13, 14, 15).** `loadobj2or3()` only falls back to `_loadobj2to3()` if `sc.load()` raises, and `sc.load()` (with its `latin` and `robust` methods) almost never does for an existing file. So the bugs inside `_loadobj2to3()` are latent, while the one path users actually hit (the fast path) loses exactly the binary data the two-pass loader was written to recover (finding 13). If finding 13 is fixed by routing py2 files to `_loadobj2to3()`, findings 1, 5, 14 and 15 become live and must be fixed at the same time.

## High severity

### 2. Importing the module silently corrupts every later pickle containing a classmethod — `legacy.py:212`, `233`

`cpreg.pickle(types.MethodType, _pickleMethod, _unpickleMethod)` runs at import time and installs a process-global entry in `copyreg.dispatch_table`, so it changes how `pickle` and `copy.copy` treat *all* bound methods everywhere in the process, not just legacy loads. `_pickleMethod()` stores `method.__self__.__class__` as the class to look the method up on. For a classmethod, `__self__` *is* the class, so `__self__.__class__` is the metaclass `type`, and the reduce records `(name, TheClass, type)`. `pkl.dumps()` succeeds; the load then does `getattr(type, name)` and fails. The bad data is already on disk by then.

```python
# mod2.py: class Thing:
#              @classmethod
#              def cmeth(cls): return 'cls'
import pickle, copy
from mod2 import Thing
pickle.loads(pickle.dumps(Thing.cmeth))()   # before the import: 'cls'
from sciris._extras import legacy            # <-- the only change
pickle.loads(pickle.dumps(Thing.cmeth))()
copy.copy(Thing.cmeth)
```

Actual, after the import:

```
AttributeError: type object 'type' has no attribute 'cmeth'   # from pickle.loads
AttributeError: type object 'type' has no attribute 'cmeth'   # from copy.copy
```

Expected: `'cls'` in both cases, as before the import. Blast radius is the whole process, and it reaches `sc.save()`/`sc.load()`: a single classmethod reference anywhere in the object graph makes the *entire* file unloadable, because `_RobustUnpickler.load()` catches the top-level error and returns one `NamedFailed` in place of the whole object.

```python
import sciris as sc
from sciris._extras import legacy
from mod2 import Thing
sc.save('cb.obj', {'callback': Thing.cmeth})
sc.load('cb.obj')['callback']
# actual:   UnpicklingWarning ... AttributeError: type object 'type' has no attribute 'cmeth'
#           then KeyError: 'callback'   (the object is a NamedFailed, not the dict)
# expected: the bound classmethod, which round-trips fine without the legacy import
```

`sc.dcp()` is unaffected (`copy.deepcopy` consults `_deepcopy_dispatch` for `MethodType` before `dispatch_table`), and ordinary bound instance methods round-trip correctly either way. Two things that look like related defects are *not*: stock pickle already re-binds an inherited method to the subclass override (verified identical with and without the import), and old pickles written *before* this registration still load, since `_unpickleMethod` handles them.

**Fix**: delete the `cpreg.pickle(types.MethodType, _pickleMethod, _unpickleMethod)` line (233) and keep the functions. Python 3 pickles bound instance methods and classmethods natively, and loading old pickles that reference `_unpickleMethod` only requires that function to exist, not the registration. (The fix originally proposed here — deriving `im_class = method.__self__ if isinstance(method.__self__, type) else method.__self__.__class__` — is wrong: `_unpickleMethod` then does `getattr(Thing, 'cmeth')`, which is already a bound classmethod, and wraps it again with `types.MethodType(..., Thing)`; calling the result raises `TypeError: Thing.cmeth() takes 1 positional argument but 2 were given`, verified on re-review.)

### 3. `parallelcmd()` is broken on Python 3.13 and hangs instead of failing — `legacy.py:466`, `551`, `560`, `570`

`_parallelcmd_task()` sets the loop variables with `exec(f'{_key} = _thisval')` and then runs `exec(_cmd)` in a separate call. Both `exec()` calls take their namespace from `locals()`, and PEP 667 (CPython 3.13) changed `locals()` in a function to return a *fresh snapshot* each call instead of a cached dict, so the assignments made by the first `exec` are discarded before the second one runs. The command therefore cannot see its own loop variables, `**kwargs` variables, or the `_returnval` it just assigned. And because `_outputqueue.put()` is the last line of the task, the resulting exception means nothing is ever sent to the parent, which blocks in `outputqueue.get()` indefinitely.

The docstring's own example:

```python
from sciris._extras import legacy
const = 4
cmd = "\nnewval = val+const\nresult = newval**2\n"
legacy.parallelcmd(cmd=cmd, parfor={'val':[3,5,9]}, returnval='result', const=const)
```

Actual: each of the three workers prints `NameError: name 'val' is not defined` followed by the bare `Exception` from line 563, and then the process hangs forever (killed by `timeout` at 60 s, exit 124). `faulthandler` confirms where:

```
Current thread ... (most recent call first):
  File ".../multiprocess/queues.py", line 101 in get
  File "/home/cliffk/sc/sciris/sciris/_extras/legacy.py", line 466 in parallelcmd
```

Expected: `[49, 81, 169]`. The mechanism, isolated:

```python
def f():
    exec('a = 1')
    return eval('a')
f()
# CPython 3.13.9: NameError: name 'a' is not defined
# CPython 3.12.3: 1
```

Since `pyproject.toml` requires Python >= 3.10, both interpreters are in contract; 3.12 and earlier work, 3.13 and later do not. The hang on worker failure is not 3.13-specific: it also occurs on 3.12 whenever `die=True` and the command raises. `die=False` does not help — `exec(f'{_returnval} = None')` is lost the same way, so `eval(_returnval)` at line 570 raises `NameError: name 'result' is not defined` outside the `try`, and again nothing is queued:

```python
import queue
from sciris._extras import legacy
q = queue.Queue()
legacy._parallelcmd_task('result = 1/0', {'i':[0]}, 'result', 0, q, None, None, None, False, {})
# actual: NameError: name 'result' is not defined ; q.empty() -> True
```

Two smaller defects in the same function: `raise Exception` at line 563 raises a new, empty exception instead of re-raising, so the real error reaches the user only via the child's stderr; and `_i, returnval = outputqueue.get()` at line 466 rebinds the caller's `returnval` (the name of the output variable) to a result value — harmless today only because all the `args` tuples are built in the preceding loop.

**Fix**: give `exec` a single explicit namespace dict and reuse it — `ns = {**globals(), **loopvars, **kwargs}; exec(_cmd, ns)` and read the result with `ns[_returnval]` — which fixes 3.13 and removes the need for the underscore-prefixed variable convention. Use one dict rather than separate globals/locals (`exec(_cmd, globals(), ns)`): with separate dicts, a `def` or `lambda` inside `cmd` cannot see `cmd`-level names such as `const`. Independently, wrap the whole body so the queue is always written (`finally: _outputqueue.put((_i, result))`, with a sentinel on failure) so a failed task can never hang the parent, and re-raise with `raise` rather than `raise Exception`.

## Medium severity

### 1. `_loadobj2to3()` calls two `sc_fileio` names that no longer exist — `legacy.py:74`, `81`, `148`, `158`

`StringUnpickler.find_class()` returns `scf.Empty` for every name in `not_string_pickleable` (`'datetime'`, `'BytesIO'`) and again for any class it cannot import, and both `loadintostring()`/`loadintobytes()` error handlers call `scf.makefailed()`. Neither attribute exists in current `sc_fileio`: `Empty` was deleted and `makefailed` became `_makefailed(module_name, name, exc, fixes, errors)` (note the renamed `exc` parameter — the legacy call site passes `error=` and `exception=`, so simply un-privatising the name is not enough). The result is that the private two-pass loader raises on any file containing a datetime or an unresolvable class, and because the second failure happens inside the handler for the first, the original diagnosis is destroyed.

```python
import gzip, pickle, datetime as dt
from sciris._extras import legacy
with gzip.open('dt.obj','wb') as f:
    pickle.dump({'d': dt.datetime(2020,1,1)}, f, protocol=2)
legacy._loadobj2to3(filename='dt.obj')
```

Actual:

```
Warning, string pickle loading failed: module 'sciris.sc_fileio' has no attribute 'Empty'
AttributeError: module 'sciris.sc_fileio' has no attribute 'makefailed'
```

Expected: the dict, with the datetime recovered by the two-pass substitution (that is exactly what `not_string_pickleable` exists to do). Restoring the historical `Empty`/`makefailed` as local shims makes the same file load (that shim is how findings 4, 5, 14 and 15 were reached), which confirms this is a stale-reference bug rather than a deeper breakage.

**Reachability (corrected on review)**: the original version of this finding said the same `AttributeError` is reached from the public `loadobj2or3()` and that "every py2 pickle containing a datetime dies". That is false for any readable file: `loadobj2or3()` only calls `_loadobj2to3()` if `sc.load()` raises, and `sc.load()` loads both the repro's `dt.obj` (a py3 pickle) and hand-assembled py2 pickles (dict with datetime, object with a list of datetimes, `Blobject`) via its `pickle`/`latin` methods without ever touching `_loadobj2to3()`. So this is a dead private function, and the severity is lowered from High to Medium.

**Fix**: reinstate a local two-line `Empty` class in `legacy.py` (it is only used here, and only needs `__init__(*args, **kwargs)` and a no-op `__setstate__`), and either call `scf._makefailed(module_name=..., name=..., exc=E)` with the current signature (verified to exist with that signature) or build the failure object locally. Both `loadintostring()`/`loadintobytes()` handlers need the argument names changed, not just the function name. For the datetime part, a simpler and more complete fix is to remove `'datetime'` from `not_string_pickleable`: Python 3's `datetime` unpickles py2 datetimes correctly under `encoding='latin1'` (verified: `pickle.load(..., encoding='latin1')` of the py2 stream gives `{'a': datetime(2020,1,1), 'l': [datetime(2021,3,4)]}`), which also fixes finding 14. The datetime branches in `recursive_substitute()` would still fire (the bytes pass still yields real datetimes), so finding 5 must still be fixed or those branches removed. Reinstating `Empty` also makes finding 15 live, so fix it at the same time.

### 6. `_parallelcmd_task()` calls `loadbalancer()` from the wrong module — `legacy.py:546`

`scp` is `sc_profiling`, but `loadbalancer()` lives in `sc_parallel` (`sc.loadbalancer.__module__` -> `'sciris.sc_parallel'`). So the whole point of `parallelcmd()`'s documented `maxcpu`, `maxmem`, `interval` and deprecated `maxload` arguments — throttling — raises in the worker. Combined with finding 3 the parent then hangs, so the user sees no error at all in the foreground.

```python
import queue
from sciris._extras import legacy
q = queue.Queue()
legacy._parallelcmd_task('r=1', {'i':[0]}, 'r', 0, q, 0.9, None, None, True, {})
# actual:   AttributeError: module 'sciris.sc_profiling' has no attribute 'loadbalancer'
#           q.empty() -> True   (so the parent's outputqueue.get() would never return)
# expected: the load balancer runs, then (i, 1) is queued
```

`if _maxcpu or _maxmem:` means the line is only reached when throttling is requested, which is why the default path fails elsewhere (finding 3) rather than here. The signature of `sc.loadbalancer()` matches the existing call, so only the module needs changing.

**Fix**: import `sc_parallel` in the header and call `scpl.loadbalancer(...)`, or call `sc.loadbalancer(...)` via the top-level namespace.

### 7. `parallel_progress()` swallows worker exceptions and reports `None` as a result — `legacy.py:523`, `526`

`pool.apply_async(fcn, ..., callback=partial(callback, idx=i))` is called with no `error_callback`, so a task that raises never invokes any callback: its slot keeps the `None` placeholder from `results *= len(inputs)`, the progress bar is never advanced for it, and `pool.join()` returns normally. The function then returns a list in which failures are indistinguishable from tasks that legitimately returned `None`. The docstring says the result "is essentially equivalent to `list(map(fcn, inputs))`", and `map` propagates.

```python
from sciris._extras import legacy
def boom(x):
    if x == 2: raise ValueError('nope')
    return x**2
legacy.parallel_progress(boom, [1,2,3,4], num_workers=2, show_progress=False)
# actual:   [1, None, 9, 16]      (no warning, no traceback, exit code 0)
# expected: ValueError: nope
```

Verified clean around it: normal operation, the zero-argument/`inputs`-as-count form, `show_progress=True`, and the `initializer` argument all work, results are returned in input order, and no worker processes are left behind.

**Fix**: pass `error_callback=` to `apply_async` and either re-raise in the parent after `pool.join()` or store the exception object in the slot (as `sc.parallelize(die=False)` does), so a failed task cannot masquerade as a `None` result.

### 9. The documented `legacy_dataframe` recovery workflow never fires — `legacy.py:254`

The class docstring — the only documentation for a class whose stated purpose is "maintained solely to allow loading old files" — tells the user to load with `remapping={'sciris.sc_dataframe.dataframe': scl.legacy_dataframe}`. But `_RobustUnpickler.find_class()` (`sc_fileio.py:2594-2604`) only consults `remapping` *after* `super().find_class()` raises, and `sciris.sc_dataframe.dataframe` still exists in current Sciris — it is now the pandas subclass. So the remapping is ignored, the old state is handed to `pandas.DataFrame.__setstate__`, and the load fails.

Repro (hand-assembled protocol-2 pickle of `sciris.sc_dataframe.dataframe` with the old `{'cols': [...], 'data': array}` state, gzipped):

```python
import sciris as sc
from sciris._extras import legacy as scl
sc.load('olddf.obj', remapping={'sciris.sc_dataframe.dataframe': scl.legacy_dataframe})
```

Actual:

```
UnpicklingWarning: ... 'error': 'Pre-0.12 pickles are no longer supported'
    NotImplementedError('Pre-0.12 pickles are no longer supported')   # from pandas/core/generic.py __setstate__
type: <class 'sciris.sc_fileio.NamedFailed'>
```

Expected: a `legacy_dataframe` with `cols=['a','b']`. The result is identical with and without the `remapping` argument, which is the proof that the argument is not consulted; the same state pickled under a non-existent class name remaps fine. The defect is reportable against this file because the workflow it documents cannot work; the mechanism, though, is in `sc_fileio`.

**Fix**: in `sc_fileio._RobustUnpickler.find_class()` (not in `legacy.py`), check the *user-supplied* portion of `remapping` (not `known_fixes`) before delegating to `super().find_class()`, so an explicit remapping can override a name that still resolves but has changed meaning. This is safe: an explicit user remapping should win. Failing that, the `legacy_dataframe` docstring should document a mechanism that works and stop advertising one that does not.

### 13. `loadobj2or3()` silently returns py2 `Blobject`/`Spreadsheet` with a `str` blob — `legacy.py:45-48`

*New on review.* The fast path `scf.loadobj(filename=filename, **kwargs)` (i.e. `sc.load()`) succeeds on py2 pickles via its `latin` method (and, for ASCII-only data, plain `pickle`), so the two-pass bytes-aware loader, whose whole purpose is recovering binary blobs, is never used. A py2 `str` blob comes back latin1-decoded as a Python 3 `str`, and every method that writes it out fails.

```python
# b3.obj: hand-assembled py2 pickle of sciris.sc_fileio.Blobject, state {'blob': b'\x89PNG\xff', 'bytes': None, 'name': 'old.xlsx'}
from sciris._extras import legacy
o = legacy.loadobj2or3('b3.obj')
repr(o.blob)    # actual: '\x89PNGÿ' (str)
o.tofile()      # actual: TypeError: a bytes-like object is required, not 'str'
o.save('x.bin') # actual: TypeError: a bytes-like object is required, not 'str'
```

Expected: `o.blob == b'\x89PNG\xff'` (bytes), and `tofile()`/`save()` work. For comparison, `_loadobj2to3('b3.obj')` (with finding 1 shimmed) returns `b'\x89PNG\xff'` correctly. No warning is emitted, so the user only finds out when trying to use the spreadsheet.

**Fix**: the simplest lossless option is to post-process after the fast path: for any `Blobject` in the result whose `blob` is a `str`, set `blob = blob.encode('latin1')` (latin1 decoding is bijective, so this recovers the exact bytes). Alternatively, restrict the fast path so py2 pickles fall through, e.g. `scf.loadobj(filename=filename, method=['pickle', 'dill'], die=True, **kwargs)` (verified: `dill` fails and plain `pickle` fails on non-ASCII bytes, so real binary blobs would reach `_loadobj2to3()`); that option requires findings 1, 5, 14 and 15 to be fixed first.

## Low severity

### 4. `recursionlimit` counts nodes visited, not recursion depth — `legacy.py:123`, `136`

`recursive_substitute()` increments `recursionlevel` on entry and then *assigns the child's return value back into its own counter* (`recursionlevel = recursive_substitute(...)`), so the counter accumulates across siblings instead of unwinding. It is a node-visit budget, not a depth limit. When it is exhausted, every remaining branch is skipped with a message that misdescribes the cause, and the caller gets a partially-substituted object: the datetimes (and blobs) that the two-pass scheme exists to recover are silently left as placeholders. With the default limit of 1000 this happens to any old object with more than ~1000 containers.

Repro (a flat object with N children, each holding one datetime; actual nesting depth 2; `scf.Empty`/`scf.makefailed` shimmed per finding 1, and the py2 stream hand-assembled — full script in the scratchpad as `mkwide.py`):

```python
out = legacy._loadobj2to3(filename='wide.obj')      # 1200 children, default recursionlimit
recovered = sum(isinstance(v.__dict__.get('when'), dt.datetime) for v in out.__dict__.values())
```

Actual:

```
total children: 1200   recovered: 1000   left as Empty: 200
nesting depth of the object: 2
n warnings printed: 200
Warning, internal recursion depth exceeded, aborting: depth=1001, <class 'myold.MyClass'> -> <class '...Placeholder'>
```

Expected: 1200 recovered, no warning — the structure is two levels deep and the limit is 1000. The same script with `recursionlimit=5` recovers exactly 5 of 1200, which shows the limit is a sibling count:

```
recursionlimit = 5 -> c1..c5 datetime, c6..c10 Empty
```

The docstring promises "Uses a recursive approach, so can set a recursion limit", and the message says "recursion depth exceeded", so both the argument and the diagnostic are misleading about what actually happened.

**Severity (corrected on review)**: originally rated High, but this code is only reached by calling the private `_loadobj2to3()` directly (see the re-verification note), so it is Low. It becomes relevant if finding 13 is fixed by routing py2 files to the two-pass loader.

**Fix**: do not thread the counter through the return value — pass `recursionlevel+1` down and discard the child's value (the depth then unwinds naturally), or drop `recursionlevel` entirely and use `len(track)`, which is already maintained and is the true depth. If a node budget is also wanted, count it separately with a distinct message.

### 5. The dict branch of `recursive_substitute()` uses `setattr()` on a dict — `legacy.py:116`

Inside `if isinstance(obj2, dict):`, a datetime value is written with `setattr(obj1, k.decode('latin1'), v)` — a copy-paste of line 129 from the object branch below, where `obj1` really does have a `__dict__`. Here `obj1` is the corresponding dict from the string pass, so the assignment raises.

Repro (hand-assembled py2 pickle of `{'a': datetime(2020,1,1), 'b': 3}` with `SHORT_BINSTRING` keys, gzipped; `scf.Empty`/`scf.makefailed` shimmed so finding 1 does not mask this):

```python
legacy._loadobj2to3(filename='py2.obj')
# actual:   AttributeError: 'dict' object has no attribute 'a' and no __dict__ for setting new attributes
# expected: {'a': datetime.datetime(2020, 1, 1, 0, 0), 'b': 3}
```

The object branch is fine — the equivalent pickle with the datetime held as an *attribute* of an instance recovers correctly (verified: `created` came back as `datetime.datetime(2019, 5, 4, 0, 0)`), so only dict containers are affected. Note also the internal inconsistency in key handling: `k.decode('latin1')` here assumes the key is always `bytes` (true only because `obj2` comes from the `encoding='bytes'` pass), whereas the recursion case five lines lower at 118-119 guards it with `if isinstance(k, (bytes, bytearray))`. The same asymmetry exists in the object branch at 129 versus 131-132.

**Severity (corrected on review)**: originally rated Medium and described as affecting "the single most common py2 Sciris pickle shape"; the public `loadobj2or3()` loads that shape correctly via `sc.load()`, so this is only reached by calling `_loadobj2to3()` directly, and is Low.

**Fix**: `obj1[k.decode('latin1') if isinstance(k, (bytes, bytearray)) else k] = v`, mirroring the key handling used for the recursion case just below.

### 14. `recursive_substitute()` never substitutes datetimes inside lists or tuples — `legacy.py:113-138`

*New on review.* `recursive_substitute()` only walks dicts and objects with `__dict__`; lists and tuples are skipped, so any datetime held in a list (e.g. a date vector) stays as the string-pass `Empty` placeholder. Reached only via direct `_loadobj2to3()` calls (with finding 1 shimmed).

```python
# py2 pickle of MyClass with attribute dates=[datetime(2020,1,1), datetime(2021,1,1)]
o = legacy._loadobj2to3(filename='l.obj')
o.dates   # actual: [<Empty object>, <Empty object>]   expected: [datetime(2020,1,1), datetime(2021,1,1)]
# likewise kids=[MyClass(when=datetime(...))] -> o.kids[0].when is an Empty
```

**Fix**: remove `'datetime'` from `not_string_pickleable` so the latin1 pass loads datetimes natively wherever they are (verified to work; see finding 1), rather than extending the substitution to lists.

### 15. `_loadobj2to3()` crashes if an unresolvable class has any nested object attribute — `legacy.py:136`

*New on review.* When the string pass cannot import a class, `find_class()` returns `Empty`, whose `__setstate__` discards the state. The bytes pass gives a `Placeholder` that keeps it, so `recursive_substitute()` then does `getattr(obj1, k)` on the empty object and aborts the whole load. This triggers once finding 1 is fixed by reinstating the historical `Empty` (as proposed there), and "a class that no longer exists" is the typical case for old files.

```python
# py2 pickle of gone_module.Gone with attribute child=myold.MyClass(x=1)
legacy._loadobj2to3(filename='u.obj')
# actual:   AttributeError: 'Empty' object has no attribute 'child'
# expected: the load completes, with the unresolvable object left as a placeholder
```

**Fix**: in the object branch, skip recursion when the target is missing, e.g. `if recursionlevel <= recursionlimit and hasattr(obj1, k): ...` (and in the dict branch, `k in obj1`), or skip recursion entirely when `obj1` is an `Empty`.

## Misplaced `# pragma: no cover`

The whole module is excluded from coverage by design, so these are noted only because I demonstrated that the code behind them runs and is wrong — not as a coverage complaint.

| Line | Branch | Reachable via |
|------|--------|---------------|
| 33 | `loadobj2or3()` | Any py2 pickle containing a `Blobject` with binary data (finding 13) |
| 52 | `_loadobj2to3()` | Direct call only: `legacy._loadobj2to3(filename=...)` (findings 1, 4, 5, 14, 15) |
| 537 | `_parallelcmd_task()` ("No coverage since pickled") | Callable directly in-process: `legacy._parallelcmd_task('r=1', {'i':[0]}, 'r', 0, queue.Queue(), 0.9, None, None, True, {})` (findings 3, 6) |
| 447 | `parallelcmd()` `maxload` deprecation | `parallelcmd(..., maxload=0.8)` reaches it; it is the only path that then hits finding 6 |

`parallelcmd()` itself carries no pragma but has no test, which is how the Python 3.13 breakage (finding 3) went unnoticed.

## Verified clean

Recorded so the same ground is not re-covered. All of the following were hypothesised, tested by execution, and found correct.

**Module import and references**: the module imports cleanly as `from sciris._extras import legacy` with no import-time exception, and every other cross-module reference in it still resolves — `scu.flexstr`, `scu.checktype`, `scu.dcp`, `scu.isnumber`, `scu.traceback`, `sco.odict`, `scf.Blobject`, `scf.loadobj` all exist (only `scf.Empty`, `scf.makefailed` and `scp.loadbalancer` do not; findings 1 and 6). Nothing else in the Sciris package imports `legacy`, so the blast radius of everything here is user code plus, for finding 2, any process that imports the module at all.

**`loadobj2or3()` / `_loadobj2to3()`**: the fast path is correct for modern files — a modern `sc.save()` file loads through `loadobj2or3()` and returns the identical object — and for py2 files without binary blobs (dicts and objects with datetimes load correctly via `sc.load()`'s `latin` method). Both stream branches of `_loadobj2to3()` are structurally sound: the `filename` branch opens `gz.GzipFile` twice under `with`, the `filestring` branch nests `closing(IO(...))` and `gz.GzipFile(...)`, and neither leaks: 20 consecutive *failing* loads left the open-fd count unchanged at 4 and created no temp files in the working directory. Reading the stream twice (once per unpickler) is deliberate and necessary, not a bug. The `filestring` branch produces the same result as the `filename` branch on the same bytes, and calling with neither argument raises the documented exception. The two-pass latin1/bytes substitution genuinely works once findings 1/4/5 are out of the way: datetimes held as *object attributes* are recovered exactly, at multiple levels, and `Blobject`/`Spreadsheet` blobs are recovered exactly (`blob` -> `b'\x89PNGdata'`, `bytes` -> `None`, plus the `name` and `created` attributes). `Placeholder.__init__(*args)` with no explicit `self` is fine, and `Placeholder.__setstate__` correctly retains the bytes-keyed state — which is why the lowercase entries in `byte_objects` (`'spreadsheet'`, `'blobject'`, which have never matched the actual class names `Spreadsheet`/`Blobject`) turn out to be harmless: falling through to `Placeholder` preserves exactly the `__dict__[b'blob']` / `__dict__[b'bytes']` that line 110-111 reads, verified end-to-end for both class names. `not_string_pickleable`/`byte_objects` are module-level lists but are only read, never mutated. The `track`/`track2` bookkeeping is correct (`track.copy()` per branch, so siblings do not share a path).

**Twisted pickling helpers**: ordinary bound instance methods round-trip correctly through `pkl.dumps`/`pkl.loads` both before and after the registration, and `sc.dcp()`/`copy.deepcopy` of a bound method is unaffected by it (`copy` checks `_deepcopy_dispatch` for `MethodType` before `dispatch_table`). Pickles written *before* the registration still load after it. `_unpickleMethod`'s `im_self is None` branch and its `AttributeError` recursion into `im_self.__class__` are consistent with upstream Twisted. The subclass-rebinding behaviour is *not* a defect introduced here: stock pickle also re-binds an inherited bound method to the subclass's override (`P.m.__get__(c)` round-trips to `C.m` with and without the import, verified identically). `pickleMethod`/`unpickleMethod` aliases exist as documented.

**`legacy_dataframe`**: all five `make()` usage forms construct correctly (`()`, `(['a','b','c'])`, `(['a','b','c'], nrows=2)`, the 2-D header form `[['a','b','c'],[1,2,3],[4,5,6]]`, and `cols`+`data`), including square list-of-lists data. `__repr__` produces correctly aligned output for zero rows, for `nrows=2` of zeros, and for populated frames; `ndigits`/`indformat` work despite `ndigits` being a `np.float64` (`'%%%is' % np.float64(1.0)` -> `'%1s'`); the `spacing` argument is honoured; the `<empty dataframe>` short-circuit works. `nrows`, `ncols` and `shape` agree with the data (`(2, 2)` for a 2x2, `(0, 0)` for an empty frame), the `ncols` corruption check does not false-positive, and the `nrows` bare `except` returning 0 is the documented empty-frame behaviour. The 1-D-data reshape rules (single column, single row, and the two dimension errors) all behave as written, and `scu.dcp()` on the 2-D header path means the caller's nested list is not mutated. (The square-dict transpose and the `legacy_dataframe(pd.DataFrame(...))` constructor crash are real but are rejected as not worth fixing; see original finding 8 below.)

**Parallelization**: `parallel_progress()` is correct apart from finding 7 — the list-of-inputs form and the count/zero-argument form both work, `num_workers` is honoured, results come back in input order, `show_progress=True` renders and closes the bar, `initializer` really does reach the workers (verified by setting a marker in the child and reading it back), and no worker processes are left alive after the call — including after the `TypeError` from a non-integer count, because `mp.Pool` spawns lazily, so there is no pool leak on that path. `results = [None]; results *= n` is correct for `n = 0` and for numpy integers. `parallelcmd()`'s queue protocol is order-safe by construction (each task returns its own index and the parent writes `outputlist[_i]`), `textwrap.dedent()` correctly de-indents a triple-quoted command, the `maxload` -> `maxcpu` deprecation warning fires with the documented `FutureWarning`, and `outputlist.tolist()` returns a plain list; the `returnval` rebinding at line 466 is genuinely harmless because every `args` tuple is built in the earlier loop.

**Not testable without a genuine old file**: nothing material. The Python 2 paths do require py2-style streams, but hand-assembling protocol-2 pickles with `SHORT_BINSTRING` keys and py2-style `GLOBAL`/`REDUCE` datetimes drove every branch of `_loadobj2to3()` (both stream sources, the dict branch, the object branch, the `Blobject` branch, both unpicklers' fallbacks, and the recursion limit), so no finding above rests on reading alone. The one thing I could not reproduce is a real pre-2018 Sciris object with lowercase `spreadsheet`/`blobject` class names; the `Placeholder` fallback makes that case work anyway, as noted above.

## Rejected on review

- **8** — `legacy_dataframe(dict)` transposes the data whenever it is square — NOT WORTH FIXING: it reproduces, but only on the constructor path of a class whose sole purpose is unpickling (which restores `__dict__` directly and never calls `make()`), and nobody builds new `legacy_dataframe`s from square dicts.
- **10** — A py2 blob without a `bytes` entry aborts the whole load — NOT WORTH FIXING: `Blobject` was introduced (commit `701b438`, 2018-09-03) with `self.bytes = None` in `__init__`, so every Sciris `Blobject` ever pickled has a `bytes` key and no real file has this shape.
- **11** — The documented import paths do not exist (`sciris.extras`, `sc_legacy`) — NOT WORTH FIXING: accurate, but a documentation-only typo worth a one-line doc edit rather than a code bug.
- **12** — `loadobj2or3()`'s bare `except:` destroys the real error — NOT WORTH FIXING: the repro's `FileNotFoundError` is the correct diagnosis, and since `sc.load()` with `die=False` almost never raises for an existing file, the masking scenario does not arise in practice.

## Suggested order of work

1. **Finding 2** — it is the only defect here that damages data outside the legacy module, it does so silently at save time, and the fix is deleting one line.
2. **Finding 3** — `parallelcmd()` needs the `exec` namespace change to work on Python 3.13 at all, and the `finally: put()` change so that a failed task can never hang the caller.
3. **Findings 6, 7** — a stale module reference and a swallowed worker exception; both small and local.
4. **Finding 13** — the one py2-loading defect users actually hit through the public entry point; the latin1 re-encode post-processing is a few lines.
5. **Findings 1, 5, 14, 15** — the private two-pass loader; fix together (removing `'datetime'` from `not_string_pickleable` covers most of 1 and all of 14), and necessarily before routing any py2 files to it as part of finding 13.
6. **Findings 4, 9** — the recursion counter in the private loader, and the `remapping` precedence in `sc_fileio._RobustUnpickler.find_class()`.

Findings 2, 7 and 13 produce plausible-looking output rather than an error, so none of them would announce itself in downstream code; findings 3 and 6 announce themselves as a hang, which is arguably worse. Each has a natural regression test (a classmethod must survive a pickle round-trip after importing the module; a raising task must not return `None`; a py2 `Blobject` must come back with a `bytes` blob; a depth-2 object with 1200 children must be fully substituted), which would be worth adding alongside the fixes — the module currently has none.
