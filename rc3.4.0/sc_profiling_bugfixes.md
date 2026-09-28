# `sc_profiling.py` bug audit

Audit of `sciris/sc_profiling.py` (1596 lines) for genuine defects: wrong numerical results, documented arguments that don't work, silent data corruption, crashes on in-contract input, and process-wide state that outlives the call that created it. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps, and performance. The file was covered in full by two parallel auditors: one covering `checkmem()`, `checkram()`, `benchmark()`, and the `profile` class (lines 36-786), the other covering `mprofile()`, `cprofile()`, `listfuncs()`, `tracecalls()`, `LimitExceeded`, and `resourcemonitor()` (lines 787-1596). **Method**: line-by-line reading of each function, followed by executed hypothesis tests — every "actual" value below was produced by running the code against the editable install (Sciris 3.3.0, editable, commit `2d69aad`), Python 3.13.9, numpy 2.4.6, pandas 3.0.5, line_profiler 5.0.2, `SCIRIS_BACKEND=agg`, and every finding was reproduced a second time independently (in a fresh interpreter, from the minimal snippet shown) before being recorded here.

**Independent re-verification.** This document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`; `sc_profiling.py` unchanged at 1596 lines, same line numbers): every finding was re-run from a minimal script in a fresh interpreter. Of the original 30 findings, 15 were confirmed (some with caveats or corrected severities, noted inline), 2 were rewritten because the description or the proposed fix was inaccurate (1: the proposed fix itself raises; 12: the anchoring is deliberate, only the docstring example is broken), and 13 were rejected as not a bug or not worth fixing (see "Rejected on review"). Four bugs missed by the original audit were added as findings 31-34. Original finding numbers are kept stable, so the numbering has gaps. The document now lists 21 findings: 5 High, 11 Medium, 5 Low.

**Nothing in this document has been applied.** All fixes are described, not made.

The most consequential defects here are leaked global hooks: a profiling tool left installed after an ordinary call degrades or breaks everything that runs later in the same process, including the test suite itself.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `sc.profile()` | An exception inside the profiled function leaves the line_profiler hook installed, permanently disabling `cProfile`/`sc.cprofile()` for the rest of the process | 496 |
| 2 | High | `sc.profile.plot()` | Raises `ValueError: output array is read-only` whenever the slowest function takes under 1 second | 757 |
| 3 | High | `sc.mprofile()` | Leaves a global trace hook installed after it returns | 820 |
| 4 | High | `sc.cprofile()` | Caches its parsed stats forever, so reusing the object silently reports the previous run | 928 |
| 5 | High | `sc.tracecalls.check_expected()` | Appends the `str` type instead of the supplied name, so string input never matches | 1326 |
| 6 | Medium | `sc.checkmem()` | On a numpy array (or any sequence of >1000 items) raises `RuntimeError` instead of reporting its size | 127, 143 |
| 8 | Medium | `sc.checkmem()` | `subtotals=False` is not passed to the recursive call, so nested subtotal rows appear anyway | 152 |
| 9 | Medium | `sc.profile.merge()` | `merge(inplace=False)` returns a half-initialised object; `p1 + p2` then breaks on `disp(skiprun=True)` | 532 |
| 11 | Medium | `sc.tracecalls()` | Crashes in `__exit__` when the trace filter matched nothing, masking the user's own exception | 1289 |
| 14 | Medium | `sc.listfuncs()` | Silently misses inherited methods, so `sc.profile(follow=SubClass)` skips half the class | 1084 |
| 16 | Medium | `sc.cprofile.to_df()` | `maxitems=...` is computed and then ignored | 998 |
| 17 | Medium | `sc.resourcemonitor()` | `kill()` enumerates every process on the system before interrupting the main thread, so a busy job is interrupted late (by a highly variable amount) or not until the body ends | 1570 |
| 18 | Medium | `sc.resourcemonitor()` | The SIGINT handler calls the original handler with no arguments, so Ctrl-C raises `TypeError` | 1441 |
| 19 | Medium | `sc.resourcemonitor()` | `kill_parent=True` raises `AttributeError` on a non-existent `self.parent_pid` and so never interrupts the main thread | 1580 |
| 31 | Medium | `sc.checkmem()` | `plot=True` pie chart includes the `Total` row, so it is always 50% of the pie and every real slice is halved | 171-173 |
| 32 | Medium | `sc.checkmem()` | Dicts with non-string keys mislabel key `0` as `'Variable'` and raise `TypeError` at `descend>=2` | 121, 136, 151, 157 |
| 7 | Low | `sc.profile.disp()`, `sc.profile.to_df()` | `maxentries` is silently ignored when `bytime=0`, so every function is dumped | 709 |
| 12 | Low | `sc.tracecalls()` | The class docstring's regex example is an invalid regex, and its `exclude` could never match | 1142 |
| 20 | Low | `sc.resourcemonitor()` | `die=False` is completely silent on a breach, contradicting its documented default verbosity | 1385 |
| 33 | Low | `sc.cprofile.to_df()` | With `columns='full'`, `percall` stays in seconds while `cumtime`/`selftime` are converted to ms | 1001-1004 |
| 34 | Low | `sc.resourcemonitor.start()`, `monitor()` | The documented `label` argument is ignored | 1428, 1486 |

Rejected on review (not listed above): 10, 13, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30. See "Rejected on review" near the end.

## Recurring patterns

**Unbalanced global hooks (the dominant, cross-cutting pattern).** Two separate sites install a process-wide trace/profile hook and fail to guarantee its removal, found independently by the two halves of this audit: `profile.run()` calls `prof.enable_by_count()` (line ~491) and only calls `prof.disable()` after `wrapper(...)` returns, with no `try`/`finally`, so an exception in the profiled function (finding 1) leaves the hook installed; and `mprofile()` calls `lp.enable_by_count()` unconditionally (line ~820) even though the `wrapper = lp(run)` call already brackets the run with its own enable/disable pair, so the explicit enable is redundant and unbalanced on every call, not just the exception path (finding 3). A milder relative, `tracecalls.stop()` clobbering rather than restoring a previously-installed `sys.setprofile` hook (original finding 13), was rejected on review as too rare to matter, although its two-line save/restore fix is safe. Any fix to one hook site should prompt a check of the others. The test suite already knows something is wrong here: `tests/test_profiling.py:126` wraps `test_cprofile()` in a `try/except ValueError` specifically for the string `'Another profiling tool is already active'` (the symptom of finding 1) and downgrades it to a warning. (The original audit also attributed the `TypeError` caught at `tests/test_profiling.py:92` to finding 3; that is speculation, and the test's own comment, "This happens when re-running this script", points to double-wrapping instead.)

**Recursive/derived calls that forget to forward their own arguments.** `checkmem()`'s recursion drops `subtotals` and hard-codes `verbose=False` (finding 8); `profile.merge()`'s `__new__`-based construction drops nine of `__init__`'s sixteen attributes (finding 9); `cprofile.to_df()` resolves six per-call overrides via `sc.ifelse` and uses five of them, re-reading `self.maxitems` instead of the resolved local (finding 16); `resourcemonitor.start()`/`monitor()` accept a documented `label` and never use it (finding 34). In each case the object or result built by the derived path is subtly not what the top-level path would build.

**Silent empty results feeding into unguarded reductions.** `tracecalls.to_df()` and `disp()` both assume at least one entry, so a filter that matched nothing turns into a confusing exception from `__exit__` that masks the user's real exception (finding 11); `cprofile.to_df()` handles the analogous empty case correctly, which shows the fix is easy and there is no structural reason for the difference.

**Falsy/wrong sentinels and mismatched attribute names.** `resourcemonitor` prints `self.parent_pid`, which is never assigned anywhere in the file (`self.parent` is the real attribute) (finding 19), and its SIGINT fallback calls `self._orig_sigint()` with zero arguments when a signal handler must take two (finding 18); `checkmem()` uses raw dict keys as string labels, so an int key `0` is treated as a missing prefix and relabelled `'Variable'` (finding 32). All are silent-until-triggered mismatches between what a piece of code assumes exists and what actually exists.

## High severity

### 1. An exception inside the profiled function leaves the line_profiler hook installed, permanently disabling `cProfile`/`sc.cprofile()` for the rest of the process — `sc_profiling.py:496`

`run()` calls `prof.enable_by_count()` (491), runs `wrapper(...)` (496), and only then calls `prof.disable()` (498). There is no `try`/`finally`, so if the profiled function raises — the single most likely thing to happen while you are profiling code you are still debugging — `prof.disable()` is never reached and the profiler stays registered. On Python 3.12+ line_profiler registers itself as a `sys.monitoring` tool, so the leak is invisible to `sys.gettrace()` but blocks every other profiling tool in the process.

```python
import sys, cProfile
import sciris as sc

def boom():
    x = 0
    for i in range(100): x += i
    raise ValueError('deliberate')

try:
    sc.profile(boom, verbose=False)
except ValueError:
    pass
print('leaked tool:', sys.monitoring.get_tool(2))
cProfile.Profile().enable()
```

Actual:

```
leaked tool: line_profiler
ValueError: Another profiling tool is already active
```

Expected: `leaked tool: None`, and `cProfile` works. After a *normal* `sc.profile()` run the state is clean (`sys.monitoring.get_tool(2)` is `None`), so this is specifically the error path. `sc.cprofile()` fails the same way:

```
--- try sc.cprofile after the leak ---
sc.cprofile FAILED: ValueError Another profiling tool is already active
```

Two further symptoms of the same leak, both observed: (a) a second `sc.profile()` call still "works" but is now running against a stale global registration; (b) at interpreter shutdown every subsequent `raise` in the process is routed through line_profiler's raise handler, producing repeated `Exception ignored in: <function _removeHandlerRef ...> AttributeError: 'NoneType' object has no attribute 'monitoring'` noise. `tests/test_profiling.py:126` already wraps `test_cprofile()` in a `try/except ValueError` specifically for the string `'Another profiling tool is already active'` and downgrades it to a warning — i.e. the test suite is already papering over this failure mode rather than catching it.

**Fix**: wrap the `wrapper()` call in `try`/`finally` and make exactly one disable call in the `finally` block:

```python
with sc.timer(verbose=self.verbose) as T:
    try:
        wrapper(*self.args, **self.kwargs)
    finally:
        prof.disable_by_count()   # or keep the existing prof.disable(), but not both
```

**Correction on review**: the original audit proposed `finally: prof.disable_by_count(); prof.disable()`. That fix is itself broken: both calls deregister the `sys.monitoring` tool, so the second one raises `ValueError: tool 2 is not in use` (from line_profiler's `_SysMonitoringState.deregister`), re-verified on 2026-09-25. Either call on its own works on both the success and the exception paths (`sys.monitoring.get_tool(2)` is `None` afterwards, timings are recorded, and `cProfile` can start).

### 2. `profile.plot()` raises `ValueError: output array is read-only` whenever the slowest function takes under 1 second — `sc_profiling.py:757`

`plot()` does `x = df.time.values` and then `x *= 1e3` to rescale to milliseconds. Under pandas 3 (copy-on-write) `Series.values` is a read-only view of the column, so the in-place multiply raises. The branch is taken whenever `x.max() < 1`, i.e. for any profile where no followed function took a whole second — the overwhelmingly common case, including the function in the class's own docstring example.

```python
import matplotlib; matplotlib.use('agg')
import sciris as sc

def slow_fn():
    n = 10000
    int_list = []
    int_dict = {}
    for i in range(n):
        int_list.append(i)
        int_dict[i] = i

p = sc.profile(slow_fn, verbose=False)
p.plot()
```

Actual:

```
  File "/home/cliffk/sc/sciris/sciris/sc_profiling.py", line 757, in plot
    x *= 1e3
ValueError: output array is read-only
```

Expected: a bar chart. Confirmed that the *only* thing keeping the method alive is the >= 1 s branch: profiling a function that takes 1.10 s plots fine, the same code profiling a 0.0055 s function raises. `plot()` is never called anywhere in Sciris or in `tests/test_profiling.py`, which is why this is unnoticed. Note that even where it does not raise, `x *= 1e3` and `ylabels[i] += ...` are intended to mutate arrays extracted from `self.df` (which `to_df()` has just assigned), so on any pandas version where `.values` returns a writeable view this silently rewrites `self.df.time` in milliseconds while leaving the column named `time` and `percent` unchanged.

**Fix**: take copies before rescaling — `x = df.time.values.copy()` and `ylabels = list(df.name.values)` — rather than mutating the dataframe's buffers.

### 3. `sc.mprofile()` leaves a global trace hook installed after it returns — `sc_profiling.py:820`

`mprofile()` calls `lp.enable_by_count()` at line 820 and never calls the matching `disable_by_count()`; the `wrapper = lp(run)` call at 822 already brackets the run with its own enable/disable pair, so the explicit enable is both redundant and unbalanced. `memory_profiler.LineProfiler.enable()` installs `sys.settrace(self.trace_memory_usage)`, so after `sc.mprofile()` returns, *every* subsequent line of Python in that interpreter is traced and memory-sampled.

```python
import sys, time
import sciris as sc

def big(): return [0]*100000

print('gettrace before:', sys.gettrace())
sc.mprofile(big, show_results=False)
print('gettrace after :', sys.gettrace())

def bench():
    t = time.time(); s = 0
    for i in range(300000): s += i
    return time.time()-t
print('after mprofile: %.3f s' % bench())
sys.settrace(None)
print('after settrace(None): %.3f s' % bench())
```

Actual:

```
gettrace before: None
gettrace after : <bound method LineProfiler.trace_memory_usage of <memory_profiler.LineProfiler object at 0x7c5f0acbc440>>
after mprofile: 0.084 s
after settrace(None): 0.006 s
```

Expected: `gettrace after` is `None` and the two benchmarks are the same. `lp.enable_count` is still `1` on return, confirming the unbalanced enable. Everything the caller does after an `sc.mprofile()` call runs ~14x slower for the rest of the session, silently, and any later profiler that wants `sys.settrace` (including a second `sc.mprofile()`) fights with the leaked one. Because `tests/test_profiling.py` calls `sc.mprofile(big_fn)` before `test_profile()`, `test_cprofile()`, `test_tracecalls()` and `test_resourcemonitor()`, the leaked hook is live for the remainder of that test module. (The original audit said this was "very likely" the cause of the `TypeError` that `tests/test_profiling.py:92-94` catches; on review that is speculation, since the test's own comment, "This happens when re-running this script", points to double-wrapping instead.)

**Fix**: delete the `lp.enable_by_count()` call at line 820 (the wrapper returned by `lp(run)` already enables/disables around the call; verified on review that memory results are still recorded without it), or wrap `wrapper(*args, **kwargs)` in `try/finally: lp.disable_by_count()`.

### 4. `cprofile` caches its parsed stats forever, so reusing the object silently reports the previous run — `sc_profiling.py:928`

`parse_stats()` guards with `if self.parsed is None or force:`, and neither `start()` (1018) nor `stop()` (1022) nor `to_df()` invalidates `self.parsed`. Since the default `show=True` makes `stop()` call `disp()` -> `to_df()` -> `parse_stats()`, `self.parsed` is always populated after the first `stop()`, so a second `start()`/`stop()` on the same object re-displays the first run's numbers, with no warning.

```python
import time
import sciris as sc

def A(): time.sleep(0.1)
def B(): time.sleep(0.3)

c = sc.cprofile()
c.start(); A(); c.stop()
print('--- second run: B, 0.3 s ---')
c.start(); B(); c.stop()
print(c.to_df().func.tolist(), c.total)
c.parse_stats(force=True); print('true total:', c.total)
```

Actual (second run):

```
--- second run: B, 0.3 s ---
                           func   cumpct  selfpct   cumtime  selftime  calls     path
0                             A  99.9483   0.0044  100.2087    0.0044      1  v2.py:2
1  <built-in method time.sleep>  99.9439  99.9439  100.2043  100.2043      1        ~
Total time: 0.100261 s
['A', '<built-in method time.sleep>'] 0.10026051800000002
true total: 0.40048254400000005
```

Expected: `B` in the table and `Total time: 0.4 s` (`cProfile` accumulates across enable/disable pairs, so the correct cumulative answer is 0.4005 s, which only `parse_stats(force=True)` reveals). The function that was actually profiled second (`B`) never appears at all, and the reported total is 4x too low. This is a wrong number presented with no indication that anything is stale, from the exact `start()`/`stop()` usage the class docstring documents as "Option 2".

**Fix**: set `self.parsed = None` (and `self.df = None`) in `start()`, or drop the cache and always re-run `pstats.Stats(self.profile)` in `to_df()`.

### 5. `tracecalls.check_expected()` appends the `str` type instead of the supplied name, so string input never matches — `sc_profiling.py:1326`

Line 1326 reads `expected.append(str)` where it should be `expected.append(item)`. For the documented "list of strings" form, the resulting `expected` set contains the builtin `str` *class*, so the intersection with `self.func_names` is always empty and `not_called` always contains `<class 'str'>`.

```python
import sciris as sc

def a(): pass

with sc.tracecalls('chk.py', verbose=False) as tc:  # 'chk.py' = the name of this script
    a()
out = tc.check_expected(['a', 'nope'])
print('called    :', out.called)
print('not_called:', out.not_called)
print('set form  :', tc.check_expected({'a', 'nope'}))
```

Actual:

```
called    : set()
not_called: {<class 'str'>}
set form  : #0. 'called':     {'a'}
#1. 'not_called': {'nope'}
```

Expected: `called = {'a'}`, `not_called = {'nope'}` for both forms. Passing an actual `set` bypasses the loop entirely (`if not isinstance(expected, set)`) and works, which is why `tests/test_profiling.py:203` — which passes an object, not strings — does not catch this. With `die=True` the function raises "the following 1 functions were not called: {<class 'str'>}" for a run in which every named function *was* called, i.e. a false alarm in the one mode designed to be used as an assertion.

**Fix**: `expected.append(item)` at line 1326.

## Medium severity

### 6. `sc.checkmem()` on a numpy array (or any sequence of >1000 items) raises `RuntimeError` instead of reporting its size — `sc_profiling.py:127`, `143`

With the default `descend=1`, the `sc.isiterable(var, exclude=str)` branch at 127 treats a numpy array as a list-like and builds one entry per element along axis 0. For an array with more than `maxitems=1000` leading elements the `n_variables > maxitems` guard then aborts the whole call, so the single most obvious use of the function — "how big is this array?" — fails with default arguments.

```python
import numpy as np, sciris as sc
sc.checkmem(np.zeros(int(1e6)))
```

Actual:

```
RuntimeError: Cannot compute the sizes of 1000000 items since maxitems is set to 1000
```

Expected: a one-row (or summarised) dataframe; `sc.checkmem(np.zeros(int(1e6)), descend=0)` correctly reports `8.001 MB / 8000813` bytes against `arr.nbytes == 8000000` and `sys.getsizeof(arr) == 8000112`. The docstring's own `np.random.rand(2483,589)` only survives because it is wrapped in a list, so the array itself is reached at `descend=0`. Where the array is small enough not to trip the guard the result is not an error but is still wrong-headed: `sc.checkmem(np.zeros((5,10)))` reports five 263-byte rows and a `Total` of `1315` bytes for an array whose own pickled size is `588` bytes.

**Fix**: exclude `np.ndarray`, `pd.Series` and `pd.DataFrame` from the descend branches the way `str` already is, so they are measured as single objects; alternatively, fall back to `descend=0` with a warning instead of raising when `n_variables > maxitems`. (Noted on review: a `pandas.DataFrame` currently goes down the `hasattr(var, '__dict__')` branch and reports meaningless `_mgr`/`_flags`/`_attrs` rows, so the exclusion must cover the `__dict__` branch too, not only `isiterable`. Separately, the docstring says `**kwargs` are "passed to `sc.load()`", but `sc.load()` is never called, so misspelled keywords such as `decend=2` are silently swallowed.)

### 8. `subtotals=False` is not passed to the recursive `checkmem()` call, so nested subtotal rows appear anyway — `sc_profiling.py:152`

The recursion at 152 forwards `descend`, `compresslevel`, `maxitems`, `plot` and `verbose` but not `subtotals`, so every level below the first reverts to the default `subtotals=True`.

```python
import numpy as np, sciris as sc
nested = dict(foo=dict(a=np.zeros(3), b=np.zeros(4)), bar=dict(c=np.zeros(5), d=np.zeros(6)))
print(sc.checkmem(nested, descend=2, subtotals=False)[['variable','is_total']].to_string())
```

Actual:

```
      variable  is_total
0  bar (total)      True
1  foo (total)      True
2        bar→d     False
3        bar→c     False
4        foo→b     False
5        foo→a     False
```

Expected: only the four leaf rows. The result is also internally inconsistent — the top-level `Total` is correctly suppressed while the intermediate `(total)` rows are not — and the surviving `(total)` rows are indistinguishable from data rows unless the caller inspects `is_total`. (The same recursive call also hard-codes `verbose=False`, so `verbose` is dropped below the first level too.)

**Fix**: add `subtotals=subtotals` to the recursive call at 152 (and pass `verbose` through).

### 9. `profile.merge(inplace=False)` returns a half-initialised object; `p1 + p2` then breaks on `disp(skiprun=True)` — `sc_profiling.py:532`

`out = self.__class__.__new__(self.__class__)` creates a bare instance and `merge()` sets only `run_func`, `follow`, `follow_funcs`, `prof`, `total`, `output` and `df`. Nine attributes that `__init__` sets are never populated, including `run_func_name`, which `_get_entries()` needs.

```python
import sciris as sc
def f1():
    s=0
    for i in range(100000): s+=i
def f2():
    s=0
    for i in range(200000): s+=i
pm = sc.profile(f1, verbose=False) + sc.profile(f2, verbose=False)
print([a for a in ['skipzero','verbose','args','kwargs','run_func_name','private','include','exclude','unwrap'] if not hasattr(pm,a)])
pm.disp(skiprun=True)
```

Actual:

```
['skipzero', 'verbose', 'args', 'kwargs', 'run_func_name', 'private', 'include', 'exclude', 'unwrap']
AttributeError: 'profile' object has no attribute 'run_func_name'
```

Expected: a merged profile that supports the same methods as its operands. `merge(inplace=True)` / `+=` is unaffected because it reuses `self`. Time and percentage merging itself is correct: `pm.total == pa.total + pb.total` and the recomputed percentages summed to 98.3% for two disjoint profiles.

**Fix**: copy the remaining scalar attributes onto `out` (or build it with `__init__(..., do_run=False)`), and set `out.run_func_name` from whichever operand's name should win.

### 11. `tracecalls` crashes in `__exit__` when the trace filter matched nothing, masking the user's own exception — `sc_profiling.py:1289`

`stop()` unconditionally calls `to_df()`, which does `df['stack'] -= df['stack'].min()` on `sc.dataframe(self.entries)`. With no recorded entries the dataframe has no columns, so this raises `KeyNotFoundError`. Because `__exit__` runs `stop()` while an exception from the body is propagating, the user's exception is replaced by a confusing Sciris error.

```python
import sciris as sc

# A) no matches, clean body
try:
    with sc.tracecalls('no_such_module_xyz') as tc:
        sum(range(100))
    print('entries:', len(tc))
except Exception as e:
    print('A)', type(e).__name__, ':', e)

# B) no matches, and the body raises
try:
    with sc.tracecalls('no_such_module_xyz'):
        raise ValueError('my real error')
except ValueError as e:
    print('B) user exception survived:', e)
except Exception as e:
    print('B) MASKED by', type(e).__name__, ':', e)
```

Actual:

```
A) KeyNotFoundError : Key "stack" is not a valid column; choices are:
B) MASKED by KeyNotFoundError : Key "stack" is not a valid column; choices are:
```

Expected: A) `entries: 0`; B) `user exception survived: my real error`. A trace pattern that matches nothing is easy to hit by accident (for example a regex-mode pattern that does not start with `.*`, see finding 12, or `trace='mypackage'` when the installed path spells the package differently), and the failure mode destroys the traceback the user actually needed. `disp()` (1273) has the same problem one step later: `max([len(label) for label in ddf.label.values])` raises `ValueError: max() arg is an empty sequence` on an empty frame.

**Fix**: return early from `to_df()`/`stop()` when `self.entries` is empty (build an empty frame with the right columns), and guard `disp()`'s `max()` with a default.

### 14. `sc.listfuncs()` silently misses inherited methods, so `sc.profile(follow=SubClass)` skips half the class — `sc_profiling.py:1084`

`get_attrs()` uses `sc.objatt(parent, ..., return_keys=True)`, which goes through `_get_obj_keys(..., use_dir=False)` and therefore reads `parent.__dict__.keys()` — only the class's *own* namespace, so every method defined on a base class is dropped. The docstring says "If class(es) are supplied, search them for methods".

```python
import sciris as sc

class Base:
    def __init__(self): pass
    def inherited(self): pass

class Child(Base):
    def own(self): pass

print('listfuncs(Child)  :', [f.__name__ for f in sc.listfuncs(Child)])
print('listfuncs(Child()):', [f.__name__ for f in sc.listfuncs(Child())])
print('listfuncs(Base)   :', [f.__name__ for f in sc.listfuncs(Base)])
```

Actual:

```
listfuncs(Child)  : ['own']
listfuncs(Child()): ['own']
listfuncs(Base)   : ['__init__', 'inherited']
```

Expected: `['__init__', 'inherited', 'own']` for `Child`. Note the inconsistency this creates: `Base` yields `__init__` (because of the `private='__init__'` default) but `Child` does not, purely because `Child` does not redefine it. `listfuncs()` is what `sc.profile()` uses for its `follow` argument (line 447), so `sc.profile(run=obj.run, follow=type(obj))` on any subclass silently profiles only the leaf class's methods and reports nothing for the inherited ones — indistinguishable from those methods not being called. Properties are also skipped (arguably by design), while `staticmethod`, `classmethod` and `functools.wraps`-decorated methods are picked up correctly.

**Caveat on review**: `check_expected()` (line 1332) uses the same `__dict__`-only convention, so this was probably a deliberate choice rather than an oversight, and it is a borderline design issue. It is kept because the docstring promises to "search them for methods" and the failure is silent.

**Fix**: do *not* simply switch to `dir(parent)`: that would change behaviour for every existing `follow=<class>` call by pulling in all base-class methods, including everything from `sc.prettyobj`/`sc.quickobj` bases. Instead, when `parent` is a class, walk `parent.__mro__` excluding `object` (and perhaps Sciris base classes), de-duplicating by name so overrides win, or add an opt-in `inherited=True` option.

### 16. `cprofile.to_df(maxitems=...)` is computed and then ignored — `sc_profiling.py:998`

Line 945 resolves the local `maxitems = sc.ifelse(maxitems, self.maxitems)`, but line 998 slices with the attribute: `self.df = self.df[:self.maxitems]`. Every other resolved local (`sort`, `mintime`, `maxfunclen`, `maxpathlen`, `columns`) is used; this one is not, so the documented per-call override — and `cpr.disp(maxitems=10)`, which forwards to `to_df()` — does nothing.

```python
import time
import sciris as sc

def f1(): time.sleep(0.05)
def f2(): time.sleep(0.05)
def f3(): time.sleep(0.05)

cpr = sc.cprofile(show=False)
cpr.start(); f1(); f2(); f3(); cpr.stop()
print('to_df()           rows:', len(cpr.to_df()))
print('to_df(maxitems=2) rows:', len(cpr.to_df(maxitems=2)))
```

Actual:

```
to_df()           rows: 4
to_df(maxitems=2) rows: 4
```

Expected: `2` for the second call. Setting it on the constructor (`sc.cprofile(maxitems=2)`) does work — verified separately, 3 rows for `maxitems=3` out of 7 — which is why this is easy to miss.

**Fix**: `self.df = self.df[:maxitems]` at line 998.

### 17. `resourcemonitor.kill()` enumerates every process on the system before interrupting the main thread, so a busy job is interrupted late or not until the body ends — `sc_profiling.py:1570`

`kill()` calls `children = parent.children(recursive=True)` unconditionally (even when `kill_children=False`) and only reaches `_thread.interrupt_main()` at line 1584, after the enumeration. `psutil.Process.children(recursive=True)` is mostly Python (it walks every PID on the box), so when the main thread is holding the GIL in a compute loop the monitor thread makes slow progress and the interrupt is delayed, often until after the runaway work has finished, in which case the `LimitExceeded` surfaces only from `stop()` (line 1472) — the opposite of "terminate the process if the specified threshold is exceeded".

```python
import time
import sciris as sc

t0 = time.time()
try:
    with sc.resourcemonitor(mem=0.0001, interval=0.05, verbose=False) as rm:
        x = 0
        while time.time()-t0 < 2.0: x += 1   # limit is breached at ~0.05 s
    print('body FINISHED after %.2f s' % (time.time()-t0))
except BaseException as e:
    print('interrupted at %.2f s by %s' % (time.time()-t0, type(e).__name__))
```

Actual:

```
interrupted at 2.00 s by LimitExceeded
```

Instrumenting `kill()` shows it is entered at 0.163 s and does not return until the main loop ends. Measured in isolation: `psutil.Process(os.getpid()).children(recursive=True)` takes 0.006 s on an idle interpreter (619 PIDs on this box) but 3.005 s when a main-thread Python loop is running, i.e. it does not complete until the main thread yields. Moving `_thread.interrupt_main()` to the front of `kill()` (everything else unchanged) fixes it:

```python
class RM(sc.resourcemonitor):
    def kill(self):
        _thread.interrupt_main()
        parent = psutil.Process(self.parent)
        children = parent.children(recursive=True)
# -> "loop interrupted after 0.153 s by LimitExceeded"
```

A control experiment confirms the signal machinery itself is fine: `threading.Timer(0.2, _thread.interrupt_main)` does interrupt a busy loop promptly, and firing it by hand against a live `resourcemonitor` raises `LimitExceeded` at 0.21 s.

**Correction on review**: the original title said a busy job is "never actually interrupted". Re-verification with `psutil.Process.children` instrumented across repeated busy-loop runs gave enumeration times of 0.05 s, 0.22 s, 0.46 s, 1.15 s and 2.12 s, and in some runs it did not finish before the 3 s body ended. So the interrupt is delayed by a highly variable amount rather than never delivered. Moving `_thread.interrupt_main()` to the top of `kill()` gave an interrupt at 0.18 s. Note also that even with the fix, a main thread blocked in `time.sleep()` or a C call is not interrupted until that call returns, because `_thread.interrupt_main()` only schedules the Python-level handler rather than delivering a real signal (verified: `time.sleep(3)` is interrupted at 3.00 s with or without the fix).

**Fix**: move `_thread.interrupt_main()` to the start of `kill()` (before any psutil work), and move `children = parent.children(recursive=True)` inside `if self.kill_children:` so it is not paid for when it is not used. Do not expect this to interrupt blocking calls.

### 18. `resourcemonitor`'s SIGINT handler calls the original handler with no arguments, so Ctrl-C raises `TypeError` — `sc_profiling.py:1441`

`start()` installs a handler that, when no limit has been breached, does `return self._orig_sigint()`. A signal handler must be called as `handler(signum, frame)`, so the fall-through path raises `TypeError` instead of the `KeyboardInterrupt` the user asked for. Worse, `signal.getsignal()` can legitimately return `signal.SIG_DFL`/`SIG_IGN` (the ints `0`/`1`), which are not callable at all.

```python
import os, signal, time
import sciris as sc

rm = sc.resourcemonitor(mem=0.99, interval=10.0, die=False, verbose=False)  # no breach
try:
    os.kill(os.getpid(), signal.SIGINT)   # i.e. Ctrl-C
    time.sleep(0.3)
    print('nothing raised')
except KeyboardInterrupt:
    print('KeyboardInterrupt (correct)')
except BaseException as e:
    print('***', type(e).__name__, ':', e)
rm.stop()
```

Actual:

```
*** TypeError : default_int_handler expected 2 arguments, got 0
```

Expected: `KeyboardInterrupt (correct)`. A second run after `signal.signal(signal.SIGINT, signal.SIG_DFL)` shows `rm._orig_sigint` is `0` and `callable(rm._orig_sigint)` is `False`, so that path would raise `TypeError: 'Handlers' object is not callable`. The practical effect: while any `sc.resourcemonitor` is running (the class starts itself on construction by default), Ctrl-C on a long job produces a `TypeError` traceback rather than a clean interrupt.

**Fix**: `return self._orig_sigint(signum, frame)`, guarded by `callable(self._orig_sigint)` (falling back to `raise KeyboardInterrupt` for `SIG_DFL` and to returning `None` for `SIG_IGN`).

### 19. `resourcemonitor(kill_parent=True)` raises `AttributeError` on a non-existent `self.parent_pid` and so never interrupts the main thread — `sc_profiling.py:1580`

`__init__` stores the PID as `self.parent` (line 1419); line 1580 prints `self.parent_pid`, which is never assigned anywhere in the file. The print sits inside `if kill_verbose:` where `kill_verbose = self.verbose is not False`, i.e. it fires for the default `verbose=None` and for `verbose=True`, and it executes *before* `parent.kill()` and before `_thread.interrupt_main()`. The monitor thread therefore dies with a traceback and the documented behaviour ("whether to also kill the parent process") does not happen, nor does the main thread get interrupted.

```python
import time
import sciris as sc

t0 = time.time()
try:
    with sc.resourcemonitor(mem=0.0001, interval=0.05, kill_parent=True) as rm:  # verbose=None
        time.sleep(1.5)
    print('body completed after %.2f s' % (time.time()-t0))
except BaseException as e:
    print('interrupted by', type(e).__name__, e)
```

Actual:

```
Exception in thread Thread-1 (monitor):
Traceback (most recent call last):
  ...
  File "/home/cliffk/sc/sciris/sciris/sc_profiling.py", line 1580, in kill
    print(f'Killing parent (PID={self.parent_pid})')
                                 ^^^^^^^^^^^^^^^
AttributeError: 'resourcemonitor' object has no attribute 'parent_pid'
...
interrupted by LimitExceeded Limits exceeded: Memory: 0.34 vs 0.00
```

The `LimitExceeded` at the end comes from `stop()`, not from the kill: the process survives the breach, runs the whole 1.5 s body, and dies only on leaving the `with` block. `hasattr(sc.resourcemonitor(start=False, kill_parent=True), 'parent_pid')` is `False`. The bug is masked by `verbose=False`, which skips the print and lets the real `parent.kill()` run (verified: that variant is SIGKILLed, exit code 137).

**Fix**: `print(f'Killing parent (PID={self.parent})')` at line 1580.

### 31. `checkmem(plot=True)` pie chart includes the `Total` row, so every real slice is halved — `sc_profiling.py:171-173`

*Added on review.* `plt.pie(df.bytesize, labels=df.variable, autopct='%0.2f')` plots every row of `df`, including the `is_total` rows appended at line 160. The `Total` slice is therefore always exactly 50% of the pie, and every real item's percentage is half its true value. With `descend>1`, the nested `(total)` rows are plotted too, which double-counts further.

```python
import numpy as np, sciris as sc, matplotlib.pyplot as plt
sc.checkmem(dict(a=np.zeros(1000), b=np.zeros(3000)), plot=True)
print([t.get_text() for t in plt.gca().texts])
```

Actual: `['Total', 'b', 'a', '50.00', '37.36', '12.64']`. Expected: two slices, `b` about 74.7% and `a` about 25.3%.

**Fix**: `pdf = df[~df.is_total]; plt.pie(pdf.bytesize, labels=pdf.variable, autopct='%0.2f')`.

### 32. `checkmem()` mislabels and crashes on dicts with non-string keys — `sc_profiling.py:121`, `136`, `151`, `157`

*Added on review.* `varnames = list(var.keys())` keeps the raw keys, which are later used as string labels. At `descend=1`, key `0` is relabelled `'Variable'`, because the recursion uses `_prefix if _prefix else 'Variable'` and `0` is falsy; at `descend>=2`, `_join.join([_prefix, varname])` raises `TypeError` for any int key; and `_prefix + ' (total)'` would also fail for int prefixes. Int-keyed dicts (e.g. results keyed by year or index) are ordinary in-contract input.

```python
import numpy as np, sciris as sc
print(sc.checkmem({0: np.zeros(3), 1: np.zeros(4)}).variable.tolist())
sc.checkmem({'x': {0: np.zeros(3), 1: np.zeros(4)}}, descend=2)
```

Actual: `['Total', 1, 'Variable']` (the `0` key is lost as `'Variable'`), then `TypeError: sequence item 1: expected str instance, int found`. Expected: rows labelled `0`, `1`, and `x→0`, `x→1`.

**Fix**: `varnames = [str(k) for k in var.keys()]` at line 121. This also makes `'0'` truthy, which fixes the `'Variable'` relabelling.

## Low severity

### 7. `maxentries` is silently ignored when `bytime=0`, so `disp()`/`to_df()` dump every function — `sc_profiling.py:709`

`_get_entries()` truncates with `if bytime == 1: entries = entries[-maxentries:]` / `elif bytime == -1: entries = entries[:maxentries]`. `bytime=0` is a documented value ("if 0, do not sort by time") and falls through both branches, so no truncation happens at all and the `maxentries` argument does nothing.

```python
import sciris as sc, sciris.sc_math as scm
p = sc.profile(run=lambda: scm.findinds([1,2,3],2), follow=scm, verbose=False)
print(len(p.output), len(p.to_df(0, 5)), len(p.to_df(1, 5)))
```

Actual:

```
34 34 5
```

Expected `34 5 5`. `p.disp(bytime=0, maxentries=5)` likewise prints all 34 full line-profiler blocks (counted 34 `Profile of` headings) instead of 5. Related and worth fixing at the same time: on the working path, `disp()`'s default `maxentries=10` drops the remaining entries with no indication that anything was omitted, even though `run()` has just printed `Profiling 34 function(s)`.

*Severity lowered from Medium to Low on review*: `bytime=0` is a non-default option and the only effect is too much output.

**Fix**: hoist the truncation out of the `bytime` conditional (`entries = entries[-maxentries:] if bytime >= 0 else entries[:maxentries]`, or slice after re-sorting), and print a "showing N of M" note when entries are dropped.

### 12. `tracecalls` class docstring's regex example is an invalid regex, and its `exclude` could never match — `sc_profiling.py:1142`

*Rewritten on review.* The original finding claimed that anchoring regex filters with `re.match` (line 1246) against the absolute `filename + '_' + name` is a silent bug. It is not: the regex-mode defaults at lines 1155-1156 are `default_trace = '.*'` and `default_exclude = {'.*<', '.*tracecalls'}`, which shows the author deliberately wrote regex patterns to be matched from the start of the absolute path. The genuine defect is the docstring example:

```python
import sciris as sc
sc.tracecalls('*mysubmodule*', exclude='^init*', regex=True, repeats=True).start()
```

Actual: `re.PatternError: nothing to repeat at position 0` (re-verified). Expected: a working example. Even if the trace pattern compiled, `exclude='^init*'` could never match, because `full_name` starts with an absolute directory path, not `init`.

**Fix**: rewrite the example to a valid pattern consistent with the anchored semantics, e.g. `sc.tracecalls('.*mysubmodule', exclude='.*__init__', regex=True, repeats=True)`, and state in the `regex` docstring that patterns are matched with `re.match` against `<absolute path>_<function name>` (so they normally start with `.*`). Optionally, switching `_check()` to `re.search()` would be backward compatible (anything `re.match` finds, `re.search` also finds) and would make regex and substring modes behave alike, but it is a usability improvement, not a bug fix.

### 20. `resourcemonitor` with `die=False` is completely silent on a breach, contradicting its documented default verbosity — `sc_profiling.py:1385`

The docstring says `verbose (bool): detail to print out (default: if exceeded; True: every step; False: no output)`. In `monitor()` the only per-step output is inside `if self.verbose:` (line 1491), and the only breach output lives in `kill()` (line 1564, `kill_verbose = self.verbose is not False`). With `die=False`, `kill()` is never called, so with the documented default `verbose=None` a breach prints nothing, raises nothing, and can only be discovered by inspecting `resmon.exception`.

```python
import time
import sciris as sc

print('=== die=False, verbose=None (documented default: print "if exceeded") ===')
rm = sc.resourcemonitor(mem=0.0001, interval=0.05, die=False)
time.sleep(0.3)
rm.stop()
print('--- end of captured region ---')
print('breach recorded internally:', repr(rm.exception))
```

Actual:

```
=== die=False, verbose=None (documented default: print "if exceeded") ===
--- end of captured region ---
breach recorded internally: LimitExceeded('Limits exceeded: Memory: 0.34 vs 0.00')
```

Expected: a `Limits exceeded: ...` line printed when the limit was exceeded. `die=False` is exactly the mode in which the user is relying on being told, and it is the mode the docstring's own second example uses (`sc.resourcemonitor(mem=0.95, cpu=0.9, time=3600, ..., die=False, ...)`).

*Severity lowered to Low on review*: the behaviour contradicts the docstring, but nothing crashes and the breach is still recorded in `resmon.exception`.

**Fix**: in `monitor()`, print `checkstr` on the `not is_ok` branch when `self.verbose is not False`, independently of `self.die`. Make sure this does not double-print when `die=True`, since `kill()` also prints the exception.

### 33. `cprofile(columns='full')`: `percall` stays in seconds while `cumtime`/`selftime` are converted to ms — `sc_profiling.py:1001-1004`

*Added on review.* When the automatic ms conversion is applied (`use_ms=None` and all `cumtime` < 1 s, which is documented behaviour), the loop only rescales `['cumtime', 'selftime']`. In the `full` column set, `percall` is shown beside them still in seconds, so the table is internally inconsistent: `percall` × `calls` ≠ `cumtime`.

```python
import time, sciris as sc
def w(): time.sleep(0.05)
c = sc.cprofile(show=False, columns='full')
c.start(); w(); w(); c.stop()
print(c.to_df()[['func','calls','percall','cumtime']])
```

Actual: `w  calls=2  percall=0.050160  cumtime=100.320575`. Expected: `percall` about `50.16` (ms), consistent with `cumtime`.

**Fix**: `for col in ['cumtime', 'selftime', 'percall']: if col in self.df.columns: self.df[col] *= 1000`.

### 34. `resourcemonitor.start(label=...)` and `monitor(label=...)` ignore the documented `label` argument — `sc_profiling.py:1428`, `1486`

*Added on review.* `start()` documents `label (str): optional label for printing progress`, but the parameter is never used, and `monitor()`'s `label` parameter is not used either. All printing uses `self.label` (set only in `__init__`, line 1408), so `rm.start(label='Phase 2')` has no effect: progress lines still read `Monitor step N: ...` (or whatever label was given to the constructor).

```python
import time, sciris as sc
rm = sc.resourcemonitor(start=False, verbose=True, interval=0.1)
rm.start(label='Phase 2'); time.sleep(0.25); rm.stop()
```

Actual: output lines begin `Monitor step 1: ...` and end `Monitor: done`. Expected: `Phase 2 step 1: ...`.

**Fix**: `if label is not None: self.label = label` at the top of `start()`, and either use or remove the unused `label` parameter of `monitor()`.

## Rejected on review

These original findings were removed from the summary and severity sections during the 2026-09-25 re-verification. Their numbers are retired, not reused.

- **10. `benchmark(which=...)` raises a bare `KeyError` for anything but the exact literals** — NOT WORTH FIXING: only malformed strings such as `' numpy'` or `'numpy only'` hit the `KeyError`; all documented values (`'python'`, `'numpy'`, `'python, numpy'`) work (deriving the key from the flags is a harmless optional hardening).
- **13. `tracecalls.stop()` clears the profile hook instead of restoring the previous one** — NOT WORTH FIXING: reproduced, but it only matters when `tracecalls` blocks are nested or another `sys.setprofile` tool is live, which is rare; the two-line save/restore fix is safe to apply opportunistically.
- **15. `listfuncs()` `include`/`exclude` match `str(f)`** — NOT WORTH FIXING: the repr contains the qualname, so realistic filters by function or class name (`include='find'`, `include='inner'`) work; the failing patterns (`'sc_math'`, `'0x'`, `'function'`) are contrived.
- **21. `checkmem()` docstring example 1 is a syntax error** — NOT WORTH FIXING: a stray `)` in a docstring, not a code bug; fix in passing.
- **22. `checkmem(order=...)` silently ignores unrecognised values** — NOT WORTH FIXING: unsorted output for `order='SIZE'` or a typo is a minor input-validation nicety.
- **23. `checkmem()` totals do not equal the object's own size** — NOT A BUG: per-item pickle overhead is an inherent, expected consequence of the documented "by dumping them to file" method.
- **24. `private=False` does not exclude single-underscore methods** — NOT WORTH FIXING: throughout Sciris (`sc.objatt()` and others) "private" means dunder; at most a docstring wording change.
- **25. `run()`'s `orig_func` save/restore is dead code** — NOT WORTH FIXING: dead code with a misleading comment; style only.
- **26. `cprofile` docstring gives the wrong default `sort`** — NOT WORTH FIXING: the docstring typo (`"cumpct"` should be `"cumtime"`) is documentation only, and a `KeyError` when sorting by a column you excluded is a reasonable error.
- **27. `cprofile.disp()` blames `mintime` for rows `maxitems` dropped** — NOT WORTH FIXING: a cosmetic message only; no numbers are wrong.
- **28. `cprofile` reports milliseconds in columns named `cumtime`/`selftime`** — NOT A BUG: the ms conversion is documented behaviour of `use_ms` (default: rescale if all durations < 1 s); the missing unit label is cosmetic (the related real inconsistency is finding 33).
- **29. `cprofile`'s `trim()` returns strings three characters too long** — NOT WORTH FIXING: trimmed names are 43 characters instead of 40; cosmetic.
- **30. `tracecalls` docstring refers to a non-existent `sys.steprofile()`** — NOT WORTH FIXING: a docstring typo; fix in passing.

## Misplaced `# pragma: no cover`

| Line | Branch | Reachable via |
|------|--------|---------------|
| 143 | `if n_variables > maxitems:` in `checkmem()` | `sc.checkmem(np.zeros(int(1e6)))` with all-default arguments (finding 6) |
| 166 | `if order == 'alphabetical':` in `checkmem()` | `sc.checkmem(nested_dict, order='alphabetical')` -> `['Total','bar','cat','foo']`, a documented value of a documented argument |
| 208 | `else:` (the `to_string=False` return) in `checkram()` | `sc.checkram(to_string=False)` — used by `checkram()`'s own docstring example, which runs correctly |
| 328 | `if return_timers:` in `benchmark()` | `sc.benchmark(repeats=2, scale=0.1, return_timers=True)` -> `objdict(['python','numpy'])`, a documented argument |
| 445 | `if self.follow is None:` in `parse_follow()` | `sc.profile(slow_fn)` — the default and the class's first docstring example; `follow_funcs` comes back as `[<function slow_fn>]` from exactly this branch |
| 813 | `if follow is None:` in `mprofile()` | `sc.mprofile(fn)` — the documented one-argument call, and the form used in `tests/test_profiling.py:92` — takes this branch every time |
| 1436 | `def handler(signum, frame):` in `resourcemonitor.start()` | Installed unconditionally by `start()` and invoked on any Ctrl-C or `_thread.interrupt_main()` while a monitor is live (finding 18) |
| 1472 | `raise self.exception` in `resourcemonitor.stop()` | With the default `die=True`, this is the line that actually surfaces `LimitExceeded` whenever the main thread was not interrupted in time (findings 17, 19) — not an unlikely fallback, it is the common path for a CPU-bound job |
| 1501 | `if self.die:` in `resourcemonitor.monitor()` | `die=True` is the default, so any breach under default settings takes this branch |
| 1562 | `def kill(self):` | Called from line 1502 on every breach with the default `die=True`; verified printing "Killing processes..." and killing the process (exit 137) in the `kill_parent`/`verbose=False` variant |

The pragmas at lines 492 (`wrapper = prof(self.run_func)`) and 496 (`wrapper(*self.args, **self.kwargs)`) sit on lines that certainly execute, but coverage cannot instrument code running under another trace/monitoring tool, so those two look legitimate rather than misplaced. The class-level `# pragma: no cover` on `resourcemonitor` (1369) is annotated as a coverage-tool workaround and is out of scope, but it is what hides most of the entries above; the `mprofile()` pragmas at 804 and 823 are genuinely hard to reach (missing package, double-wrapping) and are fine.

## Verified clean

**`checkram()`**: the unit arithmetic is correct and self-consistent — with `psutil` rss of `142938112` bytes the function returns `142938112.00 B`, `142938.11 KB`, `142.94 MB`, `0.14 GB`, i.e. the SI 1000 convention applied exactly once and matching the label in every case, agreeing with `sc.humanize_bytes(rss) == '142.938 MB'` (also 1000-based, so no double conversion anywhere in the module). The docstring example runs verbatim and reports `80.18 MB` for a `(1000, 10000)` float64 array whose `nbytes` is `80000000` — correct. `fmt` is honoured; `unit` is case-insensitive and an unknown unit raises a helpful `sc.KeyNotFoundError`. The only wart is that `start` is silently assumed to be in the same unit as the current call (`sc.checkram(unit='kb', start=sc.checkram(to_string=False))` mixes MB and KB), but `start` is undocumented and this is user error rather than a defect.

**`checkmem()`**: `descend=0`, `descend=1` and `descend=2` all run; both the `dict` and the `hasattr(var,'__dict__')` branches work (`sc.prettyobj` as in `tests/test_profiling.py:37`); strings are correctly excluded from the iterable branch; `_prefix`/`_join` build the expected `bar→item 2` paths; the `Total` row is computed from `df[np.logical_not(df.is_total)]` and so correctly sums only leaves, never double-counting nested subtotals; per-item sizes sum *exactly* to the reported total at every depth tested; `order='size'` really does sort by `bytesize` descending and `order='alphabetical'` really does sort by name; `maxitems` truncation is loud (a `RuntimeError`), never silent; `plot=True` is correctly suppressed in the recursion so only the top level draws; `compresslevel=0` produces sizes within ~0.01% of `nbytes` for large arrays; and the temporary files created by `check_one_object()` are removed on the success path. (Re-verification on 2026-09-25 found two `checkmem()` defects this paragraph missed: the `plot=True` pie chart includes the `Total` rows, finding 31, and non-string dict keys are mislabelled or crash, finding 32.)

**`benchmark()`**: normalisation is correct: MOPS is per-second and stays flat as `scale` doubles (`scale=0.25/0.5/1/2/4` gave numpy `203.9/210.2/235.4/234.2/235.7` and python `20.9/28.4/11.2/26.3/25.0`; the python spread is run-to-run noise at `repeats=2`, with no monotonic trend and certainly no factor-of-2 drift), so neither `py_ops = 10*scale*1e3*18/1e6` nor `np_ops = scale*1e6*4/1e6` is mis-scaled — this confirms flat per-second rates across `scale` are the intended and correct behavior. `return_timers=True` agrees with the printed numbers to within run-to-run noise: `py_ops/T.python.mean() = 34.99` vs a reported `34.94`, and `np_ops/T.numpy.mean() = 296.3` vs `294.8`; with a single-test `which` it correctly returns the bare `sc.timer`. `parallel=True`/`parallel=2` work (`sum(Plist)` over timers, `*ncpus` for aggregate throughput, self-correcting for contention because worker times enter the mean), `parallel='x'` raises the documented `ValueError`, and `which='np'`/`'both'`/`''` all raise the intended `ValueError` rather than doing something wrong. All three docstring examples run verbatim, and `bm['numpy'] > bm['python']` (the assertion in `tests/test_profiling.py:53`) held on every run.

**`sc.profile`**: `self.total` IS the right denominator for the reported percentages — it is the wall-clock duration of the `wrapper(...)` call (measured by the `sc.timer` context at 495), and for a self-contained run function line_profiler's own "Total time" for that function matches it to within 0.1% (`outer` reported `0.171700` s against `self.total = 0.171861`, i.e. 99.91%), so `sc.safedivide(time, self.total, np.nan)*100` at 636 is "% of the profiled call's wall time" and is right. The per-function percentages sum to more than 100% for nested code (`inner` 79.3% + `outer` 99.9% = 179.2%), but that is inherent nested-call accounting — a caller's line time includes its callees' time — not a normalisation bug, and 100% is correctly *not* forced anywhere. `skipzero=True` drops exactly the zero-time entries and nothing else (`__init__`/`_hidden` at 0.0 s removed, `inner`/`outer` retained), and `skipzero=False` (the default) keeps them, so nothing is silently dropped by default. `sort(bytime=1/-1/0)` orders by increasing time, decreasing time, and original discovery order respectively, and `_get_entries()` correctly slices `[-maxentries:]` for `bytime=1` and `[:maxentries]` for `bytime=-1` (the `bytime=0` gap is finding 7); `sort(copy=True)` does not disturb `self.output`. `follow=` was exercised for all six documented shapes and nothing was silently ignored: a bare function, `follow=None` (falls back to the run function), a list of bound methods, a class, a class instance, a single bound method, `Foo.__init__`, and a whole module (`sciris.sc_math` -> 34 functions, 34 output entries). The run function is always profiled in addition to the `follow` targets, so `follow=foo.inner` still yields both `inner` and `outer`. `include='inner'` and `exclude='inner'` both filter as documented; `unwrap=True` follows `__wrapped__` through a `functools.wraps` decorator while `unwrap=False` profiles the wrapper (line_profiler emits its own advisory `UserWarning` in the `unwrap=False` case, which is informative, not a bug). `_path_to_name()` resolves methods to `module.Class.method`, falls back to `module.py:LINE` for lambdas, and `sc.uniquename()` de-duplicates collisions. On the *normal* path `run()` does clean up after itself — `sys.monitoring.get_tool(2)` is `None` after a successful `sc.profile()` — so the leak in finding 1 really is confined to the exception path. `merge()`'s numeric behaviour is correct (`total` is the sum, percentages are recomputed against the merged total, zero-time entries do not overwrite non-zero ones, `order` is offset by `len(self.output)`), and `merge(inplace=True)`/`+=` is unaffected by the missing-attribute problem in finding 9. All three of the class's docstring examples run verbatim and produce sensible line-by-line output.

**`cprofile` numerical correctness**: built a call tree with a known 0.30 s sleep leaf and a ~0.065 s compute leaf under `middle()` under `top()`, and compared `sc.cprofile`'s dataframe against `pstats.Stats(cpr.profile).strip_dirs()` on the same profile object: `cumtime` for `top`/`middle` (0.3657 s), `leaf_sleep` (0.3002 s), `leaf_compute` (0.0655 s) and `<built-in method time.sleep>` (0.3002 s) all matched pstats exactly, `selftime` was correctly ~0 for the pass-through frames and 0.0655 for the compute leaf, `calls` matched, and `cumpct = cumtime/total_tt*100` gave the expected 99.99 / 82.08 / 17.91 split — i.e. `cprofile`'s numerical attribution matches stdlib `pstats` exactly. `percall = cumtime/calls`, `selfpct`, and the `path` column (`file:line`, `~` for builtins) were all correct. `mintime` filters on `cumtime` and therefore cannot hide a function with a large `selftime` (`cumtime >= selftime` always); `mintime=0.2` correctly kept exactly the two rows above the threshold; `mintime=1e6` produced an empty dataframe and an empty `disp()` without crashing (no `max()`-of-empty problem, unlike `tracecalls.disp()`). `columns='brief'`/`'default'`/`'full'` produce exactly the documented column sets. `calls` uses `entry[1]` (total calls including recursive), which is the same number pstats prints as `ncalls`. `sortrows` ordering is descending for numeric sort keys and ascending for string keys, matching the code's `reverse = sc.isarray(...)` intent.

**`cprofile` cleanup and nesting**: an exception raised inside a `with sc.cprofile()` block leaves no active profiler (a fresh `cProfile.Profile().enable()` immediately afterwards succeeds), i.e. `__exit__` -> `stop()` -> `disable()` runs on the exception path. Nesting two `sc.cprofile()` blocks raises `ValueError: Another profiling tool is already active` from the inner `start()`, and the outer profiler is still correctly disabled by its own `__exit__` afterwards — no leaked global profiler state in any of these paths. `stop()` disables before it displays, so a failure inside `disp()` cannot leave the profiler running.

**`tracecalls` hook removal and depth/count accuracy**: `sys.getprofile()` is `None` after a clean exit, after an exception in the body, and after an exit in which the filter matched nothing (i.e. even the `to_df()` crash of finding 11 happens after the hook is removed) — there is no leaked-hook bug, only the clobbering described in rejected finding 13. `sys.gettrace()` is `None` throughout, because the class uses `sys.setprofile`, not `sys.settrace`. Recorded depths and counts are right: for `fact(5)` the five recursive frames were recorded at relative stacks 1,2,3,4,5 with `count=5` in `df_counts`; for a mutually recursive `is_even(6)`/`is_odd` pair the seven frames alternated correctly at stacks 1..7 with `is_even` counted 4 times and `is_odd` 3 times; the shared `self.entry.stack` counter stays balanced because `c_call`/`c_return` events fall through to the no-op branch and a Python frame that raises still emits a `return` event. The non-regex `trace`/`exclude` filters behave exactly as documented (substring match on `filename + '_' + name`), `exclude=None` disables exclusion (`sc.tolist(None)` gives `[]`), and `repeats=False` deduplication by `filename+lineno` behaves as documented. `check_expected()` works correctly when given a class, an instance, or a `set` of strings (only the list/single-string path in finding 5 is broken).

**`resourcemonitor` limits and thread lifecycle**: all three limits fire: `time=0.2` raised `LimitExceeded('... Time: 0.303 vs 0.2')`, `mem=0.0001` raised on the first check, and `cpu=0.0001` raised under a busy loop. The monitor thread is reliably stopped: `thread.is_alive()` was `False` after a clean `with`-block exit and after an exception in the body (`stop()` runs from `__exit__` in both cases); it stays alive only if the user constructs a monitor and never calls `stop()`, which the docstring explicitly warns about, and even then it is a daemon thread. `signal.getsignal(signal.SIGINT)` is restored to its original value by `stop()`. `check()` computes `ok`/`ratio` correctly for all three limits, `to_df()` flattens the log correctly, and the `datalist` suppression logic (`if self.cpu < 1`, `if self.mem < 1`, `if np.isfinite(self.time)`) matches the `x if x else <no limit>` normalisation in `__init__`.

**`LimitExceeded` and `except Exception` — not a bug**: `LimitExceeded` IS caught by a plain `except Exception:`, because its multiple inheritance resolves to `MRO = [LimitExceeded, MemoryError, Exception, KeyboardInterrupt, BaseException, object]`, so `issubclass(sc.LimitExceeded, Exception)` is `True` and a plain `except Exception:` in user code does catch it (verified by raising it and catching it with `except Exception`). There is no contradiction with the docstring, and no "escapes `except Exception` because it is a `KeyboardInterrupt`" hazard. The converse is worth knowing but is a design consequence, not a defect: because it is an `Exception`, a worker function with a blanket `except Exception` will swallow a resource-limit breach.

**`listfuncs` filters and `private`**: `private=False`/`True`/`'__init__'`/`['__init__','__repr__']` all behave as designed against Sciris's own definition of "private" (dunder, per `sc_printing._get_obj_keys`): `private=True` adds every dunder method in the class's `__dict__`, a string or list adds exactly the named ones, and single-underscore methods are always included in every mode (this is the same root cause as rejected finding 24). `staticmethod`, `classmethod` and `functools.wraps`-decorated methods are all found; `property` objects are excluded (`sc.isfunc` is `False` for them), which is consistent with them not being profilable functions. `strict=True` raises `TypeError` for a class or module, which is what "raise an exception if something is not a function, rather than recurse into it" says. Modules are searched one level deep only — `sc.listfuncs(numpy)` finds 110 top-level functions and nothing from `numpy.linalg` — which reads oddly against "recursively search them for functions and classes" but is consistent with the intended module -> class -> method recursion, so it is reported here rather than as a finding.

## Suggested order of work

1. **Findings 1 and 3** — the leaked global hooks. Both are small, local changes (a `try`/`finally` with a *single* disable call for 1; deleting one line for 3), and together they are the fixes with the most outsized benefit relative to effort: they stop one profiling call from silently degrading or breaking every later profiling call or test in the same process.
2. **Findings 2, 4, 5** — the remaining High-severity findings: `plot()`'s read-only crash on the common case, `cprofile`'s stale-cache reuse, and `tracecalls.check_expected()`'s dead string comparison. All three either crash on ordinary use or silently report a wrong answer with no indication anything is stale.
3. **Medium findings (6, 8, 9, 11, 14, 16, 17, 18, 19, 31, 32)** — crashes on in-contract input (numpy arrays and int-keyed dicts into `checkmem`, empty `tracecalls` results, resourcemonitor's SIGINT/attribute bugs), silently-ignored documented arguments (`subtotals`, `maxitems`, inherited methods), and a misleading `checkmem` pie chart.
4. **Low findings (7, 12, 20, 33, 34) and the pragma list** — `maxentries` with `bytime=0`, the broken docstring regex example, `die=False` silence, the `percall` unit mismatch, and the ignored `label`. The docstring typos among the rejected findings (21, 26, 30) are worth fixing in passing while editing the file.

Splitting by symptom rather than severity: findings 4 (stale `cprofile` cache), 5 (`check_expected` false alarms), 7 (`maxentries` ignored), 8 (`subtotals` ignored), 14 (`listfuncs` missing inherited methods), 16 (`cprofile.to_df(maxitems=...)` ignored), 20 (silent breach), 31 (halved pie slices), 32 (key `0` relabelled `'Variable'`), 33 (`percall` in seconds beside ms columns) and 34 (`label` ignored) all report a plausible-looking but wrong result or silently keep the old behaviour rather than erroring — these are the ones most likely to go unnoticed in downstream code and are worth prioritizing regression tests for. By contrast, findings 1, 2, 3, 6, 9, 11, 17, 18, 19 and 32 (the `descend>=2` path) raise loudly or visibly misbehave, and so are self-announcing, though no less worth fixing given how central profiling and resource-limiting are to debugging workflows.
