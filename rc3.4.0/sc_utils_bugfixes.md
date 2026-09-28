# `sc_utils.py` bug audit

Audit of `sciris/sc_utils.py` (2582 lines) for genuine defects: wrong numerical or logical results, documented arguments that don't work, silent data corruption, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps, and performance.

**Scope**: the whole file, covered by two parallel auditors working on non-overlapping halves — Part 1 covers lines 67-941 (uuid, copy/deepcopy, sha, traceback, platform detection, web/URL fetching, HTML handling), Part 2 covers lines 949-2582 (type functions, misc string/dict/list functions, and the classes at the end of the file: `LazyModule`, `Link`, `tryexcept`, `KeyNotFoundError`, etc). **Method**: line-by-line reading of each function, followed by executed hypothesis tests, against the editable install (Sciris 3.3.0, numpy 2.4.6, commit `2d69aad`), with each finding reproduced a second time independently before being recorded here. Line numbers refer to the working tree at that commit.

**Re-verification.** This document was independently re-verified on 2026-09-25 against commit `d91898a` (where `sciris/sc_utils.py` is unchanged from `2d69aad`, so the line numbers still hold), with every finding re-run under Python 3.13 and `SCIRIS_BACKEND=agg`, and the network findings tested against a local `http.server`. Of the original 38 findings, 18 were confirmed (5 of them downgraded to Low), 2 were rewritten because the original description or fix was inaccurate (6, 16), and 18 were rejected as not a bug or not worth fixing (listed under "Rejected on review" near the end). Five bugs the original audit missed were added as findings 39-43. Original finding numbers are kept stable, so the numbering has gaps. The document now contains **25 findings: 8 High, 9 Medium, 8 Low**.

**Nothing in this document has been applied.** All fixes are described, not made.

This module holds the primitives the rest of Sciris is built on (`sc.dcp`, `sc.toarray`, `sc.mergedicts`, `sc.tolist`), so defects here don't stay local — they propagate into every module that calls them.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `sc.fast_uuid()`, `sc.uuid()` | Returns *more* than `n` UIDs, and the extras include the duplicates it was trying to eliminate | 148 |
| 2 | High | `sc.robust_dcp()`, `sc.dcp()` | Treats `tuple` as never needing a copy, so mutable data reached through a tuple is aliased between original and copy | 309 |
| 3 | High | `sc.dcp()`, `sc.robust_dcp()` | `sc.dcp(obj, die=False)` silently returns truncated nested dicts and lists, because the aborted `copy.deepcopy` leaves half-built containers in the shared memo (must ship with 13) | 334 |
| 4 | High | `sc.sha()` | Hashes `repr()`, so numpy arrays and DataFrames with more than 1000 elements collide on their truncated representation | 461 |
| 8 | High | `sc.suggest()` | `which='jaro'` returns the *least* similar option | 2030 |
| 9 | High | `sc.importbyname()`, `sc.LazyModule()` | Assigns into Sciris' own module globals instead of the caller's namespace, so the documented `sc.importbyname(pd='pandas')` defines nothing for the caller | 2064 |
| 10 | High | `sc.importbypath()` | Leaves a half-initialised module in `sys.modules` when the module raises, including under the original name when renamed | 2208 |
| 11 | High | `sc.tryexcept()` | Suppresses `BaseException`, so `KeyboardInterrupt` and `SystemExit` are swallowed | 2508 |
| 13 | Medium | `sc.robust_dcp()`, `sc.dcp()` | Raises on a plain dict, list or set holding an uncopyable element (must ship with 3) | 343 |
| 16 | Medium | `sc.urlopen()` | `response='status'` can never report a failing status, because 4xx/5xx raise before the branch is reached | 765 |
| 20 | Medium | `sc.isnumber()` / `sc.toarray()` / `sc.checktype()` | Rejects `np.bool_`, so `sc.toarray()` of a numpy boolean returns a 0-d array | 1207 |
| 24 | Medium | `sc.runcommand()` | `wait=False` blocks until the command finishes | 1877 |
| 25 | Medium | `sc.tryexcept()` | `die=True, catch=...` silently ignores `catch` | 2464 |
| 39 | Medium | `sc.checktype()`, `sc.tolist()` | Rejects abstract base classes and any class with a custom metaclass (`numbers.Number`, `abc.Mapping`, `Enum` subclasses) | 1159 |
| 40 | Medium | `sc.suggest()` | Does not case-normalize the user's input, so `'HIV'` finds no match in `['hiv', ...]` | 2022 |
| 41 | Medium | `sc.urlopen()` | `params=` appends a second `?` when the URL already has a query string | 759 |
| 42 | Medium | `sc.robust_dcp()`, `sc.dcp()` | Rebuilds containers as `obj.__class__(generator)`, which raises for `defaultdict` | 343 |
| 6 | Low | `sc.urlopen()`, `sc.wget()`, `sc.download()` | Ignores the charset the server declared, so a correctly labelled non-UTF-8 body comes back as `bytes` | 777 |
| 12 | Low | `sc.uuid()` | Silently ignores `tostring` and `length` when `uid` is supplied | 220 |
| 14 | Low | `sc.getplatform()`, `sc.iswindows()`, `sc.islinux()`, `sc.ismac()` | Raises `KeyError: 'other'` on any platform it does not recognize | 584 |
| 17 | Low | `sc.download()` | `save=False` silently collapses URLs whose final path segment matches, returning fewer results than URLs | 873 |
| 21 | Low | `sc.toarray()` | `sc.toarray(None, keepnone=True)` returns a non-iterable 0-d array, not `np.array([None])` | 1343 |
| 23 | Low | `sc.strsplit()` | Treats a multi-character `sep` string as a set of single characters | 1834 |
| 28 | Low | `sc.urlopen()` | `save=True` keeps the quotes from `Content-Disposition`, saving to a file literally named `"report.csv"` | 797 |
| 43 | Low | `sc.toarray()` | Returns a 0-d array for strings, dict views, sets and generators | 1339 |

Findings 5, 7, 15, 18, 19, 22, 26, 27, 29-38 were rejected on review; see "Rejected on review" near the end.

## Recurring patterns

**State written into a shared memo/namespace that outlives the call.** `robust_dcp()` hands its `_memo` to a `copy.deepcopy` that is *expected* to fail, then trusts what that failure left behind (finding 3) — the same shape would break any "try the fast path, then the slow path" design that shares accumulator state between the two. `_assign_to_namespace()` defaults to Sciris' own `globals()` rather than the caller's, so `sc.importbyname()` and `sc.LazyModule` assign into the library's internal module namespace instead of the caller's (finding 9). `importbypath()` registers a module in `sys.modules` before it has finished executing, so a mid-import failure leaves a half-initialised module permanently discoverable (finding 10).

**Type guards that check one type and miss its sibling.** `_numtype` covers every numpy scalar except `np.bool_` (finding 20), even though `checktype()`'s `'arraylike'` subtype already adds `_booltypes`. `checktype()` tests `type(objtype) == type`, which misses every class whose metaclass is not exactly `type`, including all ABCs (finding 39). `toarray()` special-cases numbers, 0-d arrays, pandas objects and `None`, but not dict views, sets, generators or strings (finding 43), although `tolist()` in the same file already coerces dict views.

**A "safe"/forgiving mode that is less safe than the strict one.** `sc.dcp(obj, die=False)` — the mode the `die=True` error message itself recommends — silently truncates nested containers (finding 3), aliases mutable data through tuples (finding 2), and raises outright on a plain container holding an uncopyable element (finding 13) or on a `defaultdict` (finding 42), while `die=True` at least fails loudly. `sc.tryexcept()`'s docstring calls itself "effectively an alias to `contextlib.suppress()`", but unlike `contextlib.suppress(Exception)` it also swallows `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit` (finding 11).

**Post-processing applied to only one of two paths.** `uuid()` performs `tostring`/`length` handling only inside the `if uid is None:` branch, so both arguments quietly evaporate on the conversion path (finding 12). `suggest()` lowercases the candidates but not the user's input, so the documented "case substitution is free" rule only holds in one direction (finding 40). A membership test against `[True, False, 0, 1]` is used to detect "is this a boolean" in `tryexcept.__init__`, and it is tested before `catch`, so `catch` is ignored whenever `die` is a bool (finding 25).

**`repr()` used as a canonical serialization.** `sha()` hashes `repr(obj)`, which is a display format with deliberately lossy defaults (elided numpy/pandas output) — the root cause of finding 4.

## High severity

### 1. `sc.fast_uuid(n=...)` returns *more* than `n` UIDs, including duplicates it was trying to eliminate — `sc_utils.py:148`

The uniqueness loop counts unique keys (`n_unique_keys = len(dict.fromkeys(output))`) but repairs the shortfall with `output.extend(new_uuids)` at line 161 — it appends new UIDs without ever removing the duplicates or trimming to `n`. The loop exits as soon as the *unique* count reaches `n`, so the returned list has length `n + (number of duplicate draws)` and still contains every duplicate.

```python
import sciris as sc
# 'numeric', length 6 => 1e6 possibilities; default safety=1000 allows n=1000
res = [len(sc.fast_uuid(which='numeric', length=6, n=1000, verbose=False)) for _ in range(30)]
print(sorted(set(res)))
out = next(o for _ in range(60) for o in [sc.fast_uuid(which='numeric', length=6, n=1000, verbose=False)] if len(o) != 1000)
print(len(out), len(set(out)), [x for x in set(out) if out.count(x) > 1])
```

Actual:

```
[1000, 1001, 1002, 1003]
1001 1000 ['653076']
```

Expected: `len(out) == 1000` and `len(set(out)) == 1000`. Note this is at the **default** `safety=1000`, i.e. inside the documented safe envelope (`n=1000` is exactly `allowed` here) — 5 of 20 trials came back wrong. With a smaller space it is the common case rather than the exception: `sc.fast_uuid(which='numeric', length=1, n=5, safety=1, verbose=False)` returned wrong-length or duplicate-containing lists in 14 of 20 trials, e.g. `['7','9','9','6','6','8','6','0']` for `n=5`.

Blast radius: reached by `sc.uuid(which='hex'/'ascii'/'numeric'/..., n>1)` as well, since `uuid()` delegates to `fast_uuid()` at line 202. `tests/test_utils.py:105` calls `sc.fast_uuid(n=100)` (default ascii/length 6, where collisions are astronomically unlikely) and asserts nothing about the length, so the loop body is effectively untested for the success case; the only test that enters it (`test_uuid`, `safety=1`, `n=99`, `length=2`) expects the `ValueError`.

Re-verified on review: `n=1000` returned 1001 or 1002 items, and `n=5` returned `['1','1','4','1','8','1','7','0']`.

**Fix**: after extending, deduplicate and truncate — `output = list(dict.fromkeys(output))[:n]` — or, better, build the result as a `dict.fromkeys`-backed set and draw until it reaches size `n`, then return `list(...)[:n]`. The review confirmed the dedupe-and-truncate fix is safe.

### 2. `sc.robust_dcp()` treats `tuple` as never needing a copy, aliasing mutable data reached through a tuple — `sc_utils.py:309`

Line 309 lists `tuple` (and `frozenset`) among "Immutable primitives that never need copying" and line 313 returns such objects unchanged. A tuple is only shallowly immutable: `([1,2,3],)` is a perfectly ordinary structure, and returning it as-is means the "copy" shares the inner list, so mutating the copy mutates the original.

```python
import sciris as sc
inner = [1,2,3]
t = (inner,)
c = sc.robust_dcp(t)
print(c is t, c[0] is inner)   # True True
c[0].append(999)
print(inner)                   # [1, 2, 3, 999]
```

Actual: `c is t` -> `True`, and the original list is modified through the copy. Expected: `sc.robust_dcp()` returns a structure whose mutable leaves are independent (`copy.deepcopy` handles this correctly, which is why the bug only shows up on the fallback path).

Reachable from `sc.dcp(obj, die=False)` whenever the tuple has not already been copied by the aborted `deepcopy` (e.g. the uncopyable attribute is declared before the tuple, so the memo has no entry for it):

```python
import threading, sciris as sc
class P:
    def __init__(s, t):
        s.lock = threading.Lock()   # declared first, so deepcopy dies before reaching self.pair
        s.pair = t
inner = [1,2,3]
c = sc.dcp(P((inner,)), die=False, verbose=False)
print(c.pair is not None and c.pair[0] is inner)  # True
c.pair[0].append(999); print(inner)               # [1, 2, 3, 999]
```

Same blast radius as the memo-truncation finding: silent aliasing out of `sc.dcp` is the worst possible failure for a deep-copy utility, since the caller's whole reason for calling it is isolation. Note this is a distinct defect from the memo bug — it fires even with a clean memo.

**Fix**: remove `tuple` from `primitives` and give tuples their own branch that rebuilds them from recursively copied elements (`tuple(robust_dcp(x, **kw) for x in obj)`, preserving `namedtuple` via `obj.__class__(*items)` where `_fields` exists). The namedtuple handling is required, not optional: a namedtuple's constructor takes positional fields, so the generic `obj.__class__(generator)` rebuild would fail for it. `frozenset` is safe to keep, since its members must be hashable, but only if hashability is being relied on deliberately.

### 3. `sc.dcp(obj, die=False)` silently returns truncated nested dicts and lists — `sc_utils.py:334`

`robust_dcp()` tries `copy.deepcopy(obj, memo=_memo)` first (line 334) and, on failure, falls through to its own element-wise walk using **the same `_memo`**. But a `copy.deepcopy` that raises part-way through has already registered its partially populated containers in that memo, so when the element-wise walk re-enters `robust_dcp()` for one of those containers it hits the `if obj_id in _memo: return _memo[obj_id]` fast path at line 328 and returns the *incomplete* copy. Every element after the one that triggered the failure is silently dropped.

**Trigger — this matters, because the most obvious repro does not fire.** The uncopyable element must sit at depth **>= 1** below the object handed to `dcp()`. A flat, top-level container is safe: `robust_dcp()` checks the memo *before* attempting the `deepcopy`, so the top-level object never re-reads the entry its own failed `deepcopy` poisoned, and the Mapping/Sequence branch rebuilds it in full. Every *child* container, by contrast, is reached by a second `robust_dcp()` call made after the poisoning, and so is returned truncated. Depth is the only thing that matters: shared references are not required, both dicts and lists are affected, and it does not depend on the leaf's failure mode (verified with a raising `__deepcopy__`, a `threading.Lock()`, and an open file object — all three give the same truncation).

```python
import sciris as sc

class Bad:
    def __deepcopy__(self, memo): raise RuntimeError('nope')

orig = {'inner': [0, 1, Bad(), 3, 4]}       # failing element at depth 1
copy = sc.dcp(orig, die=False, verbose=False)
print('original inner:', len(orig['inner']), 'items')
print('copied   inner:', len(copy['inner']), 'items ->', copy['inner'])
```

Actual:

```
original inner: 5 items
copied   inner: 2 items -> [0, 1]
```

Expected: 5 items, with only the uncopyable `Bad()` handled specially — that is exactly what the docstring promises ("Deep-copy anything that can be deep-copied, then try a shallow copy of that attribute, and otherwise return the original object").

The cut always lands at the failing element, so what survives depends purely on its position — `Bad` at index 0 yields `[]`, at index 3 yields `[0, 1, 2]`:

```
Bad at index 0: copy = []
Bad at index 1: copy = [0]
Bad at index 2: copy = [0, 1]
Bad at index 3: copy = [0, 1, 2]
Bad at index 4: copy = [0, 1, 2, 3]
```

**At depth >= 2 whole subtrees disappear**, because every container on the path is truncated at the point of failure:

```python
mk = lambda: [0, 1, Bad(), 3, 4]
print(sc.dcp({'a': mk()},                die=False, verbose=False))   # {'a': [0, 1]}
print(sc.dcp({'a': {'b': mk()}},         die=False, verbose=False))   # {'a': {}}
print(sc.dcp({'a': {'b': {'c': mk()}}},  die=False, verbose=False))   # {'a': {}}
```

The same holds when the container is an object attribute rather than a nested dict (`Holder({'k0':0,'k1':1,'k2':Bad(),'k3':3,'k4':4})` -> a copy whose attribute has 2 of 5 keys) and for a list nested in a list.

For contrast, the case that does **not** reproduce — the failing element at the top level of the object passed in:

```python
print(len(sc.dcp({f'k{i}': (Bad() if i==2 else i) for i in range(5)}, die=False, verbose=False)))  # 5, complete
print(len(sc.dcp([0, 1, Bad(), 3, 4], die=False, verbose=False)))                                  # 5, complete
```

The underlying memo pollution is easy to see in isolation:

```python
import copy, threading
memo = {}
inner = [10, 20, threading.Lock(), 30, 40]
try: copy.deepcopy(dict(items=inner), memo=memo)
except Exception as E: print(E)                 # cannot pickle '_thread.lock' object
print(memo[id(inner)])                          # [10, 20]
```

Blast radius: `sc.dcp()` is the most-used function in the library and `die=False` is documented as the safe/robust mode (`sc.dcp(obj, die=False)` is even suggested in the `die=True` error message at line 286). Objects that hold a lock, an open file, a socket, a database connection, or a `matplotlib` canvas are precisely the ones that reach this path, and since real objects nest, the failing leaf is almost always at depth >= 1 — the result is a copy that looks fine and is missing data. The depth dependence is also what makes this hard to notice: a minimal top-level test passes, so the behaviour looks correct until it is exercised on a real object graph. `robust_dcp()` is not called anywhere else inside Sciris (only from `dcp`), and no test exercises the `die=False` branch (`tests/test_utils.py:26-28` only asserts that `die=True` raises).

Re-verified on review, including a realistic case: an object holding a lock inside `self.results` silently loses every key of `results` after the lock. This is the most serious finding in the document.

**Fix**: do not share the memo between the failed `copy.deepcopy` attempt and the element-wise fallback — copy into a throwaway memo and merge it back only on success: `tmp = dict(_memo); dup = copy.deepcopy(obj, memo=tmp); _memo.update(tmp)`.

**This fix must ship together with finding 13.** The review applied the memo fix to an in-memory copy of `robust_dcp()`: `{'a':{'b':[0,1,Bad(),3,4]}}` then copies in full, and an object whose `results` dict holds a lock keeps all its keys (the `results` attribute falls back to a shallow copy, as the docstring describes). But a plain dict nested in a dict and holding a lock now raises `TypeError` instead of truncating, because the container branch has no per-element guard and `object.__new__(_thread.lock)` raises (finding 13). Shipped alone, the fix turns silent truncation into a crash for pure-dict structures.

### 4. `sc.sha()` hashes `repr()`, so long numpy arrays and DataFrames collide on their truncated representation — `sc_utils.py:461`

Line 461 does `obj = repr(obj)` for any non-string input. numpy's default `printoptions['threshold']` is 1000, so `repr()` of a longer array elides the middle with `...` — and therefore so does the digest. Two arrays that differ in a thousand interior values hash identically.

```python
import numpy as np, sciris as sc
a = np.arange(3000)
b = a.copy(); b[1500] = -999
print(np.array_equal(a, b))                                  # False
print(sc.sha(a, digest=True) == sc.sha(b, digest=True))      # True
c = a.copy(); c[1000:2000] = 0
print(sc.sha(a, digest=True) == sc.sha(c, digest=True))      # True
```

Actual output:

```
False
True
True
```

Expected: different arrays give different digests. Pandas is affected the same way (`repr` of a DataFrame shows only the head and tail):

```python
import pandas as pd, numpy as np, sciris as sc
df1 = pd.DataFrame({'x':np.arange(200)}); df2 = df1.copy(); df2.loc[100,'x'] = -1
print(df1.equals(df2), sc.sha(df1,True) == sc.sha(df2,True))   # False True
```

Plain Python lists are *not* affected (their `repr` is complete), which makes the failure look arbitrary from the outside: `sc.sha(list(range(3000)))` distinguishes correctly but `sc.sha(np.arange(3000))` does not. The docstring's stated contract is `sha2 != sha3` for different inputs, and the one test that exists (`tests/test_utils.py:21`) hashes a 5-element array. `sha()` is not called elsewhere inside Sciris, but it is public API whose obvious use is cache keys and change detection over exactly this kind of array data.

**Fix**: for `np.ndarray` use `obj.tobytes()` plus `str(obj.dtype)` and `obj.shape`; for pandas use `pd.util.hash_pandas_object(...).values.tobytes()`; otherwise, at minimum, wrap the `repr()` call in `np.printoptions(threshold=sys.maxsize)` so nothing is elided. Note that any of these changes the digest of *every* array or DataFrame, including short ones, so previously stored digests will no longer match; mention it in the changelog.

### 8. `sc.suggest(which='jaro')` returns the *least* similar option — `sc_utils.py:2030`

`jellyfish.jaro_similarity()` (and the pre-1.0 fallback `jellyfish.jaro_distance()`, which is also a similarity in [0,1] where 1 = identical) returns *higher = better*, but the function stores it in `distance` and then sorts ascending (`order = np.argsort(distance)`) and applies `min(distance) > threshold`, both of which assume *lower = better*. The result is exactly inverted.

```python
import sciris as sc
opts = ['temperature','humidity','pressure']
print(sc.suggest('temprature', opts, which='jaro'))
print(sc.suggest('temprature', opts))                 # damerau, for comparison
print(sc.suggest('temprature', opts, which='jaro', n=3))
```

```
which='jaro' -> humidity
which='damerau' -> temperature
jaro n=3 -> ['humidity', 'pressure', 'temperature']
```

Actual: `'humidity'` (jaro similarity 0.483). Expected: `'temperature'` (0.970 — the obvious answer, and what the two other methods return). The full ranking is reversed, and `fulloutput=True` reports the reversed order too. The threshold logic is inverted as well: `min(distance) > threshold` can never fire for jaro (similarities are <= 1 and the default threshold is `ceil(2*len(input)/3)`), so `threshold` is dead for this method and no "no suggestion" answer is ever returned. Blast radius: `which` defaults to `'damerau'` and `sc.suggest()` has no internal call sites, so the damage is confined to callers who select the documented option.

**Fix**: negate similarity-type metrics on entry (e.g. `distance[i] = 1 - jaro(...)`) so all three methods are distances, or track a `higher_is_better` flag per method and flip both the `argsort` and the threshold comparison.

### 9. `sc.importbyname()` assigns into Sciris' own module globals instead of the caller's namespace — `sc_utils.py:2064`

`_assign_to_namespace()` defaults `namespace` to `globals()`, but it is defined *inside* `sc_utils.py`, so "globals" is Sciris' own module namespace, not the caller's. Every `sc.importbyname(var='module')` call therefore rebinds a name inside `sciris.sc_utils` — where `np`, `pd`, `sc`, `re`, `sys`, `types`, `copy`, `json`, `string`, `uuid`, `warnings`, `functools`, `subprocess` and `traceback` all already live — while the caller's namespace is left untouched.

```python
import pandas as pd, sciris as sc, sciris.sc_utils as scu
sc.importbyname(pd='json')          # documented "dictionary syntax to assign to namespace"
print(scu.pd.__name__)
sc.toarray(pd.Series([1,2]))
```

```
sciris.sc_utils.pd is now: json
sc.toarray(pd.Series([1,2])) -> AttributeError: module 'json' has no attribute 'DataFrame'
'pd' in caller globals: module      # (only because the test file imported pandas itself; importbyname did not put it there)
```

The real defect is that the namespace is the wrong one: the documented usage `sc.importbyname(pd='pandas', np='numpy')` does not make `pd`/`np` available to the caller at all (re-verified on review: nothing appears in the caller's namespace, and the names are written into `sciris.sc_utils` instead). A secondary consequence, which needs a colliding variable name and is therefore fairly contrived, is that a name that collides with an `sc_utils` global silently breaks Sciris for the rest of the process. `pd` breaks `sc.toarray()` on Series/DataFrame and `sc.checktype(..., 'listlike'/'arraylike')`; `sc='scipy'` is worse, because `sc_utils` does `import sciris as sc` and uses it everywhere:

```python
import sciris as sc
sc.importbyname(sc='json')
with sc.tryexcept() as te: [][2]
# -> AttributeError: module 'json' has no attribute 'ifelse'   (from tryexcept.__init__)
```

`sc.LazyModule` shares the defect via the same helper (`sc_utils.py:2384`), so `plt = sc.importbyname(plt='matplotlib.pyplot', lazy=True)` also writes into `sc_utils` on first attribute access, which is also why a `LazyModule` never replaces the caller's variable as its docstring claims (originally filed separately as finding 38, now folded in here: fixing this finding fixes that too). Blast radius: `_assign_to_namespace()` is called from `importbyname()` (`sc_utils.py:2135`) and `LazyModule._load()` (`sc_utils.py:2384`) and nowhere else, and `sc.importbyname()` has no internal call sites, so the problem is always user-triggered. In the collision case it persists for the life of the process and surfaces as an unrelated `AttributeError` deep inside Sciris.

**Fix**: get the caller's globals instead of the library's — e.g. `namespace = inspect.stack()[k].frame.f_globals` in `importbyname()`/`LazyModule` and pass it down explicitly, or (safer) drop the implicit-namespace feature and document that the return value must be assigned. At minimum, refuse to write into `sc_utils.__dict__` when no `namespace` is supplied.

### 10. `sc.importbypath()` leaves a half-initialised module in `sys.modules` when the module raises — `sc_utils.py:2208`

`sys.modules[name] = module` is executed *before* `spec.loader.exec_module(module)` (line 2213) and there is no `try/finally`, so if the module raises during import the broken, partially-executed module object is left registered. CPython's own import machinery deletes the `sys.modules` entry on failure precisely to avoid this; `sc.importbypath()` does not.

```python
import sys, sciris as sc
open('bad.py','w').write('X = 1\nraise ValueError("boom")\n')
try: sc.importbypath('bad.py')
except ValueError as E: print('raised:', E)
print('sys.modules["bad"]:', sys.modules.get('bad'))
import bad
print('later "import bad" succeeds, X =', bad.X)
```

```
raised: boom
sys.modules["bad"]: <module 'bad' from '.../bad.py'>
later "import bad" succeeds, X = 1
```

Actual: a subsequent ordinary `import bad` (or `sc.importbypath` of anything that self-imports) returns the module with only the pre-exception half of its definitions, with no error. Expected: `sys.modules` unchanged, so the next import re-runs and re-raises. The window is widened by `renamed`: when `name != orig_name` the module is registered under *both* names (line 2210), and on the error path neither is cleaned up, so a failed `sc.importbypath('/path/to/mylib', name='newlib')` can also shadow the *installed* `mylib` for the rest of the session. This is exactly the "load two versions for comparison" use case the docstring advertises. Re-verified on review with a standard-library name: a failed `sc.importbypath('v2/csv', name='csvnew')` leaves the broken module in `sys.modules['csv']`, shadowing the stdlib `csv` for every later `import csv` in the process.

**Fix**: wrap `exec_module` in `try/except`, and on failure `sys.modules.pop(name, None)` plus restore/`pop` `orig_name` (the same restore logic that lines 2216-2219 perform on the success path); re-raise afterwards.

### 11. `sc.tryexcept()` suppresses `BaseException`, so `KeyboardInterrupt` and `SystemExit` are swallowed — `sc_utils.py:2508`

`__exit__` tests only `if exc_type is not None`, never `issubclass(exc_type, Exception)`, so it catches the whole `BaseException` hierarchy. The docstring says the class is "Effectively an alias to `contextlib.suppress()`", and the first example is annotated "Equivalent to `contextlib.suppress(Exception)`" — and `contextlib.suppress(Exception)` re-raises `KeyboardInterrupt`.

```python
import sciris as sc
n = 0
for i in range(3):
    with sc.tryexcept(verbose=0):
        n += 1
        raise KeyboardInterrupt('ctrl-C')
print('all', n, 'KeyboardInterrupts suppressed')
with sc.tryexcept(verbose=0):
    raise SystemExit(1)
print('SystemExit suppressed too')
```

```
all 3 KeyboardInterrupts suppressed
SystemExit suppressed too
```

Actual: Ctrl-C inside the block is discarded, so the loop pattern from the docstring's own "Storing the history of multiple exceptions" example cannot be interrupted from the keyboard — each iteration eats the signal and continues. `sys.exit()` inside the block is also ignored (the process keeps running with an exit request silently downgraded to a history entry), and `GeneratorExit` is swallowed too, which corrupts generator finalisation. `contextlib.suppress(Exception)` in the same position correctly re-raises. Blast radius: `sc.tryexcept` has no internal call sites (grep for `tryexcept(` outside `sc_utils.py` returns nothing), so the exposure is user code — but it is a headline feature of the module and the docstring's own loop example is exactly the pattern that becomes uninterruptible.

**Fix**: in `__exit__`, return early (propagate) unless `issubclass(exc_type, Exception)`, or add `BaseException`-derived non-`Exception` types to an always-die list; `KeyboardInterrupt`, `SystemExit` and `GeneratorExit` must never be suppressed by default. The guard should still honour an explicit request: `sc.tryexcept(catch=KeyboardInterrupt)` (or `die=False` with a `catch` list that names a `BaseException` subclass) should keep catching it, so check `catchtypes` before applying the `Exception`-only default.

## Medium severity

### 13. `sc.robust_dcp()` raises on a plain dict, list or set holding an uncopyable element — `sc_utils.py:343`

The container branches at lines 343 and 345 recurse into elements with no `try`/`except` at all, unlike the custom-object branch (lines 356-364), which guards every attribute. When an element cannot be copied, the recursion reaches the "plain custom object" branch and `object.__new__(obj.__class__)` at line 347 raises — so the function documented as "Ultra-robust deepcopying" fails outright, and `sc.dcp(obj, die=False)` raises instead of falling back.

```python
import threading, sciris as sc
sc.dcp({'lock': threading.Lock()}, die=False, verbose=False)
sc.dcp([[1,2,3], threading.Lock()], die=False, verbose=False)   # same
```

Actual: `TypeError: object.__new__(_thread.lock) is not safe, use _thread.lock.__new__()`. Expected: per the docstring, an uncopyable leaf is left as the original reference and the copy succeeds. The asymmetry is the giveaway — wrapping the same dict in an object works, because the attribute loop catches the exception.

`sc.dcp(obj, die=False)` is what the `die=True` error message at line 286 tells users to reach for, so this turns a recoverable situation into a hard failure with a confusing message about `_thread.lock.__new__`.

**Fix**: give the container branches the same per-element `try: robust_dcp(...) / except: copy.copy(...) / except: value` ladder the attribute loop uses, and guard `object.__new__` at line 347 so a non-copyable leaf returns `obj` rather than raising. Re-verified on review, including for `sc.odict` (`sc.dcp(sc.odict(lock=threading.Lock()), die=False)` raises too). **Ship this together with finding 3**: once the memo fix lands, more structures reach these unguarded container branches, so fixing 3 alone turns silent truncation into this crash. See also finding 42, a sibling defect in the same container-rebuild lines.

### 16. `response='status'` can never report a failing status, because 4xx/5xx raise before the branch is reached — `sc_utils.py:765`

The docstring documents `response='status'` as "the HTTP status" and the example `sc.urlopen('wikipedia.org', response='status')` as a way to check a site. But `urllib.request.urlopen` raises `HTTPError` for any non-2xx/3xx response, and that happens at line 765, before `output = resp.status`. So the only statuses the option can ever return are the successful ones; the interesting cases all come back as an exception, and `die=False` does not change this (`die` is documented only as controlling text-conversion failures, and is not consulted anywhere near the request).

```python
for path in ['/404', '/500']:
    try:    print(path, '->', sc.urlopen('127.0.0.1:8795'+path, response='status'))
    except Exception as E: print(path, '-> RAISED', type(E).__name__, E)
```

Actual:

```
/404 -> RAISED HTTPError HTTP Error 404: Not Found
/500 -> RAISED HTTPError HTTP Error 500: Internal Server Error
```

Expected: `404` and `500`. (For `response='text'` and the other modes, raising on a 4xx/5xx is the right behaviour and should be kept; see the fix.)

**Fix**: restrict the change to `response='status'`: `except urllib.error.HTTPError as E: if response == 'status': return E.code; raise`. The originally proposed fix (return the status *or body* for `'status'`/`'text'`, and re-raise only when `die=True`) was shown on review to be harmful: `urlopen()`'s `die` defaults to `False`, and `sc.download()` does not forward its own `die` to `urlopen()`, so every 404 or 500 would silently return the error page, and with `save=True` would save the error page's HTML as if it were the requested file.

### 20. `sc.isnumber()` rejects `np.bool_`, so `sc.toarray()` of a numpy boolean returns a 0-d array — `sc_utils.py:1207`

`_numtype = (numbers.Number,)`, and `np.bool_` is the one numpy scalar type that is *not* registered with the `numbers` ABCs (`np.float32`, `np.int64` etc. all are). So `sc.isnumber(np.bool_(True))` is False while `sc.isnumber(True)` is True, and because `toarray()`'s scalar test is `isnumber(x) or (isinstance(x, np.ndarray) and not np.shape(x))` — and a numpy scalar is not an `ndarray` instance — a numpy boolean is passed through unwrapped.

```python
import numpy as np, sciris as sc
print(sc.isnumber(True), sc.isnumber(np.bool_(True)))
print(repr(sc.toarray(np.any([1,2]))), repr(sc.toarray(True)))
```

```
True False
array(True)   array([ True])
```

`np.bool_` is not exotic: it is what `np.any()`, `np.all()`, `arr > 3` element access, and `mask[0]` all return. So `sc.toarray(np.any(x))` silently yields a 0-d array that cannot be iterated or measured, whereas `sc.toarray(bool(np.any(x)))` works. The author was aware of the gap elsewhere in the same file — `checktype()`'s `'arraylike'` branch deliberately uses `subtype = _numtype + _booltypes` (line 1179) to admit `np.bool_` — so `sc.checktype(np.array([True]), 'arraylike')` is True while `sc.isnumber(np.array([True])[0])` is False. Blast radius: `sc.isnumber()` has 25 call sites in Sciris, several of which are scalar-vs-vector dispatch switches (`sc_math.py:109` `safedivide`, `sc_math.py:597`, `sc_math.py:1202-1204` `smoothinterp`, `sc_printing.py:740`), so a `np.bool_` argument takes the vector branch there too.

**Fix**: define `_numtype = (numbers.Number, np.bool_)`. The review confirmed this is safe and consistent: `bool` is already a `Number`, and `checktype(..., 'arraylike')` already adds `_booltypes`. A narrower alternative that leaves `isnumber()` alone is to also wrap `np.generic` scalars in `toarray()`.

### 24. `sc.runcommand(wait=False)` blocks until the command finishes — `sc_utils.py:1877`

`printoutput` defaults to `True` when `wait=False`, and that branch is `while p.returncode is None: stdout, stderr = p.communicate()` — and `Popen.communicate()` waits for the process to terminate. So the default `wait=False` call blocks for exactly as long as `wait=True` would, contradicting the docstring ("whether to wait for the process to return (else, return immediately with the subprocess)") and its own example, `sc.runcommand('sleep 600; mkdir foo', wait=False) # ... the function returns immediately`.

```python
import time, sciris as sc
t0 = time.time(); p = sc.runcommand('sleep 2', wait=False);                   print(f'{time.time()-t0:.2f} s')
t0 = time.time(); p = sc.runcommand('sleep 2', wait=False, printoutput=False); print(f'{time.time()-t0:.2f} s')
```

```
wait=False elapsed: 2.00 s (docstring: "the function returns immediately")
printoutput=False elapsed: 0.00 s
```

Nothing is printed for the trouble, either: with `wait=False` the defaults are `dict(shell=True, bufsize=0)` with no `stdout=PIPE`, so the child writes straight to the terminal and `communicate()` returns `(None, None)` — the "print real-time output if `wait=False`" feature added in v3.1.1 prints nothing and only costs the caller the non-blocking behaviour they asked for. Workaround: pass `printoutput=False` explicitly.

Re-verified on review: `sc.runcommand('sleep 1.5', wait=False)` blocked for 1.50 s.

**Fix**: delete the `printoutput` loop for the `wait=False` branch, or default `printoutput` to `False` for both branches. No threaded pipe reader is needed (as the original audit proposed): with `wait=False` there is no `stdout=PIPE`, so the child already writes to the terminal in real time, and the loop only adds the blocking `communicate()`.

### 25. `sc.tryexcept(die=True, catch=...)` silently ignores `catch` — `sc_utils.py:2464`

The `elif die in [True, False, 0, 1]` branch is tested before `elif die is None and catch is not None`, and it does not look at `catch`, so `catchtypes` stays empty. Passing both is rejected with a clear `ValueError` when `die` is an exception type, but accepted-and-ignored when `die` is a bool.

```python
import sciris as sc
values = [0,1]
with sc.tryexcept(die=True, catch=IndexError, verbose=0):
    values[2]                                   # -> IndexError propagates
sc.tryexcept(die=True, catch=IndexError).catchtypes   # -> ()
sc.tryexcept(die=KeyError, catch=IndexError)          # -> ValueError (inconsistent)
```

Actual: the `IndexError` is raised despite `catch=IndexError`. Expected: the `IndexError` is caught — the `catch` docstring says "one or more exceptions to catch regardless of `die`". A user writing `die=True, catch=IndexError` to be explicit gets the opposite of the intended behaviour, silently.

**Fix**: in the boolean-`die` branch, also set `catchtypes = tolist(catch)` when `catch is not None`, so `catch` is honoured regardless of `die` as documented. The audit's original alternative (raise the "Unexpected input" `ValueError` whenever both are supplied) contradicts the docstring and should not be used.

### 39. `sc.checktype()` rejects abstract base classes and any class with a custom metaclass — `sc_utils.py:1159`

*Added on review (missed by the original audit).* `elif type(objtype) == type:` is an exact metaclass test, so any class whose metaclass is not exactly `type` falls through to the "Could not understand" `ValueError`. That covers `numbers.Number`, all the `collections.abc` classes, `Enum` subclasses and similar. The docstring says "If objtype is a type, then this function works exactly like isinstance()". `sc.tolist(..., objtype=...)` inherits the bug.

```python
import numbers, sciris as sc
from collections import abc
sc.checktype(3, numbers.Number)          # ValueError: Could not understand what type you want to check ...
sc.checktype({}, abc.Mapping)            # ValueError
sc.tolist([1,2], objtype=numbers.Number) # ValueError
```

Actual: `ValueError: Could not understand what type you want to check: should be either a string or a type, not "<class 'numbers.Number'>"` for all three. Expected: `True`, `True`, `[1, 2]`.

**Fix**: `elif isinstance(objtype, type): objinstance = objtype`. This is safe because strings, tuples and lists are handled in the neighbouring branches.

### 40. `sc.suggest()` does not case-normalize the user's input, so uppercase input misses exact case-insensitive matches — `sc_utils.py:2022`

*Added on review (missed by the original audit).* The docstring says "case substitution and stripping whitespace are not included in the distance", and the inline comment says the inputs are switched to lowercase. But only the candidate is normalized (`distance[i] = dist_func(user_input, s.strip().lower())`); `user_input` is used as-is. Every uppercase character in the input therefore costs an edit, and an all-caps input often exceeds the threshold and returns `None`.

```python
import sciris as sc
sc.suggest('HIV', ['hiv', 'hpv', 'syphilis'])  # -> None
sc.suggest('SIR', ['sir', 'seir'])             # -> None
sc.suggest('ABC', ['abc', 'xyz'])              # -> None
```

Actual: `None` for all three. Expected: `'hiv'`, `'sir'`, `'abc'`.

**Fix**: `distance[i] = dist_func(user_input.strip().lower(), s.strip().lower())`, leaving `cs_distance` as the case-sensitive tie-breaker. Checked by running a patched copy: the three docstring examples still give their documented outputs (`'Foo'`, `'Foo'`, `'Foo '`), and the cases above return `'hiv'` and `'sir'`.

### 41. `sc.urlopen(params=...)` builds an invalid URL when the URL already has a query string — `sc_utils.py:759`

*Added on review (missed by the original audit).* The code does `full_url = full_url + '?' + up.urlencode(params)` unconditionally, so a URL that already carries a query string gets a second `?`.

```python
sc.urlopen('127.0.0.1:PORT/api?key=1', params={'a': 2})
# server receives path: /api?key=1?a=2
```

Actual: the server sees `key='1?a=2'` and no `a` parameter. Expected: `/api?key=1&a=2`.

**Fix**: `full_url += ('&' if '?' in full_url else '?') + up.urlencode(params)`.

### 42. `robust_dcp()` rebuilds containers as `obj.__class__(generator)`, which raises for `defaultdict` and similar subclasses — `sc_utils.py:343`

*Added on review (missed by the original audit).* A sibling of finding 13, but it fails even when every element can be copied by the element-wise walk. The Mapping branch does `dup = obj.__class__((robust_dcp(k, **kw), robust_dcp(v, **kw)) for k, v in obj.items())`, and `defaultdict(gen)` treats the generator as its `default_factory`. So `sc.dcp(die=False)` crashes on any `defaultdict` once the `deepcopy` fast path has failed.

```python
import threading, sciris as sc
from collections import defaultdict
sc.dcp(defaultdict(list, a=[1], lk=threading.Lock()), die=False, verbose=False)
```

Actual: `TypeError: first argument must be callable or None`. Expected: a copy, as `die=False` promises.

**Fix**: build Mapping results with `dup = copy.copy(obj); dup.clear(); dup.update(...)`, falling back to `obj.__class__(...)`. Alternatively, wrap the container rebuild in the same try / shallow-copy / original ladder proposed for finding 13. Best fixed together with 3 and 13.

## Low severity

### 6. `sc.urlopen()` ignores the charset the server declared, so a correctly labelled non-UTF-8 body comes back as `bytes` — `sc_utils.py:777`

*Rewritten on review (originally High).* Line 777 is a bare `output = output.decode()`, i.e. UTF-8 regardless of the response's `Content-Type: ...; charset=...`. When the body is correctly labelled as some other encoding and is not valid UTF-8, the decode fails and, under the default `die=False`, `output` is left as the raw `bytes`.

Repro against a local server that returns `text/plain; charset=iso-8859-1` with the latin-1 bytes for `'Grüße naïve café'`:

```python
out = sc.urlopen('127.0.0.1:8795/latin1')
print(type(out).__name__, repr(out))
```

Actual:

```
bytes b'Gr\xfc\xdfe na\xefve caf\xe9'
```

Expected: `str 'Grüße naïve café'`, since the server said exactly which encoding to use. With `die=True` you get `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfc in position 2: invalid start byte`.

Two claims in the original version of this finding were wrong and have been removed. First, returning `bytes` on a decode failure is not a silent contract violation: it is the documented behaviour of `die` ("whether to raise an exception if converting to text failed"), and `die=False` is the default. Second, the "mojibake" example was backwards: it described a server that declares `charset=iso-8859-1` while actually sending UTF-8 bytes, where Sciris returns `'café'` — the text the user wants — and honouring the declared charset would give `'cafÃ©'`. The genuine defect is only the correctly-labelled non-UTF-8 case above, which is why this is now Low severity.

**Fix**: decode with the declared charset and a UTF-8 fallback: `output.decode(resp.headers.get_content_charset() or 'utf-8')`. This is safe: unlabelled responses behave exactly as today. Keep the existing `die` semantics for decode failures.

### 12. `sc.uuid()` silently ignores `tostring` and `length` when `uid` is supplied — `sc_utils.py:220`

The string conversion and the `length` trim live inside the `if uid is None:` block (lines 220-236), so on the conversion path — the one documented by "uid (str or uuid): if a string, convert to an actual UUID" — both documented arguments do nothing. `n` is ignored on that path too.

```python
import sciris as sc
u = str(sc.uuid())
print(repr(sc.uuid(uid=u, tostring=True, length=8)))
print(repr(sc.uuid(uid=u, n=3)))
```

Actual:

```
UUID('7437b1d6-8162-4007-a972-f98c267bd455')
UUID('7437b1d6-8162-4007-a972-f98c267bd455')
```

Expected: `'7437b1d6'` for the first (a length-8 string) and either a list of 3 or an error for the second. Callers who normalize input with `sc.uuid(maybe_uid, tostring=True)` get a `UUID` object rather than a string roughly half the time, depending on whether their input happened to be `None`.

*Downgraded from Medium on review*: confirmed, but low impact — it is a documented argument that silently does nothing, rather than a wrong value.

**Fix**: move the `tostring`/`length` handling out of the `if uid is None:` block so it applies to `output` regardless of provenance, and either honor or explicitly reject `n` when `uid` is given.

### 14. `sc.getplatform(expected, ...)` raises `KeyError: 'other'` on any platform it does not recognize — `sc_utils.py:584`

`mapping` has only the keys `linux`, `windows` and `mac`, but `plat` is initialized to `'other'` (line 578) and stays there for anything unrecognized. Line 584 then indexes `mapping[plat]`, so the `expected is not None` path — i.e. every call to `sc.iswindows()`, `sc.islinux()` and `sc.ismac()` — raises instead of returning `False`.

```python
import sciris as sc
sc.getplatform('linux', platform='freebsd')
```

Actual: `KeyError: 'other'` (identical for `'aix'`, `'sunos5'`, `'emscripten'`). Expected: `False` — the docstring explicitly lists `'other'` as one of the four possible return values of the function, and `platform=` is a documented argument whose stated purpose is "if supplied, map this onto one of the 'main' platforms". On a genuinely unrecognized OS this also breaks `sc.iswindows()`/`sc.islinux()`/`sc.ismac()`, which are used throughout Sciris for platform dispatch.

*Downgraded from Medium on review*: confirmed, but it only affects platforms Sciris rarely runs on. Besides FreeBSD and the other Unixes above, it also fires on Android since Python 3.13, where `sys.platform == 'android'`.

**Fix**: add `other = []` to `mapping`, or test `expected.lower() in mapping.get(plat, [])`. Trivial and safe.

### 17. `sc.download(..., save=False)` silently collapses URLs whose final path segment matches, returning fewer results than URLs — `sc_utils.py:873`

With `save=False` and no `filename`, line 873 derives the result keys as `url.split('/')[-1]`. Two URLs with the same last segment — the common case for versioned or per-source data files — produce the same key, and the `sc.objdict` built at line 900 keeps only the last one. No warning is issued, and the returned container is simply shorter than the input.

```python
import sciris as sc
u = ['http://127.0.0.1:8795/v1/payload/data', 'http://127.0.0.1:8795/v2/payload/data']
out = sc.download(u, save=False, verbose=False)
print(len(out), dict(out))
```

Actual: `1 {'data': 'PAYLOAD-data'}`. Expected: two entries (the docstring's contract for `save=False` is "Download and store in memory", with one result per URL). Because `download()` also collapses a length-1 result to a bare value at line 906, `len(out)` dropping to 1 additionally changes the *type* of the return value for a two-URL call.

*Downgraded from Medium on review*: confirmed (`.../v1/data` and `.../v2/data` collapse to one entry), but it needs URLs with matching final segments and `save=False`.

**Fix**: detect duplicate derived keys and disambiguate (e.g. fall back to the full URL, or append an index), or raise/warn; alternatively return a list when the caller supplied a list.

### 21. `sc.toarray(None, keepnone=True)` returns a non-iterable 0-d array, not `np.array([None])` — `sc_utils.py:1343`

The `keepnone` docstring says the choice is between `np.array([])` and `np.array([None], dtype=object)`. The `x is None` branch only fires when `keepnone` is False; when it is True, `None` reaches `np.array(None)` unwrapped, which is a 0-d object array. That defeats the function's entire stated purpose ("`sc.toarray(3)` will return `np.array([3])` (i.e. a 1-d array that can be iterated over)").

```python
import numpy as np, sciris as sc
a = sc.toarray(None, keepnone=True)
print(repr(a), a.ndim)
for x in a: pass
```

```
array(None, dtype=object) | ndim 0 | docstring promises array([None], dtype=object)
TypeError: iteration over a 0-d array          # len(a) also fails: "len() of unsized object"
```

Any caller that follows the documented contract and iterates or takes `len()` of the result crashes. Inside Sciris `keepnone=True` is only used with `sc.tolist()`/`sc.mergelists()` (which handle it correctly), so this is exposed to external callers.

*Downgraded from Medium on review*: confirmed, but only reachable through an explicit `keepnone=True`.

**Fix**: handle `None` before the `np.array()` call in both directions, e.g. `elif x is None: x = [None] if keepnone else []`. The review confirmed this is safe.

### 23. `sc.strsplit()` treats a multi-character `sep` string as a set of single characters — `sc_utils.py:1834`

`sep` is normalised only from `None`; otherwise the code does `for s in sep: string = string.replace(s, special)`, and iterating a `str` yields its characters. A separator of more than one character is therefore applied character by character, silently shredding the input.

```python
import sciris as sc
print(sc.strsplit('cats and dogs', sep=' and '))
print(sc.strsplit('a::b', sep='::'), sc.strsplit('key=value', sep='=='))
```

```
['c', 'ts', 'ogs']    expected ['cats','dogs']
['a', 'b'] ['key', 'value']
```

Actual: `['c', 'ts', 'ogs']` — the 'a', 'n', 'd' characters were consumed out of the middle of the words, and `skipempty=True` hid the resulting blanks. The signature is documented as `sep (str/list)` and the working example uses a one-character string (`sep='_'`), so a multi-character string is a natural extrapolation; repeated-character separators like `'::'` accidentally work, which makes the failure mode harder to spot. Nothing inside Sciris calls `sc.strsplit()`, so the blast radius is external callers only.

*Downgraded from Medium on review*: confirmed, but nothing in Sciris or its tests passes a multi-character string.

**Fix**: `sep = [sep] if isinstance(sep, str) else sep` before the loop (i.e. treat a string as one separator, and require a list for multiple). This is a behaviour change for anyone passing a character set as a string (e.g. `sep=',;'`), which currently splits on either character. That usage is undocumented and multi-character separators are the natural reading, so the change is still worth making, but note it in the changelog.

### 28. `sc.urlopen(save=True)` keeps the quotes from `Content-Disposition`, saving to a file literally named `"report.csv"` — `sc_utils.py:797`

The regex `re.findall("filename=(.+)", ...)` captures everything after `filename=`, including the surrounding double quotes required by RFC 6266 (and any trailing directives such as `; size=1234`).

```python
import os, tempfile, sciris as sc
os.chdir(tempfile.mkdtemp())
print(repr(sc.urlopen('127.0.0.1:8795/disposition', save=True)))   # header: attachment; filename="report.csv"
print(os.listdir('.'))
```

Actual: `'/tmp/tmp2qm6yjab/"report.csv"'` and `['"report.csv"']`. Expected: `report.csv`. On Windows `"` is not a legal filename character, so the save fails outright rather than producing an oddly named file. (`re.findall(...)[0]` would also raise `IndexError` for a bare `Content-Disposition: attachment` with no `filename=`.)

**Fix**: strip quotes and trailing parameters — e.g. `re.search(r'filename\*?=\s*"?([^";]+)', header)` — and guard the no-match case.

### 43. `sc.toarray()` returns a 0-d array for strings, dict views, sets and generators — `sc_utils.py:1339`

*Added on review (missed by the original audit).* Only numbers, 0-d arrays, pandas objects and `None` are special-cased (lines 1339-1347). Anything else goes straight to `np.array()`, which wraps a non-sequence iterable as a single 0-d object. This contradicts the function's stated purpose of returning a 1-d array that can be iterated over.

```python
import sciris as sc
d = {'a': 1, 'b': 2}
sc.toarray(d.keys())    # array(dict_keys(['a', 'b']), dtype=object), ndim 0
sc.toarray(d.values())  # array(dict_values([1, 2]), dtype=object), ndim 0
sc.toarray({1, 2})      # array({1, 2}, dtype=object), ndim 0
sc.toarray('abc')       # array('abc', dtype=object), ndim 0
```

Expected: `array(['a','b'])`, `array([1,2])`, `array([1,2])`, `array(['abc'], dtype=object)`. The dict-view case is the realistic one: `tolist()` in the same file already coerces dict views by default.

**Fix**: add `elif isinstance(x, (abc.KeysView, abc.ValuesView, abc.ItemsView, abc.Set, types.GeneratorType, map)): x = list(x)` and `elif isinstance(x, _stringtypes): x = [x]`. The string change is a small behaviour change (0-d to 1-d). `toarray()` has more than 20 internal call sites, so run the full test suite.

## Cross-checked between halves

A few claims that touch both fragments were re-verified from both directions rather than taken at face value:

- **`sc.mergedicts()` and the `sc_parallel` `maxcpu`/`maxmem`/`interval` clobbering** (originally finding 34, rejected on review): Part 2 confirmed this is standard `dict.update()` ordering behaviour, not a `mergedicts()` logic defect — the fault is in how `sc_parallel.py:152` orders its arguments, and that is where `sc_parallel_bugfixes.md` files it. It is mentioned here only because the `mergedicts()` docstring invites the pattern.
- **`sc.toarray()` does not alias its input.** The default call copies (`np.shares_memory` is False, and mutating the result leaves the caller's array untouched) — checked explicitly because several of the findings above involve `toarray()` producing wrong *values*, and it was worth ruling out a separate aliasing problem.
- **`sc.tolist('abc')` correctly gives `['abc']`.** A string is not iterated character-by-character; this was double-checked because the original audit filed two other `tolist()` findings (22 and 32, both since rejected on review) and it would have been easy to assume a third.
- **`ifelse()` has no short-circuit defect.** Python evaluates all arguments at the call site before `ifelse()` ever runs, and `ifelse()` never *calls* a callable argument, so a function object passed as a candidate is returned unevaluated (`sc.ifelse(None, expensive)` -> `<function expensive>`, zero calls) — this is correct, not a laziness bug.
- **`sc.LazyModule` is genuinely lazy.** `repr()`, `dir()`, `str()`, and `bool()` on an unimported `LazyModule` all leave the target module unimported (verified with `wave`, which Sciris does not import) — laziness itself is sound even though the module never replaces the caller's variable as its docstring claims (see finding 9).

## Misplaced `# pragma: no cover`

| Line | Context | Reachable via |
|---|---|---|
| 126 | `if secure: # pragma: no cover` in `fast_uuid()` | `sc.fast_uuid(secure=True)` -> `'UcPYkb'`; documented argument |
| 187 | `findinds()` shape mismatch (sc_math.py cross-reference, not this file) | n/a — see `sc_math_bugfixes.md` |
| 284 | `except Exception as E: # pragma: no cover` in `dcp()` | `sc.dcp(obj_holding_a_lock)` -> `ValueError`; also `tests/test_utils.py:28` |
| 428 | `else: # pragma: no cover` / `toprint = obj` in `pp()` | This is the **default** path: `jsonify=False` since v3.0.0, so every plain `sc.pp(obj)` runs it |
| 523 | `if verbose: # pragma: no cover` in `traceback()` | `sc.traceback(E, verbose=True)` printed the traceback |
| 588 | `else: # pragma: no cover` / `output = plat` in `getplatform()` | `sc.getplatform()` -> `'linux'`; the function's primary documented use and its documented return value |
| 758 | `if params is not None: # pragma: no cover` | `sc.urlopen('.../echo', params={'a':1,'b':'x y'})` -> server saw `/echo?a=1&b=x+y` |
| 760 | `if data is not None: # pragma: no cover` | `sc.urlopen('.../echo', data={'k':'v','n':2})` -> server saw `POST`, body `k=v&n=2` |
| 778 | `except Exception as E: # pragma: no cover` (decode failure) | Any response body that is not valid UTF-8; see finding 6 |
| 793 | `if filename is None and save: # pragma: no cover` | `sc.urlopen('.../disposition', save=True)`; documented `save=True` example |
| 805 | `with open(filename, 'wb') as f: # pragma: no cover` | `sc.urlopen('.../binary', filename='bin.dat')` wrote 256 bytes |
| 892 | `except Exception as E: # pragma: no cover` in `download()` | `sc.download([...bad url...], parallel=False, die=False)` |
| 933 | `if tostring: # pragma: no cover` in `htmlify()` | `sc.htmlify('a\nb', tostring=True)` -> `'a<br>b'`; it is a docstring example |
| 979, 986 | `flexstr()` non-string branch and multi-argument join | `sc.flexstr(b'foo', 'bar', [1,2])` — the docstring example |
| 1082, 1100 | `isiterable()` multi-argument and `minlen` branches | `sc.isiterable([1,2,3], 'abc', set(), exclude=str, minlen=1)` — the docstring example |
| 1161 | `checktype()` list-of-types conversion | `sc.checktype(3, [int, float])` — the v3.0.0 changelog feature |
| 1208 | `isnumber()` `isnan` check | `sc.isnumber(3, isnan=True)` -> False |
| 1241 | `isarray()` dtype match | `sc.isarray(np.array([1.]), dtype=float)` -> True |
| 1410, 1412 | `tolist()` `coerce='tuple'` / `coerce='array'` | `sc.tolist((1,2), coerce='tuple')` -> `[1, 2]` (documented option) |
| 1560 | `mergedicts()` renamed-keyword warning | `sc.mergedicts({'a':1}, strict=True)` -> FutureWarning |
| 1782 | `strjoin()` non-iterable argument | `sc.strjoin([1,2,3], 4, 'five')` — the docstring example |
| 2061 | `_assign_to_namespace()` (entire function) | any `sc.importbyname(...)` call, and every `LazyModule` attribute access |
| 2245 | `KeyNotFoundError.__str__` | `str(sc.KeyNotFoundError('a\nb'))`; also every `sc.odict` bad-key error |
| 2312, 2323 | `Link.__repr__`, `Link.__call__` update branch | `repr(sc.Link(3))`; `L(5)` (documented "if called with argument, update object") |
| 2373 | `if attr in _builtin_keys:` in `LazyModule.__getattr__` | *Corrected on review*: the original audit called this branch "genuinely dead", but it is reachable and load-bearing. `copy.copy(sc.LazyModule(...))` builds an instance with an empty `__dict__`, so `hasattr(y, '__setstate__')` goes through `__getattr__` to `_load` to `self._variable`, which reaches this guard; the guard is what prevents infinite recursion (with it, copying works and does not import the module) |
| 2464 | `tryexcept()` boolean `die` | `sc.tryexcept(die=True)` |
| 2530 | `tryexcept.__exit__` verbose print | `with sc.tryexcept(): [][2]` — the default (`verbose=1`) prints |

## Verified clean

**`cp()`, `dcp()`, `robust_dcp()`.** The ordinary `dcp` path is genuinely deep: mutating nested members of the copy left the original untouched for dicts, nested dicts, lists of lists, numpy arrays, `sc.odict`, and objects using `__slots__` (each checked by mutation, not just `is`). `memo` behaves correctly — a child shared by two parents stays shared in the copy (`c['x'] is c['y']` is `True`) while being distinct from the original (`c['x'] is shared` is `False`), and this also holds on the `robust_dcp` object branch. Circular references are handled by both paths (`c[2] is c` for a self-referential list; `cr.me is cr` and `cr is not r` for a self-referential object whose deepcopy fails). `die=True` raises `ValueError` chained from the underlying `TypeError` with a useful message. On the `robust_dcp` object branch, an attribute that can be neither deep- nor shallow-copied is left as the original reference and the attribute is *not* dropped (verified: all three attributes present, including the lock) — that is the documented contract, not a bug. `cp()` shallow-copies correctly and `cp(die=False)` warns and returns the original. The `verbose=True` default in `dcp` does mean `sc.dcp(x, die=False)` dumps every intermediate exception to stdout, which is loud but deliberate per the v3.2.4 changelog.

**`sha()`.** Dict key insertion order produces different digests, and the digest for a dict is stable across processes (three runs, identical). `1`, `1.0` and `True` are all distinct, as are `np.array([1,2,3])` vs its `int32` version, `float64` vs `float32`, and big-endian `>i8` vs little-endian `<i8` (numpy's repr surfaces the non-default dtype in each case). `np.int64(1)` and `1` collide, and `np.float64(1.0)` and `1.0` collide, but only because their reprs are identical in numpy 2.x. Long Python **lists** are not truncated (`sha(list(range(3000)))` distinguishes a single changed element) — only numpy/pandas objects are. `0.1+0.2` and `0.30000000000000004` correctly collide (same float). `digest=True`/`asint=True` are consistent with each other and with the docstring's assertions, and `encoding=` does reach the `encode()` call. `sc.sha('abc') == sc.sha(b'abc')`: a real collision, but an unavoidable consequence of encoding the string, and not worth changing. `sha()` is not called anywhere else inside Sciris.

**`fast_uuid()`, `uuid()`.** The safety guard arithmetic is right (`n_possibilities//safety`, raising when `n` exceeds it) and the `recursion_limit` path raises a clear `ValueError` rather than silently returning a short list. Return types are as documented: `n=1` -> `str`, `n>1` -> `list`, `forcelist=True` -> `list` even for `n=1`, `n=0` -> `[]`. `secure=True` works. `sc.uuid()` returns a `UUID`; `which=1/3/4/5` dispatch to the right `py_uuid` functions; `which='hex'` gives a 6-character hex string and `which='ascii', length=10, n=5` gives five distinct 10-character strings, matching the docstring examples; `sc.uuid(uid=s)` and `sc.uuid(uid=<UUID>)` both round-trip to the same `UUID`; `sc.uuid(tostring=True)` returns the full 36-character string; `length` larger than the UUID raises `ValueError` as tested. `sc.uuid(238)` binds to `uid`, not `which`, and falls through the `die=False` conversion path to a fresh UUID (which is what `tests/test_utils.py:79` relies on).

**`pp()`, `_printout()`.** All six `doprint` x `output` combinations behave as the docstring describes: `doprint=None` prints iff `output=False`; `doprint=True, output=True` both prints and returns; `doprint=False, output=False` is a deliberate no-op. `sort_dicts` defaults to `False` and works when set, `jsonify=True` handles an `sc.odict` without error, and `**kwargs` (tested with `width=20`) reach `pprint.pformat`.

**`urlopen()` / `download()`.** Against a local server: `headers` are sent and override the defaults (`User-Agent` replaced, custom `X-Test` present, verified server-side); `params` are urlencoded onto the query string (but see finding 41 for URLs that already have one); `data` switches the request to `POST` with the correct `application/x-www-form-urlencoded` body; 302 redirects are followed transparently; `response='json'` returns a `dict` and `response='full'` returns an open, readable handle. No file-descriptor leak on either the success path or the 404 error path (fd count flat at 4 across 30 iterations, with and without `gc.collect()`), and `-W always` produced no `ResourceWarning` on the success path — CPython closes the socket when the response is dropped. Binary saves are byte-exact (all 256 byte values written). For `download()`, content-to-filename correspondence is exact for five distinguishable payloads under both `parallel=True` and `parallel=False`, order is preserved, both `{filename: url}` and the legacy `{url: filename}` dict orders are detected correctly, and a mid-batch failure with `die=False` leaves **no** partial or truncated files on disk while the other downloads complete normally; `die=True` raises. `prefix` handling adds `http://` only when missing.

**`htmlify()`.** No double-escaping: `html.escape` handles `&` before `<`/`>`, so `'5 < 6 & 7 > 2'` -> `b'5 &lt; 6 &amp; 7 &gt; 2'` and `'&lt;'` -> `b'&amp;lt;'` (correct single escaping, not `&amp;amp;lt;`). Non-ASCII becomes numeric character references, newlines become `<br>`, tabs become four `&nbsp;`, and `tostring=True` decodes cleanly. The first and third docstring examples are exact.

**`traceback()`.** Inside an `except` block the full traceback is returned with the correct final line; explicit exception objects work, including the docstring's `sc.tryexcept()` example for both a `KeyError` and an `IndexError`; `verbose=True` prints and still returns the string; `**kwargs` reach `format_exception`/`format_exc`.

**`asciify()`.** The docstring example is exact (`sc.asciify('föö→λ ∈ ℝ') == 'foo  R'`). Combining characters decompose and drop correctly (`'éclair'` -> `'eclair'`, both precomposed and decomposed input), emoji and CJK are removed (`'🎉 emoji 🚀'` -> `' emoji '`, `'中文字'` -> `''`), ligatures and compatibility forms expand as NFKD requires (`'ﬁ'` -> `'fi'`, `'Ⅻ'` -> `'XII'`, fullwidth -> ASCII, `'½'` -> `'12'` since the fraction slash is non-ASCII), zero-width space is dropped, and characters with no decomposition are removed rather than approximated (`'ß'` -> `''`, `'Æ'` -> `''`), which is the documented `errors='ignore'` behaviour. `form=` and `errors=` both reach their calls (`form='NFC'` leaves nothing to strip and yields `'clair'`; `errors='replace'` gives `'fo?o???'`; `errors='strict'` raises).

**`getplatform()` and friends.** Alias normalization is correct for `darwin`/`osx`/`macos` -> `'mac'`, `win`/`win32`/`cygwin`/`nt` -> `'windows'`, `linux`/`posix` -> `'linux'`; `sc.getplatform('posix')` is `True` on Linux; `sc.iswindows() + sc.ismac() + sc.islinux() == 1` holds; an unknown `expected` string returns `False` rather than raising. `getuser()` returns the expected username. `isjupyter()` returns `False` and `isjupyter(detailed=True)` returns the "installed but not running" string outside IPython; the IPython-shell classification branch (line 644 onwards) was not exercised.

**`mergedicts()`.** All five documented flags work as advertised — `_strict=True` raises `TypeError` naming the offending argument index, `_overwrite=False` raises `KeyError` listing the overlapping keys, `_copy=True` deep-copies (so mutating the result no longer touches the inputs), `_die=False` downgrades both the non-dict-argument error and the `_sameclass` failure to a warning/skip, and `_sameclass=True` reproduces the class of the first *dict* argument even when earlier arguments are `None` (`sc.mergedicts(None, sc.objdict(a=1))` -> `objdict`) while falling back to `dict` when that class cannot be instantiated empty. All four docstring examples produce the documented results, including the `odict` ordering in `d3`. Merge order is args in order then `**kwargs` last, and `None` arguments are skipped. Aliasing was checked deliberately: the default (`_copy=False`) result is a *shallow* merge, so nested containers are shared with the inputs and mutating `merged['k']['x']` mutates the caller's dict — this matches the documented meaning of `_copy` ("whether or not to deepcopy the merged dictionary") and the `dict.update()` analogy, so it is not filed as a defect; nested dicts are replaced wholesale rather than recursively merged, which is also consistent with `dict.update()` and the `|` operator comparison in the docstring.

**`toarray()`.** Scalars, `bool`, 0-d arrays, `pandas.Series` and `pandas.DataFrame` (via `.values`, preserving 2-D shape), `dtype=` forwarding (including the intended truncation of `dtype=int` on floats), and `**kwargs` pass-through to `np.array()` (verified with `copy=False`, which correctly returns a view) all behave as documented. The deprecated `skipnone=True` still works and warns. The default call *copies* its input (`np.shares_memory` False, and mutating the result leaves the caller's array untouched), so there is no aliasing hazard. The v3.1.0 mixed-type claim holds for `str`-flavoured mixtures: `sc.toarray([1,'foo'])` and `sc.toarray([1.5,'foo'])` give `dtype=object` with the number preserved, and `sc.toarray([1,None])` gives `object` as well; ragged input raises numpy's own `ValueError` rather than silently building an object array. (Not covered by the original audit and added on review: dict views, sets, generators and strings give a 0-d array, finding 43.)

**`tolist()`.** A string stays whole (`sc.tolist('abc')` -> `['abc']`, including with `coerce='full'`), dicts are wrapped (`[{...}]`) while `dict_keys`/`dict_values`/`dict_items`/`range`/`map` are coerced, arrays and tuples are wrapped by default and coerced under `coerce='array'`/`'tuple'`/`'full'`, `coerce='none'` disables coercion, and all six docstring examples match. Generators are wrapped rather than coerced (asymmetric with `map`, but no data is lost and nothing is consumed). An input that is already a `list` is returned *by identity*, so `sc.tolist(mylist).append(x)` mutates the caller's list — undocumented but consistent with the "make sure it's a list" contract and relied on by `mergelists()`, which extends into a fresh list.

**`checktype()`/`isnumber()`/`isarray()`/`isiterable()`/`isstring()`/`isfunc()`/`ismodule()`.** (The review found that `checktype()` rejects ABCs and custom-metaclass types, finding 39; that case was not in the original type sweep.) Cross-checked over `int`, `float`, `bool`, `np.bool_`, `np.float32`, `np.int64`, 0-d and 1-d arrays, `Decimal`, `Fraction`, `complex`, `str`, `bytes`, `set`, a generator, `None`, and NaN. Apart from the `np.bool_` gap (finding 20), the four functions agree: `Decimal`/`Fraction`/`complex` are numbers to both `isnumber()` and `checktype(..., 'number')`; `bool` is both a number and `'bool'`; `str`/`bytes` are `'str'` and iterable but not `'listlike'`; 0-d arrays are `'array'`/`'listlike'` but not iterable; `None` is False everywhere and `checktype(None, 'none')` is True. `isnumber(x, isnan=True/False)` filters correctly for `float`, `np.float64`, `np.float32` and complex NaN (it raises `TypeError` for `Decimal`/`Fraction`, which `np.isnan` cannot handle — noted but not filed, being an unhelpful error rather than a wrong answer). `isiterable(minlen=)` and `isiterable(exclude=)` behave as documented for type and list-of-type arguments and reproduce the docstring's `[True, False, False]`; `exclude='str'` (the string spelling that `checktype()` itself accepts) raises `TypeError` from `isinstance`, which is an inconsistency in the `exclude` contract but not a wrong result. `checktype()`'s `subtype` recursion, the `'arraylike'` numeric-entry special case, and `die=True` (which returns `None` rather than `True` on success — deliberate, per the docstring) all check out; one side effect worth knowing is that the `subtype` loop iterates the object even when the result is already False, so passing a generator exhausts it.

**`tryexcept()`.** Apart from findings 11 and 25, everything works — `die=<exception>` raises that type and catches others, `catch=<exception>` catches that type and raises others, `verbose=0/1/2` control silence/one-line/full traceback, `message` substitutes `<EXCEPTION>`, the swapped-argument form `sc.tryexcept(KeyError, 'msg')` is detected and reordered, `history=` accumulates across a loop (3 entries for 3 failures out of 5 iterations) and reusing a *single* object accumulates identically without losing entries, and `exception`, `exceptions`, `died`, `__len__`, `to_df()` and `traceback(tostring=True)` all return the retained exceptions afterwards. (`exceptions`' docstring says "Retrieve the last exception" — a copy-paste of `exception`'s — while it returns all of them.)

**`importbypath()`.** The auto-unique naming path works (two loads of the same file become `goodmod` and `goodmod1`, distinct module objects), `__init__.py` handling and the folder/file name derivation are correct, a non-existent path raises `FileNotFoundError`, and the save/restore of a pre-existing `sys.modules[orig_name]` behaves as documented on the success path. Re-importing the same path twice can return stale code when the file was edited to the *same byte length* within the same second, but that is CPython's `__pycache__` invalidation rule (mtime + size) and applies equally to a normal `import`, so it is not a Sciris defect.

**`LazyModule`.** Genuinely lazy — constructing it, `repr()`, `str()`, `dir()` and `bool()` all leave the target module unimported (checked with `wave`, which Sciris does not import); only a real attribute access imports it.

**`Link`.** `Link(obj)()` returns the identical object, `uid` is picked up when present and `None` otherwise, `copy.copy`/`copy.deepcopy`/`sc.dcp` all break the link and re-raise `LinkException` on call (including when the link is nested inside a container), and `L(newobj)` updates in place.

**`ifelse()`.** All three docstring examples produce the documented answers; `default=` is returned when nothing matches (and for no arguments at all); `check=None`/`bool`/`True`/`1`/callable are all accepted. There is no short-circuiting defect to report: Python evaluates the arguments at the call site, and `ifelse()` never *calls* an argument, so a function object passed as a candidate is returned unevaluated (`sc.ifelse(None, expensive)` -> `<function expensive>`, zero calls). Minor asymmetry: `check=1` is accepted as `bool` but `check=0` raises `ValueError`.

**`strjoin()`/`newlinejoin()`/`sanitizestr()`/`transposelist()`/`uniquename()`/`autolist()`/`suggest()` (non-jaro).** All docstring examples reproduce exactly, including all four `sanitizestr()` unicode examples (`'Lukas_wanted_*500*'`, `'??scattering??Mariasaid??at?5?m??'`, `'_4pathnamestovariable'`), the three `uniquename()` cases (`'out2'`, `'file (3)'`, `'results2.csv'`), `transposelist()`'s `fix_uneven=True` padding (and its documented silent truncation when `fix_uneven=False`), `autolist`'s `+=`/`+` returning `autolist`, and `sc.suggest()`'s case- and whitespace-free tie-breaking for `which='damerau'`/`'levenshtein'` when the *input* is lowercase (uppercase input is not normalized, finding 40) with `n`, `threshold`, `threshold=-1`, `fulloutput` and the no-match `None` return.

## Rejected on review

These findings from the original audit were removed from the severity sections on re-verification (2026-09-25). Numbers are the original ones.

- **5.** `sc.sha()` digests not stable across processes — NOT WORTH FIXING: `sha()` hashes `repr()` by design and never promises cross-process stability, so an address-based `repr` is expected to vary; add a docstring sentence instead.
- **7.** `sc.toarray()` stringifies numbers in mixed number/`bytes` input — NOT WORTH FIXING: an extreme corner case; the one-line `np.character` guard is harmless to fold into other `toarray()` work if wanted.
- **15.** `sc.urlopen()` has no timeout — NOT A BUG: a feature request, and no timeout is the default for both `urllib` and `requests` (a `timeout=None` pass-through is a reasonable enhancement; a finite default would break slow downloads).
- **18.** `sc.download(die=False)` failure sentinel differs between parallel and serial — NOT WORTH FIXING: both modes mark the failure and warn, and the output values are correct, so the difference is cosmetic.
- **19.** `sc.htmlify(reverse=True)` is not an exact inverse — NOT WORTH FIXING: `reverse=True` is documented as "convert HTML to string", not as an inverse; only the `<br>`-before-`unescape` reorder is valid, and the proposed tab-restoring fix would mangle real HTML containing four `&nbsp;`.
- **22.** `sc.tolist()` raises on `None` when `objtype` is given — NOT WORTH FIXING: rejecting `None` as a type mismatch when a type was explicitly requested is defensible, and `sc.suggest('foo', None)` raising is the right outcome for invalid input.
- **26.** `sc.traceback()` with no argument returns `'NoneType: None'` outside an `except` block — NOT A BUG: it is documented as an alias for `traceback.format_exc()`, and this is exactly `format_exc()`'s behaviour.
- **27.** `sc.asciify()` does not apply `encoding` to the decode — NOT WORTH FIXING: `encoding='utf-16'` makes no sense for a function whose purpose is to produce ASCII.
- **29.** `sc.htmlify()` second docstring example shows the wrong output — NOT A BUG: a docstring typo (the input shows `\n` where it should show `\t`), trivial to fix but not a code defect.
- **30.** `sc.flexstr(None)` returns `''` and a leading list is flattened — NOT A BUG: this follows the documented `mergelists` semantics, and the proposed fix would make `sc.sanitizestr()` with no argument return `'None'`.
- **31.** `sc.toarray()` casts homogeneous string lists to object dtype — NOT A BUG: the `asobject` argument is documented as "prefer to coerce arrays to object type rather than string", which is exactly what it does; the fix would change the dtype for every string caller.
- **32.** `sc.tolist()` builds but never raises the unrecognized-`coerce` error — NOT WORTH FIXING: invalid input still raises, just with a worse message (adding the one-word `raise` is harmless if anyone touches the function).
- **33.** `sc.swapdict()` drops entries for duplicate values — NOT A BUG: the docstring says it is equivalent to `{v:k for k,v in d.items()}`, and losing duplicates is inherent to swapping.
- **34.** `sc.mergedicts()` eats keys named `_copy` etc. and does not skip `None` values — NOT A BUG: data keys literally named `_copy` are contrived, and a later `None` overriding an earlier value is standard `update()` semantics, as the original finding itself conceded.
- **35.** `sc.runcommand()` gives no way to detect a failed command — NOT A BUG: a feature request (a way to get the return code).
- **36.** `sc.importbyname(path=...)` applies `path` to every extra keyword — NOT WORTH FIXING: combining `path=` with extra `**kwargs` modules is contrived.
- **37.** `sc.Link` is broken by `sc.dcp()` but preserved by pickling — NOT A BUG: pickle keeping the link intact is fine and arguably desirable; at most a docstring wording issue.
- **38.** `sc.LazyModule` never replaces the caller's variable — NOT WORTH FIXING separately: same root cause as finding 9, and fixing 9 fixes it; the proxy works functionally and repeat imports hit the `sys.modules` cache.

## Suggested order of work

1. **Findings 3, 13, 42 and 2** (`robust_dcp()`: memo truncation, raising on plain containers, `defaultdict` rebuild, tuple aliasing) — all corrupt or crash the `die=False` path of the most-used function in the library, and all are localized to `robust_dcp()`. **3 and 13 must ship together**: the memo fix for 3 alone turns silent truncation into the finding-13 crash for pure-dict structures. 42 touches the same container-rebuild lines as 13.
2. **Finding 4** (`sha()` repr-based hashing of long arrays/DataFrames) — note in the changelog that the fix changes every array/DataFrame digest.
3. **Finding 1** (`fast_uuid` duplicate/over-length output) and **finding 8** (`suggest(which='jaro')` inverted ranking) — small, self-contained, high-impact fixes. Finding 40 (`suggest()` not lowercasing the input) is a one-line fix in the same loop.
4. **Findings 9, 10** (`importbyname`/`LazyModule` assigning into the wrong namespace; `importbypath` leaving a broken module registered) — process-lifetime state bugs (`_assign_to_namespace()`, and missing cleanup on import failure).
5. **Finding 11** (`tryexcept` swallowing `BaseException`) — a one-line guard with an outsized safety impact (Ctrl-C/`sys.exit` being silently eaten); fix alongside 25 (`catch` ignored when `die` is a bool), since both live in `tryexcept`.
6. **Findings 16, 20, 24, 39, 41** — the remaining medium-severity items: `urlopen(response='status')` on errors (use the restricted fix), `np.bool_` in `isnumber()`, `runcommand(wait=False)` blocking, `checktype()` rejecting ABCs, and `urlopen(params=)` on URLs with a query string.
7. **Findings 6, 12, 14, 17, 21, 23, 28, 43** — low-severity items: declared charset, `uuid(uid=...)` ignoring arguments, `getplatform` on unknown platforms, `download(save=False)` key collisions, `toarray(None, keepnone=True)`, multi-character `strsplit` separators (changelog note needed), `Content-Disposition` quotes, and `toarray()` of dict views/sets/strings.

Findings that **silently return wrong or incomplete data** (highest priority, because nothing announces the failure): the `dcp` memo truncation (3), the `robust_dcp` tuple alias (2), `sha()` collisions (4), `fast_uuid` returning duplicates (1), `suggest(which='jaro')` (8), and the malformed `params` URL (41). Findings that instead **raise or merely misbehave** (still worth fixing, but self-announcing): `robust_dcp` raising on plain containers and `defaultdict` (13, 42), `checktype()` rejecting ABCs (39), `getplatform`'s `KeyError` (14), `runcommand(wait=False)` blocking (24), and most of the low-severity argument-ignoring items.
