# `sc_odict.py` bug audit

Audit of `sciris/sc_odict.py` (1563 lines) for genuine defects: wrong values, documented arguments that don't work, silent data corruption, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps, and performance. The file was covered in full by two parallel auditors working on adjacent line ranges: lines 24-1195 (the `counter` and `odict` classes) and lines 1196-1563 (`objdict`, `dictobj`, `asobj`, `argparse`). **Method**: line-by-line reading of each class/method, followed by executed hypothesis tests against the editable install (Sciris 3.3.0, numpy 2.4.6, commit `2d69aad`), with every finding reproduced a second time independently before being recorded here.

This document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`; `sc_odict.py` is unchanged since `2d69aad`), with every finding re-executed under `SCIRIS_BACKEND=agg`. That review confirmed 13 of the original 26 findings, found 3 to be accurate in substance but with a wrong proposed fix (#9, #10, #16, now rewritten), and rejected 10 as not a bug or not worth fixing (listed under "Rejected on review" near the end, along with the misplaced-pragma table). It also found 6 bugs the original audit missed (#27-#32). Several severities and proposed fixes were corrected as a result; where an original fix was shown to be wrong, the finding says so. The document now lists **22 findings**: 7 High, 7 Medium, 8 Low.

**Nothing in this document has been applied.** All fixes are described, not made.

`sc.odict` is the library's central container — used throughout Sciris itself and by every downstream package that imports `sciris as sc` — so a silent index/key inconsistency here has unusually wide reach: it doesn't just affect one function, it can quietly corrupt any code that indexes an `odict` numerically after a mutation.

## Summary

| # | Severity | Class/method | Defect | Line |
|---|----------|--------------|--------|------|
| 1 | High | `odict.pop()`, `odict.remove()`, `odict.__delitem__()` | `pop()`/`remove()`/`del` by index or slice leave the key cache stale, so later integer indexing returns the wrong element | 561 |
| 2 | High | `odict.setdefault()`, `odict.__ior__()`, `odict.popitem()`, `odict.clear()` | These bypass the key cache, silently changing what `od[i]` means | 156 |
| 4 | High | `odict.insert()` | `insert()` with an already-present key silently throws the new value away | 753 |
| 5 | High | `odict.sort()` | In-place `sort()` keeps the keys its docstring says it drops, and moves them to the front | 864 |
| 6 | High | `sc.dictobj()` | `dictobj` never overrides `dict.__eq__`, so every instance compares equal to every other | 1387 |
| 7 | High | `sc.argparse()` | Converts command-line values by calling the default's class, so a `False` default can never be turned off and a list default is split into characters | 1542 |
| 23 | High | `odict.insert()`, `odict.append()` | Cannot store a value of `None`; `append(key, None)` silently stores the key as the value | 727, 699 |
| 3 | Medium | `odict.__setitem__()`, `odict.pop()` | An integer key is read as a key but written and popped by position | 237 |
| 8 | Medium | `odict.sort()`, `odict.sorted()` | `sort(sortby=..., reverse=True)` reverses the caller's list in place, and crashes for a numpy array of keys | 848, 858 |
| 9 | Medium | `odict.copy()` | `copy()` silently discards the `defaultdict` behaviour | 782 |
| 11 | Medium | `odict.disp()` | `numformat=...` is ignored, including in its own docstring example | 455 |
| 27 | Medium | `odict.sort()`, `odict.sorted()` | A numpy boolean mask (e.g. `sortby=od[:] > 2`) is rejected with `TypeError` | 849 |
| 29 | Medium | `odict.rename()` | A negative numeric index other than `-1` puts the renamed key in the wrong place | 807-811 |
| 30 | Medium | `odict.findbyval()`, `odict.filtervals()` | Crashes whenever any value in the odict is a numpy array | 646 |
| 10 | Low | `odict.sort()`, `odict.sorted()`, `odict.filter()`, `odict.map()`, `odict.fromeach()`, `odict.reverse()` | An odict with integer keys crashes in these methods | 861 |
| 13 | Low | `sc.counter` | Arithmetic (`+`, `-`, `|`, `&`) returns a plain `collections.Counter`, losing every feature the class adds | 24 |
| 16 | Low | `odict.enumkeys()`, `enumvals()`, `enumitems()`, `items()`, `iteritems()` | `transpose=True` raises `ValueError` on an empty odict | 1090 |
| 21 | Low | `odict.fromeach()` | Docstring says it returns an array; it returns an odict (docs only) | 1048 |
| 24 | Low | `sc.argparse()` | Resets `_parsed` to `False` after parsing, so the `add()`-after-parse guard never fires for the documented one-line usage | 1509 |
| 28 | Low | `odict.reverse()`, `odict.reversed()`, `odict.sort()` | Crash on odicts with tuple keys | 847-855, 878 |
| 31 | Low | `odict.__radd__()` | `dict + odict` gives the left operand precedence on duplicate keys, unlike `odict + dict` | 424-427 |
| 32 | Low | `odict.make()` | `make(n)` fills with `[]`, not `None` as the docstring says | 930-933 |

## Recurring patterns

**A hand-rolled key cache with no invalidation invariant.** `odict` keeps a manual `_cached_keys` list that is only correct if every mutation path remembers to set `_stale`, and `_ikey()` clears `_stale` as a side effect of *reading*, which races with the mutation that is supposed to set it. This single structural problem accounts for several separate findings at once: `pop()`/`remove()`/`__delitem__()` by index or slice (#1), and the inherited `dict` methods `setdefault()`, `__ior__()` (`|=`), `popitem()` and `clear()`, none of which touch `_stale` at all (#2). Because the corrupted `_cached_keys` list lives in the instance `__dict__`, it is pickled with the object, so a saved-and-reloaded file can carry the wrong cache forward. The original audit suggested that a single length check in `_ikey()`/`_cache_keys()` (`len(self) != len(self._cached_keys)` triggers a refresh) would neutralize all of these at once. Re-verification showed that is **not sufficient**: a same-length sequence of mutations (e.g. `popitem()` followed by `setdefault()`) slips past it (see #1 and #2). A length check is fine as an extra safety net, but the real fix is to set `_stale` correctly on every mutation path.

**Unguarded dict methods on `dictobj`.** `dictobj` writes always land in `__dict__`, plain `dict` storage stays empty, and any dict method not on the explicit delegation list (notably `__eq__`, `__or__`, `__reversed__`) silently operates on the empty real dict instead of failing loudly — any future dict method added to CPython is a new instance of this bug (#6).

**Numeric keys resolved as a key on read but as a position on write.** `__getitem__` and `__delitem__` try the key first and fall back to position; `__setitem__` and `pop()` go straight to position, with no fallback to an existing key. That inconsistency produces the silent mis-write on integer-keyed odicts (#3). The crashes in `sort()`/`sorted()`/`filter()`/`map()`/`fromeach()`/`reverse()` (#10) have a related cause: those methods populate a fresh empty odict via `new[key] = val`, and an integer key is taken as a position into that (empty) target. Note that the #3 fix does **not** fix #10 on its own (see #10). The class docstring calls integer keys "discouraged", which is why both are rated below High.

**`None` used as "argument not supplied".** `insert()`, `append()`, and `disp()`'s `numformat` parameter all conflate "not given" with the legitimate value `None`, producing three distinct defects: a discarded value, a discarded key, and an ignored argument.

**In-place and copy variants of the same operation implemented separately.** `sort(copy=True)` and `sort(copy=False)` are independent code paths, and only the copy path implements the documented key-filtering behaviour; likewise `copy()` and `dcp()` differ on `defaultdict` preservation. The sort pair would benefit from the in-place version being defined in terms of the copy version (e.g. `new = self.sorted(...); dict.clear(self); self.update(new)` — note the order; see #5).

**`sort()`'s type sniffing on `sortby`.** `sort()` decides what `sortby` means by checking whether every element is a `str`, a `bool` or a `numbers.Number`. Anything else is rejected, including numpy booleans (#27) and tuple keys (#28), and the string branch aliases the caller's list (#8).

## High severity

### 1. `pop()`/`remove()`/`del` by index or slice leave the key cache stale, so later integer indexing returns the wrong element — `sc_odict.py:561`

`pop()` sets `_stale = True` (line 557) and then calls `self._ikey(key)` to resolve the index; `_ikey()` sees the stale flag, calls `_cache_keys()`, and that *clears* `_stale` back to `False` (line 158) — all **before** `dict.pop()` actually removes the key. The cached key list therefore still contains the deleted key, and nothing will ever refresh it, so every subsequent integer or slice lookup is resolved against the pre-deletion key order. Every mutator that resolves a numeric key through `_ikey()` is affected: `pop(int)`, `pop(slice)`, `remove(int)`, and `del od[int]` (the `# pragma: no cover` branch at line 435, which is the "allow deletion by index" feature added in v2.0.1). `objdict` inherits the same bug.

```python
od = sc.odict(a=1, b=2, c=3, d=4)
od.pop(0)                 # remove the first item by index
print(od.keys(), od[1], od[2], od[1:3])
```

Actual:

```
['b', 'c', 'd'] 2 3 [2 3]
```

Expected: `['b', 'c', 'd'] 3 4 [3 4]`.

When the surviving key list is not a prefix of the cached one, the same bug raises instead of lying:

```python
od = sc.odict(a=1, b=2, c=3); od.pop(1)
od[1]           # KeyError: 'b'   (valid index, len(od) == 2)
od[-1]          # returns 3 only by luck
od._cached_keys # ['a', 'b', 'c'] with _stale == False
```

```python
od = sc.odict(a=1,b=2,c=3,d=4); od.pop(slice(0,2))
od[0]           # KeyError: 'a'
od = sc.odict(a=1,b=2,c=3); del od[1]
od[1]           # KeyError: 'b'
od = sc.odict(a=1,b=2,c=3); od.remove(0)
od[0]           # KeyError: 'a'
```

The corrupted cache is part of the instance `__dict__`, so it is pickled with the object, and the wrong answers survive a save/load round trip:

```python
import pickle
od = sc.odict(a=1,b=2,c=3); od.pop(0)
od2 = pickle.loads(pickle.dumps(od))
print(od2.keys(), od2._cached_keys, od2[1])   # ['b','c'] ['a','b','c'] 2   (want 3)
```

Blast radius: `pop()` by index is a headline `odict` feature; `insert()` (line 755) and in-place `sort()` (line 864) both call `self.pop()`, but only ever with string keys, so they escape. `tests/test_odict.py:194` and `:196` do call `o2.pop(0)` and `o4.pop(slice(-1))`, but never index the dict afterwards, which is why this has never been caught. This (together with #2) is the most serious bug in the file: silent wrong values from a headline feature, dating back to the OrderedDict-era `pop` logic.

**Fix**: in `pop()` and in `__delitem__`'s numeric branch, resolve the key first and set `_stale = True` *after* the deletion — e.g. `thiskey = self._ikey(key); out = dict.pop(self, thiskey, ...); self._setattr('_stale', True); return out`. Same reordering for the slice branch. Wrapping the body in `try/finally` so it always ends stale is equivalent. The shortcut originally suggested here and in #2 (refresh whenever `len(self) != len(self._cached_keys)`) must not be the only fix, because a same-length mutation sequence slips past it (see #2); it can be kept as an additional safety net.

### 2. `setdefault()`, `|=`, `popitem()` and `clear()` bypass the key cache, silently changing what `od[i]` means — `sc_odict.py:156`

`odict` keeps a manual `_cached_keys` list that is only invalidated by the methods it overrides (`__setitem__`, `setitem`, `update`, `pop`, `__delitem__`). The inherited `dict.setdefault()`, `dict.__ior__()` (`|=`), `dict.popitem()` and `dict.clear()` all mutate the dict without touching `_stale`, so integer indexing keeps resolving against the old key list.

```python
od = sc.odict(a=1, b=2)
od[-1]                  # 2 -- this call is what populates the cache
od.setdefault('c', 3)
print(len(od), od[-1])  # 3 2      <- od[-1] should be 3
od[2]                   # IndexError: index 2 out of range for dict of length 3
```

Actual output, verbatim:

```
len 3 | od[-1] = 2 (want 3)
od[2] -> IndexError index 2 out of range for dict of length 3
```

The error message is self-contradicting ("index 2 out of range for dict of length 3"), which is the giveaway. The other three:

```python
od = sc.odict(a=1,b=2); od[0]; od |= {'c':3}
od[-1]      # 2, want 3

od = sc.odict(a=1,b=2,c=3); od[0]; od.popitem()
od[-1]      # KeyError: 'c'

od = sc.odict(a=1,b=2); od[0]; od.clear()
od[0]       # KeyError: 'a'
```

Note that a dict-literal `sc.odict(...) | {...}` is fine (it builds a new object); only the in-place `|=` is affected.

A length-based cache check would not catch every case, because mutations can leave the length unchanged:

```python
od = sc.odict(a=1,b=2,c=3); od[0]; od.popitem(); od.setdefault('z',9)
len(od), len(od._cached_keys)   # 3, 3
od[-1]                          # KeyError: 'c'   (want 9)
```

**Fix**: override `setdefault`, `popitem`, `clear` and `__ior__` to set `_stale = True` after mutating. The original audit proposed comparing `len(self)` against `len(self._cached_keys)` as an alternative that "alone would catch all four"; as shown above, it does not. It is only acceptable as a safety net on top of the overrides.

### 4. `insert()` with an already-present key silently throws the value away — `sc_odict.py:753`

`insert()` pops the tail of the dict into `tmpdict`, writes the new item, then re-inserts the tail. It assumes the new key is not among the popped ones. If it is, the final re-insertion loop (line 758) overwrites the just-inserted value with the old one from `tmpdict`, so the value passed by the caller vanishes; if the key is *before* the insertion point, the new value is instead written into the existing slot and then the rotation puts it back in its original place. Either way the caller's value is discarded without a warning.

```python
od = sc.odict(a=1, b=2, c=3, d=4, e=5)
od.insert(1, 'c', 99)
print(dict(od))
```

Actual:

```
{'a': 1, 'c': 3, 'b': 2, 'd': 4, 'e': 5}
```

Expected: `c` at position 1 with value `99`. Instead `c` moved to position 1 (so the call clearly "did something") but kept its old value `3`, and `b` was displaced. More variants:

```python
sc.odict(a=1,b=2,c=3,d=4,e=5); od.insert(0,'c',99)  # -> {'c': 3, 'a': 1, 'b': 2, 'd': 4, 'e': 5}   (99 lost)
sc.odict(a=1,b=2); od.insert(1,'b',99)              # -> {'a': 1, 'b': 2}                            (no-op)
```

(`insert(4,'a',99)` at a position after the existing key does set `a=99`, but leaves it at position 0, so the position argument is silently ignored there.)

**Fix**: at the top of `insert()`, if `realkey` is already in `self`, `dict.pop` it first (and mark the cache stale), then insert at `pos` as normal, so that "insert" consistently means move-and-overwrite. The original audit's first suggestion, raising if the key already exists, would break a plausible existing idiom: `od.insert(0, existing_key, od[existing_key])` to move a key to the front, which works today because the "lost" value equals the kept one.

### 5. In-place `sort()` keeps the keys its docstring says it drops, and moves them to the front — `sc_odict.py:864`

The docstring promises "if a list of keys is provided, sort by that order (any keys not provided will be omitted from the sorted dict!)" and "if a list of boolean values is provided, then omit False entries". The `copy=True` branch (line 861) does exactly that. The in-place branch does not: it only pops and re-appends the *selected* keys, so unselected keys are never removed — they simply stay where they are and end up at the front of the result.

```python
od = sc.odict(a=1, b=2, c=3, d=4, e=5)
od.sort(sortby=['c','a'])
print(dict(od))
print(dict(sc.odict(a=1,b=2,c=3,d=4,e=5).sorted(sortby=['c','a'])))
```

Actual:

```
{'b': 2, 'd': 4, 'e': 5, 'c': 3, 'a': 1}
{'c': 3, 'a': 1}
```

Expected: both should be `{'c': 3, 'a': 1}`. The same for a boolean mask and for an index list:

```python
od = sc.odict(a=1,b=2,c=3,d=4,e=5); od.sort(sortby=[True,False,True,False,True])
# actual   {'b': 2, 'd': 4, 'a': 1, 'c': 3, 'e': 5}
# expected {'a': 1, 'c': 3, 'e': 5}
od = sc.odict(a=1,b=2,c=3,d=4,e=5); od.sort(sortby=[0,2,4])
# actual   {'b': 2, 'd': 4, 'a': 1, 'c': 3, 'e': 5}
```

This is silently wrong in two ways at once: the dict has the wrong length *and* the requested ordering is not the order you get, since the retained keys are prepended rather than appended. Full-permutation sorts (`sort()`, `sort('values')`, `sort(sortby=all_keys)`, `reverse()`) are unaffected, which is all `tests/test_odict.py:212-214` exercises.

**Fix**: in the in-place branch, delete the keys not in `allkeys` before re-inserting (e.g. `for key in origkeys: if key not in allkeys: dict.__delitem__(self, key)`, then mark the cache stale). The alternative originally given here, `self.clear(); self.update(self.sorted(...))`, is wrong as written: `self.sorted(...)` runs *after* `self.clear()` and so sorts an empty dict, and it also relies on `dict.clear`, which does not set `_stale` (#2). A correct version is `new = self.sorted(...); dict.clear(self); self.update(new)`.

### 6. `dictobj` stores its data in `__dict__` but never overrides `dict.__eq__`, so every `dictobj` compares equal to every other — `sc_odict.py:1387`

`dictobj` inherits from `dict` but keeps all of its contents in `self.__dict__` (`__init__` at `sc_odict.py:1362` writes `self.__dict__[k] = v`, and the delegation block at `sc_odict.py:1387-1401` forwards `__getitem__`/`keys`/`items`/... to `self.__dict__`). The real C-level dict storage therefore stays permanently **empty**, and every dict method that is *not* in that delegation list still reads the empty storage. `__eq__`/`__ne__` are not in the list, so comparison always compares `{} == {}`.

```python
import sciris as sc
a = sc.dictobj(x=1)
b = sc.dictobj(x=2)
print(a == b)            # actual: True    expected: False
print(a == {'x': 1})     # actual: False   expected: True
print(a == sc.dictobj()) # actual: True    expected: False
print([a, b].count(sc.dictobj(x=99)))  # actual: 2   expected: 0
```

Actual output (run twice, fresh interpreters, Sciris 3.3.0):

```
True
False
True
2
```

Because `list.__contains__`, `.count()`, `.index()`, `assert x == y` and `unittest`-style comparisons all go through `__eq__`, any test or lookup over `dictobj` instances silently succeeds for the wrong object. The same root cause breaks the other non-delegated dict operators, verified in the same run: `sc.dictobj(x=1) | {'y':2}` returns `{'y': 2}` (the `x` entry is dropped, and the result is a plain `dict`), and `list(reversed(sc.dictobj(x=1, y=2)))` returns `[]`. `json.dumps(dictobj)` returning `{}` is the same bug, and is the one instance the docstring warns about ("can't be automatically converted to a JSON (but will fail silently). Use `to_json()` instead") — the docstring does not warn that equality is meaningless.

Blast radius inside Sciris: `dictobj` is used in `sc_profiling.py:966`, `sc_profiling.py:1167` and `sc_profiling.py:1270` (`self.entries.append(entry)`). Those paths only ever read keys, and `sc.dataframe(self.entries)` at `sc_profiling.py:1288` was checked and works (pandas goes through `keys()`), so the library itself is not currently mis-comparing; the exposure is entirely user-facing. `tests/test_odict.py:282-290` is the only test of `dictobj` and never compares two of them.

**Fix**: add explicit `__eq__`/`__ne__` (and, for completeness, `__or__`/`__ror__`/`__ior__`/`__reversed__`/`__sizeof__`) to the delegation block, e.g. `def __eq__(self, other): return self.__dict__ == (other.__dict__ if isinstance(other, dictobj) else other)`, plus `__hash__ = None`. The structural alternative is to store contents in the real dict (`dict.__setitem__`) and let `__getattr__`/`__setattr__` proxy to it, which would fix all the non-delegated methods at once.

### 7. `sc.argparse()` converts command-line values by calling the default's class, so a `False` default can never be turned off and a list default is split into characters — `sc_odict.py:1542`

`keep_type()` does `default_type = self[key].__class__` and then `out = default_type(arg)`, where `arg` is always a string. For `bool` that is `bool("False") == True`; for `list` it is `list("x,y") == ['x', ',', 'y']`. Boolean flags with a `False` default (the overwhelmingly common CLI case) therefore silently become `True` no matter what the user types, and there is no way to pass `False` at all.

Repro (`apv.py`, run as a subprocess with a controlled `sys.argv`):

```python
import sys, sciris as sc
args = sc.argparse(verbose=False, tags=['a','b'])
print(sys.argv[1:], '->', dict(args))
```

Actual output:

```
$ python apv.py verbose=False
['verbose=False'] -> {'verbose': True, 'tags': ['a', 'b']}
$ python apv.py verbose=no
['verbose=no'] -> {'verbose': True, 'tags': ['a', 'b']}
$ python apv.py tags=x,y
['tags=x,y'] -> {'verbose': False, 'tags': ['x', ',', 'y']}
```

Expected: `verbose=False` -> `False`, `verbose=no` -> `False` (or an error), `tags=x,y` -> `['x','y']` (or an error). The docstring promises "converts them to the correct type"; the type is right and the value is wrong, with no warning printed (the `print()` at `sc_odict.py:1545` only fires when the constructor *raises*, and `bool()`/`list()` never raise on a string). `int`, `float` and `str` defaults were verified correct.

Blast radius: `sc.argparse` is only defined and exported here (`__all__` at `sc_odict.py:16`); it is not used elsewhere in Sciris and has **no test coverage at all** (`grep -rn argparse tests/*.py` returns nothing).

**Fix**: special-case `bool` (accept `1/0/true/false/yes/no/t/f`, case-insensitive, and error otherwise) and container defaults (split on commas, converting each element to the type of the default's first element) in `keep_type()`, instead of calling `default_type(arg)` unconditionally.

### 23. `insert()` and `append()` cannot store a value of `None` — `sc_odict.py:727`, `sc_odict.py:699`

*(Originally rated Low. Raised to High on re-verification: storing `None` is ordinary, and the `append` case silently corrupts the data rather than raising.)*

Both use `None` as the "argument not supplied" sentinel and check it with `is None`, so a legitimate `None` value is re-interpreted as a missing argument and the arguments shift.

```python
sc.odict(a=1,b=2,c=3).insert(2, 'k', None)
# IndexError: index 2 out of range for dict of length 0
#   (falls into the insert('devil', 666) form: realkey=2, realvalue='k', realpos=0)

od = sc.odict(); od.append('mykey', None); print(dict(od))
# {'key0': 'mykey'}   -- the key became the value
```

Expected: `{'a':1,'b':2,'k':None,'c':3}` and `{'mykey': None}` respectively. `insert(pos=2, key='k', value=None)` fails identically, so there is no keyword workaround.

**Fix**: use a private sentinel object (e.g. `_none = object()`) as the default for `key`/`value` instead of `None`. This changes the public signatures of `insert()` and `append()`, so re-run `cd docs && python make_api.py` afterwards, or `tests/test_api.py` will fail.

## Medium severity

### 3. An integer key is read as a key but written and popped by position — `sc_odict.py:237`

*(Originally rated High. Lowered to Medium on re-verification because the class docstring describes integer-keyed odicts as "discouraged".)*

The class docstring states: "If an odict has integer keys and the keys do not match the key positions, then the key itself will take precedence (e.g., `od[3]` is equivalent to `dict(od)[3]`, not `dict(od)[od.keys()[3]]`)", and `makefrom()`'s docstring example uses integer keys (`sc.odict.makefrom({12:'monkeys', 3:'musketeers'})`). `__getitem__` honours this by trying `dict.__getitem__` first (line 178) and `__delitem__` does the same (line 433). `__setitem__` and `pop()` do not: they test `isinstance(key, sc._numtype)` first and go straight to `self._ikey(key)`, i.e. by position. So a read and a write with the same subscript refer to different entries.

```python
od = sc.odict.makefrom({10:'a', 3:'b', 20:'c', 30:'d'})
print(od[3])          # 'b'  -- key 3, as documented
od[3] = 'ZZZ'
print(dict(od))
```

Actual:

```
'b'
{10: 'a', 3: 'b', 20: 'c', 30: 'ZZZ'}
```

Expected: `{10: 'a', 3: 'ZZZ', 20: 'c', 30: 'd'}`. The write landed on key `30` (position 3) and the entry the caller read back is untouched — silent data corruption with no error at any point.

`pop()` disagrees with `__delitem__` on the same dict:

```python
od = sc.odict.makefrom({10:'a', 3:'b', 20:'c', 30:'d'})
od.pop(3)     # 'd'   (position 3)
del od[3]     # removes key 3, leaving {10:'a', 20:'c', 30:'d'}
```

So `od[3]`, `od[3] = x`, `od.pop(3)` and `del od[3]` give three different meanings for the subscript `3` on one object.

**Fix**: mirror `__getitem__`'s precedence in `__setitem__` and `pop()`, in this narrow form only: for a numeric key, if `dict.__contains__(self, key)` use it directly as a key, otherwise fall back to `_ikey()` (positional). String-keyed odicts are unaffected because an int is never among their keys. Do not extend this to treat an *absent* integer key as a new key: that would silently change `od[5] = x` on a string-keyed odict from an `IndexError` into key creation.

### 8. `sort(sortby=..., reverse=True)` reverses the caller's list in place, and crashes for a numpy array of keys — `sc_odict.py:858`

When `sortby` is a list of keys, `allkeys = sortby` (line 848) aliases the caller's list rather than copying it, and `allkeys.reverse()` (line 858) then mutates it.

```python
mylist = ['b','a','c']
sc.odict(a=1,b=2,c=3).sorted(sortby=mylist, reverse=True)
print(mylist)
```

Actual: `['c', 'a', 'b']`; expected: `['b', 'a', 'c']` unchanged. Calling `od.sorted(sortby=mylist, reverse=True)` twice therefore gives two different answers. (The boolean-mask and index-list branches rebuild `allkeys`, so they are not affected; `sortby='values'`/`None` are not affected.)

The same line also crashes when `sortby` is a numpy array of strings and `reverse=True`, because arrays have no `.reverse()`:

```python
sc.odict(a=1,b=2).sorted(sortby=np.array(['b','a']), reverse=True)
# AttributeError: 'numpy.ndarray' object has no attribute 'reverse'
```

**Fix**: `allkeys = list(sortby)` at line 848, which fixes both the aliasing and the numpy crash.

### 9. `copy()` silently discards the `defaultdict` behaviour — `sc_odict.py:782`

`copy()` builds the new object with `self._new(super().copy())`, which does not pass `defaultdict=`, and `_defaultdict` lives in the instance `__dict__` rather than the dict contents, so it is lost. `dcp()`/`copy(deep=True)` go through `sc.dcp()` and keep it, so the two documented copy modes behave differently in a way the docstring does not mention.

```python
dd = sc.odict(a=[1], defaultdict=list)
dd['zz'].append(1)          # works
d2 = dd.copy()
d2['zz2'].append(1)         # sc.KeyNotFoundError: odict key "zz2" not found
dd.dcp()['zz2'].append(1)   # works
```

Actual:

```
copy -> KeyNotFoundError
dcp kept it -> {'a': [1], 'zz': [1]}
```

`sorted()`, `filter()`, `map()` and `fromeach()` also return plain (non-default) odicts via `self._new()`. Whether those derived results should keep the `defaultdict` is a design choice rather than a bug, so the fix below is scoped to `copy()`. In-place `sort()` keeps it, since it mutates `self`.

**Fix**: keep `new = self._new(super().copy())` as it is, then copy the attribute across if present: `dd = self.__dict__.get('_defaultdict'); if dd is not None: new._setattr('_defaultdict', dd)`. The fix originally proposed here, `self._new(super().copy(), defaultdict=...)`, would break downstream subclasses whose `__init__` does not take a `defaultdict` argument: for example `ss.Pars.__init__(self, pars=None, **kwargs)` would treat `defaultdict=None` as a parameter, and `ss.ndict.__init__(..., **kwargs)` would pass it on to `extend()`.

### 11. `disp(numformat=...)` is ignored, including in its own docstring example — `sc_odict.py:455`

The `numformat` parameter is accepted and then hard-coded to `None` when the argument dict is assembled: `sc.mergedicts(dict(..., sigfigs=sigfigs, numformat=None, maxitems=maxitems), kwargs)`. Because `numformat` is a named parameter it never appears in `**kwargs` either, so the user's value can never reach `__repr__`. The docstring example `z.disp(numformat='%0.6f')` therefore prints the `sigfigs=5` default instead.

```python
z = sc.odict().make(keys=['a'], vals=[4.293487])
z.disp(numformat='%0.6f')
```

Actual: `#0: 'a': 4.2935`; expected: `#0: 'a': 4.293487`. (Passing it through `**kwargs` is impossible — `disp(**{'numformat':'%0.6f'})` binds the named parameter.) `sigfigs` in the same call is wired up correctly, which is why the omission is easy to miss.

**Fix**: change `numformat=None` to `numformat=numformat` on line 455.

### 27. `sort()`/`sorted()` rejects numpy boolean masks, e.g. `sortby=od[:] > 2` — `sc_odict.py:849`

*(New on re-verification.)*

The docstring says "if a list of boolean values is provided, then omit False entries". The check is `isinstance(x, bool)`, and `np.bool_` is neither a `bool` nor a `numbers.Number`. So the most natural mask, built from the odict's own values, is rejected, even after converting it with `list()`.

```python
od = sc.odict(a=1, b=5, c=3)
od.sorted(sortby=od[:] > 2)
od.sorted(sortby=list(od[:] > 2))
```

Actual: `TypeError: Cannot figure out how to sort by "[False  True  True]"` for the first and `TypeError: Cannot figure out how to sort by "[False, True, True]"` for the second. Expected: `odict(b=5, c=3)` for both.

**Fix**: `elif all([isinstance(x, (bool, np.bool_)) for x in sortby]):` at line 849.

### 29. `rename()` with a negative index other than `-1` puts the key in the wrong place — `sc_odict.py:807-811`

*(New on re-verification. The original audit said numeric renames "including `-1`" were correct; only `-1` had been tested.)*

For a numeric `oldkey`, `index` is used as-is in the rotation loop, in both `range(index+1, nkeys)` and `self.keys()[index]`. A negative index therefore rotates the wrong number of times. `-1` works only by luck.

```python
for i in [-1, -2, -3]:
    r = sc.odict(a=1,b=2,c=3,d=4,e=5); r.rename(i, 'Z'); print(i, r.keys())
```

Actual:

```
-1 ['a', 'b', 'c', 'd', 'Z']
-2 ['a', 'b', 'c', 'e', 'Z']
-3 ['a', 'b', 'e', 'Z', 'd']
```

Expected: `['a','b','c','Z','e']` for -2 and `['a','b','Z','d','e']` for -3. The key name is right but the order is silently wrong.

**Fix**: normalise the index in the numeric branch: `if index < 0: index += nkeys`.

### 30. `findbyval()` crashes whenever any value in the odict is a numpy array — `sc_odict.py:646`

*(New on re-verification.)*

`if val == value:` is evaluated for every entry. For an array value this is an element-wise comparison, and `if` then raises. Arrays are the typical odict payload (`mydict[:].sum()`), so searching an odict that contains *any* array fails, even when searching for a scalar that another entry matches. `filtervals()` inherits this.

```python
z = sc.odict(a=np.array([1,2]), b=5)
z.findbyval(5)
z.findbyval(np.array([1,2]))
```

Actual: `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()` for both. Expected: `'b'` and `'a'`.

**Fix**: compare with a helper that tolerates arrays, e.g. use `np.array_equal(val, value)` if `sc.isarray(val) or sc.isarray(value)`, otherwise `val == value`. Alternatively, wrap the comparison in `try/except ValueError` and fall back to `np.array_equal`.

## Low severity

### 10. An odict with integer keys crashes in `sort()`, `sorted()`, `filter()`, `map()`, `fromeach()` and `reverse()` — `sc_odict.py:861`

*(Originally rated Medium. Lowered to Low because the class docstring calls integer keys "discouraged". The original fix claim was also wrong; see below.)*

These methods copy entries into a freshly created empty odict with `newdict[key] = value`, and for an integer key `__setitem__` interprets it as a *position* into the (empty) target, so `_ikey()` raises. `makefrom()`'s own docstring example uses integer keys (`{12:'monkeys', 3:'musketeers'}`).

```python
od = sc.odict.makefrom({10:'a', 3:'b', 20:'c', 30:'d'})
od.sorted()
```

Actual, for each method:

```
sort: IndexError: index 3 out of range for dict of length 0
sorted: IndexError: index 3 out of range for dict of length 0
filter: IndexError: index 10 out of range for dict of length 0
map: IndexError: index 10 out of range for dict of length 0
fromeach: IndexError: index 10 out of range for dict of length 0
reverse: IndexError: list index out of range
```

Expected: an integer-keyed odict sorted/filtered/mapped by key, as for string keys.

**Fix**: in `sort(copy=True)`, `filter()`, `map()` and `fromeach()`, write into the new odict with `output.setitem(key, val)` (or `_setitem` plus marking the cache stale), since the keys there are real keys taken from `self.keys()`. The in-place `sort` path also calls `self.pop(key)` with integer keys, so it needs the #3 fix as well, or `dict.pop` plus marking the cache stale. The original audit claimed that fixing `__setitem__` (#3) "fixes all six at once". That is wrong: with #3's safe fix (existing key first, else position), `new[10] = x` on an empty target still raises, and making *absent* integer keys create new keys would silently change `od[5] = x` on string-keyed odicts from an `IndexError` into key creation.

### 13. `sc.counter` arithmetic returns a plain `collections.Counter`, losing every feature the class adds — `sc_odict.py:24`

*(Originally rated Medium; lowered to Low on re-verification.)*

`counter`'s stated purpose is "Like `collections.Counter`, but with additional supported mathematical operations", implemented by forwarding unknown attributes to `self.array`. But `Counter.__add__`/`__sub__`/`__or__`/`__and__` construct their result as a hard-coded `Counter()` (CPython's implementation does not use `type(self)`), and `counter` does not override them, so the result of any arithmetic silently drops back to the base class and the array methods disappear.

```python
c1 = sc.counter('aabbc'); c2 = sc.counter('abz')
print(type(c1 + c2).__name__)
(c1 + c2).array
```

Actual:

```
Counter
(c1+c2).array -> AttributeError 'Counter' object has no attribute 'array'
```

Expected: `counter`, with `.array`, `.max()`, `.mean()` etc. still available — i.e. `(c1+c2).max()` should work exactly as `c1.max()` does. Verified for `+`, `-`, `|` and `&`; `copy()` is fine (it uses `self.__class__`), as are pickling and `sc.dcp()`. The in-place forms are also fine: `c += c2` keeps the `counter` type. Standard `Counter` semantics are otherwise intact: `-` drops non-positive counts (`sc.counter('aabbc') - sc.counter('abz')` -> `{'a':1,'b':1,'c':1}`), `most_common()` keeps insertion order for ties (`[('a',2),('b',2),('c',1)]`), and a missing key still returns `0`.

**Fix**: override `__add__`/`__sub__`/`__or__`/`__and__` to rewrap the result, e.g. `def __add__(self, other): return counter(super().__add__(other))`. The `__i*__` overrides the original audit also proposed are not needed.

### 16. `transpose=True` raises `ValueError` on an empty odict — `sc_odict.py:1090`

All five methods (`enumkeys()`, `enumvals()`, `enumitems()`, `items()`, `iteritems()`) pass their (possibly empty) list straight to `sc.transposelist()`, which does `max([len(ls) for ls in obj])` (`sc_utils.py:1480`) and so fails on an empty sequence. The non-transposed calls correctly return `[]`.

```python
sc.odict().enumitems(transpose=True)
```

Actual: `ValueError: max() iterable argument is empty` — identically for `enumkeys`, `enumvals`, `items` and `iteritems`. Expected: a tuple of empty lists that can be unpacked the same way as a non-empty result, because the realistic caller does `inds, keys = od.enumkeys(transpose=True)` or `i, k, v = od.enumitems(transpose=True)`.

**Fix**: when the odict is empty and `transpose=True`, return `([], [])` from `enumkeys`/`enumvals`/`items`/`iteritems` and `([], [], [])` from `enumitems`. Both fixes the original audit proposed (a length guard that returns `()`, or making `sc.transposelist([])` return `[]`) still fail on the unpacking above with `ValueError: not enough values to unpack`. Low value overall.

### 21. `fromeach()`'s docstring says it returns an array; it returns an odict — `sc_odict.py:1048`

Documentation-only fix. The default is `asdict=True`, so the first example line is wrong; the second line, which passes `asdict=True` explicitly and says it returns an odict, only makes sense if the default were `False`.

```python
z = sc.odict({'a':np.array([1,2,3,4]), 'b':np.array([5,6,7,8])})
z.fromeach(2)   # docstring: "Returns array([3,7])"
```

Actual:

```
#0: 'a': 3
#1: 'b': 7
```

i.e. `odict({'a':3, 'b':7})`. `z.fromeach(2, asdict=False)` does return `array([3, 7])`.

**Fix**: change the example to `z.fromeach(2) # Returns odict({'a':3, 'b':7})` and add `z.fromeach(2, asdict=False) # Returns array([3,7])`.

### 24. `sc.argparse()` resets `_parsed` to `False` after parsing, so the `add()`-after-parse guard never fires for the documented one-line usage — `sc_odict.py:1509`

`__init__` calls `self.parse()` (which ends with `self.setattribute('_parsed', True)` at `sc_odict.py:1561`) and *then* unconditionally executes `self.setattribute('_parsed', False)`. The flag is therefore always `False` on a freshly constructed, already-parsed object, and `add()`'s guard at `sc_odict.py:1514` is dead for the "Option 1" style shown in the docstring. The added argument then silently keeps its default, because `sys.argv` has already been consumed. The impact is limited to a misuse guard that never fires.

```python
# apv.py
import sciris as sc
args = sc.argparse(verbose=False, tags=['a','b'])
print('_parsed =', args.getattribute('_parsed'))
args.add(extra=1)
print(dict(args))
```

Actual output (`python apv.py verbose=False`):

```
_parsed = False
{'verbose': True, 'tags': ['a', 'b'], 'extra': 1}
```

Expected: `_parsed = True`, and `add()` raising `ValueError('Cannot add an argument to an already parsed object')`. The guard does work in the "Option 2" flow (`sc.argparse()` with no kwargs, then `add()`, then `parse()`), which was verified separately. Note also that the error message on that line reads "Cannot **and** an argument to an already parsed object".

**Fix**: move `self.setattribute('_parsed', False)` above the `if parse and len(kwargs):` block (or make it `setdefault`-like), and fix the "and" -> "add" typo.

### 28. `reverse()`/`reversed()` crash on odicts with tuple keys — `sc_odict.py:847-855`, via `878`

*(New on re-verification.)*

Tuple keys are explicitly supported (`__getitem__`/`__setitem__` special-case them, and `findkeys` searches them). But `reverse()` passes the key list to `sort()` as `sortby`, and `sort()` only accepts lists that are all strings, all bools or all numbers, so it raises. The same applies to `sort(sortby=<list of tuple keys>)`. Plain `sorted()` works.

```python
t = sc.odict({('x',1): 1, ('y',2): 2})
t.reversed()
```

Actual: `TypeError: Cannot figure out how to sort by "[('y', 2), ('x', 1)]"`. Expected: `odict({('y',2):2, ('x',1):1})`.

**Fix**: in `sort()`, treat `sortby` as a key list when every element is an existing key, e.g. check `all(isinstance(x, sc._stringtypes) or (isinstance(x, tuple) and x in self) for x in sortby)` first. Or have `reverse()` skip the type check and reorder directly.

### 31. `dict + odict` gives the left operand precedence on duplicate keys — `sc_odict.py:424-427`

*(New on re-verification.)*

`__radd__(self, dict2)` computes `self.__add__(dict2)`, i.e. `mergedicts(self, dict2)`, so the *left* operand (`dict2`) wins and the *right* operand's keys come first. `__add__` and `dict.__or__` both give the right operand precedence, so the result depends on which side the odict is on.

```python
dict({'a': 1} + sc.odict(a=2))   # {'a': 1}
dict(sc.odict(a=1) + {'a': 2})   # {'a': 2}
```

Expected: `{'a': 2}` in both cases.

**Fix**: `else: return sc.mergedicts(dict2, self)` in `__radd__`. The result type then follows `mergedicts`' rules; wrap it in `self._new(...)` if an odict result is wanted.

### 32. `make(n)` populates with `[]` rather than `None` as documented — `sc_odict.py:930-933`

*(New on re-verification.)*

The docstring says `sc.odict().make(5)  # Make an odict of length 5, populated with Nones and default key names`. But `sc.tolist(None)` returns `[]`, which takes the `nvals==0` branch, `vallist = [sc.dcp(vals) for _ in range(nkeys)]`, so each value is an empty list.

```python
dict(sc.odict().make(5))          # {'0': [], '1': [], '2': [], '3': [], '4': []}
dict(sc.odict().make(['a','b']))  # {'a': [], 'b': []}
```

Expected (per the docstring): `{'0': None, ..., '4': None}`, and likewise `None` values for `make(['a','b'])`.

**Fix**: probably correct the docstring rather than the code, since someone may rely on the `[]` default. The alternative is to special-case `vals is None` to produce `None`s, which is a behaviour change.

## Rejected on review

These findings from the original audit were re-checked on 2026-09-25 and dropped. Their numbers are kept so that cross-references stay valid.

- **#12** Float and `np.float64` indices raise `TypeError` — NOT WORTH FIXING: `od[1.0]` fails immediately with a clear `TypeError`, the same as `list` and numpy, so nothing is silently wrong, and silently truncating `od[1.5]` would be worse.
- **#14** `rename()` onto an existing key leaves the merged entry in the wrong position — NOT WORTH FIXING: renaming onto an existing key is a caller error that destroys data by definition, and only the resulting position is affected.
- **#15** `sum([od])` and `0 + od` return the original object, not a copy — NOT WORTH FIXING: it only matters for a single-element `sum()` whose result is then mutated.
- **#17** `objdict.__setitem__` bypasses the attribute-collision guard — NOT A BUG: storing a key named `values`/`items` by subscript or constructor is legitimate (e.g. data loaded from JSON), and the proposed guard would make `sc.objdict(record_with_items_field)` raise, breaking downstream data loading.
- **#18** `sc.asobj()` redirects attribute writes so wrapped classes fail at construction — NOT WORTH FIXING: `asobj` is a niche helper whose docstring says "Use at your own risk", and the repro class has no `__setitem__` anyway.
- **#19** `pop(index, default)` ignores the default — NOT WORTH FIXING: an out-of-range index together with a default is a corner case, and `list.pop` accepts no default either.
- **#20** An odd `maxitems` in `__repr__`/`disp()` shows one fewer item than claimed — NOT WORTH FIXING: it is an off-by-one in a footer message, and only for odd `maxitems`; the defaults (20, 200) are even.
- **#22** `_matchkey()` raises the original exception instead of its informative message — NOT WORTH FIXING: it needs a key whose `__str__` raises, an extreme corner case (though the fix is trivial if that code is touched anyway).
- **#25** `sc.asobj()` returns a shallow copy, not a view — NOT A BUG: nothing promises a view, and returning a new wrapper from a conversion function is normal (like `dict(d)`).
- **#26** `sc.argparse()` misassigns `--key value` arguments — NOT A BUG: the space-separated form is not one of the documented forms, and `sc.argparse` is explicitly "ultra-simple" and `=`-based (though #7 makes this look worse in practice).
- **Misplaced `# pragma: no cover` table** (23 reachable branches marked as uncovered) — NOT WORTH FIXING: this is about coverage-annotation accuracy, not behaviour.

## Verified clean

### `counter`

Base `Counter` semantics survive subclassing: `-` drops non-positive counts, `most_common()` breaks ties in insertion order, a missing key returns `0`, `copy()` returns a `counter`, `total()`/`elements()`/`subtract()`/`update()` behave normally, and pickling and `sc.dcp()` round-trip correctly. `array` reflects `values()` in insertion order; `c[0:2]` slices the array while `c[1]` looks up the key `1` (so an integer-keyed counter is key-based, not positional); `__getattr__` correctly resolves `_fundamental_attrs` before falling through to the array, so `copy`/`deepcopy`/pickle are not intercepted; and a key literally named `sum` or `mean` does not shadow the array method (`c['sum']` is the count, `c.sum` is `ndarray.sum`). An empty `counter().max()` raises numpy's own "zero-size array to reduction" error, which is the same thing `np.array([]).max()` does.

### `odict`

**Class docstring examples.** Every example in the `odict` class docstring runs and asserts as claimed: `mydict['foo'] == mydict[0]`, `mydict[:].sum() == 21`, `enumitems()` iteration, and the whole "detailed example" block (`sorted()`, get by key/index/slice/string-slice/array, `bar[3] = [3,4,5]`, `bar[0:2] = [...]`, `bar[[0,3]] = [...]`, `rename('clam','oyster')`) produce `ant/bear/oyster/donkey` in the right order. Both `defaultdict` examples work, including `defaultdict='nested'` with `nested['b']['c']['d'] = 2`. The `counter`, `__add__`, `findbyval`, `insert`, `copy`, `makefrom` (all four), `map`, `toeach` and `promote` docstring examples all produce exactly what they claim. The `make` examples produce the right keys and structure, but the "populated with Nones" comment is wrong (#32). `disp(numformat=...)` and `fromeach(2)` do not match their examples either (#11, #21).

**Slicing.** Checked `od[:]`, `od[1:3]`, `od[-1]`, `od[::2]`, `od[1:4:2]`, `od[::-1]`, `od[-2:]`, `od[3:1]` (empty array), `od[2:100]` and `od[100:200]` (clamped, no error), plus string slices `od['b':'d']` (inclusive of the stop key, as documented), `od['d':'b']` (empty), `od[:'c']` and `od['c':]`. All return values in insertion order, as a numpy array via `_sanitize_items`, and out-of-range bounds are clamped by `slice.indices()` rather than raising. Slice *assignment* was checked for `od[1:3] = [20,30]` (element-wise), `od[::2] = 0` (scalar broadcast to each key) and `od[::-1] = [1,2,3,4,5]` (reversed, correctly). Ragged values fall back to a list and string values are converted to an `object` array rather than a fixed-width `'<U'` array, as the comment intends.

**List/array key access.** `od[[0,2]]` and `od[['a','c']]` return lists, `od[np.array([0,2])]` returns an array (matching the documented "if the user supplied the keys as a list, assume they want the output as a list"), mixed `od[[0,'c']]` works, and `od[['x','y']] = [1,2]` creates both keys. `np.int64`/`np.int32`/`bool` indices resolve correctly (`od[True] == od[1]`).

**Order integrity after mutation.** For a 5-key odict, `list(od.keys())`, `list(od.values())` and `od[i] == od[key_i]` were asserted consistent after each of: `pop('c')`, `pop(1)`, `pop(-1)`, `pop([0,2])`, `pop([-1,0])`, `pop(np.array([0,2]))`, `pop(slice(1,3))`, `del od[2]`, `popitem()`, `remove('b')`, `insert(0/2/5, 'X', 99)`, `rename` at the first/middle/last position and by numeric index `0`/`2`/`-1`, `sort()`, `sort('values')`, `sort(reverse=True)`, `sort(sortby=<full key list>)`, `sort(sortby=[4,3,2,1,0])` and `reverse()`. The *ordering* is right in every one of these cases (the defects reported above are the stale key cache, which is separate, plus the partial-`sortby` case and negative rename indices other than `-1`, #29). `insert()` at 0, at a middle position and at `len(self)` all place the item exactly where asked and shift the rest correctly; `insert(len+1)` and `insert(-1)` raise rather than silently misplacing.

**`rename()`.** The tail-rotation algorithm is correct for every non-colliding rename by key name and by non-negative numeric index, and for `-1`, at the first, middle and final positions — the loop's use of `self.keys()[index]` (rather than the loop variable `i`) is deliberate and right, since each iteration shifts the list. Other negative indices are wrong (#29).

**`sort()`/`sorted()`/`reverse()`/`reversed()`.** `sort()` returns `None` and mutates in place, `sorted()` returns a copy and leaves the original alone, and `reverse()`/`reversed()` mirror that, all as documented. `sort('values')` matches `argsort` order; `sort(sortby=<full permutation of keys>)` and `sort(sortby=<full index permutation>)` are exact; `reverse=True` reverses correctly. Python-`bool` masks are correctly tested before numbers, so a boolean mask is not mistaken for indices (numpy boolean masks are rejected, #27; tuple keys are rejected, #28). `sort(sortby=['a','a','b'])` (duplicate keys) raises `KeyError` rather than corrupting, and `sort('values')` on non-orderable values (numpy arrays) raises numpy's ambiguity error rather than producing a wrong order.

**Copy semantics.** `copy()` is a genuine shallow copy (nested lists shared with the original, top-level `pop` on one not visible in the other) and `copy(deep=True)`/`dcp()` is a genuine deep copy, exactly as the `copy()` docstring's example claims. `od2 = sc.odict(od1)` copies the mapping but shares the values, matching `dict(od1)` semantics; mutating `od2['b']` does not affect `od1`, mutating `od2['a'].append(...)` does. `copy()` preserves the subclass (`_new`). Pickle and `sc.dcp()` round-trip the contents and the key order faithfully.

**Attribute/method shadowing.** `odict` defines no `__getattr__`, so keys named `keys`, `pop`, `copy`, `items` or `__dict__` are stored and retrieved normally and do not shadow or get shadowed by the real methods (`od['keys'] == 5` while `od.keys() == ['keys','pop','copy','items']`). `_cached_keys`/`_stale`/`_defaultdict` are set with `object.__setattr__` and stay out of the dict contents.

**Round-trips and iteration helpers.** `enumkeys()`, `enumvals()`, `enumvalues()`, `enumitems()`, `items()`, `iteritems()`, `dict_keys()`, `dict_values()` and `dict_items()` all agree with each other on order and content for non-empty odicts, and `transpose=True` produces correctly transposed tuples (only the empty case fails, reported above). `export()` emits valid re-evaluable Python for nested odicts. `keys()`/`values()`/`items()` return lists, as documented.

**`update`/`+`/`merge`.** `d1 + d2` returns a new odict of the correct class, right operand wins on duplicate keys, and the left operand is *not* mutated. The reflected form `dict + odict` gets precedence backwards (#31). `sum([d1, d2])` works. `update()` correctly flags the key cache stale, so `od[-1]` after `od.update(b=2)` is right.

**Find/filter family.** `index()`, `valind()`, `findkeys()` (all four methods: `re`, `in`, `startswith`, `endswith`, plus tuple keys, which are searched element-wise), `findbyval()` (exact match, `first=True/False`, the documented list-containment relaxation, and `[]` when nothing matches, for scalar- and list-valued odicts), `filter()` by key list and by pattern, `filter(exclude=True)` for both forms, and `filtervals()` all return positions and subsets consistent with `keys()`. `findbyval()`/`filtervals()` crash if any value is a numpy array (#30). `findbykey(pattern, first=False)` returning a bare value rather than a one-element list when exactly one key matches is deliberate (there is an explicit comment) rather than a bug, though it does make the return type input-dependent.

**Miscellaneous.** `setitem()` really does give plain-dict behaviour, so integer keys 0..n created that way are self-consistent for both get and set. `makefrom(force=False)` stringifies integer keys as documented. `append()` generates `key<len>` names and increments correctly; the only collision case is a pre-existing `key<len>` name, which is a narrow pre-existing-name clash rather than a logic error. `__repr__` handles an empty odict, an empty-string value, deep nesting, self-reference (bounded by `maxrecursion`) and `maxitems` truncation. `disp()`'s `sigfigs`, `maxlen`, `divider` and `maxitems` arguments all take effect (only `numformat` does not). `_slicetokeys` correctly adds 1 to a string `stop` and rejects an unknown string bound.

### `objdict` / `dictobj`

**`objdict` attribute/key protocol.** The v3.2.8 changelog claims were both tested and both hold: dunder attributes are not turned into keys (`hasattr(o, '__await__')` is `False` and leaves `o.keys() == ['a']`; same for `_ipython_canary_method_should_not_exist_`), and `o.keys = 3` raises `KeyError` (not `ValueError`) with the "use setattribute() instead" message, as does every other name that exists as an attribute. Keys with such names can still be stored by subscript or constructor and read back with `o['keys']`, which is intended. `__getattribute__` raises the correct exception type for a missing name -- `hasattr(o, 'missing')` is `False`, `getattr(o, 'missing', 'DEF')` returns `'DEF'`, and `o.missing` raises `AttributeError`, not `KeyError` -- because `__getitem__(attr, exception=E)` re-raises the original `AttributeError`; the `_fundamental_attrs` short-circuit at `sc_odict.py:1249` correctly resolves `__class__`/`__reduce__`/`__getstate__`/`__setstate__`/`__copy__`/`__deepcopy__` from the class. Keys that are not valid identifiers round-trip fine by *both* routes (`o['a b'] = 1` then `getattr(o, 'a b') == 1`; likewise `''` and `'1'`), and non-string keys (`(1,2)`, `None`) are set and retrieved correctly by subscript; integer keys are positional indices, which is documented `odict` behaviour, not a defect of this region. `==`/`!=` are correct for `objdict` (equal contents compare equal, different contents unequal, and `objdict(a=1) == {'a': 1}`), unlike `dictobj`.

**`objdict` copy/pickle/defaultdict.** `pickle.dumps`/`loads`, `sc.dcp`, `copy.deepcopy` and `copy.copy` of a nested `sc.objdict(a=[1,2], b=sc.objdict(c=3))` all round-trip with the class (`sciris.sc_odict.objdict`), the key order, and attribute access (`p.b.c == 3`) preserved -- no recursion, no downgrade to plain `dict`. A `defaultdict=list` objdict and a `defaultdict='nested'` objdict both survive pickling and `sc.dcp` with `_defaultdict` intact. Nested autovivification works through both routes (`n.a.b.c = 4` and `n2['x']['y'] = 5` produce the same structure) and the `_nested_parent`/`_nested_attr` bookkeeping is cleaned up afterwards. One behaviour worth knowing but not a bug: on a `defaultdict` objdict, `hasattr(o, 'anything')` is always `True` and, for a non-`'nested'` default, actually creates the key -- that is inherent to defaultdict read semantics. `setattribute()` correctly refuses read-only class attributes (`AttributeError` for `setattribute('keys', 4)`) and `force=True`/`getattribute`/`delattribute` behave as documented; the subclassing pattern in `tests/test_odict.py:243-280` was re-run and passes.

**`dictobj` (other than the equality bug).** `to_json()`, the class-method `fromkeys()`, and `copy()` returning a `dictobj` (v3.1.6 claim) all work; `dict(d)`, `d2.update(d)`, `pd.DataFrame([d1, d2])` and `sc.dataframe(entries)` all work because CPython's `dict_merge` and pandas both fall back to `keys()` when `__iter__` is overridden, so the empty-real-dict problem does not leak into those paths; `pickle` and `sc.dcp` round-trip a `dictobj` with class and contents preserved; `len()` and `bool()` are correct; non-string keys (`3`, `None`) work by subscript. The docstring's claim that JSON conversion "fails silently" is accurate.

### `asobj`

`strict=True` and `strict=False` genuinely differ and in the documented direction (raise vs. set a real attribute), and only for names that already exist as attributes -- a fresh name becomes a key under both settings. `hasattr`/`getattr` behave correctly on the wrapper (`AttributeError` for a missing name, defaults honoured). Reading keys by attribute works for `dict` and `list` inputs, and the derived-class pattern in `tests/test_odict.py:252-265` passes.

### `argparse`

All three documented command-line forms parse correctly for `int`/`str` defaults (`100 data.csv`, `100 output_file=data.csv`, `iterations=100 --output_file=data.csv` all give `{'iterations': 100, 'output_file': 'data.csv'}`); defaults survive an empty `sys.argv`; an unrecognised keyword raises `sc.KeyNotFoundError` listing the valid arguments; a positional after a keyword raises `ValueError`; `sys.argv` is *not* mutated (compared element-wise before and after in a subprocess) and no other global state is touched; `parse=False` suppresses parsing, and `parse=True` with no kwargs is a no-op by design. Too many positional arguments raises a bare `IndexError: list index out of range` from `keys[i]` at `sc_odict.py:1553` rather than a helpful message, which is an out-of-contract input and so not reported as a defect. A default of `None` deliberately leaves the value as a string.

## Suggested order of work

1. **Findings 1 and 2** (stale key cache via `pop`/`remove`/`del` by index, and via `setdefault`/`|=`/`popitem`/`clear`) — these return wrong *values*, not errors, on ordinary use of a headline `odict` feature, and the corrupted cache is pickled with the object, so a saved file can carry the inconsistency forward into a completely separate process. Fix both properly: reorder the stale flag in `pop()` and `__delitem__`, and override the four inherited methods. A length check alone is not enough.
2. **Finding 23** (`append(key, None)` silently swaps key and value; `insert(..., None)` crashes).
3. **Findings 5, 4, 6, 7 and 11** — silently wrong dict contents from documented calls (in-place `sort()` keeping dropped keys, `insert()` losing the value), `dictobj` equality always `True`, `argparse` bool/list conversion, and the ignored `disp(numformat=...)`. Each is a small, well-isolated fix.
4. **Findings 27, 29, 30 and 8** — numpy boolean masks rejected by `sort()`, wrong position from `rename()` with negative indices, `findbyval()` crashing on array values, and `sort(reverse=True)` mutating the caller's list.
5. **Findings 3 and 10** (integer keys; discouraged by the class docstring, and each needs its own fix), **9** (with the subclass-safe fix), **13**, **28**, **31**, **21**, **24** and **32** — lower-impact inconsistencies, documentation mismatches and dead guards. **16** is lowest value.

Findings 1, 2, 4, 5 and 23 in particular are the ones most likely to bite silently in downstream code, since none of them raise an error — they hand back a plausible-looking but wrong value, and (for 1 and 2) that wrong value can be re-serialized and reloaded as if it were correct. They should be prioritized accordingly, ahead of the crash-on-input findings, which at least announce themselves.
