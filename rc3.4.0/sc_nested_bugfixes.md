# `sc_nested.py` bug audit

Audit of `sciris/sc_nested.py` for genuine defects: wrong results, documented arguments that don't work, silent data corruption or loss, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps, and performance. The file (1369 lines) was covered end to end by two parallel auditors working on disjoint halves: one covering the nested-dict functions, `IterObj`, `iterobj()`, `mergenested()`, `flattendict()`, and `nestedloop()` (lines 26-855), the other covering `search()`, `Equal`, and `equal()` (lines 856-1369). **Method**: line-by-line reading of each function, followed by executed hypothesis tests against the editable install (Sciris 3.3.0, commit `2d69aad`); every "actual" value below was produced by running the code, and findings were reproduced independently before being recorded.

**Independent re-verification.** This document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`): every finding's repro was re-run with `SCIRIS_BACKEND=agg`. Of the original 28 findings, 14 were confirmed as written (some with caveats on the fix), 3 were confirmed but rewritten because the description or proposed fix was inaccurate (1, 4, 22; finding 22 was raised from Low to High), and 11 were rejected as not a bug or not worth fixing (listed under "Rejected on review" near the end). The review also found 4 new bugs, added as findings 29-32. The document now lists **21 findings: 7 High, 12 Medium, 2 Low**. Original finding numbers have been kept so that cross-references stay valid; numbers that were rejected are simply absent from the severity sections.

**Nothing in this document has been applied.** All fixes are described, not made.

This file also contains `sc.equal()`, one of the module's three headline functions, where the most consequential failure mode is a false positive — reporting two objects equal when they in fact differ, silently and with no error. Two such false-positive cases survived review (findings 4 and 7); two others originally reported (5 and 6) were rejected because they follow from the documented meaning of `leaf=True`.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `sc.iterobj()` | A second reference to an already-visited container is dropped entirely rather than emitted without descent | 550 |
| 2 | High | `sc.getnested()` | `safe` is inverted for objects: `safe=True` raises, `safe=False` swallows the error | 209-212 |
| 3 | High | `sc.setnested()` | `overwrite=False` fails to protect a list element, since membership is tested by value not index | 129 |
| 4 | High | `sc.equal()` | An undecidable leaf comparison is silently counted as "equal" | 1262 |
| 7 | High | `sc.equal()` | Pandas objects that differ only in their index are reported equal | 1145 |
| 8 | Medium | `sc.setnested()` | An integer key into an `odict` destroys the existing entry instead of descending into it | 118 |
| 10 | Medium | `sc.iterobj()` | The documented `*args` pass-through to `func` is unreachable, and a 3rd positional argument silently sets `inplace` | 647 |
| 11 | Medium | `sc.iterobj()` | `flatten=True` combined with `to_df=True` reports the string length as the tree depth | 637 |
| 12 | Medium | `sc.iterobj()` | `leaf=True` combined with `to_df=True` silently drops one real leaf row | 640 |
| 15 | Medium | `sc.mergenested()` | Raises on array leaves even when the two values are identical | 747 |
| 16 | Medium | `sc.iterobj()` | `atomic='default-tuple'` with `inplace=True` always raises, including for the natural tuple-to-list conversion | 537 |
| 17 | Medium | `sc.equal()` | `union` is documented and in the signature but never forwarded to `Equal` | 1365 |
| 18 | Medium | `sc.equal()` | `die` is stored on `Equal` but never used, and `convert()` errors escape the `try` | 1005 |
| 19 | Medium | `sc.search()` | Raises `ValueError` if the object contains any numpy array, Series, or dataframe | 919 |
| 21 | Low | `sc.makenested()` | Raises `ValueError(keylist)` instead of the composed error message | 115 |
| 22 | High | `sc.iterobj()` | `leaf=True` still applies `func` to the root container, so any leaf-only function crashes | 591, 612 |
| 23 | Low | `sc.getnested()` | `safe=True` does not suppress a non-integer key into a list | 203 |
| 29 | Medium | `sc.iterobj()` | A custom `rootkey` combined with `leaf=True` raises `KeyError: 'root'` (also via `sc.equal()`/`sc.search()`) | 616 |
| 30 | Medium | `sc.equal()` | `leaf=True` crashes when one object has no leaves and the other does | 1199 |
| 31 | High | `sc.iterobj()` | `inplace=True` descends into the old object rather than the value `func` returned, so nested changes are lost | 603-604 |
| 32 | Medium | `sc.iterobj()` / `sc.setnested()` | `__slots__` objects crash in `inplace=True` and in `setnested()` | 533, 228, 174 |

## Recurring patterns

**Identity-based bookkeeping drops shared children rather than just not re-descending into them.** `iterobj()`'s memo (finding 1) records every `id()` it has seen across the whole traversal. Not re-parsing a shared object is documented behavior (the `recursion` argument), and it keeps traversal cheap on heavily shared object graphs. The defect is that the second reference is not emitted at all, so it is missing from the output and, with `inplace=True`, left untransformed. The fix is to emit the node on a memo hit but not queue its children.

**"Could not be compared" is silently reduced to "equal" at both result-reduction sites.** `Equal.compare()`'s verdict line drops every `None` before calling `all()` (finding 4), and `Equal.to_df()`'s `.all(axis=1)` skips `NA` the same way. For leaf or atomic comparisons that never succeeded, both treat an undecidable comparison as a vote for "equal" rather than "not proven equal," and neither consults `self.die` (finding 18), which exists precisely to let the caller choose. (A `None` on a container whose children were all compared successfully is harmless and must stay that way.)

**The root node is special-cased inconsistently.** `sc.iterobj()` gives the top-level node the *string* key `rootkey` (default `'root'`) while every other trace is a tuple, and several sites hard-code the string `'root'` instead of `self.rootkey`. This produces: a root container passed to `func` under `leaf=True` (finding 22); a positional "drop the first row" in `to_df()` that drops a real leaf once the root is already gone (finding 12); a `KeyError` when `rootkey` is customized with `leaf=True` (finding 29); and a crash in `sc.equal()` when only one tree keeps its root row (finding 30). Deciding root inclusion once, from `leaf` and the root's own iter type, and referring to it only through `self.rootkey`, would close all four.

**Two disagreeing implementations of "is this key present" (and of "set this key").** `check_in_obj()` correctly understands dicts, list indices, and object attributes, but the final-key overwrite check in `makenested()`/`setnested()` uses a bare `in` test instead (finding 3), and `check_in_obj()` itself disagrees with `get_from_obj()`/`set_in_obj()` over how `odict` integer indexing works (finding 8). `check_in_obj()`, `set_in_obj()`, and `IterObj.setitem()` also go through `__dict__`, which `__slots__` objects don't have (finding 32). One containment helper and one setter, used everywhere and based on `hasattr`/`setattr` for objects, would close all three.

**Falsy/absent sentinels and truthiness.** `safe`, `overwrite`, and similar flags are tested or forwarded inconsistently across the dict/list/object branches of the same function (findings 2, 3, 23), producing behavior that depends on which container type is involved rather than on the documented contract.

**Inplace mode writes one object but keeps walking another.** In `iterobj(inplace=True)`, `process_obj()` writes `func(subobj)` into the parent, but traversal then continues into the old `subobj` (finding 31). Any change to descendants of a replaced container is applied to the discarded object; finding 16's tuple crash comes from the same cause.

**`*args` declared after keyword parameters.** Both `iterobj()` and `IterObj.__init__()` put `*args, **kwargs` after a long run of defaulted parameters (finding 10), which makes the documented positional pass-through to `func` unreachable and turns an attempted positional call into a silent `inplace=True`.

## High severity

### 1. `sc.iterobj()` drops a second reference to an already-visited container entirely, rather than emitting it without descending — `sc_nested.py:550`

`check_proceed()` memoizes by `id()` and skips any iterable it has seen more than `recursion` times (default 0). Not *re-parsing* a shared object is documented: `recursion` is described as "number of recursive steps to allow, i.e. parsing the same objects multiple times (default 0)", and this keeps traversal cheap on large objects with heavy sharing (e.g. Starsim sims where many modules point to the same `people`/`sim`). The defect is narrower: the second reference is not emitted as a node at all. In the default (collating) mode the second node is simply missing from the output; with `inplace=True` the function is applied to the first alias and not to the second, so the object is left half-transformed.

```python
shared = [1,2,3]
data = dict(a=shared, b=shared)
print(list(sc.iterobj(data).keys()))
data = dict(a=shared, b=shared)
sc.iterobj(data, lambda o: 'LIST' if isinstance(o, list) else o, inplace=True)
print(data)
```

```
['root', ('a',), ('a', 0), ('a', 1), ('a', 2)]        # actual; ('b',) itself is missing
{'a': 'LIST', 'b': [1, 2, 3]}                          # actual; expected {'a': 'LIST', 'b': 'LIST'}
```

The same drop propagates to the functions built on `iterobj()`: `sc.search(dict(a=dict(x=t), b=dict(y=t)), value=0)` finds only `('a','x',0)`, and `sc.equal(dict(a=s,b=s), dict(a=[1,2,3], b=[1,2,3]))` returns `False` (a false negative). Two structurally equal but distinct dicts are both visited (`dict(a={'x':1}, b={'x':1})` gives 5 nodes), so the failure only appears once the *same* object is referenced twice. `recursion=1` restores the missing nodes but also re-parses genuine cycles a second time, so it is not a safe global workaround.

**Fix**: when a memo hit occurs on an iterable, still call `process_obj()` for it (emit the node in collate mode, and apply `func`/`setitem` in inplace mode), but don't queue its children. This fixes the inplace half-transformation and the missing `('b',)` key without changing traversal cost or the meaning of `recursion`. It does not make `sc.equal()` alias-insensitive (the `('b', 0..2)` children are still absent from one tree); that would need a separate design decision.

(The original audit proposed tracking only *ancestor* ids, so that only true cycles are skipped. That was rejected on review: it changes the documented meaning of `recursion` and re-traverses every shared subtree once per reference, which can make traversal dramatically slower on heavily shared object graphs.)

### 2. `sc.getnested()` has `safe` inverted for objects: `safe=True` raises, `safe=False` swallows the error — `sc_nested.py:209-212`

The object branch of `get_from_obj()` is

```python
elif itertype == 'object':
    if safe:
        out = getattr(ndict, key)          # raises AttributeError
    else:
        out = getattr(ndict, key, default) # swallows the error
```

i.e. the two arms are swapped relative to the dict and list branches. Consequences: a strict lookup (`safe=False`, the default) on a missing attribute returns `None` instead of raising, and a forgiving lookup (`safe=True`, or any `default=`) raises `AttributeError`.

```python
o = sc.prettyobj(a=1)
sc.getnested(o, ['missing'])                   # actual: None            expected: AttributeError
sc.getnested(o, ['missing'], safe=True)        # actual: AttributeError  expected: None
sc.getnested(o, ['missing'], default='dflt')   # actual: AttributeError  expected: 'dflt'
```

Both halves are harmful: the silent `None` propagates a typo'd attribute name into downstream arithmetic, and `default=` — documented as "the value to return if the key is not found" — is unusable on objects. `sc.odict.getnested()` (`sc_odict.py:1182`) and `makenested()`'s parent-walk (`sc_nested.py:125`) both go through `get_from_obj()`; `tests/test_nested.py:129-131` only exercises the dict path, so the inversion is untested.

**Fix**: swap the two branches (`getattr(ndict, key, default)` when `safe`, plain `getattr(ndict, key)` otherwise).

### 3. `sc.setnested()`'s `overwrite=False` fails to protect a list element, because membership is tested by value instead of by index — `sc_nested.py:129`

The parent levels are checked with `check_in_obj()` (which correctly treats an integer key as a list *index*), but the final key is checked with a bare `if overwrite or lastkey not in currentlevel`. For a list, `lastkey not in currentlevel` asks whether the integer is one of the list's *values*, so `overwrite=False` guards the wrong thing in both directions, and for an object it is not a valid test at all.

```python
d = {'L': [10,20,30]}
sc.setnested(d, ['L',0], 99, overwrite=False)   # index 0 exists, so this must refuse
d = {'L': [3,4]}
sc.setnested(d, ['L',3], 99, overwrite=False)   # index 3 does not exist, so this must not refuse
sc.setnested(sc.prettyobj(a=1), ['a'], 2, overwrite=False)
```

```
{'L': [99, 20, 30]}                                              # actual: silently overwrote; expected ValueError
ValueError Not overwriting entry ['L', 3] since overwrite=False   # actual: refused a non-existent entry
TypeError argument of type 'prettyobj' is not iterable            # actual on any object
```

The first case is the damaging one: the existing element is destroyed by the very call that asked for it to be preserved, and whether it happens depends on whether the index happens to coincide with one of the stored values. The object case makes `overwrite=False` unusable with the "operate on arbitrary objects" feature added in 3.2.0 (`sc_nested.py:104`), even though `tests/test_nested.py:93` tests `overwrite=False` on a dict.

**Fix**: use `check_in_obj(currentlevel, lastkey)` for the final key, exactly as the parent loop at `sc_nested.py:118` already does. For this to also work on `__slots__` objects, `check_in_obj()` needs the `hasattr` change from finding 32.

### 4. An undecidable leaf comparison is silently counted as "equal" — `sc_nested.py:1262`

When a comparison raises or cannot be reduced to a bool, `Equal.compare()` stores `eq = None` in `self.exceptions` and `self.results`, and then the verdict line drops every `None`: `self.eq = all([v for v in self.results.values() if v is not None])`. A leaf or atomic key whose comparison never succeeded therefore *votes yes*, and if every comparison fails, `all([])` returns `True`. Nothing is printed unless `verbose` is set (`check_exceptions()` is only called under `if self.verbose`), so the failure is invisible. `to_df()` has the same defect at `sc_nested.py:1290`: `df.iloc[:, :(self.n-1)].all(axis=1)` skips NA, so a row whose only comparison result is `None` is labelled `equal=True`.

```python
class Arr:
    def __init__(self, v): self.v = np.array(v)
    def __eq__(self, other): return self.v == other.v   # returns an array -> bool() raises

sc.equal(Arr([1,2,3]), Arr([9,9,9]), method='eq', atomic=[Arr])            # actual: True    expected: False (or an error)
sc.equal(Arr([1,2,3]), Arr([9,9,9]), method='eq', atomic=[Arr], die=True)  # actual: True    expected: raise
sc.equal(dict(a=1), dict(a=2), method=lambda o: {'v': np.array([1,2])})    # actual: True    expected: False
```

```
>>> e = sc.Equal(Arr([1,2,3]), Arr([9,9,9]), method='eq', atomic=[Arr], detailed=True)
>>> dict(e.results), e.eq, list(e.exceptions.keys())
({'root': None}, True, ['root'])
>>> e.df
      equal obj0==obj1
root   True       None
```

With the default method chain `['eq','pickle']` the `Arr` case correctly gives `False` (the pickle fallback sees the difference), so the problem appears when the user restricts `method` or supplies a custom one. Both triggers use documented arguments: `atomic` is forwarded to `sc.iterobj()` via `kwargs`, and a custom callable `method` is documented ("any custom function can be provided"). Returning an array from `__eq__` is the normal idiom for array-like wrappers, which is exactly the class of object a user would mark atomic.

Blast radius: `sc.equal()`/`sc.Equal` have no internal callers inside Sciris, so the exposure is entirely to downstream users; `sc.equal()` is one of the three functions advertised in the module's "Highlights".

**Fix**: treat `None` as "not equal" (or honor `die`, finding 18) only for traces that have no compared descendants, i.e. leaves or atomic nodes, plus the case where every result is `None`. For example, after the loop, for each key whose result is `None`, check whether any other result key has it as a prefix (the currently unused `Equal.is_subkey()` helper does exactly this, which suggests it was written for this purpose); if none does, count that key as `False`. `to_df()` must apply the same rule when reducing, or the dataframe will again disagree with `eq`.

(The original audit proposed `self.eq = False if None in self.results.values() ...`. That is wrong: a root result of `None` is routine and harmless whenever the root's `==` can't be reduced to a bool but its children were compared successfully. For example, `sc.Equal(dict(a=np.array([1,2])), dict(a=np.array([1,2])), method='eq')` currently gives `results={'root': None, ('a',): True}` and `eq=True`, which is correct; under the blanket rule it would become `False`.)

### 7. `sc.equal()` reports pandas objects that differ only in their index as equal — `sc_nested.py:1145`, `sc_nested.py:1149`

`Equal.compare_special()` routes `pd.Series`/`pd.Index` to `sc.nanequal(obj, obj2, scalar=True, ...)` and `pd.DataFrame` to `sc.dataframe.equal(...)`. Both compare *values* only — `sc.dataframe.equal`'s own docstring says "same type, size, columns, and values," with no mention of the index — and because `pd.DataFrame`/`pd.Series` are in `atomic_classes`, `sc.iterobj()` does not descend into them, so the index is never compared anywhere. Two time series with completely different date indices compare equal.

```python
d1 = pd.DataFrame({'cases':[10,20,30]}, index=pd.date_range('2022-01-01', periods=3))
d2 = pd.DataFrame({'cases':[10,20,30]}, index=pd.date_range('2023-06-01', periods=3))
sc.equal(d1, d2)                                # actual: True    expected: False
d1.equals(d2)                                   # False  (pandas' own verdict)
sc.equal(dict(x=d1), dict(x=d2))                # False  (only because the pickle fallback sees the difference)
sc.equal(dict(x=d1), dict(x=d2), method='eq')   # actual: True    expected: False

s1 = pd.Series([1,2,3], index=['x','y','z'], name='foo')
s2 = pd.Series([1,2,3], index=['p','q','r'], name='bar')
sc.equal(s1, s2)                                # actual: True    expected: False
sc.equal(dict(x=s1), dict(x=s2), method='eq')   # actual: True    expected: False
```

The internal inconsistency is the clearest signal that this is a defect rather than a design choice: the *same pair of dataframes* gets `False` when nested one level under the default method (the dict `==` raises, the `pickle` fallback then sees different bytes) and `True` at the top level or with `method='eq'`, because `compare_special()` sets `compared = True` and so short-circuits the rest of the method chain. Series `name` is dropped as well.

Blast radius: `sc.dataframe.equal()` / `df.equals()` (`sc_dataframe.py:378`, `sc_dataframe.py:435`) share the value-only semantics, and `sc.dataframe.__eq__` (`sc_dataframe.py:375`) delegates to `self.equals(other)`, so any downstream `df1 == df2` on a `sc.dataframe` is index-blind too; `sc.nanequal()` is `sc_math.py:458`.

**Fix**: in `compare_special()`, compare the index (and, for Series, `name`) in addition to the values — e.g. `eq = eq and bool(np.array_equal(obj.index.values, obj2.index.values))` for the Series/DataFrame branches — or use `obj.equals(obj2)` when `equal_nan=True` (pandas' `equals` already treats NaN as equal) and fall back to the current value comparison only when `equal_nan=False`. Alternatively make `compare_special()` not set `compared = True` on a `True` result, so the remaining methods still get a chance to disagree.

### 22. `leaf=True` still applies `func` to the root container, so any leaf-only function crashes — `sc_nested.py:591`, `sc_nested.py:612`

*(Raised from Low to High on review; the original description understated the impact.)*

`iterate()` always runs `func(self.obj)` for the root: at line 591 in collate mode (the result is popped afterwards at line 616 if there is more than one node), and at line 612 in inplace mode (where the result is returned). This happens regardless of `leaf=True`. So any function meant only for leaves — the main use case of `leaf=True` — is also called on the root container and crashes:

```python
sc.iterobj({'a':1,'b':2}, lambda o: o*2, leaf=True)
sc.iterobj({'a':1,'b':2}, lambda o: o+1, leaf=True, inplace=True)
sc.iterobj({'a':1.23,'b':[2.345]}, lambda o: round(o,1), leaf=True)
```

```
TypeError: unsupported operand type(s) for *: 'dict' and 'int'     # actual; expected {('a',): 2, ('b',): 4}
TypeError: unsupported operand type(s) for +: 'dict' and 'int'     # actual; expected {'a': 2, 'b': 3}
TypeError: type dict doesn't define __round__ method               # actual; expected {('a',): 1.2, ('b', 0): 2.3}
```

When `func` does tolerate the root, the root result still leaks out: `sc.iterobj({'a': {}}, leaf=True)` returns `{'root': {'a': {}}}` (a non-leaf), and `sc.iterobj({'a': {'x': 1}}, lambda o: 'X', leaf=True, inplace=True)` returns `'X'`, i.e. `func` applied to the root.

**Fix**: in `iterate()`, compute the root's iter type first. When `leaf=True` and the root is a container, skip `func` on the root in both modes (in inplace mode, return `self.obj`), and emit the root row only when the root itself is a leaf. This also removes the need for the hard-coded `pop('root')` at line 616 (see finding 29) and the root asymmetry behind finding 30.

### 31. `iterobj(inplace=True)` descends into the old object rather than the value `func` returned, so nested changes are lost — `sc_nested.py:603-604`

*(New on review.)*

`process_obj()` writes `newobj = func(subobj)` into the parent, but `iterate()` then queues `self.iteritems(subobj, newtrace)`: the children of the *old* object, with the old object as their parent. If `func` returns a new container, all further changes to its descendants are applied to the discarded old object and lost. Some deeper nodes can still end up changed through shared references, which makes the result inconsistent rather than simply unchanged.

```python
d = {'a': {'b': 1}}
sc.iterobj(d, lambda o: dict(o) if isinstance(o, dict) else o*10, inplace=True)
print(d)

data = {'top': sc.objdict(a=sc.objdict(b=sc.objdict(c=1)))}
sc.iterobj(data, lambda o: dict(o) if isinstance(o, sc.objdict) else o, inplace=True)
```

```
{'a': {'b': 1}}                  # actual; expected {'a': {'b': 10}}
dict / objdict / dict            # actual types at top / a / a.b; expected dict / dict / dict
```

Converting container types recursively in place (objdict to dict, tuple to list, odict to dict before JSON export) is a natural use of `inplace=True`, and the failure is silent. Finding 16's tuple-to-list crash comes from the same cause.

**Fix**: have `process_obj()` return `newobj` (along with the trace), and in inplace mode queue `self.iteritems(newobj, newtrace)` instead of the old `subobj`; the memo check should then use `id(newobj)`. In collate mode, keep descending into `subobj`.

## Medium severity

### 8. `sc.setnested()` with an integer key into an `odict` destroys the existing entry instead of descending into it — `sc_nested.py:118`

`check_in_obj()` treats an `odict` as a plain dict and asks `key in parent.keys()`, which is `False` for an integer, so `makenested()` concludes the level is absent and creates a fresh one over the top of it. But `get_from_obj()`/`set_in_obj()` do honour `odict`'s positional indexing, so the very same integer key resolves fine on the way back out — the write silently discards whatever was there.

```python
od = sc.odict(a={'b':1, 'c':2})
sc.setnested(od, [0,'b'], 99)
print(dict(od))
print(sc.getnested(od, [0,'b']))
```

```
{'a': #0: 'b': 99}     # actual: 'c' has been deleted and the plain dict replaced by an odict
99                     # the round trip "succeeds", hiding the loss
```

With the equivalent string key nothing is lost: `sc.setnested(sc.odict(a={'b':1,'c':2}), ['a','b'], 99)` gives `{'b': 99, 'c': 2}`. `odict`'s interchangeable integer/string access is a headline feature (and `sc.odict.setnested()` is exported at `sc_odict.py:1186`), so the two access forms giving different results — one of them destructive — is a real trap. Related: `sc.setnested(sc.odict(), [0], 99)` raises `IndexError: index 0 out of range for dict of length 0` rather than creating the key, so integer keys cannot be used to extend an `odict` either.

**Fix**: have `check_in_obj()` delegate to the container's own containment logic for `odict`-like classes (e.g. try `get_from_obj(parent, key)` and treat a successful lookup as "present"), rather than only `key in parent.keys()`.

### 10. The documented `*args` pass-through to `func` is unreachable, and a third positional argument silently turns on `inplace` — `sc_nested.py:647`, `sc_nested.py:714`

`iterobj()` declares `*args` after thirteen keyword parameters, so a caller can only reach it by supplying all thirteen positionally; and even then `iterobj()` forwards them as `IterObj(obj=obj, func=func, ..., *args, **kwargs)`, where the positional args collide with `obj`:

```python
def f(obj, extra): return (obj, extra)
sc.iterobj({'a':1}, f, False, False, False, 0, True, 'default', None, 'root', False, False, False, 5)
```

```
TypeError: IterObj.__init__() got multiple values for argument 'obj'
```

So `*args (list): passed to func()` never works by any route (`**kwargs` does work: `sc.iterobj({'a':1}, f, extra=5)` is fine). Worse, the natural attempt binds to a real parameter instead: the third positional argument is `inplace`, so a user following the docstring silently gets in-place modification of their data.

```python
d = {'a': 1}
out = sc.iterobj(d, lambda o: 'X' if not isinstance(o, dict) else o, 5)   # intended: pass 5 to func
print(out, d)
```

```
{'a': 'X'} {'a': 'X'}     # actual: inplace=5 was set, so the caller's dict was modified in place
```

`IterObj.__init__` has the same shape (`*args, **kwargs` after all the named parameters), so `self.func_args` can never be populated there either.

**Fix**: drop `*args` from both signatures (keeping `**kwargs`), or accept the function's positional arguments explicitly as a keyword such as `func_args=None` and forward them by keyword.

### 11. `flatten=True` combined with `to_df=True` reports the string length as the tree depth — `sc_nested.py:637`

`iterobj()` applies `flatten_traces()` first, replacing the tuple keys with joined strings, and then calls `to_df()`, whose depth calculation is `len(tr)` — the length of the tuple, which is now the number of characters in the flattened key.

```python
print(sc.iterobj(dict(a=dict(x=[1,2])), flatten=True, to_df=True))
```

```
   trace  depth          value
0      a      1  {'x': [1, 2]}
1    a_x      3         [1, 2]      # expected depth 2
2  a_x_0      5              1      # expected depth 3
3  a_x_1      5              2      # expected depth 3
```

Without `flatten` the depths are correct (1, 2, 3, 3). Both keywords are documented as independent options of the same call, and the resulting `depth` column is plausible-looking small integers, so a caller grouping or filtering by depth gets silently wrong groups.

**Fix**: compute `depth` from the tuple traces before flattening (e.g. store the depth in `to_df()` from `self.output`'s original keys, or have `flatten_traces()` record the pre-flatten lengths).

### 12. `leaf=True` combined with `to_df=True` silently drops one real leaf row — `sc_nested.py:640`

`to_df(skip_root=True)` unconditionally discards the first row on the assumption that it is the root entry, but `iterate()` has already removed the root entry when `leaf=True` (`sc_nested.py:616`). The first genuine leaf is therefore dropped.

```python
print(list(sc.iterobj(dict(a=1,b=2,c=3), leaf=True).keys()))
print(sc.iterobj(dict(a=1,b=2,c=3), leaf=True, to_df=True))
```

```
[('a',), ('b',), ('c',)]
  trace  depth  value
0  (b,)      1      2         # ('a',) -> 1 is missing entirely
1  (c,)      1      3
```

Silent loss of the first data row, with no error and nothing in the output to indicate it. `leaf` and `to_df` are both documented top-level keywords of `iterobj()`.

**Fix**: in `to_df()`, drop the root row by key (`if skip_root and trace == self.rootkey`) rather than by position.

### 15. `sc.mergenested()` raises on array leaves, even when the two values are identical — `sc_nested.py:747`

The conflict test is `elif a[key] == b[key]`, whose truth value is ambiguous for a numpy array (and for a pandas Series/DataFrame). Nested dicts of arrays are ordinary input for this library, and the failure occurs even in the no-conflict case where the function should simply do nothing.

```python
sc.mergenested({'a': np.array([1,2])}, {'a': np.array([1,2])})
```

```
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

The same call with scalars works, and array leaves that appear in only one of the two dicts are fine — it is specifically a key present in both that fails. `tests/test_nested.py:166` merges scalar-valued dicts only.

**Fix**: replace the equality test with `sc.equal(a[key], b[key])` (already available in this module) or a `try/except`-guarded `bool(...)` that falls through to the conflict branch when the comparison is not scalar.

### 16. `atomic='default-tuple'` with `inplace=True` always raises, including for the natural tuple-to-list conversion — `sc_nested.py:537`

`atomic='default-tuple'` exists specifically so that tuples are descended into, but in `inplace` mode `process_obj()` calls `setitem()` for every visited node, and `setitem()` raises for a tuple parent. So the two documented options cannot be combined, even for a no-op function:

```python
sc.iterobj(dict(t=(1,2)), lambda o: o, atomic='default-tuple', inplace=True)
```

```
TypeError: Trying to set key=0 to 1 in a tuple; not possible since tuples are immutable
```

The natural use case — converting tuples to lists in place — fails too, and leaves the object half-modified:

```python
d = {'t': (1,2)}
sc.iterobj(d, lambda o: list(o) if isinstance(o, tuple) else o, atomic='default-tuple', inplace=True)
print(d)
```

```
TypeError: Trying to set key=0 to 1 in a tuple; not possible since tuples are immutable
{'t': [1, 2]}     # d after the exception: the tuple was replaced, then traversal crashed on the old tuple's children
```

The cause of the second case is finding 31: the children are traversed with the old tuple as their parent, even though `func` already replaced it with a list. The same call without `inplace` works and reports `['root', ('t',), ('t', 0), ('t', 1)]`. Because `atomic` is a global setting for the whole traversal, a single tuple anywhere in a large object aborts an otherwise valid in-place pass.

**Fix**: apply finding 31's fix (descend into `newobj`, so a tuple converted to a list is walked as a list), plus skip the write in `process_obj()` when `newobj is subobj` (nothing to do). Rebuilding the tuple in the grandparent, as originally suggested, is not necessary.

### 17. `union` is in `sc.equal()`'s signature and documented, but never forwarded to `Equal` — `sc_nested.py:1365`

The call is `e = Equal(obj, obj2, *args, method=method, detailed=detailed, equal_nan=equal_nan, leaf=leaf, verbose=verbose, die=die, **kwargs)` — `union=union` is missing. Because `union` is a named parameter of `equal()` it is not swept up by `**kwargs` either, so the value is simply discarded and `Equal`'s own default (`union=True`) always applies.

```python
sc.equal(dict(x=1), dict(x=1, y=2), leaf=True, union=False)          # actual: False (union ignored)
sc.Equal(dict(x=1), dict(x=1, y=2), leaf=True, union=False).eq       # True  -- what union=False actually does
import inspect; 'union=union' in inspect.getsource(sc.equal)          # False
```

`tests/test_nested.py:351` runs `ut.check_signatures(sc.equal, sc.Equal.__init__, ...)`, which checks that the two signatures agree but not that the arguments are passed through, so this slipped past the test suite. The documented behaviour ("if True, construct the comparison tree as the union of the trees of each object ... otherwise rows correspond to the attributes of the first object") is unobtainable from `sc.equal()`.

**Fix**: add `union=union` to the `Equal(...)` call. Once this is forwarded, `leaf=True, union=False` becomes reachable from `sc.equal()`; as shown above it ignores keys that only the second object has, which follows from the documented meaning of `union=False` (see rejected finding 6), so a docstring note on that combination is worthwhile.

### 18. `die` is stored on `Equal` but never used, and `convert()` errors escape the `try` — `sc_nested.py:1005`

`self.die = die` is the only occurrence of `die` in the whole `Equal` class (`grep -n "die" sc_nested.py` finds it at 1005 and then nothing until `mergenested`, which is a different function). The documented contract is "whether to raise an exception if an error is encountered (else return False)"; neither half holds — errors are neither raised under `die=True` nor turned into `False` under `die=False`, they are converted into `None` and then dropped from the verdict (finding 4).

```python
class Arr:
    def __init__(self, v): self.v = np.array(v)
    def __eq__(self, other): return self.v == other.v
sc.equal(Arr([1,2,3]), Arr([9,9,9]), method='eq', atomic=[Arr], die=True)  # actual: True   expected: raise
```

Separately, exceptions raised inside `convert()` are *not* covered by the `try` at all — `bconv = self.convert(baseobj, method)` sits outside it — so an object that cannot be pickled propagates the error out of `sc.equal()` regardless of `die`:

```python
f = lambda x: x
sc.equal(dict(f=f), dict(f=f), method='pickle', die=False)
# actual: PicklingError: Can't pickle <function <lambda> ...>    expected: fall through to the next method, or return False
```

**Fix**: move the two `self.convert()` calls inside the `try`, so a conversion failure falls through to the next method in the chain instead of escaping. Honour `die` only after *all* methods have failed for a given key: raise the stored exception when `self.die` is true, and count the key as `False` (per the rule in finding 4) when it is false. Caveat: `die=True` must not raise on the first exception, because in the default chain `['eq','pickle']` exceptions from `'eq'` are routine and expected (`dict == dict` raises whenever arrays are nested inside); raising early would break the default method chain.

### 19. `sc.search(..., value=...)` raises `ValueError` if the object contains any numpy array, Series, or dataframe — `sc_nested.py:919`

For `method='exact'`, `check_match()` returns `target == source` unreduced, and the caller uses it directly in a boolean context (`if check_match(v, value):` at `sc_nested.py:965`). When the traversed value is an array-like, `==` is elementwise and the `if` raises. Since `sc.search()` is advertised as the way to "find a value in a nested object" and nested scientific objects almost always contain arrays, an ordinary value search blows up.

```python
o = dict(a=np.array([1,2,3]), b=dict(c=3))
sc.search(o, value=3)      # ValueError: The truth value of an array with more than one element is ambiguous...
sc.search(o, 3)            # same (query sets both key and value)
sc.search(dict(a=sc.dataframe(x=[1,2])), value=3)
                           # ValueError: The truth value of a dataframe is ambiguous...
```

It fails identically whether or not a match exists, and it is not confined to values: `sc.search(o, 'c')` fails too, because `query` sets `value` as well as `key`. Key-only searches (`sc.search(o, key='c')`) and `type=` searches are unaffected. `tests/test_nested.py:177` only searches plain nested dicts of scalars and strings, so this is untested.

**Fix**: wrap the reduction in `check_match` in a `try/except`: `try: return bool(match) except Exception: return False`. Prefer this over reducing with `bool(np.all(match))`, which would make `value=3` match `np.array([3,3,3])` — arguably wrong.

### 29. A custom `rootkey` combined with `leaf=True` raises `KeyError: 'root'` — `sc_nested.py:616` (and `1198-1199`, `1215`)

*(New on review.)*

`iterate()` removes the root row with the hard-coded `self.output.pop('root')` instead of `self.output.pop(self.rootkey)`. `rootkey` is a documented argument of `iterobj()`, and it also reaches `sc.equal()` and `sc.search()` through their `**kwargs`.

```python
sc.iterobj({'a':1}, leaf=True, rootkey='top')
sc.equal({'a':1}, {'a':1}, leaf=True, rootkey='top')
sc.search({'a':1}, value=1, leaf=True, rootkey='top')
```

```
KeyError: 'root'     # actual, all three; expected {('a',): 1}, True, and {('a',): 1} respectively
```

`Equal.compare()` also hard-codes `'root'` (lines 1198, 1199, and the `key != 'root'` test at 1215), so with a custom `rootkey` (and `leaf=False`) the structure check at the root silently never runs.

**Fix**: use `self.output.pop(self.rootkey)` at line 616. In `Equal`, store the root key once (e.g. `rk = self.kwargs.get('rootkey', 'root')`) and use it at lines 1198, 1199, and 1215. Finding 22's fix removes the `pop` entirely.

### 30. `sc.equal(..., leaf=True)` crashes when one object has no leaves and the other does — `sc_nested.py:1199`

*(New on review.)*

When an object has no leaves (an empty container, a dict of empty dicts, or a bare scalar), `iterobj(leaf=True)` keeps its `'root'` row, because it only pops the root when `len(output) > 1`. The other object's tree has no `'root'`. With `union=True`, `'root'` ends up in `treekeys`, and the root branch then does `otree['root']` without checking that the key exists.

```python
sc.equal([], [1], leaf=True)
sc.equal(dict(a={}), dict(a={'x':1}), leaf=True)
sc.equal(1, {'a':1}, leaf=True)
sc.equal([1], [], leaf=True)
```

```
KeyNotFoundError: odict key "root" not found     # actual; expected False
KeyNotFoundError                                 # actual; expected False
KeyNotFoundError                                 # actual; expected False
False                                            # the reverse order works
```

**Fix**: in the `key == 'root'` branch, guard with `if key in otree:` and fall through to the existing "key not present gives `False`" handling. Finding 22's fix (never emitting a container root under `leaf=True`) also removes the asymmetry.

### 32. `__slots__` objects crash in `iterobj(inplace=True)` and in `setnested()` — `sc_nested.py:533`, `sc_nested.py:228`, `sc_nested.py:174`

*(New on review.)*

Version 3.2.4 added slots support for traversal (`iteritems` falls back to `__slots__`), but `IterObj.setitem()` (line 533), `set_in_obj()` (line 228), and `check_in_obj()` (line 174) all go through `parent.__dict__`, which slots objects don't have. `getnested` works because it uses `getattr`.

```python
class S:
    __slots__ = ['x', 'y']
    def __init__(self): self.x = 1; self.y = [1,2]
sc.iterobj(S(), lambda o: o*10 if isinstance(o,int) else o, inplace=True)
sc.setnested(S(), ['x'], 5)
sc.setnested(S(), ['y', 0], 5)
sc.getnested(S(), ['y', 0])
```

```
AttributeError: 'S' object has no attribute '__dict__'     # actual; expected x=10 and y=[10, 20]
AttributeError                                             # actual; expected x=5
AttributeError                                             # actual (from check_in_obj on the parent level); expected y=[5, 2]
1                                                          # works
```

**Fix**: use `setattr(parent, key, value)` in `setitem()` and `set_in_obj()`, and `hasattr(parent, key)` in `check_in_obj()`. `setattr` also respects properties and custom `__setattr__`, which writing directly to `__dict__` bypasses. The same `check_in_obj()` change is needed for finding 3's fix to work on slots objects.

## Low severity

### 21. `sc.makenested()` raises `ValueError(keylist)` instead of the error message it just built — `sc_nested.py:115`

```python
sc.makenested({}, [])
```

```
ValueError: []                                              # actual
ValueError: At least one key must be supplied, not []        # expected (the composed errormsg is discarded)
```

`tests/test_nested.py:109` asserts only that *some* exception is raised, so the message has never been checked.

**Fix**: `raise ValueError(errormsg)`.

### 23. `safe=True` does not suppress a non-integer key into a list — `sc_nested.py:203`

The list branch catches only `IndexError`, but indexing a list with a string raises `TypeError`, which escapes even with `safe=True`/`default=` set. An out-of-range integer is handled correctly (`sc.getnested({'a':[1]}, ['a',5], safe=True)` returns `None`), so the two "key not found" cases behave differently.

```python
sc.getnested({'a':[1]}, ['a','b'], safe=True, default='dd')
```

```
TypeError: list indices must be integers or slices, not str      # actual; expected 'dd'
```

This matters for the intended use of `safe`: probing a heterogeneous nested structure where you do not know whether a given level is a dict or a list.

**Fix**: catch `(IndexError, KeyError, TypeError)` in the list/tuple branch.

## Rejected on review

These original findings were rejected during the 2026-09-25 re-verification and are no longer counted. They are listed so their numbers are not reused.

- **5. `leaf=True` never compares the objects themselves, so different classes/container types compare equal** — NOT A BUG: `leaf=True` is documented as "only compare the object's leaf nodes (those with no children)", and class identity and container type are not leaves; a docstring note would suffice.
- **6. `leaf=True, union=False` misses keys that only the second object has** — NOT A BUG: `union=False` is documented as "rows correspond to the attributes of the first object", so not examining the second object's extra keys is expected (and it is only reachable through `sc.Equal`, see finding 17).
- **9. `sc.makenested()` `overwrite=False` raises only after creating intermediate levels** — NOT WORTH FIXING: with the default generator new levels are empty so the error can't fire after creating one; it needs a custom pre-filled `generator`, and the audit's second (list) example does not raise at all (it returns `{'L': [3, 4], 'x': {'L': {3: 99}}}`).
- **13. `sc.flattendict()` silently overwrites entries when the separator occurs in a key** — NOT WORTH FIXING: collisions are inherent to any separator-joined key scheme with a caller-chosen `sep`, and the examples are contrived (a literal `'a_b'` beside a nested `a.b`, or a top-level key named `'root'`).
- **14. `sc.mergenested()` inserts live references to `dict2`'s substructures** — NOT A BUG: this is normal shallow-merge aliasing (as with `dict.update` or `{**a, **b}`), and `dict2` is never mutated by the function; the docstring could mention it.
- **20. `search()` returns a `'root'` trace that `sc.getnested()` cannot resolve** — NOT WORTH FIXING: `'root'` is the documented `rootkey` convention and the root genuinely matches `type=dict`; not being able to pass it to `getnested` is an API wart, not a wrong result.
- **24. `sc.makenested()` docstring Example 2 states the wrong result** — NOT WORTH FIXING: docstring comment only (worth a one-line docs fix).
- **25. `skip=dict(instances=...)` cannot take instances** — NOT WORTH FIXING: it works when given classes (meaning `isinstance`); its overlap with `subclasses` is a design quirk, not a malfunction.
- **26. `search()` key matching slices the last character of the root key** — NOT WORTH FIXING: `key='t'` matching the root is a single-character corner case.
- **27. `search()` docstring regex example unpacks keys as if they were keys and values** — NOT WORTH FIXING: docstring only, and the example runs.
- **28. `Equal.is_subkey()` is dead code** — NOT A BUG: unused code is a style issue, not a defect (and the helper is useful for finding 4's fix).

## Misplaced `# pragma: no cover`

Three pragmas sit on reachable paths, hiding exactly the code that most needs testing. (This is coverage bookkeeping rather than a bug, and was out of scope for the re-verification.)

| Line | Branch | Reachable via |
|------|--------|---------------|
| 131 | `elif not overwrite and value is not None: # pragma: no cover` | `sc.setnested({'a':{'b':1}}, ['a','b'], 2, overwrite=False)` -> `ValueError: Not overwriting entry ['a', 'b'] since overwrite=False`; also hit by `tests/test_nested.py:93` |
| 748 | `pass # same leaf value # pragma: no cover` | `sc.mergenested({'a':1}, {'a':1})` -> `{'a': 1}` (takes this branch) |
| 751 | `if die: # pragma: no cover` | `sc.mergenested({'a':1}, {'a':2}, die=True)` -> `ValueError: Warning! Conflict at a: 1 vs. 2` |

## Verified clean

Recorded so the same ground isn't re-covered. All of the following were hypothesised, tested by execution, and found correct.

### `getnested()` / `setnested()` / `makenested()`

Set-then-get returns what was set for keylists of length 1, 2 and 5 (`sc.setnested(foo, ['a','b','c','d','e'], 42)` then `sc.getnested(...)` -> 42); for integer keys creating new levels (`sc.setnested({}, [0,1], 5)` -> `{0: {1: 5}}`, `sc.setnested({}, ['a',0], 5)` -> `{'a': {0: 5}}`); for keys that are genuine list indices (`sc.setnested({'L':[1,2,3]}, ['L',1], 99)` -> `{'L': [1, 99, 3]}`, get -> 99); and through a mixed dict/list/object/odict path (`sc.setnested({'a':[{'b':1}]}, ['a',0,'b'], 99)`, `sc.setnested(sc.prettyobj(), ['data','numbers'], [1,2,3,4])` then `sc.getnested(o, ['data','numbers',0])` -> 1). `setnested()` on a path that does not exist creates it rather than raising, for dicts, `objdict`s and `prettyobj`s, including with `generator=`; the exceptions are extending a list or an empty `odict` by index, which raises `IndexError` (arguably out of contract — a list cannot be grown by assignment), and `__slots__` objects (finding 32). `copy=True` genuinely leaves the input untouched and returns the modified deep copy; `copy=False` mutates in place and returns the same object (`res is d` -> True). `default=0` and `default=False` are *not* swallowed by the `if default is not None` guard (both correctly set `safe=True` and are returned), so this is not an instance of the falsy-sentinel pattern found in `sc_math.py`. `getnested(obj, [])` returns the object itself; a string keylist and a tuple keylist both work via `sc.tolist(..., coerce='tuple')`. Missing *intermediate* levels with `safe=True` return the default rather than raising (`sc.getnested({'a':1}, ['x','y'], safe=True)` -> None). Docstring Examples 1 and 3 of `makenested()` run exactly as written.

### `iterobj()` / `IterObj`

Traversal visits every node exactly once for acyclic objects with no shared children: a generated object with 4801 nodes produced exactly 4801 `func` calls, and `inplace=True` and `inplace=False` visited the same 16 nodes of a mixed dict/list/object tree (they differ in the aliased-child case reported in finding 1, and when `func` replaces a container, finding 31). Circular references terminate correctly in all three forms tested: a dict containing itself (`d['self'] = d` -> `['root', ('x',)]`), two objects referencing each other (`a.b = b; b.a = a` -> `['root', ('b',), ('b','val'), ('val',)]`), and a self-referencing list (`L.append(L)` -> `['root', (0,), (1,)]`); `recursion=1` re-parses each one exactly once more and still terminates. `leaf=True` visits only leaves for any object with at least two nodes (the root-node problems are reported in findings 22, 29, and 30). `depthfirst=True`/`False` produce genuinely depth-first and breadth-first key orders (`[('a',), ('a','x'), ('a','y'), ('b',), ('b','z')]` vs `[('a',), ('b',), ('a','x'), ('a','y'), ('b','z')]`). `atomic='default'` correctly refuses to descend into tuples/arrays/DataFrames; `atomic='default-tuple'` correctly descends into tuples (non-`inplace`). `skip` works in every form except `instances` given an instance (rejected finding 25): `skip='a'`, `skip=id(obj)`, `skip=dict(keys='a')`, `skip=dict(ids=id(obj))` and `skip=dict(subclasses=list)` all prune the expected subtrees, and an unrecognized dict key raises a clear `KeyError`. No `id()`-reuse false skip was observed under `inplace=True` with a `func` that frees each visited container (4801/4801 nodes still visited). `**kwargs` are forwarded to `func` correctly. The `IterObj` class docstring example (custom_type/custom_iter/custom_get with a `DataObj`) produces the expected `[1,2,3,4,5,6,7,8,9,10]`, and both `iterobj()` docstring examples produce the documented output. `check_iter_type(arr, check_array=True)` does return `'array'` (the branch is not shadowed by the `__dict__`/`__slots__` test, since ndarray has neither). Passing a single class as `custom_type` calls it as a factory (`custom_func = custom` when `callable`) but is harmless for zero-argument-constructible classes and does not misclassify. `to_df()` depths are correct without `flatten`, and `flatten_traces()` preserves the output dict's type.

### `mergenested()`

`dict1` is *not* mutated by a top-level call (verified with a `sc.dcp` snapshot before and after, including the recursive-merge case); `dict2` is never mutated by the merge itself (the result shares `dict2`'s subtrees, which is ordinary shallow-merge behavior; see rejected finding 14). Conflicting leaf types resolve in favour of `dict2` in both directions (`{'a':{'b':1}}` + `{'a':5}` -> `{'a': 5}`; `{'a':5}` + `{'a':{'b':1}}` -> `{'a': {'b':1}}`), matching "last writer wins"; `die=True` raises `ValueError` with the conflict path in the message and, because `dict1` was copied, leaves both inputs intact; `die=False` warns only under `verbose=True`. Identical scalar leaves are left alone.

### `flattendict()` / `nestedloop()` / `iternested()`

Both `flattendict()` docstring examples produce exactly the documented output, with and without `sep`. `nestedloop()` reproduces both documented orders verbatim (`[0,1]` -> `[['a',1],['a',2],['b',1],['b',2]]`, `[1,0]` -> `[['a',1],['b',1],['a',2],['b',2]]`) and generalizes consistently to three lists (`loop_order=[2,0,1]` makes list 2 the outermost loop); the reordering/un-reordering via `out[loop_order[i]] = item[i]` is self-inverse for any permutation. `iternested()` treats every non-dict value as a twig (including lists) as documented, and returns `[]` for a dict whose only value is an empty dict.

### `sc.equal()` scalar and type discrimination

All of the following are correctly reported as *not* equal, checked in both argument orders where asymmetry was possible: `1` vs `1.0`, `1` vs `True`, `0` vs `False`, and the same three nested one level deep in a dict (the `type(bconv) != type(oconv)` guard catches them even though Python's own `dict.__eq__` says the dicts are equal); `None` vs `np.nan`; `dict(a=None)` vs `dict()`; `np.float64(1.0)` vs `1.0`. An extra key or attribute in either object is caught in both directions (plain objects, dicts, and nested dicts), as is `dict(x=1, y=[])` vs `dict(x=1)` and `dict(x=1, y={})` vs `dict(x=1)` (empty container vs missing key), and `dict(a=1)` vs `dict(b=1)`. Container-type differences at the default `leaf=False` are caught: list vs tuple, list vs dict, dict vs `sc.odict`, dict vs `collections.OrderedDict`, two distinct classes with identical `__dict__`.

### `sc.equal()` depth, length, and collection contents

A single differing scalar buried at depth 4 inside a list-of-dicts-of-lists (`dict(x=[dict(y=1), dict(z=[1,2,dict(w=3)])])`) is found by all four methods (`'eq'`, `'pickle'`, `'json'`, `'str'`) and by the default chain. Prefix-length differences (`[1,2]` vs `[1,2,3]`, both orders, bare and nested) are caught. A differing element inside a `set` or a `frozenset`, a differing value inside a tuple, and a differing *tuple key* of a dict (`{(1,2):'a'}` vs `{(1,3):'a'}`) are all caught. Dicts and `sc.odict`s that differ only in key order are reported equal, which is correct: `sc.odict.__eq__` is inherited from `dict` and is order-insensitive (`sc.odict` derives from `dict`, not `OrderedDict`), so `sc.equal` agrees with the objects' own `==`; `method='pickle'` and `method='str'` disagree (they see the order), which is the documented "different methods may give different results in edge cases."

### `sc.equal()` numpy arrays

Shape differences do *not* produce a broadcast false positive: `np.array([1,2,3])` vs `np.array([[1,2,3]])`, `np.array([1,1,1])` vs `np.array([[1],[1],[1]])`, `np.array([1,1])` vs `np.array(1)`, and `np.array([1,2,3])` vs `np.array([1,2,3,4])` all return `False`, matching `np.array_equal` (`sc.nanequal(..., scalar=True)` checks shape before comparing). `np.array([1,2,3])` vs `np.array(['1','2','3'])` is `False`. Integer vs float arrays of equal value are reported equal (`method='eq'`) while `'pickle'`/`'json'`/`'str'` say not equal — this is `np.array_equal` semantics for the `'eq'` method and is treated here as intended rather than a defect, but it is worth knowing the methods disagree.

### `sc.equal()` NaN handling

`equal_nan=True/False` behaves correctly for a bare `float('nan')`, a NaN inside a list, a NaN as a dict value, a NaN inside a numpy array, and a NaN inside an array nested in a dict — `equal_nan=False` returns `False` in every case and `equal_nan=True` returns `True`. The De Morgan condition in `compare_special()` (`if not np.isnan(obj) or not np.isnan(obj2) or not self.equal_nan`) was checked and is correct. `sc.equal(np.nan, np.nan)` agrees across all four methods.

### `sc.equal()` circular references and other structure

A self-referencing dict (`d['self'] = d`) does not hang and compares correctly against both an identical and a differing copy. `pd.Index` values are compared correctly (equal and differing). Dataframes with different column *names*, different column *order*, or an extra column are correctly reported unequal (only the *index* is ignored — finding 7). `detailed=True` and `detailed=2` produce the right differing trace: an array change at `o1['b'][2]` marks only `root` and `(b,)` as unequal, a change at `o1['c']['d']` marks only `root`, `(c,)` and `(c, d)`, and in both cases the dataframe's verdict agrees with the boolean returned by `detailed=False`. `__slots__`-based objects are traversed by `sc.iterobj()` and compared correctly (writing to them is not; see finding 32). Attribute-level and key-level differences are found even when the object's own `__eq__` raises, because the child traces are compared independently of the failed root comparison.

### `sc.search()` completeness and round-tripping

Built an object with matches at five different depths — a top-level dict value, an element of a list, a dict nested inside a list, an object attribute, and an element of a list inside a nested object's attribute — and `sc.search(obj, value=7)` found all five; every returned trace round-trips through `sc.getnested()` and yields the matched object (the only trace that does *not* round-trip is `'root'`, see rejected finding 20). All five docstring examples were run verbatim; the first four behave as documented, including the `assert sc.getnested(nested, valmatches) == val` round-trip, callable `value=find` (finds all three of `3`, `4`, `8`), and `method='partial'` with `leaf=True`. No-match searches return an empty `objdict()` rather than raising, for `key=`, `value=`, and `type=`. `type=` matching works for single types and correctly excludes non-matching ones. `flatten=True` is applied after matching, so it does not disturb the matching, and `flatten` is correctly popped from `kwargs` rather than reaching `sc.iterobj()`. Duplicate matches (a node matching both `key` and `value` via `query`) are de-duplicated by the final dict comprehension, and the output preserves tree order rather than match order. Integer list indices are matched as keys (`key=2` finds `('b','cat',2)`), which is consistent with how `sc.iterobj()` builds traces. The mutual-exclusion guards for `query` vs `key`/`value` and `type` vs `key`/`value` both fire correctly.

### Argument plumbing

Every keyword in `search()`'s signature was traced into the body: `query` (sets both `key` and `value`), `key`, `value`, `type` (builds an `isinstance` lambda), `method` (read by the `check_match` closure), and `kwargs` (forwarded to `sc.iterobj()`, with `flatten` intercepted) all reach live code. In `Equal.__init__`, `method`, `detailed`, `equal_nan`, `leaf`, `union`, `verbose`, and `compare` all reach live code; only `die` does not (finding 18). `Equal.check_method()` correctly rejects an unknown method and accepts a callable; `get_method()` correctly falls back to `self.method[0]`. No in-place mutation of the compared objects was observed: `sc.equal` only reads, and the one `sc.dcp` in the hot loop (`methods = sc.dcp(self.method)`) is a copy of the method list, not of user data. `Equal.n`, `base`, `others`, `bdict`, and `odicts` are consistent with `self.objs`/`self.dicts` (`base` and `others` are unused but are documented properties of a public class, not dead private code).

## Suggested order of work

1. **Findings 22, 29, 30, 12** — the root-node handling in `iterobj()` and `Equal`. Finding 22 makes `leaf=True` unusable with ordinary leaf functions, and deciding root inclusion once (from `leaf` and the root's iter type, via `self.rootkey`) fixes all four together.
2. **Findings 31, 16, 1** — the `inplace=True` traversal: descend into the value `func` returned (31, which also fixes 16), and emit rather than drop memo hits (1). Fixing these first avoids re-testing `equal()`/`search()` twice, since both traverse via `iterobj()`.
3. **Findings 4, 18, 7** — the `sc.equal()` false positives and `die` plumbing, fixed together with the "only leaf/atomic `None`s count as unequal" rule and "`die` only after all methods fail".
4. **Findings 2, 3, 8, 32** — the `safe`/`overwrite` inversions, the `odict` integer-key data loss, and slots support in `getnested()`/`setnested()`/`makenested()`, all traceable to the same "is this key present / set this key" helpers.
5. **Findings 10, 11, 15, 17, 19** — the remaining medium-severity plumbing (`*args`, flattened depths, array leaves in `mergenested()`, `union` dropped, `search()` crashing on arrays).
6. **Findings 21, 23 and the pragma list** — narrow edge cases and coverage bookkeeping.

Findings 1, 4, 7, 8, 12, and 31 are the ones that matter most to fix carefully: each produces a silently wrong verdict (a `sc.equal()` call reporting `True` for objects that differ), silently lost data, or a half-transformed object, rather than a visible error a caller would notice and investigate.
