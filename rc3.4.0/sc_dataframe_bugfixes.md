# `sc_dataframe.py` bug audit

Audit of `sciris/sc_dataframe.py` (1257 lines), the `sc.dataframe` subclass of `pd.DataFrame`, for genuine defects: wrong results, documented arguments that don't work, silent data corruption/loss, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps, and performance.

**Scope**: the entire file, covered by two parallel auditors working on non-overlapping halves — lines 24-650 and lines 651-1257 — using the same method: line-by-line reading followed by executed hypothesis tests against the editable install (Sciris 3.3.0, numpy 2.4.6, pandas 3.0, commit `2d69aad`), with every finding reproduced a second time independently before being recorded. Line numbers refer to the working tree at that commit.

**Re-verification**: this document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`, pandas 3.0.5, numpy 2.4.6, `SCIRIS_BACKEND=agg`). All 36 original repros reproduce as written, and the line numbers still match the source. Of the original 36 findings, 21 were confirmed (two with a corrected severity), 2 were rewritten because the original description was inaccurate (18, 21), and 13 were rejected as not a bug or not worth fixing (see "Rejected on review" near the end). Six bugs the original audit missed were added as findings 37-42. Several proposed fixes were corrected (findings 1, 4, 21). The "Verified clean" section below has been corrected where it made three false claims. The document now lists **29 findings: 5 High, 19 Medium, 5 Low**.

**Nothing in this document has been applied.** All fixes are described, not made.

## Summary

| # | Severity | Method | Defect | Line |
|---|----------|--------|--------|------|
| 1 | High | `flexget()` | Integer `rows` returns an empty dataframe instead of the row | 361 |
| 2 | High | `appendrow()`, `append()`, `insertrow()`, `concat()`, `cat()`, `_sanitize_df()` | A dict of columns is silently transposed on the way in | 636-648 |
| 3 | High | `poprow()` | Removes every row sharing the popped row's index label, but reports only one | 903 |
| 4 | High | `findinds()`, `poprows()`, `filterin()`, `filterout()`, `replacecol()` | Fuzzy (`np.isclose`) matching corrupts exact-value row selection (root cause in `sc.findinds()`, `sc_math_bugfixes.md` #12) | 1073, 999 |
| 37 | High | `poprows()`, `filterin()`, `filterout()` | A list passed as `value` is treated as a boolean mask, silently deleting the wrong rows | 935, 1080 |
| 5 | Medium | `dataframe()` | Mutates the caller's data dict when columns are also passed as keywords | 99 |
| 6 | Medium | `col_name()` | Returns the wrong column (or raises `UnboundLocalError`) when any argument is `None` | 225-226 |
| 8 | Medium | `flexget()`, `__getitem__()` | Documented `cast`/`default` arguments do nothing | 324, 258 |
| 12 | Medium | `insertrow()` | Refuses multiple rows unless their count happens to equal the column count | 611-618 |
| 13 | Medium | `insertrow()` | `reset_index=False` does nothing: the index is always reset | 632 |
| 14 | Medium | `_sanitize_df()`, `appendrow()` | The shape probe crashes when a row contains a sequence | 644 |
| 15 | Medium | `addcol()` | Mutates the caller's dictionary | 786 |
| 16 | Medium | `poprow()` | Rejects numpy integers, including the output of `df.findinds()` | 896 |
| 17 | Medium | `to_odict()` | `row=<int>` raises `IndexError`, although `row` is documented as `int`/`list` | 1013 |
| 18 | Medium | `to_odict()` | Upcasts all columns to one common dtype (int to float, or everything to object) | 1013 |
| 19 | Medium | `findrow()` | `asdict=True` raises `IndexError` — the documented example does not work | 1049 |
| 20 | Medium | `filtercols()` | `die=False` mutates the list it is iterating, silently accepting a second missing column and returning a NaN column | 1128 |
| 21 | Medium | `sortrows()` | `returninds=True` returns indices that do not reproduce the sort whenever there are ties (even ascending), when `reverse=True`, or when `by` is a list | 1166 |
| 22 | Medium | `sort()` | Silently ignores `inplace`, always modifying the dataframe | 1183 |
| 24 | Medium | `sortcols()` | Discards the row index | 1203 |
| 25 | Medium | `read_csv_string()` | Only strips the outer string, so an indented CSV literal keeps the indentation in every value after the header | 1239 |
| 38 | Medium | `replacecol()` | `old=None` replaces every truthy value and leaves the `None` values alone; `old=np.nan` is a silent no-op | 999 |
| 39 | Medium | `appendrow()`, `concat()`, `_sanitize_df()` | A list of record dicts is stored as dict cells when the number of records equals `ncols` | 644-646 |
| 40 | Medium | `appendrow()`, `insertrow()`, `_sanitize_df()` | Appending or inserting a scalar row into a single-column frame crashes | 645-647 |
| 7 | Low | `get()` | Shadows `pd.DataFrame.get()` with an incompatible signature and behaviour | 248 |
| 23 | Low | `sortcols()` | Ignores `reverse` whenever `sortorder` is supplied | 1197 |
| 35 | Low | `sortcols()` | Demotes a `sc.dataframe` subclass back to the base class | 1202 |
| 41 | Low | `equal()`, `equals()` | Crashes on a categorical column containing NaN | 406 |
| 42 | Low | class docstring | Headline example calls `rmcol()`/`rmrow()`, which do not exist | 52, 60, 61 |

Rejected on review (not in this table): 9, 10, 11, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36. See "Rejected on review" near the end.

## Recurring patterns

**A dict of columns is silently transposed on the way in — found independently from both halves.** `_sanitize_df()` (lines 636-648) flattens a dict argument to `columns = list(arg.keys()); arg = list(arg.values())` and hands the column-major `arg` to `self._constructor(data=arg, columns=columns)`, which reads a list of lists as row-major. The auditor covering lines 24-650 found this via `appendrow()`/`append()` (and noted `concat()` shares the same code path); the auditor covering lines 651-1257 found the identical root cause independently via `concat()`/`cat()` (and noted `insertrow()` inherits it through `cat()`). See finding 2 for the merged writeup covering all five entry points.

**Try-the-fast-path-first, fall back on failure (sometimes a bare `except`).** `__getitem__()`, `__eq__()`, and `equals()` all attempt the pandas/base-class operation first and only apply Sciris semantics when it raises, which makes the same expression mean different things depending on the data's labels/shape/dtypes rather than the caller's intent (`df[0]` is a row or a column depending on whether the columns are integers; this label-first rule matches pandas and was rejected as a bug on review, see former findings 9 and 10). Lines 262, 264, 302, and 447 use bare `except:` blocks around multi-statement code with no `die`/`verbose` escape hatch to see the underlying exception — the `flexget()` int-row bug (finding 1) is a direct casualty: the first failure is swallowed and silently turned into an empty result instead of surfacing.

**Falsy/`None` sentinels handled inconsistently.** `col_name()`'s `None`-means-first-column branch assigns to the wrong variable (finding 6); `dataframe()`'s dict-plus-kwargs handling and `addcol()`'s identical pattern (findings 5, 15) both alias and mutate the caller's dict rather than copying it first. `replacecol()` passes `old=None` straight to `sc.findinds()`, where `None` means "find nonzero entries", so asking to replace `None` overwrites the real data instead (finding 38).

**Shape-coincidence heuristics in `_sanitize_df()`.** Whether an input is treated as one row or many is decided by `np.array(arg).shape == (self.ncols,)`. This single test is behind the dict-of-columns transpose (finding 2), the crash on a row containing a list (finding 14), list-of-record-dicts being stored as dict cells (finding 39), and the crash on a scalar row for a single-column frame (finding 40). The four fixes should be made together.

**List-valued `value` passed to `sc.findinds()`.** `sc.findinds()` treats an array-like `val` as another boolean mask rather than as a set of values to match, so every value-based row selector silently misbehaves when given a list (finding 37). Together with the relative-tolerance problem (finding 4), this means `sc.findinds()` is the root of most of the value-selection bugs in this file.

**A quantity recomputed independently of the operation it is supposed to describe.** `sortrows()`'s `returninds` is computed with its own stable `np.argsort` rather than derived from the actual (unstable by default) `sort_values()` call, so it drifts out of sync with ties, `reverse`, and list-valued `by` (finding 21). `poprow()`'s position-to-label conversion is a variant of the same shape: it converts a row position to an index label and then acts on the label, which no longer identifies the same row once labels are duplicated (finding 3).

**Arguments forwarded by literal instead of by variable.** `sort()` passes `inplace=True` instead of `inplace=inplace` (finding 22); `sortcols()` omits `reset_index` when calling `replacedata()` and so inherits its `reset_index=True` default (finding 24). Both are one-token fixes that change behaviour the caller explicitly requested.

**`.values` on a multi-column selection used as a row/dict accessor.** `to_odict()` takes `self.iloc[...].values` of a whole frame, which forces one common dtype across mixed-dtype columns and destroys per-column typing (finding 18); `findrow()` does the same, which is acceptable there because it returns a single 1-D row. The same `.values` habit is what makes `findrow(asdict=True)` feed data values back in as row positions (finding 19). `enumrows()` shows the correct alternative: it iterates the Series column-by-column and preserves dtypes exactly.

**Column/row resolution is duplicated, not shared.** `col_index()`, `col_name()`, `__getitem__()`, and `__setitem__()` each re-implement "is this a name or a position?" with slightly different rules, which is how `col_name()`'s `None` branch drifted out of sync with `col_index()`'s (finding 6). (Get/set disagreeing about bare integers was originally reported as findings 9 and 10, but is standard pandas behaviour and was rejected on review.)

## High severity

### 1. `flexget()` with an integer `rows` returns an empty dataframe instead of the row — `sc_dataframe.py:361`

The docstring documents `rows (int/list)`, but for an int the `self.iloc[rowindices, colindices]` result is a *Series* (index = column names, `.name` = the row label). Passing that Series to `self._constructor(data=output, columns=[...])` makes pandas treat `columns` as a *selection* over the Series' name rather than as a set of labels, so nothing matches and the row silently vanishes.

```python
df = sc.dataframe(cols=['x','y','z'], data=[[1238,2,-1],[384,5,-2],[666,7,-3]])
print(df.flexget(cols=['x','z'], rows=0))
```

```
actual:                        expected:
Empty dataframe                      x  z
Columns: [x, z]                0  1238 -1
Index: []
```

`.size` is `0`, so the caller gets a well-formed but empty dataframe with the right columns — no exception, and `len()`/`.size` checks are the only way to notice. `asarray=True` is unaffected (`np.array(series)` -> `[1238, -1]`), and a single column is unaffected because the `output.size == 1` branch returns the scalar. So the bug needs `asarray=False` (the default) plus two or more columns. `tests/test_dataframe.py:104` calls `df.flexget('c', 2)` — exactly the int-row case, but with one column, so it takes the scalar branch and the broken path stays untested.

Re-verified: the same empty result is returned with `cols=None`.

**Fix**: wrap an integer `rows` in a list before indexing — `if sc.isnumber(rows): rows = [rows]` — or special-case a Series result by reconstructing with `output.to_frame().T`. (Corrected on review: the original suggestion of `rowindices = sc.tolist(rows)` would wrap a slice as `[slice]` and break the slice form, so only wrap when `rows` is a number.)

### 2. `appendrow()`/`append()`/`insertrow()`/`concat()`/`cat()` silently transpose a dict of columns — `sc_dataframe.py:636-648`

`_sanitize_df()` flattens a dict to `columns = list(arg.keys()); arg = list(arg.values())` (column-major) and then feeds `arg` to `self._constructor(data=arg, columns=columns)`, which interprets a list of lists as **row-major**, so passing a dict whose values are sequences writes the data out transposed. The single-row dict case (the documented one) happens to survive because `argarray.shape == (self.ncols,)` wraps it in an extra list; a dict of multi-element lists never hits that guard. This defect was found independently by both halves of this audit — once via `appendrow()`/`append()`, and once via `concat()`/`cat()` — because `_sanitize_df()` is the single entry point shared by all of them (`concat()` calls it directly; `appendrow()`/`append()` call it; `insertrow()` calls `self.cat()`, which calls it too).

```python
df = sc.dataframe(dict(a=[1,2], b=[3,4]))
df.appendrow(dict(a=[10,20], b=[30,40]))
print(df.to_dict('list'))
```

```
actual:   {'a': [1, 2, 10, 30], 'b': [3, 4, 20, 40]}
expected: {'a': [1, 2, 10, 20], 'b': [3, 4, 30, 40]}
```

```python
df2 = sc.dataframe(dict(x=[1,2,3], y=[4,5,6]))
df2.concat(dict(x=[10,20], y=[30,40]))
```

```
actual:                              expected (what df2.concat(sc.dataframe(dict(x=[10,20], y=[30,40]))) gives):
    x   y                                  rows (10, 30) and (20, 40)
0   1   4
1   2   5
2   3   6
3  10  20     <- should be (10, 30)
4  30  40     <- should be (20, 40)
```

No warning or error is raised; the values are simply scrambled. Whether you get corruption or a crash depends on a shape coincidence — the corruption happens when the number of appended/concatenated rows equals `self.ncols`, otherwise pandas notices: for a 3-column frame, `df3.appendrow(dict(a=[10,20], b=[30,40], c=[50,60]))` raises `ValueError: 3 columns passed, passed data had 2 columns`, and `df2.concat(dict(x=[10], y=[30]))` on a 2-column frame raises `ValueError: 2 columns passed, passed data had 1 columns`. `sc.dataframe.cat(dict(a=[1,2], b=[3,4]), dict(a=[9,8], b=[7,6]))` is affected the same way, appending rows `(9, 8)` and `(7, 6)` instead of `(9, 7)` and `(8, 6)`, and `insertrow()`'s `value` argument routes through `self.cat()`, so `insertrow(1, dict(x=[1,2], y=[3,4]))` is corrupted identically. Dicts of *scalars* work correctly (`dict(x=10, y=30)`), which is the only form the tests and the `insertrow()` validator exercise, and is also the input form the module's own docstrings use everywhere else (`sc.dataframe(dict(a=[1,2], b=[3,4]))`), so the broken path is untested. `tests/test_dataframe.py:71` only exercises the scalar-valued dict (`df.appendrow(dict(a=4, b=4))`).

**Fix**: in `_sanitize_df()`, when `arg` is a dict pass it straight through to the constructor (`df = self._constructor(data=arg, **kwargs)`) instead of splitting it into `columns` + `list(values)`; keep the 1-D promotion only for the non-dict (true single row) case, e.g. by testing whether the dict values are scalars before wrapping. Scalar dicts must stay on a separate path, because `pd.DataFrame(dict(a=1, b=2))` raises without an index (confirmed on review). This fix should be made together with findings 14, 39 and 40, which share the same shape probe.

### 3. `poprow()` removes every row sharing the popped row's index label, but reports only one — `sc_dataframe.py:903`

The docstring says `poprow()` drops "by position rather than label", but the implementation converts the position to a label (`indexkey = self.index[row]`) and then calls `self.drop(indexkey)`, which is label-based and therefore removes *all* rows carrying that label. Duplicate index labels arise from ordinary Sciris calls — any `concat()`, `poprows()`, `filterin()`, or `filterout()` with `reset_index=False` leaves them — so a single `poprow()` can delete an arbitrary number of rows while returning just the one it claims to have popped.

```python
c = sc.dataframe(dict(x=[1,2])).concat(sc.dataframe(dict(x=[3,4])), reset_index=False)
print(c)             # index is 0, 1, 0, 1
r = c.poprow(0)      # pop the first row
print(r.values, c.nrows)
```

Actual:

```
   x
0  1
1  2
0  3
1  4
popped [1] remaining 2
```

i.e. `poprow(0)` on a 4-row frame leaves 2 rows (`x = 2` and `x = 4`); rows `x=1` *and* `x=3` were both destroyed, and the returned value mentions only `x=1`. Expected: 3 rows remaining, only `x=1` removed. `tests/test_dataframe.py:134` calls `df.poprow()` only on a frame with a unique default index.

**Fix**: drop positionally, e.g. `self._update_inplace(self.iloc[self._diffinds([rowindex]), :])`, or `self.drop(self.index[[rowindex]], inplace=True)` after confirming label uniqueness; alternatively route `poprow()` through `poprows()`, which is already positional and correct.

### 4. `df.findinds()` matches values it should not, so `poprows()`/`filterin()`/`filterout()`/`replacecol()` silently operate on the wrong rows — `sc_dataframe.py:1073`

`findinds()` delegates to `sc.findinds()`, which does an `np.isclose()` comparison, so an *exact-value* row lookup becomes an approximate one with numpy's default relative tolerance `rtol=1e-5`. Because the tolerance is relative, it grows with the magnitude of the data: any value within 1 part in 100,000 of the search value is treated as a match. Every value-based row selector in the class is built on this, so `poprows(value=...)`, `filterin(value=...)`, `filterout(value=...)`, and `replacecol()` all delete/keep/overwrite rows that do not hold the requested value. The `eps` argument does not help, because `rtol` stays active.

```python
df = sc.dataframe(dict(n=[100000, 100001, 200000]))
df.findinds(100000)                                  # actual: array([0, 1])   expected: array([0])
df.poprows(value=100000, col='n', inplace=False)     # actual: 1 row left (200000)   expected: 2 rows left (100001, 200000)
df.findind(100000)                                   # 0 -- the singular version is exact, and disagrees
```

Actual output:

```
[0 1]
        n
0  200000
0
```

Plain integer counts are enough to trigger it, and so are ordinary decimals and dates-as-floats:

```python
sc.dataframe(dict(v=[0.30, 0.300002, 0.5])).findinds(0.30)          # actual: array([0, 1])
sc.dataframe(dict(year=[2016.0, 2016.01, 2017.0])).findinds(2016.0) # actual: array([0, 1])
sc.dataframe(dict(pop=[1000000.0, 1000005.0, 2000000.0])).findinds(1000000.0, 'pop') # actual: array([0, 1])
d = sc.dataframe(dict(pop=[1000000.0, 1000005.0, 2000000.0])); d.replacecol('pop', 1000000.0, -1)
# actual: pop = [-1.0, -1.0, 2000000.0]   expected: [-1.0, 1000005.0, 2000000.0]
```

The class contradicts itself, which makes the bug hard to spot: `findind()` uses `coldata.tolist().index(value)`, an exact match, while `findinds()` is fuzzy. On the same frame `df.findind(1000004.0)` raises `IndexError: Item 1000004.0 not found`, but `df.findinds(1000004.0)` returns `array([0, 1])` and `df.poprows(value=1000004.0, col='pop')` deletes two rows for a value that is not present at all. Blast radius inside the file: `poprows()`, `_filterrows()` (hence `filterin()`/`filterout()`), and `replacecol()`. `tests/test_dataframe.py:38` exercises `df.poprows(value=666)` but only on well-separated values.

**Root cause**: the defect is in `sc.findinds()` itself, not in the dataframe code. At `sc_math.py:175` it calls `np.isclose(a=arr, b=val, atol=atol, **kwargs)` without setting `rtol`, so numpy's default `rtol=1e-5` stays active and `eps` only controls the absolute part. This is tracked as finding #12 in `sc_math_bugfixes.md`, which was kept as a real bug on review: `sc.findinds([100000, 100001, 100002], 100000, eps=0)` returns `[0 1]`, while passing `rtol=0` returns `[0]`. Note that `replacecol()` calls `sc.findinds()` directly (line 999) rather than going through `df.findinds()`, so a fix inside `sc_dataframe.py` alone would miss it.

**Fix**: fix it in `sc_math.py` by passing `rtol=0` by default in `sc.findinds()` (e.g. `kwargs.setdefault('rtol', 0)`), so that `eps` really is the tolerance. That fixes all five dataframe entry points at once. (Corrected on review: the original suggestion of an exact `coldata == value` match in the dataframe methods is not recommended, because it would remove the documented int-vs-float tolerance of `sc.findinds()` and would leave `findinds()` and `sc.findinds()` with different matching rules.) Until the `sc_math` fix lands, callers can pass `rtol=0` through `df.findinds(..., rtol=0)` / `poprows(..., rtol=0)`, which forward `**kwargs`; `filterin()`, `filterout()` and `replacecol()` do not forward kwargs.

### 37. `poprows(value=[...])`, `filterin(value=[...])` and `filterout(value=[...])` treat a list of values as a boolean mask and silently delete the wrong rows — `sc_dataframe.py:935`, `1080`

*Added on review (2026-09-25); missed by the original audit.*

`poprows()` documents `values (list): alternatively, search for these values to remove`, and passes the value to `sc.findinds(coldata, value)`. When `val` is array-like, `sc.findinds()` does not match the values: it treats `val` as a second boolean mask and ANDs it with the column's truthiness (`sc_math.py:176-178`, `187`). The result depends on which column entries are nonzero, not on whether they equal any of the requested values.

```python
d = sc.dataframe(dict(x=[0,1,2], y=[2,3,0]))
d.poprows(value=[5,6,7], col='y', inplace=False)   # none of 5, 6, 7 are present
```

- **Actual:** only the row `x=2, y=0` remains; every row with a nonzero `y` was deleted.
- **Expected:** all 3 rows are kept.
- **Other outcomes:** `filterout(value=[5,6,7], col='y')` gives the same silent deletion. When the list length differs from `nrows`, the call raises `ValueError: Could not handle inputs with shapes (3,) vs (5,)`.

No error or warning is raised in the equal-length case, so this is silent data loss on documented input.

**Fix**: in `poprows()`/`_filterrows()` (or in `sc.findinds()` itself), when `value` is list-like, use `np.flatnonzero(np.isin(coldata, value))`, or take the union of `sc.findinds(coldata, v)` over each `v` (which keeps the tolerance semantics once finding 4 is fixed).

## Medium severity

### 5. `dataframe()` mutates the caller's data dict when columns are also passed as keywords — `sc_dataframe.py:99`

`data.update(kwargs)` writes the keyword columns into the dict the caller passed in, so the caller's dict silently grows extra keys and any later use of it produces a different dataframe.

```python
d = dict(a=[1,2])
df = sc.dataframe(d, b=[3,4])
print(d)
```

```
actual:   {'a': [1, 2], 'b': [3, 4]}
expected: {'a': [1, 2]}
```

Consequence: `df2 = sc.dataframe(d)` afterwards unexpectedly has a `b` column. Both `sc.dataframe(d, b=...)` and `sc.dataframe(data=d, b=...)` are affected (verified separately). The same pattern occurs in `addcol()` (finding 15).

**Fix**: `data = {**data, **kwargs}` (or `data = dict(data); data.update(kwargs)`).

### 6. `col_name()` returns the wrong column (or raises `UnboundLocalError`) when any argument is `None` — `sc_dataframe.py:225-226`

The `if col is None:` branch assigns `col = 0` but never assigns `output`, and there is no re-dispatch afterwards, so `outputlist.append(output)` either raises (first iteration) or appends the value left over from the *previous* iteration. `col_index()` gets this right (`output = 0`); `col_name()` is a copy-paste of it with the assignment target wrong.

```python
df = sc.dataframe(dict(a=[1], b=[2], c=[3]))
df.col_name('b', None)   # actual: ['b', 'b']    expected: ['b', 'a']
df.col_name()            # actual: UnboundLocalError    expected: 'a'
df.col_name(None)        # actual: UnboundLocalError    expected: 'a'
```

```
--- col_name('b',None): ['b', 'b']
--- col_name(): EXC UnboundLocalError: cannot access local variable 'output' where it is not associated with a value
```

`col_name()` is documented with an explicit "return 0 if None" default (itself a copy-paste from `col_index()`; for `col_name()` it should say "the first column"), and `sc.dataframe.findrow()`/`findind()`/`sort()` all use the "`col=None` means first column" convention, so `None` is in contract. `tests/test_dataframe.py:143-145` covers only `col_name(1)`, `col_name('b')`, and `col_name(0, 2)`. Blast radius: no internal Sciris caller passes `None`, so the damage is confined to user code.

**Fix**: replace `col = 0` with `output = cols[0]` (and fix the docstring to say the first column name is returned).

### 8. `flexget()`'s documented `cast` and `default` arguments do nothing; `__getitem__()`'s `cast` likewise — `sc_dataframe.py:324`, `258`

Neither `cast` nor `default` is referenced anywhere in the bodies — `grep -n "cast" sc_dataframe.py` returns only the two signature lines (324, 258) and the docstring line (334); `default` appears only at 324/335 within these methods. `default` is documented as "the value to return if the column(s)/row(s) can't be found", but there is no `try/except` in `flexget()`, so a missing column or an out-of-range row raises regardless.

```python
df = sc.dataframe(cols=['x','y','z'], data=[[1238,2,-1],[384,5,-2],[666,7,-3]])
df.flexget(cols='q', rows=[0], default=-1)     # expected -1
df.flexget(cols='x', rows=[9], default=-1)     # expected -1
df.flexget(cols=['x','y'], rows=[0,1], asarray=True, cast=False).dtype  # same as cast=True
```

```
--- flexget(cols='q', default=-1): EXC TypeError: Unrecognized column/column type "q" <class 'str'>
--- flexget(cols='x', rows=[9], default=-1): EXC IndexError: positional indexers are out-of-bounds
--- flexget cast=False vs cast=True dtypes: (dtype('int64'), dtype('int64'))
```

**Fix**: either implement them (wrap the indexing in `try/except` returning `default`; apply `sc.toarray(..., dtype=float)`-style casting when `cast=True` and the selection is all-numeric) or delete them from the signatures and docstrings.

### 12. `insertrow()` refuses to insert multiple rows unless their count happens to equal the column count — `sc_dataframe.py:611-618`

The `die=True` validation tests `len(value) != self.ncols`, which treats `value` as a single row. For a 2-D `value` (documented: "value (array): the row(s) to insert", and the method's summary is "Insert row(s) at the specified location"), `len(value)` is the number of *rows*, so the check compares two unrelated quantities: inserting 2 rows into a 3-column frame is rejected, while inserting 3 rows into the same frame is accepted and works fine.

```python
df = sc.dataframe(dict(a=[1,2], b=[3,4], c=[5,6]))
df.insertrow(1, np.arange(6).reshape(2,3))    # ValueError: Length mismatch: expecting 3, but got 2
df.insertrow(1, np.arange(9).reshape(3,3))    # works, inserts 3 rows
df.insertrow(1, np.arange(6).reshape(2,3), die=False)  # works, inserts 2 rows
```

```
--- insertrow(1, 2x3 array) die=True: EXC ValueError: Length mismatch: expecting 3, but got 2
--- insertrow(1, 3x3 array) die=True: OK (3 rows inserted at position 1)
--- insertrow(1, 2x3 array) die=False: OK (2 rows inserted at position 1)
```

`appendrow()` has no such restriction (`df.appendrow(np.random.rand(2,3))` is exercised at `tests/test_dataframe.py:123`), so the two sibling methods disagree about what a valid row block is.

**Fix**: apply the length check per row — if `np.ndim(value) == 2` (or `value` is a list of lists), check `np.shape(value)[1]` against `self.ncols`; only use `len(value)` for the 1-D case.

### 13. `insertrow(reset_index=False)` does nothing: the index is always reset — `sc_dataframe.py:632`

`insertrow()` builds the new frame with `self.cat(before, value, after, **kwargs)`; `reset_index` is a named parameter of `insertrow()` so it is not in `**kwargs`, and `cat()` -> `concat()` defaults to `reset_index=True`. By the time `replacedata(newdf=newdf, reset_index=False)` runs, the index has already been discarded, so the original row labels are lost either way.

```python
df = sc.dataframe(dict(a=[1,2], b=[3,4]), index=[10,11])
df.insertrow(1, [9,9], reset_index=False)
print(list(df.index))
```

```
actual:   [0, 1, 2]
expected: [10, 0, 11]  (or any index that preserves the original labels)
```

`appendrow(..., reset_index=False)` does honour the flag (index becomes `[10, 11, 0]`), which confirms this is specific to `insertrow()`'s use of `cat()`.

**Fix**: pass the flag through, e.g. `newdf = self.cat(before, value, after, reset_index=False, **kwargs)` (or build the intermediate with `pd.concat` directly), leaving the final `replacedata()` call responsible for resetting. (Verified on review: passing `reset_index=False` to `cat()` gives the expected `[10, 0, 11]`.)

### 14. `_sanitize_df()`'s shape probe crashes when a row contains a sequence — `sc_dataframe.py:644`

The comment says the `np.array(arg)` conversion is "solely for checking the shape", but without `dtype=object` numpy refuses to build a ragged array, so appending a row to a frame that has an object column holding lists/arrays fails — even though the constructor accepts exactly such data. This is a distinct bug from finding 2 (both live in `_sanitize_df()`, but this one is a crash on a valid row, not a value transposition).

```python
df = sc.dataframe(dict(a=['x'], b=[[1,2]]))   # constructs fine
df.appendrow(['y', [3,4]])
```

```
actual:   ValueError: setting an array element with a sequence. The requested array has an
          inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.
expected: a 2-row frame with b = [[1,2], [3,4]]
```

The dict form fails identically (`df.appendrow(dict(a='y', b=[3,4]))`). Object columns containing arrays are a normal Sciris usage (results objects, parameter sets), and the failure is in a probe whose only purpose is to decide whether to wrap the row in a list.

**Fix**: use `np.array(arg, dtype=object)` for the probe, or avoid numpy entirely — e.g. `if not isinstance(arg[0], (list, tuple, np.ndarray, dict)) and len(arg) == self.ncols: arg = [arg]`. (Verified on review: with `dtype=object`, `['y',[3,4]]` probes as shape `(2,)` and a 2-D input still probes as `(2,2)`.) Merge with the fixes for findings 2, 39 and 40.

### 15. `addcol()` mutates the caller's dictionary — `sc_dataframe.py:786`

When columns are supplied as a dict, `data = key` aliases the caller's object and `data.update(kwargs)` then writes the keyword columns into it.

```python
nc = dict(z=[1,2,3])
sc.dataframe(dict(x=[1,2,3])).addcol(nc, w=[7,7,7])
print(nc)
```

Actual: `{'z': [1, 2, 3], 'w': [7, 7, 7]}`. Expected: `{'z': [1, 2, 3]}` — the caller's dict should not gain a `w` key. This matters when the same column-spec dict is reused across several dataframes, since each call accretes the previous call's keyword columns. Same pattern as finding 5 (`dataframe()`).

**Fix**: `data = dict(key)` (or `data = sc.dcp(key)`) before `data.update(kwargs)`.

### 16. `poprow()` rejects numpy integers, including the output of `df.findinds()` — `sc_dataframe.py:896`

The dispatch is `if isinstance(row, int)`, which is `False` for `np.int64`, so any numpy integer falls into the label branch and hits `self.index.get_indexer(row)`, which requires a list-like.

```python
d = sc.dataframe(dict(x=[1,2,3]))
d.poprow(d.findinds(2)[0])
```

Actual: `TypeError: Index(...) must be called with a collection of some kind, 1 was passed`. Expected: the row `x=2` popped. `findinds()` returns an `int64` array, so the natural "find a row, then pop it" chain fails, while `df.poprow(df.findind(2))` works because `findind()` happens to return a Python `int`. The same branch makes the documented label form fail too: on `sc.dataframe(dict(x=[1,2,3]), index=['a','b','c'])`, `df.poprow('b')` raises the same `TypeError`, so the `else` branch is dead as written.

**Fix**: use `sc.isnumber(row)` (or `isinstance(row, (int, np.integer))`) for the positional test, and wrap the label in a list for `get_indexer` (`self.index.get_indexer([row])[0]`).

### 17. `to_odict(row=<int>)` raises `IndexError`, although `row` is documented as `int`/`list` — `sc_dataframe.py:1013`

`self.iloc[<int>,:]` returns a 1-D Series, so `.values` is 1-D and the subsequent `data[:,c]` indexing fails.

```python
df = sc.dataframe(cols=['year','val'], data=[[2016,0.3],[2017,0.5]])
df.to_odict(0)
```

Actual: `IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed`. Expected: a one-row odict. `to_odict([0])` and `to_odict(slice(0,2))` work, so only the documented `int` form is broken. `tests/test_dataframe.py:122` calls `to_odict()` with no argument.

**Fix**: normalise a scalar to a list at the top (`if sc.isnumber(row): row = [row]`), as `col_index()` does for its own inputs.

### 18. `to_odict()` upcasts all columns to one common dtype — `sc_dataframe.py:1013`

*Rewritten on review (2026-09-25): the original version also blamed `findrow()` and led with the 2^53 precision case; both were misleading.*

`to_odict()` goes through `self.iloc[...].values` on the whole frame, which flattens a mixed-dtype frame to a single common dtype before splitting it back into columns. So an integer column comes back as `float64` whenever any float column is present, and every column comes back as `object` whenever a bool or string column is present. `enumrows()`, by contrast, iterates column by column and preserves per-column dtypes exactly.

```python
sc.dataframe(dict(n=[1,2], val=[0.1,0.2])).to_odict()['n'].dtype             # float64, expected int64
sc.dataframe(dict(flag=[True,False], val=[1.5,2.5])).to_odict()['val'].dtype  # object, expected float64
```

The practical impact is the changed dtype (integer IDs or counts arriving as floats, numeric columns arriving as `object`). Loss of precision is the extreme case: integers above 2^53 are rounded, e.g. `uid=[9007199254740993, 9007199254740995]` next to a float column comes back as `[9007199254740992, 9007199254740996]`.

`findrow()` is not affected in the same way and is not part of this finding. It returns a single row as a 1-D array, which must have one common dtype. The only issue there is that its docstring (line 1039) claims `array([2016, 0.3], dtype=object)` while the actual result is `array([2.016e+03, 3.000e-01])`; that is a documentation mismatch.

**Fix**: build the odict column by column (`{col: self[col].values[row] for col in self.cols}`) so each column keeps its own dtype. Separately, correct the `findrow()` docstring example.

### 19. `findrow(asdict=True)` raises `IndexError` — the documented example does not work — `sc_dataframe.py:1049`

`thisrow` at this point is the row's *values array*, but it is passed as the `row` argument of `to_odict()`, which does `self.iloc[row,:]` — i.e. the data values are used as row positions.

```python
df = sc.dataframe(cols=['year','val'], data=[[2016,0.3],[2017,0.5]])
df.findrow(2016, asdict=True)
```

Actual: `IndexError: positional indexers are out-of-bounds`. Expected, per the docstring at line 1042: `{'year':2016, 'val':0.3}`. `asdict` is unusable for any frame whose first-column values are not also valid row positions, and silently returns the wrong row when they are. Not covered by `tests/test_dataframe.py` (line 36 calls `findrow(555)` without `asdict`).

**Fix**: build the dict from the row directly, e.g. `thisrow = sc.objdict(zip(self.cols, thisrow))`, or pass the *index* rather than the values (`self.to_odict([index])`) and unwrap the length-1 arrays.

### 20. `filtercols(die=False)` mutates the list it is iterating, so a second missing column is silently accepted and returns a NaN column — `sc_dataframe.py:1128`

`cols.remove(col)` inside `for col in cols:` shifts the iteration, so the item after a missing column is skipped. The skipped name is never looked up, never reported in the "could not find" message, and stays in `cols`; the resulting `_constructor(cols=cols, data=ordered_data)` then reindexes by name and fills it with NaN — while a genuinely requested column is dropped.

```python
a = sc.dataframe(cols=['a','b','c'], data=[[1,2,3],[4,5,6]])
a.filtercols('zz','a','b', die=False)
```

Actual:

```
sc.dataframe(): could not find the following column(s): ['zz']
Choices are: ['a', 'b', 'c']
    a  b
0 NaN  2
1 NaN  5
```

Expected: column `a` = `[1, 4]` and `b` = `[2, 5]` (with only `zz` reported missing). Column `a` — which exists — has been silently replaced by NaN. The variant `a.filtercols('zz','yy','a', die=False)` reports only `zz` missing and returns a phantom `yy` column of NaN. Reachable only with `die=False` (the default `die=True` raises first), and the branch is marked `# pragma: no cover`.

**Fix**: iterate over a copy (`for col in list(cols):`) or collect the missing names first and rebuild `cols` afterwards.

### 21. `sortrows(returninds=True)` returns indices that do not reproduce the sort whenever there are ties, when `reverse=True`, or when `by` is a list — `sc_dataframe.py:1166`

*Rewritten on review (2026-09-25): the original version understated the bug and claimed the default ascending case was correct.*

`sortorder` is computed by an independent `np.argsort(self[by].values, kind='mergesort')` that never sees `ascending`/`reverse` and assumes `by` is a single column. The frame is then sorted by `sort_values()`, so the returned "indices used for sorting" describe a different permutation than the one applied. There are three separate ways they diverge.

**Ties (including the default ascending sort).** `np.argsort(kind='mergesort')` is stable, but `sort_values()` defaults to `kind='quicksort'`, which is not. With tied values the two permutations differ, so the returned indices are wrong even for a plain `d.sortrows(returninds=True)`:

```python
rng = np.random.default_rng(0)
d = sc.dataframe(dict(x=rng.integers(0,5,1000), y=np.arange(1000))); orig = d.copy()
inds = d.sortrows(returninds=True)
(orig.y.values[inds] != d.y.values).sum()   # 986 of 1000 rows differ
```

Small or tie-free data hides this, which is why the original audit reported the ascending case as correct.

**`reverse=True`.**

```python
d = sc.dataframe(dict(x=[3,1,2], y=[10,20,30]))
orig = d.copy()
inds = d.sortrows(reverse=True, returninds=True)
print(inds, d.x.tolist())
print('reproduces?', d.equals(orig.iloc[inds].reset_index(drop=True)))
```

Actual: `[1 2 0] [3, 2, 1]` and `reproduces? False` — the returned indices are the *ascending* order while the frame was sorted descending. Expected `[0 2 1]`.

**List-valued `by`.** With a list of columns the return value is not even a permutation:

```python
d = sc.dataframe(dict(a=[1,1,0], b=[2,1,3]))
np.shape(d.sortrows(by=['a','b'], returninds=True))   # actual: (3, 2)   expected: (3,)
```

Actual output `array([[0, 1], [0, 1], [0, 1]])` — `np.argsort` was applied to a 2-D block along the wrong axis. `returninds` is untested in `tests/test_dataframe.py`.

**Fix**: derive the indices from the sort that is actually performed, then apply that permutation, so the two can never disagree. The simplest version: `pos = self.reset_index(drop=True).sort_values(by=by, ascending=ascending, kind=kwargs.pop('kind', 'mergesort'), **kwargs).index.values`, then reorder the frame with `self.iloc[pos]` and return `pos`. This handles ties, `reverse`, and list-valued `by` in one place.

### 22. `sort()` silently ignores `inplace`, always modifying the dataframe — `sc_dataframe.py:1183`

`sort()` accepts `inplace` and passes the hard-coded literal `inplace=True` to `sortrows()`, so `df.sort(inplace=False)` reorders the caller's dataframe anyway. `sortrows()` itself honours the argument, so the alias and the method it aliases disagree.

```python
d = sc.dataframe(dict(x=[3,1,2], y=[10,20,30]))
d.sort(inplace=False)
print(d.x.tolist())
```

Actual: `[1, 2, 3]` (the frame was sorted). Expected: `[3, 1, 2]` unchanged, matching `sortrows(inplace=False)`, which does leave it as `[3, 1, 2]`. `sort()` is the form used in the module's own class docstring (`sc_dataframe.py:57-58`), and `docs/quarto_utils.py:282` uses it; `tests/test_dataframe.py:132` only calls `df.sort()` with the default.

**Fix**: `return self.sortrows(by=by, reverse=reverse, returninds=returninds, inplace=inplace, **kwargs)`.

### 24. `sortcols()` discards the row index — `sc_dataframe.py:1203`

`sortcols()` reorders *columns*, but it calls `replacedata()` without a `reset_index` argument, so `replacedata()`'s default `reset_index=True` applies and the row labels are replaced by `0..n-1`. There is no `reset_index` argument on `sortcols()` to switch this off.

```python
d = sc.dataframe(dict(c=[1,2], a=[3,4]), index=[7,8])
d.sortcols(inplace=False).index.tolist()
```

Actual: `[0, 1]`. Expected: `[7, 8]` — a column reordering should not renumber rows. Every sibling method that resets the index (`concat()`, `poprows()`, `filtercols()`, `sortrows()`) both documents it and lets the caller disable it.

**Fix**: pass `reset_index=False` (and optionally expose a `reset_index` argument for symmetry with the other methods).

### 25. `read_csv_string()` only strips the outer string, so an indented CSV literal keeps the indentation in every value after the header — `sc_dataframe.py:1239`

`string.strip()` removes leading whitespace from the first line only. For the triple-quoted-literal idiom the docstring itself demonstrates, any real use inside a function or class body is indented, and every line except the header retains its leading spaces. Numeric columns coerce silently so it looks correct, but string columns silently acquire the indentation.

```python
df = sc.dataframe.read_csv_string('''
    name,val
    alice,1
    bob,2
    ''')
print([repr(x) for x in df['name']])
```

Actual: `["'    alice'", "'    bob'"]`. Expected: `["'alice'", "'bob'"]`. Comparisons, joins and `findinds()` against `'alice'` then all fail. Only the first column is affected (later fields have no leading whitespace), and the header row is clean, which makes it easy to miss. The docstring example is unindented, so the failure mode is invisible from the docs.

**Fix**: `string = textwrap.dedent(string).strip()` when `strip=True` (`textwrap.dedent` ignores all-whitespace lines, so the leading blank line does not defeat it), and/or default `skipinitialspace=True`.

### 38. `replacecol(col, None, new)` replaces every truthy value and leaves the `None` values alone — `sc_dataframe.py:999`

*Added on review (2026-09-25); missed by the original audit.*

`replacecol()` calls `sc.findinds(arr=coldata, val=old)`, and `sc.findinds(arr, val=None)` means "find the nonzero entries". So asking to replace `None` overwrites all of the real (truthy) data and leaves the `None` entries untouched, which is the opposite of what was asked.

```python
sc.dataframe(dict(x=pd.array([1, None, 'b', 0], dtype=object))).replacecol('x', None, 'z')
```

- **Actual:** `x = ['z', None, 'z', 0]`.
- **Expected:** `x = [1, 'z', 'b', 0]`.
- **Related silent no-op:** `replacecol('x', np.nan, 0)` (and `df.findinds(np.nan)`) match nothing, because `np.isclose(nan, nan)` is `False`. So there is no way to use `replacecol()` to fill missing values, and no error says so. (The original audit's "Verified clean" section described this no-op as acceptable; it is not, since the requested replacement silently does not happen.)

**Fix**: in `replacecol()`, when `old is None` or `old` is NaN, use `inds = np.flatnonzero(pd.isna(coldata))`; otherwise keep `sc.findinds()`.

### 39. A list of dict records is silently stored as dict cells when the number of records equals `ncols` — `sc_dataframe.py:644-646`

*Added on review (2026-09-25); missed by the original audit.*

This is the same shape-coincidence heuristic as finding 2, but for a different standard pandas input form: a list of record dicts. `np.array([dict(...), dict(...)])` has shape `(2,)`, so on a 2-column frame `_sanitize_df()` decides it is a single row and wraps it in another list.

```python
sc.dataframe(dict(a=[1], b=[2])).appendrow([dict(a=3,b=4), dict(a=5,b=6)], inplace=False)
```

- **Actual:** one new row whose cells are `{'a': 3, 'b': 4}` and `{'a': 5, 'b': 6}`, and both columns become `object` dtype.
- **Expected:** two rows, `(3, 4)` and `(5, 6)`.
- **Comparison:** the same call on a 3-column frame (with 2 records) works correctly, so the result depends only on whether the record count happens to equal `ncols`. `concat()` is affected identically.

No error or warning is raised.

**Fix**: only wrap `arg` in a list when its elements are scalars, e.g. `if len(arg) == self.ncols and not any(isinstance(v, (dict, list, tuple, np.ndarray)) for v in arg): arg = [arg]`. This can be merged with the fixes for findings 2, 14 and 40.

### 40. Appending or inserting a scalar row into a single-column frame crashes — `sc_dataframe.py:645-647`

*Added on review (2026-09-25); missed by the original audit.*

`insertrow()`'s validation deliberately skips the iterable check when there is only one column (`if self.ncols>1:`, line 611), so a scalar is meant to be accepted. `_sanitize_df()` then computes `np.array(3).shape == ()`, which does not equal `(1,)`, so the bare scalar is passed straight to the constructor.

```python
sc.dataframe(dict(a=[1,2])).appendrow(3)       # ValueError: DataFrame constructor not properly called!
sc.dataframe(dict(a=[1,2])).insertrow(1, 3)    # same
```

- **Actual:** `ValueError: DataFrame constructor not properly called!`
- **Expected:** `a = [1, 2, 3]` and `a = [1, 3, 2]` respectively.

**Fix**: in `_sanitize_df()`, add `if np.ndim(arg) == 0 and self.ncols == 1: arg = [[arg]]` before the shape probe. This can be merged with the fixes for findings 2, 14 and 39.

## Low severity

### 7. `get()` shadows `pd.DataFrame.get()` with an incompatible signature and behaviour — `sc_dataframe.py:248`

*Severity lowered from Medium to Low on review (2026-09-25).* The contract violation is real, but no library breakage was found: pandas internals never call `.get()`, and seaborn (which calls `data.get(k, None)`) works because it copies the data into its own frame first. The impact is limited to user code that uses the defensive `df.get(key, default)` idiom.

`get()` is described as an "alias to pandas `__getitem__`", but `get` is an existing, documented `pd.DataFrame` method with the signature `get(key, default=None)` that returns the default for a missing key. Overriding it means `sc.dataframe` violates the contract of its own base class: a missing key raises instead of returning `None`, and the `default` argument is gone.

```python
import pandas as pd
p = pd.DataFrame(dict(a=[1,2])); s = sc.dataframe(dict(a=[1,2]))
p.get('b')      # None
p.get('b', 5)   # 5
s.get('b')      # KeyError: 'b'
s.get('b', 5)   # TypeError
```

```
--- pd.get('b'): None
--- pd.get('b', 5): 5
--- sc.get('b'): EXC KeyError: 'b'
--- sc.get('b', 5): EXC TypeError: dataframe.get() takes 2 positional arguments but 3 were given
```

Any downstream code (or library) that receives an `sc.dataframe` and treats it as a `pd.DataFrame` — `cols = df.get('maybe_missing')` is a common defensive idiom — changes behaviour. `tests/test_dataframe.py:96` calls `df.get('a')` with the comment "Test pandas method", i.e. the test believes it is testing pandas' method. No internal Sciris code calls `.get()` on a dataframe (checked by grep over `sciris/*.py`), so the blast radius is user code. `sc.odict.get()` (`sc_odict.py:1394`) forwards `*args, **kwargs` and does not have this problem.

**Fix**: accept and honour `default` (`def get(self, key, default=None)`, returning `default` on `KeyError`), or rename this helper to something that does not collide (e.g. `pget`/`rawget`) and keep `pd.DataFrame.get()` intact.

### 23. `sortcols()` ignores `reverse` whenever `sortorder` is supplied — `sc_dataframe.py:1197`

*Severity lowered from Medium to Low on review (2026-09-25): the combination of an explicit `sortorder` with `reverse=True` is rare, and the caller can simply pass the reversed order.*

The `if reverse: sortorder = sortorder[::-1]` line sits inside the `if sortorder is None:` block, so a user-supplied order is never reversed and the documented argument does nothing.

```python
d = sc.dataframe(dict(c=[1,2], a=[3,4], b=[5,6]))
d.sortcols(sortorder=[2,1,0], reverse=True, inplace=False).cols
```

Actual: `['b', 'a', 'c']` — identical to `reverse=False`. Expected: `['c', 'a', 'b']`. `tests/test_dataframe.py:131` only calls `sortcols(reverse=True)` with `sortorder=None`.

**Fix**: move the `if reverse: sortorder = sortorder[::-1]` outside the `sortorder is None` branch.

### 35. `sortcols()` demotes a `sc.dataframe` subclass back to the base class — `sc_dataframe.py:1202`

The line hard-codes `dataframe({...})` instead of `self._constructor({...})`. `_constructor` exists precisely "to allow subclassing", and every sibling method (`concat()`, `poprows()`, `filtercols()`, `sortrows()`, `merge()`, `cat()`, `read_csv_string()`) preserves the subclass.

```python
class MyDF(sc.dataframe): pass
type(MyDF(dict(b=[1], a=[2])).sortcols(inplace=False)).__name__
```

Actual: `dataframe`. Expected: `MyDF`. (With `inplace=True` the object identity survives, so only the returned/`inplace=False` form is affected.)

**Fix**: `newdf = self._constructor({k:self[k] for k in newcols})`. (Confirmed on review as a one-token fix.)

### 41. `sc.dataframe.equal()` crashes on a categorical column that contains NaN — `sc_dataframe.py:406`

*Added on review (2026-09-25); missed by the original audit.*

With `equal_nan=True` (the default), `equal()` does `base.fillna(sc.sc_math._nan_fill)` to make NaNs comparable. For a categorical column, pandas refuses to insert the float sentinel because it is not one of the categories.

```python
ct = sc.dataframe(dict(c=pd.Categorical(['a', None])))
sc.dataframe.equal(ct, ct)   # TypeError: Cannot setitem on a Categorical with a new category (-528876923.87569493), set the categories first
ct.equals(ct, ct)            # same, because the multi-argument path routes to equal()
```

- **Actual:** `TypeError`.
- **Expected:** `True`. `ct.equals(ct)` already returns `True`, because the single-argument path uses pandas' own `equals()`.

**Fix**: compare NaN positions directly instead of filling them, e.g. `mask = base.isna().values; eq = (mask == other.isna().values).all() and (base.values[~mask] == other.values[~mask]).all()`. Alternatively, catch the `TypeError` and convert categorical columns to `object` before filling.

### 42. The class docstring's headline example calls `rmcol()`/`rmrow()`, which do not exist — `sc_dataframe.py:52`, `60`, `61`

*Added on review (2026-09-25); missed by the original audit, whose "Verified clean" section wrongly said this example had been executed verbatim.*

This is documentation, but it is the main usage example for the class.

```python
df.rmcol('z')    # AttributeError: 'dataframe' object has no attribute 'rmcol'
df.rmrow()       # AttributeError: 'dataframe' object has no attribute 'rmrow'
df.rmrow(555)    # same
```

These methods were removed long ago (they appear only in early commits such as `ec1fc4e`), and nothing else in the repository references them.

**Fix**: replace the three calls: `df.rmcol('z')` becomes `df.popcols('z')`; `df.rmrow()` becomes `df.poprow()`; `df.rmrow(555)` becomes `df.poprows(value=555)`.

## Misplaced `# pragma: no cover`

Every pragma below sits on a reachable branch; several are already exercised by the existing test suite, and three sit on paths with confirmed bugs from this audit (finding 2's crash variant, finding 4's oracle disagreement, and finding 20's list mutation). This table is a coverage note, not a bug list.

| Line | Branch | Reachable via | Observed |
|---|---|---|---|
| 79 | `__init__`, both `cols` and `columns` given | `sc.dataframe(cols=['a'], columns=['a'], data=[[1]])` | `ValueError: The argument "cols" is an alias for "columns", do not supply both` |
| 179 | `col_index()` numeric `IndexError` | `df.col_index(10)` | `IndexError: Column "10" is not a valid index; there are 3 columns` |
| 183 | `col_index()` unrecognized type | `df.col_index('q')` | `TypeError: Unrecognized column/column type "q" <class 'str'>` |
| 232 | `col_name()` numeric `IndexError` | `df.col_name(10)` | `IndexError: Column "10" is not a valid index` |
| 235 | `col_name()` unrecognized type | `df.col_name('q')` | `TypeError: Unrecognized column/column type "q" <class 'str'>` |
| 266 | `__getitem__()` string fallback | `df['not_a_column']` — already in `tests/test_dataframe.py:92` | `KeyNotFoundError: Key "not_a_column" is not a valid column; choices are: a, b` |
| 283 | `__getitem__()` unrecognized key | `df[sc.prettyobj(...)]` — already in `tests/test_dataframe.py:94` | `KeyNotFoundError: Unrecognized dataframe key of <class 'sciris.sc_printing.prettyobj'>` |
| 311 | `__setitem__()` double-failure handler | `df['a','b'] = 5` | `IndexError: Could not understand key ('a', 'b'): ...` |
| 344, 350 | `flexget()` `cols is None` / `rows is None` | `df.flexget(rows=[0])`, `df.flexget(cols='a')` | both return the expected sub-frame |
| 374 | `__eq__()` fallback to `equals()` | `df == sc.dataframe(dict(a=[1], b=[2]))` (shape mismatch) | returns `False` (a scalar bool, not a frame) |
| 397 | `equal()` fewer than 2 args | `sc.dataframe.equal(df)` | `ValueError: There must be >=2 input arguments, not 1` |
| 507 | `replacedata()` `newdf is None` | `df.replacedata(newdata=[[5],[6]])` | replaces the data correctly |
| 818-821 | `popcols()` not-found branch | `sc.dataframe(cols=['a','b'], data=[[1,2],[3,4]]).popcols('zz', die=False)` | prints the "could not remove" message and continues; also reached by a bare `popcols()` (former finding 32, rejected on review) |
| 854 | `findind()`, `value is None` | `sc.dataframe(dict(x=[1,2,3])).findind()` | returns `2` — the documented "default: return last row index" behaviour; `findrow()` with no `value` relies on it |
| 856 | `findind()`, `closest=True` | `sc.dataframe(data=[[2016,0.3],[2017,0.5]], columns=['year','val']).findind(2013, closest=True)` | returns `0` — one of the four docstring examples; `findrow(closest=True)` routes through it |
| 861-866 | `findind()` not-found branch | `df.findind(2013)` / `df.findind(2013, die=False)` | raises the documented `IndexError` / returns `None` — both docstring-advertised paths |
| 1127-1134 | `filtercols()` not-found / `keep=False` | `df.filtercols('a','c', keep=False)` (docstring example); `df.filtercols('zz','a', die=False)` for the not-found branch | returns `['b','d']`; not-found branch is the one with the list-mutation bug (finding 20) |

## Verified clean

**`__init__()`/`cols`/`set_dtypes()`**: the statements of the long class docstring example that exercise item access behave as advertised (`df['x']`, `df[0]`, `df['x',0]`, `df[0,:] = [123,6]`, `df['y'] = [8,5,0]`, adding column `z`). *Corrected on review:* the original text said every statement of the example was executed verbatim and worked; that is false, because `df.rmcol('z')`, `df.rmrow()` and `df.rmrow(555)` raise `AttributeError` (finding 42). Otherwise, the result stays an `sc.dataframe` (not a degraded `pd.DataFrame`) after every item-set. `nrows=` preallocation works for both a list and a dict `columns` (`sc.dataframe(columns=dict(a=int,b=float), nrows=2)` gives the right dtypes); `nrows=0` correctly does nothing; `dtypes` supplied via a dict and via the `columns`-as-dict form agree, and supplying both correctly raises. The "no overlap between column names and data keys" `RuntimeWarning` fires when it should; partial overlap (`sc.dataframe(dict(a=[1],b=[2]), columns=['a','c'])`) silently drops `b` and NaN-fills `c`, but that is plain pandas constructor behaviour, not something Sciris introduces. `set_dtypes()` returns `None` and does mutate in place, as documented.

**`col_index()`/`col_name()`**: all six docstring examples return the documented values. Duplicate column names resolve to the *first* occurrence for `col_index()` and are handled sensibly by `col_name()`/`flexget()` (no crash, no wrong column). Negative indices work (`col_name(-1) == 'c'`); `col_index(-1)` returns `-1` rather than `len(cols)-1`, which is consistent with its "index" contract and works correctly downstream in `flexget()`'s `np.array(self.cols)[colindices]`. Multiple positional args return a list in the right order; a single arg returns a bare scalar (not a 1-list) as documented. Integer *column labels* are resolved label-first in both methods, which is explicitly documented in `col_name()`'s Note.

**`__getitem__()`/`__setitem__()`**: `df['x']`, `df[0]`, `df['x',0]`, `df[0,'x']`, `df[0,0]`, `df[0,:]`, `df[:2]`, `df[[0,2]]`, `df[np.array([0,2])]` all return the documented axis on a string-columned frame, and the string/non-string swap in the tuple branch works in both orders. `isinstance(k, (int, str, Ellipsis))` at line 297 is malformed (`Ellipsis` is a value, not a type) and raises `TypeError` for e.g. `df[0,:] = [...]`, but the exception is caught by the enclosing `except` and the fallback path handles the key correctly, so there is no observable defect — only a misleading error message if the fallback also fails. Setting with a `np.int64` key, a slice row (`df[0:2,'x'] = 5`) and adding a brand-new column all work.

**`flexget()`**: the docstring example is correct; non-consecutive rows and columns work; `cols=None` and `rows=None` (the `Ellipsis` paths) both work; a single-element selection returns a scalar as documented; `asarray=True` returns a plain array and preserves object dtype for mixed columns; the returned object is an `sc.dataframe` via `_constructor`. The `self._constructor(data=output, columns=...)` reindex is a no-op for name-preserving selections (including duplicate names), so no silent column loss there.

**`equal()`/`equals()`/`__eq__()`**: all four `equal()` docstring examples return the documented values, as do the assertions in `tests/test_dataframe.py:157-163`. Differing column order and differing shapes are correctly reported unequal by all three APIs. `equal_nan=True` handles NaN in numeric and in string/object columns without raising (the `sc.sc_math._nan_fill` sentinel round-trips), but not in categorical columns, where it raises (finding 41, added on review), and a frame that literally contains `-528876923.87569493` compares equal to a NaN frame — true, but by design of the shared sentinel and not worth changing. `df == df` on a NaN-containing frame gives elementwise `False` at the NaN, which is standard pandas semantics and matches `pd.DataFrame`. Three or more arguments to `equal()` are handled correctly, including the short-circuit-free `all(eqs)`.

**`appendrow()`/`append()`/`insertrow()`/`concat()`/`cat()`/`merge()`**: both `appendrow()` docstring examples and both `insertrow()` docstring examples produce the documented frames. `insertrow()` positions are correct at every boundary tested — `index=0` inserts first, `index=len(df)` and `index>len(df)` append, `index=-1` inserts before the last row (matching `list.insert(-1)`) — and no rows are dropped or duplicated in any case. Dict rows whose keys are in a different order from the columns are aligned by name, not by position (`df.appendrow(dict(c=0.9, b=9, a='zzz'))` is correct). Tuples, lists, 1-D arrays and 2-D arrays of multiple rows all work for `appendrow()`; too-few and too-many values raise (an opaque pandas message, but no silent truncation or padding). Appending a float row to an int column upcasts the column to float rather than truncating the value, and appending to a string column preserves the `str` dtype. Both docstring examples for `concat()`/`cat()` run and give the documented shapes (`df2.concat(arr1)` and `sc.dataframe.cat(arr1, pd.DataFrame(...))` both return a `sc.dataframe` of shape `(10,3)`). `concat()` preserves the class (including subclasses via `_constructor`) and per-column dtypes (`int64` and `StringDtype` both survive). Mismatched columns between two dataframes are aligned by *name* by `pd.concat`, with `NaN` fill in the non-overlapping cells and no cross-column contamination. The `columns=` argument does reach the data: `df.concat([[10,20]], columns=['y','x'])` correctly lands `10` under `y` and `20` under `x`. Concatenating onto an empty (columns-only) frame works. `merge()` preserves the class and dtypes. The only `concat`/`cat` defect found is the dict-of-columns transpose reported above (finding 2).

**`inplace=True`/`inplace=False` agreement, checked across the file**: for `appendrow()`/`append()`, `concat()`/`cat()`, `merge()`, `poprows()`, and `filtercols()`/`filterin()`/`filterout()`, the `inplace=True` and `inplace=False` paths were each verified to produce `equals()`-identical output, `inplace=True` genuinely mutates `self` and returns `self` (keeping the `sc.dataframe` subclass and column dtypes), and `inplace=False` leaves the original object completely untouched — verified not just by value comparison but by mutating the returned object afterwards and re-checking the original untouched (pandas 3.0 Copy-on-Write; note `_is_copy` no longer exists in pandas 3.0, so the classic view-aliasing check does not apply). `appendrow(reset_index=True/False)` both behave as documented, and this also holds for `poprows(reset_index=False)` specifically — both the in-place and returned paths leave the index `[0, 3, 4]` unchanged, because both share `replacedata()`. `insertrow()`'s dict-key validation (extra/missing columns) produces the right error, and the `die=False` escape hatch does bypass validation.

**`disp()`**: `nrows`/`ncols`/`width`/`precision` are all honoured, the `display.` prefix is added only for keys that exist in `pd.options.display`, extra `options`/`kwargs` are merged in the documented precedence order, and `pd.option_context` restores the global pandas options afterwards (checked `pd.options.display.precision` before and after, including when the frame's `__repr__` raises).

**`ncols`/`nrows`**: correct for empty, single-row and normal frames, including after column reordering.

**`popcols()`**: correctly removes single columns, multiple positional columns (`popcols('a','c')`) and a list (`popcols(['a','c'])`), returns `self`, and leaves remaining column order and dtypes intact; `die=False` on a missing column prints and continues without disturbing the frame. Only the bare-`popcols()` `None` message is odd (former finding 32, rejected on review as not worth fixing).

**`findind()`**: positional (not label) index, which matches what `poprow()` and `filterin()` consume; agrees with the documented examples `findind(2016) -> 0`, `findind(0.5,'val') -> 1`, `findind(2013)` raising under the default `die=True`, `findind(2013, die=False) -> None`, and `findind(2013, closest=True) -> 0`. `closest=True` works on an unsorted numeric column (it is a pure `argmin(abs(diff))`, so it does not assume sortedness), and correctly finds duplicates' *first* occurrence. On a non-numeric column `closest=True` raises `TypeError: operation 'sub' not supported` (string column) or `unsupported operand type(s) for -` (datetime column) — an unhelpful error on out-of-contract input, so not reported. Exact string lookups (`findind('b')`) work. `value=None` returns `nrows-1` as documented.

**`_diffinds()`**: correct set inversion for a list, a scalar, negative indices (`-1`, `[-1,-2]`), duplicated indices (deduplicated by `setdiff1d`), a boolean mask, an empty list, and `None`; always positional, so it stays correct on a non-default index.

**`poprows()`**: both docstring examples give the documented result. Negative indices, a scalar index, an empty list, a `value=` that matches zero rows (no-op), and a non-default index (positional, as intended) all behave correctly, and `int64`/`float64`/`StringDtype` columns keep their dtypes through the rebuild. `poprows()` on a duplicated index is *safe* (unlike `poprow()`), because it is positional.

**`enumrows()`**: all four docstring examples produce the documented output. Per-column dtypes are preserved exactly, including an `int64` above 2^53 (`9007199254740993` round-trips as a Python `int`, unlike `to_odict()`), a float and a string in the same row. `cols=` restricts and reorders correctly, `type=tuple`/`list`/`dict`/`objdict` all work, `type='bogus'` gives the intended `ValueError`, and an empty frame yields nothing. Note that the yielded index is the enumeration position, not the pandas index label (so it differs from `iterrows()` after a `reset_index=False` operation) — this looks intentional given the name, so it is not reported.

**`replacecol()`**: replaces all matching occurrences, returns `self`, honours `col` as a name or an index, defaults to the first column, and works on a string column. *Corrected on review:* the original text said that leaving `NaN` alone when asked to replace `NaN` was "a no-op rather than a wrong answer"; the no-op is itself the wrong answer, and `old=None` is worse (finding 38). The defects are the shared `sc.findinds()` tolerance (finding 4) and the `None`/NaN handling (finding 38). (The dtype-widening `TypeError`, former finding 34, was rejected on review as standard pandas 3 strict setitem.)

**`to_odict()`**: the no-argument form, a list of rows, and a slice all produce the right per-column arrays and preserve column order and names; only the `int` form (finding 17) and the dtype upcast (finding 18) are defective.

**`findrow()`**: `findrow(2016)`, `findrow(2013) -> None` (default `die=False`), `findrow(2013, closest=True)`, and `default=` (which correctly overrides `die`) all behave as documented; `die=True` with `default=None` raises. The `asdict=True` crash (finding 19) is the only defect; the single common dtype of the returned row is inherent to returning a 1-D array, and only the docstring's `dtype=object` claim is wrong (see finding 18).

**`findinds()`**: returns positional indices consistent with `poprow()`/`filterin()`; finds all duplicates (`findinds(0.3,'val') -> array([0,2])`, matching the docstring); works on a string column; `col` accepts a name or an index; forwards `**kwargs` to `sc.findinds()`. Its defects are the inherited fuzzy matching (finding 4) and the mishandling of a list-valued `value` (finding 37).

**`_filterrows()`/`filterin()`/`filterout()`**: the double set-inversion (`_diffinds()` in `_filterrows`, then again in `poprows`) is correct, not a bug: `filterin(value=2, col='y')` keeps exactly the matching rows and `filterout(value=2, col='y')` keeps exactly the rest, and the two are complementary. `inds=` and `value=` forms both work, including a boolean mask for `inds`. A `value=` matching zero rows gives an empty frame for `filterin` (correct) and a no-op for `filterout` (correct). On a non-default index the selectors stay positional, so `filterin(inds=[0,1])` picks the first two rows regardless of labels. The `verbose=True` message reports the correct *count of removed rows* and the correct removed indices for both `filterin` and `filterout` (checked specifically, since the `keep` inversion happens before the print).

**`filtercols()`**: keeps and *reorders* to the requested column order (`filtercols('d','a')` returns `['d','a']` with the data travelling correctly), `keep=False` removes the named columns and keeps the rest in original order (the docstring example gives `['b','d']`), the default `inplace=False` leaves the original alone, and dtypes (`int64`, `float64`, `StringDtype`) survive both paths. A single missing column under `die=False` behaves sanely; only the two-or-more-missing case is broken (finding 20).

**`sortrows()`/`sort()`**: the data travels with the sort — other columns, including string columns, stay on their original rows — for ascending, `reverse=True`, single-column and multi-column `by`, `by` given as an integer index, and the deprecated `col=` alias. `inplace=True` returns `self`; `inplace=False` returns a new frame and leaves `self` untouched (for `sortrows`, not `sort`); `reset_index=True`/`False` both work; `kind='mergesort'` stability is respected via `sort_values`. *Corrected on review:* the original text said ascending `returninds` reproduces the sort. That holds only for small or tie-free data; with ties it does not, because `returninds` uses a stable mergesort while `sort_values()` defaults to quicksort (finding 21). Sorting an empty frame is a no-op rather than an error.

**`sortcols()`**: alphabetical (default), `reverse=True`, and explicit `sortorder` all move the *data* with the columns (verified with distinct per-column values and a string column), dtypes are preserved per column (`int64` stays `int64`, `float64` stays `float64` — the v3.0.0 changelog claim holds), `inplace=False` leaves the original column order intact, and an empty frame is handled. The defects are the ignored `reverse` with an explicit `sortorder` (finding 23), the index reset (finding 24), and the subclass demotion (finding 35).

**`to_pandas()`**: returns a genuine `pd.DataFrame` (not the subclass), with values and dtypes intact; its `**kwargs` are silently ignored (former finding 36, rejected on review as not worth fixing).

**`read_csv()`/`read_excel()`**: `read_csv()` returns the calling class. `read_excel()` returns the calling class for a single sheet, and an `sc.objdict` of dataframes for `sheet_name=None` and for a list of sheet names, with each value converted; the keys match the sheet names.

**`read_csv_string()`**: handles a trailing newline, CRLF line endings, quoted fields containing commas (`"x,y"` -> a single value `x,y`), quoted fields containing embedded newlines, a header-only string (a 0-row frame with the right columns), `strip=False`, and forwarding of `**kwargs` to `pd.read_csv`; it returns the calling class, including for subclasses. An empty or whitespace-only string raises pandas' `EmptyDataError`, which is the same behaviour as `pd.read_csv` on an empty file, so it is not reported. Only the indentation handling is defective (finding 25).

**`_constructor`**: returning `self.__class__` correctly propagates subclasses through `concat`, `poprows`, `filtercols`, `sortrows`, `cat`, and `read_csv_string`; the only method that bypasses it is `sortcols()` (finding 35).

## Rejected on review

The following original findings were removed from the severity sections and the summary table after the 2026-09-25 re-verification. Their repros still reproduce, but the behaviour is either correct or not worth changing.

- **9.** `__getitem__()` integer keys return columns on an integer-labelled frame — NOT A BUG: label-first lookup is standard pandas behaviour, and the audit itself said no behaviour change is safe, so this is at most a documentation note.
- **10.** `df[0] = value` adds a column instead of setting row 0 — NOT A BUG: this is standard pandas `__setitem__` behaviour, required for `df['new'] = v`, and the class docstring uses the tuple form `df[0,:] = ...` for row assignment.
- **11.** `equals(equal_nan=False)` less strict than `equals()`; `equal()` type check asymmetric — NOT WORTH FIXING: `equal()` is documented to check only "type, size, columns, and values" (not index or dtypes), the fallback is by design, and the asymmetry is only a subclass `isinstance` quirk.
- **26.** `dtypes` list silently truncated by `zip()` — NOT WORTH FIXING: a `dtypes` list of the wrong length is caller error.
- **27.** `col_index()`/`col_name()` ignore `die=False` for out-of-range integers — NOT WORTH FIXING: a rare corner combining `die=False` with an out-of-range integer; a one-line fix if ever wanted.
- **28.** `__getitem__(key, die=False)` raises `UnboundLocalError` — NOT WORTH FIXING: only reachable by calling `__getitem__(set, die=False)` explicitly.
- **29.** `replacedata(newdf=..., inplace=False)` mutates the caller's frame — NOT WORTH FIXING: an internal helper whose internal callers all pass fresh frames, and `newdf` is meant as the replacement.
- **30.** `merge()` docstring says "in place" but `inplace=False` — NOT A BUG: docstring wording only.
- **31.** `addcol()` returns `None` in in-place mode — NOT WORTH FIXING: returning `None` from in-place mode is pandas convention, so changing it is an API choice, not a bug fix.
- **32.** `popcols()` with no arguments reports a missing column `None` — NOT WORTH FIXING: raising on a no-argument call is reasonable; only the message is odd.
- **33.** `enumrows()` raises `KeyError` instead of `ValueError` for an unsupported callable — NOT WORTH FIXING: only the exception type differs on invalid input.
- **34.** `replacecol()` raises `TypeError` when the new value does not fit the dtype — NOT WORTH FIXING: this is pandas 3's strict setitem, the common case (replacing with NaN) upcasts correctly, and `df.replace` covers the rest.
- **36.** `to_pandas()` silently ignores `**kwargs` — NOT WORTH FIXING: dead `**kwargs` with no wrong result.

## Suggested order of work

1. **Finding 2** (dict-of-columns transpose in `_sanitize_df()`) and **finding 4** (`findinds()`'s fuzzy matching, to be fixed in `sc.findinds()` per `sc_math_bugfixes.md` #12) — both cause silent data corruption/loss on ordinary calls with no error raised (finding 2 scrambles values across columns; finding 4 deletes, keeps, or overwrites the wrong rows), both are rooted in one function each, and both are cheap, local fixes. Fix findings 14, 39 and 40 in the same pass as finding 2, since all four live in the same shape probe, and fix finding 37 alongside finding 4.
2. **Finding 3** (`poprow()` deletes every row sharing a label) and **finding 1** (`flexget()` int-row silently returns an empty frame) — also silent data loss (rows vanish with no exception), reachable as soon as an index has duplicates (a common downstream state after `concat()`/`filterin()`) or a caller uses the documented int form of `rows`.
3. **Findings 5, 15** (`dataframe()`/`addcol()` mutate the caller's dict) — silent corruption of data the caller still holds a reference to, not just of the returned frame; small, mechanical fixes.
4. **Finding 38** (`replacecol()` with `None`/NaN) — silent wrong overwrite on an in-contract call.
5. **Findings 6, 8, 12, 13, 16-22, 24, 25** — remaining medium findings: wrong answers or crashes on documented, in-contract usage (raises rather than corrupts, in most cases), including `insertrow()`'s validation and `reset_index` bugs, `sortrows()`/`sort()`/`sortcols()` ignoring their own flags or returning mismatched indices, and `to_odict()`'s dtype upcasting.
6. **Findings 7, 23, 35, 41, 42 and the pragma list** — `get()`'s contract, minor flag handling, subclass preservation, the categorical-NaN crash in `equal()`, and the broken class docstring example.

Findings 1-4 and 37 are the ones that would not announce themselves in downstream code: no exception is raised, and the caller gets a plausible-looking dataframe that is either transposed, missing rows it should have, or missing rows it shouldn't be missing. Each has a natural regression test (round-trip a dict of columns; assert row counts before/after a value-based filter on close-but-distinct data; assert row counts after popping from a duplicated index; assert non-zero size after an int-row `flexget()`), which would be worth adding alongside the fixes.
