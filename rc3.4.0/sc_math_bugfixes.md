# `sc_math.py` bug audit

Audit of `sciris/sc_math.py` for genuine defects: wrong numerical results, documented arguments that don't work, silent data corruption, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps, and performance.

**Scope**: every function in the module except `safedivide()`, which was reviewed and fixed separately (see the `- *New in version 3.3.1:*` note in its docstring). **Method**: line-by-line reading of each function, followed by executed hypothesis tests. Every "actual" value below was produced by running the code against the editable install (Sciris 3.3.0, numpy 2.4.6). The original audit (commit `2d69aad`) recorded 32 findings, each reproduced twice.

**Re-verification**: this document was independently re-verified on 2026-09-25 against commit `d91898a`. Every repro was re-executed and all 32 original "actual" values reproduced. On review, 13 findings were rejected as not a bug or not worth fixing (see [Rejected on review](#rejected-on-review)); 4 were rewritten because the description or proposed fix was inaccurate (3, 4, 8, 17); several proposed fixes were corrected because they would have broken working behavior (3, 4, 7, 8, 21); severities were revised; 4 missed bugs were added (33-36); and all line numbers were corrected against the current source (the original numbers were about 2 lines too high). Original finding numbers are kept stable for cross-referencing, so the numbering has gaps. The document now contains **23 findings: 8 High, 9 Medium, 6 Low**.

**Nothing in this document has been applied.** All fixes are described, not made.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `sanitize()`, `rolling()`, `fillnans()` | `replacenans=False` fills NaNs with `0` instead of removing them | 386 |
| 2 | High | `count()` | Returns `arr.ndim`, not the number of matches, for any multidimensional input | 272 |
| 3 | High | `inclusiverange()` | Silently drops the endpoint for ordinary decimal steps (e.g. `0` to `1.2` by `0.2`) | 797 |
| 4 | High | `convolve()` | Edge correction is entirely wrong when the kernel is longer than the data | 1080, 1095 |
| 5 | High | `convolve()`, `smooth()`, `gauss1d()`, `gauss2d()` | Integer input silently truncates the smoothed output (four separate sites) | 1087, 1146-1147, 1394, 1523 |
| 6 | High | `smooth()` | 2-D arrays with fewer columns than the kernel length are smoothed with an over-long kernel, destroying the values | 1123 |
| 7 | High | `sem()` | Documented `**kwargs` (e.g. `ddof`) are silently ignored | 932 |
| 33 | High | `findnearest()` | Any NaN in the series makes it return the index of the first NaN | 243 |
| 8 | Medium | `findnearest()` | Unsigned-integer input underflows, giving the wrong index | 243 |
| 12 | Medium | `approx()`, `findinds()`, `count()` | `eps` cannot tighten the match: numpy's `rtol=1e-5` remains active, so `eps=0` still matches far-off values | 44, 175 |
| 13 | Medium | `sanitize()`, `rolling()` | 2-D data with `replacenans=0` raises "NaNs cannot be removed" | 378 |
| 17 | Medium | `sanitize()` (via `rolling()`, `fillnans()`) | Interpolating NaNs crashes when there are no valid values, e.g. `rolling()` with `window` longer than the data | 395, 1042 |
| 19 | Medium | `gauss1d()`, `gauss2d()` | float32 exponential underflows ~2.7x sooner than float64, returning NaN where `use32=False` works | 1383, 1511 |
| 21 | Medium | `smoothinterp()` | `growth` misanchors or crashes when all of `newx` lies outside the data range | 1279, 1282 |
| 34 | Medium | `nanequal()`, `equal()` | NaNs compare unequal between float32 and float64 arrays because the NaN sentinel rounds differently | 455, 491, 505 |
| 35 | Medium | `gauss1d()` | `use32=False` crashes on the documented list input | 1366-1372 |
| 36 | Medium | `smooth()` | Default call on a 1-D array of length 1 or 2 returns wrong values | 1123-1128 |
| 11 | Low | `normalize()` | Constant or single-element input returns all-NaN, outside the documented range | 732 |
| 14 | Low | `nanequal()` | `equal_nan=False` crashes unless the first argument is already an ndarray | 490, 496 |
| 20 | Low | `getvalidinds()` | Boolean `filterdata` produces meaningless indices (deprecated, but still exported) | 306 |
| 24 | Low | `similarity()` | Error message is missing its `f` prefix, so it prints the literal `{method}` | 963 |
| 27 | Low | `smoothinterp()` | Docstring claims the function passes exactly through every data point; it is designed not to | 1160 |
| 30 | Low | `gauss2d()` | The length-validation guard uses a chained comparison and never fires | 1501 |

## Three recurring patterns

Many of the high-severity findings are instances of three repeated mistakes, which is worth noting because fixing the pattern is cheaper and safer than fixing each site one at a time.

**Falsy sentinels tested with truthiness.** `sanitize()` uses `if replacenans is not None:` to mean "the user gave a replacement value", so `replacenans=False` (documented as "remove the NaNs") is taken as the value `0` — findings 1 and 13 are the same line of reasoning applied in opposite directions, and finding 1 silently fabricates data. The same class appears in `gauss2d()`'s chained-comparison guard (30). Any sentinel that can legitimately be `False` or `0` needs an `is` test.

**Integer dtype round-trip.** `convolve()`, `smooth()`, `gauss1d()`, and `gauss2d()` each capture the input dtype (or write float results back into an integer array) and so truncate a weighted average toward zero — a systematic ~0.5 bias, verified at `-0.5036` over 200 points of integer data. Counts, cases, and populations are exactly the integer-valued data this library is used on, so this is the single highest-value fix in the file. A shared helper (promote to float on entry, only restore a floating dtype on exit) would close all four, and would also fix finding 35.

**Reducing over a polymorphic return.** `count()` is `len(findinds(...))`, but `findinds()` documents that it returns a *tuple of index arrays* for multidimensional input, so `len()` measures dimensions. Any wrapper of `findinds()` needs to handle both return shapes.

## High severity

### 1. `sanitize()` fills NaNs with zero when asked to remove them — `sc_math.py:386`

`if replacenans is not None:` routes `replacenans=False` into the "replace with this value" branch, and `sanitized[naninds] = False` writes `0.0`. The docstring says the opposite: "If `replacenans=False`, the sanitized array may be shorter than data". `rolling()` documents the same flag as "if False, remove them", and `fillnans()` is affected identically.

```python
sc.sanitize([3., np.nan, 5.], replacenans=False)          # actual: [3. 0. 5.]   expected: [3. 5.]
sc.rolling([1.,2,3,4,5], window=2, replacenans=False)     # actual: [0. 1.5 2.5 3.5 4.5]   expected: [1.5 2.5 3.5 4.5]
```

The injected zeros are indistinguishable from real data, and in the `rolling()` case they land exactly on the window burn-in, i.e. at the start of every smoothed series. `tests/test_math.py:233` calls `sc.rolling(d, replacenans=False)` but asserts nothing about the values.

**Fix**: `if replacenans is not None and replacenans is not False:`, or normalize `False` to `None` at the top of `sanitize()`.

### 2. `count()` returns the number of dimensions for multidimensional input — `sc_math.py:272`

```python
A = np.array([[1,2,3],[3,1,3]])
sc.count(A, 3)                  # actual: 2      expected: 3
sc.count(A > 0)                 # actual: 2      expected: 6
sc.count(np.ones((2,3,4)), 1)   # actual: 3      expected: 24
```

The answer is always exactly `ndim`, so a 2-D boolean mask always reports `2` regardless of content — a plausible-looking small integer that a caller will not question. The docstring advertises "Count the number of matching elements... Similar to `numpy.count_nonzero()`"; only the 1-D case is tested (`tests/test_math.py:115`).

**Fix**: count from the boolean mask (`np.count_nonzero`) rather than `len()` of the index result.

### 3. `inclusiverange()` drops the endpoint for ordinary decimal steps — `sc_math.py:797`

`nsteps.is_integer()` is an exact float test with no tolerance, so when `(stop-start)/step` lands just below an integer the function takes the non-integer branch, truncates `stop` down by a whole step, and returns a range missing the endpoint the function's name guarantees.

```python
sc.inclusiverange(0, 0.3, 0.1)   # actual: [0. 0.1 0.2]                    expected: [0. 0.1 0.2 0.3]
sc.inclusiverange(0, 1.2, 0.2)   # actual: [0. 0.2 0.4 0.6 0.8 1. ]        expected: [... 1.2]
sc.inclusiverange(3, 5, 0.2)     # works: ends at 5.0
```

A systematic scan of exactly-divisible `(0, stop, step)` triples found 20 failures, including `0.3/0.1`, `0.6/0.1`, `0.7/0.1`, `1.2/0.1`, `0.6/0.2`, `1.4/0.2`, `2.4/0.2`, `0.15/0.05`, `0.35/0.05`. A monthly grid `sc.inclusiverange(0, 1.2, 0.1)` silently loses December. Whether a given call breaks depends on invisible float representation, which makes this hard to spot in downstream code.

**Fix**: use a tight relative tolerance, and use the result in place of `.is_integer()`:

```python
r = round(nsteps)
is_int = abs(nsteps - r) <= 1e-9*max(1, abs(nsteps))
int_steps = r if is_int else int(nsteps)
```

Verified: this gives endpoints `1.2` and `0.3` for the cases above, `[0, 3, 6, 9]` for `(0, 10, 3)`, and ends at `100000.0` for `(0, 100000.5, 1)`. The originally proposed `np.isclose(nsteps, round(nsteps))` is unsafe: numpy's default `rtol=1e-5` exceeds 1 for large step counts, so `(0, 100000.5, 1)` would be classified as an integer count, keep `stop=100000.5`, and silently stretch the step to `1.000005`.

### 4. `convolve()` is unnormalized when the kernel is longer than the data — `sc_math.py:1080`, `1095`

When `len(v) > len(a)`, the LHS/RHS re-weighting uses `minlen = min(len_a, len_v) = len(a)` (line 1080), correcting only 1-2 points with the wrong weights, and the `len_diff` trim (line 1095) then takes the middle out of an *uncorrected* `mode='same'` result. Constant input is not preserved, which is the entire purpose of the function's edge correction. This is the case the changelog line at 1067 claims to have fixed in v1.3.1, and the branch is marked `# pragma: no cover`.

```python
sc.convolve(np.ones(3), np.ones(5)/5)   # actual: [0.6 0.6 0.6]      expected: [1. 1. 1.]
sc.convolve(np.ones(2), np.ones(5)/5)   # actual: [0.4 0.4]          expected: [1. 1.]
sc.convolve(np.ones(4), np.ones(9)/9)   # actual: [0.444 ... 0.444]  expected: [1. 1. 1. 1.]
```

Errors reach ~2.9x. The reach is wider than unusual kernels: the default `sc.smooth()` call hits this on tall 2-D arrays (finding 6) and on 1-D arrays of length 1 or 2 (finding 36).

**Fix**: replace the hand-rolled cumsum re-weighting with divide-by-support normalization, rescaled by the kernel total, and keep the existing trim:

```python
num = np.convolve(a, v, 'same')
den = np.convolve(np.ones(len(a)), v, 'same')
out = num/den*v.sum()
if len_diff: out = out[lhs_trim:lhs_trim+len_a]
```

Verified: this agrees with the current `sc.convolve()` to 8.9e-16 for random (unnormalized) kernels over all `len(v)` 1-9 and `len(a)` from `len(v)` to `len(v)+5`, and gives `[1,1,1]`, `[1,1]`, `[1,1,1,1]`, and `[3.]` (for `sc.convolve([3.], [.25,.5,.25])`) in the long-kernel cases. The originally proposed plain divide-by-support was wrong in two ways: it dropped the `v.sum()` scaling, so unnormalized kernels would differ from today's (correct) output by up to 3.2; and `np.convolve(..., 'same')` returns `max(len(a), len(v))` points, so the trim is still required. Zero-sum kernels (e.g. difference kernels) divide by zero, as they do today.

### 5. Integer input silently truncates the smoothed output — `sc_math.py:1087`, `1146-1147`, `1394`, `1523`

Four independent sites, one mistake: float results are written back into an integer array, or the input dtype is restored on exit.

```python
sc.convolve([1,2,3,4,5], [1,1,1])        # actual: [ 4  6  9 12 13]   expected: [4.5 6. 9. 12. 13.5]
sc.smooth(np.arange(25).reshape(5,5))[0] # actual: [ 1  2  3  4  4]   expected: [2. 2.667 3.667 4.667 5.333]
sc.gauss1d(np.arange(5.), np.array([0,10,20,30,40]))  # actual: [15 17 20 22 24]   expected: [15.53 17.73 20. 22.27 24.47]
sc.gauss2d(np.array([0.,1,2]), np.array([0.,1,2]), np.array([0,10,21]))  # actual: [ 1 10 19]
```

`convolve()` truncates because `out[:len_lhs] = out[:len_lhs]/w_lhs` assigns a float division into an int array — but only when *both* the data and the kernel are integer, since `np.convolve` of int data with a float kernel is already float (`sc.convolve([1,2,3,4,5], [0.25,0.5,0.25])` is correct). The sites reachable with default arguments are `smooth()`'s 2-D path, which truncates *twice* (once in the row pass, once in the column pass, compounding — element `[0,0]` is 50% low; its 1-D path escapes because it rebinds `output`), and `gauss1d`/`gauss2d`, which restore `orig_dtype` at the end. Passing the identical data as a list returns floats, so the behavior is also inconsistent between equivalent calls. Measured bias over 200 integer points: `-0.5036` for positive data, `+0.4952` for negative — i.e. truncation toward zero, not rounding.

**Fix**: build the working array as float (`np.array(data, dtype=float)`), and restore `orig_dtype` only when it is a floating dtype.

### 6. `smooth()` destroys 2-D arrays with few columns — `sc_math.py:1123`

`repeats` defaults from `len(data)`, the number of *rows*, and the resulting kernel is applied along both axes. Whenever the kernel is longer than a row, the column-wise pass hits finding 4. That happens both for tall arrays (long kernel) and for any array with fewer columns than the default length-3 kernel.

```python
sc.smooth(np.ones((100,3)))[0]      # actual: [0.348 0.364 0.348]   expected: [1. 1. 1.]
sc.smooth(np.ones((5,2)))[0]        # actual: [0.333 0.75]          expected: [1. 1.]
sc.smooth(np.ones((3,100)))[0][:5]  # control, wide array: [1. 1. 1. 1. 1.]
```

A 100-timestep by 3-variable array is an entirely ordinary input, and the result is wrong by ~3x with no warning.

**Fix**: fixing finding 4 (with the corrected fix) restores correct values. Separately, derive `repeats` per axis from that axis's length, so the amount of smoothing along each axis is proportionate.

### 7. `sem()` silently ignores its documented keyword arguments — `sc_math.py:932`

The docstring says "`kwargs (dict): passed to `numpy.std`", and the signature is `sem(a, axis=None, *args, **kwargs)`, but the body calls `a.std(axis=axis)` with no forwarding. Everything the caller passes is swallowed without error.

```python
d = np.arange(10.)
sc.sem(d, ddof=1)          # actual: 0.9082951062292475   expected: 0.9574271077563381
d.std(ddof=1)/np.sqrt(10)  #                              ground truth: 0.9574271077563381
sc.sem(d, None, 999, 888)  # extra positional args also silently swallowed
```

A caller who explicitly requests the sample-std convention gets the population-std answer and no indication that the request was dropped.

**Fix**: forward the arguments positionally: `a.std(axis, *args, **kwargs)` (`n` does not depend on `ddof`, so the rest of the function is fine). The originally proposed `a.std(axis=axis, *args, **kwargs)` raises "got multiple values for argument 'axis'" whenever `args` is non-empty. Alternatively, drop `*args, **kwargs` from the signature and the kwargs claim from the docstring so unsupported arguments raise.

### 33. `findnearest()` returns the index of a NaN when the series contains NaNs — `sc_math.py:243`

*Added on re-verification.* `np.argmin` propagates NaN, so any NaN in `series` makes `abs(series-value)` NaN at that position, and `argmin` returns the index of the first NaN regardless of `value`.

```python
sc.findnearest([1, np.nan, 3], 3)                  # actual: 1   expected: 2
sc.findnearest(np.array([1., 2, np.nan, 10]), 9)   # actual: 2   expected: 3
```

This is a silently wrong index on ordinary data with missing values, and indexing the series with it returns NaN.

**Fix**: use `np.nanargmin(np.abs(series - value))`. It still raises if the series is entirely NaN, which is reasonable.

## Medium severity

### 8. `findnearest()` returns the wrong index for unsigned integers — `sc_math.py:243`

`np.argmin(abs(series-value))` is evaluated in the array's own dtype, so for any unsigned dtype `series - value` underflows wherever `series < value`, making those candidates look maximally distant.

```python
u = np.array([100, 200], dtype=np.uint32)
sc.findnearest(u, 250)     # actual: 0   expected: 1
abs(u - 250)               # [4294967146 4294967246]
sc.findnearest(np.array([0,5,10,15], dtype=np.uint64), 12)  # actual: 3   expected: 2
```

Signed and float dtypes are unaffected. A negative `value` against a `uint64` series raises `OverflowError` instead of returning index 0. Severity revised from High to Medium on review: unsigned arrays do occur (numpy and pandas produce them for some counts and IDs), but they are not common in Sciris/Starsim workflows (Starsim UIDs are signed).

**Fix**: promote only unsigned dtypes, e.g. `if series.dtype.kind == 'u': series = series.astype(np.int64)` (or `float` if values can exceed the int64 range). The originally proposed `np.asarray(series, dtype=float) - value` is wrong: it raises `UFuncTypeError` for `datetime64` series, which currently work (`sc.findnearest(datetime64_array, np.datetime64('2020-02-03'))` returns 1), and it loses precision for int64/uint64 values above 2**53.

### 12. `eps` cannot tighten a match — `sc_math.py:44`, `sc_math.py:175`

`approx()`, `findinds()`, and `count()` map `eps` onto `np.isclose(atol=...)` but leave `rtol` at numpy's default `1e-5`, so the effective tolerance is `eps + 1e-5*|val|` and `eps=0` does not force exact matching.

```python
sc.findinds([100000, 100001, 100002], 100000, eps=0) # actual: [0 1]   expected: [0]
sc.approx(1e6, 1000010, eps=0)                       # actual: True    expected: False
sc.findinds(np.array([1e6, 1000010.]), 1e6, eps=0)   # actual: [0 1]   expected: [0]
sc.count(np.array([1e6, 1000010.]), 1e6)             # actual: 2       expected: 1
```

The first case is the most damaging: plain integer values such as IDs or counts at magnitude ~1e5 falsely match their neighbours, and `eps` — which the docstrings present as the matching precision — cannot prevent it. At values of order 1e6 the tolerance is ~10, seven orders of magnitude looser than the documented 1e-6 default. The docstrings do say `eps` is "equivalent to `numpy.isclose`'s atol", so the relative term is documented only by reference; the workaround is to pass `rtol=0` through `**kwargs` (verified: `sc.findinds([100000, 100001, 100002], 100000, eps=0, rtol=0)` returns `[0]`). This is also the root cause of `sc_dataframe_bugfixes.md` finding 4, where `df.findinds()` makes `poprows()`/`filterin()`/`filterout()`/`replacecol()` operate on the wrong rows. Related: passing both `eps` and `atol` to `approx()` silently discards the explicit `atol`.

The independent review marked this "not a bug" on the grounds that it matches the documented `atol` alias; that verdict was overruled because the false matches on integer data are silent, wrong results on ordinary input. Note that the fix does change which values match for existing callers relying on the relative tolerance, which should be recorded in the changelog.

**Fix**: pass `rtol=0` by default in all three functions (`kwargs.setdefault('rtol', 0)`), so users can still override it via `**kwargs`, and state the absolute-only default in both docstrings.

### 13. 2-D data with `replacenans=0` raises "NaNs cannot be removed" — `sc_math.py:378`

The multidimensional guard is `if not replacenans:`, a truthiness test, so the legitimate replacement value `0` is treated as "remove them" and rejected. `replacenans=9` works fine, and `replacenans=0` is the docstring's own fifth example.

```python
m = np.array([[1.0, np.nan], [3.0, 4.0]])
sc.sanitize(m, replacenans=9)   # [[1. 9.] [3. 4.]]
sc.sanitize(m, replacenans=0)   # ValueError: For multidimensional data, NaNs cannot be removed...
sc.rolling(np.arange(20.).reshape(4,5), window=2, replacenans=0)  # same ValueError
```

Zero is the single most likely fill value, and the error message tells the user to do exactly what they just did.

**Fix**: `if replacenans is None or replacenans is False:` in the multidim branch.

### 17. `sanitize()` crashes when interpolating an all-NaN array — `sc_math.py:395` (reached via `rolling()` at `1042`)

When the input has no valid values, `sanitize()`'s interpolation path passes zero valid points to `smoothinterp()`, which then indexes an empty array.

```python
sc.sanitize([np.nan, np.nan], replacenans='nearest')  # ValueError: attempt to get argmin of an empty sequence
sc.sanitize([np.nan, np.nan], replacenans='linear')   # ValueError: array of sample points is empty
sc.rolling([1,2,3], window=7, replacenans='nearest')  # same ValueError
```

`rolling()` is the most common way to reach it: with `window > len(data)` every rolled value is NaN. The `rolling()` docstring's own example uses `replacenans='nearest'` with the default `window=7`, so any series shorter than 7 points crashes. `fillnans()` is affected identically. (The original audit attributed the crash to `rolling()` and also listed `operation='none'` combined with `replacenans` as a bug; on review that combination is meaningless usage, since `'none'` returns a pandas `Rolling` object, and it is not a defect.)

**Fix**: in `sanitize()`, if there are no valid indices on the interpolation path, skip interpolation and return the data unchanged (all NaN) or `defaultval`. This also fixes `rolling()` and `fillnans()`.

### 19. float32 exponential underflow — `sc_math.py:1383`, `sc_math.py:1511`

`np.exp(-dist**2)` flushes to zero at `dist >~ 10` in float32 versus `>~ 27` in float64, so when every weight underflows, `weights/np.sum(weights)` is 0/0. Extrapolating a modest distance beyond the data, or using a small `scale` on a fine grid, therefore returns NaN with the default arguments and the correct value with `use32=False`.

```python
x = np.linspace(0, 1, 40); y = x**2
sc.gauss1d(x, y, np.array([2.5]), scale=0.1)               # actual: nan   expected: ~1.0
sc.gauss1d(x, y, np.array([2.5]), scale=0.1, use32=False)  # 0.99998

rng = np.random.default_rng
x, y, z = rng(0).random(30), rng(1).random(30), rng(2).random(30); xi = np.linspace(0, 1, 30)
np.isnan(sc.gauss2d(x, y, z, xi, xi, grid=True, scale=0.02)).sum()              # 29 NaNs of 900
np.isnan(sc.gauss2d(x, y, z, xi, xi, grid=True, scale=0.02, use32=False)).sum()  # 0
```

The `gauss2d` case is entirely ordinary: random points in the unit square, interpolated onto a grid within the data range.

**Fix**: subtract the minimum distance before exponentiating (`np.exp(-(dist - dist.min()))` for the squared-distance term), which is exactly equivalent after normalization and cannot underflow; or detect an all-zero weight sum and fall back to the nearest point.

### 21. `smoothinterp()` growth breaks when all `newx` is outside the data — `sc_math.py:1279`, `1282`

Both growth branches anchor on a neighbouring element of `newy` (`pastindices[-1]+1`, `futureindices[0]-1`) without checking that an interior point exists. With every `newx` before the data the index is `len(newx)` and raises; with every `newx` after the data, `futureindices[0]-1 == -1` wraps to the last element of `newy` — itself an extrapolated point — so growth is measured backwards from the far end and the values come out *below* the data.

```python
sc.smoothinterp([-2.,-1.],   origx, origy, smoothness=0, growth=1)  # IndexError
sc.smoothinterp([2.,3.],     origx, origy, smoothness=0, growth=1)  # actual: [0.368 1.]   expected: [2.718 7.389]
sc.smoothinterp([1.,2.,3.],  origx, origy, smoothness=0, growth=1)  # correct: [1. 2.718 7.389]
```

The second and third calls have the same anchor and the same growth rate; only the presence of one interior point distinguishes them. Branch is `# pragma: no cover`.

**Fix**: fall back to anchoring on the data boundary (`finiteorigx[0]`/`finiteorigy[0]` or `finiteorigx[-1]`/`finiteorigy[-1]`) only when no interior point exists, i.e. when `pastindices[-1]+1 == len(newx)` or `futureindices[0] == 0`. Anchoring on the data boundary unconditionally, as originally proposed, would change results in the normal case, which today anchors on the *smoothed* `newy` at the last interior `newx`; and the raw `origx`/`origy` endpoints may carry non-finite values, hence the `finite*` arrays.

### 34. `nanequal()` and `sc.equal()` treat NaNs as unequal between float32 and float64 — `sc_math.py:455`, `491`, `505`

*Added on re-verification.* The NaN sentinel `_nan_fill = -528876923.87569493` is written into each array in its own dtype. In float32 it rounds to `-528876928.0`, so the sentinels no longer match across precisions and the NaN positions compare unequal.

```python
sc.nanequal(np.array([1, np.nan], dtype=np.float32), np.array([1, np.nan]))   # actual: [ True False]   expected: [True True]
sc.equal(np.array([1, np.nan], dtype=np.float32), np.array([1, np.nan]))      # actual: False            expected: True
```

Starsim uses float32 extensively, so mixed-precision comparisons (e.g. comparing saved results against freshly computed ones) are realistic.

**Fix**: drop the sentinel and compare the NaN masks directly: compute `isnan_arr = pd.isna(arr)` once, then per `other`, `eq = (arr == other) | (isnan_arr & pd.isna(other))`. This also removes the (tiny) chance of a real value colliding with the sentinel.

### 35. `gauss1d(..., use32=False)` crashes on list input — `sc_math.py:1366-1372`

*Added on re-verification.* The docstring documents `x`, `y`, and `xi` as "1D list". With `use32=True` they are converted by `_arr32()`, but with `use32=False` they are never converted to arrays, so `x - xi` fails.

```python
sc.gauss1d([0,1,2,3], [1.,2,3,4])               # works: [2.2513 2.4163 2.5837 2.7487]
sc.gauss1d([0,1,2,3], [1.,2,3,4], use32=False)  # TypeError: unsupported operand type(s) for -: 'list' and 'int'
```

**Fix**: convert unconditionally before the optional float32 cast: `x, y, xi = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(xi, dtype=float)`. Doing so, and restoring `orig_dtype` only when it is a floating dtype, also fixes the `gauss1d` site of finding 5.

### 36. `smooth()` gives wrong values for 1-D arrays of length 1 or 2 — `sc_math.py:1123-1128`

*Added on re-verification.* Same root cause as finding 4, but reachable from the default call. The kernel always has length at least 3 (`[0.25, 0.5, 0.25]`), even when the auto-computed `repeats` is 0, so a 1-D array shorter than 3 points goes through `convolve()`'s broken long-kernel branch.

```python
sc.smooth(np.array([3.]))   # actual: [1.5]           expected: [3.]
sc.smooth(np.ones(2))       # actual: [0.333, 0.75]   expected: [1., 1.]
```

Very short series occur naturally at the start of a simulation or after filtering, and the output is silently wrong rather than an error.

**Fix**: the corrected fix for finding 4 resolves this (verified: `[3.]` and `[1., 1.]`).

## Low severity

### 11. `normalize()` returns NaN for constant input — `sc_math.py:732`

After `out -= out.min()`, a flat array has `max() == 0`, so `out /= out.max()` is 0/0. The documented guarantee that the output lies in `[minval, maxval]` is violated, and the caller gets NaNs plus a bare numpy `RuntimeWarning` pointing into Sciris internals. A constant series and a length-1 array are both plausible inputs (e.g. normalizing a flat time series for a colormap). Severity revised from Medium to Low on review.

```python
sc.normalize([5,5,5])                # actual: [nan, nan, nan]
sc.normalize([7])                    # actual: [nan]
sc.normalize(np.full(4, 2.0), 10, 20)  # actual: [nan nan nan nan]
```

**Fix**: if the shifted max is 0, fill with `minval` (or the midpoint) instead of dividing.

### 14. `nanequal(equal_nan=False)` crashes on list input — `sc_math.py:490`, `496`

`sc.toarray(arr)` is applied only inside the `if equal_nan:` block (line 490), so with `equal_nan=False` the raw input reaches `other.shape != arr.shape` (line 496). Lists are an accepted input type — the docstring's second example passes one. Severity revised from Medium to Low on review.

```python
sc.nanequal(np.array([1,2,3]), [1,2,3], equal_nan=False)  # [True True True]
sc.nanequal([1,2,3], [1,2,3], equal_nan=False)            # AttributeError: 'list' object has no attribute 'shape'
sc.nanequal(1, 1, equal_nan=False)                        # AttributeError: 'int' object has no attribute 'shape'
```

Blast radius is limited: `sc_nested.py:1145` calls `sc.nanequal(..., equal_nan=self.equal_nan)` from `sc.equal()`, but that path converts to arrays first, and `sc.equal([1,2,3],[1,2,3], equal_nan=False)` was verified to still return `True`.

**Fix**: hoist `arr = sc.toarray(arr)` out of the `if equal_nan:` block, keeping the `.copy()` where the NaN fill happens. (If finding 34's fix is adopted, the fill and copy go away entirely.)

### 20. `getvalidinds()` mishandles boolean filters — `sc_math.py:306`

A boolean `filterdata` is passed through unchanged and then intersected against *integer* indices, so `True`/`False` are compared as 1/0. Severity revised from Medium to Low on review.

```python
sc.getvalidinds([3,5,8,13], np.array([True,False,False,True]))  # actual: [0 1]   expected: [0 3]
sc.getvalidinds([3,5,8,13], np.array([True,True,True,True]))    # actual: [1]     expected: [0 1 2 3]
```

The function is deprecated and `# pragma: no cover`, but it is still exported in `__all__`, and the sibling `getvaliddata()` handles boolean filters correctly, so the intent is unambiguous.

**Fix**: `filterindices = findinds(filterdata)` in the boolean branch.

### 24. `similarity()` error message is not an f-string — `sc_math.py:963`

```python
sc.similarity({1}, {2}, method='cosine')
# ValueError: Method must be "jaccard" or "dice", not "{method}"
```

The sibling message at line 781 is correctly an f-string. **Fix**: add the `f` prefix.

### 27. `smoothinterp()` docstring has its central claim backwards — `sc_math.py:1160`

"Unlike `np.interp()`, this function does exactly pass through each data point" — it does not, by design.

```python
sc.smoothinterp(origx, origx, origy, smoothness=5)  # [0.114 0.227 0.382 0.549 0.697 0.809 0.890 0.945]
origy                                               # [0.    0.2   0.1   0.9   0.7   0.8   0.95  1.   ]
```

Max deviation 0.351; only `smoothness=0` reproduces the inputs. Endpoints are additionally pulled inward by the default `keepends=True` constant padding (perfectly linear `0..4` returns `0.0778 ... 3.9222`; `keepends=False` returns it exactly). **Fix**: reword to "does *not* exactly pass through each data point (unless `smoothness=0`)", and document `keepends` in the Args block.

### 30. `gauss2d()` length guard never fires — `sc_math.py:1501`

`if len(x) != len(y) != len(z):` means `len(x) != len(y) and len(y) != len(z)`, so the common mistake — x and y consistent, z the wrong length — skips the check entirely. With `len(z) == 1` it then broadcasts to a silently wrong constant:

```python
sc.gauss2d(np.array([0.,1.,2.]), np.array([0.,1.,2.]), np.array([5.0]), use32=False)  # actual: [5. 5. 5.]
sc.gauss2d(np.array([0.,1.,2.]), np.array([0.,1.,2.]), np.array([1.,2.]), use32=False)
# ValueError: operands could not be broadcast together -- not the function's own message
```

**Fix**: `if not (len(x) == len(y) == len(z)):`.

## Misplaced `# pragma: no cover`

Six pragmas sit on reachable paths, two of them on paths with confirmed bugs — hiding exactly the code that most needs testing.

| Line | Branch | Reachable via |
|------|--------|---------------|
| 185 | `findinds()` shape mismatch | `sc.findinds(np.array([True,True]), np.array([True,True,True]))` raises this `ValueError` |
| 410, 413 | `sanitize()` verbose / `die=False` | `verbose=True` on all-NaN input; `die=False` on 2-D input |
| 604 | `numdigits()` negative + `count_minus` | The docstring's own `sc.numdigits(-12345, count_minus=True) # Returns 6` |
| 1095 | `convolve()` long-kernel branch | `sc.convolve(np.ones(3), np.ones(5)/5)` (finding 4), or `sc.smooth(np.ones(2))` (finding 36) |
| 1501 | `gauss2d()` length guard | Never fires at all (finding 30) |

## Verified clean

Recorded so the same ground isn't re-covered. All of the following were hypothesised, tested by execution, and found correct, except where a later finding is noted.

**`approx()`, `findinds()`, `findfirst()`, `findlast()`, `findnearest()`, `count()`**: every docstring example reproduces; `first`/`last`/`ind` including `ind=[0,1]` and the `first=True, last=True` `ValueError`; `die=True`/`die=False` for both 1-D and multi-D no-match cases; deprecated `val1`/`val2` kwargs plus their `FutureWarning`; multidimensional index tuples; `findfirst`/`findlast` forwarding; `findnearest` with scalar/unsorted/out-of-range/tuple/`datetime64`/`float32`/signed-int input (but see finding 33 for NaN-containing input); pandas Series and string matching; no in-place mutation anywhere (`boolarr *= arg` is safe because `sc.toarray()` copies).

**`sanitize()` and friends**: all eight `sanitize()`/`numdigits()` docstring examples and the `findnans()`/`nanequal()` examples; no input mutation on any path; all-NaN input returning a zero-length array (when not interpolating; see finding 17); `defaultval` override; single-valid-value broadcast; NaNs at both ends held flat by `'nearest'` and `'linear'`; empty input; integer pass-through; multidim `returninds`; the documented multidim `NotImplementedError` for `'nearest'`. `isprime()` exhaustively matches `sympy.isprime` over -10..200000 (the `int(n**0.5)` float-sqrt concern is unfounded: zero counterexamples exhaustively in [2**26, 2**28) and in 300k samples up to 2**45). `numdigits()` correct for 0, `+/-20000` exhaustively, all `10**k` and `10**k +/- 1` for k <= 14, 200k random integers below 1e15, decimals, `count_decimal`, `count_minus` exhaustively over -2000..-1, multi-arg mixed-sign calls, and the scalar-in/scalar-out rule. `nanequal()` with `scalar=True/False`, mismatched shapes, 2-D, ragged object arrays, and string arrays (no `<U` sentinel collision; but see finding 34 for mixed float32/float64). `getvaliddata()` and `dataindex()` correct for their documented contracts.

**`perturb()`, `normsum()`, `normalize()`, `inclusiverange()`, `randround()`, `cat()`, `linregress()`, `sem()`, `similarity()`**: `perturb()` honours `span` exactly (200k samples, `span=0.3`, min/max 0.7000/1.3000), `normal=True` gives the right std, `randseed` is reproducible and does *not* disturb the global numpy RNG (it uses `default_rng`), and no input mutation. `randround()` is genuinely unbiased over 2M samples for positive, negative, half-integer, zero, and large values (max deviation 5.6e-4), with exactly correct per-outcome probabilities. `normsum()` handles negative entries and negative totals and preserves list type. `normalize()` respects `[minval, maxval]` for negative inputs, a reversed range, and 2-D input. `inclusiverange()` reproduces all six docstring examples including `stretch=True`, descending negative-step ranges, and float-stable large offsets (`sc.inclusiverange(2000, 2020, 0.2)` ends exactly at 2020.0). `cat()` handles scalars mixed with arrays, `None`/empty args, `axis=1`, the 2-D promotion, integer-dtype preservation, and never aliases an input. `linregress()` matches `scipy.stats.linregress` to ~1e-15 unweighted. `sem()`'s `axis` handling is right for int, negative, and tuple axes. `similarity()` matches the textbook Jaccard and Dice definitions for overlapping, identical, disjoint, empty-vs-empty, and empty-vs-nonempty input, with a symmetric unit-diagonal N x N matrix for 3+ inputs.

**`rolling()`, `convolve()`, `smooth()`, `smoothinterp()`**: `rolling()` matches pandas for `'mean'`/`'median'`/`'sum'`, returns `'none'` as the `Rolling` object, is an exact identity at `window=1`, gives identical output for list/array/Series input, rolls 2-D down axis 0 preserving shape, raises cleanly on an unknown operation, and doesn't mutate. `convolve()` reproduces its docstring example, always returns `len(a)`, handles even-length kernels correctly (constant input preserved exactly), matches a divide-by-support reference (rescaled by `v.sum()`) for every `len(a) >= len(v)` case including asymmetric and unnormalized kernels, and doesn't mutate. `smooth()` preserves 1-D constants to machine precision for n = 5, 10, 20, 50 (but see finding 36 for n = 1, 2); never mutates the input; genuinely smooths 2-D in both dimensions (matches a manual rows-then-columns reference); builds a correctly normalized composite kernel for `repeats >= 2`; `legacy=True` behaves as documented (no edge correction); an unnormalized custom kernel scales the output, which is correct convolution semantics. `smoothinterp()` correctly sorts and un-sorts `newx` (`np.argsort(neworder)` is the right inverse), sorts `origx`/`origy` together, handles duplicate `origx` like `np.interp`, clamps outside the range without `growth`, extrapolates correctly whenever one `newx` is interior, drops and interpolates across NaNs with `ensurefinite=True` and reinstates them with `False`, always returns `len(newx)`, short-circuits at `smoothness=0`, supports `method='nearest'`, and doesn't mutate.

**`gauss1d()`, `gauss2d()`**: `gauss2d` reproduces an independently written Gaussian-weighted-interpolation reference to `maxdiff = 0.0` in float64, both scattered and on a meshgrid, with weights normalized to 1. `xsc = xscale*scale` / `ysc = yscale*scale` is correct and the axes are *not* swapped (a swapped reference differs by 0.139 where the correct one differs by 0.0); elongation follows the intended axis. `grid=True` returns `(len(yi), len(xi))` with `out[i,j]` corresponding to `(x=xi[j], y=yi[i])`, proven with an asymmetric `z = x` case. `grid=False` with mismatched `xi`/`yi` raises the documented error. No input mutation under either `use32` setting. The 2-D-`z` convenience path (`# pragma: no cover` at 1465) has the right `arange` convention, preserves feature location, and its transpose heuristic behaves as intended. `gauss2d` handles a single point and identical points correctly (its `scale` defaults to 1.0). Linear input is recovered to 9e-9 in the interior; edge shrinkage is the expected behavior of a truncated kernel. The missing textbook factor of 2 in `exp(-dist**2)` is not a bug, since `scale` is free and the weights are normalized.

## Rejected on review

These original findings were rejected during the 2026-09-25 re-verification. The behavior described was reproduced in every case; the verdict concerns whether it warrants a fix.

- **9. `gauss2d()` `use32=True` collapses large-offset coordinates to a constant field** — Not worth fixing: this is float32's inherent ~7-digit precision (an offset of 1.6e9 with a spread of 10 needs 8+ digits), which `use32=False` exists to avoid; at most add a sentence to the `use32` docs.
- **10. `sem()` uses the population std (`ddof=0`)** — Not a bug: the docstring defines the function as `array.std()/np.sqrt(len(array))`, so `ddof=0` is the documented contract, and once finding 7 is fixed callers can pass `ddof=1`.
- **15. `numdigits()` off by one at magnitudes >= 1e15** — Not worth fixing: only values within ~1 ULP of a power of ten at >= 1e15 are affected, and the practical impact is one extra space of column width.
- **16. `smooth(repeats=0)` still smooths** — Not worth fixing: an explicit `repeats=0` is only a loop edge case, and short data still getting one smoothing pass is desirable behavior.
- **18. `gauss1d()` returns all-NaN for a single point or zero-range `x`** — Not worth fixing: degenerate input for which a data-derived default `scale` of zero is unsurprising.
- **22. `findinds()` rejects a list among the extra positional arguments** — Not worth fixing: the docstring asks for "additional boolean arrays", and a list gives a clear `AttributeError` rather than a wrong result.
- **23. `cat()` has a dead `copy` keyword** — Not worth fixing: the unused kwarg is harmless, and removing it would break callers that still pass `copy=`.
- **25. `linregress()` `r2`/`corr` ignore `polyfit` kwargs such as `w=`** — Not worth fixing: `corr`/`r2` are the standard raw-data Pearson values and this is reachable only when passing `w=`; at most add a doc note.
- **26. `isprime()` reports non-integer floats as prime** — Not worth fixing: primality of non-integers is outside the function's contract.
- **28. `sanitize()` `die=False` fallback and stale "Returning 0." message** — Not worth fixing: wording only, and returning the original data on `die=False` is reasonable.
- **29. `findinds()` docstring says "list of tuples"** — Not worth fixing: a changelog-line doc typo (line 149).
- **31. Both Gaussian docstring examples raise `NameError`** — Not worth fixing: docstring typos (`xi` should be `xi3`; the missing `import sciris` matches the convention of other examples).
- **32. `inclusiverange()` leaks an opaque `np.linspace` error for a mis-signed step** — Not worth fixing: the call already raises, so only error-message quality is at stake.

## Suggested order of work

1. **Findings 1, 5** — silent data corruption on ordinary input (fabricated zeros; integer truncation across four functions), and both are small, local fixes.
2. **Findings 2, 3, 4, 6, 7, 33** — silently wrong numbers or indices from plain calls. Finding 4 should be fixed before 6 and 36, since both are consequences of it.
3. **Finding 12** — silent false matches on large-magnitude values; fix together with `sc_dataframe_bugfixes.md` finding 4, and note the behavior change in the changelog.
4. **Findings 8, 13, 17, 19, 21, 34, 35, 36** — dtype-dependent wrong answers, NaN returns, and crashes on in-contract input.
5. **Findings 11, 14, 20, 24, 27, 30 and the pragma list** — edge-case NaNs, deprecated functions, documentation, and error messages.

Findings 1, 2, 3, 4, 5, 6, 7, 12, and 33 all produce plausible-looking numbers rather than errors, so none of them would announce itself in downstream code. Each also has a natural regression test (constant input must be preserved; an integer input must match its float twin; the endpoint must be present; `count` must equal `np.count_nonzero`; `eps=0` must match exactly; NaNs in the series must not be selected), which would be worth adding alongside the fixes.
