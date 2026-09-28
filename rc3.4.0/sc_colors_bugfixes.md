# `sc_colors.py` bug audit

Audit of `sciris/sc_colors.py` (1030 lines) for genuine defects: wrong numerical or color results, documented arguments that don't work, silent data corruption, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps, and performance.

**Scope**: the whole file, covered end to end by two parallel auditors (lines 29-497 and lines 498-1030). **Method**: line-by-line reading of each function, followed by executed hypothesis tests -- every "actual" value below was produced by running the code against the editable install (Sciris 3.3.0, numpy 2.4.6, commit `2d69aad`), and every finding was reproduced a second time independently before being recorded here. Line numbers refer to the current working tree.

**Independent re-verification.** This document was independently re-verified on 2026-09-25 against commit `d91898a` (Sciris 3.3.0, NumPy 2.4.6, Matplotlib 3.11.1, `SCIRIS_BACKEND=agg`): every finding was re-run from scratch. Of the original 26 findings, 13 were confirmed as stated (some with corrected severity or fix), 2 were rewritten because the description or proposed fix was inaccurate (4, 10), and 11 were rejected as not a bug or not worth fixing (listed under "Rejected on review" near the end). Two missed bugs were added (27, 28). The document now contains 17 findings: 4 High, 7 Medium, 6 Low. Original finding numbers are kept stable, so the numbering has gaps.

**Nothing in this document has been applied.** All fixes are described, not made.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `sc.rgb2hsv()` | Integer RGB input truncates the hue result to 0, reporting every saturated primary/secondary as red | 182 |
| 2 | High | `sc.shifthue()` | Integer color input returns completely unshifted colors | 123 |
| 3 | High | `sc.vectocolor()` | Constant or single-element vector returns fully transparent black (0/0 -> NaN -> invisible) | 274 |
| 5 | High | `sc.colormapdemo()` (root cause `sc.fig3d()`) | Returns a blank figure as its `'3d'` entry, plots the surface into a leaked third figure, and misdirects the colorbar | 681, 686 |
| 4 | Medium | `sc.bicolormap()`, `sc.bandedcolormap()` (and other colormap constructors) | `apply=True` registers by name, so a customized colormap is silently replaced by the registered default in later plots | 809, 982 |
| 6 | Medium | `sc.colormapdemo()` | Unconditionally reseeds the global NumPy RNG with no opt-out, destroying the caller's random stream | 659 |
| 7 | Medium | `sc.hsv2rgb()` | Raises `ValueError` for any integer-dtype input | 200 |
| 8 | Medium | `sc.rgb2hex()` | Raises `UFuncTypeError` for integer black or white | 142 |
| 11 | Medium | `sc.vectocolor()` | Empty input returns a 4-tuple instead of an Nx4 array, and crashes with `asarray=False` | 293 |
| 14 | Medium | `sc.manualcolorbar()` | Raises on `data` containing NaN, and on constant `data` | 608 |
| 27 | Medium | `sc.manualcolorbar()` | Crashes when `ticklabels` is a NumPy array | 626 |
| 9 | Low | `sc.gridcolors()` | `ashex=True` is silently ignored when `asarray=True` | 101 |
| 10 | Low | `sc.sanitizecolor()` | `alpha` is silently ignored for 4-channel input | 79 |
| 16 | Low | `sc.manualcolorbar()` | Leaves the colorbar axes as the current axes, so later pyplot calls draw into it | 588 |
| 18 | Low | `sc.sanitizecolor()` | The grey-scalar shortcut only accepts an exact Python `float`, rejecting ints and `np.float32` | 70 |
| 19 | Low | `sc.gridcolors()` | `demo=True, reverse=True` plots each point at one color's coordinates but paints it a different color | 462 |
| 28 | Low | `sc.manualcolorbar()` | Accepts a `fig` argument but silently ignores it | 500 |

## Recurring patterns

**Integer dtype round-trip through a container.** `_listify_colors()` (`sc_colors.py:36`) builds `np.array(colors)` without forcing a float dtype, and all three of its consumers then assign float results back into that array (`sc_colors.py:123`, `182`, `201`). One helper line causes two silently-wrong-value bugs and one crash. This is the same defect class as findings 5 and 9 of `sc_math_bugfixes.md` -- writing a float result into an integer container -- and a partial fix was already attempted at `sc_colors.py:119` for the loop variable only.

**Zero-range normalization.** `sc.vectocolor()` divides by `maxval - minval` with no guard, exactly as `sc.normalize()` does in `sc_math.py` (finding 11 there). Both produce NaN/invisible output for constant input; a shared guarded-normalization helper would fix both.

**Flags/arguments handled in only one output branch.** `_processcolors()` applies `ashex` only in the list branch, so `ashex` + `asarray` silently loses the requested representation; `sanitizecolor()` applies `alpha` only when `len(color)==3`, so `alpha` silently loses effect for RGBA (finding 10, where the internal `vectocolor(nancolor=...)` caller depends on that behavior). In both cases the kwarg is documented unconditionally. (The analogous `manualcolorbar()` `values`-without-`colors` item, original finding 23, was rejected on review.)

**Global state written, never restored.** Three separate mechanisms in the colormap/colorbar half of the file leave the session altered after an ordinary call: `plt.set_cmap` writes `rcParams['image.cmap']` (all six colormap functions, `apply=True`), `np.random.seed` overwrites the process RNG (`colormapdemo`), and `plt.axes` leaves the colorbar axes current (`manualcolorbar`). None of them is documented as global, and none has a save/restore or an opt-out.

**Truthiness/sentinel tests on values that can legitimately be arrays or falsy.** `if ticklabels:` at line 626 crashes when the labels are a NumPy array (finding 27); `if cax is None and (axarg or axkwargs)` at line 587 has the same pattern for an ndarray rect (original finding 15, rejected on review as out of the documented contract). This is the same "sentinel tested with truthiness" family that dominated the `sc_math.py` audit (see that document's recurring patterns).

**Named-object identity assumed to survive a string round-trip.** The `apply=True` bug rests on the assumption that a colormap's *name* uniquely identifies its *contents*. It does not, once a function can produce parameterized variants under a fixed name.

## High severity

### 1. `sc.rgb2hsv()` truncates its own output to the input dtype, so integer RGB triplets return a wrong hue -- `sc_colors.py:182`

`_listify_colors()` builds the working array with `colors = np.array(colors)` (`sc_colors.py:36`), which preserves an integer dtype for a plain integer triplet such as `(0,1,1)`; the loop then writes the float HSV result back into that integer array with `colors[c] = hsvcolor` (`sc_colors.py:182`), truncating every fractional component toward zero. Since hue is always in [0,1), the hue of any integer-specified color is truncated to `0`, i.e. reported as pure red.

```python
sc.rgb2hsv([0,1,1])     # cyan
sc.rgb2hsv([0.,1.,1.])  # same color, float
```

```
[0 1 1]
[0.5 1.  1. ]
```

Actual: `[0, 1, 1]` (hue 0 = red). Expected: `[0.5, 1, 1]` (hue 0.5 = cyan), which is what `mpl.colors.rgb_to_hsv(np.array([0,1,1], dtype=float))` returns and what the float call returns. The same wrong answer `[0 1 1]` comes back for `[0,0,1]` (blue, hue 0.667), `[1,0,1]` (magenta, hue 0.833) and `(0,1,0)` (green, hue 0.333) -- every saturated primary/secondary written as integers collapses onto the same value. Pure-integer RGB triplets are the single most common way to write a saturated color by hand, and nothing warns.

Blast radius: `sc.rgb2hsv()` is public (`__all__`, `sc_colors.py:26`) and is not called elsewhere inside Sciris (the only internal `hsv2rgb` caller, `sc_colors.py:978`, passes a float array and is unaffected). `tests/test_colors.py:32` only tests a float `np.array([0.53, 0.74, 0.15])`, so the integer path is untested.

**Fix**: in `_listify_colors()`, build the array as float (`np.array(colors, dtype=float)`), or in `rgb2hsv()`/`hsv2rgb()` accumulate results in a fresh float array rather than assigning back into `colors`. `shifthue()` already does the float coercion for the *input* to `rgb_to_hsv` (`sc_colors.py:119`, commented "Required for NumPy 2.0") but not for the output array, so a single fix in `_listify_colors()` closes all three functions.

### 2. `sc.shifthue()` returns integer colors completely unshifted -- `sc_colors.py:123`

Same mechanism as the previous finding: `colors[c] = rgbcolor` (`sc_colors.py:123`) writes floats into the integer array produced at `sc_colors.py:36`. For a color written as integers the shifted RGB values are truncated to 0/1, so for most hue shifts the function silently returns its input unchanged. The docstring example itself uses integer tuples.

```python
sc.shifthue(colors=[(1,0,0),(0,1,0)], hueshift=0.1)     # docstring's input form
sc.shifthue(colors=[(1.,0.,0.),(0.,1.,0.)], hueshift=0.1)
```

```
[[1 0 0]
 [0 1 0]]
[[1.  0.6 0. ]
 [0.  1.  0.6]]
```

Actual: the input, unmodified. Expected: `[[1, 0.6, 0], [0, 1, 0.6]]`. A single color behaves the same way: `sc.shifthue(colors=(1,0,0), hueshift=0.1)` returns `[1 0 0]`. The docstring's own `hueshift=0.5` happens to be the one shift that survives truncation, because red -> cyan is exactly `(0,1,1)`; that is also the only case `tests/test_colors.py:22` exercises, and it asserts nothing.

Blast radius: `sc.shifthue()` is public and is used internally at `sc_colors.py:457` by `sc.gridcolors(hueshift=...)`, where `colors` is always a float array, so `gridcolors()` is unaffected.

**Fix**: as above -- coerce to float in `_listify_colors()` (`sc_colors.py:36`), or write into a separate float output array.

### 3. `sc.vectocolor()` returns fully transparent black for a constant or single-element vector -- `sc_colors.py:274`

`diff = maxval - minval` is zero whenever every value is identical (including any length-1 vector), so `vector = (vector - minval)/diff` at `sc_colors.py:275` is `0/0` -> `nan` for every element, with a leaked `RuntimeWarning: invalid value encountered in divide`. `nan` then falls through to `cmap(point)` at `sc_colors.py:288` (the NaN branch above it only fires when `nancolor is not None`), and Matplotlib's "bad" color is `(0,0,0,0)`, i.e. invisible. The documented promise is "It automatically scales the vector to provide maximum dynamic range for the color map" and a return of "Nx4 array of RGB-alpha color values"; an alpha of 0 for ordinary finite data is neither.

```python
sc.vectocolor([3,3,3], cmap='viridis')
sc.vectocolor([5.0], cmap='viridis')
sc.vectocolor(1, cmap='viridis')          # scalar form -> linspace(0,1,1)
sc.arraycolors(np.ones((2,2)))[0]
```

```
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
[[0. 0. 0. 0.]]
[[0. 0. 0. 0.]]
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]]
```

Expected: some valid color (any consistent choice -- `cmap(0)`, `cmap(0.5)`, or `cmap(1)`), with alpha 1. Note that `sc.vectocolor(1)` is the `n=1` case of the scalar-count form used in the `sc.animation()` docstring at `sc_plotting.py:1857`.

Blast radius: `sc.arraycolors()` inherits it (shown above), and both are used by `_process_colors()` in `sc_plotting.py:183`/`186`, so a 3-D plot whose color data happens to be flat renders as nothing at all:

```python
ax = sc.bar3d(np.ones((3,3)))
ax.collections[0].get_facecolors()[0]     # -> [0. 0. 0. 0.]  (entirely invisible plot)
```

This is the same class of defect as `sc.normalize()` returning all-NaN for constant input (finding 11 of `sc_math_bugfixes.md`).

**Fix**: guard the normalization, e.g. `if diff == 0: vector = np.full(vector.shape, 0.5)` (or `0.0`), keeping NaNs as NaNs; and consider using `sc.safedivide()`.

### 5. `colormapdemo()` returns a blank figure as its `'3d'` entry, plots the surface into a third, undeclared figure, and misdirects the 3-D colorbar -- `sc_colors.py:681`, `686`

`fig2, ax2 = sc.fig3d(returnax=True, figsize=(12,8))` returns a figure that is *not* `ax2`'s parent. `sc.fig3d` (`sc_plotting.py:45-47`) creates `fig = plt.figure(**figkwargs)` and then calls `ax3d(..., figkwargs=figkwargs)`; inside `ax3d` the test `(fig is None and figkwargs)` is true, so `ax3d` creates a *second* figure and puts the 3-D axes there. `colormapdemo` therefore (a) returns an empty figure under the key `'3d'`, (b) leaks a third figure that the caller has no handle on and cannot close, and (c) calls `fig2.colorbar(surf)`, which matplotlib flags with a `UserWarning` and attaches to the wrong figure.

```python
import matplotlib.pyplot as plt, sciris as sc
figs = sc.colormapdemo(n=10, smoothing=1, doshow=False)
print(figs['3d'].axes, len(plt.get_fignums()))
```

```
/home/cliffk/sc/sciris/sciris/sc_colors.py:686: UserWarning: Adding colorbar to a different Figure <Figure size 1200x800 with 2 Axes> than <Figure size 1200x800 with 0 Axes> which fig.colorbar is called on.
figs['3d'] axes: [] | open figures: 3
```

Expected: `figs['3d'].axes` should contain the 3-D axes, and exactly 2 figures should be open. Saving the returned figure confirms it is genuinely empty -- `figs['3d'].savefig('x.png')` produces an image whose only unique RGB value is `[255 255 255]`. Every `colormapdemo()` call in the module's own docstrings reproduces the warning (checked for `'inferno'`, `'parula'`, `'alpine'`, and the parula/turbo/banded/orangeblue "Demo and example" blocks), so this is unconditional, not an edge case. `tests/test_colors.py:83` calls `sc.colormapdemo('parula', doshow=False)` and discards the return value, so the test suite passes while emitting the warning.

Root cause (corrected on review): `figkwargs` is never empty, not merely "because it holds `figsize`". `sc.fig3d()` always builds it with `sc.mergedicts(figkwargs, kwargs, num=num)`, so it is `{'num': None}` even when called with no arguments. Consequently `f, a = sc.fig3d(returnax=True)` with no kwargs at all already returns a mismatched pair (`a.figure is f -> False`, 2 figures open; re-verified).

Blast radius: `colormapdemo()` has no internal callers, but the root cause is in `sc.fig3d(returnax=True)`, which is public and exported from `sc_plotting.py` -- every caller of `fig3d(returnax=True)`, with or without extra arguments, gets a mismatched `(fig, ax)` pair and an extra leaked figure.

**Fix**: in `sc.fig3d` (`sc_plotting.py:46`), pass the already-created figure through (`ax = ax3d(..., fig=fig, ...)`) instead of re-passing `figkwargs`; then `colormapdemo`'s `fig2.colorbar(surf)` becomes correct with no change here. A local-only workaround is `fig2 = ax2.figure` after the `fig3d` call.

## Medium severity

### 4. `apply=True` sets the colormap by *name*, so a customized `bicolormap()`/`bandedcolormap()` is silently replaced by the registered default in every subsequent plot -- `sc_colors.py:809`, `982`

`plt.set_cmap(cmap)` does `rcParams['image.cmap'] = cmap.name`, i.e. it stores a *string*. Every colormap in this module is built with a fixed name (`'bi'`, `'banded'`, `'alpine'`, ...) and that same name is already bound at import time (lines 1013-1031) to the *default-parameter* version of the colormap. So for the two functions that take shape parameters, `apply=True` records the name `'bi'`/`'banded'`, and matplotlib then resolves that name back to the default colormap -- the caller's customization is thrown away for all future artists. The already-existing image, if any, does get the custom object (`plt.set_cmap` calls `im.set_cmap(cmap)` on `gci()`), so the same call produces two different colormaps depending on when the artist was created.

```python
import numpy as np, matplotlib.pyplot as plt, sciris as sc
plt.figure(); im1 = plt.imshow(np.array([[0.,1.],[0.,1.]]))   # existing image
custom = sc.bicolormap(gap=0.9, apply=True)
plt.figure(); im2 = plt.imshow(np.array([[0.,1.],[0.,1.]]))   # new image
print(custom(0.5)[:3], im1.cmap(0.5)[:3], im2.cmap(0.5)[:3])
```

```
custom(0.5)      : [0.1 0.1 0.1]
existing im (0.5): [0.1 0.1 0.1]   <- got the custom object
new      im (0.5): [0.9 0.9 0.9]   <- got the registered default
```

Same for `sc.bandedcolormap(minvalue=0, minsaturation=0, hueshift=0.0, saturationscale=1.0, apply=True)`: `custom(0.5) = [0.7085, 0.0, 0.0002]` but the next `plt.imshow()` uses `[0.2867, 0.7428, 0.686]`, i.e. the default banded map. Verified twice in fresh interpreters.

Two secondary problems in the same three lines. (a) The docstring says `apply` is "whether to apply this colormap to the current figure", but `plt.set_cmap` mutates the *global* `rcParams['image.cmap']`, which persists for the rest of the session and every later figure; a plain `sc.parulacolormap(apply=True)` leaves the user's session with `image.cmap == 'parula'`. (b) Calling `apply=True` twice does not raise or warn (no re-registration happens at all, only an rcParam write), so the "does it break on the second call" concern does not apply -- but the flip side is that nothing ever registers the customized map, which is what makes (1) silent.

Blast radius: no internal Sciris caller uses `apply=True` (grep over the repo finds only the six definitions and the module-level registration); the entire exposure is user code. All six `if apply:` lines are `# pragma: no cover`, so no test touches this. Severity was originally given as High; on review it is Medium, because `apply` defaults to `False`, has no internal callers, and the name collision only matters for the two parameterized constructors (`bicolormap`, `bandedcolormap`).

**Fix**: when `apply=True`, register the actual object under one fixed per-function alias and set the rcParam to that alias, e.g. `alias = f'{cmap.name}-applied'; mpl.colormaps.register(cmap, name=alias, force=True); plt.set_cmap(alias)`. This keeps the registry bounded (each call overwrites the same alias). Also reword the docstring from "the current figure" to "the global default colormap".

The two fixes originally proposed here are wrong. Assigning the object directly, `plt.rcParams['image.cmap'] = cmap`, is accepted by Matplotlib 3.11, but the next `plt.imshow()` then raises `TypeError: unhashable type: 'LinearSegmentedColormap'` (re-verified). Registering under a `uuid`-suffixed name works but adds a new global registry entry on every call.

### 6. `colormapdemo()` unconditionally reseeds the global NumPy RNG, silently destroying the caller's random stream -- `sc_colors.py:659`

`if randseed is None: randseed = 8` followed by `np.random.seed(randseed)` means that a plain `sc.colormapdemo()` -- a call the user makes to *look at a colormap* -- resets the process-wide legacy NumPy RNG to seed 8 and leaves it there. The caller never asked for a seed, and there is no way to opt out: there is no `randseed=False`/`None` path that skips the seeding.

```python
import numpy as np, matplotlib.pyplot as plt, sciris as sc
np.random.seed(1)
sc.colormapdemo(n=10, smoothing=1, doshow=False)
print(np.random.rand(2))
plt.close('all')
```

```
after demo: [0.63784086 0.88290614]
```

The same two numbers come out whatever the caller seeded beforehand (`np.random.seed(12345)` gives the identical `[0.63784086 0.88290614]`; without the `colormapdemo()` call, seed 12345 gives `[0.92961609 0.31637555]`). So a notebook that seeds, displays a colormap, and then runs a stochastic model gets a silently different -- and always identical -- result. Verified twice in fresh interpreters.

Blast radius: `colormapdemo()` has no internal callers, but it is exported in `__all__` (line 211) and is the documented demo entry point in five other docstrings in this file, so it is exactly the function a user calls interactively mid-session. Severity was originally given as High; on review it is Medium, since this is a demo function.

**Fix**: use a local legacy generator, `data = np.random.RandomState(randseed).randn(n, n)`. This removes the side effect and keeps the demo images bit-identical (re-verified: equal to `np.random.seed(8); np.random.randn(n, n)`). The originally suggested `np.random.default_rng(randseed).standard_normal((n,n))` also removes the side effect but produces different numbers, so every demo image, including those in the documentation, would change.

### 7. `sc.hsv2rgb()` raises `ValueError` for any integer-dtype input -- `sc_colors.py:200`

`_listify_colors()` hands `mpl.colors.hsv_to_rgb()` an integer array (`sc_colors.py:36`), and Matplotlib's `hsv_to_rgb` uses `np.array(hsv, copy=False, dtype=promote_types(...))`, which under NumPy 2 raises rather than copying. Every integer spelling of a valid HSV triplet fails.

```python
sc.hsv2rgb([0,1,1])
```

```
ValueError: Unable to avoid copy while creating an array as requested.
If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior change in NumPy 1.x).
```

`[0,1,1]`, `np.array([0,1,1])` and `(0,1,1)` all raise; `[0.,1.,1.]` works. `sc.rgb2hsv()` does *not* raise on integers (it silently truncates -- see above), so the two halves of the same conversion pair fail in two different ways on the same input form.

Blast radius: public function; the only internal call (`sc_colors.py:978`) passes floats. `shifthue()` was already patched for exactly this NumPy 2 issue at `sc_colors.py:119` ("Required for NumPy 2.0") but the fix was applied to the loop variable, not the container, so `rgb2hsv`/`hsv2rgb` were missed.

**Fix**: same one-line fix in `_listify_colors()` (`np.array(colors, dtype=float)`).

### 8. `sc.rgb2hex()` raises `UFuncTypeError` for integer black or white -- `sc_colors.py:142`

`if all(arr<=1): arr *= 255.` is an in-place multiplication by a float; when the caller passed integers (which is the natural spelling of black and white in the 0-1 convention) `arr` is an integer array and NumPy refuses the unsafe cast.

```python
sc.rgb2hex([0,0,0])   # black
sc.rgb2hex([1,1,1])   # white
```

```
numpy.exceptions.UFuncTypeError: Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
```

`(0,0,0)` and `np.array([0,0,0])` fail identically, while `[0.,0.,0.]` returns `'#000000'` and `[255,0,0]` works (it takes the `all(arr<=1)` false branch). So integer input works for every color *except* the ones whose channels are all 0 or 1 -- exactly black, white, and the six saturated primaries/secondaries.

Blast radius: public function; reached internally from `_processcolors()` (`sc_colors.py:103`), so `sc.gridcolors(..., ashex=True)` would hit it if a basis color were ever all-0/1 integers (today the basis arrays are floats, so it does not).

**Fix**: `arr = np.array(arr, dtype=float)` at `sc_colors.py:138`, or use `arr = arr * 255.` instead of `*=`.

### 11. `sc.vectocolor([])` returns a 4-tuple instead of an Nx4 array, and crashes outright with `asarray=False` -- `sc_colors.py:293`

The empty-input branch sets `colors = (0,0,0,1)` -- a single color, not the documented "Nx4 array" -- and then passes that scalar tuple to `_processcolors()`, whose list branch iterates it element-by-element and calls `tuple(0)`.

```python
sc.vectocolor([])                    # asarray=True (default)
sc.vectocolor([], asarray=False)
```

```
(0, 0, 0, 1)
TypeError: 'int' object is not iterable
```

Actual: a 4-tuple, then a `TypeError`. Expected: an empty `(0,4)` array (and an empty list for `asarray=False`), which is what every caller indexing `colors[i]` in a loop over the (empty) data needs. The branch's comment ("It doesn't; just return black") shows empty input is meant to be supported, so this is a broken supported path, not an out-of-contract input.

Blast radius: `sc.arraycolors()` on a size-zero array hits the same branch and then fails its `sc.isarray(colorvec)` check with the misleading message "Creating array colors as a list does not make sense".

**Fix**: `colors = np.zeros((0,4))` in that branch, so `_processcolors()` returns an empty array or an empty list as appropriate.

### 14. `manualcolorbar(data=...)` raises on data containing NaN, and on constant data -- `sc_colors.py:608`

`vmin = data.min()` / `vmax = data.max()` propagate NaN, and the resulting `TwoSlopeNorm(vcenter=nan, vmin=nan, vmax=nan)` fails. Missing values in the plotted quantity are ordinary in scientific plotting, and the docstring's own use case is "supply the data used for plotting directly via `data`" -- matplotlib's own `scatter`/`pcolor` handle NaN by masking, so a colorbar built from the same array should too.

```python
import numpy as np, matplotlib.pyplot as plt, sciris as sc
sc.manualcolorbar(np.array([1.0, np.nan, 3.0]))
```

```
ValueError: vmin, vcenter, vmax must increase monotonically
```

Expected: `vmin=1.0`, `vmax=3.0`. The same line also makes constant data fail -- `sc.manualcolorbar(np.array([2.0,2.0,2.0]))` raises `ValueError: vmin, vcenter, and vmax must be in ascending order` -- which is the natural result of plotting a single-valued field. Note the error is additionally raised once inside a matplotlib draw callback, so two full tracebacks are printed to stderr before the exception surfaces, which obscures the cause.

The NaN case is the substantive one (NaN in plotted data is normal); the constant-data case is a corner case, but the same fix covers it.

**Fix**: use `np.nanmin`/`np.nanmax` at lines 608-609, and widen a degenerate range (e.g. if `vmin == vmax`, fall back to `vmin-0.5`/`vmax+0.5`, or to a plain `Normalize`) before building the norm.

### 27. `manualcolorbar()` crashes when `ticklabels` is a NumPy array -- `sc_colors.py:626`

`if ticklabels:` truth-tests the argument, so an array of labels (e.g. `np.array([...])` or `np.char.mod('%d%%', vals)`) raises. This matters most in the fully manual use case, where `ticks` is typically an array and the labels are derived from it. (Found on the 2026-09-25 re-verification.)

```python
import numpy as np, sciris as sc
sc.manualcolorbar(vmin=0, vmax=2, ticks=[0,1,2], ticklabels=np.array(['a','b','c']))
```

```
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

Actual: `ValueError`. Expected: a colorbar labelled a/b/c, which is what the same call with a list gives.

**Fix**: `if ticklabels is not None:`.

## Low severity

### 9. `ashex=True` is silently ignored when `asarray=True` in `sc.gridcolors()` -- `sc_colors.py:101`

In `_processcolors()` the hex conversion sits inside the `else` (list) branch, so when `asarray` is true the function returns early with the raw float array and `ashex` never takes effect. Both are documented as independent flags of `sc.gridcolors()` ("ashex (bool): whether to return colors in hexadecimal representation", "asarray (bool): whether to return the colors as an array instead of as a list of tuples").

```python
sc.gridcolors(3, ashex=True)
sc.gridcolors(3, ashex=True, asarray=True)
```

```
['#377eb8', '#e41a1c', '#4daf4a']
[[0.21568627 0.49411765 0.72156863]
 [0.89411765 0.10196078 0.10980392]
 [0.30196078 0.68627451 0.29019608]]
```

Actual: an RGB float array. Expected: either an array of hex strings, or an error saying the combination is unsupported -- not a silent downgrade to a different color representation, which will surface far downstream as a type error or as a mislabelled legend.

Blast radius: `_processcolors()` is also called from `sc.vectocolor()` (`sc_colors.py:296`), but that call does not pass `ashex`, so only `sc.gridcolors()` is affected.

Severity was originally given as Medium; on review it is Low (an uncommon flag combination with an obvious symptom).

**Fix**: hoist the `ashex` conversion out of the `else` branch so it applies to both, returning `np.array([...])` of strings when `asarray` is also set (or raise a clear error for the combination).

### 10. `sc.sanitizecolor()` silently ignores `alpha` when the input already has four channels -- `sc_colors.py:79`

The guard is `if alpha is not None and len(color) == 3`, so for an RGBA input the requested `alpha` is dropped instead of overriding the existing one; the docstring says only "alpha (float): if not None, include the alpha channel with this value".

```python
sc.sanitizecolor((0.1,0.2,0.3,0.9), alpha=0.5)
```

```
(0.1, 0.2, 0.3, 0.9)
```

Actual: `alpha=0.5` has no effect. Expected (per the docstring wording): `(0.1, 0.2, 0.3, 0.5)`.

However, this behavior is relied on internally. `sc.vectocolor()` at `sc_colors.py:286` calls `sanitizecolor(nancolor, alpha=True)` specifically to pad an RGB `nancolor` to RGBA while keeping a user-supplied alpha: `sc.vectocolor([1, np.nan, 2], nancolor=(1,0,0,0.3))[1]` returns `[1, 0, 0, 0.3]` (re-verified). Low severity: it is a doc/behavior mismatch with a working internal use.

**Fix**: if `sanitizecolor()` is changed to `if alpha is not None:` (overriding an existing alpha), line 286 must change at the same time, e.g. call `sanitizecolor(nancolor)` without `alpha` and append `1.0` only when the result has length 3. Otherwise, simply document that `alpha` only fills in a missing alpha channel. The original version of this finding proposed the override alone, which would have silently forced every RGBA `nancolor` to alpha 1.0.

The original finding also reported that `sc.sanitizecolor((220,20,60,0.5))` returns alpha `0.00196`, because `normalize` divides all four channels by 255. On review this is out of contract: it mixes 0-255 RGB with a 0-1 alpha, and the consistent 0-255 form `(220,20,60,255)` works correctly. It is no longer part of this finding.

### 16. `manualcolorbar()` leaves the colorbar axes as the current axes, so later pyplot calls draw into the colorbar -- `sc_colors.py:588`

`cax = plt.axes(arg=axarg, **axkwargs)` makes the new colorbar axes current and nothing restores the caller's axes, so the very next pyplot command lands in the colorbar. (Without `axkwargs` the current axes is correctly preserved, because matplotlib's `make_axes` restores it -- so the leak is specific to the `axkwargs`/`plt.axes` path, which is the third documented example.)

```python
import matplotlib.pyplot as plt, sciris as sc
fig = plt.figure(); ax = plt.gca(); ax.plot([1,2,3])
cb = sc.manualcolorbar(axkwargs=[0.1,0.5,0.8,0.1])
plt.plot([0,1],[0,1])   # user expects this on their own axes
print(len(ax.lines), len(cb.ax.lines))
```

```
user ax lines: 1 | colorbar ax lines: 1
```

Expected `2 0`; actual `1 1`. The stray line is drawn on top of the colorbar and the user's plot silently loses it.

**Fix**: capture `oldax = plt.gca()` (only if a figure already exists) before creating `cax`, and `plt.sca(oldax)` before returning; or create the axes with `fig.add_axes(...)` on an explicit figure instead of `plt.axes(...)`. The `fig.add_axes` route also fixes finding 28, since `fig.add_axes` does not change the current axes.

### 18. `sc.sanitizecolor()`'s grey-scalar shortcut only accepts an exact Python `float` -- `sc_colors.py:70`

The test is `elif isinstance(color, float)`, so the documented `sc.sanitizecolor(0.5)` shortcut is unavailable for an integer or a NumPy float32 -- the value falls through to the length check at `sc_colors.py:74` and raises. Sciris has `sc.isnumber()` for precisely this test and uses it in `sc.gridcolors()` (`sc_colors.py:384`) and in `sc.vectocolor()` (`sc_colors.py:262`).

```python
for c in [1, 0, np.float32(0.5)]:
    sc.sanitizecolor(c)
```

```
1 ValueError Cannot parse [1.] as a color: expecting length 3 (RGB) or 4 (RGBA)
0 ValueError Cannot parse [0.] as a color: expecting length 3 (RGB) or 4 (RGBA)
0.5 ValueError Cannot parse [0.5] as a color: expecting length 3 (RGB) or 4 (RGBA)
```

`0.5`, `0.0`, `1.0` and `np.float64(0.5)` all work (float64 subclasses `float`). So `sc.sanitizecolor(1.0)` is white but `sc.sanitizecolor(1)` is an error, and grey values coming out of a float32 array cannot be used at all -- a coercion the function performs for one spelling of a scalar but not an equivalent one.

**Fix**: `elif sc.isnumber(color): color = [float(color)]*3`.

### 19. `sc.gridcolors(demo=True, reverse=True)` plots each point at one color's coordinates but paints it a different color -- `sc_colors.py:462`

The demo scatter uses the *unreversed* `colors` array for the x/y/z positions but the *reversed* `output` for `c`, so with `reverse=True` the color cube is wrong: every marker sits at the RGB coordinates of color `i` while being drawn in color `n-1-i`.

```python
from unittest import mock
real = sc.scatter3d
cap = {}
def fake(x, y, z, c=None, **kw):
    cap['xyz'] = np.array([x,y,z]).T; cap['c'] = c
    return real(x, y, z, c=c, **kw)
with mock.patch.object(sc, 'scatter3d', fake):
    sc.gridcolors(5, reverse=True, demo=True)
for p, cc in zip(cap['xyz'], cap['c']):
    print(np.round(p,3), np.round(cc,3), np.allclose(p, cc))
```

```
[0.216 0.494 0.722] [1.    0.498 0.   ] False
[0.894 0.102 0.11 ] [0.635 0.306 0.6  ] False
[0.302 0.686 0.29 ] [0.302 0.686 0.29 ] True
[0.635 0.306 0.6  ] [0.894 0.102 0.11 ] False
[1.    0.498 0.   ] [0.216 0.494 0.722] False
```

Expected: every row `True` (the point at RGB coordinates `(r,g,b)` should be drawn in exactly that color; only the fixed midpoint of the reversal matches). Diagnostic-only, hence low, but the demo plot is the thing a user consults to judge the palette.

**Fix**: derive the positions from `output` (converting back from hex/list as needed), or compute the reversal once before the demo block.

### 28. `manualcolorbar()` accepts `fig=` but silently ignores it -- `sc_colors.py:500`

`fig` is in the signature but never used in the body, so the colorbar goes to the current figure regardless of the requested one. It is also missing from the Args list. (Found on the 2026-09-25 re-verification.)

```python
import matplotlib.pyplot as plt, sciris as sc
f1 = plt.figure(); f1.add_subplot(); f2 = plt.figure(); f2.add_subplot()
cb = sc.manualcolorbar(fig=f1)
print(cb.ax.figure is f1, cb.ax.figure is f2)
```

```
False True
```

Actual: the colorbar is drawn on `f2` and `f1` is untouched. Expected: `True False`.

**Fix**: if `fig` is given and `ax`/`cax` are not, use `ax = fig.gca()` instead of `plt.gca()`, create `cax` with `fig.add_axes(...)` instead of `plt.axes(...)`, and call `fig.colorbar(...)` instead of `plt.colorbar(...)`. This also fixes finding 16, since `fig.add_axes` does not change the current axes. Alternatively, remove the argument; either way, document it.

## Misplaced `# pragma: no cover`

| Line | Pragma'd code | Demonstrated reachable by |
|---|---|---|
| `sc_colors.py:74` | length check in `sc.sanitizecolor()` | `sc.sanitizecolor(1)` (see finding 18); it is an error path, but one an ordinary call reaches |
| `sc_colors.py:99` | `if reverse:` in the list branch of `_processcolors()` | `sc.gridcolors(3, reverse=True)` returns the three tuples in reversed order; this is the default (`asarray=False`) path for a documented kwarg |
| `sc_colors.py:101` | `if ashex:` in `_processcolors()` | `sc.gridcolors(3, ashex=True)` -> `['#377eb8', '#e41a1c', '#4daf4a']` -- the only working way to use a documented kwarg |
| `sc_colors.py:677`, `690` | `if doshow:` inside `colormapdemo()` | `doshow=True` is the *default*, so both `plt.show()` calls are the ordinary path; every docstring example (`sc.colormapdemo('inferno')`, `sc.colormapdemo('parula')`, `sc.colormapdemo(sc.alpinecolormap(), n=200, ...)`, and the five "Demo and example" blocks in `parulacolormap`/`turbocolormap`/`bandedcolormap`/`orangebluecolormap`/`alpinecolormap`) executes them. Coverage is zero only because `tests/test_colors.py:83` is the sole caller and it passes `doshow=False`. Confirmed reachable by running each example verbatim under `SCIRIS_BACKEND=agg`. |

The six `if apply: # pragma: no cover` lines (756, 809, 881, 944, 982, 1008) are genuinely uncovered -- no test or internal caller passes `apply=True` -- so they are *correctly* marked; they are only worth noting because finding 4 above shows the excluded code is wrong.

## Verified clean

**`_listify_colors()` and `_processcolors()`.** The dcp-then-wrap round trip is correct for both 1-D and 2-D input: a single triplet comes back as a triplet and a list of triplets comes back with the outer dimension intact, for `shifthue`, `rgb2hsv` and `hsv2rgb`. No in-place mutation of the caller's data: `sc.rgb2hsv()`, `sc.hsv2rgb()`, `sc.shifthue()` and `sc.rgb2hex()` all leave an ndarray argument bit-identical (checked with `np.array_equal` against a pre-call copy). `reverse` is correct and consistent between the array branch and the list branch (`sc.gridcolors(5)[::-1] == sc.gridcolors(5, reverse=True)` for tuples, hex strings, and arrays), and `reverse` composes correctly with `ashex`.

**`sanitizecolor()`.** All five docstring examples produce the documented results, and `'crimson'` and `(220,20,60)` agree exactly. Named colors, `'tab:'` colors, `'#rrggbb'`, 3- and 4-length tuples/lists/ndarrays, and `normalize=False` all behave as documented; `alpha=0` is correctly distinguished from `alpha=None` (it is not swallowed by a truthiness test); `asarray=True` returns a genuine `np.ndarray` of length 3 or 4; the 0-255 detection is triggered by `max()>1` as documented, so `(1,1,1)`/`(255,255,255)` both give white and `(0,0,0)` gives black without ambiguity for the 3-channel case; out-of-gamut values (`(256,0,0)`, `(-1.,0.,0.)`) are passed through unclipped rather than silently wrapped. A bare hex string without the leading `#` (`'87bc26'`) is rejected while `sc.hex2rgb()` accepts it -- an inconsistency, but the error message is accurate and actionable, so not reported.

**`rgb2hex()`/`hex2rgb()`.** Round-trip fidelity is exact: `sc.rgb2hex(sc.hex2rgb(s)) == s` for all 256 grey values and for 2000 random `#rrggbb` strings (0 failures), and for uppercase input (`'#87BC26'` -> `'#87bc26'`, correct modulo case) and for input with no leading `#`. The 3-digit expansion is correct (`'#8b2'` -> `'#88bb22'`, `'#FFF'` -> `'#ffffff'`, `'#000'` -> `'#000000'`), including uppercase 3-digit. The length guards fire correctly for 5- and 8-character strings. `sc.rgb2hex()`'s docstring example gives exactly `'#87bc26'` as claimed, and the 0-255 branch (`[255,0,0]` -> `'#ff0000'`, `[137.6,0,0]` -> `'#890000'`) is right.

**`rgb2hsv()`/`hsv2rgb()`.** For float input the round trip is essentially exact: `hsv2rgb(rgb2hsv(c))` over 2000 random float triplets has a worst-case error of `6.66e-16`. The docstring examples for both functions reproduce the documented arrays. Both correctly delegate to Matplotlib rather than reimplementing the conversion, so hue wraparound and the achromatic (`s=0`) case are handled by Matplotlib.

**`vectocolor()`.** The documented endpoints really do land on the colormap endpoints: for `np.linspace(0,10,5)` the first row equals `cmap(0.0)` and the last equals `cmap(1.0)` exactly (Matplotlib's float `__call__` maps `1.0` to the final entry rather than wrapping). The mapping is monotonic: for 101 evenly spaced inputs the nearest-colormap-index sequence is non-decreasing. `minval`/`maxval` are honored and correctly clip data outside them to the under/over colors; a reversed pair (`minval=8, maxval=2`) produces a cleanly reversed mapping rather than garbage. `midpoint` works as documented for interior values, and `midpoint` is applied *after* the 0-1 normalization so it composes correctly with explicit `minval`/`maxval`; the out-of-range assertion (`midpoint=2.0` with data in 0-1) fires with a clear message. NaN handling is correct in both directions: `minval`/`maxval` use `np.nanmin`/`np.nanmax` so a NaN does not poison the scale, `nancolor` is applied to exactly the NaN positions with alpha 1 (matching `tests/test_colors.py:62`), NaN-without-`nancolor` yields Matplotlib's bad color, and `nancolor` still works when `midpoint` is also given. `reverse` is correct for both `asarray` settings. The scalar form (`sc.vectocolor(3)`) correctly expands to `np.linspace(0,1,3)`. No in-place modification of the input: an ndarray argument and a list argument are both unchanged after the call (`sc.dcp` at `sc_colors.py:266` does its job). `cmap` accepts both a string and a `Colormap` object.

**`arraycolors()`.** Output shape is exactly `arr.shape + (4,)` for 2-D and 3-D input; the flattened result matches `sc.vectocolor(arr.reshape(-1))` element for element, so the reshape does not scramble the ordering, including for Fortran-ordered input; the input array is not mutated; `**kwargs` (checked with `nancolor` and `cmap`) reach `vectocolor()`; and `asarray=False` raises the documented explanatory error rather than returning something malformed. A Python list of lists raises `AttributeError: 'list' object has no attribute 'shape'`, which is out of contract (the docstring says "array") and so not reported. `colors = np.zeros(new_shape)` at `sc_colors.py:333` is dead (immediately reassigned) but harmless.

**`gridcolors()`.** Returns exactly `ncolors` colors, all distinct, for every `n` from 1 to 40 (checked with `np.unique(..., axis=0)`); there is no duplication at the 9/10 colorbrewer-to-kelly boundary nor at the 19/20 kelly-to-generated boundary, and the minimum pairwise RGB distance degrades smoothly (0.296 at n=9, 0.110 at n=19, 0.091 at n=20, 0.059 at n=21) rather than collapsing. `hueshift` works (it receives float arrays here, so the `shifthue` dtype bug does not bite), `hueshift=1.0` is a correct no-op modulo wraparound, and it composes with `asarray`, `ashex` and `reverse`. `basis='colorbrewer'` with `ncolors > 9` correctly falls through to the generated branch and still returns 12 distinct colors. `ncolors=0` returns an empty list without error. The iterable form (`sc.gridcolors(['a','b','c'])`, `sc.gridcolors({'a':1,'b':2})`) correctly uses `len()`. `demo=True` runs and produces a correctly labelled cube as long as `reverse` is not also set.

**`midpointnorm()`.** `vcenter` maps to exactly 0.5 (`sc.midpointnorm(vcenter=0, vmin=-1, vmax=3)(0) == 0.5`), the two arms are separately linear and monotonic over the whole range (`np.diff` of 101 samples is non-negative throughout), `vmin`/`vmax` map to 0.0/1.0 exactly even for an asymmetric range, and the docstring example (`plt.pcolor(..., cmap='bi', norm=sc.midpointnorm())`) runs. `vmin > vcenter`, `vmax < vcenter`, and `vmin == vcenter` all raise Matplotlib's `ValueError: vmin, vcenter, and vmax must be in ascending order`, which is correct behavior for a thin alias; out-of-range values map to `-inf`/`+inf`, which colormaps interpret as under/over, also standard Matplotlib. The default `vmin=None, vmax=None` correctly defers autoscaling. No defects found in this function itself; the related `vectocolor()` assertion item (original finding 12) was rejected on review.

**`manualcolorbar()`.** All four docstring examples run verbatim without error. The fully-manual example's tick/label correspondence is *correct*, not off by one: with `colors=sc.gridcolors(12)` and `values=np.sqrt(np.arange(12))`, the appended upper bound is `3.4710` (`values[-1] + diff(values[-2:])`), the 13 `BoundaryNorm` boundaries are `[0, 1, 1.4142, ..., 3.3166, 3.4710]`, all 12 requested ticks survive matplotlib's drawing pass in the requested order, and the drawn labels pair exactly as intended (`'Color 0 is nice'`@0.0, `'Color 2 is nice'`@1.4142 = `values[2]`, `'Color 10 is nice'`@3.1623, `'Color 11 is nice'`@3.3166) -- checked after an explicit `canvas.draw()`. When `values` is `None`, `np.arange(len(colors)+1)` gives the right number of bands and `norm(i) == i` for each colour. When `len(values) == len(colors)`, `values` are treated as lower band edges and the mapping `value -> colour index` is exact; `values` given as a plain Python list works as well as an array. `cmap.N` always equals `len(colors)`, so with matched lengths `BoundaryNorm` uses every colour exactly once with no off-by-one. Ticks lying outside `[vmin, vmax]` are retained rather than dropped, so no label shifting occurs from that direction. A mismatched `len(ticklabels)` *with* `ticks` supplied is caught by matplotlib with a clear `ValueError`. Without `axkwargs`/`cax` the function does *not* hijack the current axes (`plt.gca()` is still the caller's axes after the call). `axkwargs` as a dict, and as a list or tuple rect, both work; `arg=` is still a valid `plt.axes()` keyword in matplotlib 3.11. `data` is copied via `np.array(data)`, so the caller's array is never mutated. With the default `vcenter=None`, `midpointnorm` reduces to a plain linear mapping (`vcenter` is set to the exact midpoint), so a colorbar built from `data` matches a `plt.scatter(..., c=data)` drawn with the default `Normalize` -- no hidden nonlinearity. `label`/`labelkwargs` and `orientation='horizontal'` (including the vertical/horizontal branch at 627-630) all work.

**`colormapdemo()`.** All three docstring examples run (including the slow `n=200, smoothing=20` one). `cmap` accepts a registered matplotlib name, a Sciris-registered name, and a `Colormap` object. `n`, `smoothing` and `randseed` all reach the code that uses them and all change the output. The height field is correctly normalised to `[0, maxheight]` (`data -= data.min(); data /= data.max()`). The returned `'2d'` figure is correct and complete (2 axes: the pcolor plus its colorbar); only the `'3d'` entry is broken (reported above).

**Colormap constructors.** `alpinecolormap()`, `parulacolormap()`, `turbocolormap()`, `bandedcolormap()`, `orangebluecolormap()` and `bicolormap()` all return `N == 256` colormaps defined over the whole closed domain: `cmap(0.0)` equals the first control colour and `cmap(1.0)` the last (checked against the literal data tables for parula and turbo). None of them claims perceptual or luminance monotonicity except `bandedcolormap`'s "lightness mapped linearly" (a `sqrt`-vs-linear doc wording issue, original finding 26, rejected on review as documentation-only); measured relative luminance is non-monotonic for all of them, which is by design for the diverging (`bi`, `orangeblue`) and banded maps and for the terrain-like `alpine`, and parula is monotonic to within 1% of samples. `bandedcolormap`'s five shape arguments all demonstrably affect the output (max abs RGB deviation from default: `hueshift=0.0` -> 0.8992, `saturationscale=1.0` -> 0.3567, `minvalue=0.0` -> 0.3162, `minsaturation=0.0` -> 0.4839; passing a default value explicitly gives exactly 0.0000, confirming the defaults are what they say). `npts` controls the number of interpolation control points, not the output size -- the returned map is always 256 entries -- but `npts` is not documented as a colour count, so this is not a contract violation; `npts=2` works, `npts=1` raises matplotlib's own `data mapping points must start with x=0 and end with x=1`. `bicolormap`'s `gap` works in the documented direction (`cmap(0.5)` = `[1,1,1]`, `[0.9,0.9,0.9]`, `[0.5,0.5,0.5]`, `[0,0,0]` for `gap` = 0, 0.1, 0.5, 1.0) and `mingreen` sets the green channel at both extremes exactly; `epsilon=0` (three cdict nodes collapsing onto `x=0.5`) does *not* raise and yields the documented "pure red to pure blue with white in the middle" (`cmap(0.5) = [1, 0.996, 0.996]`); `gap=0` and `gap=1` are both fine.

**Module-level registration (1013-1031).** Importing `sciris` does not mutate `rcParams` (`image.cmap` is still `viridis` afterwards) and emits no warnings. Each colormap is registered under both its bare name and the `sciris-` prefix, the two aliases are numerically identical, and each registered copy carries the name it was registered under. The `if name not in existing` guard genuinely prevents the double-registration `ValueError` (`mpl.colormaps.register(cmap=sc.bicolormap(), name='bi')` by hand raises `A colormap named "bi" is already registered.`), so repeated imports and reloads are safe. `turbo` is correctly left as matplotlib's native version (which is bit-identical to Sciris's, both deriving from the same Google data) while `sciris-turbo` is registered.

## Rejected on review

These findings from the original audit were dropped on the 2026-09-25 re-verification. Their numbers are retired, not reused.

- **12.** `sc.vectocolor()` accepts `midpoint == minval`/`maxval` in its assertion, then raises from Matplotlib -- NOT WORTH FIXING: it errors either way and only the message differs; `midpoint` at a data extreme is a degenerate request (at most, tighten the assertion to `0 < vcenter < 1`).
- **13.** `manualcolorbar()` mislabels the colorbar when `ticklabels` is given without `ticks` -- NOT WORTH FIXING: this is a straight pass-through of Matplotlib semantics, Matplotlib already warns, and `ticklabels` without `ticks` is caller misuse.
- **15.** `manualcolorbar()` crashes if `axkwargs` is a NumPy array -- NOT WORTH FIXING: `axkwargs` is documented as a dict, and the ndarray rect is a corner of an undocumented extension (the one-line `axarg is not None` fix is harmless if wanted).
- **17.** `manualcolorbar()` does not validate `colors`/`values` lengths -- NOT WORTH FIXING: mismatched lengths are a caller error, and the single-color `values=[3]` case is a corner case.
- **20.** `sc.gridcolors()` `limits`/`nsteps` do nothing for `ncolors <= 19` -- NOT A BUG: the docstring states that <=19 colors use fixed palettes, which have no grid, and `basis='none'` honors both at any `ncolors`.
- **21.** `sc.gridcolors()` treats an unrecognized `basis` like `'none'` -- NOT WORTH FIXING: an input-validation nicety; documented values all behave correctly.
- **22.** `sc.rgb2hex()` truncates instead of rounding -- NOT WORTH FIXING: a 1/255 bias is imperceptible and hex round-tripping is exact.
- **23.** `manualcolorbar()` ignores `values` unless `colors` is given -- NOT A BUG: `values` is documented as "the values corresponding to the specific colors", so it only makes sense with `colors`.
- **24.** `bicolormap()` docstring wording for `epsilon`/`redbluemix` -- NOT WORTH FIXING: docstring wording only (the observations are accurate).
- **25.** `alpinecolormap()` usage example missing `import numpy as np` -- NOT WORTH FIXING: a trivial one-line docstring fix, not a code defect.
- **26.** `bandedcolormap()` value channel is `sqrt` of a linear ramp -- NOT WORTH FIXING: comment/docstring wording only; the `sqrt` is intentional.

## Suggested order of work

1. **Findings 1, 2, 7, 8** -- the integer-dtype round-trip through `_listify_colors()`. One shared one-line fix (`np.array(colors, dtype=float)`) closes four separate defects across `rgb2hsv`, `shifthue`, `hsv2rgb`, and `rgb2hex`.
2. **Findings 3, 11** -- `vectocolor()`'s zero-range-normalization and empty-input crashes, both cheap, local guards in the same function.
3. **Findings 5, 6** -- `colormapdemo()`'s leaked/blank figure (fixed in `sc.fig3d()`) and RNG reseeding; the RNG fix (finding 6) is a one-line swap to `np.random.RandomState(randseed).randn(n, n)`, which keeps the demo output identical.
4. **Finding 4** -- the `apply=True` name-collision bug; requires a slightly more involved registration fix but has no internal callers, so it can be done independently.
5. **Findings 14, 27, 16, 28** -- the `manualcolorbar()` items; 27 is a one-line fix, and switching to `fig.add_axes`/`fig.colorbar` fixes 16 and 28 together.
6. **Findings 9, 10, 18, 19 and the pragma list** -- the remaining low-severity items; note that 10 must be fixed together with its `vectocolor(nancolor=...)` caller.

Findings 1, 2, 3, 4, and 6 all produce plausible-looking colors, figures, or numbers rather than errors, so none of them would announce itself in downstream code: an integer-specified saturated color silently becomes red or stays unshifted, a flat data field silently renders as invisible, a customized colormap silently reverts to the default in later figures, and a colormap demo silently resets the caller's random stream. These are the ones most worth a regression test even before a full fix lands.
