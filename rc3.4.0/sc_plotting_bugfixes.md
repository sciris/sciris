# `sc_plotting.py` bug audit

Audit of `sciris/sc_plotting.py` (2261 lines) for genuine defects: wrong numerical or visual results, documented arguments that don't work, silent data corruption, figures/files that don't contain what the caller asked for, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, and general performance.

**Scope**: the whole file, covered by two parallel auditors working on adjacent halves. The first half (lines 36-1068) covers the 3-D plotting functions (`fig3d`, `ax3d`, `plot3d`, `scatter3d`, `surf3d`, `bar3d`) and the general plotting/tick/limit helpers (`commaticks`, `SIticks`, `setaxislim`/`setxlim`/`setylim`, `getrowscols`, `figlayout`, `boxoff`, `maximize`, `fonts`). The second half (lines 1075-2261) covers date formatting (`ScirisDateFormatter`, `dateformatter`, `datenumformatter`), figure saving and loading (`savefig`, `savefigs`, `loadfig`, `reanimateplots`, `emptyfig`), legends (`separatelegend`, `orderlegend`, `movelegend`), and the `animation` class plus `savemovie`. Environment for both halves: Sciris 3.3.0, Matplotlib 3.11.1, numpy 2.4.6, Python 3.13.9, `SCIRIS_BACKEND=agg`, commit `2d69aad`. The second half additionally notes that the `ffmpeg` binary was present (`/usr/bin/ffmpeg`), so movie encoding was genuinely exercised end to end for the `engine='matplotlib'` path, but the `ffmpeg-python` module (`import ffmpeg`) was **not** installed, so the default `engine='ffmpeg'` branch of `animation.save()` could not itself be executed and one finding below is flagged with that caveat.

**Re-verification.** This document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`; `sc_plotting.py` unchanged from the audited version), with every finding re-executed. Of the original 35 findings, 22 were confirmed (some with corrected severity or fixes), 1 was rewritten as inaccurate (17), and 12 were rejected as not a bug or not worth fixing (listed under "Rejected on review" near the end). Four new bugs found during re-verification were added as findings 36-39, giving 27 active findings: 7 High, 15 Medium, 5 Low.

**Nothing in this document has been applied.** All fixes are described, not made.

Plotting defects are unusually likely to survive review because the output still *looks like a plot*: a mislabelled axis, a figure saved from the wrong source, or an invisible bar does not raise an exception and does not look obviously wrong at a glance.

## Summary

27 active findings (7 High, 15 Medium, 5 Low). Rejected findings (12, 14, 23, 25-28, 30-34) are listed separately under "Rejected on review".

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `sc.fig3d()` | Builds a second figure internally and returns the empty one | 46 |
| 2 | High | `sc.ax3d()`, `sc.plot3d()`, `sc.scatter3d()`, `sc.surf3d()`, `sc.bar3d()` | Reuses `plt.gca()` (the global current axes) instead of the supplied figure's axes | 116 |
| 3 | High | `sc.commaticks()` | Formatter closure captures the loop variable, so several axes get labels sized for the last axis's data range | 736 |
| 5 | High | `sc.savefigs()` | Raises `TypeError` on every call that omits `filename`, including its own default `filetype='singlepdf'` example | 1505 |
| 6 | High | `sc.savefigs()` | Writes `plt.gcf()` instead of the figure being iterated, so the wrong image (or N identical images) is saved | 1522 |
| 7 | High | `sc.animation.addframe()` | Treats a supplied `Figure` as a generic `Artist`, producing a blank animation | 1968 |
| 37 | High | `sc.animation.save()` | With `sc.animation(fig=fig)`, the matplotlib engine produces a static movie (frames live in a different figure) | 2106 |
| 4 | Medium | `sc.dateformatter()` (`ScirisDateFormatter`) | Dates between 1974-08-28 and 1976-04-19 are labelled as raw year numbers | 1134 |
| 8 | Medium | `sc.bar3d()` | Constant data renders fully transparent (invisible) bars | 183 |
| 9 | Medium | `sc.plot3d()` | `c=<array>` (the docstring's own multi-colour example) raises `ValueError` | 244 |
| 10 | Medium | `sc.bar3d()` | `z=` keyword form raises `ValueError` because the bar base keeps the unflattened shape | 449 |
| 13 | Medium | `sc.figlayout()` | `figlayout(True)`/`figlayout(False)` raise `TypeError` because the two-line swap is reversed | 889 |
| 15 | Medium | `sc.dateformatter()` | `style=<Formatter instance>` can never be recognised, because the value is stringified before the `isinstance` check | 1236 |
| 16 | Medium | `sc.dateformatter()` | `axis='y'` also overwrites the x-axis formatter, corrupting the x labels | 1268 |
| 17 | Medium | `sc.savefig()` (via `sc.metadata()`) | `calling_info` is lost or misattributed because `sc.metadata()` passes `relframe+1` to `getcaller()`; same bug as `sc_versioning_bugfixes.md` #2 | 1422 (root cause `sc_versioning.py:451`) |
| 18 | Medium | `sc.savefig()` | SVG metadata is written successfully but can never be read back by `sc.loadmetadata()` | 1430 |
| 19 | Medium | `sc.savefigs()` | With an explicit `filename` and more than one figure, every figure is written to the same path | 1502 |
| 20 | Medium | `sc.orderlegend()` | Crashes on an array `order`, which is exactly what its own docstring recommends (`np.argsort()`) | 1684 |
| 21 | Medium | `sc.movelegend()` | `invisible` argument has no effect; the destination axes is blanked even when it has data | 1806 |
| 22 | Medium | `sc.animation.save()` | `imagefolder` is missing from the frame-name template handed to the `ffmpeg` engine | 2085 |
| 24 | Medium | `sc.savemovie()` | `interval` argument has no effect on the saved movie's frame rate | 2223 |
| 36 | Medium | `sc.datenumformatter()` | Without `start_date`, a real date axis is labelled ~51 years in the future (date offset added twice) | 1321 |
| 11 | Low | `sc.setaxislim()`, `sc.setxlim()`, `sc.setylim()` | A ragged list of arrays for `data` (the docstring's own example) is silently ignored | 658 |
| 29 | Low | `sc.dateformatter()`, `sc.datenumformatter()` | `start=0`/`end=0` are silently ignored by falsy-sentinel guards | 1259, 1330 |
| 35 | Low | `sc.animation.save()` | `verbose=True` default overrides the instance-level `verbose=False` set at construction | 2049 |
| 38 | Low | `sc.movelegend()` | `ncol` is not preserved (Matplotlib renamed `_ncol` to `_ncols`) | 1743 |
| 39 | Low | `sc.dateformatter()` | `style='auto'` with `dateformat=...` raises `TypeError` (Concise-only kwargs passed to `AutoDateFormatter`) | 1225 |

## Recurring patterns

Both halves independently converged on the same two shapes of mistake, which is worth stating as such rather than as isolated findings.

**Falsy-sentinel guards.** A parameter whose legitimate values include something falsy is tested for truthiness instead of `is not None`. `sc.mergedicts(None, {}, num=None)` returns the truthy dict `{'num': None}`, which is why `fig3d()` ends up building a second figure inside `ax3d()` (finding 1). `sc.movelegend()`'s `if not ax2.artists:` uses `Axes.artists` as an emptiness test, but `Line2D`/`Collection`/`Patch`/`Image` children live in `ax.lines`/`ax.collections`/`ax.patches`/`ax.images`, not `ax.artists`, so the test is essentially always true (finding 21). `sc.orderlegend()`'s `if order:` breaks on a length>1 numpy array — exactly the `np.argsort()` usage its own docstring recommends (finding 20). `sc.dateformatter()` and `sc.datenumformatter()` both guard `start`/`end` with `if start:`/`if end:`, silently dropping the legitimate value `0` (finding 29). The second half counted five such guards in its region alone; combined with `fig3d`'s `mergedicts` case and `sanitize()`'s equivalent bug documented in the `sc_math.py` audit, this is the single most repeated defect shape across the whole codebase.

**Pyplot-global calls where a figure-level call was meant.** `ax3d(fig=...)` reuses existing axes via the global `plt.gca()` instead of `fig.gca()`/`fig.axes[-1]`, so a plot silently lands in whichever figure happens to be current rather than the one the caller explicitly passed (finding 2). `sc.savefigs()` calls the pyplot-level `plt.savefig(...)`, which writes `plt.gcf()`, instead of `plot.savefig(...)` on the figure object the loop is actually iterating over (finding 6). Both are the module reaching for implicit current-figure/global state in a place where it already holds the explicit object it should use instead.

The second half additionally notes a third, narrower pattern specific to its region: **date numbers being magnitude-tested as if they were data**. `ScirisDateFormatter.format_ticks()` decides whether its input "looks like a year" (1700-2300) or "looks like an epoch offset" (`values.min() == 0`) purely from the magnitude of Matplotlib date numbers (days since 1970-01-01), so both heuristics misfire on ordinary dates that happen to produce those same magnitudes — one on 1974-1976 (finding 4), one on 1970-01-01 exactly (original finding 14, rejected on review as too narrow to be worth fixing) — and the second heuristic then also calls `date2num()` on values that are already date numbers when building its own diagnostic message.

## High severity

### 1. `sc.fig3d()` creates two figures and returns the empty one — `sc_plotting.py:46`

`fig3d()` builds `figkwargs = sc.mergedicts(figkwargs, kwargs, num=num)`, creates `fig = plt.figure(**figkwargs)`, and then passes the *same* `figkwargs` on to `ax3d()`; inside `ax3d` the guard at `sc_plotting.py:92` (`if (fig in [True, False] or (fig is None and figkwargs)) and not ax`) sees a non-empty `figkwargs` and calls `plt.figure(**figkwargs)` a *second* time, so the 3-D axes are created on a different figure than the one `fig3d` returns. Crucially this fires even with no arguments at all, because `sc.mergedicts(None, {}, num=None)` returns `{'num': None}`, which is truthy.

```python
fig, ax = sc.fig3d(returnax=True)
print(plt.get_fignums())      # actual: [1, 2]                expected: [1]
print(len(fig.axes))          # actual: 0                     expected: 1
print(ax.figure is fig)       # actual: False                 expected: True
```

The returned figure is blank, the second figure is leaked (the caller has no handle on it and cannot close it), and `fig.savefig()`/`fig.colorbar()` on the returned figure operate on the wrong object. The one case that *does* work is an explicitly-supplied `num`, because then both `plt.figure(num=...)` calls resolve to the same figure — which is exactly what `tests/test_plotting.py:25` does (`sc.fig3d(num='Blank 3D')`), so the test suite never sees it.

Blast radius: `sc_colors.py:681` (`fig2, ax2 = sc.fig3d(returnax=True, figsize=(12,8))` inside `sc.colormapdemo()`) hits this on every call; the resulting `fig2.colorbar(surf)` at `sc_colors.py:686` emits `UserWarning: Adding colorbar to a different Figure ...` and the `'3d'` figure that `colormapdemo` returns is blank. `sc.colormapdemo()` is exercised by `tests/test_colors.py:83`.

**Fix**: in `fig3d`, pass the figure it already created through to `ax3d` instead of re-passing the kwargs that made it: `ax = ax3d(nrows=nrows, ncols=ncols, index=index, returnfig=False, fig=fig, **axkwargs)`. Separately, `ax3d`'s `figkwargs`-is-truthy heuristic at line 92 should not be reached when `fig` is already a real figure, and `mergedicts(..., num=None)` should not produce a truthy dict for the no-argument case.

### 2. `ax3d(fig=...)` draws into the current figure, not the supplied one — `sc_plotting.py:116`

When the caller supplies a figure that already contains axes, `ax3d` reuses existing axes via `ax = plt.gca()` — i.e. the *global* current axes — rather than `fig.gca()`. If the supplied figure is not the current one, the plot silently lands in a different figure, while any figure-level decoration (`fig.colorbar()`) is applied to the requested one.

```python
figA, axA = plt.subplots(subplot_kw=dict(projection='3d'))
figB, axB = plt.subplots(subplot_kw=dict(projection='3d'))   # figB is now current
ax = sc.surf3d(np.arange(12.).reshape(3,4), fig=figA)
print(ax is axA, ax is axB)                # actual: False True    expected: True False
print(len(axA.collections), len(axB.collections))  # actual: 0 1    expected: 1 0
```

Actual stderr from the same run: `UserWarning: Adding colorbar to a different Figure ... which fig.colorbar is called on.` from `sc_plotting.py:379`.

This is the whole point of the `fig` argument ("if provided, use existing figure"), and it is the natural pattern for building a multi-panel figure while other figures are open. All four public 3-D plotters route through this line (`sc_plotting.py:241, 303, 366, 434`).

**Fix**: replace `ax = plt.gca()` with an axes lookup scoped to `fig` — e.g. `ax = fig.axes[-1]` (or `fig.gca()`), and additionally verify it is an `Axes3D` before reusing it, so a supplied 2-D figure gets a new 3-D subplot rather than raising.

### 3. `sc.commaticks()` labels several axes using the last axis's data range — `sc_plotting.py:736`

`commaformatter` is a closure that reads `interval = thisaxis.get_view_interval()`, but `thisaxis` is the *loop variable* of the `for ax ... for axis ...` loops below it. Every formatter installed by a single `commaticks()` call therefore shares one binding and, at draw time, all of them read the view interval of the **last axis processed by that call**. The decimal count is derived from that interval, so an axis whose range is much smaller than the last one's gets too few decimals and its labels stop matching its ticks.

**The symptom is order-dependent, and only one direction is visible.** Because axis labels have their trailing decimal zeros stripped at line 741, an axis that inherits a *larger* decimal count than it needs looks perfectly fine; only an axis that inherits a *smaller* count is visibly wrong. So the small-range axis must come **before** the large-range axis in the iteration order (`axlist` order for a figure/list, then `['x','y']` order within each axes). Formatting big-then-small shows nothing.

```python
fig, axs = plt.subplots(1, 2)
axs[0].plot([0, 0.5])          # small range -- formatted FIRST
axs[1].plot([0, 1e6])          # large range -- formatted LAST
sc.commaticks(ax=fig)
fig.canvas.draw()
print(list(np.round(axs[0].yaxis.get_ticklocs(), 2)))
print([t.get_text() for t in axs[0].get_yticklabels()])
```

```
actual   ticks : [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
actual   labels: ['-0', '0', '0', '0', '0', '0', '1', '1']
expected labels: ['-0.1', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']   # what sc.commaticks(ax=axs[0]) alone produces
```

Six distinct ticks are all labelled `0`, and the tick at `-0.1` is labelled `-0`. Swapping the two subplots (large range first, small last) gives correct labels on both, which is why this is easy to miss. This finding is **order-dependent**: it fires only when a small-range axis is formatted before a large-range one. The same asymmetry applies within a single axes when both axes are formatted at once — the documented 2.0.0 feature — where the order is fixed as x-then-y, so it fires whenever the x range is much smaller than the y range:

```python
fig, ax = plt.subplots()
ax.plot(np.linspace(0, 0.5, 10), np.linspace(0, 1e6, 10))   # x range ~0.5, y range ~1e6
sc.commaticks(ax=ax, axis=['x','y'])
fig.canvas.draw()
# actual   x labels: ['-0', '0', '0', '0', '0', '0', '1', '1']
# expected x labels: ['-0.1', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
# (axis=['y','x'], i.e. the same call with the list reversed, gives the correct x labels)
```

So both the 2.0.0 feature ("ability to set x and y axes simultaneously") and the multi-axes `ax=fig`/`ax=[...]` form are silently wrong whenever an earlier axis has a much smaller range than the last one; single-axis, single-`axis` calls (`tests/test_plotting.py:88`) are unaffected, which is why this is untested. Note also that `axis='both'` is **not accepted at all** by `commaticks()` (it raises `ValueError: Axis must be x, y, or z` from line 756), so the list form is the only way to hit it within one axes. `sc.SIticks()` is immune because its formatter does not consult the axis at all.

**Fix**: bind the axis per formatter instead of closing over the loop variable — make `commaformatter` a factory (`def make_formatter(thisaxis): def f(x, pos=None): ...; return f`) and install `mpl.ticker.FuncFormatter(make_formatter(thisaxis))`, or read the interval from the formatter's own `self.axis` inside a small `Formatter` subclass.

### 5. `sc.savefigs()` raises `TypeError: ... not filter` for every call that omits `filename` — `sc_plotting.py:1505`

`keyforfilename = filter(str.isalnum, str(key))` builds a lazy `filter` object and never joins it into a string, and that object is then handed to `sc.makefilepath(default=...)`. The branch is reached whenever the `if filename and nfigs==1` test fails, i.e. for *any* call without an explicit `filename` — including the default `filetype='singlepdf'` path, which is the function's headline documented use.

```python
import matplotlib.pyplot as plt, sciris as sc
f1 = plt.figure(); plt.plot([0,1,2])
f2 = plt.figure(); plt.plot([2,1,0])
sc.savefigs([f1, f2])            # first docstring example: "Save everything to one PDF file"
```

Actual:

```
  File "/home/cliffk/sc/sciris/sciris/sc_fileio.py", line 982, in makefilepath
    basename = os.path.basename(filename)
TypeError: expected str, bytes or os.PathLike object, not filter
```

`sc.savefigs(f1)` (a single figure, no filename) fails identically, as does `sc.savefigs([f1,f2], filetype='png')` and `sc.savefigs([f1,f2], filetype='fig')`. Three of the four examples in the function's own docstring therefore crash; only the one that passes `filename=` works. The `singlepdf` case additionally leaves the `PdfPages` object unclosed on the way out (it prints `PDF saved to .../figures.pdf` and then raises), though on this Matplotlib the partially written file is removed rather than left truncated (`figures.pdf exists: False`).

**Fix**: `keyforfilename = ''.join(filter(str.isalnum, str(key)))`, and wrap the loop so `pdf.close()` and the `plt.ion()` restore run in a `finally`. Once this is fixed, multi-figure calls without `filename` produce distinct names: `sc.odict.promote([f1,f2])` gives keys `'Key 0'`, `'Key 1'`, which the `isalnum` filter turns into the distinct `'Key0'`, `'Key1'`.

### 6. `sc.savefigs()` writes the current figure rather than the figure it was given — `sc_plotting.py:1522`

The image-saving branch calls the pyplot-level `plt.savefig(fullpath, ...)`, which writes `plt.gcf()`, instead of the figure `plot` that the loop is iterating over; the preceding `reanimateplots(plot)` does not make `plot` current (verified: `plt.gcf().number` is unchanged by it), so whenever the requested figure is not the active one the wrong image is written and no error is raised.

```python
import matplotlib.pyplot as plt, numpy as np, sciris as sc, PIL.Image
f1 = plt.figure(); plt.plot([0,1,2],[0,1,2])   # rising
f2 = plt.figure(); plt.plot([0,1,2],[2,1,0])   # falling; f2 is now current
sc.savefigs(f1, filetype='png', filename='a.png')
f1.savefig('ref1.png', dpi=200, bbox_inches='tight')
f2.savefig('ref2.png', dpi=200, bbox_inches='tight')
g = lambda p: np.asarray(PIL.Image.open(p).convert('RGB'))
print('a.png == f1?', np.array_equal(g('a.png'), g('ref1.png')), '| == f2?', np.array_equal(g('a.png'), g('ref2.png')))
```

Actual: `a.png == f1? False | == f2? True`. Expected: the file should contain `f1`, the figure that was passed. The defect hides in ordinary use because the figure you have just created is usually also the current one; it bites exactly when you collect several figures and save them afterwards. The same line makes the multi-figure case doubly wrong: the loop writes `plt.gcf()` every iteration, so all N "different" images are byte-identical. Blast radius: `reanimateplots()` is called only from `savefigs()` and `loadfig()` within Sciris; `savefigs()` is not called elsewhere in the package.

**Fix**: call `plot.savefig(fullpath, **defaultsavefigargs)` instead of `plt.savefig(...)` (and correspondingly `plt.close(plot)` is already figure-explicit, so no other change is needed).

### 7. `sc.animation.addframe(fig)` treats the figure as an artist, producing a blank animation — `sc_plotting.py:1968`

The branch test is `isinstance(fig, (list, mpl.artist.Artist))`, and `matplotlib.figure.Figure` *is* a subclass of `Artist` (`isinstance(plt.figure(), mpl.artist.Artist)` is `True`). So passing a figure explicitly — which `addframe`'s own docstring calls the typical case ("typically a figure object"), which the `fig` parameter exists for, and which `anim += fig` (`__add__`) does — skips the "render a snapshot to disk" path entirely and appends the `Figure` object to `self.frames` as if it were a line or patch. `ArtistAnimation` then toggles figure visibility, which renders nothing.

```python
import matplotlib.pyplot as plt, sciris as sc, numpy as np, subprocess, glob, os, PIL.Image
anim = sc.animation(verbose=False, dpi=50)
for i in range(3):
    fig = plt.figure(); plt.plot([0,1],[0,i+1]); plt.title(f'A{i}')
    anim.addframe(fig)
print('n_files', anim.n_files, 'n_frames', anim.n_frames)
anim.save('A.mp4', engine='matplotlib', verbose=False)
subprocess.run(['ffmpeg','-v','quiet','-y','-i','A.mp4','fr_%02d.png'])
print([len(np.unique(np.asarray(PIL.Image.open(p).convert('RGB')).reshape(-1,3), axis=0)) for p in sorted(glob.glob('fr_*.png'))])
```

Actual: `n_files 0 n_frames 3` and unique-colour counts per frame `[1126, 1, 1]` — frame 1 is the base figure's own static content and frames 2 and 3 are solid white. The same loop written with the no-argument form (`anim.addframe()` after redrawing one reused figure, as in the class docstring example) gives `n_files 3 n_frames 0` and `[1071, 1075, 1069]`, i.e. three genuinely different frames. So the documented per-figure API silently yields an empty movie while the no-argument API works.

**Fix**: exclude figures from the artist test, e.g. `isinstance(fig, list) or (isinstance(fig, mpl.artist.Artist) and not isinstance(fig, mpl.figure.FigureBase))`.

### 37. `sc.animation(fig=fig)` produces a static movie with the matplotlib engine — `sc_plotting.py:2106`

*Added on re-verification.* `save()` gets the figure via `fig = self._getfig()`, which returns `self.fig` whenever the documented `fig` constructor argument was given. However, the frames produced by `loadframes()` are `imshow` artists that live in a separate, closed `animfig`. `ArtistAnimation(self.fig, frames)` (line 2117) therefore toggles artists that are not in the figure being drawn, and every movie frame is just the live figure's final state. This is also the automatic fallback path whenever `ffmpeg-python` is not installed.

```python
import matplotlib.pyplot as plt, numpy as np, sciris as sc, subprocess, glob, PIL.Image
fig = plt.figure()
anim = sc.animation(fig=fig, verbose=False, dpi=50)
for i in range(3):
    plt.cla(); plt.plot([0,1],[0,i+1]); plt.title(f'E{i}'); anim.addframe()
anim.save('E.mp4', engine='matplotlib', verbose=False)
subprocess.run(['ffmpeg','-v','quiet','-y','-i','E.mp4','fe_%02d.png'])
ims = [np.asarray(PIL.Image.open(p).convert('RGB')).astype(int) for p in sorted(glob.glob('fe_*.png'))]
print([np.abs(ims[0]-im).sum() for im in ims])
```

Actual: `[0, 444, 444]` (the frames are essentially identical). Expected: three different frames; the same loop without `fig=` gives `[0, 244749, 146561]`. Setting `anim.fig = None` before `save()` also gives `[0, 244749, 146561]`, confirming the cause. As with finding 7, the output is a valid-looking movie, so nothing signals that it is wrong.

**Fix**: in `save()`, take the figure from the frames first, e.g. `try: fig = frames[0][0].get_figure() except: fig = self._getfig()`. Keep `self.fig` for `addframe()` only.

## Medium severity

### 4. `ScirisDateFormatter` labels genuine dates between 1974-08-28 and 1976-04-19 as raw year numbers — `sc_plotting.py:1134`

`format_ticks()` tries to detect a decimal-year axis with `if values.min() >= min_year and values.max() <= max_year` (`min_year=1700`, `max_year=2300`), but `values` are Matplotlib date numbers (days since 1970-01-01). Date number 1700 is 1974-08-28 and 2300 is 1976-04-19, so any date axis lying wholly inside that 20-month window is misread as a span of calendar years 1700-2300, converted with `sc.yeartodate()`, and every tick is labelled with its own date number instead of its date.

```python
import matplotlib.pyplot as plt, numpy as np, sciris as sc
fig, ax = plt.subplots()
x = sc.daterange('1975-01-01', '1975-07-01', asdate=True)
ax.plot(x, np.arange(len(x)))
sc.dateformatter()
fig.canvas.draw()
print([t.get_text() for t in ax.get_xticklabels()])
```

Actual: `['1826', '1857', '1885', '1916', '1946', '1977', '2007']`. Expected (what `style='concise'` gives for the identical axis): `['1975', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']`. The failure is silent — no warning, and the labels look plausible enough to pass a glance. `sc.dateformatter()` is the default-styled public entry point and `style='sciris'` is its default, so plotting historical time series from the mid-1970s (a perfectly ordinary thing to do with epidemiological data) silently produces a nonsense axis.

**Severity (corrected on re-verification)**: Medium rather than High. The bug is real and silent, but it only affects axes lying entirely within 1974-08-28..1976-04-19; any axis extending outside that window is formatted correctly.

**Fix**: the year-detection heuristic must not be applied to values that Matplotlib has already unit-converted to date numbers; decide from the axis converter/units (e.g. `axis.converter`), not from the magnitude of the tick values.

### 8. `sc.bar3d()` renders fully transparent bars for constant data — `sc_plotting.py:183`

`bar3d` always converts its `c` data through `_process_colors` -> `sc.vectocolor()`, and `vectocolor` divides by the range, so constant input becomes all-NaN and matplotlib maps NaN to transparent black. The bars exist but draw nothing.

```python
ax = sc.bar3d(np.ones((3,4)))
print(np.unique(ax.collections[0].get_facecolor(), axis=0))
# actual:   [[0. 0. 0. 0.]]                                  (alpha 0 -> invisible)
# expected: a single opaque colour, as for any other constant field
ax2 = sc.scatter3d(np.ones((3,4)))
print(np.unique(ax2.collections[0].get_facecolor(), axis=0))
# actual:   [[0.267004 0.004874 0.329415 1.]]                (fine -- scatter3d lets matplotlib normalise)
```

`RuntimeWarning: invalid value encountered in divide` is emitted from `sc_colors.py:275` during the call. Root cause is `sc.vectocolor()`'s handling of a zero range (in `sc_colors.py`, outside this region); the reason `bar3d` is affected and `scatter3d` is not is that `bar3d` pre-converts to RGBA at `sc_plotting.py:452` while `scatter3d` passes the raw numbers to `ax.scatter`. A `dz` that is constant (a common case: uniform bar heights) triggers it too, since `c` defaults to the heights.

**Fix**: primarily, make `sc.vectocolor()` return the midpoint colour (or the low end) when `max == min`; defensively, `_process_colors` could detect a zero-range input and fall back to a single colour.

### 9. `sc.plot3d(c=<array>)` raises `ValueError`; the docstring example fails — `sc_plotting.py:244`

`if c == 'index':` compares a possibly-array `c` against a string, so any array-valued `c` produces an element-wise boolean array and `if` raises. This is the second example in `plot3d`'s own docstring and the feature advertised as "*New in version 3.1.0:* Allow multi-colored line".

```python
n = 100
x = np.array(sorted(np.random.rand(n))); y = x + np.random.randn(n); z = np.random.randn(n)
c = np.arange(n)
fig = plt.figure()
sc.plot3d(x, y, z, c=c, fig=fig)
# actual: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

A list `c` of the right length does not raise but silently falls through to the single-colour branch (`sc.isarray([0,1,...])` is False), where matplotlib then rejects it: `ValueError: [0, 1, 2, ...] is not a valid value for color`. So the only way to get a multi-coloured line is the default `c='index'`.

**Fix**: guard the comparison with the type: `if isinstance(c, str) and c == 'index':`. Two cautions (corrected on re-verification; the original fix was incomplete):

- **List coercion.** The original fix said to coerce a list `c` with `sc.toarray()` before the `sc.isarray(c)` test at line 246. Done unconditionally, that would misread an RGB/RGBA list such as `[1,0,0]` as per-segment data when `n` is 3 or 4. Only coerce when `len(c) in [n, n-1]` and the elements are scalars.
- **Length `n-1`.** The code explicitly allows `len(c) in [n, n-1]`, but `_process_colors()` asserts `len(c) == len(z)`, so after the type guard is added `sc.plot3d(x, y, z, c=np.arange(n-1))` still fails with `AssertionError`. The fix should also pass a length-matched `z` (or skip the assertion) for that case.

(Note that the `# pragma: no cover` on line 246 is misplaced: that block is the default code path.)

### 10. `sc.bar3d(z=...)` raises `ValueError` because the bar base keeps the unflattened shape — `sc_plotting.py:449`

After `_process_2d_data(..., flatten=True)` has flattened `x`, `y` and the heights to 1-D, the base is built from the *original* argument: `z_base = np.zeros_like(z)`, which is still `(ny, nx)`. `ax.bar3d` then broadcasts a `(20,)` x against a `(5,4)` z and fails. The documented keyword form of the primary use case ("z (arr): 2D array of z coordinates; interpreted as the heights of the bars unless `dz` is also provided") is therefore unusable:

```python
data = np.random.rand(5,4)
sc.bar3d(data)          # works
sc.bar3d(z=data)        # ValueError: shape mismatch: objects cannot be broadcast to a single shape.
                        #   Mismatch is between arg 0 with shape (20,) and arg 2 with shape (5, 4).
sc.bar3d(x=10*np.arange(4), y=np.arange(5)+10, z=data)   # same ValueError
```

The positional form only works by accident: it puts the data in `x`, leaving the local `z` as `None`, and `np.zeros_like(None)` happens to yield the 0-d object array `array(0, dtype=object)`, which broadcasts. Two related failures from the same block: `sc.bar3d(z=[[1.,2],[3,4]], dz=np.ones((2,2)))` raises `AttributeError: 'list' object has no attribute 'flatten'` at `sc_plotting.py:440` (`z_base = z.flatten()`) although a bare list `z` is accepted when `dz` is omitted, and `sc.bar3d(data, dz=data)` — data positional, which the `x` docstring explicitly permits ("or z-coordinate data if 2D and `z` is None") — raises `AssertionError: Cannot handle x and y axes with different array shapes`.

**Fix**: build the base from the processed heights, i.e. move the block after `_process_2d_data` and use `z_base = np.zeros_like(z_height)`; coerce with `np.asarray(z)` before `.flatten()` at line 440; and let the positional-data-plus-`dz` case perform the same x/z swap that `_process_2d_data` does.

### 13. `sc.figlayout(True)` and `sc.figlayout(False)` raise `TypeError` — `sc_plotting.py:889`

The two-line swap is written in the wrong order:

```python
if isinstance(fig, bool):
    fig = None
    tight = fig # To allow e.g. sc.figlayout(False)
```

`fig` has already been set to `None`, so `tight` becomes `None` instead of the boolean, and `layout = ['none', 'tight'][tight]` raises. The comment states the intent explicitly, so the documented positional form is entirely broken.

```python
plt.figure()
sc.figlayout(False)   # actual: TypeError: list indices must be integers or slices, not NoneType
sc.figlayout(True)    # actual: TypeError: list indices must be integers or slices, not NoneType
```

**Fix**: `tight = fig; fig = None` (or `tight, fig = fig, None`).

### 15. `sc.dateformatter(style=<Formatter>)` can never be recognised — `sc_plotting.py:1236`

`style = str(style).lower()` converts the argument to a string *before* the `elif isinstance(style, mpl.ticker.Formatter)` test, so that branch is unreachable and the documented option ("options are 'sciris', 'auto', 'concise', or a Formatter object") always falls through to the error.

```python
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, sciris as sc
fig, ax = plt.subplots()
ax.plot(sc.daterange('2021-01-01', '2021-03-01', asdate=True), np.arange(60))
sc.dateformatter(style=mpl.dates.DateFormatter('%Y/%m/%d'))
```

Actual: `ValueError: Style "<matplotlib.dates.dateformatter object at 0x780b21a3d450>" not recognized; must be one of "sciris", "auto", or "concise"`. Expected: the supplied formatter installed on the axis. Unknown *string* styles are correctly caught (`sc.dateformatter(style='bogus')` raises the same, appropriate, `ValueError`).

**Fix**: move the `isinstance(style, mpl.ticker.Formatter)` test above the `str(...).lower()` coercion.

### 16. `sc.dateformatter(axis='y')` also replaces the x-axis formatter, corrupting the x labels — `sc_plotting.py:1268`

The function correctly installs the locator and formatter on the axis selected by `axis=`, and then unconditionally executes `ax.xaxis.set_major_formatter(formatter)` at the end (a leftover duplicate of the `axis=='x'` case). With `axis='y'` the numeric x-axis is therefore given a date formatter driven by the y-axis's date locator. The same oversight applies to the limit handling just above it: `start`/`end` always go through `ax.get_xlim()`/`ax.set_xlim()`, so they cannot be used to bound a `y` date axis.

```python
import matplotlib.pyplot as plt, numpy as np, sciris as sc
fig, ax = plt.subplots()
y = sc.daterange('2021-01-01', '2021-06-01', asdate=True)
ax.plot(np.arange(len(y)), y)
sc.dateformatter(ax=ax, axis='y')
fig.canvas.draw()
print('x formatter:', type(ax.xaxis.get_major_formatter()).__name__)
print('x labels:', [t.get_text() for t in ax.get_xticklabels()][:3])
print('y labels:', [t.get_text() for t in ax.get_yticklabels()][:3])
```

Actual: `x formatter: ScirisDateFormatter`, `x labels: ['Dec-12\n1969', 'Jan-01\n1970', 'Jan-21']`, `y labels: ['Jan\n2021', 'Feb', 'Mar']`. Expected: the y-axis formatted as dates and the x-axis left as plain numbers `0, 25, 50, ...`.

**Fix**: delete the trailing `ax.xaxis.set_major_formatter(formatter)` (the axis-specific `axis.set_major_formatter(formatter)` above already did the work), and take the limits from `axis.axes.get_xlim()`/`get_ylim()` according to the selected axis.

### 17. `sc.savefig()` records no caller, or the wrong caller, in `calling_info` — `sc_plotting.py:1422` (root cause `sc_versioning.py:451`)

**Same bug as `sc_versioning_bugfixes.md` #2.** The symptom shows up in `savefig()`, but the off-by-one frame count lives in `sc.metadata()`, and it should be fixed there once for both functions.

`savefig()` documents `relframe=0` as "the file calling `sc.savefig()`". It calls `sc.metadata(relframe=relframe+1)`, and that `+1` is correct under `metadata()`'s own documented convention ("if used directly, use 0; if called by another function, use 1"). The defect is one level down: `sc.metadata()` then calls `getcaller(relframe=relframe+1, tostring=False)` at `sc_versioning.py:451`. `sc.getcaller()`'s base `frame=2` is documented as "the default assuming it is being called directly", i.e. called from inside a function to find *that function's* caller, so the extra `+1` in `metadata()` takes it one frame too far out. As a result, `sc.metadata()` called directly records `N/A` (verified), a module-level `sc.savefig()` records `N/A` (the stack index is out of range and `getcaller`'s bare `except` swallows the `IndexError`), and a `savefig()` inside a function records the *caller's caller*.

```python
# helper.py
import sciris as sc
def plot_and_save(fn):
    sc.savefig(fn, verbose=False)   # helper.py line 3

# main.py
import matplotlib.pyplot as plt, sciris as sc, helper
plt.plot([1,3,7])
def user_code():
    helper.plot_and_save('h.png')   # main.py line 5
user_code()
print(sc.loadmetadata('h.png')['calling_info'])

plt.plot([1,3,7])
sc.savefig('g.png', verbose=False)  # module level
print(sc.loadmetadata('g.png')['calling_info'])
```

Actual: `{'filename': '.../main.py', 'lineno': 5}` for the first (documented answer: `helper.py`, line 3) and `{'filename': 'N/A', 'lineno': 'N/A'}` for the second. Expected: `helper.py:3` and `main.py:<the savefig line>` respectively. Passing `relframe=-1` gives the documented result in both cases, confirming a constant off-by-one. `git_info` survives because it is derived from the working directory, not from `calling_info`; the caller file and line, which are the point of the feature, do not. `tests/test_plotting.py` calls `sc.loadmetadata()` and pretty-prints it but never asserts on `calling_info`, which is why this has gone unnoticed.

**Fix**: in `sc_versioning.py:451`, change `getcaller(relframe=relframe+1, tostring=False)` to `getcaller(relframe=relframe, tostring=False)`. This fixes both `sc.metadata()` and `sc.savefig()`; `savefig()` itself needs no change. **Do not change `getcaller()`'s default `frame`.** An earlier version of this finding proposed changing `getcaller`'s base frame to 1; that would break `getcaller`'s documented semantics and its docstring examples (e.g. `sc.getcaller(frame=3)`), and a module-level `sc.getcaller()` returning `N/A` is by design. The earlier fallback (`relframe=relframe` in `savefig()`) would fix `savefig()` locally but leave `sc.metadata()` broken.

### 18. `sc.savefig()` reports success for SVG but the metadata can never be read back — `sc_plotting.py:1430`

The docstring states "Metadata can be stored and retrieved for PNG or SVG", and the SVG branch does write the metadata (Matplotlib puts it in `<dc:subject><rdf:Bag><rdf:li>sciris_metadata={...}`). But `sc.metadata(tostring=True)` produces pretty-printed, multi-line JSON, and `sc.loadmetadata()`'s SVG reader is line-based (`sc_versioning.py:571-583`): it finds the line containing `sciris_metadata=`, then slices `line[line.find(flag)+len(flag):line.find(end)]` with `end='</'`. That line is only `<rdf:li>sciris_metadata={`, `line.find('</')` returns `-1`, and the slice yields the empty string.

```python
import matplotlib.pyplot as plt, sciris as sc
plt.plot([1,3,7])
sc.savefig('h.svg', comments='hi', verbose=False)   # reports success
sc.loadmetadata('h.svg')
```

Actual: `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`. Expected: the same dict that `sc.loadmetadata('h.png')` returns (which does work, `comments='hi'` included). PDF at least fails honestly (`ValueError: Filename "ex.pdf" has unsupported type... (PDF is not supported)`), matching the docstring; SVG is the case that claims to work and does not.

**Fix**: either write the SVG payload as single-line JSON (`tostring` without indentation) or make the SVG reader join the text between `sciris_metadata=` and the closing `</rdf:li>` across lines.

### 19. `sc.savefigs()` silently writes every figure to the same file when `filename` is given — `sc_plotting.py:1502`

The `if filename and nfigs==1` test only *chooses* which default name to compute; in the `else` branch `sc.makefilepath(filename=filename, ...)` still prefers the supplied `filename` over the generated `default`, so with N>1 figures every iteration resolves to the identical path. Each figure overwrites the previous one, the returned list contains N copies of the same path, and nothing is reported.

```python
import matplotlib.pyplot as plt, sciris as sc, os
f1 = plt.figure(); plt.plot([0,1,2]); f2 = plt.figure(); plt.plot([2,1,0])
out = sc.savefigs([f1, f2], filetype='png', filename='c.png')
print([os.path.basename(o) for o in out], 'distinct files:', len({str(o) for o in out}))
```

Actual: `['c.png', 'c.png'] distinct files: 1`. Expected: two distinct files (the docstring says `filename` "only uses path if multiple files", i.e. the basename should be discarded when N>1).

**Fix**: when `nfigs > 1`, use only the directory part of `filename` and append the generated per-figure name, and warn if a generated path already exists.

### 20. `sc.orderlegend()` crashes on an array `order`, which is what its own docstring recommends — `sc_plotting.py:1684`

The guard is `if order:`. The documented type is "list or array ... as from e.g. `np.argsort()`", and truth-testing a length>1 numpy array raises.

```python
import matplotlib.pyplot as plt, numpy as np, sciris as sc
fig, ax = plt.subplots()
for l in 'ABCD': ax.plot([1,2], label=l)
sc.orderlegend(np.argsort([3,1,2,0]))
```

Actual: `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`. Expected: the legend reordered to `D, B, C, A`. The equivalent list `sc.orderlegend([3,1,2,0])` works. The same falsy-guard shape also makes `order=[0]`-style single-element intent fine but silently ignores an empty `order`.

**Fix**: `if order is not None:`.

### 21. `sc.movelegend()`'s `invisible` argument does nothing, and the destination axes is blanked even when it has data — `sc_plotting.py:1806`

`invisible` appears only in the signature and the docstring; `grep -n invisible sc_plotting.py` finds it at line 1696 (signature) and 1708 (docstring) and nowhere in the body. The body instead tests `if not ax2.artists: ax2.axis('off')`, but `Axes.artists` never contains `Line2D`, `Collection`, `Patch` or `Image` children — those live in `ax.lines`, `ax.collections`, `ax.patches`, `ax.images` — so the test is essentially always true and the destination axes is switched off unconditionally, contradicting the documented "if no artists are in the destination axes".

```python
import matplotlib.pyplot as plt, numpy as np, sciris as sc
fig, axs = plt.subplots(1, 2)
for l in 'ABCD': axs[0].plot([1,2], label=l)
axs[1].plot([1,2,3], 'k-')                       # destination axes has data
sc.movelegend(axs[0], axs[1], invisible=False)   # explicitly asking not to blank it
print('ax2.artists:', list(axs[1].artists), '| ax2.lines:', len(axs[1].lines), '| axison:', axs[1].axison)
```

Actual: `ax2.artists: [] | ax2.lines: 1 | axison: False`. Expected: `axison: True`, both because the axes has a line in it and because `invisible=False` was requested. The visual consequence is measurable: rendering the same figure with and without the `movelegend()` call gives 4226 vs 2523 dark pixels, i.e. the frame, ticks and tick labels of the destination axes disappear (the line itself is still drawn, since child artists are drawn regardless of `axison`). Handle/label pairing itself is correct — `sc.movelegend()` returns `[('A','A'), ('B','B'), ('C','C'), ('D','D')]`.

**Fix**: honour the argument, and test for real content, e.g. `if invisible and not (ax2.lines or ax2.collections or ax2.patches or ax2.images or ax2.artists): ax2.axis('off')`.

### 22. `sc.animation`'s `imagefolder` is missing from the name template handed to `ffmpeg` — `sc_plotting.py:2085`

`_getfilename()` writes frames to `self.imagefolder / (self.nametemplate % n)`, but the default `engine='ffmpeg'` branch passes the bare `self.imagefolder`-less `self.nametemplate` to `ffmpeg.input()`. With the default `imagefolder=os.getcwd()` the two happen to agree; with an explicit folder, or with the documented `imagefolder='tempfile'`, ffmpeg is pointed at a path that does not exist (or, worse, at same-named leftovers in the current directory).

```python
import matplotlib.pyplot as plt, sciris as sc, os
os.makedirs('imgs', exist_ok=True)
a = sc.animation(verbose=False, dpi=50, imagefolder='imgs'); plt.figure(); a.addframe()
print('written to', str(a.filenames[0]), '| ffmpeg.input() gets', a.nametemplate, '| exists?', os.path.exists(a.nametemplate % 0))
```

Actual: `written to imgs/animation_0000.png | ffmpeg.input() gets animation_%04d.png | exists? False`. **Caveat, stated plainly**: `ffmpeg-python` is not installed in this environment (`import ffmpeg` -> `ModuleNotFoundError`), so `save(engine='ffmpeg')` falls back to Matplotlib with a warning and the `ffmpeg.input()` call itself could not be executed. The path mismatch shown above *was* executed; the consequence for ffmpeg is inferred from it.

**Fix**: pass `str(self.imagefolder / self.nametemplate)` to `ffmpeg.input()`.

### 24. `sc.savemovie()`'s `interval` argument has no effect on the saved movie — `sc_plotting.py:2223`

`interval` is documented as "The interval between frames; alternative to using fps", but the conversion only runs when `interval is None`:

```python
if fps is None: fps = 10
if interval is None:
    interval = 1000./fps
    fps = 1000./interval   # no-op round trip
```

When the user supplies `interval`, `fps` keeps its default of 10 and that is what reaches `anim.save(fps=fps)`; `interval` is passed only to `ArtistAnimation`, where it governs on-screen playback and is ignored by the writers. The `fps = 1000./interval` line inside the branch is a no-op, which is a strong hint it was meant to be in the other branch.

```python
import matplotlib.pyplot as plt, numpy as np, sciris as sc, subprocess, json
plt.figure(); frames = [plt.plot(np.arange(5)+i) for i in range(6)]
sc.savemovie(frames, 'k.mp4', interval=50, verbose=False)   # 50 ms/frame == 20 fps
s = json.loads(subprocess.run(['ffprobe','-v','quiet','-print_format','json','-show_streams','k.mp4'], capture_output=True, text=True).stdout)['streams'][0]
print(s['r_frame_rate'])
```

Actual: `10/1`. Expected: `20/1`. `savemovie()` also *prints* the wrong number in its progress message (`Saving 8 frames at 10 fps ...` for `interval=50`), so the report agrees with the wrong behaviour rather than the request.

**Fix**: compute `fps = 1000./interval` when `interval` is supplied and `fps` is not, and error if both are given inconsistently.

### 36. `sc.datenumformatter()` without `start_date` labels a real date axis about 51 years in the future — `sc_plotting.py:1321`

*Added on re-verification.* The docstring says `start_date` is "not needed if x-axis already uses dates". In that case the code sets `start_date = mpl.dates.num2date(ax.dataLim.x0)`, i.e. the first date in the data (for example 2021-01-01, whose date number is 18628). The formatter then computes `start_date + timedelta(days=x)` on tick values that are *already* date numbers, so the date offset is added twice.

```python
import matplotlib.pyplot as plt, numpy as np, sciris as sc
fig, ax = plt.subplots()
x = sc.daterange('2021-01-01', '2021-12-31', asdate=True)
ax.plot(x, np.arange(len(x)))
sc.datenumformatter()
fig.canvas.draw()
print([t.get_text() for t in ax.get_xticklabels()][:4])
```

Actual: `['2072-Jan-02', '2072-Mar-01', '2072-May-01', '2072-Jul-01']`. Expected: `['2021-Jan-01', '2021-Mar-01', '2021-May-01', '2021-Jul-01']`, which is what `start_date=mpl.dates.num2date(0)` gives. The failure is silent and the labels look like plausible dates. (On a purely numeric axis with no `start_date`, the labels start at 1970-01-11 for tick 0, which is equally meaningless, but there is no correct answer in that case anyway.)

**Fix**: when `start_date is None`, use the Matplotlib epoch (`start_date = mpl.dates.num2date(0)`, or `sc.date(mpl.dates.get_epoch())`) rather than `ax.dataLim.x0`. Tick values on a date axis are days since the epoch, so the labels then come out right.

## Low severity

### 11. `sc.setaxislim(data=...)` silently ignores a ragged list of arrays — the shape its own docstring uses — `sc_plotting.py:658`

The `data` block is guarded by `sc.checktype(data, 'arraylike')`, which is `False` for a list of unequal-length arrays. The whole block is then skipped, so the negative values in `data` are never seen and the bottom is clamped to 0 anyway — the exact opposite of the documented promise ("will keep Matplotlib's lower limit, since at least one data value is below 0"). The docstring's own example data, `[np.array([-3,4]), np.array([6,4,6])]`, is ragged.

```python
fig, ax = plt.subplots(); ax.plot([1., 2, 3])
data = [np.array([-3,4]), np.array([6,4,6])]        # docstring's data (lengths 2 and 3)
print(sc.checktype(data, 'arraylike'))              # actual: False
print(sc.setylim(data=data), ax.get_ylim())         # actual: (0, 3.1)   (0.0, 3.1)  -- data ignored
data2 = [np.array([-3,4]), np.array([6,4])]         # same values, equal lengths
print(sc.setylim(data=data2))                       # actual: (0.9, 6)   -- lower limit kept, upper raised to 6
```

Expected for the ragged case is the same `(0.9, 6)` behaviour as the equal-length case. Note the failure direction is the dangerous one: the caller passes data specifically to protect the negative part of the plot and gets it clipped instead, with no warning.

**Severity (corrected on re-verification)**: Low rather than Medium.

**Fix**: flatten with an explicit concatenation that tolerates ragged input (e.g. `np.concatenate([sc.toarray(d).flatten() for d in sc.tolist(data)])`) rather than gating on `sc.checktype(data, 'arraylike')`, and use `np.nanmin`/`np.nanmax` so NaN-containing data are handled rather than silently treated as 0 (`min(0, np.nan)` returns `0` today).

### 29. `start=0` / `end=0` are silently ignored by both date formatters — `sc_plotting.py:1259`, `sc_plotting.py:1330`

Both functions guard with `if start:` / `if end:`. Both document `start`/`end` as `str/int`, and for `sc.datenumformatter()` the integer `0` means "the start date" — a completely legitimate lower bound that is silently discarded.

```python
import matplotlib.pyplot as plt, numpy as np, sciris as sc
fig, ax = plt.subplots(); ax.plot(np.arange(365), np.arange(365))
print('before:', ax.get_xlim())
sc.datenumformatter(start_date='2021-01-01', start=0)
print('after: ', ax.get_xlim())
```

Actual: `before: (-18.2, 382.2)` / `after:  (-18.2, 382.2)`. Expected: `(0.0, 382.2)`. Non-zero ints work (`start=10 -> (10.0, 382.2)`, `end=50 -> (-18.2, 50.0)`), which refutes the suggestion that `start`/`end` reject ints outright: integers are accepted, only `0` is dropped.

**Fix**: `if start is not None:` / `if end is not None:`.

### 35. `sc.animation.save()`'s `verbose` default overrides `sc.animation(verbose=False)` — `sc_plotting.py:2049`

`save()` declares `verbose=True` rather than `verbose=None`, so the instance-level setting documented on the class ("verbose (bool): whether to print progress") cannot suppress the save-time output; `fps`, `dpi` and `tidy` in the same block *are* correctly defaulted with `if x is None: x = self.x`.

```python
import matplotlib.pyplot as plt, sciris as sc
a = sc.animation(verbose=False, dpi=50); plt.figure()
for i in range(2): plt.cla(); plt.plot([0,i]); a.addframe()
a.save('q.mp4', engine='matplotlib')
```

Actual output despite `verbose=False`: `Saving 2 frames at 10 fps and 50 dpi to "q.mp4"...`, a progress bar, `Done; movie saved to "q.mp4"`, `File size: 4 KB`, `Time saving movie: 76.8 ms`.

**Fix**: `verbose=None` in the signature and `if verbose is None: verbose = self.verbose`.

### 38. `sc.movelegend()` does not preserve `ncol` — `sc_plotting.py:1743`

*Added on re-verification.* The docstring promises to move the legend "preserving properties". `get_legend_props()` maps `'_ncol'`, but Matplotlib renamed that attribute to `_ncols` in 3.6 (on 3.11.1, `hasattr(leg, '_ncol')` is `False` and `hasattr(leg, '_ncols')` is `True`). The column count is therefore silently dropped, while `loc`, `title` and `frameon` are preserved.

```python
import matplotlib.pyplot as plt, sciris as sc
fig, axs = plt.subplots(1, 2)
for l in 'ABCD': axs[0].plot([1,2], label=l)
axs[0].legend(ncol=2, loc='upper left', title='T', frameon=False)
new = sc.movelegend(axs[0], axs[1])
print(new._ncols)
```

Actual: `1`. Expected: `2`.

**Fix**: `mapped_attrs = {'_ncols':'ncols', '_loc':'loc', 'get_frame_on':'frameon'}`; for compatibility with Matplotlib <3.6, try `_ncols` and fall back to `_ncol`.

### 39. `sc.dateformatter(style='auto', dateformat=...)` raises `TypeError` — `sc_plotting.py:1225-1240`

*Added on re-verification.* A string or list `dateformat` is converted to `kwargs['formats']` and `kwargs['zero_formats']`, which are `ConciseDateFormatter` arguments. These kwargs are passed unconditionally to whichever formatter is chosen, and `AutoDateFormatter` accepts neither, although both `style='auto'` and `dateformat` are documented.

```python
import matplotlib.pyplot as plt, numpy as np, sciris as sc
fig, ax = plt.subplots()
ax.plot(sc.daterange('2021-01-01','2021-03-01',asdate=True), np.arange(60))
sc.dateformatter(style='auto', dateformat='%m/%d')
```

Actual: `TypeError: AutoDateFormatter.__init__() got an unexpected keyword argument 'formats'`. Expected: an `AutoDateFormatter` using `%m/%d`. `style='sciris'` and `style='concise'` work (`['01/01\n2021', '01/08', ...]` and `['01/01', '01/08', ...]`).

**Fix**: for `style in ['auto', 'matplotlib']`, map a string `dateformat` to `defaultfmt=dateformat` and drop `formats`/`zero_formats` from the kwargs.

## Cross-checked between halves

Two claims were tested from two directions, which is worth recording separately since it shows which findings were independently corroborated rather than taken on a single auditor's word.

**`dateformatter`'s `start`/`end` handling of integers.** The first half's initial hypothesis was that `start`/`end` reject integers outright. The second half's own execution (finding 29) refutes this directly: non-zero integers work fine (`start=10`, `end=50` both move the axis limit as expected); the actual defect is narrower and worse in one way — the falsy-sentinel guard `if start:`/`if end:` drops only the specific value `0`, which for `datenumformatter()` is the legitimate and meaningful "start date" value.

**The blank 3-D figure from `sc.colormapdemo()`.** The `sc_colors.py` audit observed, from the outside, that `sc.colormapdemo()`'s `'3d'` output figure comes back blank and emits a colorbar-mismatch warning. The first half of this audit traced that symptom to its root cause here: `sc.fig3d()` (finding 1) builds two figures because `ax3d()`'s `figkwargs`-truthiness guard fires even on a no-argument call, and `colormapdemo()`'s `fig2, ax2 = sc.fig3d(returnax=True, figsize=(12,8))` at `sc_colors.py:681` hits it on every invocation. The two audits' observations agree exactly: `colormapdemo()`'s own `fig2.colorbar(surf)` is the emitter of `UserWarning: Adding colorbar to a different Figure`.

## Misplaced `# pragma: no cover`

| Line | Construct | Reachable via |
|------|-----------|-----------------------------------|
| 47 | `if returnax:` in `fig3d` | Body (line 48) executes on every `sc.colormapdemo()` call, which is in `tests/test_colors.py:83` |
| 246 | multi-colour branch in `plot3d` | Body (247-250) executes on the *default* `c='index'` path; `sc.plot3d(x,y,z)` draws 9 line segments |
| 509 | `if values.ndim == 1:` in `stackedbar` | Body (510) executes for `sc.stackedbar(np.array([1.,2,3]))` |
| 512 | `if transpose:` in `stackedbar` | Body (513) executes for `tests/test_plotting.py:60` |
| 528 | label-count validation in `stackedbar` | Body (529-530) executes whenever `labels` is passed, i.e. the docstring example and `tests/test_plotting.py:56` |
| 547 | `else: label = None` in `stackedbar` | Body (548) executes on the default `labels=None` path |
| 637, 642 | `which is None` / `which == 'both'` in `setaxislim` | Bodies (638, 643-645) execute for the documented bare `sc.setaxislim()` |
| 658 | `if sc.checktype(data, 'arraylike'):` in `setaxislim` | Body (659-661) executes for any `data=` call, e.g. `sc.setylim(data=[1.,2])` |
| 692 | `def _get_axlist(ax):` (whole function excluded) | Body (696) executes via `sc.commaticks()` / `sc.SIticks()`, both in `tests/test_plotting.py:88,92` |
| 735 | `def commaformatter(...)` (whole closure excluded) | Body (736-738) executes on every draw of a `commaticks`-formatted axis |
| 778 | `def SItickformatter(...)` (whole closure excluded) | Body (780) executes on every draw of an `SIticks`-formatted axis |
| 837, 839 | `nrows is not None` / `ncols is not None` in `getrowscols` | Bodies (838, 840) execute for the documented `sc.getrowscols(10, nrows=3)` |
| 849 | `if make:` in `getrowscols` | Body (850-851) executes for the docstring example `sc.getrowscols(37, make=True)` |
| 900, 907 | `if not keep:` / `if len(kwargs):` in `figlayout` | Bodies (901, 903-904, 908) execute for the docstring example `sc.figlayout(bottom=0.3)` |
| 1127 | `ScirisDateFormatter.format_ticks()` (whole function excluded) | Produces *every* tick label on the default `style='sciris'` date axis; finding 4 (and rejected finding 14) live inside it |
| 1326 | `datenumformatter()`'s inner `formatter` | Produces every label the function exists to produce; ran on all of `sc.datenumformatter(start_date='2021-01-01')`'s ticks |
| 1335 | `datenumformatter()` `if interval:` | Exercised verbatim by the function's own second docstring example (`interval=7`), which ran correctly |
| 1430 | `savefig()` SVG/PDF metadata branch | Taken by `sc.savefig('h.svg')` and `sc.savefig('ex.pdf')`; the SVG round-trip finding above lives here |
| 1492 | `savefigs()` `singlepdf` branch | This is `savefigs()`'s *default* `filetype`; taken by `sc.savefigs([f1,f2])` |
| 1505 | `savefigs()` generated-filename branch | Taken by every `savefigs()` call that omits `filename`, i.e. the default; it is where the `filter` `TypeError` is raised |
| 1517, 1522 | `savefigs()` image-saving branch | Taken by any `filetype` other than `'fig'`; it is where the wrong-figure bug is |
| 1530 | `savefigs()` `if aslist or len(filenames)>1` | Taken by `sc.savefigs([f1,f2], filetype='png', filename='c.png')` |
| 1968 | `animation.addframe()` artist branch | Taken by `anim.addframe(fig)` for any figure, since `Figure` is an `Artist` |
| 2089 | `animation.save()` `engine == 'matplotlib'` | The automatic fallback whenever `ffmpeg-python` is absent, which it is in a default Sciris install (`ffmpeg-python` is not a Sciris dependency); every `anim.save()` executed in this audit took it |

The whole-function/whole-closure exclusions on `_get_axlist`, `commaformatter`, `SItickformatter` (692, 735, 778) and on `ScirisDateFormatter.format_ticks()` (1127) are the most significant: they hide exactly the tick-formatting logic where the `commaticks` closure bug (finding 3) and the `ScirisDateFormatter` date-magnitude bug (finding 4; also rejected finding 14) live.

## Verified clean

Recorded so the same ground isn't re-covered. All of the following were hypothesised, tested by execution, and found correct.

**3-D plumbing (`ax3d()` combinations).** For every combination that does not involve a pre-populated foreign figure, `ax.figure is fig` holds and no figure leaks: `ax3d()`, `ax3d(figkwargs=dict(figsize=...))`, `ax3d(2,2,3)`, `ax3d(221)` (the "111 format" parsing at lines 84-88 correctly expands `221` to nrows=2, ncols=2, index=3), `ax3d(fig=<empty figure>)` (reuses it, `fig is f0`), `ax3d(fig=<figure with one 3-D axes>)` (returns the existing axes), `ax3d(ax=a0)` and `ax3d(ax=a0, figkwargs=...)` (the `ax` argument correctly suppresses new-figure creation, `plt.get_fignums()` stays `[1]`), and `ax3d(fig=True)`. `ax3d` into a figure whose current axes are 2-D raises the intended `ValueError: Cannot create 3D plot into axes ...`. `elev`/`azim` were not exercised.

**`_process_2d_data()` orientation.** Verified with an asymmetric `z = np.zeros((3,5)); z[1,3] = 10`: the implicit-coordinate path puts the peak at `(x, y) = (3, 1)` (column index on x, row index on y), the explicit path with `x = [100..500]`, `y = [10,20,30]` puts it at `(400, 20)`, and the "2-D array passed positionally as `x`" path gives the same `(3, 1)` — so all three agree, and `surf3d`'s non-flattened output has `x`/`y`/`z` all `(3,5)` with `x[0] == [100..500]` and `y[:,0] == [10,20,30]`. `c='z'` and `c='index'` produce correctly-sized colour vectors on the flattened path (`len(c) == ny*nx`). None of `surf3d`/`scatter3d`/`bar3d` mutates the caller's `z` array. (`surf3d(..., c='index')` does raise `IndexError`, but `'index'` is documented only for `scatter3d`/`plot3d`, so it is not reported as a finding.)

**Docstring examples that do work.** `plot3d` example 1, both `scatter3d` examples, both `surf3d` examples (including `cmap='orangeblue'` with `c=z**2`), and both `bar3d` examples run clean; `sc.bar3d(dz=...)` with no `z` also works. Only `plot3d` example 2 fails (finding 9).

**`stackedbar()` geometry.** Checked from patch geometry rather than by eye, on `values = [[1,2,3,4],[10,20,30,40],[100,200,300,400]]`: segment k's `bottom` equals the cumulative sum of segments 0..k-1 exactly, and each bar's top equals the column sum (`np.allclose(tops, np.cumsum(values, axis=0))` is True). The same holds for `barh=True` (via `get_x`/`get_width`), for `transpose=True` on the transposed input, and for `is_cum=True` on `np.cumsum(values, axis=0)` — so `np.diff(values, prepend=0, axis=0)` at line 526 is right. `labels` map to series in order (`[c.get_label() for c in artists] == ['a','b','c']`) and `colors`/`labels` length mismatches raise as documented. Negative values stack as the cumulative sum (`[[1,2],[-3,-4],[5,6]]` gives bottoms `0, 1, -2` and totals `3, 4`, matching `values.sum(axis=0)`). 1-D input is promoted to one series. Supplying `x` shifts the bars (`get_x()` of `[9.6, 19.6, 29.6, 39.6]` for `x=[10,20,30,40]`). `flipud=True` reverses the series order and labels are applied after the flip, which matches the documented "flip the array upside down prior to plotting". No mutation of the caller's `values` array.

**Tick label correctness (single-axis calls).** With `commaticks()` or `SIticks()` applied to one axis, every label matches its tick position after `fig.canvas.draw()`, for all-positive (`0..3.4e4`), all-negative (`-5e4..-1e4`), sub-1 (`0..0.5`), mixed-magnitude (`1..1e6`), identical (`[0,0]`) data, and at the 999/1000, 1e6 and 1e9 boundaries — including the SI-prefix boundaries, where `SIticks` gives `999`, `1K`, `1.001K` and `999.75K`, `1M`, `1.00025M`, i.e. no off-by-1000. `cursor_precision` correctly gives the cursor more digits than the axis (`formatter(0.123456, None)` -> `0.123456`, `formatter(0.123456, 1)` -> `0.123`). `precision` has no visible effect on round tick values because trailing zeros are stripped for axis labels, which is deliberate (line 741). Applying `commaticks()` twice, or `commaticks()` then `SIticks()`, is idempotent/last-wins as expected. No global state is left behind: `plt.rcParams` is byte-for-byte unchanged after formatting a figure, and a subsequently created figure gets the default `ScalarFormatter`. `axis='both'` is not supported by either function (raises `ValueError: Axis must be x, y, or z`), but only `'x'`, `'y'`, `'z'` and lists thereof are documented; single-axis, single-`axis` calls are the only case that is currently tested, which is exactly why finding 3's order-dependent bug went unnoticed.

**Legend handle/label pairing.** Built a 4-series plot labelled A/B/C/D and, after each of `sc.orderlegend(reverse=True)`, `sc.orderlegend([1,0,3,2])`, `sc.orderlegend([2,0])` (partial), `sc.orderlegend([0])`, `sc.separatelegend()`, `sc.separatelegend(reverse=True)` and `sc.movelegend()`, compared each rendered legend text against the corresponding `legend_handles` `get_label()`; every handle still maps to its own label in all seven cases tested (e.g. reverse gives `[('D','D'), ('C','C'), ('B','B'), ('A','A')]`, not `D/C/B/A` labels over `A/B/C/D` handles). `sc.orderlegend()` reorders `handles` and `labels` with the same comprehension index and `sc.separatelegend()` reverses both together, so there is no way for them to desynchronise. Partial `order` lists correctly produce a shorter, correctly-paired legend rather than a mismatched full one; unlabelled artists are excluded by `ax.get_legend_handles_labels()` before ordering, so they cannot shift the indices; duplicate labels produce duplicate legend entries but each still sits next to its own handle. Figure bookkeeping: `sc.orderlegend()` and `sc.movelegend()` leave `plt.get_fignums()` unchanged and do not detach the source axes (`axs[0].axison` stays `True`, its lines stay in `axs[0].lines`); `sc.separatelegend()` adds exactly one figure (`[1] -> [1, 2]`), which is its documented purpose, and its `sc.cp(h)` handle copies do not mutate the originals.

**Date axis label/position correspondence.** For `style='sciris'` on ordinary date ranges, converted `ax.get_xticks()` via `mpl.dates.num2date()` and compared against the rendered label strings at six zoom levels: 5 days (`Apr-04\n2021`, `Apr-05`, ... every label matching its position exactly), 20 days, 90 days (month level: `Apr\n2021`, `May`, `Jun`, `Jul`), 400 days, 4 years (`2021`...`2025`) and 11 years (`2020`, `2022`, ... `2032`) — all six zoom levels including the year boundary matched. No off-by-one month or year at any level. The year-boundary case behaves correctly too: for 2020-11-15 + 120 days the labels are `Dec\n2020`, `Jan\n2021`, `Feb`, `Mar` — the year appears exactly on the first tick and on the tick where the year changes, never duplicated and never dropped, and since `show_offset=False` the offset text is empty (`get_offset() == ''`), so there is no possibility of the offset and the labels disagreeing about the year. `addyear()`'s substring test correctly suppresses a redundant year at the `%Y` zoom level. `style='concise'` and `style='auto'` also produce correct labels.

**`datenumformatter()` day-number arithmetic.** On integer tick positions the labels agree exactly with `sc.date(n, start_date=...)` for every tick tested over -50 to 400 (e.g. `-50 -> 2020-Nov-12`, `0 -> 2021-Jan-01`, `350 -> 2021-Dec-17`, `400 -> 2022-Feb-05`), and there is no off-by-one at day 0 vs day 1: day 0 is `start_date` and day 1 is the next day, matching `sc.date(0/1, start_date=...)` and `sc.day(start_date, start_date=start_date) == 0`. `start`/`end` accept both strings and non-zero integers and are converted through `sc.day()` consistently (`start='2021-02-01' -> xlim (31.0, ...)`, `end='2021-03-01' -> (..., 59.0)`, `start=10 -> (10.0, ...)`, `end=50 -> (..., 50.0)`); the only int that fails is `0` (finding 29). An `interval` that does not divide the range is handled correctly: the function's own docstring example `sc.datenumformatter(start_date='2020-04-04', interval=7, start='2020-05-01', end=50, dateformat='%m-%d')` produces ticks at 27/34/41/48 within `xlim (27.0, 50.0)` labelled `05-01`, `05-08`, `05-15`, `05-22` — correct, with the final partial interval simply omitted rather than mislabelled. `dateformat` is honoured. `sc.dateformatter()` correctly rejects an unknown *string* style with a clear `ValueError`.

**`savefig()` metadata and `_get_dpi()`.** PNG round-trips completely: `sc.loadmetadata()` returns all ten keys and `comments` (both a string and a dict) comes back byte-identical. `pipfreeze=True` and the legacy `freeze=True` alias both populate `md['pipfreeze']` (545 entries). PDF's inability to round-trip is documented and the raised message is accurate. JPG correctly warns under `die=False` and raises `ValueError` under `die=True`. `relframe` does shift the recorded frame by one per unit, so the argument is wired up (it is only mis-centred, per finding 17). `_get_dpi()`'s `min_dpi=200` floor does **not** override an explicit lower dpi: `_get_dpi(72) == 72`, `_get_dpi(50) == 50`, `_get_dpi(None) == 200`, and `sc.savefig('d50.png', dpi=50)` really produces a 320x240 image versus 1280x960 for `dpi=None` — the floor applies only when `dpi is None`, as documented.

**`savefigs()` / `loadfig()` / `animation` / `savemovie()` mechanics.** `sc.odict.promote([f1,f2])` yields keys `['Key 0', 'Key 1']`, which the `isalnum` filter turns into the distinct `'Key0'` and `'Key1'`, so once finding 5 is fixed the generated names will not collide. (Corrected on re-verification: the original audit claimed they would collide.) `savefigs()` does restore interactivity (`plt.isinteractive()` is `False` after both a successful and a failed call). `filetype='fig'` plus `sc.loadfig()` round-trips faithfully: title, axes and line data all survive (`title='ROUNDTRIP'`, 1 line), and `loadfig()`'s error path raises the intended `FileNotFoundError` with the intended message. `sc.emptyfig()` is a one-liner with nothing to get wrong. Animation frame collection is correct and complete for the working (no-argument `addframe()`) path: 5 frames give 5 sequentially numbered files `animation_0000.png`..`animation_0004.png`, all five MD5-distinct, none dropped or duplicated, `n_files == 5`, and `rmfiles()` removes all of them (only the `.mp4` is left in the directory). `fps`, `dpi` and frame count all reach the writer intact, confirmed with `ffprobe`: `sc.animation(dpi=50, fps=25)` (without `fig=`; see finding 37) gives `r_frame_rate 25/1`, 8 frames, 320x240; `sc.savemovie(frames, fps=20)` gives `20/1`, 8 frames, 960x720; `quality='low'` gives 320x240 and `quality='medium'` (the default) 960x720, so the quality-to-dpi mapping works. `bitrate` reaches `anim.save()` and demonstrably changes the output (81 kbps at `bitrate=200` versus 274 kbps without), consistent with the docstring's own "may be ignored" caveat, so no finding was raised there. The `anim_args`/`save_args`/`kwargs` merge order is correct (`kwargs` wins). The `nametemplate` `%`-formatting error path raises the intended clear `TypeError` for a template without a format specifier.

**Other single-purpose helpers.** `boxoff()`: verified from `spines[...].get_visible()` and per-tick `tick1line`/`tick2line`/`label1` visibility after a draw — the default removes exactly top and right (spines and their ticks/labels) and leaves left/bottom intact; `which='all'` removes all four spines and all ticks and labels; `which='top, bottom'` removes both x spines and the x ticks/labels while leaving the y axis alone; `which=['left']` (list form) and `which='left'` agree; `removeticks=False` removes the spine but keeps the ticks and labels; the argument-swap form `sc.boxoff('top, bottom')` works and the return value is the axes; an unrecognised spine name raises `KeyError`. `getrowscols()`: all four docstring claims reproduce exactly (`36 -> (6,6)`, `37 -> (7,6)`, `100, ratio=2 -> (15,7)`, `100, ratio=0.5 -> (8,13)`); `nrows*ncols >= n` holds for every `n` in 1..199 crossed with `ratio` in `{0.25, 0.5, 1, 2, 4}` (0 violations); `ratio` biases the shape monotonically as documented; `nrows=3`/`ncols=3` each fix that dimension; with `make=True`, `remove_extra=True` deletes exactly the right count in every configuration tested, `fig._n_subplots`/`fig._subplots_shape` are set correctly, and two non-bugs are worth knowing: the returned `axs` array still contains the deleted-but-detached axes objects (documented 3.2.2 behaviour), and `n=0`/`nrows=0` raise `ZeroDivisionError` rather than something friendlier. `figlayout()`'s normal (non-boolean-positional) paths are correct: `sc.figlayout()` leaves a `TightLayoutEngine`, `sc.figlayout(tight=False)` leaves a `PlaceHolderLayoutEngine`, and `sc.figlayout(bottom=0.3)` both applies `subplotpars.bottom == 0.3` and turns the engine back off, matching documented behaviour — only the boolean-positional form (finding 13) is broken. `maximize()` under `agg` prints the documented warning and returns normally, `die=True` raises `RuntimeError` wrapping the same message, and `plt.get_fignums()` is unchanged (not testable further without a GUI backend). `fonts()` listing returns 844 alphabetically-sorted unique names, `output='path'` returns a dict, `dryrun=True` adds nothing to `fontManager.ttflist` and leaves `rcParams['font.family']` untouched, and a nonexistent path is swallowed with a printed warning by default and raises the original exception type under `die=True` — only the `fullfont=True` docstring example (rejected finding 28, docstring-only) is wrong.

## Rejected on review

These findings from the original audit were removed from the severity sections and the summary table after independent re-verification. Numbers are kept so cross-references stay valid.

- **12**, `sc.setylim()` on all-negative data sets `(0, 0)`: NOT A BUG. The function is documented as equivalent to `plt.ylim(bottom=0)`, which on all-negative data also excludes every point (it inverts the axis), so the claimed expected `(-5.2, 0)` is not what the function promises.
- **14**, `ScirisDateFormatter` gives up when the first tick is 1970-01-01: NOT WORTH FIXING. It only fires when the leftmost tick is exactly the epoch; an ordinary axis spanning the epoch (1965-1975) formats correctly with no warning.
- **23**, `sc.animation.save()` mix-and-match guard: NOT WORTH FIXING. Mixing figure frames and artist frames is explicitly unsupported, and the proposed `if self.n_files and self.n_frames:` guard was harmful: evaluated after `loadframes()`, it would raise on every ordinary figure-based `save(engine='matplotlib')` (if ever fixed, the check must go before `loadframes()`).
- **25**, `ax3d(fig=False)` unreachable block: NOT WORTH FIXING. `fig=False` is not a documented value (docs say "existing figure, or True"), so this is dead code only.
- **26**, `sc.setaxislim()` docstring example and `which='both'` return: NOT WORTH FIXING. Docstring-only (the example passes data as `which`), and the `which='both'` return value is undocumented; worth a one-line docstring fix.
- **27**, `SIticks(fixed=...)` dead argument: NOT WORTH FIXING. The parameter is unused and undocumented; remove or ignore.
- **28**, `sc.fonts(fullfont=True)` docstring example: NOT WORTH FIXING. Docstring-only; change the example to `output='font'`.
- **30**, `datenumformatter()` negative fractional ticks: NOT WORTH FIXING. It needs fractional tick positions before `start_date` on an axis only a few days wide, an extreme corner case.
- **31**, `sc.savefigs()` swallows `**kwargs` / `filepath` / `varbose`: NOT WORTH FIXING. Docstring-only; the proposed "forward `**kwargs` into savefig args" fix would make the `position=` example crash, since `savefig` has no `position` argument, so just fix the docstring.
- **32**, `sc.loadfig()` turns on interactive mode: NOT A BUG. `plt.ion()` is deliberate and commented ("Without this, it doesn't show up"); displaying the loaded figure is the function's purpose.
- **33**, `reanimateplots()` docstring requires an odict: NOT WORTH FIXING. The helper is private (not in `__all__`) and all internal callers pass figures; only its docstring is wrong.
- **34**, `sc.animation.__radd__()` passes `self`: NOT WORTH FIXING. It only fires for `fig + anim`, which is undocumented and unlikely (the documented `anim += fig` uses `__add__`); a one-line fix if the code is touched anyway.

## Suggested order of work

1. **Findings 1, 2, 3, 6** — figure/axes plumbing bugs that silently redirect output to the wrong figure or mislabel an axis, all with small, local fixes (pass the figure object through instead of re-deriving it; bind the closure variable per-axis; call the figure's own `savefig`).
2. **Findings 5, 7, 37** — a wholly blank or static output (empty animation, static movie with `fig=`, crash on the default `savefigs()` call), all reachable from entirely ordinary calls.
3. **Findings 4, 8, 9, 10, 13, 20, 21, 36** — wrong-but-plausible date axes (mid-1970s dates, `datenumformatter()` on a date axis), wrong or invisible plot content, and hard crashes on documented keyword forms (constant-data transparency, `plot3d(c=array)`, `bar3d(z=...)`, `figlayout(True/False)`, `orderlegend`/`movelegend`).
4. **Findings 15, 16, 17, 18, 19, 22, 24** — narrower correctness and metadata-fidelity issues (unreachable `Formatter` branch, cross-axis contamination, mis-attributed `calling_info` (fix in `sc_versioning.py`, together with `sc_versioning_bugfixes.md` #2), unreadable SVG metadata, overwritten multi-figure saves, ffmpeg frame path, ignored `interval`).
5. **Findings 11, 29, 35, 38, 39 and the pragma list** — ragged `data`, falsy `start=0`, `verbose` default, lost legend `ncol`, `style='auto'` with `dateformat`.

Findings 1, 2, 3, 4, 6, 7, 8, 10, 36, and 37 all produce output that still looks like a legitimate plot, file, or movie rather than an error — a mislabelled tick, the wrong figure saved, invisible bars, a blank or static movie, or dates shifted by decades. These are the ones that would ship unnoticed, and each has a natural regression test (compare rendered labels against tick positions after a draw; compare saved-file bytes against the figure that was actually passed; check that a known constant color is opaque) that would be worth adding alongside the fixes.
