# `sc_asd.py` bug audit

Audit of `sciris/sc_asd.py` (the whole file: `_consistent_shape()`, `_improvement_ratio()`, `_validate_fval()`, `asd()`) for genuine defects: wrong or silently degenerate optimization results, documented arguments that don't work as documented, crashes on in-contract input, and misleading diagnostics. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps as such, and performance suggestions.

**Method**: line-by-line reading of all 356 lines, then an executed test per hypothesis. Special attention was paid to the items where an adaptive stochastic descent optimizer usually goes wrong: the paired plus/minus step indexing, whether `sinc`/`sdec`/`pinc`/`pdec` act on the quantity they name and in the direction the docstring claims, whether the probability vector stays normalized and positive, whether `xmin`/`xmax` can be violated or stall the sampler, whether each stopping criterion terminates for the reason reported in `exitreason`, whether the returned `x` is the best-ever point, reproducibility under `randseed`, and internal consistency of `details`. To confirm the algorithm itself is correct, `asd()` was compared against a hand-written reference implementation of the published algorithm (Kerr et al. 2018) on smooth convex test functions; the two agree **value-for-value on every seed tested**, so the core loop is sound and all findings below are in the surrounding machinery. Every "actual" value in this document was produced by running the code against the editable install (Sciris 3.3.0, numpy 2.4.6, Python 3.13.9, commit `2d69aad`), and every finding was reproduced a second time from a minimal snippet in a fresh interpreter before being recorded.

**Independent re-verification.** This document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`, `SCIRIS_BACKEND=agg`): every reproduction was re-run and all line numbers still match the source. As a result, findings 6, 10, 11 and 12 (and the former "Misplaced `# pragma: no cover`" section) were rejected and moved to [Rejected on review](#rejected-on-review), finding 4's description was corrected, finding 9's recommended fix was narrowed to a message-only change, and one missed bug was added as finding 13. Original finding numbers are kept stable, so the numbering has gaps. The document now contains 9 findings: 1 High, 5 Medium, 3 Low.

**Nothing in this document has been applied.** All fixes are described, not made.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `asd()` | `pinitial` with one entry per parameter silently disables all downhill steps and returns the starting point with a normal-looking `exitreason` | 187 |
| 2 | Medium | `asd()` | `xmin`/`xmax`/`sinitial` are neither broadcast nor length-checked, so a scalar or short bound crashes with an `IndexError` at a random iteration | 200, 197 |
| 3 | Medium | `asd()` | A starting point outside `xmin`/`xmax` is never clipped, so the returned `x` silently violates the bounds | 262 |
| 4 | Medium | `_validate_fval()` | A 0-d array objective value hits the "size-1 array" branch and crashes on `fval[0]`, even with `die=False`; a 2-d size-1 value is not scalarized and crashes later | 43 |
| 5 | Medium | `asd()` | `maxiters=0` crashes with an `IndexError` instead of doing nothing | 313 |
| 13 | Medium | `asd()` | The objective is always called with the flattened 1-D `x`, not the shape the user passed, so scalar and 2-D starting points break ordinary objectives | 220, 277 |
| 7 | Low | `asd()` | "Warning, negative objective function" prints the new *parameter* value, not the objective; fires on any ordinary run where a parameter goes negative | 291 |
| 8 | Low | `asd()` | The negative-starting-value warning is inside the loop, so it is printed once per iteration | 246 |
| 9 | Low | `asd()` | The relative-improvement `exitreason` reports the mean of the history but the criterion tests its sum, understating by a factor of `stalliters` | 329 |

## Recurring pattern

**Nothing that is sized `2*nparams` or `nparams` is ever validated.** `probabilities` (2n), `stepsizes` (2n), `xmin` (n) and `xmax` (n) are built from user input by `_consistent_shape()`, which flattens and casts but never checks or broadcasts the length. The two harmful symptoms are findings 1 (a length-n `pinitial` silently halves the search space) and 2 (a length-n `sinitial` or a scalar/short `xmin` raises `IndexError` from inside the sampling loop). (Over-long vectors are silently ignored; on review this was judged harmless, see [Rejected on review](#rejected-on-review).) A single `_check_length(vec, n, name)` helper applied at lines 187, 197, 200 and 201 (broadcasting scalars, raising on any other mismatch) would close all of them.

## High severity

### 1. `pinitial` with one entry per parameter silently turns off every downhill step — `sc_asd.py:187`

`probabilities` is internally a length-`2*nparams` vector laid out as `[plus_0..plus_{n-1}, minus_0..minus_{n-1}]`, but the docstring only says "`pinitial (None)`: Set initial parameter selection probabilities", and line 187 accepts whatever length it is given. With `nparams` entries, `cumprobs` is only `nparams` long, so `choice = np.flatnonzero(cumprobs > rng.random())[0]` can never exceed `nparams-1`, `pm = np.floor(choice/nparams)` is always `0`, and the sampler proposes *only* `x[par] + stepsizes[choice]`. Every parameter can then only ever increase. For any objective that is minimized by decreasing a parameter, no step is ever accepted, and `asd()` returns the starting point while reporting the same `exitreason` it uses for a successful, converged run.

```python
def sq(x): return float(np.sum((np.asarray(x) - 1.0)**2))   # minimum at [1,1,1], start above it
r  = sc.asd(sq, [1.5, 2.5, 3.5], pinitial=[1,1,1],       maxiters=200, verbose=0, randseed=1)
r2 = sc.asd(sq, [1.5, 2.5, 3.5], pinitial=[1,1,1,1,1,1], maxiters=200, verbose=0, randseed=1)
```

```
actual   (pinitial=[1,1,1]):       x = [1.5 2.5 3.5]  fval = 8.75                  exitreason = 'Absolute improvement too small (0 < 0.000001000)'
                                   every row of details.xvals equals the starting point: True
actual   (pinitial=[1,1,1,1,1,1]): x = [1.00019531 1. 1.00078125]  fval = 6.48e-07
expected (pinitial=[1,1,1]):       either the same answer as the 2n case, or a ValueError naming the required length
```

The failure is silent in both directions: nothing in the output distinguishes "converged" from "never took a single step", and the `fvals`/`xvals` history is a flat line that a caller reading only `result.fval` will never see. This is the only finding here that returns wrong numbers from an otherwise well-formed call. `sc.asd()` is a leaf inside Sciris (grep finds callers only in `tests/test_asd.py` and `docs/tutorials/tut_advanced.qmd`, none of which pass `pinitial`), so the blast radius is entirely external — but ASD's heaviest users are calibration drivers in Optima/Atomica-style codebases, which are exactly the callers likely to hand-tune `pinitial`.

**Fix**: validate at line 187 that `len(probabilities) == 2*nparams`, raising a `ValueError` that explains the `[plus..., minus...]` layout; optionally accept a length-`nparams` vector by tiling it (`np.concatenate((p, p))`), which is almost certainly what a user passing `n` values intends. The same layout should be stated in the docstring for both `pinitial` and `sinitial`.

## Medium severity

### 2. `xmin`/`xmax`/`sinitial` are neither broadcast nor length-checked — `sc_asd.py:200`, `197`

Lines 200-201 turn `xmin`/`xmax` into flat arrays of whatever length was supplied, and line 262 then indexes them with `par` (`0..nparams-1`); line 197 does the same for `sinitial`, indexed at line 261 with `choice` (`0..2*nparams-1`). A scalar bound — the natural way to say "the same limit for every parameter", and the way the sibling argument `stepsize` already works — or any short vector therefore raises an `IndexError` from deep inside the sampling loop, *when and only when* the RNG happens to select a high index.

```python
def sq(x): return float(np.sum((np.asarray(x) - 1.0)**2))
sc.asd(sq, [1.5, 2.5, 3.5], xmin=0,               maxiters=100, verbose=0, randseed=0)
sc.asd(sq, [1.5, 2.5, 3.5], xmax=10,              maxiters=100, verbose=0, randseed=0)
sc.asd(sq, [1.5, 2.5, 3.5], xmin=[0, 0],          maxiters=100, verbose=0, randseed=0)
sc.asd(sq, [1.5, 2.5, 3.5], sinitial=[.1, .1, .1], maxiters=100, verbose=0, randseed=0)
```

```
actual  xmin=0:                IndexError: index 1 is out of bounds for axis 0 with size 1      (sc_asd.py:262)
actual  xmax=10:               IndexError: index 1 is out of bounds for axis 0 with size 1      (sc_asd.py:263)
actual  xmin=[0,0]:            IndexError: index 2 is out of bounds for axis 0 with size 2      (sc_asd.py:262)
actual  sinitial=[.1,.1,.1]:   IndexError: index 3 is out of bounds for axis 0 with size 3      (sc_asd.py:261)
expected: scalar bounds broadcast to all parameters; a genuinely wrong length raises a ValueError up front
```

Two things make this worse than an ordinary input error. First, the exception surfaces from an internal line about `xmin[par]` rather than from argument checking, so the message never names the offending argument. Second, it is RNG- and `maxiters`-dependent: the same call can succeed on a short run and fail on a long one, since the crash only happens once the sampler picks the missing index. `_consistent_shape()` already casts and flattens these inputs, so the coercion is half-done — `stepsize` is broadcast, `xmin`/`xmax` are not.

**Fix**: after lines 197/200/201, broadcast length-1 input to the required length (`np.full(nparams, val)` / `np.full(2*nparams, val)`) and raise a `ValueError` naming the argument and the expected length for anything else. `nparams` is already known at that point.

### 3. A starting point outside `xmin`/`xmax` is never clipped, so the returned `x` can violate the bounds — `sc_asd.py:262`

`xmin`/`xmax` are documented as "Min/Max value allowed for each parameter", and lines 262-263 enforce them on every *proposal*, but `x` itself is never clipped after line 200. If `x` starts outside the box and the objective improves away from the box, every clipped proposal is worse than the current point, so nothing is ever accepted and `asd()` returns the out-of-bounds starting point (and an entire out-of-bounds `details.xvals`) with no warning.

```python
def far(x): return float((x[0] - 4.9)**2 + (x[1] - 4.9)**2)
r = sc.asd(far, [4.8, 4.8], xmin=[0, 0], xmax=[1, 1], maxiters=100, verbose=0, randseed=1)
```

```
actual:   x = [4.8 4.8]   exitreason = 'Absolute improvement too small (0 < 0.000001000)'   details.xvals.max() = 4.8
expected: x inside [0, 1] (e.g. the starting point clipped to [1, 1] and optimized from there), or an error
```

The caller is handed an infeasible parameter vector labelled as a normal convergence result; in a calibration loop those values then flow into a model that the bounds existed to protect. This is a narrower case than finding 1 (it requires an out-of-bounds `x0`, e.g. a warm start carried over from a run with looser bounds), which is why it is Medium rather than High.

**Fix**: after the bounds are built (line 201), `x = np.clip(x, xmin, xmax)` before the initial objective evaluation, and warn if anything was clipped. A reference implementation that does this converges to the boundary optimum on the repro above.

### 4. A 0-d array objective value crashes the branch written to handle size-1 arrays — `sc_asd.py:43`

`_validate_fval()` explicitly accommodates size-1 arrays: `if isinstance(fval, np.ndarray) and fval.size == 1: fval = fval[0]`. A 0-d array (`np.array(3.0)`, as produced by `np.squeeze()` on a size-1 result, or by an explicit `np.asarray(loss)`) satisfies both conditions, but `fval[0]` is invalid on a 0-d array, so the intended conversion raises instead.

```python
from sciris.sc_asd import _validate_fval
_validate_fval(np.array(3.0))                     # 0-d, size 1
_validate_fval(np.array([[3.0]]))                 # 2-d, size 1
def f0d(x): return np.array(float(np.sum((np.asarray(x) - 1.0)**2)))
sc.asd(f0d, [3., 4.], maxiters=10, verbose=0, die=False)
```

```
actual   _validate_fval(np.array(3.0)):     IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
actual   _validate_fval(np.array([[3.0]])): array([3.])          <- returned without being scalarized
actual   asd(..., die=False):               IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
expected 3.0 in both validation cases; and with die=False, no IndexError escaping asd()
```

`die=False` does not help, because the crash happens at the *initial* evaluation on line 221, which is outside the `try/except`; if it happened inside the loop instead, `die=False` would convert every trial to `np.inf` and `asd()` would silently return the starting point. The second line above is the same bug for `ndim >= 2`: `fval[0]` is still an array, so the function's own promise ("should return a scalar") is not met. This case is not silent, however (an earlier version of this document said it was): the array is rejected as soon as it is stored in the history, with an unhelpful message.

```python
def f2d(x): return np.array([[float(np.sum((x-1)**2))]])
sc.asd(f2d, [3., 4.], maxiters=50, verbose=0, randseed=0)
```

```
actual:   ValueError: setting an array element with a sequence.      (sc_asd.py:230, fvals[0] = fvalorig)
expected: runs normally, treating the size-1 array as the scalar it contains
```

1-d size-1 return values work correctly today.

**Fix**: use `fval = fval.item()` (or `fval.flat[0]`) at line 43, which handles 0-d, 1-d and n-d size-1 arrays identically. This fix should go in together with finding 13, since passing a reshaped scalar `x` makes 0-d return values more common.

### 5. `maxiters=0` crashes instead of doing nothing — `sc_asd.py:313`

The history arrays are allocated with `maxiters + 1` slots (lines 228-229) and the `count >= maxiters` test is at the *end* of the loop body (line 317), after the results for iteration `count` have already been written at line 313. With `maxiters=0` the loop still executes once and writes to index 1 of a length-1 array.

```python
def sq(x): return float(np.sum((np.asarray(x) - 1.0)**2))
sc.asd(sq, [1.5, 2.5], maxiters=0, verbose=0)
```

```
actual:   IndexError: index 1 is out of bounds for axis 0 with size 1      (sc_asd.py:313)
expected: return the starting point immediately with exitreason 'Maximum iterations reached'
```

`maxiters=1` works and does exactly one iteration, so this is purely the zero case — which matters because `maxiters=0` is the obvious way for a driver script to disable optimization for a baseline/no-op run.

**Fix**: check `count >= maxiters` before doing the work (or add `if maxiters <= 0: break` at the top of the loop body, alongside the existing "already at minimum" break).

### 13. The objective is always called with a flattened 1-D `x`, not the documented shape — `sc_asd.py:220`, `277`

*Added on review (2026-09-25).* The docstring says "`x` can be a scalar, list, or Numpy array of any size" and "`function()` accepts input `x`", and the result is reshaped back to `origshape` at line 346. However, line 135 flattens `x` and the objective is then always called with that flat length-n vector (line 220 for the initial evaluation, line 277 in the loop). An objective written for the shape the user actually passed in breaks.

```python
target = np.array([[1., 2.], [3., 4.]])
def m(x): return float(np.sum((x - target)**2))
sc.asd(m, np.zeros((2, 2)), verbose=0)
sc.asd(lambda x: float((x - 2)**2), 5., verbose=0)
```

```
actual   (2,2) start:   ValueError: operands could not be broadcast together with shapes (4,) (2,2)      (x passed to m has shape (4,))
actual   scalar start:  TypeError: only 0-dimensional arrays can be converted to Python scalars      (x passed has shape (1,))
expected: m receives a (2,2) array and result.x ~ target; the scalar objective receives a scalar/0-d x and result.x ~ 2.0
```

The scalar case is the most natural reading of "x can be a scalar". With numpy 2.x, `float()` of a shape-(1,) array raises, so any objective that calls `float(...)` on its result fails at the initial evaluation. Objectives that index `x[i, j]` or do matrix algebra on a 2-D `x` also fail, or silently compute something else (e.g. `x[0]` returns an element instead of a row). This is long-standing: the pre-refactor code in `b35a681^` also called `function(x)` on the flattened vector.

**Fix**: keep the flat vector internally, but call `function(np.reshape(x, origshape), ...)` at line 220 and `function(np.reshape(xnew, origshape), ...)` at line 277. Reshaping a size-1 flat array to shape `()` gives a 0-d array, so the fix for finding 4 (`.item()`) should go in alongside it. Risk: an existing user who passes a 2-D `x` and whose objective expects the flat vector would be affected; that seems unlikely, but the change should be noted in the changelog.

## Low severity

### 7. "Warning, negative objective function" prints the new parameter value, not the objective — `sc_asd.py:291`

```python
if newval < 0 and verbose:
    print(f'ASD: Warning, negative objective function ({newval:n}) on step {count} could lead to unexpected behavior')
```

`newval` is the proposed value of parameter `par` (set at line 261), not an objective value; the objective at that point is `fvalnew`. So the warning fires whenever a *parameter* goes negative, on runs whose objective is non-negative everywhere, and it never fires when the objective actually is negative — which is the condition the sibling check at line 246 exists to warn about.

```python
def sq(x): return float(np.sum((np.asarray(x) - 1.0)**2))   # non-negative everywhere
sc.asd(sq, [3., 4.], label='L', maxiters=20, verbose=0.1, randseed=0)
```

```
actual:
ASD: Warning, negative objective function (-2) on step 8 could lead to unexpected behavior
ASD: Warning, negative objective function (-0.4) on step 9 could lead to unexpected behavior
    L step 10 (0.0 s) -- (orig:13.00 | best:2.930 | new:3.250 | diff:0.3200)
ASD: Warning, negative objective function (-1.5) on step 20 could lead to unexpected behavior
    L step 20 (0.0 s) -- (orig:13.00 | best:0.01000 | new:6.250 | diff:6.240)
=== L Maximum iterations reached (20 steps, orig: 13.00 | best: 0.01000 | ratio: 1299.9999999999977) ===

expected: no such warnings (the objective is 13.0 -> 0.01 throughout, never negative)
```

Note also that the warning is gated only on `verbose` being truthy, not on the `1/verbose` print cadence used at line 308, so at `verbose=0.1` it produces three lines in a run the caller asked to report twice.

**Fix**: test `fvalnew < 0` and print `fvalnew`, and gate it with the same cadence as the step line (or warn only once per run).

### 8. The negative-starting-value warning is printed once per iteration — `sc_asd.py:246`

`if fvalorig < 0 and verbose:` sits inside `while True:`, but `fvalorig` never changes, so the message repeats for the life of the run.

```python
def h(x): return float(np.sum((np.asarray(x) - 1.0)**2) - 10)   # starts at -6
sc.asd(h, [3.0], maxiters=25, verbose=1, randseed=2)
```

```
actual:   the line 'ASD: Warning, negative objective function starting value (-6) could lead to unexpected behavior' is printed 25 times in a 25-step run
expected: once
```

At the default `maxiters=1000` this is 1000 identical lines. **Fix**: move the check above the loop (it only depends on `fvalorig`), next to the seed message at line 132.

### 9. The relative-improvement `exitreason` reports the mean but the criterion tests the sum — `sc_asd.py:329`

Line 328 stops when `sum(relerrorhistory) < reltol`, but line 329 formats `np.mean(relerrorhistory)` into the message, so the number shown is smaller than the number tested by a factor of `stalliters` (`10*nparams` by default). The printed comparison looks satisfied by a wide margin even when the run stopped at the threshold. This is also the only place where the two sibling criteria disagree: the absolute test at line 324 uses the mean of its history, so the meaning of `reltol` — unlike `abstol` — silently scales with the number of parameters.

```python
def slow(x): return float(1.0 + np.sum(np.abs(np.asarray(x) - 1.0)))
for rt in [1e-3, 5e-5]:
    r = sc.asd(slow, [1.02, 1.02], abstol=0, reltol=rt, maxiters=400, verbose=0, randseed=11)
    print(rt, len(r.details.fvals) - 1, r.exitreason)
```

```
actual:
0.001   58 steps   Relative improvement too small (0.00003203 < 0.001000)
5e-05   61 steps   Relative improvement too small (0 < 0.00005000)
```

The first run stopped where the message claims `3.203e-05 < 1e-3`; if that were the compared quantity, `reltol=5e-05` (still above `3.203e-05`) would have stopped at the same step, and it does not — it runs three steps further, because the quantity actually tested was `sum = 6.405e-04` (confirmed by instrumenting a private copy of the module: `reported mean = 3.202624589907499e-05 but tested sum = 0.0006405249179814998 vs reltol = 0.001, stalliters = 20`).

**Fix**: fix only the message: `strrel, strtol = sc.sigfig([sum(relerrorhistory), reltol])` at line 329, optionally with a docstring note that `reltol` is summed over `stalliters` iterations. An earlier version of this document recommended the opposite ("better") fix of testing `np.mean(relerrorhistory)`; that is unsafe, because the sum test predates the refactor (it is present in `b35a681^`) and switching to the mean would loosen `reltol` by a factor of `stalliters` (`10*nparams` by default), silently making every existing calibration stop earlier.

## Verified clean

**Core loop, step and probability adaptation (lines 240-314).** The algorithm itself is correct. A hand-written reference implementation of the published ASD update — independent code, same RNG (`np.random.default_rng(seed)`), same `choice -> (par, pm)` decoding — reproduces `asd()`'s final objective value *exactly* on every seed for a 2-D quadratic, a 5-D quadratic, 2-D Rosenbrock and a 3-D L1 function (`quad2/quad5/abs3` both reach `0.00e+00` for all 8 seeds; `rosen2` gives the identical per-seed sequence `2.74, 3.27, 3.65, 3.75, 2.61, 3.78, 3.78, 2.05` in both implementations), which pins down the parts of the loop most likely to be wrong. Specifically checked and correct: the plus/minus pairing (`stepsizes = concatenate((s, s))` means index `choice` and `choice % nparams` refer to the same parameter, `pm = floor(choice/nparams)` is 0 for the first block and 1 for the second, and `(-1)**pm` therefore adds for `choice < nparams` and subtracts for `choice >= nparams` — confirmed at `verbose=3`: `choice=2, par=0, pm=-1.0, x[par]=10.0 -> newval=9.0` for `nparams=2`); no off-by-one between `choice`, `par` and `stepsizes`/`probabilities`; `sinc` and `pinc` are applied on acceptance and `sdec`/`pdec` on rejection, each to the quantity it names and in the direction the docstring claims (verified numerically with `sinc=3, sdec=4, pinc=5, pdec=7`: an accepted `choice` went `1.0 -> 3.0 -> 9.0` in step size and `0.25 -> 1.25`, then `0.625 -> 3.125`, in normalized probability; a rejected one went `2.0 -> 0.5` in step size and `0.0357 -> 0.0051` in probability); `stepsize` reaches `stepsizes` (`stepsize=0.5` vs `0.01` on `x=[10,20]` gives `[5,10,10,10]` vs `[0.1,0.2,0.2,0.2]`); the zero-step fallbacks at lines 209-212 behave as commented.

**Probability vector.** Renormalization at line 254 happens before every draw, so the vector cannot drift to all-zeros or negative under the validated `pinc, pdec >= 1`: after 800 iterations with both tolerances disabled the minimum entry was `0.0769` and all entries were positive. `np.flatnonzero(cumprobs > rng.random())[0]` cannot come up empty because the final `cumprobs` entry is exactly 1.0 after normalization and `rng.random()` is in `[0, 1)`. The `sum(probabilities) == 0` guard at line 188 fires as intended. (`details.probabilities` is returned post-update and so does not sum to 1; that matches the pre-refactor behavior and the docstring says nothing stronger than "The probability of each step".)

**Bounds and sampler stalling.** With a starting point *inside* the box, `xmin`/`xmax` are never violated: clipping at lines 262-263 can only pull a proposal back to the boundary, and a clipped proposal that still differs from `x[par]` is legitimately evaluated, so the boundary optimum is found (`f(x)=sum((x+10)^2)` with `xmin=[0,0], xmax=[1,1]` from `[0.5,0.5]` correctly returns `[0,0]`). The `maxrangeiters=100` retry loop does not spin forever, and a fully pinned parameter (`xmin == xmax == x0`) terminates normally rather than hanging — the only defects there are finding 3 (out-of-bounds start) and the degenerate double penalty described under rejected item 10.

**Termination and `exitreason`.** Each criterion fires for the reason it reports: `maxiters` (`maxiters=5 -> 'Maximum iterations reached'` with exactly 5 recorded steps), `maxtime` (a 10 ms objective with `maxtime=0.05` exits at 0.06 s wall clock reporting `0.05157 > 0.05000`), `stoppingfunc`, `abstol` (`'Absolute improvement too small (0.0000001216 < 0.000001000)'` after 69 steps at the defaults), `minval` both as the pre-loop skip (`sc.asd(sq, [1.,1.])` -> `'Objective function already at minimum value (0.0), skipping optimization'`) and as the in-loop stop, and `reltol` (correct *behavior*; only the reported number is wrong, finding 9). The stall history is a correctly sized circular buffer: with `count` starting at 1 and index `count % stalliters`, every slot has been written at least once by `count == stalliters`, and the `count > stalliters` guard means neither criterion can be evaluated against leftover zeros — there is no off-by-one there. `stalliters=1` works. `maxiters=1` works.

**Returned values and `details` consistency.** The returned `x` is genuinely the best-ever point, not the last trial: `x` and `fval` are updated only inside the `fvalnew < fvalold` branch, so `details.fvals` is non-increasing (verified over 300- and 500-iteration runs), `result.fval == details.fvals[-1]`, `function(result.x) == result.fval` to floating-point equality, `result.x` equals the `details.xvals` row with the lowest objective, and `len(details.fvals) == details.xvals.shape[0] == count+1` (one entry for the starting point plus one per iteration) — so `fvals[i]` and `xvals[i]` are a matched pair for every `i`. Shapes round-trip through `origshape`: a scalar `x` returns a 0-d array, a `(2,2)` input returns a `(2,2)` result, and a list returns a length-`n` array.

**Reproducibility and global state.** `randseed` is fully reproducible (two calls with `randseed=42` gave bit-identical `x` and `fval`) and `asd()` does not touch the global numpy RNG (`np.random.seed(0); np.random.random()` returns the same value with and without an intervening `asd()` call), matching the docstring's "Uses its own random number stream"; `randseed=None` gives genuinely different streams (the near-identical answers on a quadratic are convergence to the same dyadic lattice point, not a seeding bug). No input is mutated: `x` (list or ndarray), `xmin`, `xmax`, `sinitial`, `pinitial` and the `args` dict are all unchanged after a run, because `_consistent_shape()` goes through `np.array(..., dtype='float')` (a copy) and `args`-as-kwargs goes through `sc.mergedicts()`. `warnings.filters` is unchanged after a `die=False` run, and the `die=True` path re-raises the original exception unwrapped.

**`die=False`, NaN and inf.** Aside from the initial evaluation not being covered (rejected item 11: raising when the starting point cannot be evaluated is reasonable), the soft path works: an objective that raises for some proposals is warned about and scored `np.inf`, the trial is rejected, and the run converges normally; `sc.sigfig()` formats `inf`/`-inf`/`nan` without raising, so the `verbose=1` step line survives an `inf` trial; an objective returning `NaN` is rejected (`max(0, fval - nan)` and `max(0, nan - 1)` both evaluate to `0` in Python, so the stall history is not poisoned) and the "objective function returned NaN" message fires; a `NaN` *starting* value does not hang — nothing can ever beat it, so the run stops after `stalliters` with `fval = nan` and `x = x0`. `_improvement_ratio()` was checked at all three of its branches, including `fvalnew = inf` (ratio 0) and both-near-zero (ratio 1); the refactor in `b35a681` flipped the final printed `ratio` from `best/orig` to `orig/best`, which is now consistent with the in-loop `ratio` convention (`>1` means improvement) — a deliberate change, not a defect. `x` containing `NaN` is rejected at line 204 as documented.

**Arguments and docstring examples.** All three documented ways of passing extra arguments give identical answers (`args=[0.5, 0.1]`, `args=dict(scale=0.5, weight=0.1)`, and bare `scale=0.5, weight=0.1` all converge to `[1, -2, -3]` with `fval = 0.0`), kwargs win over an `args` dict as `sc.mergedicts()` implies, and both docstring examples run verbatim (`sc.asd(np.linalg.norm, [1,2,3])` returns `x ~ [-8.3e-17, -1.8e-16, -2.2e-16]`). `label` is prefixed to the step lines and the summary. The `1/verbose` print cadence is correct for `verbose = 1, 0.5, 0.25, 0.1` (steps `[10, 20]` for `verbose=0.1` over 20 iterations); the extra lines that appear at other steps are the spurious warnings of finding 7, not a cadence error. `verbose=0` silences all output. `minval=None` cleanly disables both minimum-value checks. `_consistent_shape()` casts integer input to float, so there is no integer-truncation path into `x` or the step sizes.

## Rejected on review

- **6.** `stalliters=0` crashes with `ZeroDivisionError` — NOT WORTH FIXING: a stall window of 0 iterations is meaningless, so this is an unhelpful error on invalid input (out of scope per this audit's own rules); at most add input validation.
- **10.** A stalled sampling loop is penalized twice and still spends an objective evaluation — NOT WORTH FIXING (and the proposed fix was broken): it needs 100 consecutive out-of-range draws, i.e. effectively every parameter pinned, and the proposed `continue` would skip the stopping-criteria block and overrun `fvals` with `IndexError` at `count = maxiters+1` (the cited `stepsizes.min() = 3.33e-17` comes from ordinary `sdec` rejections, not this path).
- **11.** `die=False` does not cover the initial objective evaluation, and `_validate_fval()`'s `die` argument is dead — NOT WORTH FIXING: `die` is documented in terms of trials, raising when the starting point itself cannot be evaluated is sensible, and the unused `die` of a private helper is dead code rather than a bug.
- **12.** At `verbose>=3` the two lines about one step report contradictory `pm` values — NOT WORTH FIXING: cosmetic inconsistency in debug output (one line shows the sign, the other the index).
- **Former "Misplaced `# pragma: no cover`" section** (lines 317, 320, 332) — NOT A BUG: coverage annotations have no runtime effect.
- **Over-long `pinitial`/`sinitial`/`xmin`/`xmax` vectors** (from "Recurring pattern") — NOT WORTH FIXING: the extra entries are silently ignored, which is harmless, and the length validation proposed for findings 1 and 2 would cover it anyway.
