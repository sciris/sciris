# `sc_parallel.py` bug audit

Audit of `sciris/sc_parallel.py` (1092 lines) for genuine defects: wrong numerical results, documented arguments that don't work, silent data corruption, resource leaks, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps, and general performance.

**Scope**: the whole module, covered by two parallel auditors: one over lines 44-635 (`_Counter`, `_progressbar`, the `Parallel` class), one over lines 636-1092 (`parallelize()`, `TaskArgs`, `_task()`, the load functions, `loadbalancer()`). **Method**: line-by-line reading of each function, followed by executed hypothesis tests -- every "actual" value below was produced by running the code against the editable install (Sciris 3.3.0, numpy 2.4.6, psutil 7.2.2, multiprocess 0.70.19, Python 3.13.9, Linux, multiprocessing start method `fork`, commit `2d69aad`), and every finding was reproduced a second time independently before being recorded here. Line numbers refer to the current working tree.

**Nothing in this document has been applied.** All fixes are described, not made.

**Re-verification.** This document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`): every finding was re-run with a minimal repro. Of the original 23 findings, 16 were kept (13 confirmed as written, plus 3 whose description or proposed fix was corrected: 2, 7, 9; the repro for 20 was also strengthened), 7 were rejected as not a bug or not worth fixing (see "Rejected on review" near the end), and 5 new findings (24-28) were added. Original finding numbers are kept stable, so the numbering has gaps.

The single high-severity finding -- forked worker processes all inheriting an identical copy of the legacy global numpy RNG state, so every job draws the same "random" stream -- deserves particular attention: for a scientific-modelling library whose primary parallel use case is running stochastic ensembles (Covasim/Starsim-style Monte Carlo sweeps), this can silently turn an ensemble of `n` runs into one run repeated `n` times, with no error, warning, or implausible-looking output to flag it. Of every defect catalogued here, it is the one most likely to quietly invalidate a user's actual scientific results rather than just crash or misbehave visibly.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `_task()` | Forked workers share one RNG state, so every job of the default parallelizer returns identical "random" draws, and `serial=True` silently disagrees | 900 |
| 2 | Medium | `sc.parallelize()` / `Parallel.__init__()` / `_task()` | `args` given as the documented `list` type crashes with `iterarg` (found independently from both halves) | 150, 871 |
| 3 | Medium | `Parallel.__init__()` / `_task()` | `lbkwargs`'s `maxcpu`/`maxmem`/`interval` are clobbered with `None` by `sc.mergedicts()` (found independently from both halves) | 152, 876 |
| 4 | Medium | `Parallel.reset()` | Wipes configuration set by `init()`, leaving the object unrunnable | 180 |
| 6 | Medium | `Parallel.validate_args()` | `iterarg` and `iterkwargs` together are documented and implemented, but rejected by the validator (found independently from both halves) | 249, 647 |
| 7 | Medium | `Parallel.set_ncpus()` | `ncpus` as a float >= 1 (a documented type) raises `TypeError` (found independently from both halves) | 315, 666 |
| 8 | Medium | `Parallel.run_async()` / `sc.parallelize()` | Worker processes and the manager process leak when a job raises with `die=True` | 505 |
| 9 | Low | `Parallel.finalize()` | Closes the pool but never shuts down the manager, leaking one process per retained `Parallel` object | 577 |
| 10 | Medium | `Parallel.process_results()` / `_task()` | `die=False` returns `None` for a failed job, not the documented exception, and `sc.parallelize()` gives no way to tell a real `None` from a failure | 605, 913 |
| 11 | Medium | `_task()` | `capture=True` plus a failing job destroys the whole run for two of six parallelizers, and returns a non-string for the default one | 895 |
| 12 | Medium | `sc.loadbalancer()` | Silently consumes a draw from the global numpy RNG, even on the no-op "no load limit" path | 1052 |
| 13 | Low | `sc.loadbalancer()` | Always waits longer than `maxtime`, by a factor of `1 + cpu_interval/interval` (3x measured) | 1067 |
| 14 | Low | `_jobkey()` / `sc.parallelize()` | Vestigial `globaldict[_jobkey(index)]` progress writes pollute a user-supplied `globaldict` | 44 |
| 18 | Low | `Parallel.set_ncpus()` | The `maxload` deprecation shim warns and then discards the value | 306 |
| 19 | Low | `Parallel.set_method()` | `serial=True` on top of an async parallelizer raises instead of running serially | 351 |
| 20 | Low | `Parallel.make_pool()` | An empty `globaldict` is silently replaced by a fresh dict for custom parallelizers, so an empty `Manager().dict()` loses all worker writes | 418 |
| 24 | Medium | `_task()` / `make_argslist()` | A `None` element in `iterarg` is silently not passed to the function | 867 |
| 25 | Medium | `Parallel.run()` | A user's argless `RuntimeError`/`NotImplementedError` is replaced by `IndexError` in the `freeze_support` handler | 624-625 |
| 26 | Medium | `_task()` | `parallelizer='thread'` with `capture=True` leaves `sys.stdout` hijacked and misattributes output between jobs | 895-898 |
| 27 | Medium | `sc.loadbalancer()` / `_task()` | The random start stagger is identical in every forked worker, so it staggers nothing (interaction of #1 and #12) | 1052 |
| 28 | Low | `Parallel.run_async()` | `-copy` parallelizers deep-copy `globaldict`, so worker writes are lost | 499-500 |

Counts: 1 High, 13 Medium (2, 3, 4, 6, 7, 8, 10, 11, 12, 24, 25, 26, 27), 7 Low (9, 13, 14, 18, 19, 20, 28); 7 original findings rejected on review (5, 15, 16, 17, 21, 22, 23).

## Recurring patterns

**`None` defaults overwriting an explicit override dict.** Both halves independently hit the same line of reasoning: `self.lbkwargs = sc.objdict(sc.mergedicts(lbkwargs, maxcpu=maxcpu, maxmem=maxmem, interval=interval))` at `Parallel.__init__()` (line 152) merges the *explicit* keyword arguments last, so their default `None` values clobber whatever the user put inside `lbkwargs`. `_task()` then gates the load balancer on `if lbkw.maxcpu or lbkw.maxmem:` (line 876), so passing the throttle limits through the documented `lbkwargs` route disables load balancing entirely rather than enabling it (finding 3). The same idiom would silently break any future pass-through dict argument; a `None`-skipping merge is the general fix.

**Falsy or in-band sentinels.** `if self.inputdict:` (line 418, finding 20) versus the correct `is not None` used at line 451; and `if taskargs.iterval is not None:` (line 867, finding 24), which uses `None` to mean "no `iterarg`" even though `None` is a legitimate `iterarg` element. Each is a value that can legitimately be empty or `None`.

**Documented argument types that the code cannot actually accept.** `args` is documented as `list` but only a `tuple` survives concatenation with the coerced `iterval` (finding 2); `ncpus` is documented as `int/float` but any float `>= 1` is never cast and reaches `mp.Pool(processes=...)` as a float, raising `TypeError` (finding 7). Both were found independently by each auditor from opposite ends of the call path (construction-time storage versus the point where the value is actually consumed), which is itself informative: these are default-path defects, not obscure edge cases.

**`die=False` returning `None` rather than the documented exception.** `process_results()` appends the failed job's `result` (left at its initialized `None`) rather than its `exception`, and `sc.parallelize()` returns only `Parallel.results`, so a crashed job is indistinguishable from one that legitimately returned `None` (finding 10). This was found from both the `process_results()` side and the `_task()` side, since the same information is discarded at both points.

**No cleanup on the error path.** All process/pool teardown lives in `finalize()`, which is only reached on success (the `run_async()` call at line 505 and the pool-close at line 577 are both outside any `try`), and even then it does not cover the manager. There is no `try`/`finally` or context-manager use anywhere in the class, which produces both finding 8 (leaked workers plus manager on an uncaught exception) and finding 9 (leaked manager even on the success path, once the object is retained). The `capture` block has the same shape (finding 11).

**`reset()` and `init()` disagree about what "state" means.** `reset()` clears both run products and configuration, but only `init()` can rebuild the configuration (finding 4).

**Load-balancing and staggering reach for the global RNG.** `np.random.rand()` inside `loadbalancer()` (finding 12) is inconsistent with the neighboring `sc.randsleep()` call, which correctly uses a private `default_rng`; the same class of defect, at a much larger scale, is the module's single high-severity finding (finding 1). The two combine in finding 27: because forked workers share one RNG state, the stagger drawn from it is the same in every worker.

## High severity

### 1. Workers are never reseeded, so every job of the default parallelizer returns the same "random" numbers, and `serial=True` silently disagrees — `sc_parallel.py:900`

`_task()` calls `func(*args, **kwargs)` with no RNG handling at all. On Linux the default parallelizer (`multiprocess`) forks, so every worker inherits an identical copy of the legacy global `numpy` RNG state and every job draws the *same* stream. The same code run with `serial=True` -- documented at `sc_parallel.py:671` as "useful for debugging; equivalent to `parallelizer='serial'`" -- draws successive values instead, so the parallel and serial results differ for any stochastic function.

```python
import sciris as sc, numpy as np
def rnd(i): return round(float(np.random.random()), 6)
if __name__ == '__main__':
    print('parallel:', sc.parallelize(rnd, iterarg=range(3)))
    print('serial  :', sc.parallelize(rnd, iterarg=range(3), serial=True))
```

Actual:

```
parallel: [0.203256, 0.203256, 0.203256]
serial  : [0.203256, 0.81855, 0.391348]
```

Expected: three distinct draws in both cases (or, at minimum, the same answer from both code paths). The full sweep over parallelizers, seeded with `np.random.seed(1)` beforehand, was:

```
parallel (multiprocess): [0.417022004702574, 0.417022004702574, 0.417022004702574, 0.417022004702574]
serial                 : [0.417022004702574, 0.7203244934421581, 0.00011437481734488664, 0.30233257263183977]
thread                 : [0.417022004702574, 0.00011437481734488664, 0.7203244934421581, 0.30233257263183977]
fast (cf)              : [0.417022004702574, 0.417022004702574, 0.417022004702574, 0.417022004702574]
```

So both process-based parallelizers collapse to one value repeated `njobs` times, while the two in-process ones do not. Nothing warns; the values look like plausible random numbers, and a Monte Carlo ensemble of `n` runs silently becomes one run repeated `n` times. `np.random.default_rng()` inside the function is unaffected (`[0.0352, 0.1913, 0.2640, 0.2693]`), which is why this can hide for a long time in a codebase that mixes the two APIs.

Honest counter-evidence: `parallelize()`'s own Examples 2 and 4 (`sc_parallel.py:697`, `716`) both call `np.random.seed()` as the first line of the worker function, so the maintainers are clearly aware that the caller must reseed. The defect is that (a) nothing in the `Args:` list or notes says so, and (b) the `serial` argument is advertised as an equivalent debugging substitute when it is not.

Blast radius: `_task()` is the single entry point for every job of every `sc.parallelize()`/`sc.Parallel()` call, and Sciris is the parallelization layer for Covasim/Starsim-style stochastic ensembles. Re-verification caveats: this is standard numpy behavior after `fork` (the legacy global `RandomState` is not reseeded, unlike Python's `random` module), and Starsim and Covasim seed explicitly per run, so their own ensembles are not affected; the exposure is user code that relies on the global RNG. It also has a knock-on effect inside Sciris itself (finding 27). Re-verified output: parallel `[0.704433, 0.704433, 0.704433]` vs. serial `[0.704433, 0.402736, 0.139802]`.

**Fix**: this needs a design decision. If auto-reseeding, derive the per-job seeds in the parent (e.g. via `np.random.SeedSequence`, optionally from a new `seed` argument) and `np.random.seed(...)` each worker with its own seed, so that a parent-seeded run stays reproducible; a bare `np.random.seed()` in the worker would make runs irreproducible. At minimum, document the requirement in the `Args:` block next to `serial`, and drop the claim that `serial=True` is equivalent.

## Medium severity

### 2. `args` given as the documented `list` type crashes when `iterarg` is used — `sc_parallel.py:150`, `871`

`args` is documented as `args (list): positional arguments for each process, the same for all processes`, and is stored verbatim in `Parallel.__init__` (line 150) with no coercion. `_task()` then does `args = taskargs.iterval + args` (line 871) after forcing `iterval` to a tuple, so a list `args` fails on the tuple/list concatenation -- but only when `iterarg` is in play, which makes it an internal inconsistency rather than a plain type complaint.

```python
import sciris as sc
def f(x, y): return x*y
def g(a, b): return (a, b)
if __name__ == '__main__':
    print('tuple:', sc.parallelize(f, iterarg=[1,2,3], args=(10,)))
    print('list :', sc.parallelize(f, iterarg=[1,2,3], args=[10]))
    print(sc.parallelize(g, iterkwargs={'b':[1,2]}, args=[10], serial=True)) # works
    print(sc.parallelize(g, iterarg=[1,2],          args=[10], serial=True)) # fails
```

Actual:

```
tuple: [10, 20, 30]
list : FAILED TypeError can only concatenate tuple (not "list") to tuple
[(10, 1), (10, 2)]
TypeError: can only concatenate tuple (not "list") to tuple
```

Expected: both `args` forms give `[10, 20, 30]`, and the last call gives `[(1, 10), (2, 10)]` (as `args=(10,)` gives). `args=10` fails the same way (`can only concatenate tuple (not "int") to tuple`). The coercion is also internally inconsistent: `iterval` *is* coerced to a tuple two lines earlier, and in the embarrassingly-parallel case (where no concatenation happens) a list works fine -- `sc.parallelize(g, iterarg=3, args=[1,2])` returns `[3, 3, 3]`. So whether `args=[...]` works depends on which iteration form you chose. `tests/test_parallel.py` never passes `args` to `sc.parallelize()` (grep: `args=` appears only as `iterkwargs=`/`kwargs=`), which is why this survives untested.

**Fix**: make `_task()` concatenate with `args = taskargs.iterval + tuple(args)`, or normalize once in `__init__` with `tuple(args)` (after a `None` check, wrapping a scalar only if it is not already a list or tuple). Correction on review: the originally proposed `self.args = tuple(sc.tolist(args))` is wrong, because `sc.tolist((10,))` returns `[(10,)]`, so the currently working `args=(10,)` would become `((10,),)` and pass a tuple as the argument.

### 3. `lbkwargs`'s `maxcpu`, `maxmem` and `interval` are silently overwritten with `None` — `sc_parallel.py:152`, `876`

`self.lbkwargs = sc.objdict(sc.mergedicts(lbkwargs, maxcpu=maxcpu, maxmem=maxmem, interval=interval))` merges the *explicit* keyword arguments last, so their default `None` values clobber whatever the user put in `lbkwargs`. Since `_task` only invokes the balancer when `lbkw.maxcpu or lbkw.maxmem` (line 876) is truthy, passing the throttle limits via the documented `lbkwargs` route ("lbkwargs (dict): if provided, passed to `sc.loadbalancer()`") disables load balancing entirely rather than enabling it.

```python
import sciris as sc, time
def f(x): return x
P = sc.Parallel(f, iterarg=[1,2], lbkwargs=dict(maxcpu=0.7, maxmem=0.6, interval=0.3))
print(dict(P.lbkwargs))
t0 = time.time(); sc.parallelize(f, iterarg=[1,2], interval=3.0)
print(f'interval=3.0 alone: {time.time()-t0:.2f}s')
```

Actual: `{'maxcpu': None, 'maxmem': None, 'interval': None}`, and `interval=3.0 alone: 0.04s`. Expected: `{'maxcpu': 0.7, 'maxmem': 0.6, 'interval': 0.3}`. (The `interval=3.0 alone` timing is not part of this bug: `interval` configures the load check and is correctly inert without `maxcpu`/`maxmem`; see rejected finding 15.)

Instrumenting the balancer confirms it is never called (`sc.loadbalancer = lambda **kw: calls.append(kw)`; `sc.parallelize(f, iterarg=[1,2], serial=True, lbkwargs=dict(maxcpu=0.7))` gives `0` calls, versus `2` calls for `maxcpu=0.7` passed at top level). Keys that have no top-level counterpart survive correctly (`lbkwargs=dict(verbose=True, label='mine')` with `maxcpu=0.8` reaches the balancer as `{'verbose': True, 'label': 'mine', 'maxcpu': 0.8, ...}`, and `lbkwargs=dict(maxtime=0.3, cpu_interval=0.05, verbose=True)` reaches `loadbalancer()` and prints), so only the three throttling keys are affected -- exactly the ones worth overriding. `tests/test_parallel.py` never passes `lbkwargs`.

**Fix**: merge in the other order, dropping `None`s -- e.g. `lbkw = sc.mergedicts(dict(maxcpu=maxcpu, maxmem=maxmem, interval=interval), lbkwargs)`, or build the explicit dict with only the arguments the user actually supplied (`{k:v for k,v in (...) if v is not None}`) before merging `lbkwargs` on top. (The original audit also suggested gating the balancer on a bare `interval`; that was rejected on review, see finding 15 under "Rejected on review".)

### 4. `reset()` wipes configuration set by `init()`, so a reset object can no longer be run — `sc_parallel.py:180`

`reset()` is advertised in the class docstring as "reset the `Parallel` object to its initial pre-run state", but it clears `ncpus`, `njobs`, `embarrassing` and `method`, which are products of `init()` (i.e. of `validate_args()`, `set_ncpus()` and `set_method()`), not of the run. The object is left in a state no constructor can produce, and the next `run()` falls through `make_pool()` to the branch that is commented "Should be unreachable".

```python
import sciris as sc
def f(x): return x
if __name__ == '__main__':
    P = sc.Parallel(f, iterarg=[1,2])
    P.run()
    P.reset()
    print('njobs =', P.njobs, ' ncpus =', P.ncpus, ' method =', P.method)
    P.run()
```

Actual: `njobs = None  ncpus = None  method = None` then `ValueError: Invalid parallelizer "None"`. Expected: `reset()` returns the object to a runnable pre-run state (i.e. `[2, 4]` again). Note this is *not* a general re-run problem: without the `reset()` call, `P.run()` twice re-executes correctly and returns fresh results (verified with a pid/timestamp-returning function), as does `run_async()`/`finalize()` twice on an async object.

**Fix**: either have `reset()` clear only run products and re-derive the configuration by calling `self.set_ncpus()`/`self.set_method()` afterwards, or make `reset()` call `self.init()`-style re-validation at the end. (`init()` itself calls `reset()` first, so the split has to be preserved: move `ncpus`/`njobs`/`embarrassing`/`method` out of `reset()`.) While there, add `self.rawresults = None` to `reset()` (harmless; see rejected finding 5).

### 6. `iterarg` and `iterkwargs` together are documented and implemented, but rejected by the validator — `sc_parallel.py:249`, `647`

The `sc.parallelize()` docstring opens with "Either **or both** of `iterarg` or `iterkwargs` can be used", and continues "each iterable must be the same length (and the same length of `iterarg`, if it exists)"; the "nothing to parallelize" error message on line 292 also says "please supply an iterarg, iterkwargs, **or both**". But `validate_args()` (line 249, marked `# pragma: no cover`) raises if both are present, and `make_argslist()` (lines 455-469) builds `iterval` and `iterdict` completely independently, i.e. the combined mode is fully implemented downstream of the guard.

```python
import sciris as sc
def f(x, y): return x*y
if __name__ == '__main__':
    sc.parallelize(f, iterarg=[1,2,3], iterkwargs={'y':[2,3,4]})
```

Actual: `ValueError: You can only use one of iterarg or iterkwargs as your iterable, not both`. Expected per the docstring: `[2, 6, 12]`. Bypassing only the guard shows the machinery works:

```python
P = sc.Parallel(f, iterarg=[1,2,3], serial=True)
P.iterkwargs = {'y':[2,3,4]}   # set after init(), so validate_args() is not re-run
P.run(); print(P.results)      # -> [2, 6, 12]
```

Mismatched lengths *within* `iterkwargs` are already handled correctly (`All iterkwargs iterables must be the same length, not 3 vs. 2`, no silent truncation); only the cross-check between `iterarg` and `iterkwargs` is missing, because the combined case is rejected outright instead of validated.

**Fix**: either delete the guard and add the length cross-check the docstring already describes (`len(iterarg) == njobs`), which `validate_args()` is one line away from doing, or drop "or both" from the docstring and from the line 292 error message.

### 7. `ncpus` given as a float >= 1 (a documented type) raises `TypeError` — `sc_parallel.py:315`, `666`

The signature documents `ncpus (int/float)`, and `set_ncpus()` handles the fractional case with `elif 0 < ncpus < 1: ncpus = int(np.ceil(sys_cpus*ncpus))`. Any float `>= 1` skips that branch and is never cast, so `self.ncpus` stays a float and is handed straight to `mp.Pool(processes=2.0)`.

```python
import sciris as sc
def f(x): return x
if __name__ == '__main__':
    for n in [0.5, 1, 1.0, 2, 2.0]:
        try: print(f'ncpus={n!r}:', sc.parallelize(f, iterarg=[1,2], ncpus=n))
        except Exception as E: print(f'ncpus={n!r}: FAILED', type(E).__name__, E)
```

Actual:

```
ncpus=0.5: [1, 2]
ncpus=1: [1, 2]
ncpus=1.0: FAILED TypeError 'float' object cannot be interpreted as an integer
ncpus=2: [1, 2]
ncpus=2.0: FAILED TypeError 'float' object cannot be interpreted as an integer
```

Expected: `ncpus=1.0` and `ncpus=2.0` behave as `1` and `2`. This is squarely in contract for the documented type, and it is exactly what you get from the natural idiom `ncpus=sc.cpu_count()/2` or `ncpus=np.floor(...)`. `0.5`, `0.9999`, `0`, `None`, `1`, `2` and `100` all behave correctly otherwise. The test suite only exercises `ncpus=0.7` and `ncpus=-3` (`tests/test_parallel.py:153,161`), so the float-above-one case is untested.

**Fix**: `ncpus = int(np.ceil(sys_cpus*ncpus)) if 0 < ncpus < 1 else int(ncpus)` in `set_ncpus()`, applied unconditionally before `ncpus = min(ncpus, self.njobs)`. Correction on review: the original audit also suggested optionally treating `ncpus == 1.0` as a fraction via `0 < ncpus <= 1`; do not do this, because `1 == 1.0` in Python, so `ncpus=1` (int, the common "run on one CPU" request) would satisfy the test and resolve to all `sc.cpu_count()` CPUs.

### 8. Worker processes and the manager process are leaked when a job raises with `die=True` — `sc_parallel.py:505`

`run_async()` calls `output = self.map_func(_task, argslist)` with no `try`/`finally`, and `run()` only intercepts `RuntimeError` for the Windows `freeze_support` message, so when a worker re-raises with `die=True` the exception propagates out of `run_async()` before `finalize()` is ever reached and `self.pool.__exit__()` (line 578) is never called. The pool's worker processes plus the `SyncManager` process stay alive; because the `Parallel` object is reachable from the exception's traceback (a reference cycle), they are only reclaimed by the cyclic garbage collector, so a caller that catches the exception and retries accumulates them.

```python
import sciris as sc, psutil, time
def bad(x):
    if x == 1: raise ValueError('boom')
    return x
if __name__ == '__main__':
    nk = lambda: len(psutil.Process().children(recursive=True))
    for trial in range(5):
        try: sc.parallelize(bad, iterarg=[0,1,2], ncpus=3)
        except Exception: pass
        time.sleep(0.3)
        print(f'trial {trial}: live children = {nk()}')
```

Actual (`ncpus=3`, so 3 workers + 1 manager per failed call):

```
trial 0: live children = 4
trial 1: live children = 8
trial 2: live children = 12
trial 3: live children = 10
trial 4: live children = 4
```

Expected: `0` after each failed call, as happens after a successful call (verified: `children == 0` immediately after `sc.parallelize(f, iterarg=[0,2,3], ncpus=3)` succeeds). A single failed call leaves 4 processes until GC runs; `gc.collect()` clears them, confirming the cycle. With a realistic `ncpus` (e.g. 32) and a retry loop this is tens to hundreds of stray forks each holding a copy-on-write image of the parent.

The same hole applies to the async path: with `parallelizer='multiprocess-async'` and `die=True` the exception surfaces from `self.jobs.get()` at line 576, i.e. *before* the `close_pool` block on line 577, so the pool again survives.

**Fix**: wrap the body of `run_async()` (and the `get_results` block of `finalize()`) in `try`/`except` that calls `self.pool.terminate()`/`__exit__` and `self.manager.shutdown()` before re-raising, or run the whole of `run()` under a `finally` that calls `finalize(get_results=False, process_results=False)`.

### 10. With `die=False` the failed job's result is `None`, not the exception the docstring promises — `sc_parallel.py:605`, `913`

`sc.parallelize()` documents `die (bool): whether to stop immediately if an exception is encountered (otherwise, store the exception as the result)`. Neither side of the module honours it: `_task()`'s `die=False` branch stores the exception in `outdict['exception']` and leaves `result` at its initialized `None` (line 913), and `process_results()` (line 605) appends `raw['result']` to `results` and files the exception in the separate `exceptions` list. Since `sc.parallelize()` returns only `Parallel.results`, a crashed job is returned as `None` -- indistinguishable from a job that legitimately returned `None` -- and the only signal is a `RuntimeWarning` that is easy to miss (and suppressed under `warnings.simplefilter('ignore')`).

```python
import sciris as sc, warnings
def f(x):
    if x == 1: raise ValueError('boom')
    return None if x == 2 else x
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print(sc.parallelize(f, iterarg=[0,1,2,3], die=False))
```

Actual: `[0, None, None, 3]` -- job 1 (crashed) and job 2 (legitimately returned `None`) are indistinguishable. Expected per the docstring: `[0, ValueError('boom'), None, 3]`. Note the ordering and the surviving results are correct -- only the failed slot loses its information.

The information does exist if you go through the class (`P.success == [True, False, True, True]`, `P.exceptions == [None, ValueError('boom'), None, None]`), and a `RuntimeWarning` "Only N of M jobs succeeded; see exceptions attribute for details" is raised -- but that message names an attribute the `parallelize()` caller does not have. A downstream `sum(r for r in results if r is not None)`-style aggregation silently drops the failure.

Blast radius: `sc.download()` (`sc_utils.py:886`) relies on this and is internally inconsistent as a result -- its serial fallback stores `output = E` (`sc_utils.py:898`) while its parallel branch stores whatever `sc.parallelize()` returns:

```python
urls = ['http://does-not-exist.invalid/a.txt', 'http://does-not-exist.invalid/b.txt']
sc.download(urls, die=False, save=False, verbose=False)                 # parallel branch
sc.download(urls, die=False, save=False, verbose=False, parallel=False) # serial branch
```

```
parallel: #0. 'a.txt': None
          #1. 'b.txt': None
serial  : #0. 'a.txt': URLError(gaierror(-2, 'Name or service not known'))
          #1. 'b.txt': URLError(gaierror(-2, 'Name or service not known'))
```

**Fix**: in `process_results()`, append `raw['exception'] if raw['exception'] is not None else raw['result']`, or have `_task` set `result = E` in its `die=False` branch -- either makes the returned list self-describing and matches both the docstring and `sc.download()`'s serial path.

### 11. `capture=True` plus a failing job destroys the whole run for two of the built-in parallelizers, and returns a non-string for the default one — `sc_parallel.py:895`

In `_task()` the capture block is

```python
if taskargs.capture:
    with sc.capture() as stdout:
        result = func(*args, **kwargs)
    stdout = str(stdout)
```

If `func` raises, the `with` block unwinds and `stdout = str(stdout)` is skipped, so the live `sc.capture` object -- which holds `self.stdout = sys.stdout`, i.e. a `TextIOWrapper` (`sc_printing.py:1719`) -- is what ends up in `outdict['stdout']` and gets sent back to the parent. Stdlib `pickle` cannot serialize it, so the whole run dies even though `die=False` was explicitly requested to keep going.

```python
import sciris as sc
def f(x):
    print(x)
    if x == 1: raise ValueError('boom')
    return x
if __name__ == '__main__':
    print(sc.parallelize(f, iterarg=[0,1], capture=True, die=False, parallelizer='fast'))
```

Actual:

```
FAILED: TypeError cannot pickle 'TextIOWrapper' instances
control (no capture): [0, None]
```

Expected: `[0, None]` in both cases. Across parallelizers (3 jobs, job 1 raising, `capture=True, die=False`):

```
=== parallelizer=None (multiprocess) ===
  types: ['str', 'capture', 'str']
  join: 'output from job 0\noutput from job 1\noutput from job 2\n'
=== parallelizer=fast ===
  RUN FAILED: TypeError cannot pickle 'TextIOWrapper' instances
=== parallelizer=multiprocessing ===
  RUN FAILED: MaybeEncodingError Error sending result: '[{'result': None, 'success': False, 'exception': ValueError('boom'), ...
=== parallelizer=thread ===
  types: ['str', 'capture', 'str']
  join: 'output from job 0\noutput from job 2\n'
```

So: two of the six parallelizers lose the entire run; the default one survives (dill can pickle it) but leaves `Parallel.stdout` as a mixed list of `str` and `capture` objects; and the `thread` case silently drops the failed job's captured text. Note this is the *only* failure mode of `die=False` that is not contained -- without `capture`, the same run returns `[0, None]` cleanly.

**Fix**: put the `str()` conversion in a `finally`, or capture into a plain `io.StringIO` and read `getvalue()` in a `finally`, so `outdict['stdout']` is always plain text. See also finding 26, a separate `capture=True` defect that affects `thread` even on the success path.

### 12. `loadbalancer()` silently consumes a draw from the global numpy RNG — even on the "no load limit" path where it does nothing else — `sc_parallel.py:1052`

`pause = interval*2*np.random.rand()` uses the *global* legacy RNG, and it is computed unconditionally, before the "return immediately if no max load" guard at `sc_parallel.py:1059`. So calling `sc.loadbalancer()` anywhere inside a seeded pipeline shifts every subsequent draw by one, whether or not the call actually sleeps or checks anything.

```python
import sciris as sc, numpy as np
if __name__ == '__main__':
    np.random.seed(42); print('no call  :', np.random.random(2))
    np.random.seed(42); sc.loadbalancer(maxcpu=1.0, maxmem=1.0); print('after lb :', np.random.random(2))
```

Actual:

```
no call  : [0.37454012 0.95071431]
after lb : [0.95071431 0.73199394]
```

Expected: identical rows; a load balancer should not be observable in the caller's random stream. (`maxcpu=1.0, maxmem=1.0` is the early-return path: it returns `None` in 0.00 s having done nothing but burn the draw.) The effect propagates through `parallelize()`: with an otherwise-identical seed, `sc.parallelize(f, iterarg=[1,2], serial=True)` gives `[0.5488135039273248, 0.7151893663724195]` and the same call with `maxcpu=0.99` gives `[0.7151893663724195, 0.5448831829968969]` -- enabling load balancing changes the numbers.

Note `sc.randsleep()` is clean here: it uses `np.random.default_rng(seed)` (`sc_datetime.py:1528`), so this one line is the only leak.

**Fix**: use a private generator (`np.random.default_rng().random()`, or `random.random()`) for the stagger, and move the computation after the early-return guard. A fresh unseeded `default_rng()` also fixes finding 27, since it draws independent OS entropy in each forked worker.

### 24. A `None` element in `iterarg` is silently not passed to the function — `_task()`, `sc_parallel.py:867`

`_task()` uses `if taskargs.iterval is not None:` to mean "is there an `iterarg`?". But `make_argslist()` sets `iterval = iterarg[index]`, so an `iterarg` element that is legitimately `None` is indistinguishable from "no `iterarg`", and the function is called with no positional argument. This gives silently wrong results when iterating over options such as `[None, 'a', 'b']` or seeds `[None, 1, 2]`.

```python
import sciris as sc
def fd(x=5): return x
if __name__ == '__main__':
    print(sc.parallelize(fd, iterarg=[None, 2]))
    print(sc.parallelize(fd, iterarg=[None, 2], serial=True))
```

Actual: `[5, 2]` in both cases. Expected: `[None, 2]`. If the parameter has no default, the result is `TypeError: missing 1 required positional argument` instead.

**Fix**: in `make_argslist()`, pass an explicit flag (e.g. `has_iterval = iterarg is not None`) through `TaskArgs`, and test that flag in `_task()` instead of `taskargs.iterval is not None`.

### 25. `Parallel.run()` replaces a user's `NotImplementedError` or argless `RuntimeError` with `IndexError` — `sc_parallel.py:624-625`

The Windows `freeze_support` handler catches `RuntimeError` and does `if 'freeze_support' in E.args[0]`. Any `RuntimeError` subclass raised with no args (`raise NotImplementedError`, `raise RuntimeError()`), or with a non-string first arg, crashes the handler, so the caller receives `IndexError` or `TypeError` instead of their own exception. This happens with the default `die=True` in both serial and parallel modes, so `except NotImplementedError:` in calling code stops working.

```python
import sciris as sc
def nie(x): raise NotImplementedError
if __name__ == '__main__':
    sc.parallelize(nie, iterarg=[1,2])
```

Actual: `IndexError: tuple index out of range` (re-verified with `serial=True` as well). Expected: `NotImplementedError`.

**Fix**: `if E.args and isinstance(E.args[0], str) and 'freeze_support' in E.args[0]:` (or `'freeze_support' in str(E)`), and use a bare `raise` in the else branch.

### 26. `parallelizer='thread'` with `capture=True` leaves `sys.stdout` hijacked and misattributes output between jobs — `_task()`, `sc_parallel.py:895-898`

`sc.capture` is a `contextlib.redirect_stdout`, which swaps the process-global `sys.stdout`. With overlapping thread jobs, each job's `__exit__` restores whatever `sys.stdout` was when that job entered, which is often another job's capture buffer. After the run, the parent's `sys.stdout` still points at a dead capture buffer, so every subsequent `print()` in the session silently disappears, and the per-job captured text is attributed to the wrong jobs. (The original audit's "verified clean" result for thread capture only held because its jobs did not overlap.)

```python
import sciris as sc, sys, time
def slowprint(x):
    time.sleep([0.05,0.3,0.1,0.2][x]); print(f'job{x}'); time.sleep([0.3,0.05,0.2,0.1][x]); return x
if __name__ == '__main__':
    real = sys.stdout
    P = sc.Parallel(slowprint, iterarg=range(4), parallelizer='thread', capture=True); P.run()
    ok = sys.stdout is real; sys.stdout = real
    print(ok, [str(s) for s in P.stdout])
```

Actual: `False ['', '', '', 'job0\njob2\njob3\njob1\n']`. Expected: `True ['job0\n', 'job1\n', 'job2\n', 'job3\n']`.

**Fix**: `redirect_stdout` cannot work per-thread. Either reject or warn on `capture=True` with `method == 'thread'` (or fall back to serial capture), or install a single thread-aware stdout proxy for the duration of the run that dispatches writes to a per-thread `StringIO` (via `threading.local`).

### 27. The load balancer's random start stagger is identical in every forked worker, so it staggers nothing — `loadbalancer()`, `sc_parallel.py:1052`

`_task()` calls `sc.loadbalancer(**lbkw)` without `index`, so the stagger is `pause = interval*2*np.random.rand()` from the global RNG. Every forked worker inherits the same RNG state (finding 1), so every worker computes the same pause and then checks the load at the same instant. That defeats the stated purpose ("Give it time to asynchronize"): all workers see the same load reading and start together.

```python
import sciris as sc, time
def stamp(x): return time.time()
if __name__ == '__main__':
    t0 = time.time()
    out = sc.parallelize(stamp, iterarg=range(4), ncpus=4, maxcpu=0.99, interval=1.0)
    print([round(t-t0, 3) for t in out])
```

Actual: `[2.086, 2.088, 2.087, 2.088]`. Expected: start times spread over roughly 0.1-2.1 s.

**Fix**: the same as finding 12's: `pause = interval*2*np.random.default_rng().random()`. An unseeded `default_rng()` draws fresh OS entropy in each process, so this fixes both findings.

## Low severity

### 9. `finalize()` closes the pool but never shuts down the manager, so one process leaks per `Parallel` object — `sc_parallel.py:577`

`make_pool()` starts a `SyncManager` process (lines 408-415) for the shared `globaldict` and the `_Counter` list, but `finalize()` only closes `self.pool`; there is no `self.manager.shutdown()` anywhere in the module (`grep -n "shutdown" sciris/sc_parallel.py` returns nothing). The manager survives for the lifetime of the retained `Parallel` object, so the documented pattern of keeping the object around (the class docstring example ends with `print(P.times)`) leaks one process per object.

```python
import sciris as sc, psutil, time
def f(x): return x
if __name__ == '__main__':
    nk = lambda: len(psutil.Process().children(recursive=True))
    keep = []
    for i in range(5):
        P = sc.Parallel(f, iterarg=[1,2], ncpus=2); P.run(); keep.append(P)
        time.sleep(0.2)
        print(f'{i+1} retained Parallel objects -> live children = {nk()}')
```

Actual:

```
1 retained Parallel objects -> live children = 1
2 retained Parallel objects -> live children = 2
3 retained Parallel objects -> live children = 3
4 retained Parallel objects -> live children = 4
5 retained Parallel objects -> live children = 5
```

Expected: 0 live children after `finalize()`. A single object shows it directly: `P.run_async()` -> 4 children; `P.finalize()` -> 1 child, and that one is the manager (`P.manager` is still a live `multiprocess.managers.SyncManager`). Re-running the *same* object does not accumulate (the old manager is replaced and refcount-collected), so the leak is per-object, not per-run. `sc.parallelize()` is unaffected in practice because it discards the `Parallel` object, which triggers the manager's own finalizer.

Severity: Low (downgraded from Medium on review). The leak costs one idle process per retained object, and `sc.parallelize()` never leaks.

**Fix**: in `finalize()`, when `close_pool` is true, first snapshot everything served by the manager into plain objects (at least `self.globaldict = dict(self.globaldict)`, plus anything else that holds a manager proxy), then call `self.manager.shutdown()` (guarded by `if self.manager:` and the same `try`/`except` used for the pool) and set `self.manager = None`. Correction on review: the originally proposed fix (shut down the manager with no snapshot) breaks documented behavior, because `P.globaldict` is a proxy served by that manager; after `P.manager.shutdown()`, `dict(P.globaldict)` raises `BrokenPipeError: [Errno 32] Broken pipe`, and reading worker writes from `P.globaldict` after the run is a documented use.

### 13. `loadbalancer()` always waits longer than `maxtime`, by a factor of `1 + cpu_interval/interval` (3x measured) — `sc_parallel.py:1067`

`maxtime` is documented as "maximum amount of time to wait to start the task"; it is implemented as a poll *count*, `maxcount = maxtime/float(interval)`, which assumes each poll costs `interval`. Each poll actually costs `cpuload(interval=cpu_interval)` (a blocking `psutil.cpu_percent()`, 0.1 s by default) **plus** `sc.randsleep(interval)` (uniform on `[0, 2*interval]`, mean `interval`). The initial `time.sleep(pause)` of up to `2*interval` is not counted either. So the wait overruns the documented cap by `1 + cpu_interval/interval` -- always, including at the defaults.

```python
import sciris as sc, time
if __name__ == '__main__':
    for maxtime, interval in [(0.5, 0.05), (1.0, 0.1), (2.0, 0.5)]:
        t0 = time.time()
        out = sc.loadbalancer(maxcpu=0.0001, maxmem=0.99, interval=interval, cpu_interval=0.1, maxtime=maxtime, verbose=0)
        print(f'maxtime={maxtime} interval={interval}: actual {time.time()-t0:.2f}s ({out.split(";")[1].strip()})')
```

Actual (two consecutive runs, an idle 20-core box):

```
maxtime=0.5 interval=0.05: actual 1.53s  (process  queued 10 times)
maxtime=1.0 interval=0.1:  actual 2.12s  (process  queued 10 times)
maxtime=2.0 interval=0.5:  actual 3.36s  (process  queued 4 times)
maxtime=0.5 interval=0.05: actual 1.52s  (process  queued 10 times)
maxtime=1.0 interval=0.1:  actual 1.67s  (process  queued 10 times)
maxtime=2.0 interval=0.5:  actual 2.79s  (process  queued 4 times)
```

Expected: <= `maxtime` in every row. At `interval=0.05` the overrun is 3x; at the default `interval=0.5` it is still 1.4-1.7x, i.e. the documented 10-hour ceiling is really 14-17 hours, and a caller who passes a small `interval` to poll responsively (`interval=0.01` is legal -- the floor is 1 ms) gets `maxtime/0.01` polls of >= 0.1 s each, i.e. a 10-hour cap that can block for weeks. Note that the lower bound is `maxcount*cpu_interval` regardless of load, so this is not a load-dependent artifact.

Blast radius: `_task()` calls `sc.loadbalancer(**lbkw)` for every job whenever `maxcpu`/`maxmem` is set, and does not pass `maxtime`, so every such job inherits the 36000 s default and the same overrun factor.

**Fix**: bound the loop on elapsed wall time rather than a poll count, e.g. `t0 = sc.time()` before the loop and `while toohigh and (sc.time()-t0) < maxtime:`, keeping `count` only for the message. Severity: Low (downgraded on review); the overrun is modest at the default settings.

### 14. The `globaldict[_jobkey(index)]` progress writes are vestigial and pollute a user-supplied `globaldict` — `sc_parallel.py:44`

`_jobkey()` exists only to write per-job progress flags into the shared dict (`globaldict[_jobkey(index)] = 0` at line 887 and `= 1` at line 903). Nothing reads them any more: `grep -rn "_jobkey" sciris/ tests/` returns only the definition and those two writes, because v3.3.0 replaced the "sum the global dictionary" progress scheme with `_Counter`. The dead writes are still visible to the user, since the same dict is handed to each worker as the `globaldict` kwarg and exposed as `P.globaldict`.

```python
import sciris as sc
def uses(x, globaldict=None): return x
if __name__ == '__main__':
    P = sc.Parallel(uses, iterarg=[1,2], globaldict={'mine':1}); P.run()
    print(dict(P.globaldict))
```

Actual: `{'mine': 1, '_job0': 1, '_job1': 1}`. Expected: `{'mine': 1}`. Any caller that iterates or aggregates over its own globaldict (`sum(gd.values())`, `len(gd)`, `for k in gd`) sees `njobs` spurious entries. They also cost two manager round-trips per job, which is the very cost `_Counter` was introduced to remove.

**Fix**: delete the two `globaldict[_jobkey(index)]` assignments and `_jobkey()` itself; the counter already supplies the progress information.

### 18. The `maxload` deprecation shim warns and then discards the value — `sc_parallel.py:306`

`set_ncpus()` pops `maxload` out of `self.kwargs`, warns that it has been renamed to `maxcpu`, and assigns `self.maxcpu = maxload` -- but `self.maxcpu` is read nowhere in the module (`grep -n "self\.maxcpu" sciris/sc_parallel.py` -> only line 306), and `self.lbkwargs` was already frozen in `__init__` (line 152). The renamed argument therefore has no effect at all, so a caller migrating from v1.x is told their code still works while the throttle is silently switched off.

```python
import sciris as sc
calls = []; sc.loadbalancer = lambda **kw: calls.append(kw)
def f(x): return x
P = sc.Parallel(f, iterarg=[1,2], serial=True, maxload=0.8); P.run()
print('warned; lb calls =', len(calls), 'lbkwargs =', dict(P.lbkwargs), 'self.maxcpu =', P.maxcpu)
```

Actual: `FutureWarning: sc.loadbalancer() argument "maxload" has been renamed "maxcpu" as of v2.0.0`, then `warned; lb calls = 0 lbkwargs = {'maxcpu': None, 'maxmem': None, 'interval': None} self.maxcpu = 0.8`. Expected: `lb calls = 2` with `maxcpu=0.8`. (The pop itself is correct -- `maxload` does not leak through to the user's function.)

**Fix**: `self.lbkwargs.maxcpu = maxload` instead of `self.maxcpu = maxload` (this works because `lbkwargs` is an `objdict`). Deleting the five-year-old shim entirely is also reasonable.

### 19. `serial=True` on top of an async parallelizer raises instead of running serially — `sc_parallel.py:351`

`serial` is documented as "whether to skip parallelization and run in serial (useful for debugging; equivalent to `parallelizer='serial'`)", i.e. a one-flag debugging switch. `set_method()` honours it (line 334 forces `parallelizer = 'serial'`) but the async check later in the method tests the *original* `self.parallelizer` string against the *new* `self.method`, so flipping `serial=True` on an existing async call is rejected with a message about a combination the user did not ask for.

```python
import sciris as sc
sc.parallelize(lambda x: x, iterarg=[1,2], parallelizer='multiprocess-async', serial=True)
```

Actual: `ValueError: You have specified to use async with "serial", but async is only supported for: multiprocess, multiprocessing.` Expected: `[1, 2]`, run in serial. (The genuinely invalid `parallelizer='serial-async'` must keep raising, and does -- `tests/test_parallel.py:169` covers it.)

**Fix**: skip the async check when `self.serial` is true (or clear `is_async` in the `if self.serial:` branch).

### 20. An empty `globaldict` is silently replaced by a fresh dict for custom parallelizers, so an empty `Manager().dict()` loses all worker writes — `sc_parallel.py:418`

`if self.inputdict:` is a truthiness test on a dict, whereas the companion line 451 (`useglobal = True if self.inputdict is not None else False`) correctly uses `is not None`. For `method == 'custom'` the true branch is what binds the user's actual dict object (`globaldict = self.inputdict`, per the code comment "in case it's something special"), so an empty dict skips it and the workers are handed a throwaway `dict()` instead.

With a plain `{}` and an in-process custom map this is only an object-identity difference (`P.globaldict is gd` is `False`, but the writes still land in `P.globaldict`). The case that actually loses data is the one the branch exists for: an empty `Manager().dict()` passed with a process-based custom parallelizer.

```python
import sciris as sc, multiprocess as mp
def w(x, globaldict=None): globaldict[f'w{x}'] = x; return x
if __name__ == '__main__':
    man = mp.Manager(); pool = mp.Pool(2)
    for init in ({}, {'pre':0}):
        gd = man.dict(init)
        sc.Parallel(w, iterarg=[1,2], parallelizer=pool.map, globaldict=gd).run()
        print(dict(gd))
```

Actual (re-verified):

```
{}
{'pre': 0, '_job0': 1, '_job1': 1, 'w1': 1, 'w2': 2}
```

Expected: the worker writes `w1`, `w2` present in both cases. Whether a custom parallelizer's workers can communicate back through the caller's dict thus depends on whether that dict happened to be non-empty, and the failure is silent. The non-custom methods are unaffected in practice (there the true branch only performs a no-op `update`).

**Fix**: `if self.inputdict is not None:`.

### 28. `-copy` parallelizers deep-copy `globaldict`, so worker writes are lost — `run_async()`, `sc_parallel.py:499-500`

`'serial-copy'` and `'thread-copy'` apply `sc.dcp()` to every `TaskArgs` (line 500), including its `globaldict`. `_Counter` was given a `__deepcopy__` returning `self` for exactly this reason, but `globaldict` was not. So `globaldict` works under `serial`, `thread` and `multiprocess`, but silently does nothing under the `-copy` variants that exist to emulate multiprocess semantics.

```python
import sciris as sc
def gw(x, globaldict=None): globaldict[f'w{x}'] = x; return x
if __name__ == '__main__':
    print(dict(sc.Parallel(gw, iterarg=[1,2], parallelizer='serial-copy', globaldict={'a':0}).run().globaldict))
    print(dict(sc.Parallel(gw, iterarg=[1,2], parallelizer='serial',      globaldict={'a':0}).run().globaldict))
```

Actual: `{'a': 0}` for `serial-copy`, versus `{'a': 0, 'w1': 1, 'w2': 2, ...}` for `serial`. Expected: the worker writes are visible for both. Niche, so low priority.

**Fix**: in the copy branch, deep-copy with a memo that maps the shared dict to itself (`copy.deepcopy(arg, memo={id(self.globaldict): self.globaldict})`), or re-attach `arg.globaldict = self.globaldict` after copying.

## Misplaced `# pragma: no cover`

Twenty-two pragmas sit on reachable paths -- most of them ordinary documented behavior, not defensive corners -- hiding exactly the code that most needs testing. Two (lines 320, 345) are already exercised by `tests/test_parallel.py`; the rest are not. (This section concerns test coverage rather than bugs, so it was not re-graded in the 2026-09-25 review.)

| Line | Branch | Reachable via |
|------|--------|---------------|
| 249 | `if iterarg is not None and iterkwargs is not None:` | `sc.Parallel(f, iterarg=[1,2], iterkwargs={'y':[1,2]})` -> `ValueError`; this is a *documented* call form (finding 6) |
| 259 | `except Exception as E:` (unparseable `iterarg`) | `sc.Parallel(f, iterarg=object())` -> `TypeError: Could not understand iterarg ...` |
| 269 | `if not sc.isiterable(val):` | `sc.Parallel(f, iterkwargs={'x':1})` -> `TypeError: iterkwargs entries must be iterable, not <class 'int'>` |
| 275 | `if len(val) != njobs:` | `sc.Parallel(f, iterkwargs={'x':[1,2],'y':[1,2,3]})` -> `ValueError: All iterkwargs iterables must be the same length, not 2 vs. 3` |
| 282 | `if not isinstance(item, dict):` | `sc.Parallel(f, iterkwargs=[1,2])` -> `TypeError: If iterkwargs is a list, each entry must be a dict ...` |
| 286 | `else:` (bad `iterkwargs` type) | `sc.Parallel(f, iterkwargs=5)` -> `TypeError: iterkwargs must be a dict of lists, a list of dicts, or None ...` |
| 305 | `if maxload is not None:` | `sc.Parallel(f, iterarg=[1,2], maxload=0.8)` -> `FutureWarning` (finding 18) |
| 320 | `if not ncpus > 0:` | `sc.Parallel(f, iterarg=[1,2], ncpus=-3)` -> `ValueError: No CPUs to run on ...`; **already exercised by `tests/test_parallel.py:161`** |
| 345 | `else:` -> `self.method = 'custom'` | `sc.Parallel(f, iterarg=[1,2], parallelizer=pool.map)` -> `P.method == 'custom'`; **already exercised by `tests/test_parallel.py:145`** |
| 402 | `else:` -> `Invalid parallelizer` ("Should be unreachable") | `P = sc.Parallel(f, iterarg=[1,2]); P.run(); P.reset(); P.run()` -> `ValueError: Invalid parallelizer "None"` (finding 4) |
| 592 | `if self.rawresults is None:` | `P = sc.Parallel(f, iterarg=[1,2], parallelizer='multiprocess-async'); P.run_async(); P.finalize(get_results=False)` -> `ValueError: Cannot process results: results not ready yet` |
| 611 | `if not all(self.success):` | `sc.parallelize(bad, iterarg=[0,1,2], die=False)` -> `RuntimeWarning: Only 2 of 3 jobs succeeded ...`; this is the normal `die=False` path |
| 906 | `except Exception as E:` in `_task()` | any failing job; exercised with both `die=True` (raises with the note "Task 1 failed: set die=False to keep going instead.") and `die=False` (returns `[0, None, None, 3]`) -- the single path that implements the documented `die` argument |
| 937 | `if taskargs.callback:` | `sc.parallelize(f, iterarg=[1,2], callback=cb)` -- the callback fired in both serial and multiprocess mode, printing `CALLBACK index 0 result 2` etc.; `callback` is a documented argument |
| 1025 | `if maxload is not None:` | reached by `loadbalancer()`'s *own* docstring example (`maxload=0.5`), which emits the `FutureWarning` (rejected finding 22) |
| 1038 | `if interval is None:` | this is the default: `sc.loadbalancer(maxcpu=0.999, maxmem=0.999, cpu_interval=0.02, maxtime=1)` returns a normal status string; any call that does not pass `interval` lands here |
| 1041 | `if interval < min_interval:` | `sc.loadbalancer(..., interval=1e-5)` -> `UserWarning: sc.loadbalancer() "interval" should not be less than 0.001 s` |
| 1048 | `else:` (`label += ': '`) | `sc.loadbalancer(label='myjob', ...)` -> `'myjob: CPU ... (0.00<0.99), ...'`; `label` is a documented argument |
| 1054 | `else:` (`pause = index*interval`) | `sc.loadbalancer(index=3, interval=0.1, ...)` -> 0.35 s elapsed, message `starting process 3 after 1 tries`; `index` is a documented argument |
| 1059 | `if (not 0 < maxcpu < 1) and ...` | `sc.loadbalancer(maxcpu=None, maxmem=None)` -> returns `None` in 0.00 s; also reached by `maxcpu=1.0, maxmem=1.0` (finding 12) |
| 1080 | `if cpu_toohigh:` | `sc.loadbalancer(maxcpu=0.0001, maxmem=0.99, interval=0.05, cpu_interval=0.05, maxtime=0.2, verbose=True)` -> four `CPU load too high (0.16>0.00)` lines |
| 1083 | `elif mem_toohigh:` | `sc.loadbalancer(maxcpu=1.0, maxmem=0.0001, interval=0.05, cpu_interval=0.05, maxtime=0.2, verbose=True)` -> four `Memory load too high (0.33>0.00)` lines |

Not demonstrated reachable, and probably correctly marked: lines 74 and 83 (`_Counter` exception fallbacks), 468 (`iterkwargs` type re-check in `make_argslist`), 580 (pool close failure), 624 (Windows `freeze_support`).

## Verified clean

Recorded so the same ground isn't re-covered. All of the following were hypothesised, tested by execution, and found correct; together they bound how much of the module can be trusted despite the findings above.

**Ordering.** Deliberately inverted durations (job 0 sleeps longest, the last job sleeps shortest) return results in `iterarg` submission order, not completion order, for every parallelizer tried: `multiprocess`, `multiprocess-async`, `concurrent.futures`, `thread` and `serial`. Ordering is also preserved when a middle job fails with `die=False` (`[0, None, 20]` for `iterarg=[0,1,2]`), i.e. surviving results stay in their correct positions rather than shifting up. Serial and parallel runs of a deterministic function agree exactly (the one property `tests/test_parallel.py:97` already covers), and `die=True` correctly identifies the failing job (`E.__notes__[0]` names the task index and notes `set die=False to keep going instead`). `Parallel.process_results()` keeps `results`, `success`, `exceptions`, `stdout` and `times.jobs` index-aligned throughout.

**The v3.3.0 `_Counter` fix.** `P.counter.value == P.njobs` exactly after the run for all seven parallelizer variants (`serial`, `serial-copy`, `thread`, `thread-copy`, `multiprocess`, `multiprocessing`, `concurrent.futures`), and also when a job failed under `die=False`; no off-by-one and no count above `njobs`. With `progress=True` the drawn bars run `1/n ... n/n` and the final bar is always `n/n` (checked over 8 trials of 8 simultaneous instant jobs, and for staggered durations at n = 2, 3, 4, 8), so the bar cannot finish before the jobs. The only artifact is that under a tight race two workers can print the same value (`... 5, 5, 6 ...`), because each reads the shared count after its own increment -- cosmetic, no wrong final state. `_Counter.__deepcopy__` returning `self` was tested directly via the `-copy` parallelizers, which do `sc.dcp()` on every `TaskArgs`: the count still reaches `njobs`, i.e. the counter is genuinely shared rather than copied per job. `monitor()` on an async run terminates and prints a final `Job 10/10 ... 100%`.

**Resource cleanup on the success path.** Child process count returns to zero after `sc.parallelize()` completes normally -- the pool *is* closed properly outside the failure modes covered by findings 8 and 9. `P.run()` twice on the same object genuinely re-executes (different worker pids and timestamps each time) rather than returning stale results, as does `run_async()`/`finalize()` twice on an async object; a second `finalize()` on an already-finalized async object is idempotent. Repeated runs of one object do not accumulate manager processes. Only the `reset()` path is broken (finding 4). `self._running` is never set to `True` anywhere, so `P.status` never reports `'running'` for a blocking, non-async run -- but since those calls block, no caller can observe the difference; `running`/`ready`/`status` behave correctly for the async path.

**Parent-side global state.** After running the same job through `multiprocess`, `concurrent.futures`, `thread` and `serial` in sequence, `dict(sc.options)`, `np.random.get_state()` and `matplotlib.rcParams` are all byte-identical to before, and `os.environ` is unchanged (an apparent diff turned out to be `PYSIDE6_OPTION_PYTHON_ENUM`, set by matplotlib's backend probing, not by `sc_parallel`). The `maxload` pop mutates `self.kwargs`, not the caller's dict. The bare `except: pass` around the globaldict setup does not mask anything in normal use, since `globaldict` is always a real dict or manager dict by then.

**Aliasing.** For `iterarg=[0,0,0]` with a function returning a fresh list, the three returned objects have distinct ids and mutating `results[0]` leaves `results[1]` and `results[2]` untouched, under `multiprocess`, `thread` and `serial` alike. The `results`/`success`/`exceptions`/`stdout`/`times.jobs` lists are rebuilt from scratch in `process_results()`, so they are not shared between runs. The one genuine serial/parallel divergence is inherent to in-process execution rather than a defect: a function that mutates its own argument sees the parent's object under `serial=True` (`iterarg=[shared, shared]` -> the caller's list becomes `[1, 999, 999]` and both results alias it) whereas under `multiprocess` each worker mutates a copy (`[1, 999]`, distinct objects); `parallelizer='serial-copy'` exists for exactly this, and the docstring flags `serial` as a debugging aid. Hashing/comparing the caller's objects before and after runs under `serial`, `thread`, `thread-copy` and the default `multiprocess` confirms the caller's `kwargs` dict, its nested dict, its `iterarg` list, and a list-of-dicts `iterkwargs` are all unchanged afterwards, and `Parallel.kwargs` is not polluted with the injected `globaldict` key (`_task()` re-merges into a fresh dict via `sc.mergedicts()` before assigning `kwargs['globaldict']`). The only mutation found is internal and benign: `_task()` rewrites `taskargs.iterval` in place to a 1-tuple, so for the non-copying parallelizers `P.argslist[0].iterval` reads `(10,)` instead of `10` after a run; it is idempotent and `make_argslist()` rebuilds the list from `self.iterarg` on a re-run, so nothing downstream is affected.

**Argument plumbing.** `iterarg=[(1,2),(3,4)]` (list of tuples), `iterkwargs={'a':[1,2],'b':[3,4]}` (dict of lists) and `iterkwargs=[{'a':1,'b':3},{'a':2,'b':4}]` (list of dicts) all give `[(1, 3), (2, 4)]`, matching docstring Example 3. `args=(10,)` and `kwargs={...}` and bare `**func_kwargs` all arrive correctly. `iterarg` accepts a list, a list of tuples, an integer (embarrassing mode: `sc.parallelize(g, iterarg=3, args=[1,2])` -> `[3,3,3]`), a `range`, and a numpy array; it rejects a generator and a dict, which are not documented as supported. A single job (`iterarg=[5]`) works; `ncpus=1` works; `ncpus=100` with 4 jobs is clamped to 4 and works. `iterarg=1` (embarrassing mode) does not pass the index to the function, as documented. Note (not a bug, but a trap): a tuple element of `iterarg` is unpacked into multiple positional arguments while a list element or a row of a 2-D array is passed as a single argument, so `sc.parallelize(g, iterarg=[[1,2],[3,4]])` and the numpy-array equivalent both give `[3, 7]` while `sc.parallelize(g, iterarg=[(1,2),(3,4)])` raises `g() takes 1 positional argument but 2 were given`; the tuple-unpacking half of that is explicitly documented by Example 3. `capture=True` works on the success path for `multiprocess` and `serial` (`P.stdout` collects each worker's stdout in job order and it really is suppressed from the terminal), and for `thread` only when jobs do not overlap (see finding 26); `progress=True` renders and reaches 100%; `callback` receives a dict with exactly `index, njobs, args, kwargs, globaldict, outdict` and is called once per job in both serial and parallel mode; `elapsed` is measured around the function call only (the load-balancer wait is deliberately excluded, since `start = sc.time()` comes after it). `globaldict` round-trips (workers' writes are visible in `P.globaldict`) for the manager-backed methods and for `serial`/`thread`, but not for the `-copy` variants (finding 28) or an empty dict with a custom parallelizer (finding 20). Not verified clean on review: an `iterarg` element of `None` is dropped (finding 24).

**The load functions.** The three v3.3.0 aliases are literal name bindings (`sc.cpu_count is sc.cpucount`, `sc.cpu_load is sc.cpuload`, `sc.mem_load is sc.memload` all `True`), so the two naming conventions cannot drift. `cpuload()` returns `psutil.cpu_percent(interval=...)/100` and `memload()` returns `psutil.virtual_memory().percent/100`; both were observed in `[0, 1]`, matching the docstrings' "a float between 0-1" claim, i.e. neither returns a percentage. The only inaccuracy is `cpu_count()`'s docstring calling itself an alias to `multiprocessing.cpu_count()` when it actually calls `multiprocess.cpu_count()` -- the same value, so not reported as a defect. `loadbalancer()` terminates in every configuration tried, including the pathological `maxcpu=0.0001` (bounded by `maxcount`, albeit at the inflated wall time from finding 13) and `maxtime=0`. `verbose=0` **is** respected -- the v3.2.9 changelog claim holds: with `verbose=0` and a deliberately-too-high load nothing is printed, whereas `verbose=None` prints the four "CPU load too high" lines; `verbose=False` is likewise silent and `verbose=True` prints even when the load is fine. `maxcpu`/`maxmem` are both actually enforced and are correctly reported independently (`Memory load too high (0.33>0.00)` fires with `maxcpu=1.0`). `index` staggers as documented (`index=3, interval=0.1` -> 0.35 s). Values `>1` are interpreted as percentages as the docstring comment says (`maxcpu=90, maxmem=90` -> `0.90<`). `interval` is floored at 1 ms with a warning. `maxcpu=None`/`False` and `maxmem=None`/`False` are treated as no limit and return immediately. Non-conflicting top-level `maxcpu`/`maxmem` and non-conflicting `lbkwargs` keys (`label`, `verbose`) all reach `sc.loadbalancer()` intact, including when `ncpus` is also given. One oddity that is not counted as a defect: the early-return path returns `None` while every other path returns the status string, and the early return is additionally conditioned on whether the caller passed `interval` (`default_interval is None`), so `sc.loadbalancer(maxcpu=None, maxmem=None, interval=0.05)` does a full (pointless) load check instead of returning immediately -- the return value is undocumented either way.

**Docstring examples.** The class docstring example runs verbatim and produces `results == [0, 1, 4, 9, ..., 81]`, sensible `P.times` (`started`, `finished`, `elapsed`, per-job `jobs`), and `repr(P) == 'Parallel(jobs=10; cpus=10; method=multiprocess; status=done)'`. All six of `parallelize()`'s docstring examples were executed verbatim and pass, including `assert results1 == results2 == results3` in Example 3, the `parallelizer='multiprocessing'` and `kwargs={'x':3,'y':8}` variants in Example 4, the `pool.map` custom parallelizer in Example 5, and the Dask wrapper in Example 6 (`[[1,1,1,1],[2,2,2,2,2],[3,3,3,3,3,3]]`). `set_method()`'s alias mapping resolves every documented spelling, and `'invalid-parallelizer'` raises `sc.KeyNotFoundError` as tested.

## Rejected on review

These original findings were removed from the severity sections and the summary table after re-verification on 2026-09-25. Numbers are kept so cross-references stay valid.

- **5. `reset()` does not clear `rawresults`** -- NOT WORTH FIXING: only reachable by misuse (processing results on a never-run object, or after an explicit `reset()`), and it still gives a clear error; adding `self.rawresults = None` to `reset()` can ride along with the fix for finding 4.
- **15. `interval` on its own never reaches the load balancer** -- NOT A BUG: `interval` is documented as the pause "for checking load", so it being inert without `maxcpu`/`maxmem` is consistent with the docs, and the proposed gate change would add an unrequested start stagger.
- **16. `iterkwargs` length validation depends on dict key order** -- NOT WORTH FIXING: it needs a zero-length first `iterkwargs` entry plus a non-empty later one, and it still errors (`IndexError`), just with a worse message.
- **17. An empty iterable raises instead of returning `[]`** -- NOT A BUG: this is a deliberate, explicit `ValueError` with a clear message, and returning `[]` would be a design change.
- **21. Documented `ncpus`/`maxcpu` interaction is false** -- NOT A BUG (docs only): the docstring claims are wrong but the code is correct; a cheap docstring tweak is still worthwhile.
- **22. `loadbalancer()` docstring example uses `maxload` and the wrong default** -- NOT WORTH FIXING (docs only): a stale docstring example, trivial to edit.
- **23. Doubled space in `process_str`** -- NOT WORTH FIXING: a cosmetic doubled space in a log message.

## Suggested order of work

1. **Finding 1** -- the RNG-collapse defect. High severity, and the fix (reseed per worker, or at minimum document the requirement and stop calling `serial=True` an equivalent) is small and local relative to its blast radius.
2. **Findings 24, 25, 8** -- silently wrong results for `None` `iterarg` elements, user exceptions replaced by `IndexError`, and leaked processes on the exception path. All are small, local fixes.
3. **Findings 2, 3, 6, 7** -- documented argument types/combinations that plain in-contract calls crash on (`args` as a list, `ncpus` as a float, `iterarg`+`iterkwargs` together, and the `lbkwargs` clobbering). These are all one- or two-line fixes once identified.
4. **Findings 4, 10, 11, 26** -- the `reset()` state bug, the two `die=False` failure modes (result loses the exception; `capture=True` can crash the whole run), and `thread` capture hijacking `sys.stdout`.
5. **Findings 12, 27** -- `loadbalancer()`'s RNG use, which both leaks a draw from the caller's stream and makes the stagger identical across workers; one-line fix for both. Lower urgency since the balancer is opt-in.
6. **Findings 9, 13, 14, 18, 19, 20, 28 and the pragma list** -- low-severity leaks, plumbing and dead code. For finding 9, use the corrected fix (snapshot `globaldict` before shutting down the manager).

Findings 1, 3, 10, 20, 24, 26, 27 and 28 are silent: they produce plausible-looking output (repeated "random" numbers, a disabled load balancer, `None` in place of an exception or of a `None` argument, lost `globaldict` writes, vanished `print()` output) with no error or warning that would prompt investigation. By contrast, findings 2, 6, 7, 8, 11 and 25 announce themselves immediately as a `TypeError`/`ValueError`/pickling error on an otherwise-ordinary call, which is disruptive but at least undeniable -- a caller finds out before shipping results built on top of them.
