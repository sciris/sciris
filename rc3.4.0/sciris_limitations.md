# Sciris limitations encountered in gf_skills_index.py

Notes from trying to replace stdlib usage in `gf_skills_index.py` with sciris equivalents. Tested against sciris 3.3.0.

Adopted without trouble: `sc.loadjson`, `sc.savejson`, `sc.runcommand`, `sc.timer`, `sc.heading`, `sc.getdate`, `sc.time`. The items below are the ones that stayed stdlib, plus one that turned out to be adoptable after all.

## 1. `sc.argparse` assigns bare `--flags` positionally

**Blocking.** This is the only item here that would introduce a silent correctness bug rather than just extra lines.

### Behavior

`sc.argparse` matches bare `--flag` arguments by position rather than by name, so the flag's name is ignored entirely. With three boolean options declared:

```python
args = sc.argparse(full=False, provenance=False, verbose=False)
```

| Command line | Parsed result | Expected |
| --- | --- | --- |
| `--verbose` | `full=True`, `provenance=False`, `verbose=False` | `verbose=True` |
| `--provenance` | `full=True`, `provenance=False`, `verbose=False` | `provenance=True` |
| `--full --verbose` | `full=True`, `provenance=True`, `verbose=False` | `full=True`, `verbose=True` |
| `--bogus` | `full=True`, `provenance=False`, `verbose=False` | error |

The `key=value` form works correctly (`verbose=True` sets `verbose`), so the feature is usable — just not with conventional flag syntax.

In this project that is actively dangerous: `./update --verbose` would set `full=True` and leave verbose off, silently launching a ~300-call full rescan instead of changing the display. Unknown arguments are also accepted silently and assigned to the first parameter, so `--bogus` would do the same.

### What would unblock it

Match `--name` against declared argument names, setting that argument to `True` when it was declared with a boolean default, and reserve positional assignment for arguments given without a `--` prefix. Raising on an unrecognized `--name` rather than silently consuming it positionally would also help. Stdlib `argparse`'s `action='store_true'` is the reference behavior.

## 2. `sc.parallelize` cannot stop queued tasks early

**Blocking for this use case**, though less severe than it first appeared.

### Behavior

My initial assumption — that `sc.parallelize` only returns once every task finishes, making incremental checkpointing impossible — was wrong. The `callback` parameter is called per task as it completes, and receives the result:

```python
{'index': 0, 'njobs': 3, 'args': (1,), 'kwargs': {},
 'outdict': {'result': 2, 'success': True, 'exception': None, 'stdout': '', 'elapsed': 2.4e-06}}
```

That is enough to flush the cache and output JSON as results arrive, which is most of what this project needs.

The remaining gap is cancellation. `gf_skills_index.py` must support Ctrl-C mid-scan: in-flight repos finish, results are flushed, and the next run resumes from the cache. `sc.parallelize` exposes no way to stop the remaining queued tasks — there is no `stop`/`cancel` parameter, and no documented way for a callback to request that the run wind down. With ~300 queued repos, an interrupt would have to be enforced by every worker checking a global flag and returning early, which means every task still gets scheduled and the "stop" is a convention rather than a guarantee.

`concurrent.futures` handles this directly: tasks are submitted in batches, and the loop checks the stop flag at each batch boundary, so nothing further is submitted after an interrupt.

### What would unblock it

Any of:

- A sentinel a `callback` can return (or an exception it can raise) meaning "stop scheduling further tasks", with the completed results still returned.
- A `stop` parameter accepting a `threading.Event`, checked before each task is dispatched.
- Returning a handle exposing the underlying pool/futures so the caller can cancel pending work.

Partial results being returned after an interrupt, rather than the call raising and discarding them, would be the key property.

## 3. No `sc.Lock` / `sc.Event`

**Minor.** `sciris` does not wrap `threading` primitives (`hasattr(sc, 'Lock')` and `hasattr(sc, 'Event')` are both `False`), so `import threading` stays for the two locks guarding rate-limit backoff and progress-bar drawing, plus the stop event.

This is arguably out of scope for sciris. It is only worth noting because `sc.parallelize` users who need coordination between workers currently have to reach for `threading` anyway, so a thin re-export would let simple threaded scripts avoid the import.

### What would unblock it

Re-export `threading.Lock` and `threading.Event` as `sc.Lock` / `sc.Event`. Low value on its own; more useful bundled with the cancellation support in item 2.

## 4. `json` — not a limitation

Listed here because I initially recorded it as one and was wrong.

I assumed `sc.loadjson(string=...)` would obscure the distinction between a malformed API response (retry) and a rate-limit message (back off), because `gh()` branches on `json.JSONDecodeError`. In fact `sc.loadjson` propagates the real exception:

```python
sc.loadjson(string='{not json')  # raises json.JSONDecodeError
```

So `sc.loadjson(string=out)` is a drop-in for `json.loads(out)`. The `import json` could be dropped entirely by catching `ValueError`, which `JSONDecodeError` subclasses. This is worth doing; it just was not part of the change already made.

One small wart: `sc.loadjson` has no `die`/`default` parameter, so "load this file or give me an empty dict" needs a `try/except` wrapper (`load_cache()` in this project). A `default=` argument would collapse that to one line.
