# `sc_settings.py` bug audit

Audit of `sciris/sc_settings.py` (all 834 lines: `parse_env()`, the `ScirisOptions` class, and the module-level `sc.help()`) for genuine defects, with the emphasis the module deserves: global-state correctness. Specifically tested: whether `sc.options.context()` restores exactly the previous state on exit (including on an exception and when nested), whether `set()`/`reset()` round-trip, whether invalid option names and values are rejected, whether the `SCIRIS_*` environment variables are read *and applied* with the right type coercion, whether `dpi`/`font`/`fontsize`/`backend`/`style` take effect and can be undone, and whether `with_style()` leaks rcParams between calls. Deliberately **out of scope**: unhelpful errors on deliberately wrong types, style, naming, missing type hints, performance, and the `font_size`/`font_family`/`show_type` deprecation shims as such.

**Method**: line-by-line reading, then one executed script per hypothesis. Every "actual" line below was produced by running the code against the editable install (Sciris 3.3.0, Matplotlib 3.11.1, NumPy 2.4.6, Python 3.13.9, commit `2d69aad`), and every finding was reproduced a second time from the minimal snippet shown, in a fresh interpreter, before being recorded. Because the subject is global state, each repro below is a complete standalone program: run it in its own process. Unless stated otherwise the environment is `MPLBACKEND=agg` with no `SCIRIS_*` variables set.

**Independent re-verification.** This document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`): every repro was re-run in a fresh subprocess and the line numbers were confirmed against the current source. As a result, 9 findings were confirmed (some with corrected fixes), 1 was rewritten (5), 5 low-value items were moved to "Rejected on review" (11-15), and 1 missed bug was added (16). Original finding numbers are kept stable, so the numbering has gaps.

**Nothing in this document has been applied.** All fixes are described, not made.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `sc.options.context()` / `__exit__` | Nested contexts raise `AttributeError` on the outer exit and leak the outer setting permanently | 377 |
| 2 | High | `sc.options.set('defaults')` / `reset()` | `kwargs = self.orig_options` (no copy) lets `reset()` overwrite the stored *default* backend with `'agg'`, and aliases the live `rc` dict to its stored default | 309 |
| 3 | High | `sc.options.set()` | An invalid value is stored before validation, so one bad `dpi` poisons every later `sc.options()` call | 345 |
| 4 | Medium | `ScirisOptions.__init__` | The Matplotlib-related `SCIRIS_*` env vars are read but never applied; `SCIRIS_BACKEND=agg` (used by `tests/pytest.ini`) does not switch the backend | 708 |
| 5 | Medium | `with_style()`, `use_style()` | Called with no style, they ignore the `style` option because `kwargs['style'] = None` defeats the `self.style` fallback | 642 |
| 6 | Medium | `sc.options.reset()` | Does not restore rcParams touched by a named Matplotlib style: 15 keys stay modified | 579 |
| 7 | Medium | `ScirisOptions.__setitem__` | The lock is bypassed by `update()`/`setdefault()`: a typo silently becomes a new option, and real options don't reach Matplotlib | 179 |
| 8 | Medium | `sc.help()` | The documented `flags` argument crashes with `TypeError` whenever `ignorecase` is left at its default | 760 |
| 16 | Medium | `sc.options.context(backend=...)` | The backend is not restored on exit, because the saved default `''` is stored as-is and then skipped by `set_matplotlib_global()` | 343, 399 |
| 9 | Low | `sc.options.help(detailed=True)` | Advertises `SCIRIS_SHOWTYPE` and `SCIRIS_FONTSIZE`; the variables actually read are `SCIRIS_SHOW_TYPE` and `SCIRIS_FONT_SIZE` | 506 |
| 10 | Low | `sc.options.help()` | `output=True` is silently ignored unless `detailed=True` | 494 |

Totals: 3 High, 6 Medium, 2 Low (11 open findings). Findings 11-15 were rejected on review; see "Rejected on review" near the end.

## Recurring patterns

**One global slot for re-entrant state.** `context()` stores the pre-entry values in a single attribute, `on_entry`, and `__exit__` deletes it. That is fine exactly once; nested use (1) clobbers it and then crashes, and a failed `context()` strands it (rejected item 15). A stack (a list of dicts pushed/popped) fixes both, and would also make `__exit__` idempotent. Separately, what `context()` saves is the *option value*, not the Matplotlib state, so an option whose default is a "not yet discovered" placeholder (`backend = ''`) cannot be restored (16).

**Mutate first, validate later.** `set()` writes the new value into the options dict at line 345 and only then hands it to Matplotlib (349) or to `use_style()` (354), where the actual validation happens. So every rejected value survives in the options object, and a bad `dpi` is re-applied — and re-rejected — on the next call (3). Validating (or try/restore-ing) around lines 345-354 would make `sc.options` transactional.

**Options that are stored but never applied.** The options dict is treated as the source of truth in some paths and as a passive record in others: nothing calls `set()` at import, so the Matplotlib env vars do nothing (4); `with_style()`/`use_style()` never consult `self.style`, so the `style` option does nothing after the moment it is assigned (5); and `update()` writes values that never reach Matplotlib (7). All three are the same missing link between the dict and `set_matplotlib_global()`/`use_style()`. The fixes need care, though: `set()` calls `use_style()` on *every* call, so any fix that makes that path re-apply the full style would clobber the user's own rcParams (see the corrected fixes for 5 and 6).

## High severity

### 1. Nested `sc.options.context()` crashes on exit and leaks the outer setting — `sc_settings.py:377`

`context()` stores the pre-entry values in a single attribute (`self.setattribute('on_entry', ...)`, line 377), and `__exit__` deletes it (line 223). Nesting therefore overwrites the outer block's saved state with the inner block's, and the outer `__exit__` finds no `on_entry` at all: it takes the `except AttributeError` branch (line 224, which is marked `# pragma: no cover`) and raises a misleading "Please use sc.options.context() if using a with block". The outer block's original value is never restored, so `sc.options` is left permanently modified.

```python
import sciris as sc
try:
    with sc.options.context(dpi=111):
        with sc.options.context(dpi=222):
            pass
except AttributeError as E:
    print('AttributeError:', E)
print('dpi is now', sc.options.dpi, '(expected 100)')
```

Actual:

```
AttributeError: Please use sc.options.context() if using a with block
dpi is now 111 (expected 100)
```

Expected: no exception, and `dpi is now 100`.

Blast radius: `context()` is the module's only advertised way to change non-plotting options temporarily (`sc.options.help()` and the class docstring both point at it), and nesting is the normal consequence of two library functions both wanting a temporary setting — e.g. a `with sc.options.context(aspath=True):` block inside user code that itself calls a Sciris function using a context. `tests/test_settings.py:23` exercises only the single-level case. Nothing else in `sciris/` calls `context()`, so this bites downstream code (Starsim/Covasim-style callers), not Sciris itself.

**Fix**: make the saved state a stack — `context()` appends `{k:self[k] for k in kwargs}` to a list attribute, `__exit__` pops the last entry and only deletes the attribute when the list is empty. That also removes the need for the `except AttributeError` fallback to double as the "you didn't call context()" error, which should instead be a check for an empty stack.

### 2. `sc.options.reset()` overwrites the stored default backend — `sc_settings.py:309`

`set('defaults')` does `kwargs = self.orig_options` — an alias, not a copy. Line 326 then writes `kwargs['backend'] = 'agg'` when the *default* value of `interactive` is falsy, which mutates `orig_options` itself. With `SCIRIS_INTERACTIVE=0` (a documented environment variable), a single `sc.options.reset()` permanently replaces the stored default backend with `'agg'`, after which `sc.options(interactive=True)` restores `'agg'` instead of the real backend, and there is no way to get interactivity back in that session. The lazy backend-discovery at lines 405-406 (`if not self.orig_options['backend']`) can no longer fire either, because the slot is no longer empty.

Repro (note the environment variable; run as `SCIRIS_INTERACTIVE=0 python repro.py` on a machine with an interactive backend available):

```python
import matplotlib.pyplot as plt
import sciris as sc
print('start backend      :', plt.get_backend())
print('default backend    :', repr(sc.options.get_default('backend')))
sc.options.reset()
print('default backend now:', repr(sc.options.get_default('backend')))
sc.options(interactive=False); print('interactive=False ->', plt.get_backend())
sc.options(interactive=True);  print('interactive=True  ->', plt.get_backend())
```

Actual:

```
start backend      : qtagg
default backend    : ''
default backend now: 'agg'
interactive=False -> agg
interactive=True  -> agg
```

Expected: `default backend now: ''` (or `'qtagg'`), and `interactive=True  -> qtagg`.

For comparison, the same round trip *without* an intervening `reset()` works correctly (`qtagg -> agg -> qtagg`), which confirms the mutation of `orig_options` is the sole cause. The `reset()` here is an ordinary, side-effect-free-looking call — `sc.options('defaults')` and `sc.options.set('default')` are the same path.

Blast radius: `orig_options` is the reference used by `get_default()`, `changed()`, and every `value in [None, 'default']` reset inside `set()`, so corrupting it silently changes what "default" means for the rest of the process. `changed('backend')` also starts reporting `False` for a backend the user never chose.

**Second consequence, no environment variables needed** (found on re-verification). The same alias also affects `rc`. After any `sc.options.reset()`, the per-key loop runs `super().__setitem__('rc', self.orig_options['rc'])`, so the live `rc` dict and the stored default become the same object. Any later in-place change to `sc.options.rc` then silently rewrites the default too:

```python
import sciris as sc
sc.options.reset()
print('alias:', sc.options.rc is sc.options.orig_options['rc'])
sc.options.rc['axes.grid'] = True
print('default rc:', sc.options.get_default('rc'), '| changed:', sc.options.changed('rc'))
```

Actual: `alias: True` then `default rc: {'axes.grid': True} | changed: False`. Expected: `alias: False`, `default rc: {}`, `changed: True`. Because it needs no environment variable, this is the more commonly hit form of finding 2.

**Fix**: `kwargs = sc.dcp(self.orig_options)` at line 309. Use `sc.dcp` rather than a shallow `dict(self.orig_options)`: a shallow copy fixes the backend case but not the `rc` alias, since the nested dict would still be shared. The `interactive` block at 319-326 then writes into the copy and no longer touches `orig_options`.

### 3. An invalid option value is stored before it is validated, breaking all later calls — `sc_settings.py:345`

`set()` writes the value into the options dict (`super().__setitem__(key, value)`, line 345) and *then* pushes it to Matplotlib (line 349) or to `use_style()` (line 354), which is where validation actually happens. When validation fails, the bad value is already committed. Worse, `set()` ends with an unconditional `use_style()`, and `with_style()`'s `pop_keywords()` re-injects every *changed* option into the rc dict — so the stored garbage is re-validated and re-rejected on every subsequent call, including calls that touch a completely unrelated option.

```python
import sciris as sc
try: sc.options(dpi='banana')
except ValueError as E: print('1st:', E)
print('sc.options.dpi =', repr(sc.options.dpi))
try: sc.options(aspath=True)
except ValueError as E: print('2nd (valid call):', E)
```

Actual:

```
1st: Key figure.dpi: Could not convert 'banana' to float
sc.options.dpi = 'banana'
2nd (valid call): Key figure.dpi: Could not convert 'banana' to float
```

Expected: the first call raises and leaves `sc.options.dpi == 100`; the second call succeeds.

For `style` it is only half as bad: `sc.options(style='nosuchstyle')` raises from `_handle_style()` (line 588) and leaves `sc.options.style == 'nosuchstyle'`, but it does *not* break later calls, because `with_style()` never re-validates `self.style` (see finding 5), so a later `sc.options(sep='.')` succeeds. Note also that in the second call above, `aspath` *was* set to `True` before the exception was raised, so a multi-key `set()` applies keys up to the failure and then aborts — a partially applied change with no rollback. `sc.options.reset()` is the only recovery, and only because it rewrites every key from `orig_options`.

Blast radius: any user-facing wrapper that forwards a config value into `sc.options()` (e.g. reading `dpi` from a YAML file) turns one bad entry into a permanently broken `sc.options` for the rest of the session. The partial-application problem is visible in the module's own class docstring: its example `sc.options.set(fontsize=18, show=False, backend='agg', precision=64)` (line 152) raises `ValueError: Option "show" not recognized` (neither `show` nor `precision` is an option), after `fontsize=18` has already been applied. That example should drop the two invalid keys.

**Fix**: in the per-key loop, capture `old = self[key]`, apply, and restore `old` in an `except` around lines 345-351; likewise wrap the trailing `use_style()` (354) so a bad `style` is rolled back. Validating `dpi` (numeric) and `style` (against `plt.style.library` plus the three aliases) before assignment would be even cheaper.

## Medium severity

### 4. The Matplotlib-related `SCIRIS_*` environment variables are read but never applied — `sc_settings.py:708`

`get_orig_options()` reads `SCIRIS_DPI`, `SCIRIS_FONT`, `SCIRIS_FONT_SIZE`, `SCIRIS_BACKEND`, `SCIRIS_INTERACTIVE` and `SCIRIS_STYLE` into the options dict, but the module-level construction (`options = ScirisOptions()`, line 708) only calls `set_show_type()` (line 709) — it never calls `set()`, `set_matplotlib_global()`, or `use_style()`. So the values are recorded and reported correctly but have no effect on Matplotlib until the user happens to call `sc.options(...)` for some other reason. The class docstring states "Each setting can also be set with an environment variable, e.g. SCIRIS_DPI", and `tests/pytest.ini` sets `SCIRIS_BACKEND=agg` precisely to keep the test suite headless.

Run as `SCIRIS_BACKEND=agg SCIRIS_DPI=200 SCIRIS_FONT_SIZE=18 SCIRIS_STYLE=fancy python repro.py` (with no `MPLBACKEND`):

```python
import matplotlib.pyplot as plt
import sciris as sc
print('backend  option', repr(sc.options.backend), '| plt.get_backend()', plt.get_backend())
print('dpi      option', sc.options.dpi, '| rcParams figure.dpi', plt.rcParams['figure.dpi'])
print('fontsize option', repr(sc.options.fontsize), '| rcParams font.size', plt.rcParams['font.size'])
print('style    option', repr(sc.options.style), '| rcParams axes.facecolor', plt.rcParams['axes.facecolor'])
sc.options.reset()
print('after reset(): dpi rc', plt.rcParams['figure.dpi'], '| backend', plt.get_backend())
```

Actual:

```
backend  option 'agg' | plt.get_backend() qtagg
dpi      option 200 | rcParams figure.dpi 100.0
fontsize option '18' | rcParams font.size 10.0
style    option 'fancy' | rcParams axes.facecolor white
after reset(): dpi rc 200.0 | backend agg
```

Expected: `figure.dpi 200.0`, `font.size 18.0` and `plt.get_backend() agg` immediately after `import sciris`. The last line is the proof of mechanism: an explicit `reset()` (which does route through `set()`) applies exactly the values the env vars asked for, so only the import-time call is missing. (`SCIRIS_STYLE` stays broken even after `reset()` for the separate reason in finding 5.)

Blast radius: `SCIRIS_BACKEND=agg` is the documented way (`CLAUDE.md`, `tests/pytest.ini`, `.github/workflows`) to run Sciris without a display, and it does not work — anything that actually opened a window would still try to. `SCIRIS_SEP`, `SCIRIS_ASPATH`, `SCIRIS_SHOW_TYPE` and `SCIRIS_JUPYTER` are unaffected, because those are read out of the dict at point of use (or applied by the explicit `set_show_type()` call).

**Fix**: after constructing `options`, apply the Matplotlib-affecting settings once, e.g. call `options.set_matplotlib_global(k, options[k])` for the keys whose env var was actually present (checking `os.getenv` so an unset variable does not force `plt.get_backend()`, which the comment on line 282 warns is slow), and apply `options.style` if `SCIRIS_STYLE` was set. Guarding on "env var present" keeps import cheap and avoids changing behaviour for users who set nothing.

### 5. `use_style()` and `with_style()` with no style ignore the `style` option — `sc_settings.py:642`

`with_style()` intends `self.style` to be the fallback when no style is passed: line 647 reads `style = kwargs.pop('style', self.style)`. But line 642 has already done `kwargs['style'] = style` unconditionally, so when `style is None` the key *exists* with value `None` and `pop()` never reaches its default. `_handle_style(None)` then returns `self.rc` (normally `{}`) plus any changed option keys, so no style is applied. Consequently `sc.options.use_style()` — documented as "Shortcut to set Sciris's current style as the global default", with a worked example — does not apply the current style, and `with sc.options.with_style():` is effectively an empty context.

```python
import matplotlib.pyplot as plt
import sciris as sc
sc.options.set(style='fancy', use=False)
sc.options.use_style()
print('use_style(): axes.facecolor =', plt.rcParams['axes.facecolor'], '(expected #f2f2ff)')
with sc.options.with_style():
    print('with_style(): axes.facecolor =', plt.rcParams['axes.facecolor'], '(expected #f2f2ff)')
```

Actual:

```
use_style(): axes.facecolor = white (expected #f2f2ff)
with_style(): axes.facecolor = white (expected #f2f2ff)
```

Expected: `#f2f2ff` in both lines, since `sc.options.style == 'fancy'`.

Passing the style explicitly works (`sc.options.use_style('fancy')` -> `#f2f2ff`), so the only broken case is the no-argument fallback — which is the case that makes a stored `sc.options.style` (and hence `SCIRIS_STYLE`) meaningful after the moment it is assigned. `sc.options(style='fancy')` works because `set()` forwards `kwargs.get('style')` explicitly on line 354.

*Correction on re-verification*: the original version of this finding also claimed that a later `sc.options(dpi=150)` "silently drops the user's chosen style". That is false. That call runs `use_style(style=None)`, which applies only a partial dict, and `plt.style.use()` of a partial dict leaves the fancy keys in place: `axes.facecolor` is still `#f2f2ff` after `sc.options(style='fancy'); sc.options(dpi=150)`.

Blast radius: the no-argument forms of `use_style()` and `with_style()` are advertised in `sc.options.help()` and the docstrings; `tests/test_settings.py:25` only ever passes a style explicitly.

**Fix**: keep the `self.style` fallback only for direct user calls to `with_style()`/`use_style()`, and do not let it apply to the `use_style()` call that `set()` makes at the end of every call. For example, add a private flag (`_fallback=True` by default, with `set()` passing `_fallback=False`) or a sentinel default, and at line 642 insert the key only when a style was actually given or the fallback is wanted. Leave line 354 as `kwargs.get('style')`, so `set()` only passes a style when `'style' in kwargs`.

*Why the original fix was wrong*: the original proposal (insert the key at line 642 only when not `None`, and change line 354 to `kwargs.get('style', self.style)`) would make every `sc.options(...)` call — including ones for non-plotting options such as `aspath` or `sep` — re-apply the whole Sciris style (at least the 10 keys in `style_default`), overwriting rcParams the user has set themselves. Verified: after `plt.style.use('ggplot')`, the current code keeps `axes.facecolor = #E5E5E5` after `sc.options(dpi=150)`, while the proposed fallback path (`use_style(style=sc.options.style)`) resets it to `white`.

### 6. `sc.options.reset()` does not restore rcParams set by a named Matplotlib style — `sc_settings.py:579`

`_handle_style()` builds a *partial* rc dict: for a named style it starts from `style_default` (10 keys, lines 36-48) and updates it with the requested style. Applying it goes through `plt.style.use(dict)`, which only overwrites the keys present in the dict. So switching to any style with a broader footprint than those 10 keys — i.e. any entry of `plt.style.library` — and then calling `sc.options.reset()` leaves the remaining keys permanently modified, with nothing in `sc.options` recording that they changed.

```python
import matplotlib.pyplot as plt
import sciris as sc
before = dict(plt.rcParams)
sc.options(style='fivethirtyeight')
sc.options.reset()
diff = [k for k in before if repr(before[k]) != repr(plt.rcParams[k])]
print(len(diff), 'rcParams not restored:', sorted(diff)[:6], '...')
print('axes.linewidth', before['axes.linewidth'], '->', plt.rcParams['axes.linewidth'])
```

Actual:

```
15 rcParams not restored: ['axes.edgecolor', 'axes.labelsize', 'axes.linewidth', 'axes.prop_cycle', 'axes.titlesize', 'figure.subplot.bottom'] ...
axes.linewidth 0.8 -> 3.0
```

Expected: `0 rcParams not restored`. The full leaked set is `axes.edgecolor`, `axes.labelsize`, `axes.linewidth`, `axes.prop_cycle`, `axes.titlesize`, `figure.subplot.bottom`, `figure.subplot.left`, `figure.subplot.right`, `lines.solid_capstyle`, `patch.edgecolor`, `patch.linewidth`, `xtick.major.size`, `xtick.minor.size`, `ytick.major.size`, `ytick.minor.size`. The colour cycle being among them means every plot in the process keeps the wrong palette after a "reset".

The built-in `'simple'` and `'fancy'` styles are unaffected because every key they touch is also in `style_default` — which is exactly why this was easy to miss. The same leak applies to anything put in `options.rc` whose keys are outside `style_default`. Option *values* do round-trip perfectly (verified: `to_dict()` is identical before and after), so `reset()` looks successful.

**Fix**: have `reset()`/`set('defaults')` restore Matplotlib properly rather than by applying a 10-key patch: snapshot `dict(plt.rcParams)` when `ScirisOptions` is constructed, excluding `backend` and the other keys in `matplotlib.style.core.STYLE_BLACKLIST`, and restore from that snapshot when `style` is reset to default. Apply this only on the `style='default'`/reset path, not on every `set()` call, so that user rc changes are not clobbered (the same trap as the original fix for finding 5). Do not use `mpl.rcdefaults()`: it would also wipe the user's `matplotlibrc` and any rcParams they set themselves. Alternatively, record the keys each style application touched so they can be reverted individually.

### 7. `sc.options.update()` and `setdefault()` bypass the option lock — `sc_settings.py:179`

`ScirisOptions` overrides `__setitem__` to reject unknown keys and to route known ones through `set()` (the changelog entry is "*New in version 3.2.5:* locked attributes to prevent accidental modification"). But the inherited `dict.update()` and `dict.setdefault()` do not go through `__setitem__`, so they neither validate the key nor apply the value. A typo becomes a permanent new option, and a correctly spelled option is written into the dict without ever reaching Matplotlib — after which `changed()` reports it as modified and `with_style()` will start injecting it.

```python
import matplotlib.pyplot as plt
import sciris as sc
sc.options.update(dpi=300, dpii=999)
print('dpi option', sc.options.dpi, '| rcParams figure.dpi', plt.rcParams['figure.dpi'])
print('bogus key present:', 'dpii' in sc.options, '->', sc.options.dpii)
```

Actual:

```
dpi option 300 | rcParams figure.dpi 100.0
bogus key present: True -> 999
```

Expected: either a `KeyNotFoundError` for `dpii` (as `sc.options['dpii'] = 5`, `sc.options.dpii = 5` and `sc.options(dpii=5)` all correctly give), or at minimum `figure.dpi` actually set to 300. `sc.options.setdefault('dpii', 5)` behaves the same way. `update()` is a plausible thing to reach for when applying a dict of settings (`sc.options.update(sc.loadjson('opts.json'))` looks like the obvious companion to `sc.options.save()`), and it silently half-works. Impact is low in practice, but it defeats the lock added in 3.2.5.

**Fix**: override `update()` (and `setdefault()`) on `ScirisOptions` to delegate to `self.set(**kwargs)`, so both the key check and `set_matplotlib_global()` run. `__init__` (line 167) would then need `super().update(options)` or `dict.update` directly for the initial population.

### 8. `sc.help(flags=...)` always crashes with the default `ignorecase` — `sc_settings.py:760`

`flags` is collected into a list and `re.I` is appended when `ignorecase` is true (lines 758-760), then splatted: `re.findall(pattern, line, *flags)` (line 797). With the default `ignorecase=True`, supplying any `flags` therefore passes two positional flag arguments to `re.findall()`, which takes at most three arguments in total.

```python
import re, sciris as sc
try: sc.help('json', flags=[re.M])
except TypeError as E: print('TypeError:', E)
```

Actual:

```
TypeError: findall() takes from 2 to 3 positional arguments but 4 were given
```

Expected: the search runs with `re.M | re.I`. The documented argument is usable only in the one combination `ignorecase=False` with a single flag; the docstring describes it as "additional flags to pass to `re.findall()`", i.e. explicitly additive to `re.I`. `tests/test_settings.py` never passes `flags`.

**Fix**: combine flags bitwise instead of positionally — build `flagval = 0`, OR in each entry of `sc.tolist(flags)`, OR in `re.I` if `ignorecase`, and call `re.findall(pattern, line, flagval)`. (Confirmed correct on re-verification.)

### 16. `sc.options.context(backend=...)` does not restore the backend on exit — `sc_settings.py:343` and `:399`

*Added on re-verification (2026-09-25).* The default value of `options.backend` is `''`, meaning "not yet discovered". `context()` therefore saves `on_entry = {'backend': ''}`, and `__exit__` calls `set(backend='')`. In `set()`, `''` is not in `[None, 'default']` (line 343), so it is stored as-is, and `set_matplotlib_global('backend', '')` then skips the switch because of the `if value:` guard (line 399). The backend stays switched after the block, and the option now reads `''` while Matplotlib is on the other backend.

Run with `MPLBACKEND=pdf` (or on a desktop with `qtagg`):

```python
import matplotlib.pyplot as plt
import sciris as sc
print('before', plt.get_backend())
with sc.options.context(backend='svg'):
    print('inside', plt.get_backend())
print('after', plt.get_backend(), repr(sc.options.backend))
```

Actual:

```
before pdf
inside svg
after svg ''
```

Expected: `after pdf 'pdf'` (or `after pdf ''`).

`sc.options(backend='default')` does work, because it substitutes the lazily discovered `orig_options['backend']`, and `context(interactive=False)` also works, because it goes through `orig_options['backend']`. Only the plain `backend` key fails. Blast radius: `with sc.options.context(backend='agg'):` is a natural pattern for rendering one plot headlessly, and it leaves the whole session on the temporary backend.

**Fix**: in `set()`, treat an empty backend as "default": `if value in [None, 'default'] or (key == 'backend' and not value): value = self.orig_options[key]`. This works because by exit time `set_matplotlib_global()` has already stored the real backend in `orig_options['backend']` (lines 405-406). Alternatively, `context()` could record `plt.get_backend()` in `on_entry` when `'backend' in kwargs` and `self.backend` is empty.

## Low severity

### 9. `sc.options.help(detailed=True)` prints the wrong environment variable names — `sc_settings.py:506`

The help output derives the variable name mechanically as `f'SCIRIS_{key.upper()}'` (with the comment "NB, hard-coded above!"), which is wrong for the two option names that contain an implicit word break: `showtype` is read from `SCIRIS_SHOW_TYPE` (line 258) and `fontsize` from `SCIRIS_FONT_SIZE` (line 273).

```python
import sciris as sc
with sc.capture(): d = sc.options.help(detailed=True, output=True)
print('help says:', d.showtype.variable, '/', d.fontsize.variable)
```

Actual: `help says: SCIRIS_SHOWTYPE / SCIRIS_FONTSIZE`. Following that advice does nothing (`SCIRIS_SHOWTYPE=1 SCIRIS_FONTSIZE=20` -> `showtype False`, `fontsize '10.0'`), whereas the real names work (`SCIRIS_SHOW_TYPE=1 SCIRIS_FONT_SIZE=20` -> `showtype True`, `fontsize '20'`).

**Fix**: store the environment variable name alongside each option in `get_orig_options()` (a third `objdict`, e.g. `optenv.showtype = 'SCIRIS_SHOW_TYPE'`) and read it in `help()` instead of reconstructing it — which also removes the "hard-coded above" hazard for future options.

### 10. `sc.options.help(output=True)` is ignored unless `detailed=True` — `sc_settings.py:494`

The `if not detailed:` early return (lines 494-496) prints the docstring and returns before the `if output:` branch, so `sc.options.help(output=True)` returns `None`. Actual: `returned None`; expected: the `objdict` of options (as `help(detailed=True, output=True)` returns). **Fix**: build and return `optdict` regardless of `detailed`, using `detailed` only to decide what to print.

## Misplaced `# pragma: no cover`

Each line below was demonstrated reachable from public API in this audit.

| Line | Code | How it is reached |
|---|---|---|
| 224 | `except AttributeError as E:` in `__exit__` | Nested `sc.options.context()` (finding 1) |
| 333 | `if key in rename.keys():` (deprecated names) | `sc.options(font_size=18)` — the example in this module's own docstring, line 7 |
| 337 | `if key not in self.keys():` (unknown option) | `sc.options(dpii=5)` raises the `ValueError` from line 341 |
| 587 | `else:` (unknown style) in `_handle_style()` | `sc.options(style='nosuchstyle')` raises the `ValueError` from line 589 |
| 638 | `if isinstance(style, dict):` in `with_style()` | `with sc.options.with_style({'xtick.alignment':'left'}):` — already in `tests/test_settings.py:28` |
| 805 | `if not len(matches):` in `sc.help()` | `sc.help('zzqqxyzzy')` prints "No matches for ... found among 287 available functions." |

Correctly placed, for the record: `111` (`parse_env` with an unrecognised `which`, only reachable with a bad argument), `417` (`sc.isjupyter()`), and `574`/`590` inside `_handle_style()` — the dict branch and `reset=True` branch are genuinely dead, since the only caller (line 648) has already converted dicts to kwargs and always passes `reset=False`. Finding 5 and rejected item 14 are about that dead code rather than about the pragmas. This table was confirmed accurate on re-verification, but it does not affect behavior and was not scored.

## Verified clean

**`parse_env()`.** The four type coercions were checked against all the ways a variable can be falsy, and they are distinct where they should be: with `which=bool` and `default=True`, `'0'`, `'false'` and `''` all give `False` while *unset* gives `True` (so an explicitly-empty variable is not confused with an absent one); `'f'`, `'0.0'` and `'none'` are likewise `False`, and any other non-empty string is `True`. A falsy non-string default is passed through rather than being replaced by the type's zero (`parse_env('X', default=0, which=int)` -> `0`), and a truthy default is coerced (`default=3.5, which=float` -> `3.5`). `which=None` returns the raw value uncast, as documented, and `which` accepts both the actual types (`int`, `float`, `bool`, `str`) and their string spellings. The non-plotting variables `SCIRIS_SEP`, `SCIRIS_ASPATH`, `SCIRIS_SHOW_TYPE` and `SCIRIS_JUPYTER` all take effect correctly at import (they are read from the dict at point of use, or applied by the explicit `set_show_type()` on line 709) — only the Matplotlib-affecting ones are inert (finding 4).

**Single-level `context()`.** Tested and correct in every respect I could think of apart from nesting and the `backend` key (finding 16): it restores the option value, it restores the rcParams it touched (`figure.dpi` went `100.0 -> 333.0 -> 100.0`), it restores correctly when the body raises (`aspath` back to `False` after a `RuntimeError` propagated out), and it correctly skips keys whose value the body left unchanged. `context()` with no keyword arguments is also harmless.

**`with_style()` as a context manager.** No leakage between calls: because it returns `plt.style.context(rc)`, all rcParams are restored on exit, including after an exception raised inside the block (a full `dict(plt.rcParams)` diff before and after showed zero differences). Entering `with_style('fancy')`, `'simple'`, `'default'` and `'fivethirtyeight'` in turn leaves nothing behind. The `kwargs`-not-in-`plt.rcParams` check (line 675) correctly raises `KeyError`, and `pop_keywords()` correctly uses `is not None` so that `grid=False` and `facecolor=None` are distinguished. `self.rc` is not mutated by `with_style()` (the aliasing `rc = self.rc` at line 573 is only dangerous in the dead dict branch; the separate `rc`/`orig_options['rc']` alias created by `reset()` is covered in finding 2).

**Option round-trip.** Setting all eight scalar options away from their defaults (`dpi`, `fontsize`, `font`, `sep`, `aspath`, `showtype`, `style`, `jupyter`) and then calling `reset()` reproduces `to_dict()` exactly — no key is missed by `reset()`, and the comparison is on values, not just keys. `showtype` round-trips its NumPy side effect too (`np.float64(3)` prints as `np.float64(3.0)` with `showtype=True` and as `3.0` after `sc.options(showtype='default')`). `sc.options.save()` / `sc.options.load()` also round-trips (`dpi=123, aspath=True` survived a `reset()` in between). Only the *Matplotlib* side of `reset()` is incomplete (finding 6).

**Individual Matplotlib options take effect and can be undone.** `sc.options(dpi=321)`, `(fontsize=19)` and `(font='serif')` each set the corresponding rcParam, and passing `None` (or `'default'`) restores the original exactly: `figure.dpi 100.0 -> 321.0 -> 100.0`, `font.size 10.0 -> 19.0 -> 10.0`, `font.family ['sans-serif'] -> ['serif'] -> ['sans-serif']`. The `backend` option round-trips as well when reset explicitly (`qtagg -> agg -> qtagg` via `sc.options(backend='agg')` then `sc.options(backend='default')`), though not through `context()` (finding 16), as does `interactive` (`qtagg -> agg -> qtagg`) — the lazy population of `orig_options['backend']` at lines 405-406 is what makes this work, and it is correct *provided* `reset()` has not corrupted the slot first (finding 2). `sc.options(style='fancy'|'simple'|'default')` applies and reverts correctly, since those two styles only touch keys present in `style_default`.

**Typo rejection.** Three of the five ways to set an option correctly reject an unknown name: `sc.options.dpii = 5` and `sc.options['dpii'] = 5` raise `KeyNotFoundError` (via the `_locked` check in `__setitem__`), and `sc.options(dpii=5)` raises `ValueError` listing the valid options. Only `update()`/`setdefault()` slip through (finding 7). The `_locked` mechanism itself is sound: `getattribute('_locked')` inside a `try` correctly treats "attribute absent" as unlocked so that `__init__` can populate the dict.

**Module-level `sc.help()`.** Aside from the `flags` crash (finding 8), it behaves: `sc.help()`, `sc.help('smooth')`, `sc.help('JSON', ignorecase=False, context=True)` and `sc.help('pickle', source=True, context=True)` all run; `func_ok()` correctly filters dunders, `sc_*` modules and the self-referential names; and `inspect.getsource()` was checked against every object in `dir(sc)` that survives that filter — no object raises anything other than the `OSError` that is already caught, so the narrow `except OSError` is adequate today (the one module that survives the filter, `sc.ansi`, has retrievable source, which merely adds noise to `source=True` searches). The `maxlnolen` computation cannot divide by zero because it only runs for keys that have at least one match.

**Miscellaneous.** `style_simple` and `style_fancy` are not mutated at import — the "Replaced with Mulish in `load_fonts()`" comment on line 56 is stale (there is no `load_fonts` anywhere in the package), so both dicts keep `font.family = 'sans-serif'` and the `sc.dcp()` on line 61 correctly prevents `style_fancy` from aliasing `style_simple`. `orig_options` is a genuine deep copy of the initial options, so `options['rc']` and `orig_options['rc']` start out as distinct dicts (until the first `reset()`; see finding 2). `to_dict()` is a shallow copy, so the returned `'rc'` entry *is* the live dict (`to_dict()['rc'] is options.rc` -> `True`) — worth knowing, but it takes deliberate mutation of a nested value to cause harm, so I am not reporting it. `changed()` correctly returns `None` (not `False`) for a key that is not an option, which is what `pop_keywords()` relies on for its non-option keys `grid` and `facecolor`. `sc.options.load()` on a file containing an unknown key raises a bare `KeyError` from line 551, but since the only documented producer of that file is `sc.options.save()`, I treated that as out of contract rather than a defect.

## Rejected on review

These were moved out of the findings on re-verification (2026-09-25). They are accurate observations but not worth fixing as bugs; the docstring items (12, 13) and the `optdesc.rc` wording (14) are worth a quick edit while someone is in the file.

- **11. `fontsize` is coerced to a string** (line 273) — NOT WORTH FIXING: the default is `'10.0'`, but Matplotlib accepts it and no Sciris code path breaks (`changed('fontsize')` compares like with like), so switching `which` to `float` is safe but cosmetic.
- **12. `context()`'s docstring contradicts its behaviour** (line 372) — NOT WORTH FIXING: plotting options set via `context()` do take effect, but this is a wording fix, not a bug.
- **13. `sc.help(output=True)` returns a string, not the documented dictionary** (line 723) — NOT WORTH FIXING: a one-word docstring correction, not a behavioural bug.
- **14. `options.rc` is never populated** (line 284) — NOT WORTH FIXING: it has no visible effect beyond finding 5, and the proposed `reset=True` fix in `with_style()` would make a temporary `with sc.options.with_style('fancy'):` permanently change `options.rc`, leaking the style out of the context; at most, correct the `optdesc.rc` text.
- **15. A failed `context()` strands `on_entry`** (line 377) — NOT WORTH FIXING: only reachable by a bare `with sc.options:` after a failed `context()`, and the stack fix for finding 1 removes it at no extra cost.

## Suggested order of work

1. Finding 1 (the `on_entry` stack) — smallest change, removes a crash and a state leak from the module's advertised temporary-settings API (and incidentally rejected item 15).
2. Finding 2 (`sc.dcp` on line 309) — one call, prevents irreversible corruption of the defaults, including the `rc` alias.
3. Finding 3 (restore-on-failure in `set()`) — makes every other option change safe to attempt; fix the class docstring example at the same time.
4. Finding 16 (treat an empty backend as default in `set()`) — one condition, makes `context(backend=...)` restore correctly.
5. Findings 5 and 4 (make `style` and the env vars actually apply) — use the corrected fix for 5, so `set()` does not re-apply the style on every call.
6. Finding 6 (rcParams snapshot for `reset()`) — the largest change, and best done after 5 so that "what style is active" is well defined.
7. Findings 7 and 8, then the documentation items 9 and 10.
