# `sc_datetime.py` bug audit

Audit of `sciris/sc_datetime.py` for genuine defects: wrong numerical results, documented arguments that don't work, silent data corruption, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps, and performance.

**Scope**: the entire file (1532 lines), covered by two parallel auditors — one over lines 31-727 (the date functions: `date()`, `readdate()`, `getdate()`, `day()`, `daydiff()`, `daterange()`, `datedelta()`, `yeartodate()`, `datetoyear()`) and one over lines 728-1532 (the timing functions: `tic()`/`toc()`/`toctic()`, `timer`, `elapsedtimestr()`, `timedsleep()`, `randsleep()`). **Method**: line-by-line reading of each function, followed by executed hypothesis tests — every "actual" value below was produced by running the code against the editable install (Sciris 3.3.0, numpy 2.4.6, commit `2d69aad`), and every finding was reproduced a second time independently before being recorded here.

**Re-verification**: this document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`); line numbers still match the current source. Of the original 26 findings, 21 were confirmed (four of them with corrected or safer fixes: 3, 4, 8, 14), one was rewritten as inaccurate (12: the defect is real but the proposed fix was wrong), and four were rejected (11, 16, 18, 22; see "Rejected on review" near the end). The review found four further bugs, added as findings 27-30. The document now lists 26 active findings: 3 high, 13 medium, 10 low.

**Nothing in this document has been applied.** All fixes are described, not made.

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | High | `sc.day()` | Reuses the first list element's implicit start date for every later element | 455 |
| 2 | High | `sc.datedelta()` | Silently discards the `days` argument whenever `years` is fractional | 623 |
| 3 | High | `sc.timer.toctic()` / `.tt()` | Silently resets the module-level tic clock, corrupting a later `sc.toc()` | 1122 |
| 4 | Medium | `sc.date()` | `to='datetime'` always raises, despite being a documented v3.1.0 feature | 384 |
| 5 | Medium | `sc.daterange()` | `interval=<int>` raises `TypeError`, contradicting the documented "if an int, number of days" | 553 |
| 6 | Medium | `sc.daterange()` | Month/year interval drifts off month-ends and never reaches the requested `end_date` | 566 |
| 7 | Medium | `sc.getdate()` | Raises `AttributeError` on a `datetime.date`, exactly what `sc.date()` returns | 116 |
| 8 | Medium | `sc.datetoyear()` | Raises `TypeError` on a `datetime.datetime` such as `sc.now()` | 722 |
| 9 | Medium | `sc.datetoyear()` | Documented `dateformat` argument does nothing and emits a spurious deprecation warning | 721 |
| 10 | Medium | `sc.timer.toc()` / `.toctic()` / `.tt()` / `.tto()` / `.tocout()` / `.stop()` | `unit=...` (and `elapsed=...`) collides with an internally-passed keyword and raises `TypeError` | 1122 |
| 12 | Medium | `sc.timer.indivtimings` / `.cumtimings` / `.plot()` | Charge idle time between a re-entered context manager's blocks (or between combined timers) to the next lap | 1206 |
| 13 | Medium | `sc.toctic()` | Swallows positional arguments into `returntic`/`returntoc` instead of passing them to `sc.toc()` | 908 |
| 14 | Medium | `sc.randsleep()` | Supplying only `low=` or only `high=` is silently ignored, giving a 0-2 s sleep | 1521 |
| 27 | Medium | `sc.daterange()` | Ignores `readformat` when the end date comes from `datedelta` kwargs (`weeks=`, `days=`, ...), so custom-format start dates crash | 549 |
| 29 | Medium | `sc.elapsedtimestr()` | Crashes on timezone-aware input, including ISO 8601 strings with `Z` or an offset and `sc.now(utc=True)` | 1382 |
| 30 | Medium | `sc.elapsedtimestr()` | Rejects `datetime.date` (what `sc.date()`/`sc.datedelta()` return), so its own docstring example crashes | 1375 |
| 15 | Low | `sc.datedelta()` | Returns a bare scalar for a single-element list, unlike every other date function | 654 |
| 17 | Low | `sc.daydiff()`, `sc.datetoyear()`, `sc.yeartodate()` | Three docstring examples give the wrong answer or raise (docs only) | 481, 705, 673 |
| 19 | Low | `sc.readdate()` | `verbose=True` produces no detail for `dateformat='dmy'`/`'mdy'` | 249 |
| 20 | Low | `sc.elapsedtimestr()` | Returns "0 days ago" for the last second before 24 hours | 1408 |
| 21 | Low | `sc.timer.toc()` (`.string` / `.message`) | Always reported in seconds, ignoring the timer's `unit` | 1095 |
| 28 | Low | `sc.datedelta()` | Silently ignores `outformat`, although `**kwargs` are documented as passed to `sc.date()` | 652 |
| 23 | Low | `sc.timedsleep()` | "Delay less than elapsed time" warning is unreachable; prints "Pausing for 1e-12 s" instead | 1488 |
| 24 | Low | `sc.toc()`, `sc.timer()` (`unit` argument) | `_convert_time_unit()` accepts the misspelling "milisecond" but rejects "millisecond" and "msec" | 769 |
| 25 | Low | `sc.timer.plot()` | Documents a `cumulative` argument that does not exist and crashes if passed (docs only) | 1263 |
| 26 | Low | `sc.timer.toctotal()` | Drops the timer's formatting kwargs (`sigfigs`, `baselabel`, etc.) | 1187 |

## Recurring patterns

**Per-element state written back into the enclosing loop variable.** Two defects share this exact shape: a value that is logically per-element is written back into the enclosing variable inside a loop over a list, so the first element silently determines the answer for every later element. `sc.day()` does it with the implicit `start_date` default (`sc_datetime.py:455-458`, high severity — the resulting numbers are simply wrong with no error). `sc.datedelta()` does it by rebinding the loop variable over the argument name `datestr`, so the post-loop `isinstance(datestr, list)` check at `sc_datetime.py:654` inspects the last element instead of the original argument. (`sc.datedelta()` also carries `as_date` across iterations at `sc_datetime.py:645`, but that only matters for mixed-type lists and was rejected on review as not worth fixing; see original finding 16.) A sweep for `for X in ...:` where `X` or a nearby default is assigned inside the body would catch these.

**Shared global/instance state between `tic`/`toc` and `timer`.** `sc.timer` methods forward `**kwargs` straight into the module-level `toc()` without reconciling them against the attributes they duplicate (`sc_datetime.py:1122`). That one line is responsible for three separate findings: `unit=`/`elapsed=` colliding into a `TypeError`, `reset=True` silently overwriting the module-level `_tictime` that `sc.tic()`/`sc.toc()` own, and (by omission on line 1095) `.string`/`.message` being cached in seconds regardless of the timer's `unit`. Separately, `indivtimings` is computed as successive differences from `_tics[0]`, which is right for a timer tic'd once (including plain non-resetting `toc()` use) but charges idle time to the next lap when `__enter__`, `start()` or `__iadd__` add further tics. (`total` and `toctotal()` also span from the first tic, but that is their documented meaning.)

**Argument forwarded to a deprecated alias.** `sc.datetoyear()` forwards `dateformat=` into `sc.date()`, where `dateformat` means `outformat` rather than `readformat` — so the argument silently does nothing and warns about a function the user never called. `sc.date()` maintains three such underscore/legacy aliases (`startdate`, `asdate`, `format`, `dateformat`); any internal call that forwards a user-facing name into that `**kwargs` block is worth re-checking.

## High severity

### 1. `sc.day()` reuses the first element's implicit start date for every later element in a list — `sc_datetime.py:455`

When no `start_date` is supplied, `day()` is documented to "return the number of days into the current year" for each date. The default is computed *inside* the per-element loop and assigned to the same `start_date` local (`sc_datetime.py:455-458`), so once the first element sets `start_date = date(f'{d.year}-01-01')`, the `if start_date:` guard is truthy on every subsequent iteration and all remaining dates are measured from the *first* element's January 1st.

```python
import sciris as sc
print(sc.day(['2021-03-01', '2022-03-01']))
print(sc.day('2021-03-01'), sc.day('2022-03-01'))
```

Actual:

```
[59, 424]
59 59
```

Expected: `[59, 59]` -- both dates are the 60th day of their own year, and each scalar call confirms `59` is the right per-year answer. The same leak affects the varargs form (`sc.day('2021-03-01','2022-03-01')` -> `[59, 424]`), array input (`sc.day(np.array(['2021-03-01','2022-03-01']))` -> `array([59, 424])`), and produces a spurious negative when the years are descending (`sc.day(['2022-03-01','2021-03-01'])` -> `[59, -306]`). Note that the second value is not even self-consistent: it is neither days-into-its-own-year nor days-from-a-user-supplied-origin, so a caller cannot correct for it without knowing the first element.

Blast radius: `sc.day()` is used internally only at `sc_plotting.py:1330-1331` (`sc.datenumformatter()`), where `start`/`end` are scalars and an explicit `start_date` is always passed, so it is unaffected. The existing test at `tests/test_datetime.py:92` (`sc.day([1000, None, sc.now()], start_date='2020-04-04')`) always passes `start_date`, so this branch is untested.

**Fix**: compute the fallback origin into a separate loop-local variable, e.g. hoist `if start_date is not None: start_date = date(start_date)` above the loop and use `this_start = start_date if start_date is not None else date(f'{d.year}-01-01')` inside it, so the default is recomputed for each element. (Also note `if start_date:` should be `if start_date is not None:` -- a falsy-but-valid origin such as `start_date=0` currently falls through to the default.)

### 2. `sc.datedelta()` silently discards the `days` argument whenever `years` is fractional — `sc_datetime.py:623`

The inner helper `years_to_days(days, years, start_year=None)` takes `days` as a parameter and then immediately overwrites it (`days = int(round(frac_year*days_per_year))`, `sc_datetime.py:623`) before writing it back with `kw['days'], kw['years'] = days, int_years` at `sc_datetime.py:626`. The user's `days` is never added, so any `days=` passed alongside a fractional `years=` is silently dropped. Integer `years` takes the non-fractional path and works correctly, which makes the discrepancy easy to miss.

```python
import sciris as sc
print(sc.datedelta('2020-06-01', days=10, years=0.25))
print(sc.datedelta('2020-06-01', days=0,  years=0.25))
print(sc.datedelta('2020-06-01', days=10, years=1))
print(sc.datedelta(days=10, years=0.25))
```

Actual:

```
2020-09-01
2020-09-01
2021-06-11
relativedelta(days=+91)
```

Expected: `2020-09-11` for the first line (91 days for the quarter year plus the requested 10 days), and `relativedelta(days=+101)` for the last. The first two lines being identical is the proof that `days=10` was thrown away; the third line shows `days=10` is honoured as soon as `years` is an integer. `weeks=` is unaffected (it is a separate `relativedelta` field), so `sc.datedelta('2020-06-01', weeks=1, years=0.25)` is correct while the `days=` equivalent is not.

Blast radius: within Sciris, `datedelta()` is called at `sc_datetime.py:549`/`563` (from `sc.daterange()`) and `sc_datetime.py:684` (from `sc.yeartodate()`); none of those pass a fractional `years` together with `days`, so the damage is confined to direct user calls. Fractional years are a documented v3.2.0 feature and are not covered in `tests/test_datetime.py`.

**Fix**: accumulate rather than overwrite -- `kw['days'] = days + int(round(frac_year*days_per_year))` -- and drop the shadowing local. The same helper should also be made to read its `days` argument rather than the closure, so the "function arguments remain the ground truth" comment at `sc_datetime.py:625` actually holds.

### 3. `sc.timer().toctic()` / `.tt()` silently resets the module-level tic clock, corrupting a later `sc.toc()` — `sc_datetime.py:1122`

`timer.toc()` forwards its whole `kwargs` dict to the module-level `toc()` (`output = toc(elapsed=self.elapsed, unit=self.unit, verbose=verbose, **kwargs)`), and `timer.toctic()`/`tt()`/`tto()` put `reset=True` into that dict. Module-level `toc(reset=True)` does `global _tictime; _tictime = pytime.time()`, so every timer lap silently overwrites the state owned by `sc.tic()`/`sc.toc()`. The intended per-object reset is a separate statement three lines further down (`if kwargs.get('reset'): self.tic()`, line 1125), so the global write is pure collateral damage.

```python
import sciris as sc
sc.tic()                    # start the global timer
T = sc.timer(verbose=False) # an unrelated timer
sc.timedsleep(0.05); T.tt() # timer lap 1
sc.timedsleep(0.05); T.tt() # timer lap 2
sc.timedsleep(0.05)
sc.toc()                    # should be ~0.15 s
```

Actual: `Elapsed time: 0.0503 s`. Expected: `Elapsed time: 0.151 s`. The reported total is only the time since the *last* timer lap, and it silently shrinks as more laps are added -- the exact "top-level `sc.tic()` ... `sc.toc()` around a loop that uses `sc.timer`" pattern that both objects are documented for. Nothing inside Sciris interleaves the two, so the blast radius is user code (`sc.toctic()` is used in `tests/test_datetime.py:146`, but not interleaved with a `timer`). Note the plain `T.toc()` path does *not* corrupt the global (verified: reports 0.151 s correctly), which makes the failure look intermittent and dependent on which timer method the user happened to pick.

**Fix**: in `timer.toc()`, pop `reset` out of `kwargs` into a local before forwarding, so the module-level `toc()` never sees it, and use the local afterwards: `reset = kwargs.pop('reset', False)` before line 1122, then `if reset: self.tic()` in place of `if kwargs.get('reset'): self.tic()` at line 1125. (Corrected on review: the original fix said to simply pop `reset`, claiming it was "already consumed locally", but the `kwargs.get('reset')` check runs *after* the forwarding call, so a bare pop would make it always falsy and silently stop `toctic()`/`tt()` from re-ticking the timer.)

## Medium severity

### 4. `sc.date(..., to='datetime')` always raises, even though "datetime" output is a documented v3.1.0 feature — `sc_datetime.py:384`

The docstring advertises `to='datetime'` ("- *New in version 3.1.0:* allow 'datetime' output") and the input-conversion block handles `to == 'datetime'` in two places (`sc_datetime.py:359` and `sc_datetime.py:377`). But the output dispatch chain at `sc_datetime.py:384-394` has branches only for `'date'`, `str`/`'str'`/`'string'`, `'pandas'` and `'numpy'`, so `'datetime'` falls through to the `else` and raises -- for every input type.

```python
import datetime as dt, sciris as sc
try:
    sc.date('2020-04-05', to='datetime')
except Exception as E:
    print(type(E).__name__, E, '| cause:', repr(E.__cause__))
```

Actual:

```
ValueError Conversion of "2020-04-05 00:00:00" to a date failed | cause: ValueError('Could not understand to="datetime": must be "date", "str", "pandas", or "numpy"')
```

Expected: `datetime.datetime(2020, 4, 5, 0, 0)`. The failure is unconditional across the input matrix -- `str`, `dt.date`, `dt.datetime`, `np.datetime64`, `pd.Timestamp`, and `int` + `start_date` all raise identically. Note the wrapper also hides the real cause behind a misleading "Conversion of ... failed" message, and the nested error message itself omits `'datetime'` from its list of valid values.

Blast radius: no internal Sciris caller uses `to='datetime'`, and `grep -rn "to='datetime'" tests/ sciris/` returns nothing, so the feature has never been exercised.

**Fix**: add a `'datetime'` branch to the dispatch chain at `sc_datetime.py:384`, and add `"datetime"` to the error message at `sc_datetime.py:393`. A bare `elif to == 'datetime': out = d` returns a `pd.Timestamp` rather than a plain `datetime` when the input was a `pd.Timestamp` or an `np.datetime64` (the latter is converted to `pd.Timestamp` at line 368); that is tolerable since `pd.Timestamp` subclasses `datetime`, but for a plain `datetime` every time use `out = d.to_pydatetime() if isinstance(d, pd.Timestamp) else d` (and promote a bare `dt.date` to `dt.datetime` if one slips through).

### 5. `sc.daterange(interval=<int>)` raises `TypeError`, contradicting its own documented "if an int, the number of days" — `sc_datetime.py:553`

`interval` is documented as `(int/str/dict)` with "if an int, the number of days". The normalisation block at `sc_datetime.py:553-556` only maps `None`/`'day'`/`'week'`/`'month'`/`'year'` to dicts; an integer passes through unchanged and is then splatted at `sc_datetime.py:563` as `datedelta(**interval)`.

```python
import sciris as sc
try:
    sc.daterange('2020-01-01', '2020-01-10', interval=3)
except Exception as E:
    print(type(E).__name__, E)
```

Actual:

```
TypeError sciris.sc_datetime.datedelta() argument after ** must be a mapping, not int
```

Expected: `['2020-01-01', '2020-01-04', '2020-01-07', '2020-01-10']`, which is what the documented-equivalent `interval=dict(days=3)` returns. The string and dict forms both work; only the documented int form is broken. `tests/test_datetime.py:112` loops over the four string intervals only.

**Fix**: add `if sc.isnumber(interval): interval = dict(days=interval)` before the string dispatch at `sc_datetime.py:553`.

### 6. `sc.daterange()` with a month or year interval drifts off month-ends and never reaches the requested `end_date` — `sc_datetime.py:566`

The loop accumulates `curr_date += delta` (`sc_datetime.py:566`) instead of anchoring each step to `start_date`. `relativedelta` month/year arithmetic clamps to the end of the shorter target month and is therefore not additive, so once a step lands on a clamped day the sequence is permanently shifted and the `inclusive=True` end date is skipped.

```python
import sciris as sc
print(sc.daterange('2020-01-31', '2020-06-30', interval='month'))
print(sc.daterange('2020-02-29', '2025-03-01', interval='year'))
```

Actual:

```
['2020-01-31', '2020-02-29', '2020-03-29', '2020-04-29', '2020-05-29', '2020-06-29']
['2020-02-29', '2021-02-28', '2022-02-28', '2023-02-28', '2024-02-28', '2025-02-28']
```

Expected (the anchored equivalent, `start + relativedelta(months=1)*i`, which is how `relativedelta` is meant to be scaled):

```
['2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30', '2020-05-31', '2020-06-30']
['2020-02-29', '2021-02-28', '2022-02-28', '2023-02-28', '2024-02-29', '2025-02-28']
```

So the March entry is the 29th rather than the 31st, the requested inclusive end `2020-06-30` is silently replaced by `2020-06-29`, and the 2024 leap day is lost even though the anchor is itself a Feb 29. Same effect with an explicit dict: `sc.daterange('2020-01-31','2021-01-31', interval=dict(months=2))` ends `'2021-01-30'` and `'2021-01-31' in result` is `False`, despite `inclusive=True` being the default. Ordinary `interval='day'`/`'week'` and integer-day dict intervals are unaffected because day arithmetic is additive.

**Fix**: index off the anchor instead of accumulating -- `i = 0; while True: curr = start_date + delta*i; if curr >= end_date: break; dates.append(curr); i += 1` -- which `du.relativedelta.relativedelta` supports via `__mul__`.

### 7. `sc.getdate()` raises `AttributeError` on a `datetime.date`, which is exactly what `sc.date()` returns — `sc_datetime.py:116`

`getdate()` validates its input with `obj.timetuple()` (`sc_datetime.py:110`), which succeeds for both `dt.date` and `dt.datetime`, but then unconditionally computes `timestamp = obj.timestamp()` at `sc_datetime.py:116` before the `astype` dispatch. `dt.date` has no `.timestamp()`, so every `astype` fails -- including `'str'` and `'dateobj'`, which never use the timestamp.

```python
import sciris as sc
try:
    sc.getdate(sc.date('2020-01-01'))  # sc.date() returns a datetime.date by default
except Exception as E:
    print(type(E).__name__, E)
```

Actual:

```
AttributeError 'datetime.date' object has no attribute 'timestamp'
```

Expected: `'2020-Jan-01 00:00:00'` (the equivalent `sc.getdate(dt.datetime(2020,1,1))` returns exactly that). All four `astype` values fail identically (`'str'`, `'int'`, `'dateobj'`, `'float'`). This is squarely in-contract: `sc.date()` -- the library's own recommended converter -- returns `dt.date`, so the natural pipeline `sc.getdate(sc.date(x))` cannot work.

Blast radius: the only internal caller, `sc_versioning.py:456`, calls `sc.getdate()` with no argument (so it goes through `now()` and gets a `datetime`), and `tests/test_datetime.py:54` always passes `None` or `sc.now(utc=True)`, so the `dt.date` path is untested.

**Fix**: move `timestamp = obj.timestamp()` inside the `'int'`/`'float'` branches, or promote a bare `dt.date` to `dt.datetime` first (`if not isinstance(obj, dt.datetime): obj = dt.datetime(obj.year, obj.month, obj.day)`).

### 8. `sc.datetoyear()` raises `TypeError` on a `datetime.datetime` such as `sc.now()` — `sc_datetime.py:722`

The input is normalised only when it is a string or a `pd.Timestamp` (`sc_datetime.py:719-721`); a plain `dt.datetime` is left alone and then hits `dateobj - dt.date(...)` at `sc_datetime.py:722`, which Python forbids for `datetime - date`.

```python
import datetime as dt, sciris as sc
try:
    sc.datetoyear(dt.datetime(2010, 7, 1))
except Exception as E:
    print(type(E).__name__, E)
```

Actual:

```
TypeError unsupported operand type(s) for -: 'datetime.datetime' and 'datetime.date'
```

Expected: `2010.4958904109589`, the value the string and `pd.Timestamp` forms both return. `sc.datetoyear(sc.now())` fails the same way, as does `sc.datetoyear(np.datetime64('2010-07-01'))` (`AttributeError: 'numpy.datetime64' object has no attribute 'year'`). Since `pd.Timestamp` (a `dt.datetime` subclass) *is* handled, and since `sc.now()` is the library's own way of getting "now", the naive-`datetime` gap looks like an oversight rather than a deliberate contract.

**Fix**: for `dt.datetime` input (including `sc.now()`), compute the fraction directly so the time of day is kept: `(dateobj - dt.datetime(dateobj.year, 1, 1, tzinfo=dateobj.tzinfo)) / year_length`. Only strings and `np.datetime64` need to go through `date(dateobj, readformat=dateformat)` (which also fixes finding 9). (Corrected on review: the original fix, an unconditional `dateobj = date(dateobj, readformat=dateformat)`, converts every `datetime` to a `date` and discards the time of day, so `datetoyear(dt.datetime(2010,7,1,18))` would equal midnight. The existing `pd.Timestamp` path already loses the time this way, so that fix would be consistent, but a decimal-year function should keep it.)

### 9. `sc.datetoyear()`'s documented `dateformat` argument does nothing and emits a spurious deprecation warning — `sc_datetime.py:721`

`dateformat` is documented as "If dateobj is a string, the optional date conversion format to use", but it is forwarded as `date(dateobj, dateformat=dateformat)` at `sc_datetime.py:721`. In `sc.date()`, `dateformat` is a *deprecated alias for `outformat`* (`sc_datetime.py:308-310`), not for `readformat` -- so the value never reaches `readdate()`, the read still uses the default format list, and the user gets a `FutureWarning` about an argument they did not pass.

```python
import warnings, sciris as sc
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    try:
        print(sc.datetoyear('01/07/2010', dateformat='dmy'))
    except Exception as E:
        print(type(E).__name__, E)
    print([str(x.message)[:60] for x in w])
```

Actual:

```
ValueError Conversion of "01/07/2010" to a date failed
['sc.date() argument "dateformat" has been deprecated as ']
```

Expected: `2010.4958904109589`. Any date that actually *needs* a custom read format is unparseable via `sc.datetoyear()`, and any date that parses without one warns pointlessly. Note the warning names `sc.date()`, so the message misdirects the user to a function they never called.

**Fix**: pass `readformat=dateformat` instead of `dateformat=dateformat` at `sc_datetime.py:721`.

### 10. `timer.toc(unit=...)` and every alias that accepts kwargs raise `TypeError` on the documented `unit` argument — `sc_datetime.py:1122`

`timer.toc()`'s docstring says "see `sc.toc()` for keyword arguments", and `unit` is `sc.toc()`'s headline keyword, but line 1122 already passes `unit=self.unit` positionally-by-name alongside `**kwargs`, so a user-supplied `unit` collides.

```python
import sciris as sc
T = sc.timer()
sc.timedsleep(0.05)
T.toc(unit='ms')
```

Actual:

```
  File "/home/cliffk/sc/sciris/sciris/sc_datetime.py", line 1122, in toc
    output = toc(elapsed=self.elapsed, unit=self.unit, verbose=verbose, **kwargs)
TypeError: sciris.sc_datetime.toc() got multiple values for keyword argument 'unit'
```

Expected: `Elapsed time: 50.1 ms`. The same crash occurs for `T.toctic(unit='min')`, `T.tt('lbl', unit='ms')`, and `T.tocout(unit='ms')`. Passing `unit` to the *constructor* works, so this is purely the per-call override. `elapsed` collides identically. No internal Sciris caller passes `unit` to `timer.toc()`, so this is user-facing only.

**Fix**: `unit = kwargs.pop('unit', self.unit)` (and likewise for `elapsed`) before the call, mirroring how `verbose` is already handled on line 1121.

### 12. `indivtimings` charges idle time between a re-entered context manager's blocks (or between combined timers) to the next lap — `sc_datetime.py:1206`

*Rewritten on review: the defect is real, but the original description overstated it and the original fix would have broken the common non-resetting case.* `indivtimings` is `np.diff(sc.cat(self._tics[0], self._tocs))`, i.e. successive differences of the toc times starting from the first tic. That is deliberate and correct for the ordinary cases: it turns the *cumulative* timings from plain non-resetting `T.toc()` calls into per-lap times (verified: three plain `T.toc()` calls 0.03 s apart give `timings` `[0.030, 0.060, 0.091]`, `indivtimings` `[0.030, 0.030, 0.030]`, with `len(_tics) == 1`), and it is also right for `toctic()` laps. It goes wrong only when there is idle time between a toc and the next tic: `__enter__` and `tic()`/`start()` append to `_tics`, and `__iadd__` concatenates two timers' `_tics`/`_tocs` (without sorting), so a re-entered context manager or a combined timer charges any wall-clock time spent *outside* a timed interval to the next lap.

```python
import sciris as sc
T = sc.timer('lap', verbose=False)
with T: sc.timedsleep(0.05)
sc.timedsleep(0.10)          # dead time outside the block
with T: sc.timedsleep(0.05)
print('timings     :', [round(v,3) for v in T.timings.values()])
print('indivtimings:', [round(v,3) for v in T.indivtimings.values()])
```

Actual:

```
timings     : [0.05, 0.05]
indivtimings: [0.05, 0.151]
```

Expected `indivtimings` to read `[0.05, 0.05]`, matching `timings`. The combined-timer case is the same defect: two 0.05 s timers created 0.2 s apart give `timings {'t1': 0.050, 't2': 0.050}` but `indivtimings {'t1': 0.050, 't2': 0.251}`. Blast radius inside the module: `timer.plot()` (line 1281) bars `self.indivtimings`, so plotting a re-entered or combined timer shows silently wrong bars for every lap after the first -- and `sc.timer` addition/`sum()` exists precisely to aggregate separately-run timers (*New in version 3.1.0*). `total` and `toctotal()` (and hence `cumtimings[-1]`) span from the first tic to the last toc, so for these timers `total` (0.301 s) exceeds `sum()` (0.100 s); that is their documented meaning ("from the first tic"), so it is not treated as part of this defect.

**Fix**: for each toc `i`, compute the lap as `toc[i] - max(latest tic <= toc[i], toc[i-1])` (with `toc[-1]` treated as minus infinity). This gives the right answer for plain non-resetting `toc()` use, for `toctic()` laps, and for a re-entered context manager. Combined timers additionally need per-timer handling (or `__iadd__` must keep each timer's tics and tocs paired), because the concatenated lists are not in time order. (Corrected on review: the original fix -- derive `indivtimings` from `self.timings`, or use `np.array(self._tocs) - np.array(self._tics[:len(self._tocs)])` -- breaks the non-resetting case: the first returns the cumulative values, and the second fails with a shape mismatch because there is only one tic for three tocs. The original suggestion to make `total`/`cumtimings` sum the intervals is also dropped, since spanning from the first tic is documented behaviour.)

### 13. `sc.toctic()` swallows positional arguments into `returntic`/`returntoc` instead of passing them to `sc.toc()` — `sc_datetime.py:908`

The signature is `def toctic(returntic=False, returntoc=False, *args, **kwargs)`, but the docstring says "Arguments are passed to `sc.toc()`" and the sibling APIs put `start` and `label` first (`sc.toc(start, label)`, `timer.toctic(label)`). Any positional call therefore binds the caller's start time or label to `returntic`, where it is silently discarded *and* flips the return value.

```python
import sciris as sc
T = sc.tic()
sc.timedsleep(0.05)
sc.tic()                # something else calls tic()
print('toctic(T) ->', end=' '); sc.toctic(T)   # T silently ignored
print('toc(T)    ->', end=' '); sc.toc(T)      # correct
```

Actual:

```
toctic(T) -> Elapsed time: 0.0000184 s
toc(T)    -> Elapsed time: 0.0513 s
```

`sc.toctic('mylabel')` likewise prints the default `Elapsed time: ...` with the label dropped and returns the new tic timestamp instead of `None`. Keyword form (`sc.toctic(label='x')`) works. `timer.toctic()` is a different method and is unaffected.

**Fix**: make the flags keyword-only -- `def toctic(*args, returntic=False, returntoc=False, **kwargs)` -- so positional arguments reach `toc()`.

### 14. `sc.randsleep(low=...)` or `randsleep(high=...)` alone is silently ignored, giving a 0-2 s sleep — `sc_datetime.py:1521`

The guard is `if low is None or high is None:` and its body overwrites *both* bounds from `delay`/`var`. So supplying only one of the two documented bounds ("low (float): optionally define lower bound of sleep", "high (float): optionally define upper bound of sleep") discards it and falls back to `low = delay*(1-var) = 0`, `high = delay*(1+var) = 2`.

```python
import numpy as np, sciris as sc
d = np.array([sc.randsleep(low=0.001, high=0.002) for i in range(20)])
print('low+high: min %.4f max %.4f' % (d.min(), d.max()))
d = np.array([sc.randsleep(low=0.001) for i in range(5)])
print('low only: min %.4f max %.4f  (documented: >= 0.001)' % (d.min(), d.max()))
```

Actual:

```
low+high: min 0.0011 max 0.0020
low only: min 0.2319 max 1.6698  (documented: >= 0.001)
```

Expected the second line to stay at or above 0.001 s. The failure mode is a ~1000x longer pause than asked for, which in a loop looks like a hang rather than a wrong number. Sciris's own callers (`sc_parallel.py:1082`, `sc_parallel.py:1085`, `sc_parallel.py:127`) all use the positional `delay` form, so the blast radius is user code; `tests/test_datetime.py:152` only tests `sc.randsleep(0.1)` and the `[low, high]` list form.

**Fix**: compute the delay-derived bounds once, `dlow, dhigh = delay*(1-var), delay*(1+var)`, then fill in only the missing bound: `low = dlow if low is None else low`, `high = dhigh if high is None else high`. (Refined on review: the original suggestion `high = 2*delay if high is None else high` ignored `var`.)

### 27. `sc.daterange()` ignores `readformat` when the end date comes from `datedelta` kwargs — `sc_datetime.py:549`

*Added on review.* `readformat` is used to read `start_date` (`sc_datetime.py:550`), but when `weeks=`, `days=` etc. are given instead of an explicit `end_date`, line 549 first calls `datedelta(start_date, **kwargs)` without `readformat`. Any start date that needs a custom read format therefore crashes, even though the same call with an explicit end date works.

```python
import sciris as sc
print(sc.daterange('01-03-2020', '03-03-2020', readformat='dmy'))  # works
print(sc.daterange('01-03-2020', weeks=1, readformat='dmy'))       # crashes
```

Actual:

```
['2020-03-01', '2020-03-02', '2020-03-03']
ValueError: Conversion of "01-03-2020" to a date failed
```

Expected for the second call: the eight dates from `'2020-03-01'` to `'2020-03-08'`.

**Fix**: `end_date = datedelta(start_date, readformat=readformat, **kwargs)` at `sc_datetime.py:549`. Alternatively, convert `start_date` with `date(start_date, readformat=readformat)` before computing `end_date`.

### 29. `sc.elapsedtimestr()` crashes on timezone-aware input, including ISO 8601 strings with `Z` or an offset — `sc_datetime.py:1382`

*Added on review.* `now_time = dt.datetime.now()` is naive. `readdate()` explicitly supports ISO strings with `Z` or `+hh:mm` (the `iso-tz` formats, new in v3.3.0) and returns an aware datetime, so the comparison `pasttime > now_time` at `sc_datetime.py:1382` raises. The docstring promises "a datetime object or a string in ISO 8601 format", and Sciris's own `sc.now(utc=True)` fails too.

```python
import sciris as sc, datetime as dt
sc.elapsedtimestr('2023-10-12T00:05:12Z')
sc.elapsedtimestr(sc.now(utc=True) - dt.timedelta(hours=3))
```

Actual: `TypeError: can't compare offset-naive and offset-aware datetimes`, for both calls. Expected: `'12 Oct 2023'` and `'3 hours ago'`.

**Fix**: `now_time = dt.datetime.now(pasttime.tzinfo) if pasttime.tzinfo is not None else dt.datetime.now()`. This has to happen after `pasttime` has been parsed, so move the `now_time` assignment below the input-handling block.

### 30. `sc.elapsedtimestr()` rejects `datetime.date`, so its own docstring example crashes — `sc_datetime.py:1375`

*Added on review.* The only accepted object type is `dt.datetime` (`sc_datetime.py:1375`); anything else hits the `TypeError` branch. But `sc.date()`, `sc.datedelta()` and `sc.daterange()` all return `dt.date` by default, and the docstring example `yesterday = sc.datedelta(sc.now(), days=-1); sc.elapsedtimestr(yesterday)` fails because `datedelta` on a datetime returns a `dt.date`.

```python
import sciris as sc
sc.elapsedtimestr(sc.datedelta(sc.now(), days=-1))
```

Actual: `TypeError: User-supplied value 2026-09-24 is neither a datetime object nor an ISO 8601 string.` Expected: `'yesterday'`, or similar (e.g. `'N hours ago'`, since a date means midnight).

**Fix**: add `elif isinstance(pasttime, dt.date): pasttime = dt.datetime(pasttime.year, pasttime.month, pasttime.day)` after the `dt.datetime` branch (the order matters, since `dt.datetime` subclasses `dt.date`). Alternatively, fix the docstring example to use `sc.now() - dt.timedelta(days=1)`.

## Low severity

### 15. `sc.datedelta()` returns a bare scalar for a single-element list, unlike every other date function — `sc_datetime.py:654`

The guard `if not isinstance(datestr, list) and len(newdates) == 1` at `sc_datetime.py:654` tests `datestr`, but `datestr` has been rebound by the `for datestr in datelist:` loop at `sc_datetime.py:644` and now holds the *last element*, not the original argument. For a one-element list the element is a string/date, so the list wrapper is stripped.

```python
import sciris as sc
print(repr(sc.datedelta(['2021-07-07'], days=1)))
print(repr(sc.date(['2021-07-07'])))
```

Actual:

```
'2021-07-08'
[datetime.date(2021, 7, 7)]
```

Expected `['2021-07-08']`, to match `sc.date()`, `sc.day()` and `sc.readdate()`, all of which preserve list-ness for a single-element list via `scu._sanitize_output()`. Code that does `for d in sc.datedelta(dates, months=1)` iterates over the characters of a string when `dates` happens to have length 1.

**Fix**: capture the original argument before the loop (e.g. `was_list = isinstance(datestr, list)`) and test that, or use `scu._sanitize_iterables`/`_sanitize_output` as the sibling functions do.

### 17. Three docstring examples give the wrong answer or raise — `sc_datetime.py:481`, `sc_datetime.py:705`, `sc_datetime.py:673`

```python
import sciris as sc
print(sc.daydiff('2022-03-20'))          # sc_datetime.py:481 claims 79
try: print(sc.datetoyear(2010.5))        # sc_datetime.py:705 claims datetime.date(2010, 7, 2)
except Exception as E: print(type(E).__name__, E)
try: print(sc.yeartodate('2010-07-01'))  # sc_datetime.py:673 claims approximately 2010.5
except Exception as E: print(type(E).__name__, E)
```

Actual:

```
78
AttributeError 'float' object has no attribute 'year'
ValueError invalid literal for int() with base 10: '2010-07-01'
```

`sc.daydiff('2022-03-20')` is 78, not 79 (2022-01-01 is day 0, and 31 + 28 + 19 = 78) -- the docstring is simply off by one. The `sc.datetoyear(2010.5)` example is a leftover from the removed `reverse=` argument (see the v3.2.1 changelog note two lines below it) and now raises. The `sc.yeartodate()` example is a copy-paste of the `sc.datetoyear()` example -- it passes a date string to a function that expects a decimal year, and raises; the correct example is `sc.yeartodate(2010.5)`, which returns `datetime.date(2010, 7, 2)`.

**Fix**: change `# Returns 79` to `# Returns 78` at `sc_datetime.py:481`; delete the stale reverse example at `sc_datetime.py:705`; replace the `sc.yeartodate()` example at `sc_datetime.py:673` with `sc.yeartodate(2010.5) # Returns datetime.date(2010, 7, 2)`.

### 19. `sc.readdate(verbose=True)` produces no detail for `dateformat='dmy'`/`'mdy'` because the verbose block is nested inside the wrong `if` — `sc_datetime.py:249`

`verbose` is documented as "return detailed error messages", but the loop that appends the per-format exceptions (`sc_datetime.py:249-251`) is nested inside `if dateformat not in ['dmy', 'mdy']:` (`sc_datetime.py:247`), whose actual purpose is to add the "use dateformat='dmy'" hint.

```python
import sciris as sc
try:
    sc.readdate('not a date', dateformat='dmy', verbose=True)
except Exception as E:
    print(len(str(E)), 'chars, has per-format detail:', 'date:' in str(E))
```

Actual:

```
99 chars, has per-format detail: False
```

With `dateformat=None` the same call yields a 1994-character message that does include the detail, so `verbose=True` is a no-op precisely in the ambiguous-format case where it is most useful.

**Fix**: dedent the `if verbose:` block at `sc_datetime.py:249` one level so it is a sibling of the `dateformat not in ['dmy','mdy']` check rather than its child.

### 20. `sc.elapsedtimestr()` returns "0 days ago" for the last second before 24 hours — `sc_datetime.py:1408`

The hours branch is guarded by `elif elapsed_time < dt.timedelta(seconds=60 * 60 * 24 - 1)`. The stray `- 1` opens a one-second gap: an elapsed time in `[86399, 86400)` s skips the hours branch, reaches the days branch, and formats `elapsed_time.days`, which is still `0`.

```python
import datetime as dt, sciris as sc
now = dt.datetime.now()
for s in [86398, 86399, 86400]:
    print(s, 's ago ->', repr(sc.elapsedtimestr(now - dt.timedelta(seconds=s))))
```

Actual:

```
86398 s ago -> '23 hours ago'
86399 s ago -> '0 days ago'
86400 s ago -> 'yesterday'
```

Expected `'23 hours ago'` for 86399 s. Narrow window (1 s in 86400), but the output is nonsense rather than merely imprecise, and the same `days == 0` formatting would be produced by any future widening of that branch.

**Fix**: drop the `- 1` (use `dt.timedelta(days=1)`); alternatively make the days branch fall back to hours when `elapsed_time.days == 0`.

### 21. `timer.string` and `timer.message` are always in seconds, ignoring the timer's `unit` — `sc_datetime.py:1095`

Line 1095 calls `toc(start=self._start, output='all', verbose=False)` without `unit`, so the cached display strings use `sc.toc()`'s default `unit='s'`; the correctly-formatted string produced by the second call (line 1122, which does pass `unit=self.unit`) is only printed, never stored.

```python
import sciris as sc
T = sc.timer(unit='ms')
sc.timedsleep(0.05)
T.toc()
print('T.string  =', repr(T.string))
print('T.message =', repr(T.message))
```

Actual:

```
Elapsed time: 50.1 ms
T.string  = '0.0501 s'
T.message = 'Elapsed time: 0.0501 s'
```

The printed line and the stored strings disagree, so `.string`/`.message` cannot be used to reproduce what the timer showed the user. This also affects the default `unit='auto'` (a 0.05 s lap prints `50.3 ms` but stores `'0.0503 s'`). `.string` is documented as a feature (*New in version 3.2.5*).

**Fix**: pass `unit=self.unit` (or `unit=unit` after resolving per-call overrides) to the first `toc()` call, or assign `self.string`/`self.message` from the second call's `output='all'` result.

### 23. `sc.timedsleep()`'s "delay less than elapsed time" warning is unreachable; it prints "Pausing for 1e-12 s" instead — `sc_datetime.py:1488`

`remaining = max(1e-12, delay - elapsed - _sleep_overhead)` clamps to a positive number, so the very next test `if remaining > 0 and verbose` is always true and the `elif verbose` branch (line 1491) can never run. The diagnostic that was meant to tell the user their loop body overran the requested period is dead, and the message that does print is misleading.

```python
import time, sciris as sc
sc.timedsleep('start')
time.sleep(0.05)
sc.timedsleep(0.01, verbose=True)  # requested delay already blown by 5x
```

Actual: `Pausing for 1e-12 s`. Expected: `Warning, delay less than elapsed time (0.01 vs. 0.05)`. The `# pragma: no cover` on line 1491 is correctly placed in the sense that the line is genuinely unreachable -- it is the code that is wrong, not the pragma.

**Fix**: compute `remaining = delay - elapsed - _sleep_overhead`, branch on its sign for the message, and clamp only at the `pytime.sleep(max(0, remaining))` call.

### 24. `_convert_time_unit()` accepts the misspelling "milisecond" but rejects "millisecond" and "msec" — `sc_datetime.py:769`

The alias list is `'ms' : dict(factor=1e-3, aliases=['ms', 'milisecond', 'miliseconds'])` -- single `l`. Every other unit spells out its long form correctly (`second`/`seconds`, `microsecond`, `nanosecond`, `minute`, `hour`), so the correctly-spelled words are the ones that fail.

```python
import sciris as sc
for u in ['milisecond', 'millisecond', 'msec']:
    try:    print(u, '->', sc.toc(elapsed=1.5, unit=u, output='message'))
    except Exception as E: print(u, '->', type(E).__name__, str(E)[:40])
```

Actual:

```
milisecond -> Elapsed time: 1500 ms
millisecond -> ValueError Could not understand "millisecond"; all
msec -> ValueError Could not understand "msec"; all
```

**Fix**: add `'millisecond', 'milliseconds', 'msec', 'msecs'` to the `'ms'` alias list (keeping the misspellings for backwards compatibility). While there: `'sec'`/`'secs'` are accepted for seconds but `'usec'`/`'nsec'` are not, and `'day'` is absent entirely.

### 25. `timer.plot()` documents a `cumulative` argument that does not exist and crashes if passed — `sc_datetime.py:1263`

The docstring's argument list starts with `cumulative (bool): how the timings will be presented, individual or cumulative`, but the signature is `plot(self, fig=None, figkwargs=None, grid=True, **kwargs)` and the method unconditionally draws both an individual and a cumulative subplot. A caller following the docstring has the argument forwarded into `plt.barh()`.

```python
import matplotlib; matplotlib.use('agg')
import sciris as sc
T = sc.timer(verbose=False)
for i in range(3):
    sc.timedsleep(0.02); T.toctic()
T.plot(cumulative=True)
```

Actual: `AttributeError: Rectangle.set() got an unexpected keyword argument 'cumulative'`. Expected: either a respected argument or no mention of it in the docstring.

**Fix**: delete the `cumulative` line from the docstring (the method always shows both), or implement it.

### 26. `timer.toctotal()` drops the timer's formatting kwargs — `sc_datetime.py:1187`

`toctotal()` explicitly re-applies only `unit` and `verbose` (`kwargs.setdefault(...)`) and never consults `self.kwargs`, so formatting options given to the constructor and honoured by every other method are ignored for the total line.

```python
import sciris as sc
T = sc.timer(label='mylabel', baselabel='BASE: ', sigfigs=6)
sc.timedsleep(0.05); T.toc(); T.toctotal()
```

Actual:

```
BASE: mylabel: 50.1676 ms
Total: 50.4 ms
```

The `sigfigs=6` and `baselabel='BASE: '` that shaped the first line are silently absent from the second. (Overriding `label` with `'Total'` is intentional and correct.)

**Fix**: merge `self.kwargs` into `kwargs` (excluding `label`) the same way `timer.toc()` does on lines 1101-1103.

### 28. `sc.datedelta()` silently ignores `outformat` — `sc_datetime.py:652`

*Added on review.* `kwargs` is documented as "passed to `sc.date()`", but it only reaches the *reading* call (`date(datestr, **kwargs)`, `sc_datetime.py:647`). That call returns a date object, so `outformat` has no effect there. The output call `date(newdate, as_date=as_date)` at `sc_datetime.py:652` does not get `outformat`, so string output always uses `'%Y-%m-%d'`.

```python
import sciris as sc
print(sc.datedelta('2021-07-07', days=1, outformat='%d/%m/%Y'))
```

Actual: `'2021-07-08'`. Expected: `'08/07/2021'`. A related effect: a string read with a custom `readformat` comes back in ISO format rather than the input format (`sc.datedelta('07/08/2021', days=1, readformat='dmy')` returns `'2021-08-08'`). This is arguably acceptable, but it shows that "return as input type" preserves the type and not the format.

**Fix**: split the kwargs. Pass `outformat` (if present) to the second `date()` call at `sc_datetime.py:652`, and pass the rest to the first.

## Misplaced `# pragma: no cover`

Reachability in lines 31-727 was demonstrated with `sys.settrace()`, checking that the *body* line inside each pragma'd block executed (not just the condition line). Reachability in lines 728-1532 was confirmed by running `coverage run --include='*sc_datetime.py'` and reading `excluded_lines` from `coverage json`, then demonstrating each as reachable by executing the snippet shown.

| Line(s) | Pragma'd construct | Demonstrated reachable by |
|---|---|---|
| 108 | `if sc.isstring(obj):` in `sc.getdate()` | `sc.getdate('already a string')`; body line 109 traced (marginal -- not a documented use, listed for completeness) |
| 120 | `elif astype in ['float','number','timestamp']:` in `sc.getdate()` | `sc.getdate(astype='float')` -- a verbatim docstring example (`sc_datetime.py:96`); body line 121 traced |
| 229 | `elif sc.isnumber(datestr):` in `sc.readdate()` | `sc.readdate(1611661666)` -- a verbatim docstring example, and already exercised by `tests/test_datetime.py:47` (`sc.readdate(sc.tic())`); body lines 230-231 traced |
| 330 | `if as_date is not None:` in `sc.date()` | any `sc.date(..., as_date=...)` (a documented argument, used in `tests/test_datetime.py:73`) **and** every single `sc.daterange()` call, which passes `as_date=` unconditionally at `sc_datetime.py:569`; body line 331 traced |
| 354 | `if d is None:` in `sc.date()` | `sc.date([None, '2020-01-01'])` -- None input is the documented v3.0.0 "allow None" feature; body lines 355-356 traced |
| 858-859 | `if isinstance(start, str): # pragma: no cover` (start/label swap) in `sc.toc()` | `sc.tic(); sc.toc('swapped-label')` -> `swapped-label: 0.0202 s` |
| 872-873 | `else: base = baselabel # pragma: no cover` in `sc.toc()` | `sc.tic(); sc.toc(baselabel='BASE-ONLY: ')` -> `BASE-ONLY: 0.0202 s` |
| 878-879 | `else: base = '' # pragma: no cover` (label='') in `sc.toc()` | `sc.tic(); sc.toc(label='')` -> `0.0203 s` |
| 895-903 | `if output: # pragma: no cover` in `sc.toc()` | Exercised by *every* `sc.timer()` lap -- `timer.toc()` line 1095 calls `toc(..., output='all')`; also `sc.toc(output=True)` returns `4.29e-06` |
| 1135-1136 | `if not len(self._tics): # pragma: no cover` in `timer.total` | `sc.timer(start=False).total` -> `0` |
| 1141-1142 | `if not len(self._tocs): # pragma: no cover` in `timer.total` | `repr(sc.timer())` -> `Total time: 1.88351e-05 s` (the `__repr__` of any un-`toc`ed timer) |
| 1300-1302 | `else:` (nothing timed) in `timer.plot()` | `sc.timer(start=False).plot()` -> `RuntimeWarning: Looks like nothing has been timed...` |
| 1350-1351 | `else: month_token = '%B' # pragma: no cover` in `elapsedtimestr()` | `sc.elapsedtimestr(now - dt.timedelta(days=10), shortmonths=False)` -> `'29 August'` |
| 1356-1357 | `else: date_str = date.strftime('%d ' + month_token) # pragma: no cover` in `elapsedtimestr()` | `sc.elapsedtimestr(now + dt.timedelta(hours=1))` -> `'8 Sep'` (same-year path) |
| 1369-1374 | `if isinstance(pasttime, str): # pragma: no cover` in `elapsedtimestr()` | `sc.elapsedtimestr('2020-04-04')` -> `'4 Apr 2020'` (the docstring advertises ISO-8601 string input) |
| 1377-1427 | `else: # pragma: no cover` -- the entire past-time branch of `elapsedtimestr()`, i.e. the function's whole reason to exist | `sc.elapsedtimestr(sc.now() - dt.timedelta(hours=3))` -> `'3 hours ago'`; `tests/test_datetime.py:123` already exercises it |

The `elapsedtimestr()` pragma at 1377-1427 is the notable case: it sits on the `else:` of the "is this a future date" test, so coverage.py excludes roughly 50 lines containing all of the minute/hour/day/date formatting logic -- which is where the "0 days ago" defect above (finding 20) lives, undetected.

Legitimately pragma'd in the date-functions region (deprecation shims and error paths, body not reached in normal use): lines 72, 112, 122, 249, 309, 379, 433, 461, 714.

## Verified clean

**`sc.time()`, `sc.now()`, `sc.getdate()`.** `sc.time()` matches `time.time()`. `sc.now(utc=True)` returns a genuinely re-offset UTC instant, not a relabelled local time: on a UTC-4 host, `sc.now(utc=True).replace(tzinfo=None) - sc.now()` was `+14400.0` s and the value agreed with `dt.datetime.now(dt.timezone.utc)` to the microsecond. `sc.now(timezone='US/Pacific')` likewise returned `16:49-07:00` against a local `19:49`, with `tzfile('America/Los_Angeles')` attached. The `isinstance(utc, str)` shortcut works (`sc.now(utc='US/Pacific')` is equivalent to `timezone='US/Pacific'`). `astype` in `'dateobj'`/`'str'`/`'int'`/`'float'` and `dateformat=` all behave as documented, and `dateformat` correctly forces `astype='str'`. `getdate()` on a `dt.datetime` is correct for all `astype` values (the `dt.date` crash is reported above).

**`sc.readdate()`.** Every one of the 21 entries in the default `formats_to_try` dict parses its own canonical example correctly, with no cross-format mis-parse (checked one sample per key, all 21 returned the expected datetime; e.g. `'20200321'` -> 2020-03-21, `'21 March 2020'` -> 2020-03-21, `'Sat Mar 21 23:09:29 2020'` -> the correct datetime). Format ordering does not cause a wrong-but-parseable answer: `'default2'` (`%Y-%m-%dT%H:%M:%S.%f`) being listed before `'iso'`/`'iso-tz'` does not shadow them, and `'iso-min'` does not swallow tz-suffixed strings. The v3.3.0 ISO 8601/timezone claims hold: `'2023-10-12T00:05:12Z'` -> tz-aware UTC datetime; `'+05:30'` and `'-08:00'` offsets -> correct `timezone(timedelta(...))` with the wall-clock fields left exactly as written (not silently shifted), and fractional-second plus offset (`'...12.123456+05:30'`) also parses. `sc.date()` on those strings returns the as-written calendar date, and `to='pandas'` preserves the offset (`Timestamp('2023-10-12 00:05:12+0530', tz='UTC+05:30')`) -- so no silent shift anywhere in the chain. `'2023-10-12T00:05Z'` (minute precision plus zone) is genuinely unsupported and raises cleanly rather than mis-parsing. `dateformat='dmy'` and `'mdy'` disambiguate `'04-03-2020'` to 2020-03-04 and 2020-04-03 respectively; the bare ambiguous string raises rather than guessing; and `dateformat='dmy'` correctly *refuses* an ISO string rather than silently falling back. `return_defaults=True` returns the 21-entry dict and takes precedence over a supplied `datestr`. Numeric input works for `'posix'` (default) and `'ordinal'`/`'matplotlib'`, raises a clear error for a strftime format, and mixed string/number varargs (`sc.readdate('20200321', 1611661666)`) return both correctly. List, `np.ndarray` and varargs inputs all preserve length, order and container type.

**`sc.date()`.** The `to` x input-type matrix was run exhaustively over `to` in `'date'`/`'str'`/`'string'`/`'pandas'`/`'numpy'` against `str`, `dt.date`, `dt.datetime`, `np.datetime64`, `pd.Timestamp` and `int`+`start_date`: all 30 combinations give the right calendar instant (only `to='datetime'`, reported above, fails). `as_date=True/False`, `asdate=`, `outformat=`, `readformat=` (`'dmy'`, a custom strftime, `'posix'`, `'ordinal'`) and the `format=` alias all reach the code that should use them. `outformat` is correctly ignored when the output is not a string. `sc.date(1986, 4, 4)` and `sc.date(year=1986, month=4, day=4)` both build the right date, and a partial y/m/d triple raises. List input returns a list of the same length and order (checked with a deliberately unsorted 4-element list including `2019-12-31` and `2020-02-29`); `np.ndarray` input returns an array; `None` elements pass through. `sc.date()` does not mutate or alias its input: `_sanitize_iterables()` deep-copies, the input list and array were byte-identical afterwards, and neither `sc.date(dt.date(...))` nor `sc.date(dt.datetime(...), to='pandas')` returns the same object it was given. `2020-02-29` parses and `2100-02-29` correctly raises.

**Calendar and date-arithmetic checks (`sc.day()`, `sc.daydiff()`, `sc.daterange()`, `sc.datedelta()`, `_get_year_length()`, `sc.yeartodate()`, `sc.datetoyear()`).** With an explicit `start_date`, `sc.day()` is correct including negative results (`sc.day('2021-01-21', start_date='2022-02-22')` -> -397, and the docstring's `[-397, 772]` two-element example is right). Day-of-year is right at both year boundaries and across the leap day (`'2020-01-01'` -> 0, `'2020-12-31'` -> 365, `'2021-12-31'` -> 364). DST transitions do not perturb the count (`sc.day(['2021-03-13','2021-03-15'], start_date='2021-03-13')` -> `[0, 2]`, not 1 or 3), because the arithmetic is done on `dt.date` objects. Numeric elements pass through as ints and mix correctly with strings. `sc.daydiff()` pairs consecutive arguments as documented -- `output[i] = days[i+1] - days[i]` -- verified with 2, 3 and 4 arguments (`'2020-01-01','2020-01-02','2020-01-04','2020-01-08'` -> `[1, 2, 4]`), returns a scalar for exactly one pair, signs correctly for reversed arguments (-16), spans the leap day correctly, and accepts a single list argument (rejecting list-plus-extra-args). The one-argument days-since-Jan-1 form computes the right number (the docstring's *stated* number is wrong; see finding 17). `sc.daterange()`'s `inclusive=True/False` really does include/exclude the end for day-granularity ranges: `'2020-03-01'`..`'2020-04-04'` gives exactly 35 dates ending `'2020-04-04'`, and `inclusive=False` gives 34 ending `'2020-04-03'`. A single-day range returns a one-element list (and `[]` with `inclusive=False`); an end before the start returns `[]` rather than raising, in both string and `asdate=True` modes. The leap day and the year boundary are included with no gap or duplicate (`'2020-02-27'`..`'2020-03-02'` and `'2019-12-30'`..`'2020-01-02'`). `interval='week'` and `interval=dict(days=3)` step correctly; the `datedelta()`-kwargs form (`weeks=5`) yields 36 dates as expected for an inclusive 35-day span (but ignores `readformat`; finding 27); output type follows the input type unless `as_date`/`asdate` overrides it, and `outformat` is honoured. Only the month/year `interval` drift (finding 6) and the int `interval` crash (finding 5) failed. `_get_year_length()` gets the century leap rules right: 1900 -> 365, 2000 -> 366, 2020 -> 366, 2021 -> 365, 2100 -> 365, 2400 -> 366. `datedelta()` month/year clamping matches `relativedelta` semantics: `'2021-01-31'` + 1 month -> `'2021-02-28'`, `'2021-03-31'` - 1 month -> `'2021-02-28'`, `'2020-02-29'` + 1 year -> `'2021-02-28'`, - 1 year -> `'2019-02-28'`, and day arithmetic crosses the year boundary correctly in both directions. `+1 month` then `-1 month` from `'2021-01-31'` gives `'2021-01-28'` rather than the original -- **not** reported as a bug: calendar month arithmetic is not invertible once a day-of-month is clamped, and this is documented `dateutil` behaviour that any fix would have to break. Combining `days`/`weeks`/`months`/`years` in one call is order-independent in the right way (`relativedelta` applies the year/month fields before the day/week fields, so `sc.datedelta('2021-01-31', months=1, days=1)` -> `'2021-03-01'` matches doing the months first then the days manually). `sc.datedelta(days=3)` with no date returns a bare `relativedelta`. `sc.yeartodate()`/`sc.datetoyear()` round-trip exactly in the date -> year -> date direction for all eight dates tried, including `'2020-02-29'`, `'2000-02-29'`, `'2020-12-31'`, `'2019-12-31'` and `'2100-03-01'`; the year -> date -> year direction is exact at 2000.0/2000.5/2020.0/2020.5 and elsewhere off by at most one day's worth (<= 0.0014 yr), which is the documented "to the nearest day" rounding, and `_get_year_length()` is correctly consulted so a leap year uses 366. `yeartodate(as_date=False)` and the `asdate=` alias both return the string form; an integer year returns Jan 1.

**Unit conversion (`_convert_time_unit()`, `toc(unit=...)`).** The factor and the printed label are always drawn from the same mapping entry and never applied twice: forcing a known elapsed time of 1.5 s gives `1.50 s`, `1500 ms`, `1500000 mus`, `1500000000 ns`, `0.0250 min`, `0.000417 hr`, and the numeric spellings `unit=1`, `60`, `3600`, `1e-3`, `1e-6`, `1e-9` resolve to `('s')`, `('min')`, `('hr')`, `('ms')`, `('mus')`, `('ns')` with the matching factors -- all 33 accepted spellings were checked against the table. The `label` loop variable is only returned after a `break`, so it cannot leak a stale key. `sigfigs` affects only the formatted string, never the returned value (`sigfigs=1,2,3,10` all return exactly `1.23456789`). The `unit='auto'` ladder is monotone and its boundaries are self-consistent (`<1e-7` ns, `<1e-4` mus, `<1e-1` ms, else s); `elapsed=0` reports `0.00 ns` and a negative elapsed reports ns, which is odd but harmless. `sc.toc(unit=...)` returns raw seconds regardless of unit, which matches the documented meaning of `unit` ("the unit of time to display"). `sc.toc()`'s `doprint` deprecation shim warns and then correctly maps onto `verbose`, and an unrecognized unit raises a clear `ValueError` listing the alternatives.

**`timer` bookkeeping.** For the intended resetting workflow the accounting is exactly right: four 0.05 s `toctic()` laps give `timings` `[0.0501, 0.0502, 0.0503, 0.0501]` summing to `0.2007` against `total 0.2013` and `toctotal() 0.2021` (the ~0.6 ms excess is the `pytime.time()`/formatting overhead between the elapsed calculation and the `_tocs.append`, not a double count), with `len(T) == len(T.timings) == 4`, no duplicated or missing first entry, and `cumtimings[-1] == total`. Labels land on the right entries: constructor `label`/`baselabel` compose as `BASE: ctor: ...`, a per-call `T.toc('override')` replaces the label for that entry only, `auto=True` produces `(0)`, `(1) named`, ... exactly as documented, and a repeated identical label is disambiguated as `same`, `(1) same`, `(2) same` with no overwrite. Using the timer as a context manager still records the lap when the block raises (`timings {'block': 0.0502}` after a `ValueError` propagates), nesting an inner timer inside an outer one gives 0.05 / 0.10 correctly, `reset=True` re-tics the object, and `__iadd__`/`sum()` never lose or overwrite an entry even when three timers all use the label `'a'` (keys become `a`, `(1) a`, `(2) a`). `rawtimings`, `sum()`, `min()`, `max()`, `mean()`, `std()` all agree with `timings`. Note (not a bug, but easy to misread): plain non-resetting `T.toc()` records *cumulative* times (`[0.050, 0.101, 0.151]`), so `sum() = 0.302` is double `total = 0.151` and `mean`/`max` describe cumulative rather than per-lap times -- `sum()`'s docstring only claims to be "similar to `timer.total`". `toctotal()` correctly measures from the first tic (0.101 s after two 0.05 s laps) and deliberately does not add an entry to `timings`. `sc.timer` also works as a decorator, `Timer` is a true alias, and `timer.plot()` produces both subplots with the axis label taken from the same `_convert_time_unit()` call used to scale the bars.

**`elapsedtimestr()`.** Every boundary except the 86399 s case above (finding 20) is right: 60 s -> `a minute ago`, 119 s -> `a minute ago`, 120 s -> `2 mins ago`, 3599 s -> `59 mins ago`, 3600 s -> `1 hour ago`, 86400 s -> `yesterday`, 47.9 hr -> `yesterday`, 48 hr -> `2 days ago`, 4.9 days -> `4 days ago`, 5 days -> falls through to the date `3 Sep` (matching the documented `maxdays=5`). Singular/plural wording is correct throughout (`a minute ago` vs `N mins ago`, `1 hour ago` vs `N hours ago`, `yesterday` vs `N days ago`). `minseconds` behaves as an inclusive threshold (`minseconds=10` -> `just now` at exactly 10 s, `11 secs ago` at 11 s; `minseconds=60` -> `just now` at 30 s), `maxdays` shortens the days window as documented (`maxdays=1` -> `7 Sep` for 25 hr ago) and does not disturb the sub-day branches, and `shortmonths=False` gives `29 August`. A future timestamp takes the print-the-date path and correctly includes the year only when it differs (`8 Sep` for one hour ahead, `13 Oct 2027` for 400 days ahead). Leading-zero stripping in `print_date` works, and the string-input path parses naive ISO dates (`'2020-04-04'` -> `'4 Apr 2020'`); timezone-aware strings and datetimes crash (finding 29), and `dt.date` input is rejected (finding 30). The "yesterday"/"N days ago" wording is based on elapsed 24-hour blocks rather than calendar days, so something at 23:00 today can read `yesterday` -- that is what the code says it does and is not treated as a defect.

**`timedsleep()`.** Timing is accurate and the compensation logic works: 20 iterations of `timedsleep(0.005)` around a real computation took 0.105 s (versus the 0.10 s ideal), and an explicit `start=` argument correctly shortens the pause (asked for 0.05 s with 0.02 s already elapsed, slept 0.0300 s). The `'start'` sentinel and the bare-`None` call are equivalent, the `global _delaytime` is properly deleted after use, and a subsequent call with no `'start'` falls back to "sleep the full delay" rather than erroring or sleeping a stale interval. The `_sleep_overhead` subtraction does not overshoot into a negative sleep because of the `max()` clamp.

**`randsleep()`.** With both bounds, or with the list form, or with the `delay`/`var` form, the draws match the documentation exactly (200 draws each): `delay=0.001` -> [2.1e-05, 0.0020] mean 0.00105; `delay=0.002, var=0.1` -> [0.00180, 0.00220] mean 0.00200; `delay=[0.0005, 0.0015]` -> [0.00051, 0.00150] mean 0.00101; `low=0.0005, high=0.0015` -> [0.00050, 0.00149] mean 0.00100. The returned value is the duration actually slept (returned 0.001713 vs measured 0.001787). `seed` is genuinely reproducible (two seeded calls returned bit-identical `0.001250190933209334`) and, importantly, does **not** disturb the global numpy RNG: `np.random.rand()` returned `0.9296160928171479` both before and after a seeded `randsleep`, because the function builds its own `np.random.default_rng(seed)`. Unseeded calls differ from each other. `low > high` is rejected by numpy (`ValueError: high - low < 0`) rather than silently sleeping.

## Rejected on review

These original findings were removed from the severity sections and the summary table after re-verification on 2026-09-25. Their numbers are retired, not reused.

- **11. `sc.timer(start=False).toc()` silently reports ~1.79e9 s** — Not worth fixing: the docstring says that with `start=False` you must call `timer.tic()` explicitly, so calling `toc()` first is user misuse, and the result is obviously wrong unless an unrelated `sc.tic()` ran earlier (raising a clear error would be a nicety, not a bug fix).
- **16. `sc.datedelta()` decides the output type from the first list element only** — Not worth fixing: the loop-carried `as_date` is real, but it only matters for lists that mix `str` and date elements, which is unusual, and the output is still a valid date in a sensible type.
- **18. `sc.date(to='numpy')` returns a different `datetime64` unit depending on input type** — Not worth fixing: partly false, since the two values compare equal (`sc.date('2020-04-05', to='numpy') == sc.date(dt.date(2020,4,5), to='numpy')` is `True`) and the unit difference shows only in the repr, while object dtype for array input is shared by every `to=` option and matches the documented "list/array of" return.
- **22. `sc.randsleep()` does not validate that the low bound is non-negative** — Not worth fixing: `var` is documented as giving a range of "0 to 2*interval" at its default of 1, so `var > 1` (a negative lower bound) is a user input error, not a bug.

## Suggested order of work

1. **Findings 1, 2, 3** — the high-severity group. All three are silently wrong numbers on ordinary calls, and each is a small, local, well-understood fix (recompute a per-element default inside the loop; accumulate instead of overwrite; pop a keyword before forwarding).
2. **Finding 12** — the `timer` bookkeeping hole (`indivtimings` for a re-entered or combined timer). It is silently-wrong-not-crashing, and the fix needs care so the non-resetting `toc()` case keeps working.
3. **Findings 5, 6, 4, 7, 8, 9, 10, 13, 14, 27, 29, 30** — the remaining medium findings: documented arguments and input types that crash or are silently ignored (`daterange` int/month-year interval and `readformat` with kwargs, `date(to='datetime')`, `getdate`/`datetoyear` on in-contract types, `toctic` positionals, `randsleep` partial bounds, `elapsedtimestr` on timezone-aware datetimes and on `dt.date`).
4. **Findings 15, 17, 19, 20, 21, 23-26, 28** — the low-severity group: type/unit inconsistencies, docstring errors, dead or ignored arguments (including `datedelta`'s `outformat`), and an unreachable warning branch.

Findings 1, 2, 3, 6, 12, and 20 are the ones most worth fixing first for a second reason beyond severity: they return a plausible-looking wrong value rather than raising an exception. A wrong day-of-year, a dropped `days=` argument, a shrunken elapsed time, a date-range that quietly stops short of its endpoint, or an "0 days ago" label do not look like errors to a caller — they look like data, so they can propagate silently for a long time before anyone questions them.
