# `sc_printing.py` bug audit

Audit of `sciris/sc_printing.py` (1738 lines) for genuine defects: wrong or lost output, documented arguments that don't work, silent data corruption, and crashes on in-contract input. Deliberately **out of scope**: unhelpful errors on deliberately wrong input types, style, naming, missing type hints, test-coverage gaps, and performance. The file was covered end to end by two parallel auditors: one over lines 51-1083 (object display, spacing, and data representation — `prepr()`/`pr()`, `objmeth()`/`classatt()`/`objrepr()`, `sigfig()`/`sigfiground()`, `arraymean()`/`arraymedian()`, `printarr()`, `humanize_bytes()`, `createcollist()`), the other over lines 1090-1738 (colour, `heading()`, `progressbar()`/`percentcomplete()`, `capture`, `slacknotification()`). **Method**: line-by-line reading followed by executed hypothesis tests against the editable install (Sciris 3.3.0, numpy 2.4.6, commit `2d69aad`; `requests` 2.32.5 where relevant), with every "actual" value produced by running the code and reproduced a second time before being recorded. Line numbers refer to the current working tree.

**Independent re-verification.** This document was independently re-verified on 2026-09-25 against commit `d91898a` (branch `rc3.4.0`, Python 3.13.9, `SCIRIS_BACKEND=agg`, plus Python 3.10 for finding 30), re-running every reproduction. Of the original 29 findings, 21 were confirmed (some with corrected details or fixes), 1 was rewritten as inaccurate (finding 1, also downgraded from High to Medium), and 7 were rejected as not a bug or not worth fixing (see "Rejected on review"). Three missed bugs were added as findings 30-32. The document now lists 25 findings: 4 High, 16 Medium, 5 Low.

**Nothing in this document has been applied.** All fixes are described, not made.

Two findings here are unusual in a way worth flagging up front: merely *displaying* an object with `repr()`/`sc.pr()` runs every property getter three times (finding 1), and a supposedly protective function can silently delete visible text rather than just leaving unwanted characters behind (`strip_ansi()` eating ordinary characters, not just escape codes, finding 4).

## Summary

| # | Severity | Function | Defect | Line |
|---|----------|----------|--------|------|
| 1 | Medium | `sc.prepr()`, `sc.pr()`, `sc.objmeth()`, `sc.classatt()`, `sc.objrepr()` | Merely `repr()`-ing an object evaluates every property getter three times | 121 |
| 2 | High | `sc.sigfiground()`, `sc.arraymean(tostring=False)` | Silently overflows to a large negative `int64` for values above ~9.2e18 and for `inf` | 734 |
| 3 | High | `sc.printarr()` | Prints float32 (and float16) arrays with zero decimal places | 955 |
| 4 | High | `sc.strip_ansi()` | Deletes visible text when the string contains any non-SGR CSI sequence | 1215 |
| 5 | High | `sc.capture()` | Inherits from both `UserString` and `str`, so its `str` payload is permanently empty | 1687 |
| 6 | Medium | `sc.sigfig()`, `sc.sigfigs()` | Ignores the documented per-character interpretation of a string `formats`, so its own docstring example is wrong | 650 |
| 7 | Medium | `sc.sigfig()`, `sc.sigfigs()` | `keepints` is ignored for negative numbers | 672 |
| 8 | Medium | `sc.sigfig()`, `sc.sigfigs()` | `sigfig(sep=...)` ignores the separator character it is given | 688 |
| 9 | Medium | `sc.arraymean()`, `sc.printmean()` | `arraymean(axis=...)` returns an unformatted repr of two lists instead of a rounded summary | 782 |
| 10 | Medium | `sc.arraymedian()`, `sc.printmedian()` | Mutates the caller's list of quantiles in place, and rejects a tuple | 851 |
| 11 | Medium | `sc.printarr()` | Crashes on an empty array | 954 |
| 12 | Medium | `sc.printarr()` | Loses column alignment as soon as any value is negative | 954 |
| 13 | Medium | `sc.colorize()` | Bare `sc.colorize()` emits nothing, so it does not reset the colour it is documented to reset | 1170 |
| 14 | Medium | `sc.colorize()`, `sc.heading()` | Silently discards the string when the colour name is not recognised | 1173 |
| 15 | Medium | `sc.heading()` | Cannot forward `fg`/`bg`/`style` to `colorize()` because `color` defaults to `'cyan'` | 1252 |
| 16 | Medium | `sc.slacknotification()` | Keeps the trailing newline when the webhook is read from a file | 1406 |
| 18 | Medium | `sc.slacknotification()` | Never checks the HTTP status, so a failed send reports success even with `die=True` | 1419 |
| 19 | Medium | `sc.capture()` | Re-appends the whole buffer if the same object is used twice, duplicating text | 1729 |
| 20 | Low | `sc.createcollist()` (and `sc.objatt()`/`objmeth()`/`objprop()`/`classatt()`/`prepr()`) | Scrambles the column layout when the item count leaves the last column empty | 57 |
| 21 | Low | `sc.prepr()`, `sc.pr()` | The "time exceeded" note always claims exactly 1 entry was not shown | 346 |
| 22 | Low | `sc.sigfig()`, `sc.sigfigs()` | Prints one digit more than requested when rounding crosses a power of ten | 682 |
| 28 | Low | `sc.percentcomplete()` | Prints at the wrong granularity for loops shorter than 100 iterations | 1482 |
| 30 | Medium | `sc.sigfiground()`, `sc.arraymean(tostring=False)` | Crashes on Python 3.10/3.11 whenever a list input rounds to all integers | 743 |
| 31 | Medium | `sc.arraymean(tostring=False)` | Returns NaN for the mean when the data has zero spread | 780-796 |
| 32 | Low | `sc.arraymedian()`, `sc.printmedian()` | Prints unrounded bounds when the median is exactly 0 | 864-869 |

Findings 17, 23, 24, 25, 26, 27 and 29 were rejected on re-verification and are listed under "Rejected on review" near the end.

## Recurring patterns

**Unsigned comparisons used as magnitude guards.** `sigfig()`'s `keepints` test (`x > (10**sigfigs)`, line 672) compares the signed value against a positive threshold, so negative inputs silently take the wrong branch. `printarr()`'s width calculation (`arr.max()`, line 954) is the same mistake in width form. Both want `abs()`. (`humanize_bytes()` has the same unsigned test, but negative byte counts are out of contract; see rejected finding 24.)

**Non-finite intermediate values.** `sigfiground()` casts `inf` to int64 (finding 2), `arraymean()` feeds an infinite sig-fig count into `sigfiground()` when the spread is zero (finding 31), and `arraymedian()` does the same into `sigfig()` when the median is zero (finding 32). All three come from `log10(0)` or `inf` flowing into rounding code that assumes finite values.

**Stale magnitude / stale length after mutation.** `sigfig()` computes `magnitude` before rounding and then formats with a `decimals` derived from that stale value (682); `prepr()` truncates `labels` and then computes `len(labels) - a` from the truncated list (346). In both cases a quantity captured before an operation is used as if it were still valid afterwards.

**Dtype/precision assumptions.** `printarr()`'s `arr.dtype == float` test matches only float64, so float32/float16 arrays fall through to a zero-decimal format (finding 3); `sigfiground()`'s unconditional `.astype(np.int64)` overflows for magnitudes above ~9.2e18 and silently flips sign (finding 2). Both assume a value that is only sometimes true of real scientific data (reduced-precision arrays, large magnitudes).

**Bare `except` hiding a real defect.** `sigfig()`'s `except: output.append(sc.flexstr(x))` (line 691) is what turns the `axis=` bug in `arraymean()` into silently unformatted output rather than an error (finding 9), and `_is_meth()`'s `except: return False` (line 126) turns a raising property into a silent misclassification rather than a report.

**Warn and continue with a known-bad value.** `colorize()`'s unknown-colour branch (finding 14) and `slacknotification()`'s unchecked HTTP response (finding 18) both keep going with state that is known-bad, converting a clear failure into a silent wrong result (a dropped string, a "Message sent." for a failed send).

**Accumulators never reset.** `capture._io` is never truncated or replaced between uses, so reusing the object duplicates the previously captured text (finding 19); in the same class, the `str` payload fixed at construction can never be updated at all (finding 5) — both stem from mutating one view of the object while a second view goes stale.

**A documented argument silently ignored.** This shows up on both halves of the file: `sigfig()`'s string-form `formats` (finding 6) and `sep` (finding 8) are accepted but not honoured; `printarr()`'s `decimals` is inoperative for non-float64 dtypes (finding 3); and `colorize()`/`heading()` silently drop the *string* payload itself when the colour name is not recognised (finding 14) rather than just ignoring the colour request.

## High severity

### 2. `sigfiground()` silently overflows to a large negative integer for values above ~9.2e18 and for `inf` — `sc_printing.py:734`

*Confirmed on re-verification. Note that the `inf` case (`sc.sigfiground(np.inf)`) makes this more likely to be hit in practice than the 9.2e18 framing alone suggests.*

`round_arr()` ends with `if np.all(exponent <= 0): out = out.astype(np.int64)`, on the theory that "they're all integers". For any input whose magnitude exceeds the int64 range the cast wraps, producing `-9223372036854775808` (with only a `RuntimeWarning: invalid value encountered in cast`, which is easy to miss and is suppressed under `warnings.simplefilter('ignore')` or pytest capture).

```python
sc.sigfiground(1e19)               # actual: -9223372036854775808   expected: 1e+19
sc.sigfiground([2e19, 3e19])       # actual: [-9223372036854775808, -9223372036854775808]
sc.sigfiground(np.inf)             # actual: -9223372036854775808   expected: inf
sc.arraymean([1e19,2e19,3e19], tostring=False)
# actual: (-9223372036854775808, -9223372036854775808)   expected: (2e+19, 8.2e+18)
```

The sign flip is the dangerous part: a positive quantity becomes a large negative one, so downstream comparisons (`if x > 0`) reverse. Magnitudes above 9.2e18 are ordinary in scientific use (bytes, joules, particle counts, currency in minor units), and `sc.arraymean(..., tostring=False)` funnels straight into this path. Note the cast is *unnecessary* whenever it is dangerous: the values are already exactly representable as floats. Mixed-magnitude input is unaffected (`sigfiground([1e20, 1.0])` keeps float dtype, because `exponent <= 0` is not true for every element), which makes the failure input-dependent and easy to miss in testing. `tests/test_printing.py` does not test `sc.sigfiground()` at all.

**Fix**: only cast when the values fit, e.g. `if np.all(exponent <= 0) and np.isfinite(out).all() and np.abs(out).max() <= np.iinfo(np.int64).max:`; otherwise leave the float dtype.

### 3. `printarr()` prints float32 (and float16) arrays with zero decimal places — `sc_printing.py:955`

The auto-format branch tests `if arr.dtype == float`, which is only true for float64; every other floating dtype falls into the `else` branch and gets `'%{maxdigits}.0f'`, i.e. all fractional information is discarded from the printout.

```python
sc.printarr(np.array([1.5, 2.25, 3.125], dtype=np.float32))   # actual: '2  2  3'
sc.printarr(np.array([1.5, 2.25, 3.125]))                     # float64: '1.50  2.25  3.12'
```

Actual: `2  2  3`. Expected: `1.50  2.25  3.12`. The printed digits do not merely lose precision, they are a different number, and there is no indication that anything was truncated — the user is looking at a debugging display and will believe it. float32 is exactly the dtype scientific code adopts to halve memory, and Sciris itself produces it (`sc.gauss1d`/`sc.gauss2d` with the default `use32=True`). The `decimals` argument is silently inoperative in the same case. `tests/test_printing.py:67-69` only exercises `np.random.rand(...)`, which is float64.

**Fix**: test `arr.dtype.kind == 'f'` (or `np.issubdtype(arr.dtype, np.floating)`) instead of `arr.dtype == float`.

### 4. `strip_ansi()` deletes visible text when the string contains any non-SGR CSI sequence — `sc_printing.py:1215`

`strip_ansi()` delegates to `ansi.strip_color()`, whose regex is `re.sub('\x1b\\[(K|.*?m)', '', s)` (`sciris/_extras/ansicolors.py:122`; line 113 is the `def strip_color` line). The `.*?m` alternative is not anchored to the SGR grammar, so for any escape sequence that is *not* `ESC[...m` or exactly `ESC[K` the regex keeps consuming ordinary printable characters until it finds the next literal `m` anywhere in the text, and deletes all of it. Every character between the escape and that `m` is silently lost. This is the common case for captured terminal output: `ESC[2K` (erase line, emitted by `tqdm`), `ESC[A` (cursor up, emitted by `tqdm.moveto` — `tqdm.utils._term_move_up()` returns `'\x1b[A'` on this platform), `ESC[2J`, `ESC[H`, `ESC[?25l`.

```python
import sciris as sc
sc.strip_ansi('\x1b[2K\rDownloading 45% complete')
sc.strip_ansi('\x1b[2Jsome message')
sc.strip_ansi('\x1b[?25lhi mom')
sc.strip_ansi('\x1b[31mred\x1b[0m then \x1b[2K more text')
```

Actual (run twice, fresh interpreters, sciris 3.3.0, commit `2d69aad`):

```
'\x1b[2K\rDownloading 45% complete'          -> 'plete'
'\x1b[2Jsome message'                        -> 'e message'
'\x1b[?25lhi mom'                            -> 'om'
'\x1b[31mred\x1b[0m then \x1b[2K more text'  -> 'red then ore text'
```

Expected: `'\rDownloading 45% complete'` (or with the `\r` also removed), `'some message'`, `'hi mom'`, `'red then  more text'`. Note the failure mode is not "the escape survives" but "your text is eaten", which is much harder to notice; the same non-SGR sequences that are left *intact* when nothing follows them (`sc.strip_ansi('hello\x1b[2K')` -> `'hello\x1b[2K'`) destroy content when text follows them.

The docstring of `strip_color()` explicitly disclaims completeness ("does not try to strip every possible one"), so the *missing* strips (`ESC[2K`, `ESC[nA`, `ESC[2J`) are arguably by design; the deletion of visible characters is not. Blast radius: `sc.strip_ansi()` is new in 3.3.0 and is only used at `tests/test_printing.py:32` inside Sciris, but it is the documented way to clean up captured output, and `sc.capture()` around anything using `tqdm` (i.e. `sc.progressbar()`, `sc.progressbars()`, `sc.parallelize(..., progress=True)`) produces exactly the input that triggers this.

**Fix**: replace the regex with a full CSI matcher, e.g. `re.sub('\x1b\\[[0-?]*[ -/]*[@-~]', '', s)` (optionally plus `\x1b[@-Z\\\\-_]` for two-character escapes). That strips SGR, 256-colour, truecolour, cursor-movement and erase sequences, cannot run past the terminating byte, and leaves a truncated sequence at end-of-string alone.

### 5. `capture` inherits from both `UserString` and `str`, so its `str` payload is permanently empty — `sc_printing.py:1687`

`class capture(co.UserString, str, redirect_stdout)` gives every instance two independent views of its text. Python-level operations resolve through `UserString` and see the captured text in `self.data`; anything that reads the object *as* a `str` (C-level consumers, `str` methods called unbound, `%`-formatting) sees the value fixed by `str.__new__` at construction time, which for `sc.capture()` is `''` and can never be updated because `str` is immutable. The two views therefore disagree permanently, and the disagreement is silent.

```python
import sciris as sc, sys, re, json
with sc.capture() as c:
    print('hello world')

str(c)              # 'hello world\n'   (UserString.__str__)
len(c)              # 12               (UserString.__len__)
str.__len__(c)      # 0                <-- the actual str payload
''.join([c])        # ''               <-- silently empty
re.search('hello', c)  # None          <-- silently no match
json.dumps(c)       # '""'             <-- silently empty
str.upper(c)        # ''
sys.stdout.write(c) # writes nothing, returns 0
'%s' % c            # RecursionError
```

Actual, re-verified in a fresh interpreter:

```
C 'hello world\n' 0 '' None ""
D RecursionError
```

Expected: `str.__len__(c) == 12`, `''.join([c]) == 'hello world\n'`, `re.search` matching, `json.dumps(c) == '"hello world\\n"'`, `sys.stdout.write(c)` writing the text, `'%s' % c == 'hello world\n'`. The `RecursionError` has a separate mechanism worth noting: because `capture` is a `str` subclass, `'%s' % c` gives priority to the reflected operation `c.__rmod__('%s')`, and `collections.UserString.__rmod__` is `return self.__class__(str(template) % self)`, which re-enters `'%s' % c` forever (`/software/conda/lib/python3.13/collections/__init__.py:1439`, ~1000 frames then `RecursionError`).

This matters because `sc.capture()` is documented as a testing helper ("useful for testing the output of functions that are supposed to print certain output"): `assert re.search('warning', txt)` and `assert 'x' in ''.join(lines)` fail *quietly and wrongly*, i.e. a test that should pass fails, or an assertion of absence passes vacuously. Blast radius inside Sciris is nil — all three internal call sites consume it safely (`sc_parallel.py:898` does `str(stdout)`, `sc_settings.py:450` does `'Warning' not in stderr`, `sc_profiling.py:626` does `txt.strip().split(...)`) — so this only bites user code.

**Fix**: drop `str` from the base classes (`class capture(co.UserString, redirect_stdout)`); `UserString` already supplies the whole `str` API, and the only thing lost is `isinstance(txt, str)`. Caveat from re-verification: losing `isinstance(txt, str)` is a visible behaviour change, e.g. `sc.isstring(txt)` becomes False, so any user code that type-checks a capture result will change; the 3 internal call sites (listed above) are unaffected. If `isinstance` compatibility must be kept, override `__new__` to seed the `str` payload and have `stop()`/`__exit__` return a fresh real `str`, but the two-views problem is unfixable in general for a mutable accumulator. Removing `str` also fixes the `__rmod__` recursion.

## Medium severity

### 1. Merely `repr()`-ing an object calls every one of its property getters, three times each — `sc_printing.py:121`

*Rewritten on re-verification (2026-09-25): the getter-evaluation claim is confirmed, but the original also claimed that the `cached_property` "mutation" was caused by `_is_meth()` and would be fixed by the proposed change. It is not: `prepr()` deliberately evaluates class attributes (including `cached_property`) in order to display them. Severity lowered from High to Medium: the impact is extra cost and getter side effects, not data corruption.*

`_is_meth()` decides whether a `dir()` entry is a method by doing `obj = getattr(obj, attr, None)` on the *instance* (line 121), so every property is **evaluated** just to find out that it is not a method. `_is_prop()` does the safe thing (`getattr(type(obj), attr)`), but `_is_meth()` is called first, over all of `dir(obj)`, by both `objmeth()` and `classatt()`. `prepr()` calls `classatt(return_keys=True)` once and then `objrepr()`, whose `assemble()` helper evaluates `objmeth()` and `classatt()` again (unconditionally — the `show*` flags only gate whether the string is *used*), so each property getter runs exactly 3 times per `sc.pr()`. The value obtained is then thrown away: properties are excluded from the attribute list, so the cost and any side effects are paid but nothing is displayed.

```python
import sciris as sc
n = [0]
class C(sc.prettyobj):
    def __init__(self): self.x = 1
    @property
    def p(self):
        n[0] += 1          # any side effect: a counter, a lazy load, a DB hit
        return 'v'
c = C()
repr(c)                    # just printing it
print(n[0])                # 3
```

Actual: `3` getter calls from one `repr()`. Expected: `0` getter calls, since property values are never displayed.

Not part of this defect: a `functools.cached_property` is not a `property` instance, so `classatt()` lists it as a class attribute, and `prepr()` then `getattr`s it on purpose to show its value, which caches the result into the instance `__dict__`. That is a consequence of displaying it, by design, and the fix below does not change it (re-verified: with a class-first `_is_meth()`, `cp` still appears in `__dict__` after `repr()`). If that is unwanted, it needs a separate decision to treat `cached_property` like `property` in `classatt()`/`_is_prop()`.

Two further consequences of the getter evaluation: (a) a property whose getter raises is swallowed by `_is_meth()`'s bare `except`, so a broken property is silently misclassified rather than reported; (b) the work is not covered by `maxtime`, which only guards the attribute-value loop, so an expensive property makes `sc.pr()` slow with no way to opt out.

Blast radius: this fires for any class with properties, including Sciris' own. Measured on `sc.counter`, whose `array` property materialises all values into a NumPy array (`sc_odict.py:41`):

```python
c = sc.counter({i:i for i in range(2_000_000)})
# c.array alone: 0.053 s
# sc.prepr(c):   0.161 s   -> ratio 3.0x, and 'array' appears only as a name under Properties
```

`sc.dataframe` inherits pandas' properties, so `sc.pr(df)` (the first example in `prepr()`'s own docstring) evaluates `df.values`, `df.T`, `df.style`, ... 3x each — verified with a `sc.dataframe` subclass that counts `values` accesses: 3 per `sc.prepr()`. `sc.prepr()` is the `__repr__` of `sc.prettyobj`, `sc.quickobj`, `sc.objdict`-style classes and `sc_fileio.py:1771`/`sc_utils.py:2314,2494`.

**Fix**: in `_is_meth()`, look the attribute up on the class first (`getattr(type(obj), attr, None)`), return False for non-function descriptors found there (`property`, `functools.cached_property`, and anything else whose type defines `__get__` but is not a function/classmethod/staticmethod), and fall back to the instance only when the class has nothing. Re-verified by monkeypatching: the getter count drops to 0, and methods, classmethods, staticmethods and instance-assigned lambdas are all still classified correctly. Separately, make `objrepr.assemble()` lazy (pass a callable, or guard each `show*` flag) so the sections that are not shown are not computed.

### 6. `sigfig()` ignores the documented per-character interpretation of a string `formats`, so its own docstring example is wrong — `sc_printing.py:650`

The docstring says: "`formats` (str/list): custom format suffixes; if str (e.g. 'kmb'), split into chars". The code does `formats_list = sc.tolist(formats)`, and `sc.tolist('kmb')` returns `['kmb']` (a string is one item, not three characters), so the whole string becomes the single suffix for 1e3 and there is no suffix for 1e6 or above.

```python
sc.sigfig(3432.3842, SI=True, formats='kmb')   # actual: '3.432kmb'   docstring: '3.432k'
sc.sigfig(3.4e6,     SI=True, formats='kmb')   # actual: '3400kmb'    expected: '3.400m'
sc.sigfig(3432.3842, SI=True, formats=list('kmb'))  # '3.432k'  <- only the list form works
```

The output is both wrong and misleading: `'3400kmb'` reads as 3400 million-billions. The list form documented alongside it works correctly, so only the string shorthand is broken. Not covered by `tests/test_printing.py` (which never passes `formats`).

**Fix**: `formats_list = list(formats) if isinstance(formats, str) else sc.tolist(formats)`.

### 7. `keepints` is ignored for negative numbers — `sc_printing.py:672`

The guard is `elif x > (10**sigfigs) and not SI and keepints:` — an unsigned comparison, so no negative value ever reaches the keep-integer branch and negatives are rounded while their positive counterparts are not.

```python
sc.sigfig( 23432.23, sigfigs=3, keepints=True)   # actual:  '23432'
sc.sigfig(-23432.23, sigfigs=3, keepints=True)   # actual: '-23400'   expected: '-23432'
```

Actual: sign-dependent behaviour. Expected: `abs(x)` compared, so `-23432.23` gives `'-23432'`. A column of values summarised with `keepints=True` therefore mixes full and rounded integers depending on sign — e.g. a table of net changes shows gains exactly and losses rounded. `tests/test_printing.py:87` tests only positive values.

**Fix**: `elif abs(x) > (10**sigfigs) and not SI and keepints:`.

### 8. `sigfig(sep=...)` ignores the separator character it is given — `sc_printing.py:688`

The docstring documents `sep (bool/str): if provided, use as thousands separator`, but both code paths hardcode `format(roundnumber, ',')` (lines 674 and 688), so only the truthiness of `sep` is used and the character is discarded.

```python
sc.sigfig(1234567.0, sep='.')   # actual: '1,235,000'   expected: '1.235.000'
sc.sigfig(1234567.0, sep=' ')   # actual: '1,235,000'   expected: '1 235 000'
```

Actual: commas regardless. Expected: the requested separator. This silently produces the wrong convention for European-format output, and Sciris' own test suite assumes the argument is meaningful — `tests/test_printing.py:85` calls `sc.sigfig(..., sep='.')` and `:88` calls `sep=','`, neither asserting on the separator. Two secondary effects of the same block: converting back through `float(string)` drops requested trailing zeros (`sc.sigfig(3.10, sigfigs=3, sep=True)` -> `'3.1'`, vs `'3.100'` without `sep`), and for small magnitudes it reintroduces scientific notation (`sc.sigfig(1e-12, sep=True)` -> `'1e-12'`, vs `'0.000000000001000'`).

**Fix**: `string = format(roundnumber, ',')` then `string.replace(',', sep)` when `sep` is a string; keep `','` when `sep is True`. Better still, insert the separator into the already-formatted integer part rather than round-tripping through `float()`.

### 9. `arraymean(axis=...)` returns an unformatted repr of two lists instead of a rounded summary — `sc_printing.py:782`

With `axis` given, `val`, `err` and therefore `relsize` are arrays, so `vsf = esf + relsize` makes the *number of significant figures* an array. `sigfig()` loops over the values of `val` but keeps `sigfigs` as the whole array, so `round(x*factor)` gets an array and raises; `sigfig()`'s bare `except` (line 691) swallows it and appends `sc.flexstr(x)`, i.e. the raw unrounded value. The result is a string containing two Python list reprs, with the means at full float precision and the errors rounded.

```python
d = np.array([[10.,20,30],[11,19,33],[12,21,29]])
sc.arraymean(d, axis=0)
```

Actual: `"['11.0', '20.0', '30.666666666666668'] ± ['1.6', '1.6', '3.4']"`. Expected something like `'11.0 ± 1.6, 20.0 ± 1.6, 30.7 ± 3.4'`, or at minimum consistently rounded values. `axis` is a documented, advertised argument (*New in version 3.2.0*) and is untested in `tests/test_printing.py`. The `tostring=False` path is fine (`(array([11., 20., 30.7]), array([1.6, 1.6, 3.4]))`), because `sigfiground()` broadcasts.

**Fix**: when `val` is not a scalar, either format element-wise (loop over `zip(val, err, vsf, esf)` and join) or reduce `vsf` to a scalar; separately, `sigfig()`'s bare `except` at line 691 should not silently return the unformatted value — that is what hid this.

### 10. `arraymedian()` mutates the caller's list of quantiles in place, and rejects a tuple — `sc_printing.py:851`

In the "pair of percentiles" branch, `quantiles = ci` aliases the caller's list and `quantiles[i] = q/100` writes back into it, so the caller's variable is silently rescaled by 1/100. The same two lines make a tuple — the most natural way to write a fixed pair, and equally covered by the docstring ("If a pair of ints or floats is provided") — raise `TypeError`.

```python
bounds = [5, 95]
sc.arraymedian([1,2,3,4,5.], bounds)
print(bounds)                             # actual: [0.05, 0.95]   expected: [5, 95]

sc.arraymedian([1,2,3,4,5.], (5, 95))
# TypeError: 'tuple' object does not support item assignment
```

Actual: caller's list becomes `[0.05, 0.95]`; tuple input crashes. Expected: input untouched; tuple accepted. A caller who keeps a module-level `CI_BOUNDS = [5, 95]` and passes it to `sc.printmedian()` will find it silently converted, and any of their own code that later treats it as a percentile pair (e.g. passing it to `np.percentile`) now gets 0.05/0.95. Related, in the same block: a NumPy integer (`ci=np.int64(95)`) satisfies `sc.isnumber()` but neither `isinstance(ci, int)` nor `isinstance(ci, float)`, so `x` is never assigned and the function raises `UnboundLocalError: cannot access local variable 'x'`. `tests/test_printing.py:104-105` passes list literals, so the mutation is invisible there.

**Fix**: `quantiles = [q/100 if isinstance(q, int) else q for q in ci]` — builds a new list, works for tuples and arrays, and drops the index assignment. Add a `float(ci)`-based fallback for the scalar branch so NumPy scalars work.

### 11. `printarr()` crashes on an empty array — `sc_printing.py:954`

`maxdigits = sc.numdigits(arr.max())` is evaluated before any shape check, so an empty array raises inside NumPy rather than printing nothing.

```python
sc.printarr(np.array([]))      # ValueError: zero-size array to reduction operation maximum which has no identity
sc.printarr(np.zeros((0,3)))   # same
```

Actual: `ValueError` from `np.max`. Expected: an empty (or "empty array") printout. Empty arrays arise routinely from filtering (`arr[arr>threshold]`), and `printarr()` is a debugging aid, i.e. exactly what you reach for when a result is unexpectedly empty.

**Fix**: return early (`if not arr.size: string = '<empty array>'`) or guard the `arr.max()` call.

### 12. `printarr()` loses column alignment as soon as any value is negative — `sc_printing.py:954`

The field width is derived from `arr.max()` only, so the minus sign and the digits of large-magnitude negative values are unaccounted for and overflow the field.

```python
sc.printarr(np.array([-100.5, 1.0, 3.25]))
```

Actual: `-100.50  1.00  3.25` (fields of 7, 4, 4 characters). Expected fixed-width columns, e.g. `-100.50     1.00     3.25`. Because the width comes from the maximum, an array whose largest element is small but whose smallest is very negative is misaligned throughout, which defeats the whole purpose of the function ("Print a numpy array nicely") for 2-D and 3-D data, where the columns are meant to line up between rows.

Re-verified for 2-D data too: floats give `'-100.50  1.00'` over `'3.25  4.00'`, and ints give `'-5  10'` over `' 3  -200'`.

**Fix**: compute the width from `max(sc.numdigits(arr.max()), sc.numdigits(arr.min()))` and add 1 when `arr.min() < 0`. The `+1` is required because `sc.numdigits(-100.5) == 3` does not count the sign.

### 13. Bare `sc.colorize()` emits nothing, so it does not reset the colour it is documented to reset — `sc_printing.py:1170`

With no colour argument, `sc.tolist(None)` returns `[]`, so `ansicolor` stays `''` and (with `string=None`) the function prints an empty line and returns `''`. Two of the function's own docstring examples rely on the opposite behaviour: `sc.colorize(['yellow', 'bgblack']); print('Hello world'); print('Goodbye world'); colorize() # Colorize all output in between` and `sc.colorize() # Stop typing in magenta`. Since `sc.colorize('magenta')` (no string) emits a bare `\x1b[35m` with no reset — correct, that form is meant to be persistent — the terminal stays magenta for every subsequent line of the session.

```python
import sciris as sc
sc.colorize(output=True)             # actual: ''          expected: '\x1b[0m'
sc.colorize('magenta', output=True)  # '\x1b[35m'          (persistent, by design)

with sc.capture() as txt:
    sc.colorize('magenta')
    print('mid')
    sc.colorize()                    # docstring: "Stop typing in magenta"
str(txt)  # actual: '\x1b[35m\nmid\n\n'  -- no \x1b[0m anywhere
```

`tests/test_printing.py:18` works around this by calling `sc.colorize('reset')` and keeping the docstring's comment ("Colorize all output in between"), which suggests the author knew the reset form was `'reset'` and the docstring was never updated. Two smaller docstring defects in the same block: the example writes `colorize()` unqualified, which is a `NameError` in user code, and the `**Examples**` fences use single backticks rather than triple backticks (a pre-existing pattern across this module, so probably a doc-build convention).

**Fix**: either default the colour to reset when nothing else is supplied (`if color is None and string is None and not alt_usage: color = 'reset'`), or change the two docstring examples to `sc.colorize('reset')`.

### 14. `colorize()` silently discards the string when the colour name is not recognised — `sc_printing.py:1173`

The unknown-colour guard prints a message about the colour and then `return`s — with no value and, more importantly, without ever emitting `string`. A one-character typo in the colour therefore deletes the payload instead of the decoration, and because the guard runs before anything else, `output=True` returns `None` rather than the plain text.

```python
import sciris as sc
sc.colorize('rd', 'important message', output=True)
# printed:  Color "rd" is not available, use colorize(showhelp=True) to show options.
# returned: None      expected: 'important message' (uncoloured) or a raised ValueError

sc.heading('Hi', color='blu')
# printed:  Color "blu" is not available, ...    -- the heading itself never appears
```

Both pasted from the run. The `output=True` case is worse than a lost message: the `None` propagates, so the documented pattern `print("prefix: " + sc.colorize(..., output=True))` raises `TypeError` at a site far from the typo. Contrast the `fg=`/`bg=` path, which raises cleanly (`sc.colorize('hi', fg='notacolour')` -> `ValueError("Could not parse color 'notacolour'")`), and contrast the module's own fallback for unsupported terminals, which deliberately keeps the string (`ansistring = str(string)`). Line 1173 is marked `# pragma: no cover`.

**Fix**: print the warning but continue with `ansicolor = ''` (falling through to the uncoloured string), or raise `ValueError`; either way do not swallow `string`.

### 15. `sc.heading()` cannot forward `fg`/`bg`/`style` to `colorize()` because `color` defaults to `'cyan'` — `sc_printing.py:1252`

`heading()` documents `kwargs (dict): passed to sc.colorize()`, and `colorize()` documents `fg`/`bg`/`style` as the alternative way to specify colour. But `heading()` always passes its own `color='cyan'` through, and `colorize()` raises as soon as both are present.

```python
sc.heading('Hi', fg='red', output=True)
# ValueError: You can supply either color or fg, but not both
sc.heading('Hi', color=None, fg='red', output=True)   # works: '\x1b[31m\n\n——————————\nHi\n——————————\n\x1b[0m'
```

Pasted from the run. Any of the three alternate-usage keywords — including the plausible `style='bold'` — fails on the documented default, and the workaround (`color=None`) is not mentioned anywhere.

**Fix**: in `heading()`, only pass `color` when none of `fg`/`bg`/`style` is present in `kwargs`; or in `colorize()`, treat a `color` equal to the default as absent when the alternate keywords are used.

### 16. `slacknotification()` keeps the trailing newline when the webhook is read from a file — `sc_printing.py:1406`

`slackurl = f.read()` is used verbatim, with no `.strip()`. The documented default is exactly this path — "a plain text file containing a single line which is the Slack webhook... By default it will look for the file `.slackurl` in the user's home folder" — and any text file written by an editor ends with a newline. `requests` does not strip it; it percent-encodes it into the path, so the POST goes to `.../services/XXX%0A`, which is not the webhook.

```python
open('wh.txt','w').write('http://127.0.0.1:9/FROM-FILE\n')
sc.slacknotification(message='hi', webhook='wh.txt', verbose=0)
# recorded url: 'http://127.0.0.1:9/FROM-FILE\n'
requests.models.PreparedRequest().prepare_url('http://127.0.0.1:9/x\n', None)  # -> 'http://127.0.0.1:9/x%0A'
```

Both lines pasted from the run (requests 2.32.5); no request was sent. Combined with finding 18, the user sees "Message sent." and nothing arrives.

**Fix**: `slackurl = f.read().strip()`, and apply the same strip to the `$SLACKURL` and inline-string branches.

### 18. `slacknotification()` never checks the HTTP status, so a failed send reports success even with `die=True` — `sc_printing.py:1419`

`response = post(...)` is only echoed at `verbose>=3`, never inspected; `printv('Message sent.', 2, verbose)` runs unconditionally. Slack signals bad tokens, revoked webhooks and malformed payloads with a non-2xx status and a body such as `invalid_token`, none of which raise in `requests`, so the documented `die=True` strictness cannot fire for the most likely failure.

```python
import requests, sciris as sc
resp = requests.models.Response(); resp.status_code = 404; resp._content = b'invalid_token'
requests.post = lambda url=None, data=None, **kw: resp
sc.slacknotification(message='hi', webhook='https://hooks.slack.com/services/BAD', verbose=2, die=True)
```

Actual output: `'  Sending Slack message\n    Message sent.\n'` — no exception, no warning. Expected: a `RuntimeError` under `die=True` (or a printed warning under `die=False`). Verified with a fabricated `Response`; no network request was made.

**Fix**: after the `post`, add `if not response.ok:` and route through the same `die` branch, e.g. `errormsg = f'Slack message failed ({response.status_code}): {response.text}'`.

### 19. `capture` re-appends the whole buffer if the same object is used twice, duplicating text — `sc_printing.py:1729`

`__exit__` does `self.data += self._io.getvalue()` but never truncates or replaces `self._io`, so a second `start()`/`stop()` cycle (or a second `with` on the same object) adds the *entire* buffer again, including everything captured the first time.

```python
import sciris as sc
t = sc.capture().start(); print('one'); t.stop()
t.start();                print('two'); t.stop()
str(t)   # actual: 'one\none\ntwo\n'   expected: 'one\ntwo\n'

w = sc.capture()
with w: print('A')
with w: print('B')
str(w)   # actual: 'A\nA\nB\n'         expected: 'A\nB\n'
```

Both actual values above are pasted from the run. The docstring advertises the `start()`/`stop()` form as a first-class idiom, and reusing the object to capture a second block is the obvious thing to try; the result is not just extra text but text in a misleading order. Related, same method: calling `stop()` twice (or on a capture that was never started) appends the buffer again *and then* raises `IndexError('pop from empty list')` from `contextlib.redirect_stdout.__exit__`, leaving `self.data` corrupted.

**Fix**: in `__exit__`, do `self.data += self._io.getvalue()` then `self._io = io.StringIO()` (and rebind `redirect_stdout`'s target), or `self._io.seek(0); self._io.truncate(0)`; and guard `stop()` against being called when not active. The `seek(0); truncate(0)` form is the simpler choice, since `redirect_stdout` keeps a reference to the same stream object and so needs no rebinding.

### 30. `sigfiground()` crashes on Python 3.10/3.11 whenever a list input rounds to all integers — `sc_printing.py:743`

*Added on re-verification (2026-09-25).*

`out = [int(x) if x.is_integer() else x for x in out]` runs after `out.tolist()`. When `round_arr()` has cast to `int64` (every exponent <= 0, i.e. the result is all integers; line 734), `tolist()` returns Python `int`s, and `int.is_integer()` only exists from Python 3.12. Sciris declares `requires-python = ">=3.10"` and CI tests 3.10, but nothing in `tests/` calls `sigfiground()` or `arraymean(tostring=False)`, so CI never hits this.

```python
sc.sigfiground([1234, 5678], 2)
```

Actual on 3.10/3.11: `AttributeError: 'int' object has no attribute 'is_integer'` (verified that `(5).is_integer()` raises on Python 3.10, and on 3.13 that the elements reaching line 743 are `int`). Expected: `[1200, 5700]`, which is what 3.12+ returns. The docstring's own third example (`[3.3, 830000, 0, -84000]`) avoids this only because it contains `3.28343`, which keeps the float dtype.

**Fix**: `out = [int(v) if float(v).is_integer() else v for v in out]`, or skip the comprehension when `out.dtype.kind == 'i'`.

### 31. `arraymean(tostring=False)` returns NaN for the mean when the data has zero spread — `sc_printing.py:780-796`

*Added on re-verification (2026-09-25).*

When all values are equal, `err == 0`, so `relsize = floor(log10(|val|)) - floor(log10(0)) = +inf` (line 780) and `vsf = esf + relsize = inf` (line 782). `sigfiground(val, inf)` then computes `factor = 10**inf = inf` and `round(val*inf)/inf = inf/inf = nan`.

```python
sc.arraymean([5, 5, 5], tostring=False)   # actual: (nan, 0)   expected: (5, 0) or (5.0, 0)
sc.arraymean([5, 5, 5])                   # '5.0 ± 0'  (survives only via sigfig()'s bare except, unrounded)
```

Actual: `(nan, 0)`. Expected: `(5, 0)`. This is a silent wrong numeric result, and constant data is ordinary (an all-zero or saturated output, a single repeated measurement). The original "Verified clean" pass checked only the string path.

**Fix**: guard non-finite sig-fig counts, e.g. `if not np.isfinite(vsf): vsf = esf` (and the same for `esf`); alternatively, in `sigfiground()`, leave values unchanged where `exponent` is not finite.

## Low severity

### 20. `createcollist()` scrambles the column layout when the item count leaves the last column empty — `sc_printing.py:57`

The function computes `nrow` and then flattens the columns row-wise with `newkeys += items[x::nrow]`, but prints a fixed `ncol` items per row. That is only self-consistent when at most the final row is short; when `len(items) % ncol == 1` (so the last grid column would hold a single item), the printed rows are re-chunked out of step and items land in the wrong row.

```python
print(sc.createcollist(list('abcd')))
print(sc.createcollist(list('abcdefg')))
```

Actual (4 items):
```
  a                       c                       b
  d
```
Expected:
```
  a                       c
  b                       d
```
Actual (7 items) is `a d g / b e c / f`; expected `a d g / b e / c f`. Reading down the columns — the point of a columnated list — gives `a, d, b` instead of `a, b, c`. This affects every `sc.pr()` of any object with 4, 7, 10, 13, ... methods, attributes or properties, so it is common in practice; counts of 5, 6, 8, 9, 11, 12 are laid out correctly. It is visible in ordinary `sc.pr()` output, e.g. a four-method object prints `Methods: c() m() f()` on one row and `s()` on the next.

**Fix**: build the rows explicitly (`rows = [items[i::nrow] for i in range(nrow)]`, then join each row) instead of flattening and re-chunking, or reduce `ncol` to `ceil(len(items)/nrow)` before flattening.

### 21. `prepr()`'s "time exceeded" note always claims exactly 1 entry was not shown — `sc_printing.py:346`

`labels` is truncated to `labels[:a]` and then appended to on line 345, so by the time line 346 runs, `len(labels) - a` is `1` no matter how many attributes remain.

```python
import time, sciris as sc
class Slow:
    def __repr__(self):
        time.sleep(0.2); return 'slow'
print(sc.prepr(sc.prettyobj({f'k{i}':Slow() for i in range(20)}), maxtime=0.5))
```

Actual: `etc. (time exceeded): 1 entries not shown` after showing 3 of 20 attributes. Expected: `17 entries not shown`. Purely diagnostic, but it is the number the user needs in order to decide how much to raise `maxtime` by, and the analogous `maxitems` message ("7 entries not shown") is correct, so the inconsistency is misleading.

**Fix**: capture `nlabels = len(labels)` before truncating and report `nlabels - a`.

### 22. `sigfig()` prints one digit more than requested when rounding crosses a power of ten — `sc_printing.py:682`

`magnitude` (line 678) is computed *before* the rounding at line 680, and `decimals` (line 682) is derived from that stale magnitude, so when rounding pushes the value up into the next decade the format string keeps one decimal too many. The *value* is correct — this is a display-digit defect, not a wrong number.

```python
sc.sigfig(0.99995, 4)   # actual: '1.0000'   expected: '1.000'
sc.sigfig(9.9999,  4)   # actual: '10.000'   expected: '10.00'
sc.sigfig(999.99,  4)   # actual: '1000.0'   expected: '1000'
sc.sigfig(999999, 4, SI=True)  # actual: '1000.0K'  expected: '1.000M'
```

Actual: 5 significant digits for a 4-significant-figure request. The same stale magnitude is why `SI=True` reports `999999` as `'1000.0K'` rather than promoting to `'1.000M'`. Everything away from a decade boundary is exact (verified for every power of ten from 1e-12 to 1e12).

**Fix**: recompute `magnitude = np.floor(np.log10(abs(x)))` after the rounding step, before computing `digits`/`decimals` (and, for the `SI` case, re-check the magnitude against `format_map` after rounding).

### 28. `percentcomplete()` prints at the wrong granularity for loops shorter than 100 iterations — `sc_printing.py:1482`

`onepercent = max(stepsize, round(maxsteps/100*stepsize))`, whose own comment says "not smaller than 1". The clamp should therefore be `max(1, ...)`; clamping to `stepsize` means that whenever `maxsteps < 100` the print interval becomes `stepsize` *iterations* rather than `stepsize` *percent*, so the requested granularity is multiplied by `100/maxsteps`.

```python
for i in range(50): sc.percentcomplete(i, 50, stepsize=10)   # actual: 0% 20% 40% 60% 80%   expected: 0% 10% ... 90%
for i in range(20): sc.percentcomplete(i, 20, stepsize=5)    # actual: 0% 25% 50% 75%       expected: 0% 5% ... 95%
for i in range(1000): sc.percentcomplete(i, 1000, stepsize=10) # 0% 10% ... 90% (correct)
```

Actual values pasted from the run. Output-only, hence low.

**Fix**: `onepercent = max(1, round(maxsteps/100*stepsize))`.

### 32. `arraymedian()` prints unrounded bounds when the median is exactly 0 — `sc_printing.py:864-869`

*Added on re-verification (2026-09-25).*

`relsize = np.floor(np.log10(abs(median))) - ...` (line 864) is `-inf` when `median == 0`, so `sf - relsize` is `+inf`. `sigfig(bound, inf)` (lines 868-869) then overflows inside `round()`, and `sigfig()`'s bare `except` returns `sc.flexstr(bound)`, the raw float.

```python
sc.arraymedian([-1.234, -0.5, 0, 0.5, 1.234])
```

Actual: `'0 (95% CI: -1.1605999999999999, 1.1605999999999999)'`. Expected: `'0 (95% CI: -1.16, 1.16)'`. A zero median is common for symmetric or centred data (residuals, differences, integer counts). Display-only, hence low.

**Fix**: when `median == 0` (or `relsize` is not finite), fall back to `sf` significant figures for the bounds, e.g. `relsize = np.where(np.isfinite(relsize), relsize, 0)`.

## Misplaced `# pragma: no cover`

| Line | Pragma on | Reachable via |
|------|-----------|----------------|
| 950 | `if arr.dtype == object or arr.dtype.kind in ['U','S','O']:` in `printarr()` | `sc.printarr(np.array([['cat','nudibranch'],[23,2423482]], dtype=object))` — this is `printarr()`'s own second docstring example, and it prints correctly |
| 957 | `else:` (the non-float64 format branch) in `printarr()` | `sc.printarr(np.array([1,20,300]))` prints `1   20  300`; any integer array reaches it, as does any float32 array (finding 3) |
| 1121 | `if not enable:` in `colorize()` | `enable` is a documented argument whose entire purpose is to be set to `False`; `sc.colorize('red', 'hi', output=True, enable=False)` -> `'hi'` |
| 1173 | `if color not in ansicolors.keys():` in `colorize()` | Reached by any colour typo, including via `sc.heading(color=...)`; see finding 14, where it silently swallows the string — worth covering precisely because its behaviour is wrong |
| 1544 | `if every < 1:` in `progressbar()` | This is the documented fractional form of `every` ("if float and <1, print every maxiters*every iteration"). Demonstrated: `sc.progressbar(5, 10, every=0.5, length=4, output=True)` -> `'\r ••—— 50%'`, and the bar prints only at i=0, 5, 10. Nothing in `tests/test_printing.py` passes a float `every` |

## Verified clean

**`sigfig()` / `sigfigs()` / `sigfiground()` numerics.** Swept every power of ten from 1e-12 to 1e12 at `sigfigs=4`: the printed digits match the value and the significant-figure count is exact in all 25 cases (`1e-12` -> `'0.000000000001000'`, `1e12` -> `'1000000000000'`). No double-rounding: `decimals` derived from the pre-round magnitude is always >= the number needed, so `strformat % x` never re-rounds the already-rounded value (the only consequence is the extra trailing digit reported in finding 22). Exact halves round half-to-even consistently at every magnitude (`0.5`/`1.5`/`2.5` at 1 sf -> `'0.5'`/`'2'`/`'2'`; negatives symmetric), and this carries through to larger magnitudes consistently (`sc.sigfig(1250, 2)` -> `'1200'`, matching `round(12.5) == 12`). `0`, `-0.0` -> `'0'`; `nan`, `inf`, `-inf` pass through as `'nan'`/`'inf'`/`'-inf'` (via the bare except, but the output is right). Negatives are formatted with the correct field width at every magnitude tested; no leading-space padding leaked into any output. `sigfigs=None` gives `'3432.38'` as documented. List, tuple and ndarray inputs all return the matching container type (`list`, `tuple`, `list`), and the input array is not mutated. `SI=True` prefixes match the magnitude at every boundary: 999 -> `'999.0'`, 1000 -> `'1.000K'`, 1e6 -> `'1.000M'`, 1e9 -> `'1.000B'`, 1e12 -> `'1.000T'`, 1e15/1e18 -> `'1.000e15'`/`'1.000e18'`, and negatives keep their sign and prefix (`-1e6` -> `'-1.000M'`); there is no off-by-1000. `formats` given as a *list* maps correctly to 1e3/1e6/1e9 in the right order despite the internal `reverse()`. `sc.sigfig()` and `sc.sigfiground()` agree on the same input for every value tested (3.28343, 834874, 0.0001234, 999.99, 0.99995, 1250, 2.5, -83742, 1e-12) once the string is parsed back to a float. All three of `sigfiground()`'s docstring examples reproduce exactly (`3.283`, `834880`, `[3.3, 830000, 0, -84000]`), it handles zero and all-zero input without dividing by zero, and it preserves NaN (the int64 cast is skipped because `exponent` is NaN).

**`arraymean()` / `arraymedian()` interval semantics.** `stds` is applied literally and correctly: `stds=1` gives `± 320` and `stds=2` gives `± 650` for `np.std(data) == 324.03`, `stds=3` gives `± 970` — so the multiplier is not double-counted. **`ci=95` correctly maps to the 2.5/97.5 percentiles** (`x = ci/100/2` -> `[0.025, 0.975]`, matching `np.quantile(data, [0.025, 0.975])`); this is *not* the common half-alpha error. The float form (`ci=0.95`) gives the identical result, `ci='iqr'` gives exactly `np.quantile(data, [0.25,0.75])` labelled `IQR`, and `ci='range'`/`'minmax'` gives the true min and max labelled `min, max`. `mean_sf` and `err_sf` each control the quantity named in the docstring (`mean_sf=2` -> `'1400 ± 600'`, `err_sf=4` -> `'1427.0 ± 648.1'`, both together -> `'1427.0 ± 600'`), and the automatic `relsize` logic gives the error 2 significant figures with the mean scaled to match. The string form's numbers agree with the numeric form's for scalar input (`'1430 ± 650'` vs `(1430, 650)`). NaN data propagates to `'nan ± nan'` rather than a crash; all-identical data gives `'5.0 ± 0'` on the string path (with a `divide by zero encountered in log10` RuntimeWarning, and the mean printed unrounded because `vsf` becomes `inf`); re-verification found that the same input on the `tostring=False` path returns `(nan, 0)`, which is finding 31, and that a median of exactly zero in `arraymedian()` leaves the bounds unrounded, which is finding 32; a mean of exactly zero gives `'0 ± 1.6'`. Not reported as defects, but worth knowing: `arraymean()` uses `np.std`'s default `ddof=0` (population SD), so for n=5 the interval is `sqrt((n-1)/n)` = 10% narrower than the sample-SD equivalent, and `**kwargs` go to `sigfig()`, so there is no way to request `ddof=1`; and `tostring=False` returns the tuple regardless of `doprint`, so `sc.printmean(data, tostring=False)` prints nothing.

**`humanize_bytes()`.** The divisor is applied exactly once (the loop only records `factor`/`label`, it does not divide) and always matches the label it prints: 0 -> `'0 B'`, 1 -> `'1 B'`, 999 -> `'999 B'`, 1000 -> `'1.000 KB'`, 1023 -> `'1.023 KB'`, 1024 -> `'1.024 KB'`, 1025 -> `'1.025 KB'`, 1e6 -> `'1.000 MB'`, 1e9 -> `'1.000 GB'`. `decimals` is honoured for everything at or above 1 KB and deliberately forced to 0 below it (per the code comment), and the `2.3423887e6, decimals=2` -> `'2.34 MB'` result matches the docstring value.

**Object-display partitioning (`objatt`/`classatt`/`objmeth`/`objprop`).** Built a class with an instance attribute, a single-underscore attribute, a name-mangled private, a class attribute, a plain method, a `classmethod`, a `staticmethod`, a `property`, a `functools.cached_property` and a property whose getter raises, and confirmed every one of the ten members appears in **exactly one** of `objatt`/`classatt`/`objmeth`/`objprop` with **no duplicates and none lost**: instance/private/mangled -> `objatt`; class attribute and `cached_property` -> `classatt`; method, classmethod and staticmethod -> `objmeth` (bound and unbound alike, via `types.MethodType`/`FunctionType`); both properties, including the raising one -> `objprop`. The raising property does not break any of the four functions, nor `objrepr()` or `prepr()`. `_is_prop()` correctly looks the attribute up on the type, so it never evaluates anything. `private=False` filters only dunders (single-underscore and name-mangled names are always shown, which is the documented intent), `private=<str>` and `private=<list>` re-admit the named dunders, and `sort=False` preserves insertion order in both `objatt()` and `prepr()`. `__slots__`-only classes fall through to the `__slots__` branch and unset slots are reported as `'N/A'` rather than raising.

**`prepr()` / `pr()` robustness.** A self-referencing object (`o.me = o`) renders in 0.01 s with the nested repr truncated by `maxlen`, no hang and no `RecursionError`; the `maxrecurse` stack check fires as designed. An attribute whose `__repr__` raises falls back to `object.__repr__` and the surrounding output is complete. `maxitems=3` shows 3 attributes and reports `etc. (too many items): 7 entries not shown` with the correct count; `maxlen=20` truncates each value to 20 characters and appends ` [...]` so the truncation is visible; `skip` works as both a list and a bare string and preserves the order of the remaining keys; `vals=False` (i.e. `sc.quickobj`) lists the attribute names without evaluating them. `sc.pr()` is a faithful `print(sc.prepr(...))`. The `labels = objkeys` aliasing at line 307 does mutate `objkeys` in place via `labels += classatt(...)`, but the polluted list is only passed on to a `classatt()` call whose result is discarded (`showclassatt=False`), so it has no observable effect — worth knowing if that call is ever made lazy.

**`printarr()` / `printvars()` / `blank()` / `indent()`.** `printarr()` field widths and `decimals` are correct for non-negative float64 data in 1-D, 2-D and 3-D (`decimals=4` propagates through the 2-D and 3-D recursion because `fmt` is computed once at the top level and passed down), the 3-D vertical separator length matches the data extent of a row (10 characters for two 4-character values plus one separator, excluding the trailing separator), a scalar is accepted via `sc.toarray()`, `doprint=False` returns exactly the string that `doprint=True` prints, and mixed object arrays are right-aligned to the widest entry. `printvars()` takes the caller's namespace as an explicit `locals()` argument rather than walking frames, so there is no frame off-by-one: called from inside a function it reports that function's locals (`x: 42`, `y: 'inner'`), called at module level it reports the module's, and its docstring example (`sc.printvars(locals(), ['a','b'], color='green')`) works. A name missing from the mapping prints `Warning, could not be printed` rather than raising; omitting `localvars` entirely (documented as required) degrades to that same message for every variable instead of an error, which is soft but not wrong. `spaces` and `divider` behave as documented. `indent()` handles multi-line text, `width=None` (no wrapping), and the `n=` fake-prefix removal correctly; `blank()` is a one-liner with no defect.

**`colorize()`.** Reset discipline is correct in every case except the bare `colorize()` documented in finding 13: `colorize('red', 'hi')` emits `\x1b[31mhi\x1b[0m`, consecutive calls each carry their own reset (no bleed between them), a multi-colour list concatenates the codes and still emits exactly one trailing reset (`sc.colorize(['yellow','bgblack'], 'hello', output=True)` -> `'\x1b[33m\x1b[40mhello\x1b[0m'`), the alternate `fg`/`bg`/`style` path resets via `ansicolors.color()`'s `'\x1b[{0}m{1}\x1b[0m'` template, and the colour-only form (`colorize('magenta')`) deliberately omits the reset. The error path emits nothing at all, so no reset is needed there, and `enable=False` returns/prints the bare string. `fg`/`bg`/`style` resolve to the claimed codes: `fg='#ffa044', bg='blue', style='italic+underline'` -> `\x1b[38;2;255;160;68;44;3;4m`, i.e. truecolour 255/160/68, bg 44, styles 3 and 4, matching `STYLES.index()`; every name in the 17-entry `ansicolors` dict maps to the standard 30-37/40-47/0 codes. The `string`/`color` argument swap in the alternate-usage path is correct, and supplying both `color` and `fg` raises as documented. `doprint`/`output` are both honoured via `sc.sc_utils._printout()`. One latent oddity not counted as a finding because Sciris documents `fg` as a `str`: an integer `fg=0` (a legal 256-colour index per `_color_code`) is silently dropped by the `if fg:` truthiness test in `_extras/ansicolors.py:95`, so `sc.colorize('hi', fg=0, output=True)` -> `'hi'` while `fg=1` -> `'\x1b[38;5;1mhi\x1b[0m'`.

**`strip_ansi()`.** Correctly strips basic SGR (`\x1b[31m`), compound SGR (`\x1b[1;31m`), 256-colour (`\x1b[38;5;196m`), truecolour (`\x1b[38;2;255;160;68m`), the standard reset (`\x1b[0m`), the terse reset (`\x1b[m`) and `\x1b[K`. It leaves an incomplete sequence at end-of-string (`'hello\x1b['`, `'hello\x1b[31'`) and a lone `\x1b` not followed by `[` untouched — harmless, no corruption. `len(sc.strip_ansi(s))` equals the visible width for every `sc.colorize()` output tested (single colour, colour lists, and the `fg`/`bg`/`style` path: 5 == len('hello'), 14 == len('cat in the hat')), so the docstring example and `tests/test_printing.py:32` both hold. It coerces its argument with `str()`, so `sc.strip_ansi(sc.capture())` works.

**`heading()`.** `tight=True` does exactly what it claims: `'\x1b[36m\n——————————\nHi\n——————————\x1b[0m'` versus the default `'\x1b[36m\n\n——————————\nHi\n——————————\n\x1b[0m'`, i.e. one leading newline instead of two and none after, so with `print()`'s own newline there is one blank line before and none after; the `tests/test_printing.py:37` invariant (`normal.count('\n') == tight.count('\n') + 2`) is satisfied. `spaces`/`spacesafter` are independent and `spaces=0, spacesafter=0` yields no padding at all; `divider=''` correctly skips the divider rows via the `if fulldivider:` guard; `minlength`/`maxlength` clamp through `np.median`, including the inverted case `minlength=300, maxlength=200` (-> 200); multiple positional args are joined with `sep`; a 250-character string is capped at `maxlength`. Two cosmetic non-bugs: a multi-line heading sizes its divider from the total length including the newlines, and a multi-character `divider` produces `length*len(divider)` characters (`divider='=-'` -> a 20-character rule for `length=10`), overshooting `maxlength`.

**`progressbar()`.** The filled fraction is exactly right at the boundaries: with `maxiters=5, length=10`, i=0 gives 0 filled and `0%`, i=1 gives 2 and `20%`, i=4 gives 8 and `80%`, and i=5 gives 10 and `100%` — there is no off-by-one, `int(length*i//maxiters)` reaches `length` precisely at `i == maxiters`, and the final newline is emitted only on `lastiter`. `every=3` prints at i=0,3,6,9 plus the last iteration; `every=0.5` correctly becomes 5; `label` and `length` are both honoured (`'\rWorking ••—— 50%'`); `maxiters` given as a sized object uses its length; `maxiters=0` short-circuits to `100%` with a full bar rather than dividing by zero; `empty`/`full`/`newline`/`flush` all reach their uses. The iterable/`None` branch returns a real `tqdm` and maps `label` onto `desc` without clobbering an explicit `desc`.

**`percentcomplete()`.** For the docstring's own examples the arithmetic is right: `maxsteps=500, stepsize=1` prints 100 lines, on every 5th iteration as claimed, and `stepsize=10` prints 10 lines on every 50th, also as claimed; `prefix` is prepended verbatim (`'Completeness: 1%'`) and a numeric `prefix` becomes that many spaces. Reaching 99% rather than 100% for `for i in range(maxsteps)` is the ordinary 0-indexing convention, and passing `maxsteps` inclusive does print `100%`.

**`capture`.** `sys.stdout` is restored both on normal exit and when the body raises (the exception propagates and `contextlib.redirect_stdout.__exit__` still runs); nesting two captures works and each gets only its own text (`outer` -> `'o1\no2\n'`, `inner` -> `'i1\n'`). Of the two views, all the Python-level operations agree with each other and with the captured text: `str(c)`, `c.data`, `len(c)`, `==` (and `hash()`, which matches the equal plain string), `in`, `bool()`, `+` in both directions, `.strip()`, `.rstrip()`, `.split()`, `.splitlines()`, `.upper()`, and f-string/`.format()` interpolation; it is only the `str` payload that disagrees (finding 5). It makes no claim about threads or subprocesses, and in fact captures output from child *threads* (because it swaps the process-global `sys.stdout`, which also means it silently swallows unrelated threads' output for the duration) but not from *subprocesses* (`subprocess.run([sys.executable,'-c','print("x")'])` inside a capture writes straight to the real fd 1 and is not captured). `self.stdout`, saved in `__init__`, is never read anywhere in the class.

**`printtologfile()`.** No handle leak on any path: the `with open(...)` is nested inside the `try`, so a write failure still closes the file and an `open` failure never creates one. A missing target directory is handled gracefully (`Warning, could not write to logfile /nonexistent-dir-xyz/log.txt: [Errno 2] No such file or directory`), appends accumulate correctly (`'\nhello\n\nagain\n'`), `message=None` returns immediately, and the default `tempfile.gettempdir()/logfile` target works. A non-string message is reported as a warning rather than coerced (`can only concatenate str (not "int") to str`), which is the deliberately out-of-scope wrong-type case; note the function has no `die` argument, so it can never be made strict.

**`printv()`.** Threshold (`verbose >= thisverbose`), indent width (`' '*thisverbose*indent`), `sc.flexstr()` coercion of non-strings, and `**kwargs` forwarding to `print()` all behave as documented.

**`tqdm_pickle` / `progressbars`.** `progressbars(3, total=[5,10,15], label=['a','b','c'])` assigns per-bar totals and labels correctly (`descs ['a','b','c']`, `totals [5,10,15]`), a scalar `label` becomes `'Sim 0'`, `'Sim 1'`, ..., `update(index)` advances the right bar, `desc` passed through `kwargs` overrides `label`, and the whole object round-trips through `pickle.dumps` (which is the point of `tqdm_pickle.__getstate__` dropping `sp`/`fp`).

## Rejected on review

These original findings were rejected during the 2026-09-25 re-verification and are no longer counted. Numbers are kept so that cross-references remain valid.

- **17. `slacknotification()` discards an explicitly-passed webhook in favour of `$SLACKURL`** — NOT WORTH FIXING: it only triggers for an explicit webhook that is neither a `hooks.slack.com` URL nor an existing file, which is outside the documented usage, and the `$SLACKURL` fallback is a defensible design.
- **23. `arraymean()`'s docstring example shows half the value the function returns** — NOT A BUG: the code is correct and only the docstring example (and the matching comment at `tests/test_printing.py:93`) predates the `stds=2` default, so it needs a one-line doc edit (`± 320` -> `± 650`), not a bug fix.
- **24. `humanize_bytes()` stops at GB and mislabels negative sizes as bytes** — NOT WORTH FIXING: `'1000.000 GB'` is arithmetically correct (no TB/PB is a feature gap), negative byte counts are out of contract, and the `sc.humansize` example is a docs typo.
- **25. `printarr()` double-spaces 2-D arrays but single-spaces the 2-D slices of a 3-D array** — NOT WORTH FIXING: purely cosmetic, and the blank line between 2-D rows may well be intentional.
- **26. `colorize(showhelp=True)` combined with `fg`/`bg`/`style` raises `UnboundLocalError`** — NOT WORTH FIXING: combining help mode with the alternate-usage colour arguments is a nonsensical call.
- **27. `printred()` and friends ignore `ansi_support`, unlike `colorize()`** — NOT WORTH FIXING: it only affects Windows without colorama, and modern Windows consoles render ANSI natively.
- **29. `progressbar(output=True)` does not return what the print path emits** — NOT A BUG: returning the bar content without the trailing `\r`/`\n` terminal control is a reasonable API choice, and `i > maxiters` overflowing the bar is caller error.

## Suggested order of work

1. **Findings 4, 5** — `strip_ansi()` deleting visible text and `capture`'s split `str`/`UserString` identity, both because they corrupt or lose captured text silently and both fixes are small and local (a regex swap; dropping one base class, noting the `isinstance(..., str)` caveat).
2. **Findings 2, 3, 30, 31** — `sigfiground()`'s int64 overflow, `printarr()`'s float32 truncation, the Python 3.10/3.11 crash in `sigfiground()`, and `arraymean(tostring=False)` returning NaN for constant data: all produce wrong output or crashes from ordinary calls on supported Pythons.
3. **Findings 1, 6-12** — property getters evaluated by `repr()`, `sigfig()`'s `formats`/`sep`/`keepints`, `printarr()`'s empty-array crash/misalignment, and `arraymean()`/`arraymedian()`'s formatting and mutation bugs: documented arguments that don't work, hidden cost, and a caller-visible mutation.
4. **Findings 13-16, 18, 19** — the `colorize()`/`heading()` colour-name and forwarding defects, plus `slacknotification()`'s newline and status defects and `capture()`'s buffer-reuse duplication: mostly single-branch fixes, with delivery consequences worth prioritizing despite being "medium".
5. **Findings 20-22, 28, 32 and the pragma list** — cosmetic layout, display-digit and diagnostic-message correctness.

Findings that silently *alter or lose data* rather than merely raising or misformatting: finding 4 (`strip_ansi()` deletes visible characters), finding 2 (the int64 overflow flips sign on large magnitudes and on `inf`), finding 31 (`arraymean(tostring=False)` returns NaN instead of the mean for constant data), finding 3 (float32 arrays lose their decimals in `printarr()`), finding 10 (`arraymedian()` rescales the caller's own list by 1/100), and finding 19 (`capture` duplicates previously captured text on reuse). Finding 1 runs property getters (and so any side effects they have) during display, but does not itself corrupt the object. Findings 5-9, 11-16, 18, 20-22, 28, 30 and 32 are output-formatting, argument-forwarding, or crash/message defects — wrong or missing display rather than corrupted state.
