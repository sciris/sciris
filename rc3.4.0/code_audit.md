# Code audit

- **Project**: `/home/cliffk/sc/sciris`
- **Tier**: 1 (Software library or digital public good used by many people for many years)
- **Strictness**: 1 (strict)
- **Config**: none
- **Overall score**: 92/100
- **Status**: PASS
- **Date**: 2026-08-09
- **Version**: idm-standards:audit-code v2.0_2026.06.10
- **Time spent**: 190s

## Summary

| Category | Score | Weight |
| -- | -- | -- |
| Quality | 90/100 | 40% |
| Usability | 89/100 | 40% |
| Safety | 100/100 | 20% |
| **Total** | **92/100** | 100% |

| Metric | Score | Notes |
| -- | -- | -- |
| correct | 9/10 | 94 test functions with ~95% measured coverage, CI across 5 OS/Python combinations plus scheduled and downstream test workflows; JOSS-published and peer-reviewed. A few `# pragma: no cover` exit branches in `sc_asd.py` remain untested. |
| clear | 9/10 | One `sc_*.py` module per functional domain, descriptive names, comprehensive Google-style docstrings with version history and runnable examples on nearly all public functions; a handful of small methods lack docstrings. |
| concise | 9/10 | No significant copy-paste duplication across ~22k lines; numpy/pandas used appropriately rather than hand-rolled loops. |
| simple | 9/10 | Unified `import sciris as sc` namespace, one-liner common workflows, specific and helpful error messages; the large set of aliases makes the canonical name occasionally ambiguous. |
| powerful | 9/10 | Extensive kwargs-based configurability and composable/subclassable base classes (`sc.prettyobj`, `sc.odict`, `sc.objdict`); no docs section demonstrates subclassing patterns. |
| performant | 8/10 | Appropriate algorithms and data structures throughout, with real profiling/benchmarking infrastructure (`sc.benchmark()`, `sc.profile`, `sc.cprofile`, `sc.mprofile`) and load-balanced parallelization; no continuous benchmark-regression tracking in CI. |
| documented | 9/10 | Multiple READMEs, high docstring coverage with runnable examples, 10 Quarto tutorials, an API reference, and a style guide; the "when not to use / tradeoffs" material lives only in `CLAUDE.md`, not the public docs. |
| accessible | 10/10 | Public GitHub repo, MIT license, changelog, contributing guide, code of conduct, 1-command install, on PyPI, community support channels, and AI-optimization (Claude plugin, `context7.json`, GitMCP). |
| compliant | 10/10 | MIT license, no secrets or credentials found, all 20 declared dependencies permissively licensed (BSD/MIT/Apache/PSF/MPL), all community files present, no PII in test data. |
| reproducible | 10/10 | Loosely specified dependencies (correct for library code), 20+ semantic-version git tags, maintained changelog, published on PyPI, seeds exposed as configurable arguments. |

Sciris scores very well against the Tier 1 rubric. Safety is flawless: MIT-licensed with permissively-licensed dependencies, no secrets, and dependency/version practices that are exactly right for a library (loose bounds, semver tags, PyPI publication). Quality is similarly strong — comprehensive tests at ~95% coverage, CI across five OS/Python combinations, downstream-package regression testing, and a JOSS publication behind it. The remaining gaps are all refinements rather than defects: a few untested termination branches in `sc_asd.py`, no benchmark-regression gate in CI, and user-facing "when not to use Sciris" guidance that currently exists only in `CLAUDE.md` rather than in the published docs.

## Recommendations

1. **[performant] — Add a CI benchmark-regression check** *(effort: medium; automated: yes)*
   Sciris already has `sc.benchmark()` and a benchmark exercised in `tests/test_profiling.py`, but nothing tracks results across commits. Add a workflow (or extend `.github/workflows/test_sciris.yaml`) that runs `sc.benchmark()` on a fixed runner, stores the result as an artifact, and fails or warns when a hot path regresses beyond a threshold. Alternatively adopt `asv` for the handful of critical paths (`sc.odict` access, `sc.dcp`, `sc.save`/`sc.load`, `sc.parallelize` overhead).

2. **[documented] — Publish the "gotchas and limitations" guidance in the docs** *(effort: quick; automated: no)*
   The "Common Gotchas and Limitations" material in `CLAUDE.md` (when *not* to use Sciris: performance-critical inner loops, strict typing requirements, distributed computing, memory-constrained environments; plus migration strategies) is exactly the tradeoff coverage the Tier 1 rubric asks for at 10/10, but end users of docs.sciris.org never see it. Move or mirror it into a `docs/` page.

3. **[correct] — Cover the remaining `sc_asd.py` termination branches** *(effort: quick; automated: yes)*
   `sciris/sc_asd.py` lines 317, 320, and 332 are marked `# pragma: no cover` — the `maxiters`, `maxtime`, and `stoppingfunc` exit paths. Add three short tests in `tests/test_asd.py` that trigger each termination condition and assert on the returned `exitreason`.

4. **[powerful] — Document subclassing and extension patterns** *(effort: medium; automated: no)*
   `sc.odict`, `sc.objdict`, and `sc.prettyobj` are designed to be subclassed and downstream packages (Starsim, Covasim) do exactly that, but `docs/tutorials/tut_advanced.qmd` covers nested dicts and context managers without showing extension. Add a section demonstrating a real subclass with custom `__repr__`/validation.

5. **[simple] — Clarify canonical names versus aliases** *(effort: quick; automated: no)*
   Overlapping names (`sc.load`/`sc.loadobj`, `sc.tolist`/`sc.promotetolist`, and similar pairs) leave newcomers unsure which to reach for. Mark the canonical name in each docstring pair (e.g. "alias for `sc.load()`") and/or group aliases separately in the API reference.

6. **[clear] — Fill the remaining docstring gaps** *(effort: quick; automated: yes)*
   A small number of public-facing methods lack docstrings, e.g. `disp`/`__getitem__` in `sciris/sc_odict.py` and `save`/`try_load` in `sciris/sc_fileio.py`. One-line docstrings would close the gap.

## Proposed solutions

**Recommendation 2 — Publish gotchas and limitations in the docs** *(automated: no)*

Create `docs/whennottouse.qmd` (or add a section to an existing overview page) and register it in the Quarto TOC. Suggested outline, drawn from the existing `CLAUDE.md` content:

```markdown
# When (not) to use Sciris

## Where Sciris helps most
...

## Where to reach for something else
- **Performance-critical inner loops** — `sc.odict` carries small overhead versus built-in `dict`; use `dict` in hot loops.
- **Strict typing requirements** — Sciris's flexibility can mask type errors in low-level libraries.
- **Multi-machine distributed computing** — use Dask, Ray, or Celery instead of `sc.parallelize()`.
- **Memory-constrained environments** — convenience features cost memory.

## Growing out of Sciris
Replace individual functions incrementally; prototype with Sciris, then optimize bottlenecks;
keep Sciris for I/O and utilities while moving compute-heavy parts to specialized libraries.
```

The judgment calls about how prominently to position this (a standalone page versus a section in the intro) are a human decision about the docs' framing, which is why this is not automated.

**Recommendation 4 — Document subclassing and extension patterns** *(automated: no)*

Add an "Extending Sciris" section to `docs/tutorials/tut_advanced.qmd`. A worked example is more valuable than a list; sketch:

```python
class Results(sc.objdict):
    """ Container for simulation results, with automatic summary printing. """
    def __init__(self, npts, **kwargs):
        super().__init__(**kwargs)
        self.npts = npts

    def addresult(self, name, values=None):
        self[name] = np.zeros(self.npts) if values is None else values
        return self

    def summarize(self):
        return sc.objdict({k: v.mean() for k, v in self.items() if isinstance(v, np.ndarray)})
```

Then show the same for `sc.prettyobj` (custom `__repr__` for free) and note which methods are safe to override versus which `sc.odict` relies on internally. Choosing an example that reflects how downstream packages actually subclass these is the human part.

**Recommendation 5 — Clarify canonical names versus aliases** *(automated: no)*

Deciding which member of each pair is canonical is a maintainer judgment, not something to infer mechanically. Suggested process: (1) list the alias pairs — grep for repeated assignments in each `sc_*.py` and in `__init__.py`; (2) for each pair, pick the canonical name (usually the shorter, more modern one); (3) in the non-canonical docstring, open with "Alias for `sc.<canonical>()`."; (4) in the Quarto API reference, either group aliases into a single "Aliases" section or suppress them from the main listing. No deprecation is implied — the aliases stay, they are just labelled.

## Full Results

```yaml
project: /home/cliffk/sc/sciris
tier: 1
strictness: 1
overall_score: 92
failed: false
config: none
quality:
  score: 90
  correct:
    score: 9
    weight: 7
    reason: "94 test functions with ~95% measured coverage, CI across 5 OS/Python combinations plus scheduled and downstream test workflows; sc.asd() cites its peer-reviewed publication and the library is JOSS-published. A few '# pragma: no cover' termination branches in sc_asd.py remain untested."
  clear:
    score: 9
    weight: 2
    reason: "One sc_*.py module per functional domain, descriptive names, comprehensive Google-style docstrings with version history and runnable examples on nearly all public functions; a handful of small methods in sc_odict.py and sc_fileio.py lack docstrings."
  concise:
    score: 9
    weight: 1
    reason: "No significant copy-paste duplication across ~22k lines reviewed; numpy/pandas used appropriately throughout rather than hand-rolled loops."
usability:
  score: 89
  simple:
    score: 9
    weight: 3
    reason: "Unified 'import sciris as sc' namespace with one-liner common workflows and specific error messages; the large number of aliases makes the canonical name occasionally ambiguous for newcomers."
  powerful:
    score: 9
    weight: 2
    reason: "Extensive kwargs-based configurability (sc.parallelize exposes ncpus, maxcpu, maxmem, parallelizer, die, lbkwargs) and composable/subclassable base classes; no docs section demonstrates subclassing patterns."
  performant:
    score: 8
    weight: 2
    reason: "Appropriate algorithms and data structures with real profiling/benchmarking infrastructure (sc.benchmark, sc.profile, sc.cprofile, sc.mprofile) and load-balanced parallelization; no continuous benchmark-regression tracking in CI."
  documented:
    score: 9
    weight: 2
    reason: "Multiple READMEs, high docstring coverage with runnable examples, 10 Quarto tutorials, API reference, and style guide; the 'when not to use' tradeoff material lives only in CLAUDE.md, not the public docs."
  accessible:
    score: 10
    weight: 1
    reason: "Public GitHub repo, MIT license, CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT, 1-command pip/conda/uv install, published on PyPI, community support channels, and AI-optimization (claude_plugin/, context7.json, GitMCP)."
safety:
  score: 100
  compliant:
    score: 10
    weight: 6
    reason: "MIT LICENSE, no hardcoded secrets or credentials found, all 20 declared dependencies permissively licensed (BSD/MIT/Apache/PSF/MPL), all community files present, no PII in test data."
  reproducible:
    score: 10
    weight: 4
    reason: "Dependencies specified without version pins (appropriate for library code), 20+ semantic-version git tags with a maintained CHANGELOG, published on PyPI, seeds exposed as configurable arguments rather than hardcoded."
```
