# Sciris: suggested improvements for the AI-authored era

*Grounded in Sciris 3.2.9 (283 public callables across 17 `sc_*` modules). This is an opinionated list, ordered by return on investment. Some suggestions are deliberately bold and would shift what Sciris is; those are flagged. A closing section defends what should **not** change.*

## Why this list exists

Sciris was designed for a human sitting at a REPL who "dislikes coding and wants something that just works" (its own STYLE_GUIDE). That human learns the library once, keeps its quirks in their head, and reaches for `sc.findnearest` because they remember it exists. Almost every design decision — flexible input types, soft failures via `die=False`, generous aliases, "sensible defaults" chosen at runtime, lowercase class names, no type hints — optimizes for *that* person in *that* moment.

But a large and growing share of code that imports Sciris is now written by an LLM agent, and an agent's constraints are almost the inverse of the REPL user's:

- **It can't see runtime state.** The human runs `sc.load('x.pkl')`, sees `None` come back, and investigates immediately. The agent writes a 60-line script, and a silently-returned `None` surfaces three functions later as an inscrutable `AttributeError`. Soft failures are *worse* for agents than for humans.
- **It reasons from static signals.** Signatures, type information, stub files, docstrings, and — above all — error messages are what the model "sees" without executing. Sciris deliberately exposes almost none of this to tooling: no type hints, no stubs, types only in prose inside docstrings.
- **Its knowledge is frozen and lags.** A model trained before v3.x will confidently emit v2.x-shaped calls. Aliases and flexible signatures multiply the number of plausible-but-wrong incantations, so the agent has more ways to be subtly outdated.
- **It defaults to the popular thing.** NumPy, pandas, and Matplotlib are overwhelmingly represented in training data; Sciris is a rounding error by comparison. Left to its own devices, an agent writes the six-line NumPy version instead of `sc.findnearest`, because it doesn't *know* the one-liner exists. Sciris's core value — brevity — is invisible to the model unless something surfaces it.
- **Errors are its only feedback loop.** When generated code fails, the agent reads the traceback and tries again. An error that says *"invalid method 'expon'; did you mean 'exp'? valid options are [...]"* is fixed in one turn. A bare `ValueError` or a silent `None` costs iterations, tokens, and sometimes the agent just gives up and hand-rolls the logic.

The good news: Sciris's central bet — **brevity through simple interfaces** — is *already* the single most AI-friendly thing about it. `sc.save('f.pkl', data)` is fewer tokens, fewer bugs, and more legible than the pickle+gzip boilerplate it replaces. The suggestions below are mostly about making the *rest* of the library as machine-legible as its brevity is machine-efficient — and doing so, wherever possible, without taking anything away from the human.

Sciris is also ahead of most scientific libraries here: it already ships a `claude_plugin/` with 10 skills, a `context7.json`, and MCP wiring. Several suggestions build directly on that investment rather than starting from scratch.

---

## Theme A — Make Sciris legible to machines

### 1. Ship a structured, machine-readable API index (and an `llms.txt`)

**Summary.** Generate one canonical, machine-readable catalogue of the public API — every callable's name, signature, one-line summary, aliases, canonical example, and deprecation status — and ship it both inside the package and as a docs artifact (`llms.txt` / `api.json`).

**Details.** With only ~283 public callables, the *entire* Sciris surface fits in a single file well within an LLM context window. Auto-generate it from the source (introspect `dir(sc)`, `inspect.signature`, first docstring line, and the `**Example**::` blocks) so it never drifts. Publish two forms from the same data: a dense human/LLM-readable `llms.txt` at the docs root (the emerging convention agents and RAG systems look for), and a `sciris/_api.json` shipped in the wheel so any tool — including `sc.help` — can consume it offline. A single entry might read: `findnearest(series, value=None) — find the index of the array element closest to value; aliases: none; example: sc.findnearest([1,2,3,4,5], 3.7) -> 3`.

**Value.** This is the highest-ROI item on the list. It directly attacks the "the agent doesn't know the one-liner exists" problem: a RAG-equipped agent can load the whole map of the library in one shot and *discover* that `sc.dateformatter` or `sc.safedivide` exists before hand-rolling it. It anchors generation to the *current* version, mitigating training lag. And it becomes the single source of truth that stubs, skills, and docs are generated from (see #8).

**Effort.** Low–Medium. A ~150-line generator script plus a CI step. The data already exists in the docstrings; this just projects it into machine-consumable form.

**Challenges.** Keeping the canonical example per function short and genuinely runnable requires some curation. The `llms.txt` convention is still stabilizing, so the exact format may need to track community norms.

### 2. Ship type stubs (`.pyi`) and a `py.typed` marker — without touching the source

**Summary.** Add PEP 561 stub files carrying type information for the public API, so IDEs, type checkers, and LLMs get real signatures, while the hand-written source stays exactly as type-hint-free as the STYLE_GUIDE demands.

**Details.** Sciris's STYLE_GUIDE explicitly rejects inline type hints, and makes a fair case: annotating genuinely polymorphic parameters like `sc.date()`'s input would produce unreadable `Union` "monstrosities." Stub files resolve this tension cleanly — the *human-facing source* keeps its clean, annotation-free look, while a parallel `.pyi` layer (plus an empty `py.typed`) hands machines the signatures they crave. Types can be liberal (`ArrayLike`, `str | Path | None`) where inputs are flexible and precise where they aren't — exactly the distinction the STYLE_GUIDE already draws for docstrings. Seed the stubs semi-automatically from the docstring `Args:` blocks (which already encode types as prose like `filename (str/Path)`), then hand-refine the high-traffic functions (`save`/`load`, `parallelize`, `odict`, `dataframe`, the date functions).

**Value.** This is what turns on autocomplete grounding, `mypy`/`pyright` checking in *downstream* code, and — critically — gives the LLM a static contract for return types, which docstring prose conveys inconsistently today. An agent that knows `sc.load` returns `Any` but `sc.sha` returns `str` writes better downstream code without executing anything. It also improves the human experience in modern IDEs at zero cost to source readability.

**Effort.** Medium–High. 283 callables is a real amount of surface, but it's bounded, it can be seeded automatically, and it can land incrementally (highest-traffic modules first). Stubs then need light maintenance as signatures change — which #8's generation approach can partly automate.

**Challenges.** Sciris's polymorphism means some stubs will be honestly loose (`Any`), which limits checker value for those functions. Stubs can drift from source if not generated/tested; a CI check that the `.pyi` signatures match the runtime signatures mitigates this. Philosophically, this is the closest thing here to reversing a stated design decision — but because it lives outside the source, it respects the *spirit* (clean code for humans) while serving machines.

### 3. Make discovery AI-native: upgrade `sc.help()` and add `sc.api()`

**Summary.** `sc.help()` already searches docstrings — evolve it from a prose-printing regex search into a structured, fuzzy, machine-consumable discovery tool, and add a companion `sc.api()` that returns the index from #1.

**Details.** Today `sc.help('smooth')` does a substring/regex scan over docstrings and prints results for a human. Three upgrades make it agent-native: (a) fuzzy matching so `sc.help('nearest neighbor')` surfaces `findnearest` — `jellyfish` is *already* a dependency, so this is nearly free; (b) structured return (`output=True` exists; make the returned object a clean list of `{name, signature, summary, example}` records rather than a match dict); and (c) a task-oriented entry point, e.g. `sc.api('save data as json')`, that ranks candidate functions. Think of `sc.api()` as the runtime twin of the shipped index in #1 — same data, live introspection.

**Value.** Gives an agent (or human) a first-class "is there a Sciris function for X?" affordance it can call mid-task instead of guessing or reaching for NumPy. Directly counteracts the "defaults to the popular library" failure mode. For humans, fuzzy search is strictly better than the current exact/regex behavior.

**Effort.** Low–Medium. Builds on existing `sc.help` and an existing dependency; most of the work is the structured return format, which #1 already defines.

**Challenges.** Ranking quality for the task-oriented `sc.api('...')` mode is the hard part; a simple fuzzy-over-summaries baseline is fine to start and avoids adding an LLM dependency to the library itself.

---

## Theme B — Tighten the failure feedback loop

### 4. Make every "invalid input" error self-correcting, and give Sciris typed exceptions

**Summary.** Standardize the pattern already present in a few places — *"X not recognized; must be one of [...]"* — across every validated argument, add "did you mean?" suggestions via the already-present `jellyfish`, and introduce a small Sciris exception hierarchy so callers can branch on error type.

**Details.** Sciris already does this well in spots (`sc.options` lists valid options; `sc_parallel` and `sc_nested` say "must be one of ..."), but many sites still `raise ValueError(errormsg)` with a bare message, and none suggest the nearest valid value. Add an internal helper — `sc._raise_invalid(name, value, options)` — that every choice-validating call routes through, producing: *"invalid `method` 'expon' for sc.smooth(); did you mean 'exp'? valid options: ['exp', 'flat', 'hanning']."* Separately, define `sc.exceptions` (`ScirisError`, `LoadError`, `VersionError`, ...) subclassing the built-ins, so that soft-failure paths and downstream `try/except` can target Sciris-specific conditions instead of catching a generic `ValueError` that might mean anything.

**Value.** Error messages are the agent's feedback loop, and this is the highest-leverage place to spend a modest effort. A self-correcting message collapses a multi-turn guess-and-check into a single fix; the fuzzy suggestion frequently gets it right on the first retry. Typed exceptions let agents (and robust human code) write precise recovery logic. Both are pure wins for humans too.

**Effort.** Medium. Mechanically touch the (bounded) set of `raise` sites and route them through the helper; define the exception classes once. No new dependencies.

**Challenges.** Care needed not to change *which* exception type is raised where downstream code already catches `ValueError`/`TypeError` — the new classes should *subclass* the current ones so existing `except ValueError` keeps working. That preserves backward compatibility while adding specificity.

### 5. Make soft failures loud, and add a global strict mode (bold)

**Summary.** Guarantee that `die=False` paths *never* fail silently — always emit a visible warning — and add `sc.options(strict=True)` to flip every `die` default to raise, so agent-authored pipelines and CI can opt into fail-fast globally.

**Details.** Sciris's `die` keyword is a genuinely good idea, but two things hurt machine users. First, some soft-failure paths return `None` (or a partial result) with little or no signal; a human notices at the REPL, an agent's script sails on until it crashes elsewhere. Make every soft failure route through `sc.warn(...)` so there is always a breadcrumb on stderr. Second, `die` defaults are mixed across the library (~40 functions default to `False`, ~33 to `True`), which is impossible to reason about statically. A single global switch — `with sc.options(strict=True):` or an env var `SCIRIS_STRICT=1` — that makes all `die` defaults raise gives agents and CI one lever to turn the whole library fail-fast, without changing any per-call defaults for existing human users.

**Value.** Silent `None` propagation is one of the most expensive failure modes for agent-generated code, because the traceback points far from the cause. Loud warnings + an opt-in strict mode convert a class of confusing downstream crashes into an immediate, local, correctable error. Humans debugging someone else's (or an agent's) pipeline benefit identically.

**Effort.** Low–Medium. The strict-mode plumbing is a single option consulted when `die` is left at its default; the "always warn" change is localized to the soft-failure branches.

**Challenges.** This is philosophically bold: it nudges Sciris from "forgiving by default" toward "forgiving but never silent." Some users rely on quiet `None` returns as control flow; the mitigation is that default behavior is unchanged unless strict mode is explicitly enabled, and warnings (not errors) are the only always-on addition. Deciding the exact default for `strict` (off, to preserve compatibility) matters and should stay off.

---

## Theme C — Keep the ground truth trustworthy and current

### 6. Make docstring examples executable and verify them in CI

**Summary.** Convert (or mirror) the `**Example**::` blocks into runnable doctests and run them in CI, so every published example is guaranteed to work on the current version.

**Details.** Sciris has 100+ curated examples, but they live in RST `**Example**::` blocks that are *not executed* by the test suite (only a handful of `>>>` doctests exist). That means examples can silently drift from behavior across releases. Agents ingest these examples heavily — via training data, via RAG over the docs, and via the shipped skills — and treat them as ground truth. Un-verified examples are therefore a direct source of confidently-wrong generated code. Adopt `pytest --doctest-modules` (or Sybil for the RST-style blocks) so examples are executed on every push; keep the human-friendly narrative examples where prose is clearer, but ensure the *canonical* example per function is machine-checked.

**Value.** Turns the docstring corpus into a trustworthy, always-current signal for both training and RAG, and catches API drift the moment it happens. Humans copy-pasting from docs get examples that actually run. This also produces the verified `example` field that #1's index and #3's `sc.help` surface.

**Effort.** Medium. Normalizing 100+ example blocks into doctest-parseable form is real work, and some examples (plots, randomness, timing) need `# doctest: +SKIP` or determinism shims. Best done module-by-module.

**Challenges.** Plotting/parallel/timing examples are inherently non-deterministic or side-effecting and will need skips or careful fixtures, so not every example becomes a hard assertion — but even executing-without-error catches most drift.

### 7. Centralize deprecations and designate canonical names

**Summary.** Replace the ad-hoc deprecation warning strings scattered through the code with a single `@sc.deprecated(...)` decorator that feeds a machine-readable registry, and explicitly mark one canonical name per concept among the many aliases.

**Details.** Deprecations today are hand-written `warnmsg = 'sc.x argument "y" deprecated ...'` lines in individual functions — good for humans, invisible to tooling. A `@sc.deprecated(replacement='sc.new', since='3.2.0')` decorator would (a) emit a consistent, greppable warning, and (b) populate a registry that the API index (#1) and stubs (#2) expose, so an agent generating from stale training data gets a clear, machine-legible "use X instead" signal. Separately, Sciris has many aliases (`save`/`saveobj`, `load`/`loadobj`, numerous pandas passthroughs). Aliases are convenient and should stay, but *designating a canonical name* per concept (and marking the rest as aliases in the index/stubs/docs) gives agents a single form to converge on, reducing the "five valid-looking calls, model picks a different one each time" inconsistency.

**Value.** Machine-readable deprecations are one of the few direct levers against training lag: the agent's outdated call still runs but self-documents the migration. Canonical-name designation reduces generation variance and makes downstream code more uniform and reviewable.

**Effort.** Low–Medium. The decorator is small; the registry piggybacks on #1. Auditing aliases to pick canonical names is a modest one-time review.

**Challenges.** Purely additive and low-risk, as long as aliases keep working (they should — the point is to *label*, not remove). The main cost is the judgment calls on which name is canonical.

### 8. Generate the AI-facing artifacts from one source of truth (bold)

**Summary.** Auto-generate the plugin skills, the API index/`llms.txt`, the stubs, and the docs' API reference from the source in CI, so the growing constellation of AI-facing artifacts can never silently drift from the actual library.

**Details.** Sciris already ships hand-written assets that describe its API to machines: 10 skill files in `claude_plugin/`, `context7.json`, MCP wiring, and the Sphinx API docs. Every one of these is a copy of the truth that lives in the source, and every copy can rot independently — a renamed argument or a new function won't propagate. The bold move is to make the source the *only* place API facts are written, and generate everything else: the skills' function lists, the `llms.txt` (#1), the stubs (#2), and the docs API pages all fall out of one introspection pass, checked in CI so a stale artifact fails the build. Hand-written prose (the *why*, the tutorials, the philosophy) stays hand-written; only the *enumerated API facts* are generated.

**Value.** This is what makes suggestions #1, #2, #6, and #7 *stay* true release after release with near-zero ongoing effort. Given that Sciris is already investing in multiple AI-facing surfaces, the drift risk is real and compounding; this caps it. It's arguably the difference between "Sciris did an AI-friendliness project once" and "Sciris is durably AI-friendly."

**Effort.** Medium–High. Requires building the generators and reworking the hand-maintained skills to be templated around generated data. Front-loaded, but it *reduces* long-run maintenance.

**Challenges.** Generated skills may read less naturally than lovingly hand-written ones; the fix is to generate the factual scaffolding and let humans own the narrative sections. There's also a bootstrapping cost before the payoff. This reframes the plugin from an artifact into a build target, which is a mindset shift for the maintainers.

---

## Theme D — Make it just-work in a fresh sandbox

### 9. A slim, fast-installing core with heavy dependencies as optional extras

**Summary.** Move heavyweight or narrow-purpose dependencies (`gitpython`, `line_profiler`, `memory_profiler`, and similar) out of the base install and behind `pip install sciris[all]` / feature-specific extras, with lazy imports that raise a clear "install sciris[profiling]" message on first use.

**Details.** Sciris currently declares ~20 hard dependencies, several of which (git bindings, C-extension profilers, `dill`, `zstandard`, `multiprocess`) are only needed for specific features. Agents constantly operate in fresh, ephemeral environments — CI runners, sandboxes, containers — where `pip install sciris` happens on every run. A heavy dependency tree makes that slower and more failure-prone (compiler-dependent wheels, transitive conflicts), which is precisely the environment where an agent is least able to debug an install failure. A lean core (`numpy`/`pandas`/`matplotlib` plus small pure-Python deps) with optional extras keeps the common path fast and reliable, while `sc.profile()` et al. raise an actionable ImportError pointing at the right extra.

**Value.** Faster, more reliable installs in exactly the throwaway environments agents live in, fewer "it won't even import" dead-ends, and a smaller attack/maintenance surface. Humans in constrained or offline environments benefit too. This is the most orthogonal-to-AI item here, but the fresh-sandbox reliability angle is genuinely AI-shaped.

**Effort.** Medium. Repartition `pyproject.toml` extras and add lazy-import guards at each feature boundary; Sciris's existing `SCIRIS_LAZY` machinery and delayed-import patterns give it a head start.

**Challenges.** A real backward-compatibility consideration: code (and agents) currently assume `import sciris as sc` brings *everything*. Mitigations: keep a `sciris[all]` that reproduces today's behavior, make the error messages unmistakable, and version the change clearly. This is the suggestion most likely to surprise existing users, so it needs a loud changelog and a deprecation period.

---

## What should *not* change (the soul of Sciris)

Being bold cuts both ways — some of what makes Sciris tempting to "clean up" is exactly what makes it valuable, for humans and machines alike:

- **Brevity and the simplifying interfaces.** `sc.save`, `sc.parallelize`, `sc.dateformatter`, `sc.odict` — collapsing boilerplate into one legible call is Sciris's whole reason to exist, and it happens to be deeply AI-friendly (fewer tokens, fewer bugs, more legible diffs). Do not sacrifice this for purity.
- **Flexible input types.** Accepting a scalar, list, or array; a string, number, or `date` object — this is real ergonomic value. The fix for machines is to *describe* the flexibility (stubs, docstrings, the index), not to remove it.
- **The clean, hint-free source.** The STYLE_GUIDE's rejection of inline type hints is defensible. Serve machines with stubs *beside* the source, not annotations *in* it.
- **The `die` philosophy.** Fine-grained, locally-scoped strictness is a good idea. Make its failures *loud* and globally *togglable* (#5); don't take the knob away.
- **Aliases and lowercase class names.** Convenient and memorable. Label a canonical form for machines (#7); keep the alternatives for humans.

The through-line: **don't remove Sciris's flexibility — make it observable.** Almost every suggestion here adds a machine-readable *description* of behavior (types, errors, examples, deprecations, an index) rather than constraining the behavior itself. That is how Sciris can become dramatically more AI-friendly while getting *more* human-friendly, not less.

---

## Prioritization (ROI matrix)

| # | Suggestion | Value | Effort | ROI | Alters identity? |
|---|-----------|-------|--------|-----|------------------|
| 1 | Machine-readable API index + `llms.txt` | High | Low–Med | **Highest** | No |
| 4 | Self-correcting errors + typed exceptions | High | Med | **Very high** | No |
| 3 | AI-native discovery (`sc.help`/`sc.api`) | Med–High | Low–Med | **Very high** | No |
| 5 | Loud soft-failures + global strict mode | High | Low–Med | **High** | Somewhat |
| 2 | Type stubs (`.pyi`) + `py.typed` | High | Med–High | **High** | Slightly (source untouched) |
| 6 | CI-verified doctest examples | Med–High | Med | **High** | No |
| 7 | Centralized deprecations + canonical names | Med | Low–Med | **Medium–High** | No |
| 8 | Single-source-of-truth generation | High (compounding) | Med–High | **High (long-run)** | Mindset shift |
| 9 | Slim core + optional heavy deps | Medium | Med | **Medium** | Yes (install UX) |

**If you do only three things:** #1 (the index — it unlocks #2, #3, #7, #8), #4 (self-correcting errors — the cheapest large improvement to the agent feedback loop), and #5 (never fail silently — kills the most expensive class of agent bug). Together they are low-to-medium effort and would meaningfully change how reliably an LLM can write correct Sciris code, while making the library strictly nicer for humans.
