"""
Generate a machine-readable index of the Sciris API.

The whole public Sciris API fits comfortably in a single file, which makes it
possible for a tool (an IDE, a documentation search, or an LLM agent) to load
the entire map of the library in one shot. This script introspects `sciris`
and projects it into three artifacts, all generated from the same data and all
published with the docs (e.g. <https://docs.sciris.org/llms.txt>):

- `api.json`: the canonical index, as structured data
- `llms.txt`: a compact Markdown index (name, signature, summary, aliases)
- `llms-full.txt`: as above, plus the canonical example for each entry

This is a docs build tool: nothing here is part of the Sciris package itself.

To regenerate the artifacts::

    cd docs && python make_api.py

To check that they are up to date (used by the test suite, hence CI)::

    cd docs && python make_api.py --check
"""

import re
import sys
import inspect
import sciris as sc

__all__ = ['make_index', 'load', 'write', 'check']

thisdir = sc.thispath(__file__)
jsonfile = thisdir / 'api.json'
llmsfile = thisdir / 'llms.txt'
llmsfullfile = thisdir / 'llms-full.txt'

max_summary = 300 # Maximum number of characters in a summary
max_example = 12 # Maximum number of lines in an example

description = 'Fast, flexible tools to simplify scientific Python' # Matches pyproject.toml
notes = [
    'Sciris is a library of utilities for scientific computing: containers, file I/O, dates, arrays, plotting, parallelization, and profiling.',
    'Everything listed here is available from the top level: `import sciris as sc`, then e.g. `sc.findnearest()`. Do not import submodules directly.',
    'Signatures are as introspected from the current version; summaries are the first paragraph of each docstring.',
    'Aliases are alternative names for the same object; the canonical name is the one listed, and is the one to prefer when writing new code.',
]
links = [ # Extra pointers, following the llms.txt convention
    ('Documentation', 'https://docs.sciris.org', 'tutorials, API reference, and the style guide'),
    ('Source', 'https://github.com/sciris/sciris', 'the Sciris repository'),
    ('Paper', 'https://doi.org/10.21105/joss.05076', 'Sciris: Simplifying scientific software in Python (JOSS 2023)'),
]

# The order the modules are listed in, matching the docs; anything else is appended alphabetically
module_order = ['sc_math', 'sc_asd', 'sc_odict', 'sc_dataframe', 'sc_fileio', 'sc_versioning',
                'sc_printing', 'sc_plotting', 'sc_colors', 'sc_parallel', 'sc_profiling',
                'sc_utils', 'sc_datetime', 'sc_nested', 'sc_settings']

module_titles = sc.objdict(
    sc_math       = 'Math and arrays',
    sc_asd        = 'Optimization',
    sc_odict      = 'Dictionaries',
    sc_dataframe  = 'Dataframes',
    sc_fileio     = 'File I/O',
    sc_versioning = 'Versioning and metadata',
    sc_printing   = 'Printing and formatting',
    sc_plotting   = 'Plotting',
    sc_colors     = 'Colors and colormaps',
    sc_parallel   = 'Parallelization',
    sc_profiling  = 'Profiling and resource monitoring',
    sc_utils      = 'Utilities',
    sc_datetime   = 'Dates and times',
    sc_nested     = 'Nested objects',
    sc_settings   = 'Settings and help',
    other         = 'Other',
)


def _getdoc(obj):
    """ Get an object's own docstring (not an inherited one, which may be from e.g. pandas) """
    doc = getattr(obj, '__doc__', None)
    if not isinstance(doc, str) and inspect.isclass(obj): # Some classes document themselves in __init__
        doc = getattr(getattr(obj, '__init__', None), '__doc__', None)
    return doc if isinstance(doc, str) else ''


def _getsummary(doc):
    """ Extract the first paragraph of a docstring, collapsed onto a single line """
    if not doc:
        return ''
    paragraphs = re.split(r'\n\s*\n', doc.strip())
    summary = ' '.join(paragraphs[0].split())
    if len(summary) > max_summary:
        summary = summary[:max_summary].rsplit(' ', 1)[0] + ' […]'
    return summary


def _getsignature(obj):
    """ Get the call signature of a function or class, falling back gracefully """
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError): # pragma: no cover # e.g. some C-implemented callables
        return '(...)'


def _getexample(doc):
    """ Extract the canonical example: the first code block, preferring one under an "Example" heading """
    if not doc:
        return ''
    blocks = [(m.start(), m.group(1)) for m in re.finditer(r'```(?:python)?\n(.*?)```', doc, flags=re.DOTALL)]
    if not blocks:
        return ''
    heading = doc.find('**Example')
    example = None
    if heading >= 0:
        for start, block in blocks:
            if start > heading:
                example = block
                break
    if example is None:
        example = blocks[0][1]
    lines = [line for line in example.strip('\n').rstrip().splitlines()]
    if len(lines) > max_example:
        lines = lines[:max_example] + ['# [...]']
    return '\n'.join(lines)


_deprecated = re.compile(r'(?i)^(?:\*{0,2}note:?\*{0,2}\s*)?(?:this (?:function|class|method) is deprecated|deprecated\b)')

def _isdeprecated(doc):
    """ Whether the docstring marks the object as deprecated (in the summary, or in a paragraph of its own) """
    if not doc:
        return False
    if 'deprecat' in _getsummary(doc).lower():
        return True
    paragraphs = re.split(r'\n\s*\n', doc.strip())
    return any(_deprecated.match(' '.join(paragraph.split())) for paragraph in paragraphs)


def _modulekey(module):
    """ Sort key for grouping entries by module, following the docs order """
    short = module.rsplit('.', 1)[-1] if module else 'other'
    if short not in module_order and short not in module_titles:
        short = 'other'
    return short


def make_index():
    """
    Introspect Sciris and return the API index as a dictionary.

    The index has the keys `version`, `description`, `notes`, `n_entries`, `aliases`
    (a mapping of alias name to canonical name), and `entries` (a list of records,
    each with `name`, `kind`, `module`, `signature`, `summary`, `example`, `aliases`,
    and `deprecated`).

    **Example**:

    ```python
    import make_api
    index = make_api.make_index()
    print(index['entries'][0])
    ```
    """
    names = [name for name in dir(sc) if not name.startswith('_')]

    # Group names by object, so aliases (e.g. sc.save/sc.saveobj) collapse into one entry
    groups = {} # Map id(obj) to the list of names pointing at it
    objs = {}
    for name in names:
        obj = getattr(sc, name)
        if inspect.ismodule(obj) or not callable(obj): # Skip submodules and data (e.g. sc.style_fancy)
            continue
        key = id(obj)
        groups.setdefault(key, []).append(name)
        objs[key] = obj

    entries = []
    aliasmap = {}
    for key, group in groups.items():
        obj = objs[key]
        realname = getattr(obj, '__name__', None)
        canonical = realname if realname in group else sorted(group, key=len)[0] # Prefer the object's own name
        aliases = sorted(name for name in group if name != canonical)
        for alias in aliases:
            aliasmap[alias] = canonical
        doc = _getdoc(obj)
        entries.append(dict(
            name       = canonical,
            kind       = 'class' if inspect.isclass(obj) else 'function',
            module     = _modulekey(getattr(obj, '__module__', '')),
            signature  = _getsignature(obj),
            summary    = _getsummary(doc),
            example    = _getexample(doc),
            aliases    = aliases,
            deprecated = _isdeprecated(doc),
        ))

    entries = sorted(entries, key=lambda entry: entry['name'].lower())
    index = dict(
        version     = sc.__version__,
        description = description,
        notes       = notes,
        links       = [dict(title=title, url=url, description=desc) for title,url,desc in links],
        n_entries   = len(entries),
        aliases     = dict(sorted(aliasmap.items())),
        entries     = entries,
    )
    return index


def _sortmodules(index):
    """ Return the module keys present in the index, in docs order """
    present = {entry['module'] for entry in index['entries']}
    modules = [mod for mod in module_order if mod in present]
    modules += sorted(present - set(modules))
    return modules


def make_llms_txt(index=None, examples=False):
    """
    Render the API index as an llms.txt-style Markdown document.

    Args:
        index (dict): the index from `make_index()` (default: generate it)
        examples (bool): whether to include the canonical example for each entry (i.e. llms-full.txt)

    Returns:
        The document, as a string.
    """
    index = sc.ifelse(index, make_index())
    filename = 'llms-full.txt' if examples else 'llms.txt'

    lines = [f'# Sciris v{index["version"]}', '', f'> {index["description"]}', '']
    for note in index['notes']:
        lines += [f'- {note}']
    if not examples:
        lines += ['- A version of this file including a usage example for each function is available at llms-full.txt.']
    lines += ['', f'This file lists all {index["n_entries"]} public Sciris functions and classes. It is generated from the source '
              f'by `python make_api.py`; do not edit it by hand.', '']

    lines += ['## Links', '']
    for link in index['links']:
        lines += [f'- [{link["title"]}]({link["url"]}): {link["description"]}']
    lines += ['']

    bymodule = {}
    for entry in index['entries']:
        bymodule.setdefault(entry['module'], []).append(entry)

    for module in _sortmodules(index):
        title = module_titles.get(module, module)
        header = f'{title} ({module})' if module.startswith('sc_') else title
        lines += [f'## {header}', '']
        for entry in bymodule[module]:
            extras = []
            if entry['aliases']:
                extras.append('aliases: ' + ', '.join(f'sc.{alias}()' for alias in entry['aliases']))
            if entry['deprecated']:
                extras.append('DEPRECATED')
            suffix = f' [{"; ".join(extras)}]' if extras else ''
            summary = entry['summary'] or '(no description available)'
            lines += [f'- `sc.{entry["name"]}{entry["signature"]}`: {summary}{suffix}']
            if examples and entry['example']:
                lines += ['', '  ```python']
                lines += [f'  {line}'.rstrip() for line in entry['example'].splitlines()]
                lines += ['  ```', '']
        lines += ['']

    lines += [f'<!-- Generated by `python make_api.py` for Sciris v{index["version"]}: {filename} -->', '']
    return '\n'.join(lines)


def load():
    """
    Load the generated API index (`docs/api.json`).

    **Example**:

    ```python
    import make_api
    index = make_api.load()
    print(index['n_entries'])
    ```
    """
    return sc.loadjson(jsonfile)


def write(verbose=True):
    """
    Regenerate all the API artifacts: `api.json`, `llms.txt`, and `llms-full.txt`.

    Args:
        verbose (bool): whether to print progress

    Returns:
        The list of files written.
    """
    index = make_index()
    sc.savejson(jsonfile, index)
    written = [jsonfile]

    for path,examples in [(llmsfile, False), (llmsfullfile, True)]:
        sc.savetext(path, make_llms_txt(index, examples=examples))
        written.append(path)

    if verbose:
        for path in written:
            print(f'  Wrote {path.name} ({path.stat().st_size/1e3:0.1f} kB)')
        print(f'Indexed {index["n_entries"]} Sciris functions and classes for v{index["version"]}')
    return written


def check(verbose=True):
    """
    Check whether the generated artifacts match the current API.

    Args:
        verbose (bool): whether to print which files are out of date

    Returns:
        The list of files that are out of date (empty if everything matches).
    """
    index = make_index()
    stale = []

    def compare(path, expected):
        """ Compare a file to what it should contain, treating a missing file as stale """
        if not path.exists():
            return 'missing'
        actual = sc.loadjson(path) if path.suffix == '.json' else sc.loadtext(path)
        return None if actual == expected else 'out of date'

    for path,expected in [(jsonfile, index),
                          (llmsfile, make_llms_txt(index, examples=False)),
                          (llmsfullfile, make_llms_txt(index, examples=True))]:
        reason = compare(path, expected)
        if reason:
            stale.append(path)
            if verbose:
                print(f'  {path.name} is {reason}')
    if verbose and not stale:
        print(f'All API artifacts are up to date ({index["n_entries"]} entries, v{index["version"]})')
    return stale


if __name__ == '__main__':
    if '--check' in sys.argv:
        stale = check()
        if stale:
            print('Run "python make_api.py" to regenerate.')
            sys.exit(1)
    else:
        write()
