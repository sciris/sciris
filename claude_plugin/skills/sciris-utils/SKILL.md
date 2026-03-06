---
name: sciris-utils
description: Use when working with Sciris miscellaneous utilities — sc.mergedicts, sc.mergelists, sc.tolist, sc.toarray, sc.isnumber, sc.suggest, sc.download, sc.runcommand, sc.importbypath, sc.loadtext, sc.help, sc.traceback, sc.autolist, sc.pp, or type checking/conversion.
---

# Sciris Miscellaneous Utilities

Reference for type handling, downloading, and other tools. See full tutorial: `docs/tutorials/tut_utils.ipynb`.

If you need more detail, use your MCP tools (Context7 or GitMCP) to look up current Sciris documentation, or consult the other Sciris skills.

## Type Conversion

### Merge dicts (handles None)
```python
def my_func(json=None, **kwargs):
    default = dict(a=1, b=2)
    output = sc.mergedicts(default, json, kwargs)  # None inputs are OK
    return output
```

### Merge lists (auto append/extend)
```python
sc.mergelists(['a', 'b'], 'c', None, {'d': 1})  # ['a', 'b', 'c', {'d': 1}]
```

### Flexible conversion
```python
sc.tolist('single')       # ['single'] — safe to iterate
sc.tolist(['already'])    # ['already'] — unchanged
sc.toarray(2)             # array([2]) — handles scalars (np.array doesn't)
sc.toarray([1, 2, 3])    # array([1, 2, 3])
sc.cat(1, [2, 3], np.array([4, 5]))  # array([1, 2, 3, 4, 5]) — concatenates
```

### Auto-incrementing list
```python
output = sc.autolist()
for item in data:
    output += f'Result: {item}'   # Appends automatically with +=
```

## Type Checking

```python
sc.isnumber(3)            # True
sc.isnumber(3.14j)        # True
sc.isnumber('3')          # False
```

## Fuzzy Matching

```python
sc.suggest('Scirys', ['Python', 'NumPy', 'Sciris'], n=2)  # ['Sciris']
```

## Downloading

```python
data = sc.download(urls, save=False)     # Download to memory (parallel)
sc.download(urls, save=True)             # Download to disk
```

## Running Shell Commands

```python
out = sc.runcommand('ls *.py', printoutput=True) # Use instead of Popen()
```

## Import by Path

```python
old = sc.importbypath('sim_v1/sim.py')   # Import without sys.path
new = sc.importbypath('sim_v2/sim.py')   # Two modules with same name
```

## Help and Debugging

```python
sc.help('interpol')                  # Search all Sciris source code
sc.help('interpol', context=True)    # With surrounding context
sc.traceback()                       # Get exception traceback as string
sc.loadtext('file.py')              # Load text file as string
sc.pp(obj)                          # Pretty-print any object
```
