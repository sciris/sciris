---
name: sciris-files
description: Use when saving or loading files with Sciris — sc.save, sc.load, sc.savejson, sc.loadjson, sc.saveyaml, sc.readyaml, sc.savearchive, sc.loadarchive, sc.savefig, sc.loadmetadata, sc.getfilelist, sc.thispath, sc.makefilepath, sc.rmpath, sc.metadata, sc.compareversions, sc.require, or version/reproducibility tracking.
---

# Sciris Files and Versioning

Reference for file I/O and version management. See full tutorial: `docs/tutorials/tut_files.ipynb`.

If you need more detail, use your MCP tools (Context7 or GitMCP) to look up current Sciris documentation, or consult the other Sciris skills.

## Save/Load Any Object

```python
sc.save('my-sim.obj', sim)           # Gzipped pickle (works with any Python object)
sim = sc.load('my-sim.obj')          # Load — methods and class still work
sc.zsave('fast.obj', sim)            # Zstandard compression (slightly faster)
# sc.load() auto-detects compression format
```

## JSON

```python
sc.savejson('data.json', obj)        # Saves JSONifiable parts of any object
data = sc.loadjson('data.json')      # Returns dict (not original object)
data = sc.readjson(json_string)      # Parse JSON from string
```

## YAML (JSON superset with comments)

```python
data = sc.readyaml(yaml_string)      # Parse YAML (supports comments)
sc.saveyaml('config.yaml', data)
```

## File Utilities

```python
sc.getfilelist('*.ipynb')            # List files matching pattern
sc.thispath()                        # Path of current file (use instead of pathlib.Path)
sc.makefilepath('data/out.csv', makedirs=True)  # Ensure path exists
sc.rmpath('file_or_folder')         # Remove file or folder (auto-detects)
```

## Versioning and Reproducibility

```python
md = sc.metadata()                   # Collect all environment metadata
sc.compareversions(np, '>1.0')       # Version comparison (returns True/False)
sc.require('numpy>1.20')            # Warn/raise if requirement not met
```

### Metadata-enhanced figures
```python
sc.savefig('fig.png', comments='My analysis')  # Saves with metadata (use instead of plt.savefig)
md = sc.loadmetadata('fig.png')                # Retrieve metadata later
```

### Metadata-enhanced archives
```python
sc.savearchive('sim.zip', sim, files='script.py', comments='Full run')
sim = sc.loadarchive('sim.zip')      # Restore object
md = sc.loadmetadata('sim.zip')      # Get metadata separately
```
