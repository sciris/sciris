# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Publication Information

**Citation**: Kerr CC, Sanz-Leon P, Abeysuriya RG, et al. Sciris: Simplifying scientific software in Python. *Journal of Open Source Software*. 2023;8(88):5076. doi:10.21105/joss.05076

**Key Applications**: Used by Covasim (COVID-19 modeling), Optima HIV, Optima Nutrition, Atomica, and other scientific software tools adopted in 30+ countries.

## Quick Reference

### Essential Commands
- **Run tests:** `cd tests && ./run_tests`
- **Install dev:** `pip install -e .`
- **Build docs:** `cd docs && ./build_docs`
- **Standard import:** `import sciris as sc`

### Core Functions
- **Containers:** `sc.odict()`, `sc.dcp()` (deep copy)
- **I/O:** `sc.save()`, `sc.load()`, `sc.savejson()`, `sc.loadjson()`
- **Arrays:** `sc.toarray()`, `sc.findinds()`, `sc.findnearest()`
- **Parallel:** `sc.parallelize()`, `sc.loadbalancer()`
- **Plotting:** `sc.boxoff()`, `sc.commaticks()`, `sc.dateformatter()`
- **Timing:** `sc.tic()`, `sc.toc()`, `sc.profile()`

### Key Parameters
- **`die`**: Exception handling (`die=True` raises, `die=False` warns)
- **`verbose`**: Output control (`verbose=True/False/None`)
- **`copy`**: Data copying behavior for containers

## Development Commands

### Testing
- **Run all tests:** `cd tests && ./run_tests` (uses pytest with parallel execution)
- **Run specific test:** `cd tests && pytest test_<module>.py -v`
- **Check test coverage:** `cd tests && ./check_coverage` (generates HTML report at tests/htmlcov/index.html)
- **Environment:** Tests run with `SCIRIS_BACKEND=agg` to prevent matplotlib windows

### Code Quality
- **Linting:** No formal linter configured (project follows internal style guide)
- **Type checking:** Not enforced (Sciris prioritizes flexibility over strict typing)
- **Pre-commit hooks:** Not configured
- **Style guide:** Follow existing code patterns and Python conventions

### Documentation
- **Build docs:** `cd docs && ./build_docs` (builds Sphinx documentation)
- **Build docs without notebooks:** `cd docs && ./build_docs never`
- **Debug mode (serial):** `cd docs && ./build_docs debug`
- **Output:** Documentation built to docs/_build/html/index.html

### Development Setup
- **Install package:** `pip install -e .` (editable install from pyproject.toml)
- **Test requirements:** `pip install -r tests/requirements.txt`
- **Python version:** Requires Python >=3.9

## Architecture Overview

Sciris is a scientific Python utilities library with a modular design organized around functional domains. Published in the Journal of Open Source Software (JOSS), Sciris aims to streamline the development of scientific software by making it easier to perform common tasks through "simplifying interfaces" that reduce boilerplate code and focus on scientific clarity.

## Design Philosophy

Sciris follows four core principles:

**Brevity through simple interfaces**: Packages common patterns requiring multiple lines of code into single, simple functions. Provides succinct plotting tasks, ensures consistent types and containers, and extends functionality of established objects like `OrderedDict` via `sc.odict`.

**Dejargonification**: Uses plain function names (e.g., `sc.smooth`, `sc.findnearest`, `sc.safedivide`) to make code scientifically clear and human-readable. Provides MATLAB-like functionality with familiar names (e.g., `sc.tic`/`sc.toc`, `sc.boxoff`).

**Fine-grained exception handling**: Uses the `die` keyword across classes and functions to enable locally scoped strictness levels. When `die=False`, Sciris handles exceptions softly with warnings; when `die=True`, it raises exceptions directly.

**Version management**: Provides methods to save and load metadata including Git information (`sc.savefig`, `sc.gitinfo`, `sc.loadmetadata`), compare module versions (`sc.compareversions`), and require specific versions (`sc.require`).

## Key Features and Examples

### Feature-rich Containers (`sc.odict`)
The `odict` class combines the best features of lists, dictionaries, and numeric arrays:
```python
data = sc.odict(a=[1,2,3], b=[4,5,6]) 
assert data['a'] == data[0]  # Integer indexing
assert data[:].sum() == 21   # Slicing and array operations
```

### Numerical Utilities
Type-agnostic array operations that handle mixed types:
```python 
sc.findinds([2,3,6,3], 3.0)  # Returns array([1,3])
sc.findnearest([1,2,3,4,5], 3.7)  # Returns 3 (index) and 4 (value)
sc.toarray([1,2,3], dtype=float)  # Flexible array conversion
```

### Simplified Parallelization
Easy parallel execution with automatic CPU detection:
```python
def f(x, y): return x*y
out = sc.parallelize(func=f, iterarg=[(1,2),(2,3),(3,4)])
# Alternative syntax:
out = sc.parallelize(func=f, iterkwargs={'x':[1,2,3], 'y':[2,3,4]})
```

### Enhanced Plotting
Shortcuts for common plot customizations:
```python
sc.options(font='Garamond')        # Set custom font
sc.dateformatter()                # Auto x-axis date formatting  
sc.commaticks()                    # Add commas to tick labels
sc.boxoff()                        # Remove top/right spines
sc.setylim()                       # Auto y-axis limits
```

### File I/O Simplification
Unified save/load with automatic compression:
```python
data = {'results': [1,2,3], 'params': {'n': 100}}
sc.save('myfile.pkl', data)        # Automatic compression
loaded = sc.load('myfile.pkl')     # Automatic decompression
sc.savejson('data.json', data)     # JSON with complex object support
```

### Real-world Workflow Examples

#### Scientific Data Processing Pipeline
```python
import sciris as sc
import numpy as np

# Load and process experimental data
sc.tic()
raw_data = sc.loadspreadsheet('experiment_data.xlsx')
processed = sc.odict()

for condition in raw_data.keys():
    # Clean and standardize data
    clean_data = sc.toarray(raw_data[condition], dtype=float)
    clean_data = clean_data[~np.isnan(clean_data)]  # Remove NaN
    
    # Statistical analysis with parallel processing
    results = sc.parallelize(
        func=statistical_analysis,
        iterkwargs={'data': [clean_data], 'method': ['bootstrap', 'permutation']}
    )
    processed[condition] = results

# Save intermediate results with metadata
sc.save('processed_data.pkl', processed, verbose=True)
sc.toc('Data processing pipeline')
```

#### Time Series Analysis
```python
# Generate date range and load time series data
dates = sc.daterange('2023-01-01', '2023-12-31', as_date=True)
timeseries = sc.odict()

for dataset in ['temperature', 'humidity', 'pressure']:
    # Load and smooth noisy sensor data
    raw = sc.load(f'{dataset}_raw.pkl')
    smoothed = sc.smooth(raw, window=7)  # 7-day moving average
    timeseries[dataset] = smoothed

# Create comprehensive plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
colors = sc.gridcolors(3, asarray=True)

for i, (key, data) in enumerate(timeseries.items()):
    plt.sca(axes[i])
    plt.plot(dates, data, color=colors[i], linewidth=2)
    plt.title(f'{key.title()} Over Time')
    sc.dateformatter()
    sc.boxoff()

sc.savefig('timeseries_analysis.png', metadata=True)  # Include Git info
```

### Type-agnostic Input Handling
Sciris functions accept multiple input types (lists, arrays, scalars) and handle conversions automatically, reducing the need for explicit type checking and conversion code.

### Core Module Structure
- **sc_utils.py** - Fundamental utilities, type checking, convenience functions
- **sc_odict.py** - Enhanced dictionary class (OrderedDict + list features)
- **sc_nested.py** - Nested dictionary and complex object operations
- **sc_math.py** - Mathematical operations and array manipulations
- **sc_dataframe.py** - DataFrame utilities and extensions
- **sc_datetime.py** - Date and time handling
- **sc_fileio.py** - File I/O, pickling, JSON, YAML, Excel operations
- **sc_plotting.py** - Matplotlib extensions and 3D plotting
- **sc_colors.py** - Color management and utilities
- **sc_printing.py** - Enhanced printing, formatting, progress display
- **sc_parallel.py** - Parallelization using multiprocess
- **sc_profiling.py** - Performance profiling and timing
- **sc_settings.py** - Global configuration management
- **sc_versioning.py** - Object versioning and compatibility

### Key Design Patterns

**Unified Namespace:**
- All functions accessible via `import sciris as sc`
- Flat namespace design (e.g., `sc.odict()`, `sc.save()`, `sc.parallel()`)
- Common import pattern: `import sciris as sc` used internally

**Core Data Structure:**
- `odict` class is central - combines dict, list, and array features
- Supports integer indexing, slicing, and key-based access
- Used throughout library as flexible container

**API Consistency:**
- Save/load function pairs (e.g., `save()`/`load()`, `savejson()`/`loadjson()`)
- Common parameters: `verbose`, `copy`, `die` (error handling)
- Type-agnostic inputs with automatic conversion

**Environment Awareness:**
- Platform detection utilities (`iswindows()`, `islinux()`, `ismac()`)
- Thread control via `SCIRIS_NUM_THREADS` environment variable
- Optional lazy loading with `SCIRIS_LAZY`

## Environment Variables

### Core Variables
- **`SCIRIS_NUM_THREADS`**: Controls maximum number of threads for parallel operations
  - Default: Auto-detected based on CPU count
  - Example: `export SCIRIS_NUM_THREADS=4`

- **`SCIRIS_BACKEND`**: Matplotlib backend selection
  - Default: System default
  - Example: `export SCIRIS_BACKEND=agg` (for headless environments)

- **`SCIRIS_LAZY`**: Enable lazy loading for optional dependencies
  - Default: False
  - Example: `export SCIRIS_LAZY=1`

### Usage in Code
```python
import os
os.environ['SCIRIS_NUM_THREADS'] = '8'  # Set before importing sciris
import sciris as sc
```

### File Organization Principles

**Main Package:** All core functionality in `sciris/` directory with `sc_` prefix
**Optional Components:** `sciris/_extras/` for optional dependencies
**No Circular Imports:** Strategic use of delayed imports and `import sciris as sc`

## Import Conventions

### Standard Import Pattern
```python
import sciris as sc  # Always use this form
```

### What NOT to do
```python
# Avoid these patterns:
from sciris import odict, save, load     # Breaks namespace consistency  
import sciris.sc_utils as utils          # Internal module imports
from sciris.sc_plotting import *         # Star imports
```

### Internal Development
When developing Sciris itself, modules use:
```python
import sciris as sc  # Even internal modules use this pattern
```

### Specific Module Access
For performance-critical code or when you need only specific functionality:
```python
import sciris as sc
# Then use: sc.odict(), sc.save(), etc.
# This is preferred over direct module imports
```

## Common Development Patterns

### Adding New Functions
1. Choose appropriate module based on functional domain
2. Follow naming conventions (snake_case, descriptive names)
3. Use consistent parameter patterns (`verbose=None`, error handling)
4. Add comprehensive docstrings with examples
5. Import in module's `__all__` list for namespace inclusion

### Testing New Features
1. Add test in corresponding `tests/test_<module>.py`
2. Test both success and failure cases
3. Use `sc_test_utils.py` helper functions where applicable
4. Run coverage check to ensure adequate testing

### Performance Considerations
- Library designed for scientific computing efficiency
- Use NumPy operations where possible
- Consider parallel execution for expensive operations
- Respect thread limits set by `SCIRIS_NUM_THREADS`

## File I/O Patterns

Sciris provides unified I/O through `sc_fileio.py`:
- `save()`/`load()` - Universal pickling with compression
- `savejson()`/`loadjson()` - JSON with jsonpickle for complex objects
- `saveobj()`/`loadobj()` - Alternative object serialization
- Excel support through `loadspreadsheet()` and `Blobject.export()`

All I/O functions handle both file paths and file objects, with automatic compression detection.

## Common Gotchas and Limitations

### When NOT to Use Sciris
- **Performance-critical inner loops**: `sc.odict` adds small overhead vs built-in `dict` (~500ms per million operations)
- **Strict typing requirements**: Sciris's flexibility can mask type errors in low-level libraries  
- **Large-scale distributed computing**: Use Dask, Ray, or Celery instead of `sc.parallelize` for multi-machine setups
- **Memory-constrained environments**: Sciris's convenience features use additional memory

### Common Pitfalls
- **`die` parameter confusion**: Remember `die=False` = soft warnings, `die=True` = strict exceptions
- **Import order**: Always use `import sciris as sc` rather than individual module imports
- **Thread limits**: Respect `SCIRIS_NUM_THREADS` environment variable in parallel code
- **Type assumptions**: Don't assume specific return types - Sciris prioritizes flexibility over strict typing

### Migration Strategies
When outgrowing Sciris:
- Replace individual functions incrementally rather than wholesale migration
- Use Sciris for prototyping, then optimize specific bottlenecks with lower-level alternatives
- Keep Sciris for I/O and utilities while switching compute-intensive parts to specialized libraries

## Debugging Tips

### Exception Handling with `die`
```python
# Strict mode - raises exceptions immediately
result = sc.load('file.pkl', die=True)  

# Forgiving mode - prints warning, returns None
result = sc.load('missing.pkl', die=False)
if result is None:
    print("File not found, using defaults")
```

### Verbose Output
```python
# Enable detailed output for debugging
sc.save('data.pkl', mydata, verbose=True)
# Output: Saving data.pkl... (6.2 MB, 0.12 s)

# System-wide verbosity
sc.options(verbose=True)  # All operations become verbose
```

### Performance Debugging
```python
# Profile specific functions
with sc.profile('data_processing'):
    processed = process_large_dataset(data)
# Output: data_processing: 2.34 s

# Timer for code blocks
sc.tic()
complex_calculation()
sc.toc('Complex calculation')
# Output: Complex calculation: 1.23 s
```

### Memory and Thread Debugging
```python
# Check thread usage
print(f"Using {sc.cpu_count()} CPUs")
print(f"Thread limit: {os.environ.get('SCIRIS_NUM_THREADS', 'auto')}")

# Monitor memory usage during operations
sc.checkmem()  # Shows current memory usage
```

### Common Debug Patterns
```python
# Type checking and conversion
data = sc.promotetoarray(data, verbose=True)  # Shows conversion details
sc.checktype(data, 'arraylike', die=False)    # Non-fatal type check

# Container inspection
od = sc.odict(a=1, b=2, c=3)
print(od.disp())  # Detailed container info
```