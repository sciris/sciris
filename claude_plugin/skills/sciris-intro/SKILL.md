---
name: sciris-intro
description: Use when the user needs to implement a basic Sciris features — finding array values, plotting with date formatting, objdict containers, saving/loading objects, or parallelization.
---

# Sciris introduction

Quick reference for Sciris' most commonly used features. See the full tutorial: `docs/tutorials/tut_intro.ipynb`.

If you need more detail, use your MCP tools (Context7 or GitMCP) to look up current Sciris documentation, or consult the other Sciris skills.

## Core Patterns

### Array operations
Use the `sciris-arrays` skill for more details.
```python
import numpy as np
import sciris as sc

data = np.random.rand(50)
inds = sc.findinds(data > 0.9)          # Find indices matching condition
mean_str = sc.arraymean(data)            # Mean ± std as string
joined = sc.strjoin(inds)               # Join values into comma-separated string
```

### Quick plotting with dates
Use the `sciris-dates` skill for more details.
```python
dates = sc.daterange('2022-01-01', '2022-02-28', as_date=True)
values = 1e6 * np.random.randn(59)**2

data = sc.dataframe(x=dates, y=values)  # Shortcut to pd.DataFrame
plt.scatter(data.x, data.y)
sc.dateformatter()                       # Format date axis
sc.SIticks()                             # SI notation on y-axis
```

### Flexible containers
Use the `sciris-dicts` skill for more details.
```python
data = sc.objdict(a=[1,2,3], b=[4,5,6])
assert data.a == data['a'] == data[0]    # Three ways to access
assert data[:].sum() == 21              # Slice and sum
for i, key, value in data.enumitems():   # Enumerate with keys
    print(f'Item {i}: {key} = {value}')
```

### Save/load any object
Use the `sciris-files` skill for more details.
```python
sc.save('my-sim.obj', sim)               # Save (gzipped pickle)
new_sim = sc.load('my-sim.obj')          # Load — methods still work!
new_sim.plot()
```

### Parallelization
Use the `sciris-parallel` skill for more details.
```python
results = sc.parallelize(func, iterkwargs=dict(scale=[40,30,20,10]), x_offset=5, y_offset=10)
```

### Plot configuration
Use the `sciris-plotting` skill for more details.
```python
sc.options(dpi=120, jupyter=True)        # Set DPI and backend
sc.boxoff()                              # Remove top/right spines
```
