---
name: sciris-arrays
description: Use when working with NumPy arrays in Sciris — finding indices, nearest values, concatenation, handling NaN/missing values, smoothing data, 2D Gaussian smoothing, or linear regression with sc.findinds, sc.findnearest, sc.cat, sc.rmnans, sc.fillnans, sc.smooth, sc.rolling, sc.linregress.
---

# Sciris Math and Array Tools

Reference for array utilities that simplify NumPy workflows. See full tutorial: `docs/tutorials/tut_arrays.ipynb`.

If you need more detail, use your MCP tools (Context7 or GitMCP) to look up current Sciris documentation, or consult the other Sciris skills.

## Array Indexing

```python
import numpy as np
import sciris as sc

data = np.random.rand(100)

# Find indices where condition is true
inds = sc.findinds(data > 0.9)           # vs (data>0.9).nonzero()[0]

# Find nearest value
ind = sc.findnearest(data, 0.5)          # vs np.argmin(abs(data - 0.5))

# Works on lists too (not just arrays)
ind = sc.findnearest([81, 78, 66, 25], 50)
```

## Creating/Concatenating Arrays

```python
sc.cat(1, 2, 3)                          # array([1, 2, 3])

# Add row to matrix — vs np.concatenate([data, np.atleast_2d(np.array([1,2]))])
data = sc.cat(data, [1, 2])
```

## Missing Values

```python
d0 = [1, 2, np.nan, 4, np.nan, 6, np.nan, np.nan, np.nan, 10]

d1 = sc.rmnans(d0)                       # Remove NaNs
d2 = sc.fillnans(d0, 0)                  # Replace with zeros
d3 = sc.fillnans(d0, 'linear')           # Linear interpolation
```

## Smoothing

```python
# 1D smoothing
smooth = sc.smooth(data, 7)              # Gaussian-like smoothing
roll = sc.rolling(data, 7)               # Rolling average

# 2D Gaussian smoothing
smooth2d = sc.gauss2d(raw_2d, scale=2)
```

## Other Math

```python
# Random rounding: 0.7 rounds up 70% of the time
rounded = sc.randround(data)

# Linear regression
m, b = sc.linregress(x, y)

# 3D plotting helpers
ax = sc.ax3d(121)                        # Create 3D subplot
sc.bar3d(data, ax=ax)                    # 3D bar plot
```
