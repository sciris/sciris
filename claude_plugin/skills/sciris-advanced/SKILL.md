---
name: sciris-advanced
description: Use when working with advanced Sciris features — nested dictionaries (sc.makenested, sc.getnested, sc.setnested, sc.iternested, sc.search, sc.iterobj), context blocks (sc.capture, sc.tryexcept), sc.smoothinterp interpolation, sc.asd optimization, sc.animation, sc.savemovie, or sc.printjson.
---

# Sciris Advanced Features

Reference for nested dicts, context blocks, interpolation, optimization, and animation. See full tutorial: `docs/tutorials/tut_advanced.ipynb`.

If you need more detail, use your MCP tools (Context7 or GitMCP) to look up current Sciris documentation, or consult the other Sciris skills.

## Nested Dictionaries

```python
nest = {}
sc.makenested(nest, ['key1', 'key1.1'])     # Create nested structure
sc.makenested(nest, ['key2', 'key2.1', 'key2.1.1'])

# Iterate over all "twigs" (leaf paths)
for twig in sc.iternested(nest):
    sc.setnested(nest, twig, value)          # Set value at path
    val = sc.getnested(nest, twig)           # Get value at path

# Search by key or value
sc.search(nest, 'key2.1.1')                 # Find by key pattern
sc.search(nest, value=5)                    # Find by value

# Transform all values
sc.iterobj(nest, transform_func, inplace=True)

sc.printjson(nest)                           # Pretty-print as JSON
```

## Context Blocks

### Capture output
```python
with sc.capture() as text:
    verbose_function()                       # All print output → text variable
lines = text.splitlines()
```

### Exception handling
```python
# Simple: exit gracefully at first exception
with sc.tryexcept():
    risky_function()

# Advanced: accumulate exception history
tryexc = None
for i in range(1000):
    with sc.tryexcept(history=tryexc, verbose=False) as tryexc:
        fickle_function()
tryexc.disp()                                # Show all exceptions
```

## Interpolation

```python
# Smoother than np.interp, more conservative than scipy cubic spline
y_new = sc.smoothinterp(newx, origx, origy, smoothness=5)
```

## Optimization (Adaptive Stochastic Descent)

```python
result = sc.asd(objective_func, x0, verbose=False)
# Often faster and more accurate than scipy.optimize.minimize for noisy problems
```

## Animation

```python
frames = [plt.plot(np.cumsum(np.random.randn(100))) for i in range(20)]
sc.savemovie(frames, 'animation.gif')
```
