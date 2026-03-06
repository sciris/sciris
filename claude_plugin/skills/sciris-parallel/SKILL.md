---
name: sciris-parallel
description: Use when parallelizing code or profiling performance with Sciris — sc.parallelize, sc.Parallel, iterarg, iterkwargs, maxcpu, maxmem, async parallelization, sc.profile, sc.benchmark, sc.memload, sc.checkram, line profiling, or CPU/memory monitoring.
---

# Sciris Parallelization and Profiling

Reference for parallel execution and performance analysis. See full tutorial: `docs/tutorials/tut_parallel.ipynb`.

If you need more detail, use your MCP tools (Context7 or GitMCP) to look up current Sciris documentation, or consult the other Sciris skills.

## Basic Parallelization

```python
# Simple: iterate over one argument
results = sc.parallelize(func, [1, 2, 3])

# With keyword arguments
results = sc.parallelize(complex_func, [1, 2, 3], arg2=10)

# Multiple iteration arguments (dict of lists)
results = sc.parallelize(mult, iterkwargs={'x': [1,2,3], 'y': [2,3,4]})

# Multiple iteration arguments (list of tuples)
results = sc.parallelize(mult, iterarg=[(1,2), (2,3), (3,4)])

# Multiple iteration arguments (list of dicts)
results = sc.parallelize(mult, iterkwargs=[{'x':1, 'y':2}, {'x':2, 'y':3}])
```

## Resource-Limited Parallelization

```python
results = sc.parallelize(
    func     = my_func,
    iterarg  = range(200),
    maxcpu   = 0.8,       # Don't exceed 80% CPU
    maxmem   = 0.9,       # Don't exceed 90% memory
    interval = 0.2,       # Re-check interval (seconds)
)
```

## Advanced: Parallel Class

```python
P = sc.Parallel(
    func = slow_func,
    iterarg = range(10),
    parallelizer = 'multiprocess-async',  # Async execution
    die = False,                          # Continue on failure
)
P.run_async()
P.monitor()                               # Watch progress
P.finalize()                              # Collect results

print(P.success)       # Per-job success flags
print(P.exceptions)    # Any exceptions that occurred
print(P.results)       # Results (None for failed jobs)
print(P.times)         # Per-job timing
```

## Profiling

### Benchmarking
```python
bm = sc.benchmark()    # CPU performance in MOPS
ml = sc.memload()      # System memory load
ram = sc.checkram()    # Python process RAM usage
```

### Line Profiling
```python
sc.profile(my_func)    # Shows time per line (like line_profiler)
```
