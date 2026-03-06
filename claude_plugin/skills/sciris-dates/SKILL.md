---
name: sciris-dates
description: Use when working with dates, times, or timing in Sciris — sc.timer, sc.tic, sc.toc, sc.date, sc.daterange, sc.datedelta, sc.now, sc.getdate, sc.time, sc.timedsleep, sc.randsleep, date format conversion, or timing code blocks.
---

# Sciris Dates and Times

Reference for timing and date utilities. See full tutorial: `docs/tutorials/tut_dates.ipynb`.

If you need more detail, use your MCP tools (Context7 or GitMCP) to look up current Sciris documentation, or consult the other Sciris skills.

## Timing Code

### Timer object
```python
T = sc.timer()

zeros = np.zeros((n, n))
T.tt('Make zeros')          # toc-then-tic (prints elapsed, resets)

rand = np.random.rand(n, n)
T.tt('Make random')
```

### Context manager
```python
with sc.timer('My operation'):
    expensive_work()
```

### Timer statistics
```python
T = sc.timer()
for i in range(10):
    work(i)
    T.tt(f'Iteration {i}')

print(T.mean(), T.std(), T.min(), T.max())
T.plot()                    # Plot timing results
```

## Sleep Utilities

```python
sc.timedsleep(0.3)                  # Like time.sleep()

# In a loop: accounts for computation time so each iteration = exact duration
for i in range(5):
    sc.timedsleep('start')
    do_work()
    sc.timedsleep(0.3, verbose=True)  # Total iteration = 0.3s

sc.randsleep(0.2)                    # Sleep random 0-0.4s (mean 0.2)
```

## Date Utilities

### Current time
```python
sc.time()                # Unix timestamp (time.time())
sc.now()                 # datetime.datetime.now()
sc.getdate()             # Formatted string: '2024-Mar-15 14:30:00'
```

### Date conversion
```python
sc.date('2022-03-04')                        # str → datetime
sc.date('04-03-2022', format='mdy')          # Explicit format
sc.date('04-03-2022', format='dmy')          # Day-month-year
```

### Date ranges and math
```python
dates = sc.daterange('2022-02-02', '2022-03-04')          # List of date strings
dates = sc.daterange('2022-01-01', '2022-12-31', as_date=True)  # As datetime objects
newdates = sc.datedelta(dates, months=10)                  # Add 10 months
```
