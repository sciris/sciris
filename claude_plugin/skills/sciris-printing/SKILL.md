---
name: sciris-printing
description: Use when printing or formatting output with Sciris — sc.heading, sc.printgreen, sc.printblue, colored output, sc.strjoin, sc.newlinejoin, sc.pr, sc.prettyobj, sc.indent, sc.progressbar, sc.printmedian, or monitoring loop progress.
---

# Sciris Printing Tools

Reference for output formatting and progress display. See full tutorial: `docs/tutorials/tut_printing.ipynb`.

## Headings and Colors

```python
sc.heading('Section Title')           # Bold section heading with divider
sc.printgreen('Success message')      # Green text
sc.printblue('Info message')          # Blue text
sc.indent(text)                       # Indent text block
```

## String Joining

```python
sc.strjoin(['a', 'b', 'c'])          # 'a, b, c' (shortcut to ', '.join())
sc.newlinejoin(['a', 'b', 'c'])      # 'a\nb\nc' (auto str conversion)
# Especially useful in f-strings and error messages
```

## Object Inspection

```python
sc.pr(obj)                            # Pretty representation (attributes, types, sizes)
# Much more informative than dir(obj)
```

### prettyobj base class
```python
class MySim(sc.prettyobj):            # Inherit for nice print(sim) output
    def __init__(self):
        self.n = 10
        self.ready = False

sim = MySim()
print(sim)                            # Shows all attributes with types and values
```

## Progress Monitoring

```python
for i in sc.progressbar(range(100)):  # tqdm-based progress bar
    do_work(i)
```

## Numeric Formatting

```python
sc.printmedian(data)                  # Print median with IQR
sc.sigfig(123456, 3, sep=True)       # '123,000' (significant figures with separator)
```
