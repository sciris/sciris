---
name: sciris-dicts
description: Use when working with Sciris dictionaries or dataframes — sc.odict, sc.objdict, sc.dataframe, integer indexing of dicts, enumitems, object-syntax access, dataframe creation with dtypes, appendrow, or sc.dataframe.cat.
---

# Sciris Dictionaries and Dataframes

Reference for Sciris container types. See full tutorial: `docs/tutorials/tut_dicts.ipynb`.

If you need more detail, use your MCP tools (Context7 or GitMCP) to look up current Sciris documentation, or consult the other Sciris skills.

## odict — Ordered Dict with Index Access

```python
od = sc.odict(a=['some', 'strings'], b=[1, 2, 3])

od['a']                  # Key access (like dict)
od[0]                    # Integer index access
od.keys()[0]             # Keys returns a list (not dict_keys)

for i, k, v in od.enumitems():   # Enumerate with key and value
    print(f'Item {i}: {k} = {v}')
```

**When NOT to use odict:** When your dict has integer keys (ambiguous with index access).

**odict vs objdict:** odict is slightly faster (nanoseconds per op). Use odict for millions of operations; objdict for everything else.

## objdict — odict + Object Syntax

```python
ob = sc.objdict(key1=[1, 2], key2=[3, 4])

ob.key1                  # Object syntax (no quotes!)
ob['key1']               # Dict syntax
ob[0]                    # Index syntax
# All three are equivalent

# Especially handy in f-strings:
print(f'{ob.key1 = }')  # No nested quote issues
```

## dataframe — Enhanced pd.DataFrame

### Creation shortcuts
```python
# Simpler than pd.DataFrame(dict(x=x, y=y, z=z))
df = sc.dataframe(x=x, y=y, z=z)

# With dtypes
df = sc.dataframe(x=x, y=y, z=z, dtypes=[str, float, bool])

# Columns with types
df = sc.dataframe(columns=dict(x=str, y=float, z=bool), data=data)
```

### Display
```python
df.disp()                               # Show full dataframe (no truncation)
df.disp(precision=1, ncols=5, nrows=10) # Customized display
```

### Indexing
```python
df['values', 1]          # Column + row access
df[1]                    # Auto iloc fallback (KeyError in pandas)
```

### In-place manipulation
```python
df.appendrow(['d', 4, 0])               # Append row in-place

# Concatenate anything
df = sc.dataframe.cat(
    sc.dataframe(x=['a'], y=[1]),        # Dataframe
    dict(x=['b'], y=[2]),                # Dict
    [['c', 3]],                          # Raw data
)
```
