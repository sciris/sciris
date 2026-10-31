---
name: sciris-dicts
description: Use when working with Sciris dictionaries or dataframes — sc.obj, sc.objdict, sc.odict, sc.dataframe, integer indexing of dicts, enumitems, object-syntax access, dataframe creation with dtypes, appendrow, or sc.dataframe.cat.
---

# Sciris Dictionaries and Dataframes

Reference for Sciris container types. See full tutorial: `docs/tutorials/tut_dicts.qmd`.

If you need more detail, use your MCP tools (Context7 or GitMCP) to look up current Sciris documentation, or consult the other Sciris skills.

## sc.obj — Dict and Object in One

A real `dict` that is as fast as a plain object. Use it by default.

```python
ob = sc.obj(key1=[1, 2], key2=[3, 4])

ob.key1                  # Object syntax (no quotes!)
ob['key1']               # Dict syntax
ob[0]                    # Index syntax
# All three are equivalent

ob.keys()[0]             # Keys returns a list (not dict_keys)

for i, k, v in ob.enumitems():   # Enumerate with key and value
    print(f'Item {i}: {k} = {v}')

print(f'{ob.key1 = }')   # Especially handy in f-strings: no nested quote issues
```

**When NOT to use sc.obj:** When your dict has integer keys (ambiguous with index access).

**Methods are protected:** `ob = sc.obj(clear=0.1)` keeps `ob.clear()` working; get the key with `ob['clear']`. Setting `ob.clear = 0.1` raises an error.

**Subclassing:** Methods work as usual. Attributes that aren't keys must be declared in `__slots__`; set them with `setattribute()`, and give them starting values in `__new__`. Overriding `__setitem__()` also applies to `ob.x = 3`, but makes that subclass slower.

## Other dicts

- `sc.objdict`: a slower `sc.obj` that also supports setting items by position (`ob[0] = 3`), lists of keys (`ob[['a','b']]`), `defaultdict`, and attributes that aren't keys (`setattribute()`) without `__slots__`.
- `sc.odict`: like `sc.objdict`, but with dict syntax only.
- `sc.dictobj`: a simpler dict with object syntax, without indexing or the `sc.odict` methods; keys can replace methods.

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
