---
name: sciris-plotting
description: Use when plotting with Sciris or Matplotlib — sc.options, sc.dateformatter, sc.commaticks, sc.SIticks, sc.boxoff, sc.setylim, sc.figlayout, sc.getrowscols, sc.vectocolor, sc.gridcolors, sc.scatter3d, sc.savefig, plot styles (sciris.simple, sciris.fancy), colormaps (parula, orangeblue), or 3D plotting.
---

# Sciris Plotting and Colors

Reference for Matplotlib extensions and color utilities. See full tutorial: `docs/tutorials/tut_plotting.ipynb`.

If you need more detail, use your MCP tools (Context7 or GitMCP) to look up current Sciris documentation, or consult the other Sciris skills.

## Setup and Options

```python
sc.options(jupyter=True)              # High-res retina backend for Jupyter
sc.options(dpi=120)                   # Set figure DPI
sc.options(font='serif')              # Change font globally
sc.options(font='default')            # Reset font
```

## Plot Styles

```python
with plt.style.context('sciris.simple'):   # Clean, minimal style
    make_plot()
with plt.style.context('sciris.fancy'):    # Seaborn-like style
    make_plot()
```

## Axis Formatting

```python
sc.dateformatter()                    # Auto date formatting on x-axis
sc.commaticks()                       # Comma-separated tick labels (1,000,000)
sc.SIticks()                          # SI notation (1M, 2.5k)
sc.setylim()                          # Auto y-limits (starts at 0)
sc.boxoff()                           # Remove top/right spines
sc.figlayout()                        # Tight layout (remove whitespace)
```

## Layout Helpers

```python
rows, cols = sc.getrowscols(14)       # Auto grid for N subplots
```

## Colors

### Continuous (for ordered data)
```python
colors = sc.vectocolor(n, cmap='turbo')       # Map values to colormap
colors = sc.vectocolor(values, cmap='parula')  # Custom Sciris colormaps: parula, orangeblue
c = sc.arraytocolor(data_2d)                   # 2D version
```

### Categorical (for distinct groups)
```python
colors = sc.gridcolors(n)             # n<=9: ColorBrewer, 10-19: Kelly's, 20+: uniform RGB
colors = sc.gridcolors(n, asarray=True)  # As numpy array
```

## 3D Plotting

```python
ax = sc.ax3d(121)                     # Create 3D subplot
sc.scatter3d(x, y, z, c=colors, ax=ax)
sc.bar3d(data, ax=ax)
sc.surf3d(data, cmap='orangeblue')
```

## Saving Figures

```python
sc.savefig('fig.png')                 # Publication quality + metadata
sc.savefig('fig.png', comments='v2')  # With custom comments
```
