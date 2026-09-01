# Quiver Plots

A quiver plot displays vector components, for example eastward and northward wind. Convert meteorological “direction from” conventions carefully before calculating components.

```python
ax.quiver(x, y, u_east, v_north)
```

Include a vector key or documented scale; otherwise arrow lengths cannot be interpreted quantitatively.
