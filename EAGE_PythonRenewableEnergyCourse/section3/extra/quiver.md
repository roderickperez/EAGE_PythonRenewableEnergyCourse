# Quiver Plots

A quiver plot displays vector components, for example eastward and northward wind. Convert meteorological “direction from” conventions carefully before calculating components.

```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic eastward/northward velocity components at four grid points.
x, y = np.meshgrid([0, 1], [0, 1])
u_east = np.array([[2, 3], [1, 2]])
v_north = np.array([[1, 0], [2, 1]])
fig, ax = plt.subplots()
ax.quiver(x, y, u_east, v_north)
ax.set(xlabel="Grid x", ylabel="Grid y", title="Synthetic wind vectors (components in m/s)")
plt.show()
```

Include a vector key or documented scale; otherwise arrow lengths cannot be interpreted quantitatively.
