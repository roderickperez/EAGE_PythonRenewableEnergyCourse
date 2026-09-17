# Streamplots

`streamplot` visualises a continuous 2-D vector field on a regular grid.

```python
import numpy as np
import matplotlib.pyplot as plt

x_grid = np.linspace(-2, 2, 30)
y_grid = np.linspace(-2, 2, 30)
x, y = np.meshgrid(x_grid, y_grid)
u, v = -y, x  # synthetic rotating vector field, not observed weather
fig, ax = plt.subplots()
ax.streamplot(x_grid, y_grid, u, v, density=1.2)
ax.set(xlabel="Grid x", ylabel="Grid y", title="Synthetic flow field")
plt.show()
```

It is useful for conceptual flow fields, but it does not show measured particle trajectories and should not imply temporal evolution from a static field.
