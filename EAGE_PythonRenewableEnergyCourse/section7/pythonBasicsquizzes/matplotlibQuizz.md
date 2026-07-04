---
kernelspec:
  name: python3
  display_name: Python 3
---
# Quiz: Matplotlib

Test your ability to create clear and informative visualisations with Matplotlib.

---

:::{admonition} Question 1
:class: note

**Line Plot**

Monthly average wind speed (m/s) for a wind farm site is recorded over one year:

```python
months = range(1, 13)
wind_speed = [5.2, 6.1, 7.4, 8.0, 9.3, 10.1, 9.8, 8.5, 7.2, 6.4, 5.8, 5.5]
```

Create a line plot with:
- X-axis: Month number (1–12), labelled `"Month"`
- Y-axis: labelled `"Wind Speed (m/s)"`
- Title: `"Monthly Average Wind Speed"`
- Markers at each data point.
:::

:::{admonition} Question 1 (Solution)
:class: tip, dropdown

We use `plt.plot()` with the `marker` argument:

```{code-cell} python
import matplotlib.pyplot as plt

months = range(1, 13)
wind_speed = [5.2, 6.1, 7.4, 8.0, 9.3, 10.1, 9.8, 8.5, 7.2, 6.4, 5.8, 5.5]

plt.figure(figsize=(8, 4))
plt.plot(months, wind_speed, marker="o", color="steelblue", linewidth=2)
plt.xlabel("Month")
plt.ylabel("Wind Speed (m/s)")
plt.title("Monthly Average Wind Speed")
plt.xticks(months)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```
:::

---

:::{admonition} Question 2
:class: note

**Bar Chart**

The installed renewable capacity (GW) by energy type in a hypothetical country is:

```python
types = ["Wind", "Solar", "Hydro", "Geothermal"]
capacity = [45, 38, 120, 12]
```

Create a bar chart with:
- Each bar in a different colour.
- X-axis labelled `"Energy Type"` and Y-axis labelled `"Capacity (GW)"`.
- Title: `"Installed Renewable Capacity by Type"`.
- Values displayed on top of each bar.
:::

:::{admonition} Question 2 (Solution)
:class: tip, dropdown

We use `plt.bar()` and annotate each bar with `plt.text()`:

```{code-cell} python
import matplotlib.pyplot as plt

types = ["Wind", "Solar", "Hydro", "Geothermal"]
capacity = [45, 38, 120, 12]
colors = ["steelblue", "gold", "mediumseagreen", "tomato"]

plt.figure(figsize=(7, 5))
bars = plt.bar(types, capacity, color=colors, edgecolor="black")

for bar, val in zip(bars, capacity):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             str(val), ha="center", va="bottom", fontsize=11)

plt.xlabel("Energy Type")
plt.ylabel("Capacity (GW)")
plt.title("Installed Renewable Capacity by Type")
plt.tight_layout()
plt.show()
```
:::

---

:::{admonition} Question 3
:class: note

**Scatter Plot**

For 20 randomly generated wind turbines, plot wind speed (x-axis) vs. power output (y-axis) as a scatter plot. Use:

```python
import numpy as np
np.random.seed(7)
wind_speed = np.random.uniform(3, 15, 20)
power_output = 0.5 * wind_speed**2 + np.random.normal(0, 5, 20)
```

Label axes and add a title.
:::

:::{admonition} Question 3 (Solution)
:class: tip, dropdown

Scatter plots reveal correlations between two continuous variables:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)
wind_speed = np.random.uniform(3, 15, 20)
power_output = 0.5 * wind_speed**2 + np.random.normal(0, 5, 20)

plt.figure(figsize=(7, 5))
plt.scatter(wind_speed, power_output, color="darkorange", edgecolors="black", s=80)
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Output (kW)")
plt.title("Wind Speed vs. Power Output")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```
:::

---

:::{admonition} Question 4
:class: note

**Subplots**

Using `plt.subplots()`, create a figure with two side-by-side plots:
1. **Left** — A line plot of daily solar irradiance over 30 days (use sine curve data).
2. **Right** — A histogram of the same irradiance data (20 bins).

Add titles and axis labels to both subplots.
:::

:::{admonition} Question 4 (Solution)
:class: tip, dropdown

`plt.subplots(1, 2)` creates a 1-row, 2-column layout:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

days = np.arange(1, 31)
irradiance = 500 + 300 * np.sin(np.linspace(0, np.pi, 30))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: line plot
ax1.plot(days, irradiance, color="goldenrod", linewidth=2)
ax1.set_xlabel("Day")
ax1.set_ylabel("Irradiance (W/m²)")
ax1.set_title("Daily Solar Irradiance")
ax1.grid(True, linestyle="--", alpha=0.5)

# Right: histogram
ax2.hist(irradiance, bins=20, color="skyblue", edgecolor="black")
ax2.set_xlabel("Irradiance (W/m²)")
ax2.set_ylabel("Frequency")
ax2.set_title("Irradiance Distribution")

plt.tight_layout()
plt.show()
```
:::

---

:::{admonition} Question 5
:class: note

**Customisation**

Plot the following geothermal well temperature profile:

```python
depth    = [0, 100, 200, 400, 800, 1500, 2500, 3500]
temp_C   = [15, 25, 40, 70, 120, 180, 240, 310]
```

Requirements:
- Y-axis inverted (depth increases downward).
- Line colour: dark red (`"darkred"`), line style: `"--"`.
- Markers: `"s"` (squares), markersize 8.
- Grid enabled.
- Title: `"Geothermal Temperature Gradient"`.
- X-axis: `"Temperature (°C)"`, Y-axis: `"Depth (m)"`.
:::

:::{admonition} Question 5 (Solution)
:class: tip, dropdown

We invert the Y-axis with `ax.invert_yaxis()`:

```{code-cell} python
import matplotlib.pyplot as plt

depth  = [0, 100, 200, 400, 800, 1500, 2500, 3500]
temp_C = [15, 25, 40, 70, 120, 180, 240, 310]

fig, ax = plt.subplots(figsize=(5, 7))
ax.plot(temp_C, depth, color="darkred", linestyle="--",
        marker="s", markersize=8, linewidth=2)
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Depth (m)")
ax.set_title("Geothermal Temperature Gradient")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```
:::
