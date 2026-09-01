---
kernelspec:
  name: python3
  display_name: Python 3
---
# Quiz: Eolic (Wind) Energy — Fundamentals

This quiz focuses on the fundamental concepts of eolic (wind) energy, wind resource assessment, and turbine components.

---

:::{admonition} Question 1
:class: note

**Wind Resource Assessment**

The average wind power density (W/m²) is an indicator of wind resource quality. The 12 monthly mean speeds below support only a coarse teaching approximation: true wind power density requires higher-frequency speed observations because submonthly variability is lost.  
It is calculated as:

$$WPD = \frac{1}{2} \rho \overline{v^3}$$

Given 12 monthly average wind speeds (m/s):

```python
v_monthly = [6.1, 6.8, 7.4, 8.2, 9.1, 10.3,
             10.8, 9.7, 8.5, 7.3, 6.6, 6.2]
```

Calculate:
1. The annual average wind speed.
2. The average wind power density (assuming $\rho = 1.225$ kg/m³).
:::

:::{admonition} Question 1 (Solution)
:class: tip, dropdown

For this approximate exercise, $\overline{v^3}$ is the mean of the cubed monthly means, not the cube of their annual mean. Do not interpret it as a bankable resource assessment:

```{code-cell} python
import numpy as np

v_monthly = [6.1, 6.8, 7.4, 8.2, 9.1, 10.3,
             10.8, 9.7, 8.5, 7.3, 6.6, 6.2]

rho = 1.225
v_arr = np.array(v_monthly)

annual_avg = v_arr.mean()
WPD = 0.5 * rho * np.mean(v_arr**3)

print(f"Annual average wind speed: {annual_avg:.2f} m/s")
print(f"Wind Power Density:        {WPD:.2f} W/m²")
```
:::

---

:::{admonition} Question 2
:class: note

**Capacity Factor**

A 3 MW wind turbine generates 7,500 MWh of electricity in one year.

1. What is the annual capacity factor (%)?  
   > $CF = \frac{\text{Annual Energy}}{P_{rated} \times 8760}$
2. How many full-load equivalent hours does this represent?
:::

:::{admonition} Question 2 (Solution)
:class: tip, dropdown

We apply the capacity factor formula:

```{code-cell} python
P_rated_MW     = 3
annual_MWh     = 7500
hours_per_year = 8760

CF = annual_MWh / (P_rated_MW * hours_per_year) * 100
full_load_hours = annual_MWh / P_rated_MW

print(f"Annual capacity factor:     {CF:.1f}%")
print(f"Full-load equivalent hours: {full_load_hours:.0f} h/year")
```
:::

---

:::{admonition} Question 3
:class: note

**Wind Shear — Logarithmic Profile**

Wind speed increases with height above ground according to the log-law:

$$v(z) = v_{ref} \cdot \frac{\ln(z / z_0)}{\ln(z_{ref} / z_0)}$$

where:
- $v_{ref} = 8$ m/s measured at $z_{ref} = 10$ m  
- $z_0 = 0.03$ m (roughness length — grassland)

Calculate the wind speed at hub heights of 60, 80, 100, and 120 m.
:::

:::{admonition} Question 3 (Solution)
:class: tip, dropdown

We apply the log-law at each hub height:

```{code-cell} python
import numpy as np

v_ref = 8.0    # m/s
z_ref = 10.0   # m
z0    = 0.03   # m (grassland)

hub_heights = [60, 80, 100, 120]

print(f"{'Height (m)':>12}  {'Wind Speed (m/s)':>18}")
print("-" * 35)
for z in hub_heights:
    v_z = v_ref * np.log(z / z0) / np.log(z_ref / z0)
    print(f"{z:>12}  {v_z:>18.3f}")
```
:::

---

:::{admonition} Question 4
:class: note

**Turbine Comparison**

Compare two turbines at the same site with average wind speed $\bar{v} = 9$ m/s:

| Turbine | Rotor Diameter (m) | $C_P$ |
|---------|-------------------|-------|
| A       | 82                | 0.40  |
| B       | 112               | 0.45  |

For $\rho = 1.225$ kg/m³, calculate the power output of each turbine and determine which is more efficient per unit swept area.
:::

:::{admonition} Question 4 (Solution)
:class: tip, dropdown

Power efficiency per unit area = $P / A$:

```{code-cell} python
import math

rho = 1.225
v   = 9.0

turbines = {
    "A": {"D": 82,  "Cp": 0.40},
    "B": {"D": 112, "Cp": 0.45},
}

print(f"{'Turbine':>8}  {'A (m²)':>10}  {'P (kW)':>10}  {'P/A (W/m²)':>12}")
print("-" * 46)
for name, t in turbines.items():
    A = math.pi * (t["D"] / 2)**2
    P = 0.5 * rho * A * t["Cp"] * v**3
    print(f"{name:>8}  {A:>10.1f}  {P/1000:>10.2f}  {P/A:>12.2f}")
```
:::

---

:::{admonition} Question 5
:class: note

**Wind Farm Wake Effect**

In a wind farm, downstream turbines experience reduced wind speeds due to wake effects.
A simplified model assumes that each turbine row reduces wind speed by 8 %.

Starting with $v_0 = 10$ m/s, a turbine with $D = 90$ m, $C_P = 0.42$, and $\rho = 1.225$ kg/m³:

Compute the power output (kW) for each of 5 successive turbine rows (the first row sees $v_0$, each subsequent row sees 8 % less).
:::

:::{admonition} Question 5 (Solution)
:class: tip, dropdown

We iteratively reduce the wind speed and calculate power for each row:

```{code-cell} python
import math

rho   = 1.225
D     = 90
Cp    = 0.42
v     = 10.0
A     = math.pi * (D / 2)**2
wake  = 0.08   # 8% speed reduction per row

print(f"{'Row':>5}  {'v (m/s)':>10}  {'P (kW)':>10}")
print("-" * 30)
for row in range(1, 6):
    P = 0.5 * rho * A * Cp * v**3 / 1000
    print(f"{row:>5}  {v:>10.3f}  {P:>10.2f}")
    v *= (1 - wake)
```
:::
