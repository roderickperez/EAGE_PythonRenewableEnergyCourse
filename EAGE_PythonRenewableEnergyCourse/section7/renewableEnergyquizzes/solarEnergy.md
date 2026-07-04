---
kernelspec:
  name: python3
  display_name: Python 3
---
# Quiz: Solar Energy

Test your knowledge of solar photovoltaic systems and Python calculations.

---

:::{admonition} Question 1
:class: note

**PV Panel Power**

The DC power output of a PV panel is:

$$P_{DC} = G \cdot A \cdot \eta_{panel}$$

where $G$ is solar irradiance (W/m²), $A$ is panel area (m²), and $\eta_{panel}$ is panel efficiency.

Calculate the DC power (W) for:
- $G = 850$ W/m², $A = 1.6$ m², $\eta_{panel} = 0.19$.
:::

:::{admonition} Question 1 (Solution)
:class: tip, dropdown

A direct substitution into the formula:

```{code-cell} python
G        = 850    # W/m²
A        = 1.6    # m²
eta      = 0.19   # panel efficiency

P_DC = G * A * eta
print(f"DC Power output: {P_DC:.2f} W")
```
:::

---

:::{admonition} Question 2
:class: note

**Temperature Derating**

PV panel power decreases with temperature above the Standard Test Condition (STC) of 25 °C.
The corrected power is:

$$P_T = P_{STC} \left[1 + \gamma (T - 25)\right]$$

where $\gamma = -0.004$ /°C (temperature coefficient).

Calculate $P_T$ for $P_{STC} = 380$ W and temperatures `T = [20, 25, 35, 45, 55, 65]` °C.
Print a table of results.
:::

:::{admonition} Question 2 (Solution)
:class: tip, dropdown

We loop over the temperature list and apply the derating formula:

```{code-cell} python
P_stc = 380      # W
gamma = -0.004   # /°C
temps = [20, 25, 35, 45, 55, 65]

print(f"{'Temp (°C)':>10}  {'P_T (W)':>10}")
print("-" * 25)
for T in temps:
    P_T = P_stc * (1 + gamma * (T - 25))
    print(f"{T:>10}  {P_T:>10.2f}")
```
:::

---

:::{admonition} Question 3
:class: note

**Solar Farm Energy**

A solar farm has 2000 panels, each rated at 400 W (STC).  
The farm operates for 5 peak sun hours (PSH) per day with a system efficiency of 80 % (accounting for inverter losses, cabling, etc.).

Calculate:
1. Total installed capacity (kW).
2. Daily energy production (kWh).
3. Annual energy production (MWh).
:::

:::{admonition} Question 3 (Solution)
:class: tip, dropdown

We scale from individual panel to farm level:

```{code-cell} python
n_panels    = 2000
P_panel_W   = 400      # W
PSH         = 5        # hours/day
eta_system  = 0.80

capacity_kW = n_panels * P_panel_W / 1000
daily_kWh   = capacity_kW * PSH * eta_system
annual_MWh  = daily_kWh * 365 / 1000

print(f"Installed capacity:       {capacity_kW:,.0f} kW")
print(f"Daily energy production:  {daily_kWh:,.0f} kWh")
print(f"Annual energy production: {annual_MWh:,.1f} MWh")
```
:::

---

:::{admonition} Question 4
:class: note

**Irradiance vs. Power Curve**

For a single PV panel ($A = 1.65$ m², $\eta = 0.20$), plot power output (W) as a function of irradiance from 0 to 1000 W/m² in steps of 50 W/m².

Add axis labels and a title.
:::

:::{admonition} Question 4 (Solution)
:class: tip, dropdown

We use NumPy for the irradiance array and Matplotlib to plot:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

A   = 1.65   # m²
eta = 0.20

G_values = np.arange(0, 1050, 50)   # W/m²
P_values = G_values * A * eta

plt.figure(figsize=(7, 4))
plt.plot(G_values, P_values, color="goldenrod", linewidth=2, marker="o", markersize=4)
plt.xlabel("Solar Irradiance (W/m²)")
plt.ylabel("Power Output (W)")
plt.title("PV Panel Power vs. Solar Irradiance")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```
:::

---

:::{admonition} Question 5
:class: note

**Capacity Factor**

A solar plant with a rated capacity of 10 MW produces the following monthly energy (MWh):

```python
monthly_energy = [1200, 1350, 2100, 2800, 3500, 3800,
                  3700, 3400, 2500, 1800, 1300, 1100]
```

Calculate:
1. Total annual energy (MWh).
2. The annual capacity factor (%).

> *Capacity Factor = Annual Energy / (Rated Power × 8760 hours)*
:::

:::{admonition} Question 5 (Solution)
:class: tip, dropdown

We sum the monthly values and apply the capacity factor formula:

```{code-cell} python
rated_MW = 10
monthly_energy = [1200, 1350, 2100, 2800, 3500, 3800,
                  3700, 3400, 2500, 1800, 1300, 1100]

annual_MWh   = sum(monthly_energy)
CF           = annual_MWh / (rated_MW * 8760) * 100

print(f"Annual energy production: {annual_MWh:,} MWh")
print(f"Annual capacity factor:   {CF:.1f}%")
```
:::
