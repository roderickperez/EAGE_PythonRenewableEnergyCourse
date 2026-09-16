---
kernelspec:
  name: python3
  display_name: Python 3
---

# Geothermal-Energy Quiz and Python Challenge

These eight questions connect thermodynamics and subsurface concepts to basic and intermediate Python.

:::{admonition} Quiz 1 — Gradient, heat flux, or power?
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Match the quantities to units: geothermal gradient, conductive heat flux, thermal power, and electrical energy.

Units: W, W/m², °C/km, MWh.

:::

:::{admonition} Solution
:class: tip, dropdown

- Geothermal gradient: °C/km.
- Conductive heat flux: W/m².
- Thermal power: W.
- Electrical energy: MWh.
:::

:::{admonition} Quiz 2 — Minimum depth
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

With surface temperature 12 °C and a 34 °C/km linear gradient, calculate temperature at 3.5 km and the depth at which the linear model reaches 160 °C. State one limitation.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
surface_c, gradient_c_km = 12.0, 34.0
temperature_3_5km = surface_c + gradient_c_km * 3.5
target_depth_km = (160 - surface_c) / gradient_c_km
print(f"T(3.5 km) = {temperature_3_5km:.1f} °C")
print(f"Depth for 160 °C = {target_depth_km:.2f} km")
```

The constant gradient is a simplified extrapolation and does not establish flow or permeability.
:::

:::{admonition} Quiz 3 — Write a well-power function
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Write a function returning thermal, gross, and net MW using $\dot m c_p\Delta T$, conversion efficiency, and parasitic fraction. Validate inputs and test the ordering of outputs.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import numpy as np

def well_power_mw(mass_flow, production_c, reinjection_c,
                  conversion_efficiency, parasitic_fraction=0.1, cp=4180):
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [mass_flow, production_c, reinjection_c, conversion_efficiency, parasitic_fraction, cp]):
        raise ValueError("model inputs must be finite")
    mass_flow = np.asarray(mass_flow, dtype=float)
    delta_t = np.asarray(production_c, dtype=float) - reinjection_c
    if np.any(mass_flow < 0) or np.any(delta_t < 0):
        raise ValueError("mass flow and temperature difference must be non-negative")
    if not 0 <= conversion_efficiency <= 1 or not 0 <= parasitic_fraction < 1:
        raise ValueError("invalid efficiency or parasitic fraction")
    if cp <= 0:
        raise ValueError("specific heat must be positive")
    thermal = mass_flow * cp * delta_t / 1e6
    gross = thermal * conversion_efficiency
    net = gross * (1 - parasitic_fraction)
    return thermal, gross, net

thermal, gross, net = well_power_mw(80, 165, 70, 0.13, 0.12)
assert thermal >= gross >= net >= 0
print(f"Thermal={thermal:.2f} MW, gross={gross:.2f} MW, net={net:.2f} MW")
```
:::

:::{admonition} Quiz 4 — Carnot bound in Kelvin
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Plot Carnot efficiency for hot-source temperatures 100–250 °C and a 25 °C sink. Also calculate the incorrect Celsius-ratio result and show the difference.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import matplotlib.pyplot as plt

hot_c = np.linspace(100, 250, 100)
cold_c = 25
carnot = 1 - (cold_c + 273.15) / (hot_c + 273.15)
incorrect = 1 - cold_c / hot_c

plt.figure(figsize=(8, 4))
plt.plot(hot_c, carnot, label="Correct Kelvin ratio")
plt.plot(hot_c, incorrect, linestyle="--", label="Incorrect Celsius ratio")
plt.xlabel("Hot-source temperature (°C)")
plt.ylabel("Efficiency bound")
plt.title("Temperature ratios require an absolute scale")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
assert np.all((carnot > 0) & (carnot < 1))
```
:::

:::{admonition} Quiz 5 — Analyze a well field
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Create a DataFrame for four wells with mass flows `[62, 78, 55, 90]` kg/s and temperatures `[150, 168, 142, 175]` °C. Calculate net output by well, total it, and create a bar chart.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import pandas as pd

wells = pd.DataFrame({
    "well": ["A", "B", "C", "D"],
    "mass_flow_kgs": [62, 78, 55, 90],
    "production_c": [150, 168, 142, 175],
})
_, _, wells["net_mw"] = well_power_mw(
    wells["mass_flow_kgs"], wells["production_c"],
    reinjection_c=70, conversion_efficiency=0.12, parasitic_fraction=0.11,
)
wells.plot.bar(x="well", y="net_mw", figsize=(7, 4), legend=False, color="firebrick")
plt.ylabel("Net power (MW)")
plt.title(f"Well-field net power = {wells['net_mw'].sum():.1f} MW")
plt.tight_layout()
assert wells["net_mw"].ge(0).all()
```
:::

:::{admonition} Quiz 6 — Sensitivity surface
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

For an 80 kg/s well at 165/70 °C, vary conversion efficiency from 8–18% and parasitic fraction from 5–20%. Plot net power with `contourf`.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
efficiency_axis = np.linspace(0.08, 0.18, 60)
parasitic_axis = np.linspace(0.05, 0.20, 50)
ETA, PAR = np.meshgrid(efficiency_axis, parasitic_axis)
thermal_mw = 80 * 4180 * (165 - 70) / 1e6
NET = thermal_mw * ETA * (1 - PAR)

plt.figure(figsize=(8, 5))
contour = plt.contourf(ETA * 100, PAR * 100, NET, levels=20, cmap="inferno")
plt.colorbar(contour, label="Net power (MW)")
plt.xlabel("Conversion efficiency (%)")
plt.ylabel("Parasitic fraction (%)")
plt.title("Geothermal net-power sensitivity")
plt.tight_layout()
```
:::

:::{admonition} Quiz 7 — Decline, availability, and energy
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

For 12 MW initial net power, annual decline 1.5%, and availability 94%, calculate annual GWh over 25 years and cumulative energy. Plot both on separate axes.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
years = np.arange(1, 26)
net_power_mw = 12 * (1 - 0.015) ** (years - 1)
annual_gwh = net_power_mw * 8760 * 0.94 / 1000
cumulative_gwh = np.cumsum(annual_gwh)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(years, annual_gwh, marker="o", markersize=3)
axes[0].set(xlabel="Year", ylabel="Annual energy (GWh)", title="Annual output")
axes[1].plot(years, cumulative_gwh, color="darkorange")
axes[1].set(xlabel="Year", ylabel="Cumulative energy (GWh)", title="Cumulative output")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
assert np.all(np.diff(annual_gwh) < 0) and np.all(np.diff(cumulative_gwh) > 0)
```
:::

:::{admonition} Quiz 8 — Detect invalid observations
:class: note

**Reference:** Grant and Bixley, Chapters 2–3 [@grant2011].

Add two faulty rows to the well table: a negative flow and production below reinjection temperature. Write a validation function that returns problem rows rather than silently calculating negative power.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
faulty = pd.DataFrame({
    "well": ["E", "F"], "mass_flow_kgs": [-5, 40], "production_c": [160, 60]
})
audit = pd.concat([wells[["well", "mass_flow_kgs", "production_c"]], faulty], ignore_index=True)

def invalid_wells(frame, reinjection_c):
    invalid = (frame["mass_flow_kgs"] < 0) | (frame["production_c"] <= reinjection_c)
    return frame.loc[invalid].copy()

problems = invalid_wells(audit, 70)
print(problems.to_string(index=False))
assert set(problems["well"]) == {"E", "F"}
```
:::
