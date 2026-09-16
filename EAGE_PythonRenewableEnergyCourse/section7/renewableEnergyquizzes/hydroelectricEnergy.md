---
kernelspec:
  name: python3
  display_name: Python 3
---

# Hydropower Quiz and Python Challenge

Complete the questions before opening the solutions. The sequence progresses from concepts and scalars to functions, arrays, data validation, and plots.

:::{admonition} Quiz 1 — Choose the correct statement
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

Which statement is correct?

A. Capacity factor and turbine efficiency are interchangeable.  
B. Net head equals gross head plus friction loss.  
C. MW is power; MWh is energy.  
D. Pumped storage produces more energy than it consumes.

:::

:::{admonition} Solution
:class: tip, dropdown

**C.** MW is a rate of energy transfer; multiplying MW by hours produces MWh. Net head is gross head minus losses, capacity factor and efficiency describe different ratios, and round-trip storage efficiency is below one.
:::

:::{admonition} Quiz 2 — Trace the units
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

Show that $\eta\rho gQH$ has units of watts. Then calculate output for $Q=55$ m$^3$/s, gross head 72 m, 5 m head loss, and efficiency 0.89.

:::

:::{admonition} Solution
:class: tip, dropdown

$\mathrm{kg/m^3\;m/s^2\;m^3/s\;m=kg\,m^2/s^3=W}$.

```{code-cell} python
rho, gravity = 1000.0, 9.81
flow, gross_head, loss, efficiency = 55.0, 72.0, 5.0, 0.89
net_head = gross_head - loss
power_mw = efficiency * rho * gravity * flow * net_head / 1e6
print(f"Net head: {net_head:.1f} m; power: {power_mw:.2f} MW")
assert 32 < power_mw < 33
```
:::

:::{admonition} Quiz 3 — Write and test a function
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

Write `hydro_power_mw(flow, net_head, efficiency=0.9, rated_mw=None)`. Reject negative values and invalid efficiency, accept arrays, and clip to the optional rating. Test zero flow and a clipped case.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import numpy as np

def hydro_power_mw(flow, net_head, efficiency=0.9, rated_mw=None):
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [flow, net_head, efficiency]):
        raise ValueError("model inputs must be finite")
    flow = np.asarray(flow, dtype=float)
    head = np.asarray(net_head, dtype=float)
    if np.any(flow < 0) or np.any(head < 0):
        raise ValueError("flow and head must be non-negative")
    if not 0 <= efficiency <= 1:
        raise ValueError("efficiency must be between 0 and 1")
    power = efficiency * 1000 * 9.81 * flow * head / 1e6
    if rated_mw is not None:
        if not np.isfinite(rated_mw) or rated_mw < 0:
            raise ValueError("rating must be finite and non-negative")
        power = np.minimum(power, rated_mw)
    return float(power) if power.ndim == 0 else power

assert hydro_power_mw(0, 50) == 0
assert hydro_power_mw(100, 80, rated_mw=40) == 40
```
:::

:::{admonition} Quiz 4 — Environmental flow and plant limit
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

For river flows `[8, 15, 25, 40, 65, 90]` m$^3$/s, reserve 12 m$^3$/s for environmental flow. Calculate plant power at 50 m net head, 88% efficiency, and 25 MW rating. Print a table.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
river_flow = np.array([8, 15, 25, 40, 65, 90], dtype=float)
usable_flow = np.maximum(river_flow - 12, 0)
plant_power = hydro_power_mw(usable_flow, 50, 0.88, rated_mw=25)

print("river  usable  power")
for river, usable, power in zip(river_flow, usable_flow, plant_power):
    print(f"{river:5.0f}  {usable:6.0f}  {power:5.2f} MW")
assert np.all(plant_power >= 0) and np.all(plant_power <= 25)
```
:::

:::{admonition} Quiz 5 — Unequal time steps
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

A sensor records `[10, 12, 14, 8]` MW for intervals lasting `[0.25, 0.25, 0.5, 1.0]` hours. Calculate energy correctly and compare it with the incorrect unweighted sum.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
power_mw = np.array([10, 12, 14, 8], dtype=float)
duration_h = np.array([0.25, 0.25, 0.5, 1.0])
energy_mwh = np.sum(power_mw * duration_h)
incorrect = power_mw.sum()
print(f"Correct: {energy_mwh:.1f} MWh; unweighted sum: {incorrect:.1f}")
assert np.isclose(energy_mwh, 20.5)
```
:::

:::{admonition} Quiz 6 — Flow-duration and power-duration curves
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

Create 365 reproducible daily flows, calculate clipped power, sort both arrays descending, and plot them against exceedance probability. What causes a horizontal segment in the power-duration curve?

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
daily_flow = np.clip(rng.lognormal(mean=3.7, sigma=0.45, size=365), 0, 130)
daily_power = hydro_power_mw(daily_flow, 45, 0.90, rated_mw=30)
exceedance = 100 * np.arange(1, len(daily_flow) + 1) / (len(daily_flow) + 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(exceedance, np.sort(daily_flow)[::-1])
axes[0].set(xlabel="Exceedance probability (%)", ylabel="Flow (m³/s)", title="Flow duration")
axes[1].plot(exceedance, np.sort(daily_power)[::-1], color="tab:green")
axes[1].set(xlabel="Exceedance probability (%)", ylabel="Power (MW)", title="Power duration")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
```

The horizontal high-power segment is generator clipping at rated power.
:::

:::{admonition} Quiz 7 — Monthly energy and capacity factor
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

Use the 2024 monthly mean powers below. Calculate monthly GWh with actual calendar hours, annual GWh, and capacity factor for a 30 MW plant. Create a bar chart.

```python
mean_power_mw = [15, 16, 18, 22, 25, 27, 24, 20, 17, 15, 14, 13]
```

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import pandas as pd

months = pd.period_range("2024-01", periods=12, freq="M")
mean_power = np.array([15, 16, 18, 22, 25, 27, 24, 20, 17, 15, 14, 13])
hours = months.days_in_month.to_numpy() * 24
monthly_gwh = mean_power * hours / 1000
annual_gwh = monthly_gwh.sum()
capacity_factor = annual_gwh * 1000 / (30 * hours.sum())

plt.figure(figsize=(9, 4))
plt.bar(months.astype(str), monthly_gwh, color="steelblue")
plt.xticks(rotation=45)
plt.ylabel("Energy (GWh)")
plt.title(f"2024 hydropower energy; CF = {capacity_factor:.1%}")
plt.tight_layout()
assert 0 <= capacity_factor <= 1
```
:::

:::{admonition} Quiz 8 — Pumped-storage comparison
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

For 2.5 GWh pumping input and round-trip efficiency from 0.70 to 0.88, create a function returning recovered energy and loss. Plot both. Which quantity must always be larger?

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
def storage_cycle(input_gwh, round_trip_efficiency):
    eta = np.asarray(round_trip_efficiency, dtype=float)
    if input_gwh < 0 or np.any((eta < 0) | (eta > 1)):
        raise ValueError("invalid input energy or efficiency")
    recovered = input_gwh * eta
    return recovered, input_gwh - recovered

eta = np.linspace(0.70, 0.88, 10)
recovered, losses = storage_cycle(2.5, eta)
plt.figure(figsize=(8, 4))
plt.plot(eta, recovered, marker="o", label="Recovered")
plt.plot(eta, losses, marker="s", label="Loss")
plt.xlabel("Round-trip efficiency")
plt.ylabel("Energy (GWh)")
plt.title("Pumped-storage cycle")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
assert np.all(recovered <= 2.5) and np.all(losses >= 0)
```

Pumping input is always at least as large as recovered electricity.
:::
