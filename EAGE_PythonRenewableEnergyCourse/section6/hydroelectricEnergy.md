---
kernelspec:
  name: python3
  display_name: Python 3
---

# Hydroelectric Energy

## Learning goals

By the end of this lesson you should be able to:

- distinguish head, flow, power, and energy;
- calculate gross and net hydraulic power with consistent SI units;
- implement reusable Python functions with input checks;
- convert a flow time series into power and energy;
- interpret capacity factor, turbine limits, and pumped-storage efficiency;
- create and explain engineering plots rather than only calculate one number.

Hydropower converts the gravitational potential energy of water into shaft power and then electricity. Conventional storage plants, run-of-river plants, and pumped-storage plants use the same energy balance but have different operating constraints. Head and flow determine the resource; turbine-generator efficiency and equipment limits determine the electrical output. See the U.S. Department of Energy overview and turbine guidance [@doeHydropowerWorks; @doeHydroTurbines].

## Concepts and equations

### Gross head, head loss, and net head

The gross head is the upstream-to-downstream elevation or hydraulic-head difference. Friction and local losses reduce the head available at the turbine:

$$H_{net}=H_{gross}-h_{loss}$$

Always use $H_{net}$ in the plant power equation. Treating gross head as net head overestimates production.

### Power and energy

For water density $\rho$, gravity $g$, volume flow $Q$, net head $H_{net}$, and combined turbine-generator efficiency $\eta$:

$$P_e=\eta\rho gQH_{net}$$

| Quantity | SI unit |
|---|---|
| $\rho$ | kg/m$^3$ |
| $g$ | m/s$^2$ |
| $Q$ | m$^3$/s |
| $H$ | m |
| $P$ | W |

Power is an instantaneous rate. Energy over intervals $\Delta t_i$ is

$$E=\sum_i P_i\Delta t_i$$

If $P$ is in MW and $\Delta t$ in hours, $E$ is in MWh. For rated power $P_r$ and capacity factor $CF$ over $T$ hours:

$$E=P_r\,CF\,T,\qquad CF=\frac{E}{P_rT}$$

Capacity factor includes hydrology, outages, dispatch, and other constraints; it is not turbine efficiency.

### Equipment limits and storage

The generator limits electrical output:

$$P_{plant}=\min(P_e,P_r)$$

For pumped storage, round-trip efficiency compares recovered generation with pumping energy:

$$\eta_{RT}=\frac{E_{generated}}{E_{pumping}}$$

It is below one. Pumped storage shifts energy in time; it is not a net energy source.

## A tested Python model

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

RHO_WATER = 1000.0  # kg/m3
G = 9.81            # m/s2

def hydro_power_mw(flow_m3s, gross_head_m, efficiency=0.90,
                   head_loss_m=0.0, rated_power_mw=None):
    """Return hydroelectric power in MW for scalar or array inputs."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [flow_m3s, gross_head_m, efficiency, head_loss_m]):
        raise ValueError("model inputs must be finite")
    flow = np.asarray(flow_m3s, dtype=float)
    net_head = np.asarray(gross_head_m, dtype=float) - head_loss_m
    if np.any(flow < 0):
        raise ValueError("flow_m3s must be non-negative")
    if np.any(np.asarray(head_loss_m) < 0):
        raise ValueError("head loss must be non-negative")
    if np.any(net_head < 0):
        raise ValueError("head loss cannot exceed gross head")
    if not 0 <= efficiency <= 1:
        raise ValueError("efficiency must be between 0 and 1")
    power = efficiency * RHO_WATER * G * flow * net_head / 1e6
    if rated_power_mw is not None:
        if not np.isfinite(rated_power_mw) or rated_power_mw < 0:
            raise ValueError("rated_power_mw must be non-negative")
        power = np.minimum(power, rated_power_mw)
    return float(power) if power.ndim == 0 else power

assert hydro_power_mw(0, 50) == 0
assert np.isclose(hydro_power_mw(100, 50, 0.90), 44.145)
```

The function combines basic Python ideas: arguments, defaults, conditionals, exceptions, NumPy arrays, and a return value.

## Worked time-series example

The following synthetic daily flows are teaching data, not a site assessment. A real study needs measured flow, an environmental-flow rule, a flow-duration curve covering wet and dry years, head-loss estimates, and an equipment-efficiency curve.

```{code-cell} python
rng = np.random.default_rng(7)
days = np.arange(1, 366)
seasonal_flow = 75 + 35 * np.sin(2 * np.pi * (days - 60) / 365)
flow_m3s = np.clip(seasonal_flow + rng.normal(0, 7, len(days)), 10, None)

power_mw = hydro_power_mw(
    flow_m3s, gross_head_m=48, head_loss_m=3,
    efficiency=0.90, rated_power_mw=35,
)
energy_gwh = power_mw.sum() * 24 / 1000
capacity_factor = energy_gwh * 1000 / (35 * 8760)

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(days, flow_m3s, color="tab:blue")
axes[0].set(ylabel="Flow (m³/s)", title="Synthetic daily river flow")
axes[1].plot(days, power_mw, color="tab:green")
axes[1].axhline(35, color="black", linestyle="--", label="Rated power")
axes[1].set(xlabel="Day of year", ylabel="Power (MW)",
            title=f"Energy = {energy_gwh:.1f} GWh; CF = {capacity_factor:.1%}")
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
```

The flat segments at 35 MW are generator clipping, not constant river flow. This is why resource and output plots should be viewed together.

## Guided exercises

:::{admonition} Exercise 1 — Scalar calculation
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

For $Q=42$ m$^3$/s, $H_{gross}=65$ m, $h_{loss}=4$ m, and $\eta=0.88$, calculate net head and power in MW. Check that the result is between 20 and 25 MW.

```python
# TODO: calculate net_head_m
# TODO: call hydro_power_mw and print the result
```
:::

:::{admonition} Exercise 1 — Solution
:class: tip, dropdown

1. Subtract the head loss.
2. Pass the given flow and efficiency to the model.
3. Compare MW with the stated interval.

```{code-cell} python
net_head_m = 65 - 4
exercise_power_mw = hydro_power_mw(42, 65, efficiency=0.88, head_loss_m=4)
print(f"Net head: {net_head_m:.1f} m; power: {exercise_power_mw:.2f} MW")
assert 20 < exercise_power_mw < 25
```
:::

:::{admonition} Exercise 2 — Lists, loops, and arrays
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

Evaluate flows `[10, 20, 30, 40, 50]` m$^3$/s at 40 m net head. Store powers in a list, print a two-column table, repeat with NumPy, and compare with `np.allclose`.

```python
flows = [10, 20, 30, 40, 50]
# TODO: use a comprehension or loop
# TODO: repeat with np.array and compare
```
:::

:::{admonition} Exercise 2 — Solution
:class: tip, dropdown

1. Calculate one power for each flow.
2. Repeat with a NumPy array.
3. Compare the two methods.

```{code-cell} python
flows = [10, 20, 30, 40, 50]
power_list = [hydro_power_mw(flow, 40) for flow in flows]
for flow, power in zip(flows, power_list):
    print(f"{flow:3d} m³/s -> {power:6.2f} MW")
power_array = hydro_power_mw(np.array(flows), 40)
assert np.allclose(power_list, power_array)
```
:::

:::{admonition} Exercise 3 — Environmental-flow dispatch
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

Write `dispatch_hydro(flow, environmental_flow, ...)`. Only `max(flow - environmental_flow, 0)` reaches the turbine. Test zero, the environmental-flow boundary, and normal operation.

```python
def dispatch_hydro(flow, environmental_flow, gross_head_m, rated_power_mw):
    # TODO: calculate usable flow
    # TODO: call hydro_power_mw
    pass
```
:::

:::{admonition} Exercise 3 — Solution
:class: tip, dropdown

1. Reserve environmental flow.
2. Clamp usable flow at zero.
3. Test no flow, the reserve boundary and normal flow.

```{code-cell} python
def dispatch_hydro(flow, environmental_flow, gross_head_m, rated_power_mw):
    if environmental_flow < 0:
        raise ValueError("environmental_flow must be non-negative")
    usable_flow = np.maximum(np.asarray(flow, dtype=float) - environmental_flow, 0)
    return hydro_power_mw(usable_flow, gross_head_m, rated_power_mw=rated_power_mw)

assert dispatch_hydro(0, 5, 40, 20) == 0
assert dispatch_hydro(5, 5, 40, 20) == 0
assert dispatch_hydro(30, 5, 40, 20) > 0
```
:::

:::{admonition} Exercise 4 — Duration curves
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

Sort the worked-example flow and power arrays from largest to smallest. Plot both against exceedance percentage. Explain the flat rated-power region.

```python
# TODO: calculate exceedance percentages
# TODO: sort flow_m3s and power_mw descending
# TODO: create two labelled plots
```
:::

:::{admonition} Exercise 4 — Solution
:class: tip, dropdown

1. Sort flows from largest to smallest.
2. Assign exceedance positions.
3. Compare flow and capped power.

```{code-cell} python
exceedance = 100 * np.arange(1, len(flow_m3s) + 1) / (len(flow_m3s) + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(exceedance, np.sort(flow_m3s)[::-1])
axes[0].set(xlabel="Exceedance (%)", ylabel="Flow (m³/s)", title="Flow duration")
axes[1].plot(exceedance, np.sort(power_mw)[::-1], color="tab:green")
axes[1].set(xlabel="Exceedance (%)", ylabel="Power (MW)", title="Power duration")
for axis in axes:
    axis.grid(alpha=0.3)
plt.tight_layout()
```

The horizontal section is generator clipping at rated power.
:::

:::{admonition} Exercise 5 — Unequal month lengths
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

Convert monthly mean power to GWh using each month's actual number of hours. Create a bar chart and reconcile the annual total.

```python
monthly_mean_mw = [15, 16, 18, 22, 25, 27, 24, 20, 17, 15, 14, 13]
# TODO: obtain days per month with pandas
# TODO: calculate and plot monthly GWh
```
:::

:::{admonition} Exercise 5 — Solution
:class: tip, dropdown

1. Build the monthly calendar.
2. Multiply mean MW by each month’s actual hours.
3. Convert to GWh and reconcile the yearly sum.

```{code-cell} python
import pandas as pd

months = pd.period_range("2024-01", periods=12, freq="M")
monthly_mean_mw = np.array([15, 16, 18, 22, 25, 27, 24, 20, 17, 15, 14, 13])
monthly_gwh = monthly_mean_mw * months.days_in_month.to_numpy() * 24 / 1000
annual_gwh = monthly_gwh.sum()
plt.figure(figsize=(9, 4))
plt.bar(months.astype(str), monthly_gwh)
plt.xticks(rotation=45)
plt.ylabel("Energy (GWh)")
plt.title(f"Monthly hydropower energy; total = {annual_gwh:.1f} GWh")
plt.tight_layout()
assert np.isclose(annual_gwh, sum(monthly_gwh))
```
:::

:::{admonition} Exercise 6 — Pumped-storage cycle
:class: note

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

For 2.0 GWh pumping input and round-trip efficiencies 0.70–0.85, calculate and plot recovered energy and loss. Validate both outputs.

```python
# TODO: write storage_cycle(input_gwh, efficiency)
# TODO: plot recovered energy and loss
```
:::

:::{admonition} Exercise 6 — Solution
:class: tip, dropdown

1. Multiply pumping input by each round-trip efficiency.
2. Subtract recovered electricity from input to obtain losses.
3. Plot both and verify they sum to the input.

```{code-cell} python
def storage_cycle(input_gwh, efficiency):
    efficiency = np.asarray(efficiency, dtype=float)
    if input_gwh < 0 or np.any((efficiency < 0) | (efficiency > 1)):
        raise ValueError("invalid energy or efficiency")
    recovered = input_gwh * efficiency
    return recovered, input_gwh - recovered

round_trip_efficiency = np.linspace(0.70, 0.85, 10)
recovered, loss = storage_cycle(2.0, round_trip_efficiency)
plt.figure(figsize=(8, 4))
plt.plot(round_trip_efficiency, recovered, marker="o", label="Recovered")
plt.plot(round_trip_efficiency, loss, marker="s", label="Loss")
plt.xlabel("Round-trip efficiency")
plt.ylabel("Energy (GWh)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
assert np.all(recovered <= 2.0) and np.all(loss >= 0)
```
:::

## Common mistakes

- Using gross head when the equation requires net head.
- Confusing efficiency with capacity factor.
- Multiplying MW by seconds and labelling the result MWh.
- Ignoring generator clipping or environmental-flow requirements.
- Applying capacity factor to power that is already a time average.
- Treating pumped storage as a source of net electricity.

## Check your understanding

Continue with the [hydropower quiz and Python challenge](../section7/renewableEnergyquizzes/hydroelectricEnergy.md).
