---
kernelspec:
  name: python3
  display_name: Python 3
---

# Hydropower Quiz

Unless a question states otherwise, use water density 1,000 kg/m³ and gravitational acceleration 9.81 m/s².

Complete the questions before opening the solutions. The sequence progresses from concepts and scalars to functions, arrays, data validation, and plots.

**4 Easy · 8 Medium · 8 Hard.** Try each question before opening its solution. The code examples build on a shared Python session: run the preceding setup and solution code in order, or provide equivalent imports, data and functions yourself. In the sandbox, **Include preceding examples** does not run answers hidden in solution dropdowns.

In the sandbox, after opening a code answer, **Include preceding practice answers** loads the earlier runnable examples and answers plus that answer into the editor. Review the code and select **Run Python**; loading it does not run it automatically.

:::{admonition} Quiz 1 — Choose the correct statement
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

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

<span class="difficulty-badge difficulty-easy">Easy</span>

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

:::{admonition} Quiz 3 — Convert flow units
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

**Reference:** [@jica2011; @ifc2015].

Write a function to convert litres per second to m³/s. Check 2,500 L/s.

:::

:::{admonition} Solution
:class: tip, dropdown

There are 1,000 litres in one cubic metre.

```{code-cell} python
def litres_to_m3s(litres_s):
    return litres_s / 1000

print('2,500 L/s:', litres_to_m3s(2500), 'm³/s')
assert litres_to_m3s(2500) == 2.5
```
:::

:::{admonition} Quiz 4 — Convert generation units
:class: note

<span class="difficulty-badge difficulty-easy">Easy</span>

**Reference:** [@jica2011; @ifc2015].

Convert 14,500 kWh to MWh using a function. Explain why the answer is not in MW.

:::

:::{admonition} Solution
:class: tip, dropdown

Both units measure energy.

```{code-cell} python
def kwh_to_mwh(energy):
    return energy / 1000

print('14,500 kWh:', kwh_to_mwh(14500), 'MWh')
assert kwh_to_mwh(14500) == 14.5
```
:::

:::{admonition} Quiz 5 — Write and test a function
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

Write `hydro_power_mw(flow, net_head, efficiency=0.9, rated_mw=None)`. Reject negative values and invalid efficiency, accept flow and head arrays with scalar efficiency and rating, and clip to the optional rating. For paired records, align labels first and pass matching one-dimensional arrays in the same order; the model pairs values by position and requires unique, nonmissing and matching pandas indexes when inputs are labeled. If a labeled series is paired with an unlabeled array, check their order yourself; the helper cannot verify that pairing. Test zero flow and a clipped case.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
import numpy as np

def hydro_power_mw(flow, net_head, efficiency=0.9, rated_mw=None):
    """Pair by position; labeled inputs must already have matching indexes."""
    indexes = [getattr(value, "index", None) for value in (flow, net_head)]
    indexes = [index for index in indexes if index is not None and not callable(index)
               and callable(getattr(index, "equals", None))]
    for index in indexes:
        if hasattr(index, "is_unique") and not index.is_unique:
            raise ValueError("Labeled inputs must have unique, nonmissing indexes")
        if callable(getattr(index, "to_frame", None)):
            name = list(range(index.nlevels)) if index.nlevels > 1 else 0
            if index.to_frame(index=False, name=name).isna().to_numpy().any():
                raise ValueError("Labeled inputs must have unique, nonmissing indexes")
    if len(indexes) > 1 and any(not indexes[0].equals(index) for index in indexes[1:]):
        raise ValueError("Labeled inputs must have matching indexes in the same order")
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [flow, net_head, efficiency]):
        raise ValueError("model inputs must be finite")
    flow = np.asarray(flow, dtype=float)
    head = np.asarray(net_head, dtype=float)
    if (flow.ndim > 1 or head.ndim > 1 or
            (flow.ndim == head.ndim == 1 and flow.shape != head.shape)):
        raise ValueError("flow and head must be scalars or matching 1-D arrays")
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

print('Zero-flow and capped electrical power (MW):', hydro_power_mw(0, 50), hydro_power_mw(100, 80, rated_mw=40))
assert hydro_power_mw(0, 50) == 0
assert hydro_power_mw(100, 80, rated_mw=40) == 40
```
:::

:::{admonition} Quiz 6 — Environmental flow and plant limit
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

For river flows `[8, 15, 25, 40, 65, 90]` m$^3$/s, aim to release 12 m$^3$/s as environmental flow to a separate bypass reach upstream of the turbine return, releasing at most the available river flow. Report any target shortfall and the flow available afterward. Calculate plant power at 50 m net head, 88% combined turbine–generator efficiency, and 25 MW rating; explain why electrical clipping alone does not determine turbine flow or bypass. Print a table.

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
river_flow = np.array([8, 15, 25, 40, 65, 90], dtype=float)
release = np.minimum(river_flow, 12)
shortfall = 12 - release
available_flow = river_flow - release
plant_power = hydro_power_mw(available_flow, 50, 0.88, rated_mw=25)

print("river  release  shortfall  available  power")
for river, env, deficit, available, power in zip(river_flow, release, shortfall, available_flow, plant_power):
    print(f"{river:5.0f}  {env:7.0f}  {deficit:9.0f}  {available:9.0f}  {power:5.2f} MW")
print("Available flow is not necessarily turbine flow: electrical clipping alone does not specify bypass.")
assert np.all(plant_power >= 0) and np.all(plant_power <= 25)
```
:::

:::{admonition} Quiz 7 — Unequal time steps
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

A sensor records interval-mean powers `[10, 12, 14, 8]` MW for intervals lasting `[0.25, 0.25, 0.5, 1.0]` hours. Calculate energy correctly and compare it with the incorrect unweighted sum.

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

:::{admonition} Quiz 8 — Flow-duration and power-duration curves
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

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

:::{admonition} Quiz 9 — Derate for head loss
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@jica2011; @ifc2015].

Write a function for net head floored at zero. Evaluate gross head 18 m with head losses 1, 7 and 20 m.

:::

:::{admonition} Solution
:class: tip, dropdown

The floor describes zero generation when the assumed loss exhausts gross head.

```{code-cell} python
def net_head(gross, loss):
    if gross < 0 or loss < 0:
        raise ValueError('Head and loss must be non-negative')
    return max(gross-loss,0)

print('Net heads (m):', [net_head(18, loss) for loss in [1,7,20]])
assert [net_head(18, loss) for loss in [1,7,20]] == [17,11,0]
```
:::

:::{admonition} Quiz 10 — Water accounting at the intake
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@jica2011; @ifc2015].

Build a function that returns environmental release, turbine flow and bypass. For river flows of 15 and 2 m³/s, a 4 m³/s environmental target and an 8 m³/s turbine limit, verify the balance and report any environmental shortfall. Treat the target as a release to a separate bypass reach upstream of the turbine return; water allocated to that release does not pass through the turbine. The target is hypothetical, and actual compliance depends on the applicable location and rules.

:::

:::{admonition} Solution
:class: tip, dropdown

Reserve the release before allocating water to the turbine.

```{code-cell} python
def allocate(flow, reserve, limit):
    if not all(np.isfinite(x) and x >= 0 for x in (flow, reserve, limit)):
        raise ValueError("Flow, reservation and limit must be finite and non-negative")
    release = min(flow,reserve)
    turbine = min(flow-release,limit)
    return release,turbine,flow-release-turbine

for flow in (15, 2):
    release, turbine, bypass = allocate(flow, 4, 8)
    shortfall = 4 - release
    print(f"River {flow} m³/s: release {release}, turbine {turbine}, bypass {bypass}; environmental shortfall {shortfall} m³/s")
    assert np.isclose(release + turbine + bypass, flow) and min(release, turbine, bypass) >= 0
assert allocate(15,4,8) == (4,8,3)
assert allocate(2,4,8) == (2,0,0)
```
:::

:::{admonition} Quiz 11 — Infer turbine efficiency
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@jica2011; @ifc2015].

At 5.5 m³/s and 22 m net head, calculate the combined turbine–generator efficiency implied by 900 kW electrical output. Check that it is physically possible.

:::

:::{admonition} Solution
:class: tip, dropdown

Convert kW to W before dividing by hydraulic power in W.

```{code-cell} python
def efficiency(power_kw,flow,head):
    return power_kw*1000/(1000*9.81*flow*head)

print('Implied efficiency:', efficiency(900,5.5,22))
assert .75 < efficiency(900,5.5,22) < .76
```
:::

:::{admonition} Quiz 12 — Calculate turbine throughput per MWh
:class: note

<span class="difficulty-badge difficulty-medium">Medium</span>

**Reference:** [@jica2011; @ifc2015].

For a 2 MW turbine taking 9 m³/s, calculate turbine throughput in m³ per MWh. This flow passing through the turbine is not a measure of consumptive water use or a water footprint. Evaluate 45-minute and 2-hour intervals with a reusable function.

:::

:::{admonition} Solution
:class: tip, dropdown

Duration cancels under steady conditions.

```{code-cell} python
def intensity(flow,power,hours):
    volume=flow*hours*3600
    return volume/(power*hours)

for hours in (.75, 2):
    print(f"Turbine throughput over {hours:g} h: {intensity(9,2,hours):.0f} m³/MWh")
assert intensity(9,2,.75) == intensity(9,2,2) == 16200
```
:::

:::{admonition} Quiz 13 — Monthly energy and capacity factor
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

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
print(f"Annual energy: {annual_gwh:.3f} GWh; capacity factor: {capacity_factor:.1%}")
plt.title(f"2024 hydropower energy; CF = {capacity_factor:.1%}")
plt.tight_layout()
assert 0 <= capacity_factor <= 1
```
:::

:::{admonition} Quiz 14 — Pumped-storage comparison
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** JICA, Chapter 3 and Chapters 7–8 [@jica2011]; IFC, Chapter 7 [@ifc2015].

For 2.5 GWh pumping input and round-trip efficiency from 0.70 to 0.88, create a function returning recovered energy and loss. Plot both. Which is larger, pumping input or recovered energy?

:::

:::{admonition} Solution
:class: tip, dropdown

```{code-cell} python
def storage_cycle(input_gwh, round_trip_efficiency):
    eta = np.asarray(round_trip_efficiency, dtype=float)
    if not np.isfinite(input_gwh) or not np.isfinite(eta).all() or input_gwh < 0 or np.any((eta < 0) | (eta > 1)):
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

:::{admonition} Quiz 15 — Route a reservoir during a storm
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@jica2011; @ifc2015].

Start with 4,000 m³, a 12,000 m³ reservoir and one-hour inflows [0,10,0] m³/s. In this discrete teaching convention, each hour’s total inflow is available before that hour’s releases; it does not establish continuous-release feasibility. Prioritise 1 m³/s release, then up to 3 m³/s for generation. Verify water balance every hour and report spill and environmental shortfall.

:::

:::{admonition} Solution
:class: tip, dropdown

Within each step, the full interval inflow is credited before release and spill. A finer time step is needed to assess whether releases can be sustained as water arrives.

```{code-cell} python
def route(inflows):
    storage=4000
    spill_total=shortage_total=0
    for q in inflows:
        available=storage+q*3600
        release=min(3600,available)
        turbine=min(10800,available-release)
        rest=available-release-turbine
        spill=max(0,rest-12000)
        storage=rest-spill
        assert available==storage+release+turbine+spill
        spill_total+=spill
        shortage_total+=3600-release
    return storage,spill_total,shortage_total

print("Final storage, spill, environmental shortfall (m³):", route([0,10,0]))
assert route([0,10,0]) == (0,9600,0)
```
:::

:::{admonition} Quiz 16 — Search a turbine limit under low flows
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@jica2011; @ifc2015].

Compare turbine flow limits [4,6,8] m³/s for six flows [3,6,10,12,7,2] m³/s, each constant for one hour, with a 2 m³/s target for a separate bypass reach upstream of the turbine return, 30 m net head and 90% combined turbine–generator efficiency. Allocate the available environmental release before turbine flow. Select the smallest limit that captures 90% of unlimited electrical energy under the same release policy and check the next lower limit fails. The target is hypothetical; actual compliance depends on the applicable location and rules.

:::

:::{admonition} Solution
:class: tip, dropdown

Compare captured electrical energy against the same release policy in every scenario.

```{code-cell} python
flows=[3,6,10,12,7,2]
usable=[max(0,q-2) for q in flows]
def energy(limit):
    return sum(min(q,limit)*1000*9.81*30*.9/1e6 for q in usable)

target=.9*energy(float('inf'))
choice=next(limit for limit in [4,6,8] if energy(limit)>=target)
print(f"Selected limit {choice} m³/s; selected energy {energy(choice):.3f} MWh; unlimited {energy(float('inf')):.3f} MWh")
print(f"90% target {target:.3f} MWh; next lower limit 6 m³/s gives {energy(6):.3f} MWh and does not meet the target")
assert choice==8 and energy(6)<target
```
:::

:::{admonition} Quiz 17 — Recover flow and net head from measured power
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@jica2011; @ifc2015].

Given 1.5 MW electrical output, gross head 34 m and 90% combined turbine–generator efficiency, solve for flow on [0,15] m³/s when head loss is 0.03Q² m. Use bisection, check the point lies on the increasing branch, calculate net head and verify the residual.

:::

:::{admonition} Solution
:class: tip, dropdown

The bracket bounds a monotone operating region; the entire cubic need not be monotone.

```{code-cell} python
def output(q):
    return .9*1000*9.81*q*(34-.03*q*q)/1e6

lo,hi=0,15
assert 34-3*.03*hi**2>0 and output(lo)<1.5<output(hi)
for _ in range(40):
    mid=(lo+hi)/2
    if output(mid)<1.5:
        lo=mid
    else:
        hi=mid
flow=(lo+hi)/2
net_head=34-.03*flow**2
slope=.9*1000*9.81*(34-3*.03*flow**2)/1e6
residual=output(flow)-1.5
print(f"Flow: {flow:.3f} m³/s; net head: {net_head:.3f} m")
print(f"Power slope: {slope:.4f} MW/(m³/s); power residual: {residual:.3e} MW")
assert slope>0 and abs(residual)<1e-9
```
:::

:::{admonition} Quiz 18 — Compare policies over unequal intervals
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@jica2011; @ifc2015].

Model 3 intervals of [2,6,3] h with flows [3,12,5] m³/s constant within their respective intervals, 38 m net head, 85% combined turbine–generator efficiency and a 2 MW electrical cap. Compare environmental targets of 2 and 4 m³/s, returning energy and environmental shortfall volume; reconcile release, turbine water and bypass. Treat the target as a hypothetical release into a separate bypass reach upstream of the turbine return, so reserved water does not generate electricity. Any remaining flow above the turbine's capped flow also bypasses the turbine. Actual compliance depends on the applicable location and rules.

:::

:::{admonition} Solution
:class: tip, dropdown

Integrate volumes in seconds and energy in hours.

```{code-cell} python
import numpy as np
q=np.array([3,12,5]); dt=np.array([2,6,3])
def policy(reserve):
    release=np.minimum(q,reserve)
    factor=1000*9.81*38*.85/1e6
    turbine=np.minimum(q-release,2/factor)
    bypass=q-release-turbine
    assert np.allclose((release+turbine+bypass)*dt*3600,q*dt*3600)
    volumes = dict(inflow=np.dot(q,dt)*3600, release=np.dot(release,dt)*3600,
                   turbine=np.dot(turbine,dt)*3600, bypass=np.dot(bypass,dt)*3600)
    print(f"Target {reserve} m³/s water volumes (m³):", volumes)
    return np.dot(factor*turbine,dt),np.dot(reserve-release,dt)*3600

low,high=policy(2),policy(4)
print('Target 2 and 4: energy MWh, shortfall m³:', low, high)
assert low[0]>high[0] and low[1]==0 and high[1]==7200
```
:::

:::{admonition} Quiz 19 — Quantify missing-hour energy bounds
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@jica2011; @ifc2015].

Four one-hour interval-mean meter powers are [0.4, missing, 0.8, missing] MW. With a 1.5 MW limit, compute observed energy, coverage and defensible lower and upper bounds. Write a function that refuses readings above rating.

:::

:::{admonition} Solution
:class: tip, dropdown

The lower bound permits zero in a gap; it does not impute zero observations. If every hour is missing, the observed total is unavailable while the lower bound is still zero under the stated nonnegative-power assumption.

```{code-cell} python
import numpy as np
def energy_bounds(power,rating):
    p=np.asarray(power,dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(rating) or rating < 0:
        raise ValueError("Provide a non-empty series and finite non-negative rating")
    if np.isinf(p).any():
        raise ValueError("Infinite readings are invalid; use NaN for missing")
    seen=~np.isnan(p)
    if np.any((p[seen]<0)|(p[seen]>rating)):
        raise ValueError('Recorded power is outside rating')
    valid_sum=p[seen].sum()
    observed=valid_sum if seen.any() else np.nan
    return observed,seen.mean(),valid_sum+rating*(~seen).sum()

observed,coverage,upper=energy_bounds([.4,np.nan,.8,np.nan],1.5)
lower=observed if np.isfinite(observed) else 0.0
observed_label = "unavailable" if np.isnan(observed) else f"{observed:.3f} MWh"
print(f"Observed energy: {observed_label}; lower bound: {lower:.3f} MWh; coverage: {coverage:.0%}; upper bound: {upper:.3f} MWh")
assert np.allclose(energy_bounds([.4,np.nan,.8,np.nan],1.5),[1.2,.5,4.2])
```
:::

:::{admonition} Quiz 20 — Check pumped-storage arbitrage
:class: note

<span class="difficulty-badge difficulty-hard">Hard</span>

**Reference:** [@jica2011; @ifc2015].

Buy 6 MWh at €38/MWh, with pumping efficiency 0.86 and generating efficiency 0.9. Compare selling at €45 and €60 per MWh. Write a function returning recovered energy, margin and break-even sale price.

:::

:::{admonition} Solution
:class: tip, dropdown

The break-even sale price rises when round-trip efficiency falls; this example excludes other costs.

```{code-cell} python
def trade(energy,buy,sell):
    recovered=energy*.86*.9
    return recovered,recovered*sell-energy*buy,buy/(.86*.9)

low=trade(6,38,45); high=trade(6,38,60)
print('Recovered MWh, margin €, break-even €/MWh:', low, high)
assert low[1]<0<high[1] and 45<high[2]<60
```
:::
