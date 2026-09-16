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


[Download this complete chapter as a Jupyter notebook](hydroelectricEnergy.ipynb). In standard Jupyter viewers, answer headings and code are visible below each prompt; the book provides collapse controls.

## Chapter practice — 20 Python exercises

**5 easy · 10 medium · 5 hard.** Work through the exercises in order. All inputs are synthetic teaching data. Each solution is directly below its question and starts collapsed in the book. Open it after trying your own code. Each solution runs independently; run the full notebook from the first cell when studying the chapter. References identify the underlying concepts rather than copied textbook problems.

(hydro-exercise-01)=
### Exercise 01 — Net head

**Difficulty:** Easy

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Gross head is 40 m and head loss 3 m. Calculate net head.

```{code-cell} python
# Your solution for hydro exercise 01.
```

::::{dropdown} Step-by-step answer — Hydro 01

1. Enter gross head and hydraulic losses.
2. Subtract losses before calculating useful head.
3. Check the head is positive and report it.

```{code-cell} python
# Step 1: Enter gross head and hydraulic losses.
gross, loss = 40, 3
# Step 2: Subtract losses before calculating useful head.
net = gross-loss
# Step 3: Check the head is positive and report it.
print(net,'m'); assert net == 37
```

**Interpretation:** Net head is 37 m; losses reduce rather than increase usable head.

::::

(hydro-exercise-02)=
### Exercise 02 — Electrical power

**Difficulty:** Easy

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Use Q=10 m³/s, net head 30 m, efficiency 0.9, density 1000 kg/m³ and g 9.81 m/s². Calculate MW.

```{code-cell} python
# Your solution for hydro exercise 02.
```

::::{dropdown} Step-by-step answer — Hydro 02

1. Store parameters in SI units.
2. Convert hydraulic power to electricity and MW.
3. Check a hand-calculated result.

```{code-cell} python
# Step 1: Store parameters in SI units.
q, head, eta, rho, g = 10,30,0.9,1000,9.81
# Step 2: Convert hydraulic power to electricity and MW.
power = rho*g*q*head*eta/1e6
# Step 3: Check a hand-calculated result.
print(power,'MW'); assert abs(power-2.6487)<1e-12
```

**Interpretation:** 2.6487 MW omits generator clipping.

::::

(hydro-exercise-03)=
### Exercise 03 — Daily hydro energy

**Difficulty:** Easy

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** A hydro generator averages 2 MW for 24 hours. Calculate MWh and kWh.

```{code-cell} python
# Your solution for hydro exercise 03.
```

::::{dropdown} Step-by-step answer — Hydro 03

1. Enter average power and duration.
2. Integrate and convert energy units.
3. Verify and print both units.

```{code-cell} python
# Step 1: Enter average power and duration.
power, hours = 2,24
# Step 2: Integrate and convert energy units.
mwh=power*hours; kwh=mwh*1000
# Step 3: Verify and print both units.
print(mwh,'MWh;',kwh,'kWh'); assert mwh==48 and kwh==48000
```

**Interpretation:** The power must be an interval average for this calculation.

::::

(hydro-exercise-04)=
### Exercise 04 — Environmental reservation

**Difficulty:** Easy

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** River flow is 8 m³/s and the environmental reservation is 3 m³/s. Calculate available turbine flow.

```{code-cell} python
# Your solution for hydro exercise 04.
```

::::{dropdown} Step-by-step answer — Hydro 04

1. Enter river flow and reservation.
2. Reserve water before assigning turbine flow.
3. Check non-negativity and report the result.

```{code-cell} python
# Step 1: Enter river flow and reservation.
river, reserve = 8,3
# Step 2: Reserve water before assigning turbine flow.
usable=max(river-reserve,0)
# Step 3: Check non-negativity and report the result.
print(usable,'m³/s'); assert usable==5
```

**Interpretation:** This simplified reservation is not a site-specific environmental-flow prescription.

::::

(hydro-exercise-05)=
### Exercise 05 — Water volume

**Difficulty:** Easy

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** A turbine passes 3 m³/s for 2 hours. Calculate water volume.

```{code-cell} python
# Your solution for hydro exercise 05.
```

::::{dropdown} Step-by-step answer — Hydro 05

1. Convert hours to seconds.
2. Integrate the constant flow rate.
3. Check and label the volume.

```{code-cell} python
# Step 1: Convert hours to seconds.
seconds=2*3600
# Step 2: Integrate the constant flow rate.
volume=3*seconds
# Step 3: Check and label the volume.
print(volume,'m³'); assert volume==21600
```

**Interpretation:** Flow is volume per second; multiplying by hours without conversion is incorrect.

::::

(hydro-exercise-06)=
### Exercise 06 — Flow to power table

**Difficulty:** Medium

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** For river flows [2,5,10,20] m³/s, reserve 3, use head 25 m, eta 0.9, and a 2 MW rating. Tabulate usable flow and electrical MW.

```{code-cell} python
# Your solution for hydro exercise 06.
```

::::{dropdown} Step-by-step answer — Hydro 06

1. Calculate usable flow for each river condition.
2. Apply the hydraulic model and generator cap.
3. Check dry and clipped cases and print a table.

```{code-cell} python
import numpy as np
import pandas as pd
# Step 1: Calculate usable flow for each river condition.
river=np.array([2,5,10,20]); usable=np.maximum(river-3,0)
# Step 2: Apply the hydraulic model and generator cap.
power=np.minimum(1000*9.81*usable*25*0.9/1e6,2)
# Step 3: Check dry and clipped cases and print a table.
print(pd.DataFrame({'river_m3s':river,'usable_m3s':usable,'power_mw':power}))
assert power[0]==0 and power[-1]==2
```

**Interpretation:** Capping power does not mean all available water must pass through the turbine.

::::

(hydro-exercise-07)=
### Exercise 07 — Generator flow limit

**Difficulty:** Medium

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Find maximum turbine flow for a 2 MW generator, net head 25 m and eta 0.9. Compare against available flow 12 m³/s.

```{code-cell} python
# Your solution for hydro exercise 07.
```

::::{dropdown} Step-by-step answer — Hydro 07

1. Rearrange P=rho*g*Q*H*eta for Q.
2. Limit turbine flow to equipment capacity.
3. Recalculate power as a check.

```{code-cell} python
# Step 1: Rearrange P=rho*g*Q*H*eta for Q.
q_limit=2e6/(1000*9.81*25*0.9)
# Step 2: Limit turbine flow to equipment capacity.
q_turbine=min(12,q_limit); bypass=12-q_turbine
# Step 3: Recalculate power as a check.
power=1000*9.81*q_turbine*25*0.9/1e6
print(q_limit,'m³/s limit;',bypass,'m³/s bypass'); assert abs(power-2)<1e-12
```

**Interpretation:** Bypass is water not used for generation; environmental flow has already been reserved.

::::

(hydro-exercise-08)=
### Exercise 08 — Quadratic head loss

**Difficulty:** Medium

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** At Q=[2,4,6] m³/s use gross head 30 m and loss=0.1Q² m. Calculate net head and MW at eta 0.9.

```{code-cell} python
# Your solution for hydro exercise 08.
```

::::{dropdown} Step-by-step answer — Hydro 08

1. Model the assumed flow-dependent loss.
2. Calculate head before calculating power.
3. Check head remains physically valid.

```{code-cell} python
import numpy as np
# Step 1: Model the assumed flow-dependent loss.
q=np.array([2,4,6]); loss=0.1*q**2
# Step 2: Calculate head before calculating power.
head=30-loss; power=1000*9.81*q*head*0.9/1e6
# Step 3: Check head remains physically valid.
print(head,'m;',power,'MW'); assert np.all(head>0) and np.all(head<30)
```

**Interpretation:** The loss coefficient is synthetic and needs hydraulic calibration for a real conduit.

::::

(hydro-exercise-09)=
### Exercise 09 — Monthly durations

**Difficulty:** Medium

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Mean output is 2 MW in January and February 2025. Use pandas calendar month lengths to calculate MWh for each month.

```{code-cell} python
# Your solution for hydro exercise 09.
```

::::{dropdown} Step-by-step answer — Hydro 09

1. Create the two calendar months.
2. Multiply average output by actual month hours.
3. Verify the January and non-leap February results.

```{code-cell} python
import pandas as pd
# Step 1: Create the two calendar months.
months=pd.date_range('2025-01-01',periods=2,freq='MS')
# Step 2: Multiply average output by actual month hours.
energy=2*24*months.days_in_month.to_numpy()
# Step 3: Verify the January and non-leap February results.
print(energy,'MWh'); assert list(energy)==[1488,1344]
```

**Interpretation:** Constant monthly mean output does not imply equal monthly energy.

::::

(hydro-exercise-10)=
### Exercise 10 — Flow-duration ranking

**Difficulty:** Medium

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Sort daily flows [8,2,10,4] descending and assign exceedance plotting positions rank/(n+1).

```{code-cell} python
# Your solution for hydro exercise 10.
```

::::{dropdown} Step-by-step answer — Hydro 10

1. Rank observed daily mean flows from largest to smallest.
2. Use the stated plotting-position convention.
3. Print paired values and check endpoints.

```{code-cell} python
import numpy as np
# Step 1: Rank observed daily mean flows from largest to smallest.
flow=np.sort([8,2,10,4])[::-1]
# Step 2: Use the stated plotting-position convention.
exceedance=np.arange(1,len(flow)+1)/(len(flow)+1)
# Step 3: Print paired values and check endpoints.
print(list(zip(exceedance,flow)))
assert np.all(np.diff(flow)<=0) and np.allclose(exceedance,[.2,.4,.6,.8])
```

**Interpretation:** A duration curve loses chronology and cannot directly route reservoir storage.

::::

(hydro-exercise-11)=
### Exercise 11 — Efficiency sensitivity

**Difficulty:** Medium

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** At Q 8 m³/s and net head 40 m, compare efficiencies 0.75,0.85,0.95. Print MW and the gain from lowest to highest.

```{code-cell} python
# Your solution for hydro exercise 11.
```

::::{dropdown} Step-by-step answer — Hydro 11

1. Define conversion-efficiency scenarios.
2. Keep resource inputs fixed across scenarios.
3. Verify linear sensitivity and report gain.

```{code-cell} python
import numpy as np
# Step 1: Define conversion-efficiency scenarios.
eta=np.array([0.75,0.85,0.95])
# Step 2: Keep resource inputs fixed across scenarios.
power=1000*9.81*8*40*eta/1e6
# Step 3: Verify linear sensitivity and report gain.
print(power,'MW; gain:',power[-1]-power[0],'MW')
assert np.allclose(power/power[0],eta/eta[0])
```

**Interpretation:** Real turbine efficiency varies with operating point; these are comparison assumptions.

::::

(hydro-exercise-12)=
### Exercise 12 — Pumped-storage round trip

**Difficulty:** Medium

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Store 100000 m³ through 100 m head. With pump efficiency 0.85 and generation efficiency 0.9, find gravitational, input and recovered MWh.

```{code-cell} python
# Your solution for hydro exercise 12.
```

::::{dropdown} Step-by-step answer — Hydro 12

1. Convert gravitational joules to MWh.
2. Account for pumping and generating conversion losses.
3. Check round-trip efficiency.

```{code-cell} python
import numpy as np
# Step 1: Convert gravitational joules to MWh.
stored=1000*9.81*100000*100/3.6e9
# Step 2: Account for pumping and generating conversion losses.
input_energy=stored/0.85; output=stored*0.9
# Step 3: Check round-trip efficiency.
print(stored,input_energy,output,'MWh stored, input, output')
assert np.isclose(output/input_energy,0.85*0.9)
```

**Interpretation:** Storage shifts electricity and consumes net energy; it is not a new primary-energy source.

::::

(hydro-exercise-13)=
### Exercise 13 — Validate resource readings

**Difficulty:** Medium

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** For river flows [3,−1,missing,5] m³/s, flag invalid readings and report valid fraction without filling gaps.

```{code-cell} python
# Your solution for hydro exercise 13.
```

::::{dropdown} Step-by-step answer — Hydro 13

1. Preserve the measured values and missing marker.
2. Replace only physically invalid negatives with missing.
3. Report coverage rather than a complete-period generation claim.

```{code-cell} python
import pandas as pd
# Step 1: Preserve the measured values and missing marker.
flow=pd.Series([3,-1,None,5],dtype=float)
# Step 2: Replace only physically invalid negatives with missing.
clean=flow.mask(flow.lt(0))
# Step 3: Report coverage rather than a complete-period generation claim.
print(clean,'coverage:',clean.notna().mean()); assert clean.notna().sum()==2
```

**Interpretation:** Do not infer zero flow from an absent measurement.

::::

(hydro-exercise-14)=
### Exercise 14 — Environmental shortage

**Difficulty:** Medium

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Daily flows[1,4,7] m³/s must first provide up to 3 m³/s environmental release. Calculate release, shortage and remaining turbine flow.

```{code-cell} python
# Your solution for hydro exercise 14.
```

::::{dropdown} Step-by-step answer — Hydro 14

1. Store daily mean flows and the reservation target.
2. Allocate available water before calculating shortages.
3. Reconcile water and print all three quantities.

```{code-cell} python
import numpy as np
# Step 1: Store daily mean flows and the reservation target.
river=np.array([1,4,7]); target=3
# Step 2: Allocate available water before calculating shortages.
release=np.minimum(river,target); shortage=target-release; turbine=river-release
# Step 3: Reconcile water and print all three quantities.
assert np.array_equal(release+turbine,river)
print('Release, shortage, turbine m³/s:',release,shortage,turbine)
assert np.array_equal(shortage,[2,0,0])
```

**Interpretation:** A shortage means the river cannot meet the target; no negative turbine flow is allowed.

::::

(hydro-exercise-15)=
### Exercise 15 — Plot a power-duration curve

**Difficulty:** Medium

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Daily flows[2,4,8,12,16] m³/s drive head 20 m, eta 0.9 and cap 2 MW with no reservation in this example. Plot ranked MW and calculate five-day MWh.

```{code-cell} python
# Your solution for hydro exercise 15.
```

::::{dropdown} Step-by-step answer — Hydro 15

1. Calculate daily average electrical powers.
2. Sort only for the duration plot; integrate all equal-duration samples.
3. Plot the stated exceedance convention and check conservation of the sum.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Calculate daily average electrical powers.
power=np.minimum(1000*9.81*np.array([2,4,8,12,16])*20*0.9/1e6,2)
# Step 2: Sort only for the duration plot; integrate all equal-duration samples.
ranked=np.sort(power)[::-1]; energy=power.sum()*24
# Step 3: Plot the stated exceedance convention and check conservation of the sum.
fig,ax=plt.subplots(figsize=(6,3)); ax.plot(np.arange(1,6)/6*100,ranked,'o-')
ax.set(xlabel='Exceedance plotting position (%)',ylabel='Power (MW)',title='Synthetic hydro duration curve'); fig.tight_layout(); plt.show()
print(energy,'MWh'); assert np.isclose(ranked.sum()*24,energy)
```

**Interpretation:** Sorting preserves total energy for equal intervals but removes the order needed for storage simulation.

::::

(hydro-exercise-16)=
### Exercise 16 — Reservoir routing

**Difficulty:** Hard

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Route hourly inflows[2,0,8,16,0,0] m³/s; initial storage 20000 m³ and maximum 50000. Prioritise 1 m³/s environment, then up to 3 turbine; spill excess. Use head 20 m and eta 0.9. Check every water balance.

```{code-cell} python
# Your solution for hydro exercise 16.
```

::::{dropdown} Step-by-step answer — Hydro 16

1. Set storage state and output records.
2. Allocate each hour of incoming water and spill overflow last.
3. Check total spill and final storage and report generated energy.

```{code-cell} python
import numpy as np
# Step 1: Set storage state and output records.
storage=20000.; rows=[]
# Step 2: Allocate each hour of incoming water and spill overflow last.
for q in [2,0,8,16,0,0]:
    before=storage; water=storage+q*3600
    environmental=min(water,3600); water-=environmental
    turbine=min(water,3*3600); water-=turbine
    spill=max(water-50000,0); storage=water-spill
    assert np.isclose(before+q*3600,storage+environmental+turbine+spill)
    energy=1000*9.81*turbine*20*0.9/3.6e9
    rows.append((storage,spill,energy,3600-environmental))
# Step 3: Check total spill and final storage and report generated energy.
print('Final storage:',storage,'m³; total generation:',sum(r[2] for r in rows),'MWh')
assert storage==21200 and sum(r[1] for r in rows)==7600
```

**Interpretation:** Within-hour inflow is immediately available; head is constant and evaporation is omitted.

::::

(hydro-exercise-17)=
### Exercise 17 — Storage-dependent head

**Difficulty:** Hard

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Route inflows[2,0,3] m³/s for three hours from 20000 m³. Withdraw up to 2 m³/s, no spill limit. Use beginning-of-step head H=10+S/10000 m and eta 0.9; compare with constant 12 m.

```{code-cell} python
# Your solution for hydro exercise 17.
```

::::{dropdown} Step-by-step answer — Hydro 17

1. Initialise storage and energy counters.
2. Use beginning-of-step head and limit withdrawal to available water.
3. Compare model outputs and verify the final storage.

```{code-cell} python
import numpy as np
# Step 1: Initialise storage and energy counters.
storage=20000.; variable_energy=0.; fixed_energy=0.
# Step 2: Use beginning-of-step head and limit withdrawal to available water.
for inflow in [2,0,3]:
    before=storage; head=10+before/10000
    available=before+inflow*3600; volume=min(2*3600,available)
    storage=available-volume
    variable_energy+=1000*9.81*volume*head*0.9/3.6e9
    fixed_energy+=1000*9.81*volume*12*0.9/3.6e9
    assert np.isclose(before+inflow*3600,storage+volume)
# Step 3: Compare model outputs and verify the final storage.
print(variable_energy,fixed_energy,'MWh variable/fixed head')
assert storage==16400 and variable_energy<fixed_energy
```

**Interpretation:** Beginning-of-step head is a numerical approximation; smaller time steps may be needed for rapid level changes.

::::

(hydro-exercise-18)=
### Exercise 18 — Environmental-flow trade-off

**Difficulty:** Hard

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** For daily river flows[2,5,8,12] m³/s, compare reservations 1,3,5 m³/s, head 30 m, eta 0.9, cap 2 MW. Calculate four-day energy and environmental-shortage volume.

```{code-cell} python
# Your solution for hydro exercise 18.
```

::::{dropdown} Step-by-step answer — Hydro 18

1. Keep the same river observations for every policy scenario.
2. Apply reservation, clipping and interval conversions.
3. Check the energy trade-off and report units.

```{code-cell} python
import numpy as np
# Step 1: Keep the same river observations for every policy scenario.
river=np.array([2,5,8,12]); results=[]
# Step 2: Apply reservation, clipping and interval conversions.
for reserve in [1,3,5]:
    usable=np.maximum(river-reserve,0)
    power=np.minimum(1000*9.81*usable*30*0.9/1e6,2)
    shortage=np.maximum(reserve-river,0).sum()*86400
    results.append((reserve,power.sum()*24,shortage))
# Step 3: Check the energy trade-off and report units.
print('Reservation m³/s, energy MWh, shortage m³:',results)
assert np.all(np.diff([r[1] for r in results])<=0)
```

**Interpretation:** The model compares water allocation, not ecological adequacy or regulatory compliance.

::::

(hydro-exercise-19)=
### Exercise 19 — Search a generator rating

**Difficulty:** Hard

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** Flows[2,4,8,12] m³/s last 24 h each, head 30 m and eta 0.9. Choose the smallest candidate rating[1,2,3,4] MW that captures 95% of uncapped energy.

```{code-cell} python
# Your solution for hydro exercise 19.
```

::::{dropdown} Step-by-step answer — Hydro 19

1. Calculate the uncapped electrical resource.
2. Evaluate retained energy for each candidate rating.
3. Select and verify the first feasible rating.

```{code-cell} python
import numpy as np
# Step 1: Calculate the uncapped electrical resource.
available=1000*9.81*np.array([2,4,8,12])*30*0.9/1e6
# Step 2: Evaluate retained energy for each candidate rating.
results=[(r,np.minimum(available,r).sum()/available.sum()) for r in [1,2,3,4]]
# Step 3: Select and verify the first feasible rating.
best=next(row for row in results if row[1]>=0.95)
print('MW rating, retained energy fraction:',best); assert best[0]==3
```

**Interpretation:** This is an energy-capture criterion; construction cost and seasonal hydrology are not represented.

::::

(hydro-exercise-20)=
### Exercise 20 — Pumped-storage dispatch and water balance

**Difficulty:** Hard

**Reference:** Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.

**Task:** An upper reservoir starts empty with capacity 10000 m³ and head 100 m. Pump with 1 MW for two hours at 85%, then generate for two hours at up to 0.7 MW and 90%. Track water and electricity.

```{code-cell} python
# Your solution for hydro exercise 20.
```

::::{dropdown} Step-by-step answer — Hydro 20

1. Set state and energy totals at constant head.
2. Limit pumping by space and generation by available water.
3. Account for remaining stored energy as well as conversion losses.

```{code-cell} python
import numpy as np
# Step 1: Set state and energy totals at constant head.
storage=0.; used=0.; recovered=0.; joules_per_m3=1000*9.81*100
# Step 2: Limit pumping by space and generation by available water.
for action in ['pump','pump','generate','generate']:
    before=storage
    if action=='pump':
        volume=min(1e6*3600*0.85/joules_per_m3,10000-storage)
        storage+=volume; used+=volume*joules_per_m3/0.85/3.6e9
    else:
        volume=min(0.7e6*3600/(joules_per_m3*0.9),storage)
        storage-=volume; recovered+=volume*joules_per_m3*0.9/3.6e9
    assert 0<=storage<=10000 and abs(storage-before)<=10000
# Step 3: Account for remaining stored energy as well as conversion losses.
stored_equivalent=storage*joules_per_m3*0.9/3.6e9
print(used,recovered,stored_equivalent,'MWh pumping, exported, recoverable remaining')
assert np.isclose(recovered+stored_equivalent,used*0.85*0.9)
```

**Interpretation:** The final reservoir need not be empty; recovered/pumped energy over a partial cycle is not full round-trip efficiency.

::::

## Common mistakes

- Using gross head when the equation requires net head.
- Confusing efficiency with capacity factor.
- Multiplying MW by seconds and labelling the result MWh.
- Ignoring generator clipping or environmental-flow requirements.
- Applying capacity factor to power that is already a time average.
- Treating pumped storage as a source of net electricity.

## Check your understanding

Continue with the [hydropower quiz and Python challenge](../section7/renewableEnergyquizzes/hydroelectricEnergy.md).
