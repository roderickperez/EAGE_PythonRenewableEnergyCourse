---
kernelspec:
  name: python3
  display_name: Python 3
---

# Geothermal Energy

## Learning goals

After this lesson you should be able to:

- distinguish geothermal gradient, heat flux, thermal power, gross electrical power, and net electrical power;
- use Kelvin correctly in thermodynamic efficiency calculations;
- estimate heat carried by produced fluid;
- include conversion efficiency, parasitic load, and production decline;
- implement validated Python functions and scenario plots;
- explain why temperature alone does not establish a viable geothermal resource.

Electricity generation requires useful heat plus a way to transport it to the surface. Conventional hydrothermal systems rely on heat, fluid, and permeability. Enhanced geothermal systems create or improve fluid pathways, while closed-loop systems circulate fluid in sealed wells [@doeGeothermalElectricity; @doeEGS]. Dry steam, flash steam, and binary-cycle plants use different surface processes and operating ranges.

## Concepts and equations

### Geothermal gradient is not heat flux

A linear temperature-depth approximation is

$$T(z)=T_0+G_Tz$$

where $G_T$ is the geothermal gradient, for example °C/km. This is a local approximation; gradients vary with geology, groundwater flow, and depth.

Conductive heat flux follows Fourier's law:

$$\mathbf{q}=-k\nabla T$$

The negative sign indicates heat flows toward lower temperature. Gradient (K/m) and heat flux (W/m$^2$) are different quantities.

### Thermal power in produced fluid

For mass-flow rate $\dot m$, specific heat $c_p$, production temperature $T_p$, and reinjection temperature $T_r$:

$$P_{th}=\dot m c_p(T_p-T_r)$$

If $c_p$ is J/(kg K), the result is W. A constant liquid-water $c_p$ is a simplified sensible-heat model; steam, flashing, brines, and phase change require enthalpy:

$$P_{th}=\dot m(h_p-h_r)$$

### Efficiency, parasitic load, and net power

The reversible Carnot bound is

$$\eta_C=1-\frac{T_c}{T_h}$$

where both temperatures must be in Kelvin. A plant cannot simply be assigned Carnot efficiency; real conversion efficiency is lower and technology-specific.

$$P_{gross}=\eta_{conv}P_{th}$$

$$P_{net}=P_{gross}-P_{pumps}-P_{aux}$$

Net power, not gross generator power, is the relevant exported power.

### Energy and decline

For discrete time steps:

$$E_{net}=\sum_iP_{net,i}\Delta t_i$$

A simple annual decline scenario is

$$P_y=P_1(1-d)^{y-1}$$

This is a scenario model, not a reservoir forecast. Sustainable production depends on reservoir response, injection strategy, pressure, thermal breakthrough, and operational constraints.

## Tested Python functions

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

def temperature_at_depth_c(depth_km, surface_c=15.0, gradient_c_per_km=30.0):
    depth = np.asarray(depth_km, dtype=float)
    if np.any(depth < 0) or gradient_c_per_km < 0:
        raise ValueError("depth and gradient must be non-negative")
    return surface_c + gradient_c_per_km * depth

def geothermal_net_power_mw(mass_flow_kgs, production_c, reinjection_c,
                            conversion_efficiency, parasitic_fraction=0.10,
                            cp_jkgk=4180.0):
    """Simplified single-phase sensible-heat model."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [mass_flow_kgs, production_c, reinjection_c, conversion_efficiency, parasitic_fraction, cp_jkgk]):
        raise ValueError("model inputs must be finite")
    mass_flow = np.asarray(mass_flow_kgs, dtype=float)
    delta_t = np.asarray(production_c, dtype=float) - reinjection_c
    if np.any(mass_flow < 0) or np.any(delta_t < 0):
        raise ValueError("mass flow and production-minus-reinjection temperature must be non-negative")
    if not 0 <= conversion_efficiency <= 1:
        raise ValueError("conversion_efficiency must be between 0 and 1")
    if not 0 <= parasitic_fraction < 1:
        raise ValueError("parasitic_fraction must be in [0, 1)")
    if cp_jkgk <= 0:
        raise ValueError("specific heat must be positive")
    thermal_mw = mass_flow * cp_jkgk * delta_t / 1e6
    gross_mw = thermal_mw * conversion_efficiency
    net_mw = gross_mw * (1 - parasitic_fraction)
    return float(net_mw) if np.ndim(net_mw) == 0 else net_mw

def carnot_efficiency(hot_c, cold_c):
    hot_k = np.asarray(hot_c, dtype=float) + 273.15
    cold_k = np.asarray(cold_c, dtype=float) + 273.15
    if np.any(hot_k <= cold_k) or np.any(cold_k <= 0):
        raise ValueError("hot temperature must exceed cold temperature in Kelvin")
    return 1 - cold_k / hot_k

assert temperature_at_depth_c(0) == 15
assert geothermal_net_power_mw(0, 160, 70, 0.12) == 0
assert 0 < carnot_efficiency(150, 25) < 1
```

## Worked well-field example

```{code-cell} python
years = np.arange(1, 31)
initial_mass_flow = 90.0
decline_rates = [0.00, 0.01, 0.02]

fig, ax = plt.subplots(figsize=(9, 4.5))
for decline in decline_rates:
    flow = initial_mass_flow * (1 - decline) ** (years - 1)
    net_power = geothermal_net_power_mw(
        flow, production_c=165, reinjection_c=70,
        conversion_efficiency=0.13, parasitic_fraction=0.12,
    )
    ax.plot(years, net_power, marker="o", markersize=3,
            label=f"{decline:.0%} annual flow decline")

ax.set(xlabel="Operating year", ylabel="Net power (MW)",
       title="Simplified geothermal decline scenarios")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
```

The plot varies only mass flow. In a real reservoir, temperature, pressure, scaling, downtime, reinjection response, and parasitic pumping may also change.


[Download this complete chapter as a Jupyter notebook](geothermalEnergy.ipynb). In standard Jupyter viewers, answer headings and code are visible below each prompt; the book provides collapse controls.

## Chapter practice — 20 Python exercises

**5 easy · 10 medium · 5 hard.** Work through the exercises in order. All inputs are synthetic teaching data. Each solution is directly below its question and starts collapsed in the book. Open it after trying your own code. Each solution runs independently; run the full notebook from the first cell when studying the chapter. References identify the underlying concepts rather than copied textbook problems.

(geothermal-exercise-01)=
### Exercise 01 — Temperature at depth

**Difficulty:** Easy

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Surface temperature 15 °C and gradient 30 °C/km: calculate temperature at 2 km with a linear model.

```{code-cell} python
# Your solution for geothermal exercise 01.
```

::::{dropdown} Step-by-step answer — Geothermal 01

1. Store depth and gradient with matching kilometre units.
2. Add the gradient-induced temperature increase.
3. Check and label the estimate.

```{code-cell} python
# Step 1: Store depth and gradient with matching kilometre units.
surface,gradient,depth=15,30,2
# Step 2: Add the gradient-induced temperature increase.
temperature=surface+gradient*depth
# Step 3: Check and label the estimate.
print(temperature,'°C'); assert temperature==75
```

**Interpretation:** A linear conductive gradient is not a prediction of a convective reservoir.

::::

(geothermal-exercise-02)=
### Exercise 02 — Heat flow in produced water

**Difficulty:** Easy

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Water flow 50 kg/s cools from 150 to 70 °C. With cp 4180 J/(kg K), calculate thermal MW.

```{code-cell} python
# Your solution for geothermal exercise 02.
```

::::{dropdown} Step-by-step answer — Geothermal 02

1. Calculate the temperature difference.
2. Apply sensible heat flow and convert W to MW.
3. Check the hand calculation.

```{code-cell} python
# Step 1: Calculate the temperature difference.
delta=150-70
# Step 2: Apply sensible heat flow and convert W to MW.
thermal=50*4180*delta/1e6
# Step 3: Check the hand calculation.
print(thermal,'MW thermal'); assert abs(thermal-16.72)<1e-12
```

**Interpretation:** The model assumes single-phase liquid and constant heat capacity.

::::

(geothermal-exercise-03)=
### Exercise 03 — Gross and net electricity

**Difficulty:** Easy

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Thermal power 20 MW has gross electrical efficiency 12% and parasitic load 10% of gross electricity. Calculate gross and net MW.

```{code-cell} python
# Your solution for geothermal exercise 03.
```

::::{dropdown} Step-by-step answer — Geothermal 03

1. Convert thermal input into gross electricity.
2. Subtract the parasitic fraction of gross electricity.
3. Check the result and distinguish the two powers.

```{code-cell} python
# Step 1: Convert thermal input into gross electricity.
gross=20*0.12
# Step 2: Subtract the parasitic fraction of gross electricity.
net=gross*(1-0.10)
# Step 3: Check the result and distinguish the two powers.
print(gross,net,'MW gross/net'); assert abs(net-2.16)<1e-12
```

**Interpretation:** Parasitic fraction here applies to gross electrical power, not thermal input.

::::

(geothermal-exercise-04)=
### Exercise 04 — Availability-adjusted energy

**Difficulty:** Easy

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** A plant produces 3 MW net when available, with 95% availability in a 365-day year. Calculate GWh.

```{code-cell} python
# Your solution for geothermal exercise 04.
```

::::{dropdown} Step-by-step answer — Geothermal 04

1. Calculate expected operating hours.
2. Integrate net power and convert to GWh.
3. Check the numerical result.

```{code-cell} python
# Step 1: Calculate expected operating hours.
hours=365*24*0.95
# Step 2: Integrate net power and convert to GWh.
energy=3*hours/1000
# Step 3: Check the numerical result.
print(energy,'GWh'); assert abs(energy-24.966)<1e-12
```

**Interpretation:** This assumes full net output whenever available.

::::

(geothermal-exercise-05)=
### Exercise 05 — Kelvin and temperature differences

**Difficulty:** Easy

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Convert 150 and 70 °C to kelvin, then show that the temperature difference is unchanged.

```{code-cell} python
# Your solution for geothermal exercise 05.
```

::::{dropdown} Step-by-step answer — Geothermal 05

1. Store Celsius temperatures.
2. Add273.15 to both absolute temperatures.
3. Check that the difference is80 K.

```{code-cell} python
# Step 1: Store Celsius temperatures.
hot,cold=150,70
# Step 2: Add273.15 to both absolute temperatures.
hot_k,cold_k=hot+273.15,cold+273.15
# Step 3: Check that the difference is80 K.
print(hot_k,cold_k,'K; difference:',hot_k-cold_k,'K')
assert abs((hot_k-cold_k)-(hot-cold))<1e-10
```

**Interpretation:** Use absolute kelvin temperatures in thermodynamic ratios; Celsius differences are valid for sensible heat.

::::

(geothermal-exercise-06)=
### Exercise 06 — Depth-temperature table

**Difficulty:** Medium

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** At depths[1,2,3] km, compare gradients 25 and 35 °C/km with surface 15 °C. Build a pandas table.

```{code-cell} python
# Your solution for geothermal exercise 06.
```

::::{dropdown} Step-by-step answer — Geothermal 06

1. Store depths in a common unit.
2. Calculate each gradient scenario separately.
3. Print labelled values and check one scenario.

```{code-cell} python
import numpy as np
import pandas as pd
# Step 1: Store depths in a common unit.
depth=np.array([1,2,3])
# Step 2: Calculate each gradient scenario separately.
table=pd.DataFrame({f'{g} C_per_km':15+g*depth for g in [25,35]},index=depth)
# Step 3: Print labelled values and check one scenario.
print(table.rename_axis('Depth km')); assert table.iloc[-1,1]==120
```

**Interpretation:** Gradient uncertainty changes inferred temperature substantially at depth.

::::

(geothermal-exercise-07)=
### Exercise 07 — Compare well flows

**Difficulty:** Medium

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** For flows[30,50,70] kg/s, temperatures 150/70 °C, cp 4180, efficiency 0.12 and parasitic fraction 0.1, calculate net MW.

```{code-cell} python
# Your solution for geothermal exercise 07.
```

::::{dropdown} Step-by-step answer — Geothermal 07

1. Store well-flow cases.
2. Apply the same thermal and electrical assumptions.
3. Check proportionality to flow and display power.

```{code-cell} python
import numpy as np
# Step 1: Store well-flow cases.
flow=np.array([30,50,70])
# Step 2: Apply the same thermal and electrical assumptions.
net=flow*4180*(150-70)*0.12*0.9/1e6
# Step 3: Check proportionality to flow and display power.
print(net,'MW net'); assert np.allclose(net/net[0],flow/flow[0])
```

**Interpretation:** Pump demand is represented only by a fixed fractional loss in this model.

::::

(geothermal-exercise-08)=
### Exercise 08 — Reinjection temperature sensitivity

**Difficulty:** Medium

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** At 50 kg/s and production 160 °C, compare reinjection[60,80,100]°C with cp 4180, efficiency 0.12 and parasitic 0.1.

```{code-cell} python
# Your solution for geothermal exercise 08.
```

::::{dropdown} Step-by-step answer — Geothermal 08

1. Store reinjection-temperature alternatives.
2. Calculate thermal drawdown and net electrical power.
3. Verify the direction of the simplified sensitivity.

```{code-cell} python
import numpy as np
# Step 1: Store reinjection-temperature alternatives.
reinjection=np.array([60,80,100])
# Step 2: Calculate thermal drawdown and net electrical power.
net=50*4180*(160-reinjection)*0.12*0.9/1e6
# Step 3: Verify the direction of the simplified sensitivity.
print(net,'MW net'); assert np.all(np.diff(net)<0)
```

**Interpretation:** Lower reinjection temperatures may increase modeled heat recovery but create chemical and reservoir constraints.

::::

(geothermal-exercise-09)=
### Exercise 09 — Multiple well-field totals

**Difficulty:** Medium

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Two wells have flows[40,60] kg/s and production[150,170]°C. Reinjection is 70 °C, cp 4180, conversion 0.12 and parasitic 0.1. Calculate individual and total net MW.

```{code-cell} python
# Your solution for geothermal exercise 09.
```

::::{dropdown} Step-by-step answer — Geothermal 09

1. Align each well flow with its production temperature.
2. Calculate and sum net well contributions.
3. Compare the total against direct thermal accounting.

```{code-cell} python
import numpy as np
# Step 1: Align each well flow with its production temperature.
flow=np.array([40,60]); temperature=np.array([150,170])
# Step 2: Calculate and sum net well contributions.
net=flow*4180*(temperature-70)*0.12*0.9/1e6
# Step 3: Compare the total against direct thermal accounting.
print(net,'MW by well;',net.sum(),'MW total')
assert np.isclose(net.sum(),4180*(40*80+60*100)*0.12*0.9/1e6)
```

**Interpretation:** Adding well powers assumes the shared plant can accept all of the available resource.

::::

(geothermal-exercise-10)=
### Exercise 10 — Parasitic-load scenarios

**Difficulty:** Medium

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Gross output is 5 MW. Compare parasitic fractions[0.05,0.10,0.20] and calculate net MW and 24-hour MWh.

```{code-cell} python
# Your solution for geothermal exercise 10.
```

::::{dropdown} Step-by-step answer — Geothermal 10

1. Store alternative operating-loss assumptions.
2. Deduct each parasitic fraction from gross output.
3. Check ordering and label both quantities.

```{code-cell} python
import numpy as np
# Step 1: Store alternative operating-loss assumptions.
fraction=np.array([0.05,0.10,0.20])
# Step 2: Deduct each parasitic fraction from gross output.
net=5*(1-fraction); energy=net*24
# Step 3: Check ordering and label both quantities.
print(net,'MW;',energy,'MWh'); assert np.all(np.diff(net)<0)
```

**Interpretation:** Parasitic demand includes pumps and auxiliaries; a constant fraction simplifies their operating behavior.

::::

(geothermal-exercise-11)=
### Exercise 11 — Thermodynamic upper bound

**Difficulty:** Medium

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Compute Carnot efficiency for hot 150 °C and cold 25 °C using kelvin. Compare a 12% conversion assumption.

```{code-cell} python
# Your solution for geothermal exercise 11.
```

::::{dropdown} Step-by-step answer — Geothermal 11

1. Convert both reservoir temperatures to kelvin.
2. Calculate the ideal reversible-engine efficiency.
3. Verify the illustrative actual conversion is below the bound.

```{code-cell} python
# Step 1: Convert both reservoir temperatures to kelvin.
hot=150+273.15; cold=25+273.15
# Step 2: Calculate the ideal reversible-engine efficiency.
carnot=1-cold/hot
# Step 3: Verify the illustrative actual conversion is below the bound.
print(carnot,'Carnot fraction; assumed actual:',0.12); assert 0<0.12<carnot<1
```

**Interpretation:** Carnot is an upper bound between fixed-temperature reservoirs, not a practical geothermal cycle model.

::::

(geothermal-exercise-12)=
### Exercise 12 — Conductive heat flux

**Difficulty:** Medium

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Use thermal conductivity 2.5 W/(m K) and gradient 30 K/km. Calculate upward conductive heat-flux magnitude W/m².

```{code-cell} python
# Your solution for geothermal exercise 12.
```

::::{dropdown} Step-by-step answer — Geothermal 12

1. Convert the temperature gradient to K per metre.
2. Apply the magnitude form of Fourier conduction.
3. Check units and numerical value.

```{code-cell} python
# Step 1: Convert the temperature gradient to K per metre.
gradient=30/1000
# Step 2: Apply the magnitude form of Fourier conduction.
flux=2.5*gradient
# Step 3: Check units and numerical value.
print(flux,'W/m²'); assert abs(flux-0.075)<1e-12
```

**Interpretation:** The magnitude is75 mW/m². Heat flux requires conductivity; a temperature gradient alone is not flux.

::::

(geothermal-exercise-13)=
### Exercise 13 — Compounded net-power decline

**Difficulty:** Medium

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** A 5 MW net plant declines 2% per year. Calculate years 1–10 and annual GWh at 95% availability.

```{code-cell} python
# Your solution for geothermal exercise 13.
```

::::{dropdown} Step-by-step answer — Geothermal 13

1. Start the decline exponent at zero for year1.
2. Calculate annual net power and availability-adjusted energy.
3. Check first-year energy and decreasing power.

```{code-cell} python
import numpy as np
# Step 1: Start the decline exponent at zero for year1.
years=np.arange(1,11)
# Step 2: Calculate annual net power and availability-adjusted energy.
power=5*0.98**(years-1); energy=power*8760*0.95/1000
# Step 3: Check first-year energy and decreasing power.
print(energy,'GWh/year'); assert np.isclose(energy[0],41.61) and np.all(np.diff(power)<0)
```

**Interpretation:** A prescribed decline curve is a scenario, not a calibrated reservoir forecast.

::::

(geothermal-exercise-14)=
### Exercise 14 — Invalid well records

**Difficulty:** Medium

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Flows[40,−2,50] kg/s and production[150,160,60]°C have reinjection 70 °C. Retain only finite positive-flow records hotter than reinjection; report exclusions.

```{code-cell} python
# Your solution for geothermal exercise 14.
```

::::{dropdown} Step-by-step answer — Geothermal 14

1. Preserve aligned well observations.
2. Construct a physical-validity mask before calculating heat.
3. Report excluded records and verify the retained count.

```{code-cell} python
import numpy as np
# Step 1: Preserve aligned well observations.
flow=np.array([40,-2,50]); production=np.array([150,160,60])
# Step 2: Construct a physical-validity mask before calculating heat.
valid=np.isfinite(flow)&np.isfinite(production)&(flow>0)&(production>70)
thermal=flow[valid]*4180*(production[valid]-70)/1e6
# Step 3: Report excluded records and verify the retained count.
print('Excluded indices:',np.where(~valid)[0],'; thermal MW:',thermal)
assert valid.sum()==1 and np.isclose(thermal[0],13.376)
```

**Interpretation:** Excluded values need investigation; silently converting negative heat to positive power would hide errors.

::::

(geothermal-exercise-15)=
### Exercise 15 — Plot decline scenarios

**Difficulty:** Medium

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Compare initial 5 MW with annual decline 0%,1%,3% over 20 years and 95% availability. Plot annual GWh and print cumulative values.

```{code-cell} python
# Your solution for geothermal exercise 15.
```

::::{dropdown} Step-by-step answer — Geothermal 15

1. Create the year axis and initialise a labelled plot.
2. Model each prescribed decline path and retain total energy.
3. Check cumulative ordering and show units and legend.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Create the year axis and initialise a labelled plot.
years=np.arange(1,21); totals=[]; fig,ax=plt.subplots(figsize=(6,3))
# Step 2: Model each prescribed decline path and retain total energy.
for rate in [0,0.01,0.03]:
    energy=5*(1-rate)**(years-1)*8760*0.95/1000
    totals.append(energy.sum()); ax.plot(years,energy,label=f'{rate:.0%} decline')
# Step 3: Check cumulative ordering and show units and legend.
ax.set(xlabel='Operating year',ylabel='Annual energy (GWh)',title='Synthetic geothermal scenarios'); ax.legend(); fig.tight_layout(); plt.show()
print(totals,'GWh cumulative'); assert np.all(np.diff(totals)<0)
```

**Interpretation:** Long-run scenarios are sensitive to decline assumptions and intervention choices.

::::

(geothermal-exercise-16)=
### Exercise 16 — Fit decline without future leakage

**Difficulty:** Hard

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Create ten annual powers 5×0.98^(year−1) MW. Fit log(power) on years 1–7 only, predict 8–10, and compare MAE against the last training observation.

```{code-cell} python
# Your solution for geothermal exercise 16.
```

::::{dropdown} Step-by-step answer — Geothermal 16

1. Generate the synthetic series and split chronologically.
2. Estimate log-linear parameters using training data only.
3. Evaluate holdout predictions and recover the known decline rate.

```{code-cell} python
import numpy as np
# Step 1: Generate the synthetic series and split chronologically.
year=np.arange(1,11); power=5*0.98**(year-1); split=7
# Step 2: Estimate log-linear parameters using training data only.
slope,intercept=np.polyfit(year[:split]-1,np.log(power[:split]),1)
prediction=np.exp(intercept+slope*(year[split:]-1))
# Step 3: Evaluate holdout predictions and recover the known decline rate.
mae=np.abs(prediction-power[split:]).mean(); baseline=np.abs(power[split:]-power[split-1]).mean()
print('Annual decline:',1-np.exp(slope),'MAE/baseline MW:',mae,baseline)
assert np.isclose(1-np.exp(slope),0.02) and mae<baseline
```

**Interpretation:** Exact recovery occurs because the noiseless observations follow the fitted model by construction.

::::

(geothermal-exercise-17)=
### Exercise 17 — Well field with plant clipping

**Difficulty:** Hard

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Three wells have flows[40,50,60] kg/s and production[150,160,170]°C. Use reinjection 70 °C, cp 4180, conversion 0.12, parasitic 0.1 and a 4 MW net plant cap. Compare all operating versus outage of each well.

```{code-cell} python
# Your solution for geothermal exercise 17.
```

::::{dropdown} Step-by-step answer — Geothermal 17

1. Calculate each well potential net contribution.
2. Apply the shared plant limit after summing operating wells.
3. Verify outages cannot increase output and compare with baseline.

```{code-cell} python
import numpy as np
# Step 1: Calculate each well potential net contribution.
flow=np.array([40,50,60]); temp=np.array([150,160,170])
potential=flow*4180*(temp-70)*0.12*0.9/1e6
# Step 2: Apply the shared plant limit after summing operating wells.
all_on=min(potential.sum(),4)
outage=[min(potential.sum()-p,4) for p in potential]
# Step 3: Verify outages cannot increase output and compare with baseline.
print('Baseline MW:',all_on,'single-well outages MW:',outage)
assert all_on==4 and all(0<=p<=all_on for p in outage)
```

**Interpretation:** Clipping can hide some well loss. This allocation ignores interactions between wells and shared equipment.

::::

(geothermal-exercise-18)=
### Exercise 18 — Net output with pumping penalty

**Difficulty:** Hard

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Compare flows 20–100 kg/s in steps 10. Gross MW=m×4180×80×0.12/1e 6 and pump MW=0.00004 m². Find the discrete flow maximising net power.

```{code-cell} python
# Your solution for geothermal exercise 18.
```

::::{dropdown} Step-by-step answer — Geothermal 18

1. Define candidate flow rates and a synthetic pump-load model.
2. Calculate gross and net power consistently in MW.
3. Select the best candidate and verify the energy subtraction.

```{code-cell} python
import numpy as np
# Step 1: Define candidate flow rates and a synthetic pump-load model.
flow=np.arange(20,101,10); pump=0.00004*flow**2
# Step 2: Calculate gross and net power consistently in MW.
gross=flow*4180*80*0.12/1e6; net=gross-pump
# Step 3: Select the best candidate and verify the energy subtraction.
best=np.argmax(net)
print('Best flow:',flow[best],'kg/s; net:',net[best],'MW')
assert np.allclose(net+pump,gross) and flow[best]==100
```

**Interpretation:** The best candidate is the upper search boundary, not an established physical optimum; do not extrapolate the pump model.

::::

(geothermal-exercise-19)=
### Exercise 19 — Lifetime energy with an intervention

**Difficulty:** Hard

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** Starting at 5 MW, power declines 3% per year through year 10. At the start of year 11 restore power to 5 MW and then decline 3% again through year 20. Compare cumulative GWh with uninterrupted decline;95% availability.

```{code-cell} python
# Your solution for geothermal exercise 19.
```

::::{dropdown} Step-by-step answer — Geothermal 19

1. Calculate the uninterrupted reference trajectory.
2. Reset the age exponent at the intervention year.
3. Integrate the incremental generation and verify the reset.

```{code-cell} python
import numpy as np
# Step 1: Calculate the uninterrupted reference trajectory.
year=np.arange(1,21); baseline=5*0.97**(year-1)
# Step 2: Reset the age exponent at the intervention year.
age=np.where(year<=10,year-1,year-11)
intervention=5*0.97**age
# Step 3: Integrate the incremental generation and verify the reset.
gain=np.sum(intervention-baseline)*8760*0.95/1000
print(gain,'GWh additional'); assert intervention[10]==5 and gain>0
```

**Interpretation:** Intervention cost, downtime and reservoir feasibility are omitted; additional energy is not automatically economic benefit.

::::

(geothermal-exercise-20)=
### Exercise 20 — Joint uncertainty in flow and temperature

**Difficulty:** Hard

**Reference:** Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.

**Task:** With seed 5 sample 2000 independent flows uniform 40–60 kg/s and production temperatures uniform 140–170 °C. Use reinjection 70 °C, cp 4180, eta 0.12, parasitic 0.1. Report net-power median,5th/95th percentiles and probability above 2 MW.

```{code-cell} python
# Your solution for geothermal exercise 20.
```

::::{dropdown} Step-by-step answer — Geothermal 20

1. State the synthetic uncertainty distributions and independence assumption.
2. Propagate paired samples through the net-power model.
3. Summarise outcomes and check the physical range.

```{code-cell} python
import numpy as np
# Step 1: State the synthetic uncertainty distributions and independence assumption.
rng=np.random.default_rng(5); flow=rng.uniform(40,60,2000); temp=rng.uniform(140,170,2000)
# Step 2: Propagate paired samples through the net-power model.
net=flow*4180*(temp-70)*0.12*0.9/1e6
# Step 3: Summarise outcomes and check the physical range.
print('5th/median/95th MW:',np.quantile(net,[0.05,0.5,0.95]),'P(net>2):',np.mean(net>2))
assert np.all(net>0) and np.all(net<=60*4180*100*0.12*0.9/1e6)
```

**Interpretation:** The distributions and independence are assumptions, not measured reservoir uncertainty or a confidence interval.

::::

## Common mistakes

- Using °C rather than Kelvin in a thermodynamic temperature ratio.
- Calling geothermal gradient a heat flux.
- Ignoring phase change when enthalpy, not constant $c_p\Delta T$, is required.
- Reporting gross power while omitting pumping and auxiliary loads.
- Treating a high subsurface temperature as proof of permeability, flow, or commercial viability.
- Subtracting a fixed decline percentage linearly instead of compounding the retained fraction.

## Check your understanding

Continue with the [geothermal quiz and Python challenge](../section7/renewableEnergyquizzes/geothermalEnergy.md).
