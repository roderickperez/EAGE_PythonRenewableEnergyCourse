---
kernelspec:
  name: python3
  display_name: Python 3
---

# Solar

[Review the theory and worked examples](../../section6/solarEnergy.md). [Download the exercise notebook](solar.ipynb).

## Chapter practice — 20 Python exercises

**4 Basic · 8 Medium · 8 Hard.** Basic exercises connect concepts, units and short calculations. Medium exercises apply functions and arrays. Hard exercises combine modelling, validation and interpretation. Each solution runs independently.

## Basic

(solar-exercise-01)=
### Exercise 01 — Panel DC power

**Difficulty:** Basic

**Reference:** PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3; [@goswami2015], §§9.2–9.2.2.

**Task:** At 800 W/m², a 2 m² panel has efficiency 20%. Calculate DC watts.

```{code-cell} python
# Your solution for solar exercise 01.
```

::::{dropdown} Step-by-step answer — Solar 01

1. Store irradiance, area and efficiency.
2. Multiply incident power by conversion efficiency.
3. Label and verify the result.

```{code-cell} python
# Step 1: Store irradiance, area and efficiency.
g, area, eta = 800, 2, 0.20
# Step 2: Multiply incident power by conversion efficiency.
power_w = g*area*eta
# Step 3: Label and verify the result.
print(power_w, 'W DC'); assert power_w == 320
```

**Interpretation:** 320 W assumes constant efficiency and uniform irradiance.

::::

(solar-exercise-02)=
### Exercise 02 — Array nameplate

**Difficulty:** Basic

**Reference:** PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3; [@goswami2015], §§9.2–9.2.2.

**Task:** An array contains 24 modules rated at 400 W DC. Calculate its STC rating in kW.

```{code-cell} python
# Your solution for solar exercise 02.
```

::::{dropdown} Step-by-step answer — Solar 02

1. Enter module count and rating.
2. Add module ratings and convert to kW.
3. Print and check the nameplate.

```{code-cell} python
# Step 1: Enter module count and rating.
count, module_w = 24, 400
# Step 2: Add module ratings and convert to kW.
array_kw = count*module_w/1000
# Step 3: Print and check the nameplate.
print(array_kw,'kW DC'); assert array_kw == 9.6
```

**Interpretation:** Nameplate describes STC; actual operating output varies.

::::

(solar-exercise-03)=
### Exercise 03 — Four-hour irradiation

**Difficulty:** Basic

**Reference:** Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3; [@goswami2015], §§9.2–9.2.2.

**Task:** Four hourly mean irradiances are [0, 200, 600, 400] W/m². Calculate four-hour irradiation in kWh/m².

```{code-cell} python
# Your solution for solar exercise 03.
```

::::{dropdown} Step-by-step answer — Solar 03

1. Store hourly means and duration.
2. Integrate and convert Wh to kWh.
3. Verify and label the four-hour result.

```{code-cell} python
# Step 1: Store hourly means and duration.
irradiance = [0,200,600,400]; hours = 1
# Step 2: Integrate and convert Wh to kWh.
irradiation = sum(irradiance)*hours/1000
# Step 3: Verify and label the four-hour result.
print(irradiation,'kWh/m²'); assert irradiation == 1.2
```

**Interpretation:** This is a four-hour window, not necessarily a complete day.

::::

(solar-exercise-04)=
### Exercise 04 — Inverter conversion

**Difficulty:** Basic

**Reference:** PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3; [@goswami2015], §§9.2–9.2.2.

**Task:** A PV array delivers 5 kW DC to a 96%-efficient inverter with a 6 kW AC limit. Find AC kW.

```{code-cell} python
# Your solution for solar exercise 04.
```

::::{dropdown} Step-by-step answer — Solar 04

1. Enter the DC input and equipment assumptions.
2. Apply conversion efficiency and the output limit.
3. Check the output cannot exceed input or rating.

```{code-cell} python
# Step 1: Enter the DC input and equipment assumptions.
dc_kw, efficiency, rating_kw = 5, 0.96, 6
# Step 2: Apply conversion efficiency and the output limit.
ac_kw = min(dc_kw*efficiency, rating_kw)
# Step 3: Check the output cannot exceed input or rating.
print(ac_kw,'kW AC'); assert ac_kw == 4.8 and ac_kw <= dc_kw
```

**Interpretation:** No clipping occurs in this example; efficiency is held constant.

::::

## Medium

(solar-exercise-05)=
### Exercise 05 — Capacity factor from AC energy

**Difficulty:** Medium

**Reference:** Capacity-factor energy accounting: [@manwell2009], §2.5; this exercise explicitly uses the AC plant rating.

**Task:** A 10 kW AC plant exports 48 kWh in 24 hours. Calculate AC-based CF and state the denominator.

```{code-cell} python
# Your solution for solar exercise 05.
```

::::{dropdown} Step-by-step answer — Solar 05

1. Set exported energy, AC capacity and duration.
2. Calculate the AC-based capacity factor.
3. Print a labelled percentage.

```{code-cell} python
# Step 1: Set exported energy, AC capacity and duration.
energy_kwh, capacity_kw, hours = 48, 10, 24
# Step 2: Calculate the AC-based capacity factor.
denominator_kwh = capacity_kw * hours
cf = energy_kwh / denominator_kwh
# Step 3: Print a labelled percentage.
print(f'AC-rated energy: {denominator_kwh} kWh; AC-based CF: {cf:.0%}')
assert denominator_kwh == 240 and cf == 0.2
```

**Interpretation:** 20% uses AC nameplate. A DC-based capacity factor would have a different denominator.

::::

(solar-exercise-06)=
### Exercise 06 — Cell-temperature estimate

**Difficulty:** Medium

**Reference:** Specified simplified NOCT approximation; compare the fuller [Sandia/SAM thermal model](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/cell-temperature/noct-cell-temperature/).

**Task:** At ambient 25 °C and irradiance [0,400,800] W/m², use NOCT 45 °C and Tc=Ta+G(NOCT−20)/800. Calculate temperatures.

```{code-cell} python
# Your solution for solar exercise 06.
```

::::{dropdown} Step-by-step answer — Solar 06

1. Store irradiance and temperature parameters.
2. Apply the approximate NOCT relationship.
3. Verify the 800 W/m² case.

```{code-cell} python
import numpy as np
# Step 1: Store irradiance and temperature parameters.
g = np.array([0,400,800]); ambient = 25; noct = 45
# Step 2: Apply the approximate NOCT relationship.
cell = ambient+g*(noct-20)/800
# Step 3: Verify the 800 W/m² case.
print(cell,'°C'); assert np.allclose(cell,[25,37.5,50])
```

**Interpretation:** This simple thermal relationship omits wind and mounting effects.

::::

(solar-exercise-07)=
### Exercise 07 — Temperature coefficient

**Difficulty:** Medium

**Reference:** Linear temperature correction: [@pvlibDocs].

**Task:** At 1000 W/m², a 10 kW STC array has gamma −0.004/°C. Evaluate DC output at cell temperatures [25,45,65]°C.

```{code-cell} python
# Your solution for solar exercise 07.
```

::::{dropdown} Step-by-step answer — Solar 07

1. Convert temperature cases to an array.
2. Apply the temperature multiplier relative to 25°C.
3. Check direction and expected output.

```{code-cell} python
import numpy as np
# Step 1: Convert temperature cases to an array.
temperature = np.array([25,45,65]); gamma = -0.004
# Step 2: Apply the temperature multiplier relative to 25°C.
dc_kw = 10*(1+gamma*(temperature-25))
# Step 3: Check direction and expected output.
print(dc_kw,'kW DC'); assert np.allclose(dc_kw,[10,9.2,8.4])
```

**Interpretation:** Higher cell temperature reduces output for a negative power coefficient.

::::

(solar-exercise-08)=
### Exercise 08 — Clipping budget

**Difficulty:** Medium

**Reference:** PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3; [@goswami2015], §§9.2–9.2.2. The constant-efficiency clipping rule is specified here.

**Task:** Hourly DC output [0,4,8,10] kW passes through a 97% inverter capped at 7 kW AC. Calculate AC kWh and clipping kWh.

```{code-cell} python
# Your solution for solar exercise 08.
```

::::{dropdown} Step-by-step answer — Solar 08

1. Calculate AC power before applying the rating.
2. Clip output and calculate discarded potential AC energy.
3. Check energy accounting across one-hour intervals.

```{code-cell} python
import numpy as np
# Step 1: Calculate AC power before applying the rating.
dc = np.array([0,4,8,10]); available_ac = dc*0.97
# Step 2: Clip output and calculate discarded potential AC energy.
ac = np.minimum(available_ac,7); clipping = available_ac-ac
# Step 3: Check energy accounting across one-hour intervals.
print(ac.sum(),'kWh AC;',clipping.sum(),'kWh clipping')
assert np.allclose(ac+clipping,available_ac) and np.isclose(ac.sum(),17.88)
```

**Interpretation:** Clipping is measured after modeled inverter conversion; it is distinct from inverter losses.

::::

(solar-exercise-09)=
### Exercise 09 — Sequential losses

**Difficulty:** Medium

**Reference:** PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3; [@goswami2015], §§9.2–9.2.2.

**Task:** Start with 1000 kWh of estimated DC energy before soiling and wiring losses. Apply 3% soiling, 2% wiring and 4% inverter losses sequentially. Compare with simply subtracting 9%.

```{code-cell} python
# Your solution for solar exercise 09.
```

::::{dropdown} Step-by-step answer — Solar 09

1. Express loss fractions as retained fractions.
2. Multiply the successive retained fractions.
3. Print the difference and check the product.

```{code-cell} python
import numpy as np
# Step 1: Express loss fractions as retained fractions.
retained = 1-np.array([0.03,0.02,0.04])
# Step 2: Multiply the successive retained fractions.
output = 1000*np.prod(retained); additive = 1000*(1-0.09)
# Step 3: Print the difference and check the product.
print(output,'kWh;',output-additive,'kWh difference')
assert np.isclose(output,912.576)
```

**Interpretation:** Multiplicative losses apply to changing energy bases; simple addition is an approximation.

::::

(solar-exercise-10)=
### Exercise 10 — Performance ratio

**Difficulty:** Medium

**Reference:** Yield and performance-ratio definitions: [Sandia PV performance metrics](https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/).

**Task:** A 100 kW DC array exports 12,000 kWh with POA irradiation 150 kWh/m². Use reference irradiance 1 kW/m² to calculate PR.

```{code-cell} python
# Your solution for solar exercise 10.
```

::::{dropdown} Step-by-step answer — Solar 10

1. Calculate final yield in hours from AC energy and DC rating.
2. Calculate reference yield and their ratio.
3. Verify and display the dimensionless result.

```{code-cell} python
# Step 1: Calculate final yield in hours from AC energy and DC rating.
final_yield = 12000/100
# Step 2: Calculate reference yield and their ratio.
reference_yield = 150/1; pr = final_yield/reference_yield
# Step 3: Verify and display the dimensionless result.
print(f'PR: {pr:.0%}'); assert pr == 0.8
```

**Interpretation:** PR normalises for irradiation; it is not cell conversion efficiency.

::::

(solar-exercise-11)=
### Exercise 11 — Specific yield

**Difficulty:** Medium

**Reference:** Yield and performance-ratio definitions: [Sandia PV performance metrics](https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/).

**Task:** Two systems generate [4000,9000] kWh from [5,10] kW DC. Calculate kWh/kWp and identify the higher specific yield.

```{code-cell} python
# Your solution for solar exercise 11.
```

::::{dropdown} Step-by-step answer — Solar 11

1. Store energy and DC nameplate values.
2. Normalise production by installed capacity.
3. Compare the normalised values.

```{code-cell} python
import numpy as np
# Step 1: Store energy and DC nameplate values.
energy = np.array([4000,9000]); capacity = np.array([5,10])
# Step 2: Normalise production by installed capacity.
specific_yield = energy/capacity
# Step 3: Compare the normalised values.
winner = int(np.argmax(specific_yield)) + 1
print(specific_yield,'kWh/kWp; higher yield: system',winner)
assert np.allclose(specific_yield,[800,900]) and winner==2
```

**Interpretation:** The periods must match before comparing yields; higher yield does not alone establish lower cost.

::::

(solar-exercise-12)=
### Exercise 12 — Unequal daylight intervals

**Difficulty:** Medium

**Reference:** Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

**Task:** Average AC powers [1,4,2] kW persist for [0.5,2,1.5] hours. Integrate their energy.

```{code-cell} python
# Your solution for solar exercise 12.
```

::::{dropdown} Step-by-step answer — Solar 12

1. Store paired power and duration arrays.
2. Multiply each interval before summing.
3. Verify the hand calculation.

```{code-cell} python
import numpy as np
# Step 1: Store paired power and duration arrays.
power = np.array([1,4,2]); duration = np.array([0.5,2,1.5])
# Step 2: Multiply each interval before summing.
energy = (power*duration).sum()
# Step 3: Verify the hand calculation.
print(energy,'kWh'); assert energy == 11.5
```

**Interpretation:** A sum of kW samples without duration is not energy for unequal intervals.

::::

## Hard

(solar-exercise-13)=
### Exercise 13 — Nighttime and invalid data

**Difficulty:** Hard

**Reference:** PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3; [@goswami2015], §§9.2–9.2.2. Array calculations: [@numpyDocs].

**Task:** Four intervals have irradiances [0,200,missing,−10] W/m² and durations [2,0.5,1,0.5] h. The first interval is night. Write a function for valid-time coverage, daylight coverage and observed irradiation. Preserve night zero and exclude non-finite or negative readings.

```{code-cell} python
# Your solution for solar exercise 13.
```

::::{dropdown} Step-by-step answer — Solar 13

1. Preserve measured zero and the two distinct invalid states.
2. Integrate valid intervals and weight coverage by duration.
3. Reconcile coverage and observed irradiation against hand calculations.

```{code-cell} python
import numpy as np
# Step 1: Preserve measured zero and the two distinct invalid states.
g=np.array([0,200,np.nan,-10]); dt=np.array([2,.5,1,.5])
daylight=np.array([False,True,True,True])
# Step 2: Integrate valid intervals and weight coverage by duration.
def audit_irradiance(g, dt, daylight):
    valid=np.isfinite(g)&(g>=0)
    return (dt[valid].sum()/dt.sum(), dt[valid&daylight].sum()/dt[daylight].sum(),
            np.dot(g[valid],dt[valid])/1000)

# Step 3: Reconcile coverage and observed irradiation against hand calculations.
result=audit_irradiance(g,dt,daylight)
print(result,'total coverage, daylight coverage, observed kWh/m²')
assert np.allclose(result,[.625,.25,.1])
```

**Interpretation:** Zero at night is a valid observation; a missing daylight observation cannot be treated as zero.

::::

(solar-exercise-14)=
### Exercise 14 — Compounded degradation

**Difficulty:** Hard

**Reference:** Original compounded-degradation scenario; Array calculations: [@numpyDocs].

**Task:** First-year output is 100 MWh. Write a reusable forecast for annual and cumulative output over 25 years. Compare 0.5% and 1% annual degradation and find the first year below 90 MWh in each case.

```{code-cell} python
# Your solution for solar exercise 14.
```

::::{dropdown} Step-by-step answer — Solar 14

1. Index years with no degradation in the first year.
2. Apply annual retained output multiplicatively.
3. Check the first and last years and print the sum.

```{code-cell} python
import numpy as np
# Step 1: Index years with no degradation in the first year.
years = np.arange(1,26)
# Step 2: Apply annual retained output multiplicatively.
def forecast(rate):
    return 100*(1-rate)**(years-1)
# Step 3: Check the first and last years and print the sum.
totals=[]; first=[]
for rate in [.005,.01]:
    energy=forecast(rate)
    first.append(next((int(year) for year,e in zip(years,energy) if e<90),None))
    totals.append(energy.sum())
    assert energy[0]==100 and np.isclose(energy[4],100*(1-rate)**4)
print(totals,'MWh cumulative; first year under 90 MWh:',first)
assert first==[23,12] and totals[0]>totals[1]
```

**Interpretation:** The rate is an assumption; weather and availability are held constant.

::::

(solar-exercise-15)=
### Exercise 15 — Daily profile plot

**Difficulty:** Hard

**Reference:** PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3; [@goswami2015], §§9.2–9.2.2. Plotting: [@matplotlibDocs].

**Task:** Model 24 hourly mean irradiances as max(0,800 sin(pi(h−6)/12)). Use 20 m², 20% module efficiency and 96% inverter efficiency. Plot DC and delivered AC for inverter caps of 2 and 3 kW. Integrate DC, delivered AC and clipping in MWh and reconcile the energy budget.

```{code-cell} python
# Your solution for solar exercise 15.
```

::::{dropdown} Step-by-step answer — Solar 15

1. Generate a labelled synthetic hourly irradiance profile.
2. Convert incident radiation to DC power and energy.
3. Plot scenarios and reconcile energy through both conversion boundaries.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Generate a labelled synthetic hourly irradiance profile.
h = np.arange(24); g = np.maximum(0,800*np.sin(np.pi*(h-6)/12))
# Step 2: Convert incident radiation to DC power and energy.
dc_kw = g*20*0.2/1000; potential=.96*dc_kw
# Step 3: Plot scenarios and reconcile energy through both conversion boundaries.
fig,ax=plt.subplots(figsize=(6,3)); ax.plot(h,dc_kw,label='DC')
for cap in [2,3]:
    ac=np.minimum(potential,cap)
    clipping=(potential-ac).sum()/1000
    delivered=ac.sum()/1000
    assert np.isclose(dc_kw.sum()/1000, delivered+clipping+(dc_kw-potential).sum()/1000)
    print(cap,'kW inverter:',delivered,'MWh delivered;',clipping,'MWh clipped')
    ax.plot(h,ac,label=f'AC, {cap} kW cap')
ax.set(xlabel='UTC hour',ylabel='Power (kW)',title='Synthetic PV day')
ax.legend(); fig.tight_layout(); plt.show()
```

**Interpretation:** The irradiance values are treated as interval means; no real location or date is represented.

::::

(solar-exercise-16)=
### Exercise 16 — Size an inverter by clipping target

**Difficulty:** Hard

**Reference:** PV device and system relationships: [@foster2010], Chapter 5; [@wade2003], Chapter 3; [@goswami2015], §§9.2–9.2.2. Original discrete inverter search.

**Task:** For hourly DC [0,2,5,9,12,8,3,0] kW and 97% efficiency, search integer AC ratings 5–12 kW for at most 5% clipped potential AC energy.

```{code-cell} python
# Your solution for solar exercise 16.
```

::::{dropdown} Step-by-step answer — Solar 16

1. Calculate the common available AC profile.
2. Evaluate each candidate using the same clipping denominator.
3. Select the smallest rating that satisfies the target.

```{code-cell} python
import numpy as np
# Step 1: Calculate the common available AC profile.
available = np.array([0,2,5,9,12,8,3,0])*0.97
# Step 2: Evaluate each candidate using the same clipping denominator.
trials = [(rating, np.maximum(available-rating,0).sum()/available.sum()) for rating in range(5,13)]
# Step 3: Select the smallest rating that satisfies the target.
best = next(t for t in trials if t[1]<=0.05)
print('Rating kW and clipping fraction:',best)
assert best[0]==10 and all(f>0.05 for r,f in trials if r<best[0])
```

**Interpretation:** An energy constraint is not a financial optimum; inverter prices and lifetime behavior are omitted.

::::

(solar-exercise-17)=
### Exercise 17 — Temperature and irradiance sensitivity grid

**Difficulty:** Hard

**Reference:** Specified simplified NOCT approximation; compare the fuller [Sandia/SAM thermal model](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/cell-temperature/noct-cell-temperature/). [@pvlibDocs].

**Task:** For irradiance [200, 600, 1000] W/m² and air temperature [10, 25, 40] °C, calculate a 3×3 grid of AC power in kW. Use 10 kW DC nameplate, an 8 kW AC cap, NOCT 45 °C, a power coefficient of −0.004/°C and inverter efficiency 0.97.

```{code-cell} python
# Your solution for solar exercise 17.
```

::::{dropdown} Step-by-step answer — Solar 17

1. Broadcast irradiance columns against ambient-temperature rows.
2. Calculate cell temperature, DC and limited AC power.
3. Check bounds and thermal direction and print a labelled grid.

```{code-cell} python
import numpy as np
import pandas as pd
# Step 1: Broadcast irradiance columns against ambient-temperature rows.
g=np.array([200,600,1000])[None,:]; ambient=np.array([10,25,40])[:,None]
# Step 2: Calculate cell temperature, DC and limited AC power.
cell=ambient+g*25/800
dc=np.maximum(10*g/1000*(1-0.004*(cell-25)),0); ac=np.minimum(dc*0.97,8)
# Step 3: Check bounds and thermal direction and print a labelled grid.
assert ac.shape==(3,3) and np.all(ac<=8) and np.all(np.diff(ac,axis=0)<=0)
print(pd.DataFrame(ac,index=[10,25,40],columns=[200,600,1000]).rename_axis('Ambient °C / G W m⁻²'))
```

**Interpretation:** At fixed irradiance, higher modeled cell temperature cannot increase power here; clipping can flatten sensitivity.

::::

(solar-exercise-18)=
### Exercise 18 — PV and battery self-consumption

**Difficulty:** Hard

**Reference:** Storage losses: [@wade2003], Chapter 5; [@manwell2009], §10.7. The dispatch policy is an original course assumption.

**Task:** PV hourly AC [0,4,6,0] kW meets constant 2 kW load with an initially empty 3 kWh battery, 2 kW limits and 90% efficiencies. Report unmet load, curtailed PV and final SOC.

```{code-cell} python
# Your solution for solar exercise 18.
```

::::{dropdown} Step-by-step answer — Solar 18

1. Initialise storage and bookkeeping in kWh.
2. Route each one-hour interval through direct use, charging and discharge.
3. Verify storage limits and show final energy accounting.

```{code-cell} python
import numpy as np
# Step 1: Initialise storage and bookkeeping in kWh.
soc=0.; unmet_total=0.; curtailed_total=0.; rows=[]
# Step 2: Route each one-hour interval through direct use, charging and discharge.
for pv in [0,4,6,0]:
    before=soc; direct=min(pv,2)
    charge=min(pv-direct,2,(3-soc)/0.9); discharge=min(2-direct,2,soc*0.9)
    soc+=charge*0.9-discharge/0.9
    unmet=2-direct-discharge; curtailed=pv-direct-charge
    served=direct+discharge
    losses=charge*0.10+discharge*(1/0.9-1)
    assert np.isclose(pv+before,served+curtailed+losses+soc)
    assert np.isclose(served+unmet,2)
    unmet_total+=unmet; curtailed_total+=curtailed; rows.append(soc)
# Step 3: Verify storage limits and show final energy accounting.
print(unmet_total,curtailed_total,soc,'kWh unmet, curtailed, final SOC')
assert np.isclose(unmet_total,2) and all(0<=s<=3+1e-12 for s in rows)
```

**Interpretation:** Stored energy remaining at the end has not yet served the load. Battery economic sizing needs longer data.

::::

(solar-exercise-19)=
### Exercise 19 — PV LCOE with degradation

**Difficulty:** Hard

**Reference:** Discounted cost and energy: [@ifc2015], §13.2.5 and Chapter 14; [@goswami2015], §1.4.3, Eq. 1.6; [@irena2026], methodology annex. Original degradation scenario.

**Task:** Use €1m capital, €20k annual O&M, 1500 MWh first-year AC, 0.5% annual degradation and 25 years. Calculate LCOE at 0%,3%,6% real discount rates.

```{code-cell} python
# Your solution for solar exercise 19.
```

::::{dropdown} Step-by-step answer — Solar 19

1. Define year-end energy and a reusable discounted-cost calculation.
2. Evaluate the interest-rate scenarios with identical technical inputs.
3. Check the undiscounted result and scenario direction.

```{code-cell} python
import numpy as np
# Step 1: Define year-end energy and a reusable discounted-cost calculation.
years=np.arange(1,26); energy=1500*0.995**(years-1)
def lcoe(rate):
    discount=(1+rate)**years
    return (1_000_000+np.sum(20_000/discount))/np.sum(energy/discount)
# Step 2: Evaluate the interest-rate scenarios with identical technical inputs.
values=[lcoe(r) for r in [0,0.03,0.06]]
# Step 3: Check the undiscounted result and scenario direction.
print(values,'EUR/MWh')
assert np.isclose(values[0],1_500_000/energy.sum()) and np.all(np.diff(values)>0)
```

**Interpretation:** See IFC Chapter 14 and IRENA cost methodology. Costs are synthetic; taxes, replacements and salvage are omitted.

::::

(solar-exercise-20)=
### Exercise 20 — Detect underperformance without future leakage

**Difficulty:** Hard

**Reference:** Data handling: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html). Chronological baseline evaluation; thresholds are course assumptions.

**Task:** Expected daily energy is [10,20,30,40,50,60] kWh; observed is [9,18,27,36,30,36]. Calibrate an observed/expected ratio on days 1–4, then flag holdout days below 80% of calibrated expectation.

```{code-cell} python
# Your solution for solar exercise 20.
```

::::{dropdown} Step-by-step answer — Solar 20

1. Split the chronological observations before calibration.
2. Estimate one scale factor only from training energy.
3. Flag and check the holdout underperformance.

```{code-cell} python
import numpy as np
# Step 1: Split the chronological observations before calibration.
expected=np.array([10,20,30,40,50,60]); observed=np.array([9,18,27,36,30,36])
train=slice(0,4); test=slice(4,None)
# Step 2: Estimate one scale factor only from training energy.
factor=observed[train].sum()/expected[train].sum()
prediction=expected[test]*factor; relative=observed[test]/prediction
# Step 3: Flag and check the holdout underperformance.
flags=relative<0.8
print('Calibration:',factor,'holdout relative performance:',relative,'flags:',flags)
assert np.isclose(factor,0.9) and flags.all()
```

**Interpretation:** Flags identify a discrepancy, not its cause. Weather-model errors, outages or sensor faults need separate diagnosis.

::::

