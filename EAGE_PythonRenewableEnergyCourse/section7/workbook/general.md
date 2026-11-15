---
kernelspec:
  name: python3
  display_name: Python 3
---

# General

[Review the theory and worked examples](../../section0/introRenewableEnergy.md). [Download the exercise notebook](general.ipynb).

## Chapter practice — 20 Python exercises

**4 Basic · 8 Medium · 8 Hard.** Basic exercises connect concepts, units and short calculations. Medium exercises apply functions and arrays. Hard exercises combine modelling, validation and interpretation. Each solution runs independently.

## Basic

(general-exercise-01)=
### Exercise 01 — Convert energy units

**Difficulty:** Basic

**Reference:** Unit definitions/conversions: [NIST SP 811, Appendix B](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors).

**Task:** Convert 2.5 MWh to kWh and MJ. Use 1 kWh = 3.6 MJ.

```{code-cell} python
# Your solution for general exercise 01.
```

::::{dropdown} Step-by-step answer — General 01

1. Store the energy in MWh.
2. Convert to kWh, then MJ.
3. Print labelled results and check the conversion.

```{code-cell} python
# Step 1: Store the energy in MWh.
energy_mwh = 2.5
# Step 2: Convert to kWh, then MJ.
energy_kwh = energy_mwh * 1000
energy_mj = energy_kwh * 3.6
# Step 3: Print labelled results and check the conversion.
print(energy_kwh, 'kWh;', energy_mj, 'MJ')
assert energy_kwh == 2500 and energy_mj == 9000
```

**Interpretation:** Energy is conserved by a unit conversion: 2,500 kWh = 9,000 MJ.

::::

(general-exercise-02)=
### Exercise 02 — Power over a time interval

**Difficulty:** Basic

**Reference:** Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

**Task:** A generator averages 3 MW for 6 hours. Calculate MWh and mean MW from the energy.

```{code-cell} python
# Your solution for general exercise 02.
```

::::{dropdown} Step-by-step answer — General 02

1. Enter interval-average power and duration.
2. Multiply power by elapsed time.
3. Recover mean power and verify the result.

```{code-cell} python
# Step 1: Enter interval-average power and duration.
power_mw, hours = 3, 6
# Step 2: Multiply power by elapsed time.
energy_mwh = power_mw * hours
# Step 3: Recover mean power and verify the result.
print(energy_mwh, 'MWh;', energy_mwh / hours, 'MW')
assert energy_mwh == 18
```

**Interpretation:** 18 MWh is energy; 3 MW is power.

::::

(general-exercise-03)=
### Exercise 03 — Daily capacity factor

**Difficulty:** Basic

**Reference:** Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

**Task:** A 5 MW plant delivers 48 MWh in 24 hours. Calculate its daily capacity factor.

```{code-cell} python
# Your solution for general exercise 03.
```

::::{dropdown} Step-by-step answer — General 03

1. Calculate maximum possible energy over this day.
2. Divide actual energy by that maximum.
3. Print a percentage and check bounds.

```{code-cell} python
# Step 1: Calculate maximum possible energy over this day.
maximum_mwh = 5 * 24
# Step 2: Divide actual energy by that maximum.
capacity_factor = 48 / maximum_mwh
# Step 3: Print a percentage and check bounds.
print(f'{capacity_factor:.1%}')
assert abs(capacity_factor - 0.4) < 1e-12
```

**Interpretation:** 40% describes this day, not the annual resource or conversion efficiency.

::::

(general-exercise-04)=
### Exercise 04 — Conversion losses

**Difficulty:** Basic

**Reference:** Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

**Task:** A device receives 100 MWh and converts 85% to useful energy. Calculate output and losses.

```{code-cell} python
# Your solution for general exercise 04.
```

::::{dropdown} Step-by-step answer — General 04

1. Enter input energy and fractional efficiency.
2. Calculate useful energy and the remainder.
3. Reconcile the energy balance.

```{code-cell} python
# Step 1: Enter input energy and fractional efficiency.
input_mwh, efficiency = 100, 0.85
# Step 2: Calculate useful energy and the remainder.
output_mwh = input_mwh * efficiency
loss_mwh = input_mwh - output_mwh
# Step 3: Reconcile the energy balance.
print(output_mwh, 'MWh useful;', loss_mwh, 'MWh losses')
assert output_mwh + loss_mwh == input_mwh
```

**Interpretation:** 85 MWh useful and 15 MWh lost; efficiency is dimensionless.

::::

## Medium

(general-exercise-05)=
### Exercise 05 — Generation shares

**Difficulty:** Medium

**Reference:** Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

**Task:** Solar, hydro, wind and geothermal produce 10, 30, 40 and 20 MWh. Print each source share.

```{code-cell} python
# Your solution for general exercise 05.
```

::::{dropdown} Step-by-step answer — General 05

1. Store source energies in a dictionary.
2. Sum the energy and divide each source by the total.
3. Print percentages and confirm they sum to one.

```{code-cell} python
# Step 1: Store source energies in a dictionary.
energy = dict(solar=10, hydro=30, wind=40, geothermal=20)
# Step 2: Sum the energy and divide each source by the total.
total = sum(energy.values())
shares = {name: value / total for name, value in energy.items()}
# Step 3: Print percentages and confirm they sum to one.
for name, share in shares.items():
    print(name, f'{share:.0%}')
assert abs(sum(shares.values()) - 1) < 1e-12
```

**Interpretation:** Shares use an energy denominator, not installed capacity.

::::

(general-exercise-06)=
### Exercise 06 — Unequal sampling intervals

**Difficulty:** Medium

**Reference:** Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

**Task:** Average powers [2,4,1] MW last [0.5,1,2] hours. Calculate total energy and time-weighted average power.

```{code-cell} python
# Your solution for general exercise 06.
```

::::{dropdown} Step-by-step answer — General 06

1. Represent power and duration as matching arrays.
2. Integrate interval energy and divide by elapsed time.
3. Verify energy against a hand calculation.

```{code-cell} python
import numpy as np
# Step 1: Represent power and duration as matching arrays.
p = np.array([2, 4, 1]); dt = np.array([0.5, 1, 2])
# Step 2: Integrate interval energy and divide by elapsed time.
energy = np.sum(p * dt); mean_power = energy / dt.sum()
# Step 3: Verify energy against a hand calculation.
print(energy, 'MWh;', mean_power, 'MW')
assert energy == 7 and mean_power == 2
```

**Interpretation:** The arithmetic mean of the three powers is inappropriate for unequal durations.

::::

(general-exercise-07)=
### Exercise 07 — Read generation from CSV

**Difficulty:** Medium

**Reference:** Data handling: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html).

**Task:** Read this CSV using pandas: source,mwh with rows solar,12; wind,18; hydro,10. Calculate total energy.

```{code-cell} python
# Your solution for general exercise 07.
```

::::{dropdown} Step-by-step answer — General 07

1. Load the supplied CSV text.
2. Validate values before aggregating.
3. Print the table and checked total.

```{code-cell} python
from io import StringIO
import pandas as pd
# Step 1: Load the supplied CSV text.
csv = 'source,mwh\nsolar,12\nwind,18\nhydro,10\n'
df = pd.read_csv(StringIO(csv))
# Step 2: Validate values before aggregating.
assert df.source.is_unique and df.mwh.notna().all() and df.mwh.ge(0).all()
total = df.mwh.sum()
# Step 3: Print the table and checked total.
print(df.to_string(index=False)); print(total, 'MWh')
assert total == 40
```

**Interpretation:** StringIO behaves like a file; replace it with a CSV path for local data.

::::

(general-exercise-08)=
### Exercise 08 — Missing generation is not zero

**Difficulty:** Medium

**Reference:** Data handling: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html).

**Task:** For [1, missing,3,2] hourly MW readings, report coverage and observed energy, but do not invent a four-hour total.

```{code-cell} python
# Your solution for general exercise 08.
```

::::{dropdown} Step-by-step answer — General 08

1. Preserve the missing measurement.
2. Calculate coverage and energy for observed hours only.
3. Label the result and verify it.

```{code-cell} python
import pandas as pd
# Step 1: Preserve the missing measurement.
p = pd.Series([1, None, 3, 2], dtype=float)
# Step 2: Calculate coverage and energy for observed hours only.
coverage = p.notna().mean(); observed_mwh = p.sum(min_count=1)
# Step 3: Label the result and verify it.
print(f'Coverage: {coverage:.0%}; observed energy: {observed_mwh} MWh')
assert coverage == 0.75 and observed_mwh == 6
```

**Interpretation:** 6 MWh covers three observed hours; the full four-hour total is unknown.

::::

(general-exercise-09)=
### Exercise 09 — Exact duplicate readings

**Difficulty:** Medium

**Reference:** Data handling: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html).

**Task:** Times [00,01,01,02] have MW [1,2,2,3]. Remove only exact timestamp/value duplicates and check uniqueness.

```{code-cell} python
# Your solution for general exercise 09.
```

::::{dropdown} Step-by-step answer — General 09

1. Build a timestamped table.
2. Remove exact repeated records, then check for conflicting times.
3. Sum the remaining hourly mean powers.

```{code-cell} python
import pandas as pd
# Step 1: Build a timestamped table.
df = pd.DataFrame({'time': pd.to_datetime(['2025-01-01 00:00', '2025-01-01 01:00', '2025-01-01 01:00', '2025-01-01 02:00'], utc=True), 'mw': [1,2,2,3]})
# Step 2: Remove exact repeated records, then check for conflicting times.
clean = df.drop_duplicates()
assert clean.time.is_unique
# Step 3: Sum the remaining hourly mean powers.
print(clean.to_string(index=False)); print(clean.mw.sum(), 'MWh')
assert len(clean) == 3 and clean.mw.sum() == 6
```

**Interpretation:** Different values at one timestamp would require investigation, not arbitrary deletion.

::::

(general-exercise-10)=
### Exercise 10 — Match supply and demand

**Difficulty:** Medium

**Reference:** Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units). Original no-storage balance.

**Task:** Hourly renewable supply is [1,5,2] MW and demand [3,3,3] MW. Calculate served, curtailed and unmet energy.

```{code-cell} python
# Your solution for general exercise 10.
```

::::{dropdown} Step-by-step answer — General 10

1. Store aligned hourly power arrays.
2. Allocate direct supply and residuals.
3. Check both balances and print one-hour energy sums.

```{code-cell} python
import numpy as np
# Step 1: Store aligned hourly power arrays.
supply = np.array([1,5,2]); demand = np.array([3,3,3])
# Step 2: Allocate direct supply and residuals.
served = np.minimum(supply, demand)
curtailed = supply - served; unmet = demand - served
# Step 3: Check both balances and print one-hour energy sums.
assert np.array_equal(served + curtailed, supply)
assert np.array_equal(served + unmet, demand)
print('MWh served, curtailed, unmet:', served.sum(), curtailed.sum(), unmet.sum())
```

**Interpretation:** Results are 6, 2 and 3 MWh; total supply alone cannot establish demand coverage.

::::

(general-exercise-11)=
### Exercise 11 — Aggregate daily energy

**Difficulty:** Medium

**Reference:** Data handling: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html). Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

**Task:** Generate 48 consecutive UTC hourly means of 2 MW and aggregate to daily MWh.

```{code-cell} python
# Your solution for general exercise 11.
```

::::{dropdown} Step-by-step answer — General 11

1. Create an hourly UTC index and matching mean powers.
2. Multiply by one hour and sum each day.
3. Check complete coverage and print daily energy.

```{code-cell} python
import pandas as pd
# Step 1: Create an hourly UTC index and matching mean powers.
p = pd.Series(2.0, index=pd.date_range('2025-01-01', periods=48, freq='h', tz='UTC'))
# Step 2: Multiply by one hour and sum each day.
daily = (p * 1.0).resample('D').sum(min_count=24)
# Step 3: Check complete coverage and print daily energy.
print(daily.rename('MWh'))
assert (daily == 48).all() and daily.sum() == 96
```

**Interpretation:** A complete UTC day has 24 hourly intervals; local daylight-saving days require care.

::::

(general-exercise-12)=
### Exercise 12 — Compare fleet capacity factors

**Difficulty:** Medium

**Reference:** Resource and energy accounting: [@manwell2009], §§2.3–2.5; aerodynamics: §3.2.

**Task:** Plant A: 2 MW and 24 MWh; plant B: 8 MW and 144 MWh over the same 24 hours. Calculate plant and fleet CF.

```{code-cell} python
# Your solution for general exercise 12.
```

::::{dropdown} Step-by-step answer — General 12

1. Store nameplate capacities and energy.
2. Compute each ratio and the fleet ratio from totals.
3. Check the fleet weighting.

```{code-cell} python
import numpy as np
# Step 1: Store nameplate capacities and energy.
capacity = np.array([2,8]); energy = np.array([24,144])
# Step 2: Compute each ratio and the fleet ratio from totals.
individual = energy / (capacity * 24)
fleet = energy.sum() / (capacity.sum() * 24)
# Step 3: Check the fleet weighting.
print(individual, 'individual CF;', fleet, 'fleet CF')
assert np.isclose(fleet, 0.7)
assert np.isclose(fleet, np.average(individual, weights=capacity))
```

**Interpretation:** Fleet CF is capacity-weighted when the time window is the same.

::::

## Hard

(general-exercise-13)=
### Exercise 13 — SQL and pandas agreement

**Difficulty:** Hard

**Reference:** Data handling: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html). [@sqliteDocs].

**Task:** Solar delivers 10 MWh and 12,000 kWh; wind delivers 20 MWh and 18,000 kWh. Normalise mixed units, reject unsupported units and reconcile separate SQL and pandas totals by source.

```{code-cell} python
# Your solution for general exercise 13.
```

::::{dropdown} Step-by-step answer — General 13

1. Create a small table of source observations.
2. Convert each reading to MWh and reject unknown units.
3. Aggregate independently using SQL and pandas.
4. Reconcile totals and display the result.

```{code-cell} python
import sqlite3
import pandas as pd
# Step 1: Create a small table of source observations.
df = pd.DataFrame({'source': ['solar','solar','wind','wind'],
                   'energy': [10,12000,20,18000], 'unit': ['MWh','kWh','MWh','kWh']})
# Step 2: Convert each reading to MWh and reject unknown units.
def to_mwh(values, units):
    scale = units.map({'MWh': 1, 'kWh': .001})
    if scale.isna().any() or values.isna().any() or values.lt(0).any():
        raise ValueError('Need non-negative readings with known units')
    return values * scale

df['mwh'] = to_mwh(df.energy, df.unit)
# Step 3: Aggregate independently using SQL and pandas.
with sqlite3.connect(':memory:') as con:
    df.to_sql('energy', con, index=False)
    sql = pd.read_sql_query('SELECT source, SUM(mwh) AS mwh FROM energy GROUP BY source ORDER BY source', con).set_index('source').mwh
expected = df.groupby('source').mwh.sum().sort_index()
# Step 4: Reconcile totals and display the result.
pd.testing.assert_series_equal(sql, expected)
assert sql.to_dict() == {'solar': 22.0, 'wind': 38.0}
print(sql)
```

**Interpretation:** Solar totals 22 MWh and wind 38 MWh; matching code outputs still requires valid source definitions.

::::

(general-exercise-14)=
### Exercise 14 — Forecast benchmark

**Difficulty:** Hard

**Reference:** Chronological evaluation: [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

**Task:** Use monthly MWh [10,12,14,16] as training and [15,17] as test. Compare fixed last-value, training-mean and walk-forward last-value forecast MAE. In walk-forward prediction, reveal each test observation only after predicting it.

```{code-cell} python
# Your solution for general exercise 14.
```

::::{dropdown} Step-by-step answer — General 14

1. Separate earlier training observations from later test observations.
2. Construct forecasts without reading test values.
3. Evaluate both forecasts on the same test window.

```{code-cell} python
import numpy as np
# Step 1: Separate earlier training observations from later test observations.
train = np.array([10,12,14,16]); test = np.array([15,17])
# Step 2: Construct forecasts without reading test values.
last_forecast = np.repeat(train[-1], len(test)); mean_forecast = np.repeat(train.mean(), len(test))
# Step 3: Evaluate both forecasts on the same test window.
walk_forecast = np.r_[train[-1],test[:-1]]
mae = {name: np.abs(test-predictions).mean() for name,predictions in
       [('fixed last',last_forecast),('training mean',mean_forecast),('walk-forward',walk_forecast)]}
print(mae,'MWh MAE')
assert mae == {'fixed last':1, 'training mean':3, 'walk-forward':1.5}
```

**Interpretation:** Two test months illustrate evaluation; they do not establish long-term forecast skill.

::::

(general-exercise-15)=
### Exercise 15 — Plot a source mix

**Difficulty:** Hard

**Reference:** Plotting: [@matplotlibDocs]. Power, energy and conversion: [@wade2003], Chapter 2; [@jica2011], §3.1.2 (use coherent SI units).

**Task:** For two one-hour intervals, use solar [0,20], hydro [15,20], wind [20,10], geothermal [7,8] and demand [50,45], all in MW. Plot stacked source output and demand. Reconcile interval and total energy, and explain how surplus and shortfall coexist.

```{code-cell} python
# Your solution for general exercise 15.
```

::::{dropdown} Step-by-step answer — General 15

1. Align per-source power and demand at each hour.
2. Stack source power and overlay the demand line.
3. Calculate served energy, surplus and shortfall separately for each hour.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Align per-source power and demand at each hour.
sources = {'Solar':[0,20], 'Hydro':[15,20], 'Wind':[20,10], 'Geothermal':[7,8]}
demand = np.array([50,45])
# Step 2: Stack source power and overlay the demand line.
fig, ax = plt.subplots(figsize=(6,3)); bottom = np.zeros(2)
for name, power in sources.items():
    ax.bar([1,2],power,bottom=bottom,label=name)
    bottom += power
ax.plot([1,2],demand,'ko-',label='Demand')
ax.set(xlabel='Hour',ylabel='Power (MW)',title='Synthetic grid'); ax.legend()
fig.tight_layout(); plt.show()
# Step 3: Calculate served energy, surplus and shortfall separately for each hour.
served = np.minimum(bottom,demand)
surplus = bottom-served; shortfall = demand-served
assert np.allclose(served+surplus,bottom) and np.allclose(served+shortfall,demand)
assert (bottom.sum(),served.sum(),surplus.sum(),shortfall.sum()) == (100,87,13,8)
print('MWh available, served, surplus, shortfall:',100,87,13,8)
```

**Interpretation:** The second-hour surplus cannot cover the earlier shortfall without storage or another balancing source.

::::

(general-exercise-16)=
### Exercise 16 — Battery dispatch with losses

**Difficulty:** Hard

**Reference:** Storage losses: [@wade2003], Chapter 5; [@manwell2009], §10.7. The dispatch policy is an original course assumption.

**Task:** Supply [0,4,4,0] MW meets constant 2 MW demand. Start empty with 3 MWh usable storage, 2 MW charge/discharge limits and 90% efficiency each way. Route four hours and prove energy balance.

```{code-cell} python
# Your solution for general exercise 16.
```

::::{dropdown} Step-by-step answer — General 16

1. Set storage state, limits and accounting totals.
2. Dispatch direct generation, then charge or discharge within limits.
3. Check limits and report energy outcomes.

```{code-cell} python
import numpy as np
# Step 1: Set storage state, limits and accounting totals.
soc = 0.0; rows = []
# Step 2: Dispatch direct generation, then charge or discharge within limits.
for supply in [0,4,4,0]:
    before = soc; direct = min(supply, 2)
    charge = min(supply-direct, 2, (3-soc)/0.9)
    discharge = min(2-direct, 2, soc*0.9)
    soc += charge*0.9 - discharge/0.9
    curtailed = supply-direct-charge; unmet = 2-direct-discharge
    served = direct + discharge
    losses = charge * 0.10 + discharge * (1/0.9 - 1)
    # Powers in MW equal interval energies in MWh numerically for a one-hour step.
    assert np.isclose(supply + before, served + curtailed + losses + soc)
    assert np.isclose(served + unmet, 2)
    rows.append((soc, curtailed, unmet))
# Step 3: Check limits and report energy outcomes.
assert all(0 <= row[0] <= 3+1e-12 for row in rows)
print('SOC MWh, curtailment MWh, unmet MWh by hour:', rows)
assert np.isclose(sum(r[2] for r in rows), 2)
```

**Interpretation:** Charging losses consume supply and discharge losses consume stored energy; final stored energy is not delivered energy.

::::

(general-exercise-17)=
### Exercise 17 — Discounted energy cost

**Difficulty:** Hard

**Reference:** Discounted cost and energy: [@ifc2015], §13.2.5 and Chapter 14; [@goswami2015], §1.4.3, Eq. 1.6; [@irena2026], methodology annex.

**Task:** For €1000 initial cost, €20 annual O&M and 100 MWh/year over 10 years, calculate LCOE at real rates 0%,5%,10%, with year-end flows.

```{code-cell} python
# Your solution for general exercise 17.
```

::::{dropdown} Step-by-step answer — General 17

1. Represent operating years and create a reusable cost function.
2. Compare consistent real-rate scenarios.
3. Check the zero-rate hand result and print units.

```{code-cell} python
import numpy as np
# Step 1: Represent operating years and create a reusable cost function.
years = np.arange(1,11)
def lcoe(rate):
    factors = (1+rate)**years
    return (1000+np.sum(20/factors))/np.sum(100/factors)
# Step 2: Compare consistent real-rate scenarios.
results = [lcoe(r) for r in [0,0.05,0.10]]
# Step 3: Check the zero-rate hand result and print units.
print(results, 'EUR/MWh')
assert np.isclose(results[0], 1.2) and np.all(np.diff(results)>0)
```

**Interpretation:** Teaching costs are intentionally small. This omits replacements, taxes and end-of-life costs; see IFC Chapter 14 for economic boundaries.

::::

(general-exercise-18)=
### Exercise 18 — Strict multi-file alignment

**Difficulty:** Hard

**Reference:** Data handling: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html).

**Task:** Join two CSV-like tables with hourly UTC times: solar [0,1,2] and wind [2,1,0]. Reject duplicate times or mismatched coverage before calculating their sum.

```{code-cell} python
# Your solution for general exercise 18.
```

::::{dropdown} Step-by-step answer — General 18

1. Create separate source tables with their own timestamp columns.
2. Validate matching coverage and perform a one-to-one join.
3. Test both completeness and the known sum.

```{code-cell} python
import pandas as pd
import numpy as np
# Step 1: Create separate source tables with their own timestamp columns.
time = pd.date_range('2025-01-01', periods=3, freq='h', tz='UTC')
solar = pd.DataFrame({'time':time, 'solar_mw':[0,1,2]})
wind = pd.DataFrame({'time':time[::-1], 'wind_mw':[0,1,2]})
# Step 2: Validate matching coverage and perform a one-to-one join.
def align(solar, wind):
    for table in (solar, wind):
        if table.time.isna().any() or not table.time.is_unique:
            raise ValueError('Each source must have valid, unique timestamps.')
    if set(solar.time) != set(wind.time):
        raise ValueError('Sources must cover the same timestamps.')
    return solar.merge(wind, on='time', validate='one_to_one', sort=True)

joined = align(solar, wind)
joined['total_mw'] = joined.solar_mw + joined.wind_mw
# Step 3: Test both completeness and the known sum.
assert len(joined)==3 and np.allclose(joined.total_mw, 2)
assert joined.time.is_monotonic_increasing
print(joined.to_string(index=False))
```

**Interpretation:** An inner join alone can silently discard unmatched times; coverage must be checked first.

::::

(general-exercise-19)=
### Exercise 19 — Capacity search under a reliability target

**Difficulty:** Hard

**Reference:** Original discrete design search under the stated shortfall target; Array calculations: [@numpyDocs].

**Task:** Wind produces [0,1,2,1] MW per unit; firm geothermal supplies 1 MW and demand is 2 MW each hour. Search 0–4 wind units for coverage at least 80%. Report the smallest feasible count, unmet and curtailed MWh.

```{code-cell} python
# Your solution for general exercise 19.
```

::::{dropdown} Step-by-step answer — General 19

1. Set aligned generation shapes and a coverage target.
2. Evaluate each discrete design without equating supply with served energy.
3. Select the smallest feasible design and check it.

```{code-cell} python
import numpy as np
# Step 1: Set aligned generation shapes and a coverage target.
wind = np.array([0,1,2,1]); demand = np.full(4,2); target = 0.8
# Step 2: Evaluate each discrete design without equating supply with served energy.
results = []
for units in range(5):
    supply = 1 + units*wind; served = np.minimum(supply, demand)
    results.append((units, served.sum()/demand.sum(), (demand-served).sum(), (supply-served).sum()))
# Step 3: Select the smallest feasible design and check it.
best = next(row for row in results if row[1]>=target)
print('Units, coverage, unmet MWh, curtailed MWh:', best)
assert best[0]==1 and best[1]==0.875
```

**Interpretation:** No amount of this wind shape fixes the zero-wind hour. A coverage target is not the same as hourly reliability.

::::

(general-exercise-20)=
### Exercise 20 — Monte Carlo uncertainty

**Difficulty:** Hard

**Reference:** Original uncertainty propagation with stated distributions; [NumPy random sampling](https://numpy.org/doc/stable/reference/random/index.html).

**Task:** Assume 1000 independent illustrative annual-energy outcomes uniformly distributed between 80 and 120 GWh. Use seed 7; report mean, 5th and 95th percentiles and probability of less than 90 GWh.

```{code-cell} python
# Your solution for general exercise 20.
```

::::{dropdown} Step-by-step answer — General 20

1. Sample a stated synthetic uncertainty model reproducibly.
2. Summarise the distribution and threshold frequency.
3. Check physical support and explain the probability convention.

```{code-cell} python
import numpy as np
# Step 1: Sample a stated synthetic uncertainty model reproducibly.
rng = np.random.default_rng(7); energy = rng.uniform(80,120,1000)
# Step 2: Summarise the distribution and threshold frequency.
interval = np.quantile(energy,[0.05,0.95]); probability = np.mean(energy<90)
# Step 3: Check physical support and explain the probability convention.
print('Mean GWh:', energy.mean(), '5th/95th GWh:', interval, 'P(E<90):', probability)
assert energy.min()>=80 and energy.max()<=120 and 0<=probability<=1
```

**Interpretation:** These are simulation percentiles under an assumed distribution, not a confidence interval or calibrated project P90.

::::

