---
kernelspec:
  name: python3
  display_name: Python 3
---

# 20 Python exercises for renewable energy

**5 easy · 10 medium · 5 hard.** These original exercises adapt concepts from the supplied books and documents. Every numerical dataset is synthetic. References identify the underlying concept, not a claim that the source contains these exact problems or numbers.

Complete Sections 1–3 for Python fundamentals and read the corresponding Section 6 technology lesson. Easy exercises practice scalar calculations; medium exercises combine arrays, functions and data; hard exercises integrate models, constraints and validation. Hard exercises 16–20 build on the ideas in 6, 13; 3, 7; 8; 5, 14; and 12, 14 respectively, but their solution code is self-contained.

Try each prompt in a new code cell. Use the starter cell for your work, then open **Solution** to reveal numbered steps, complete executable code, checks and interpretation. All solution dropdowns start closed in the book. Each solution can run independently with its own imports and data. Existing course quizzes remain available as additional practice.

Use interval-average values unless stated otherwise. Write units beside results and record assumptions. An assertion checks a known result or conservation law; passing it does not validate every real-world use of the model.

## Exercise map

| No. | Level | Topic | Python practice |
|---|---|---|---|
| 01 | Easy | [Energy and capacity factor](#exercise-01) | variables, arithmetic, formatted output |
| 02 | Easy | [Hydropower from net head](#exercise-02) | variables, unit conversion |
| 03 | Easy | [Solar panel and array output](#exercise-03) | functions, multiplication |
| 04 | Easy | [Wind speed and the cubic law](#exercise-04) | math.pi, powers, ratios |
| 05 | Easy | [Geothermal thermal and net electrical power](#exercise-05) | sequential calculations |
| 06 | Medium | [Environmental flow and daily hydro energy](#exercise-06) | NumPy masks, clipping, integration |
| 07 | Medium | [Temperature-corrected PV and inverter clipping](#exercise-07) | arrays, DataFrame, minimum/maximum |
| 08 | Medium | [A turbine curve with tested boundaries](#exercise-08) | functions, masks, boundary tests |
| 09 | Medium | [Energy from unequal intervals](#exercise-09) | arrays, validation, weighted mean |
| 10 | Medium | [A transparent solar-data quality check](#exercise-10) | pandas, duplicates, missing data |
| 11 | Medium | [Compare SQL and pandas generation totals](#exercise-11) | SQLite, groupby, reconciliation |
| 12 | Medium | [Monthly PV performance ratio](#exercise-12) | pandas, ratios, labelled bar chart |
| 13 | Medium | [Pumped-storage water and electricity budget](#exercise-13) | energy conversion, round-trip efficiency |
| 14 | Medium | [Geothermal decline scenarios](#exercise-14) | functions, arrays, scenario plots |
| 15 | Medium | [Wind direction as circular data](#exercise-15) | trigonometry, validation, sector counts |
| 16 | Hard | [Reservoir routing with a mass-balance check](#exercise-16) | stateful loops, conservation, DataFrame |
| 17 | Hard | [Hybrid supply with a lossy battery](#exercise-17) | dispatch loops, balances, constraints |
| 18 | Hard | [Wind AEP sensitivity with a Weibull resource](#exercise-18) | probability bins, convergence, scenario analysis |
| 19 | Hard | [Fit a geothermal decline model without future leakage](#exercise-19) | log-linear fit, holdout, baseline comparison |
| 20 | Hard | [Discounted PV cost with degradation and sensitivity](#exercise-20) | discounting, functions, scenario tables |

(exercise-01)=
## Exercise 01 — Energy and capacity factor

**Difficulty:** Easy  
**Python skills:** variables, arithmetic, formatted output

**Reference:** Wade, Chapter 2, electricity and energy [@wade2003]; Python tutorial [@pythonDocs].

**Task:** A 2 MW wind turbine has mean electrical output 0.6 MW over a complete 24-hour day. Calculate MWh, kWh and daily capacity factor. Assume each hourly mean is the same; capacity factor is not rotor efficiency.

```{code-cell} python
# Exercise 01: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 01

1. Multiply mean MW by 24 hours.
2. Multiply MWh by 1,000 for kWh.
3. Divide energy by rated MW times the same 24 hours.

```{code-cell} python
import math
mean_mw, rated_mw, hours = 0.6, 2.0, 24
energy_mwh = mean_mw * hours
energy_kwh = energy_mwh * 1000
capacity_factor = energy_mwh / (rated_mw * hours)
print(f"{energy_mwh:.1f} MWh; {energy_kwh:.0f} kWh; CF={capacity_factor:.1%}")
assert math.isclose(energy_mwh, 14.4)
assert math.isclose(capacity_factor, 0.30)
```

**Interpretation and checks:** Expected: 14.4 MWh, 14,400 kWh and 30%. This is a one-day capacity factor, not an annual estimate.

::::

(exercise-02)=
## Exercise 02 — Hydropower from net head

**Difficulty:** Easy  
**Python skills:** variables, unit conversion

**Reference:** JICA, Chapter 3, especially 3.1.2–3.1.3 [@jica2011].

**Task:** Use flow 12 m³/s, gross head 30 m, head loss 2 m, combined efficiency 0.85, water density 1,000 kg/m³ and gravity 9.81 m/s². Calculate net head and electrical MW. Ignore generator clipping.

```{code-cell} python
# Exercise 02: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 02

1. Subtract head loss from gross head.
2. Apply P = efficiency × density × gravity × flow × net head in watts.
3. Divide watts by one million and check zero flow.

```{code-cell} python
import math
net_head_m = 30 - 2
def hydro_mw(flow_m3s):
    if not math.isfinite(flow_m3s) or flow_m3s < 0:
        raise ValueError("flow must be finite and non-negative")
    return 0.85 * 1000 * 9.81 * flow_m3s * net_head_m / 1e6
result = hydro_mw(12)
print(f"Net head {net_head_m} m; electrical power {result:.6f} MW")
assert math.isclose(result, 2.801736)
assert hydro_mw(0) == 0
```

**Interpretation and checks:** Expected: 28 m and 2.801736 MW. Using gross head would overestimate output.

::::

(exercise-03)=
## Exercise 03 — Solar panel and array output

**Difficulty:** Easy  
**Python skills:** functions, multiplication

**Reference:** Wade, Chapter 3, photovoltaic panels [@wade2003]; Foster et al., photovoltaic conversion [@foster2010].

**Task:** Twenty panels each have area 2 m² and efficiency 20% at plane-of-array irradiance 800 W/m². Calculate one panel and total DC kW. Hold efficiency constant and omit inverter and temperature losses.

```{code-cell} python
# Exercise 03: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 03

1. Calculate incident watts per panel from irradiance times area.
2. Multiply by efficiency and convert to kW.
3. Scale by panel count; include a zero-panel check.

```{code-cell} python
import math
def array_kw(count):
    if not isinstance(count, int) or count < 0:
        raise ValueError("panel count must be a non-negative integer")
    return 800 * 2 * 0.20 * count / 1000
print(f"Panel: {array_kw(1):.2f} kW; array: {array_kw(20):.2f} kW DC")
assert math.isclose(array_kw(20), 6.4)
assert array_kw(0) == 0
```

**Interpretation and checks:** Expected: 0.32 kW per panel and 6.4 kW DC. This is power, not daily energy.

::::

(exercise-04)=
## Exercise 04 — Wind speed and the cubic law

**Difficulty:** Easy  
**Python skills:** math.pi, powers, ratios

**Reference:** Manwell et al., Sections 2.2 and 3.2 [@manwell2009]; Wagner and Mathur, turbine fundamentals [@wagner2009].

**Task:** For rotor radius 20 m and density 1.225 kg/m³, calculate available wind kW at 5 and 10 m/s. Show the ratio. Do not apply a power coefficient: these are available kinetic-energy fluxes, not electricity.

```{code-cell} python
# Exercise 04: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 04

1. Calculate swept area as pi times radius squared.
2. Evaluate 0.5 × density × area × speed cubed.
3. Convert W to kW and divide the two results.

```{code-cell} python
import math
area_m2 = math.pi * 20**2
power_5_kw = 0.5 * 1.225 * area_m2 * 5**3 / 1000
power_10_kw = 0.5 * 1.225 * area_m2 * 10**3 / 1000
ratio = power_10_kw / power_5_kw
print(f"Available power: {power_5_kw:.2f}, {power_10_kw:.2f} kW; ratio={ratio:.0f}")
assert math.isclose(ratio, 8)
```

**Interpretation and checks:** Expected: about 96.21 and 769.69 kW, a factor of eight. A controlled generator does not follow this law beyond rated output.

::::

(exercise-05)=
## Exercise 05 — Geothermal thermal and net electrical power

**Difficulty:** Easy  
**Python skills:** sequential calculations

**Reference:** Grant and Bixley, Chapters 2–3, fluid heat and simplified models [@grant2011].

**Task:** Single-phase water flows at 50 kg/s and cools from 150 to 70 °C. Use cp = 4,180 J/(kg K), conversion efficiency 12% and parasitic load 10% of gross electricity. Calculate thermal, gross electrical and net electrical MW.

```{code-cell} python
# Exercise 05: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 05

1. Use the temperature difference with mass flow and specific heat.
2. Multiply thermal power by conversion efficiency.
3. Subtract the parasitic fraction of gross electrical power.

```{code-cell} python
import math
thermal_mw = 50 * 4180 * (150 - 70) / 1e6
gross_mw = thermal_mw * 0.12
net_mw = gross_mw * (1 - 0.10)
print(f"Thermal {thermal_mw:.3f}; gross {gross_mw:.4f}; net {net_mw:.5f} MW")
assert math.isclose(net_mw, 1.80576)
assert thermal_mw > gross_mw > net_mw > 0
```

**Interpretation and checks:** Expected: 16.72 MW thermal, 2.0064 MW gross and 1.80576 MW net. Phase change or brine properties require a more appropriate enthalpy model.

::::

(exercise-06)=
## Exercise 06 — Environmental flow and daily hydro energy

**Difficulty:** Medium  
**Python skills:** NumPy masks, clipping, integration

**Reference:** IFC, Chapter 7, hydrology and energy; Chapter 12, environmental impacts [@ifc2015].

**Task:** Daily mean river flows are [4, 8, 12, 20] m³/s. Reserve up to 5 m³/s for the environment, use 25 m net head, efficiency 0.9 and a 2 MW generator. Calculate usable flow, electrical power, four-day MWh and capacity factor. All samples represent 24-hour means.

```{code-cell} python
# Exercise 06: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 06

1. Subtract environmental flow and clamp usable flow at zero.
2. Calculate hydraulic conversion and cap output at 2 MW.
3. Integrate four 24-hour periods and use the 96-hour denominator.

```{code-cell} python
import numpy as np
river = np.array([4, 8, 12, 20], dtype=float)
usable = np.maximum(river - 5, 0)
power = np.minimum(1000 * 9.81 * usable * 25 * 0.9 / 1e6, 2)
energy = power.sum() * 24
cf = energy / (2 * 4 * 24)
print("Usable m3/s:", usable, "Power MW:", power)
print(f"Energy {energy:.3f} MWh; CF {cf:.2%}")
assert np.isclose(energy, 100.974)
assert np.all((power >= 0) & (power <= 2))
```

**Interpretation and checks:** Expected: 100.974 MWh and about 52.59%. A low river flow may be insufficient even for the environmental target; the model never invents extra water.

::::

(exercise-07)=
## Exercise 07 — Temperature-corrected PV and inverter clipping

**Difficulty:** Medium  
**Python skills:** arrays, DataFrame, minimum/maximum

**Reference:** Foster et al., photovoltaic performance [@foster2010]; pvlib PVWatts DC equation [@pvlibDocs].

**Task:** Use four hourly mean plane-of-array irradiances [0, 400, 800, 1000] W/m² and ambient temperatures [20, 22, 25, 28] °C. A 10 kW DC / 7 kW AC array has NOCT 45 °C, gamma = -0.004 per °C and inverter efficiency 0.97. Use Tc = Ta + G(45−20)/800. Tabulate cell temperature, DC and clipped AC kW; calculate four-hour energy.

```{code-cell} python
# Exercise 07: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 07

1. Estimate cell temperature separately for each interval.
2. Apply Pdc = 10(G/1000)[1 + gamma(Tc−25)].
3. Apply inverter efficiency before clipping; sum AC kW times one hour.

```{code-cell} python
import numpy as np
import pandas as pd
g = np.array([0, 400, 800, 1000], dtype=float)
ambient = np.array([20, 22, 25, 28], dtype=float)
cell = ambient + g * (45 - 20) / 800
dc = 10 * g / 1000 * np.maximum(1 - 0.004 * (cell - 25), 0)
ac = np.minimum(dc * 0.97, 7)
print(pd.DataFrame({"G_Wm2": g, "cell_C": cell, "DC_kW": dc, "AC_kW": ac}))
energy_kwh = ac.sum() * 1.0
print(f"Four-hour AC energy: {energy_kwh:.5f} kWh")
assert np.isclose(energy_kwh, 17.71656)
assert ac[0] == 0 and ac[-1] == 7
```

**Interpretation and checks:** The final interval clips at 7 kW. The temperature model is approximate; real performance also depends on wind, mounting, spectral effects and shading.

::::

(exercise-08)=
## Exercise 08 — A turbine curve with tested boundaries

**Difficulty:** Medium  
**Python skills:** functions, masks, boundary tests

**Reference:** Manwell et al., Sections 2.5 and 8.3 [@manwell2009].

**Task:** Implement a 3 MW turbine: zero below 3 m/s, cubic interpolation between 3 and 12, rated power from 12 to below 25, zero at or above 25. Use P = 3(v³−3³)/(12³−3³) in the rising region. Evaluate [0, 3, 6, 12, 24.9, 25] m/s and reject negative or non-finite speeds.

```{code-cell} python
# Exercise 08: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 08

1. Validate the array before evaluating the curve.
2. Apply separate masks for the rising and rated regions.
3. Check exact cut-in, rated and cut-out boundaries.

```{code-cell} python
import numpy as np
def turbine(speed):
    v = np.asarray(speed, dtype=float)
    if not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError("speed must be finite and non-negative")
    p = np.zeros_like(v)
    rising = (v >= 3) & (v < 12)
    p[rising] = 3 * (v[rising]**3 - 3**3) / (12**3 - 3**3)
    p[(v >= 12) & (v < 25)] = 3
    return p
output = turbine([0, 3, 6, 12, 24.9, 25])
print(output)
assert np.allclose(output, [0, 0, 1/3, 3, 3, 0])
for invalid in [-1, np.nan, np.inf]:
    try:
        turbine([invalid])
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input was accepted")
```

**Interpretation and checks:** This is an instructional curve. A manufacturer curve, air-density correction and site losses are required for real energy assessment.

::::

(exercise-09)=
## Exercise 09 — Energy from unequal intervals

**Difficulty:** Medium  
**Python skills:** arrays, validation, weighted mean

**Reference:** JICA, Section 3.1.3, energy generation [@jica2011]; NumPy guide [@numpyDocs].

**Task:** Interval-average powers [2, 4, 1, 3] MW persist for [0.25, 0.5, 1, 0.25] hours. Write a function checking equal one-dimensional shapes, finite values, non-negative power and positive durations. Return MWh and duration-weighted mean MW.

```{code-cell} python
# Exercise 09: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 09

1. Check lengths, shapes and numerical validity.
2. Sum power times its corresponding duration.
3. Divide energy by total duration for the mean power.

```{code-cell} python
import numpy as np
def integrate(power, duration):
    p, h = np.asarray(power, float), np.asarray(duration, float)
    if p.ndim != 1 or p.shape != h.shape or p.size == 0:
        raise ValueError("nonempty, equal one-dimensional shapes required")
    if not np.isfinite(p).all() or not np.isfinite(h).all():
        raise ValueError("finite values required")
    if np.any(p < 0) or np.any(h <= 0):
        raise ValueError("invalid generation or interval duration")
    energy = np.sum(p * h)
    return energy, energy / h.sum()
energy, mean = integrate([2, 4, 1, 3], [0.25, 0.5, 1, 0.25])
print(f"{energy:.2f} MWh; mean {mean:.3f} MW")
assert np.isclose(energy, 4.25) and np.isclose(mean, 2.125)
try:
    integrate([1, 2], [1])
except ValueError:
    pass
else:
    raise AssertionError("shape mismatch accepted")
```

**Interpretation and checks:** Expected: 4.25 MWh and 2.125 MW. The unweighted mean is incorrect here. Instantaneous measurements would require a stated interpolation/integration rule.

::::

(exercise-10)=
## Exercise 10 — A transparent solar-data quality check

**Difficulty:** Medium  
**Python skills:** pandas, duplicates, missing data

**Reference:** Wade, Chapter 10, monitoring and maintenance [@wade2003]; pandas indexing [@pandasDocs].

**Task:** Build UTC readings at 00:00, 01:00, 01:00, 03:00 on 1 June 2025 with irradiance [0, 200, 200, −5] W/m². Remove only the exact repeated timestamp/value row, reject any remaining duplicate time, mark negative irradiance missing and reindex to all four hours. Report completeness; do not impute or calculate a misleading full-period energy total.

```{code-cell} python
# Exercise 10: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 10

1. Remove the documented duplicate row, not arbitrary repeated timestamps.
2. Flag impossible values and explicitly reveal the missing 02:00 observation.
3. Count valid observations over all expected timestamps.

```{code-cell} python
import numpy as np
import pandas as pd
raw = pd.DataFrame({"time": pd.to_datetime([
    "2025-06-01 00:00Z", "2025-06-01 01:00Z",
    "2025-06-01 01:00Z", "2025-06-01 03:00Z"]),
    "G_Wm2": [0.0, 200.0, 200.0, -5.0]})
clean = raw.drop_duplicates().set_index("time").sort_index()
if clean.index.has_duplicates:
    raise ValueError("conflicting timestamp duplicates require investigation")
clean.loc[clean.G_Wm2 < 0, "G_Wm2"] = np.nan
expected = pd.date_range("2025-06-01", periods=4, freq="h", tz="UTC")
clean = clean.reindex(expected)
completeness = clean.G_Wm2.notna().mean()
print(clean)
print(f"Completeness: {completeness:.0%}")
assert clean.G_Wm2.isna().sum() == 2
assert completeness == 0.5 and len(raw) == 4
```

**Interpretation and checks:** Expected: 50% completeness. Missing at 02:00 and invalid at 03:00 are different problems, and neither is assumed to be zero.

::::

(exercise-11)=
## Exercise 11 — Compare SQL and pandas generation totals

**Difficulty:** Medium  
**Python skills:** SQLite, groupby, reconciliation

**Reference:** IFC, Chapter 7, energy accounting [@ifc2015]; SQLite SQL language [@sqliteDocs]; pandas [@pandasDocs].

**Task:** Create four records: solar 10 and 12 MWh on days 1 and 2, wind 20 and 18 MWh on those days. Store them in an in-memory SQLite table with a day/source primary key and a non-negative, non-null energy constraint. Compute totals by source in SQL and pandas and compare.

```{code-cell} python
# Exercise 11: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 11

1. Create a table with the intended row grain and constraints.
2. Use parameterized inserts, then GROUP BY source.
3. Sort both results by source before comparison.

```{code-cell} python
import sqlite3
import numpy as np
import pandas as pd
records = [(1, "solar", 10.0), (2, "solar", 12.0),
           (1, "wind", 20.0), (2, "wind", 18.0)]
df = pd.DataFrame(records, columns=["day", "source", "mwh"])
with sqlite3.connect(":memory:") as con:
    con.execute("CREATE TABLE generation (day INTEGER NOT NULL, "
                "source TEXT NOT NULL, mwh REAL NOT NULL CHECK(mwh >= 0), "
                "PRIMARY KEY(day, source))")
    con.executemany("INSERT INTO generation VALUES (?, ?, ?)", records)
    sql = pd.read_sql_query("SELECT source, SUM(mwh) AS mwh FROM generation "
                            "GROUP BY source ORDER BY source", con).set_index("source")
totals = df.groupby("source").mwh.sum().sort_index()
print(sql)
assert sql.index.tolist() == totals.index.tolist()
assert np.allclose(sql.mwh, totals)
assert totals.to_dict() == {"solar": 22.0, "wind": 38.0}
```

**Interpretation and checks:** Expected: solar 22 MWh, wind 38 MWh. Do not mix aggregate renewable totals with their component sources.

::::

(exercise-12)=
## Exercise 12 — Monthly PV performance ratio

**Difficulty:** Medium  
**Python skills:** pandas, ratios, labelled bar chart

**Reference:** Foster et al., PV systems [@foster2010]; PV performance conventions [@pvlibDocs].

**Task:** For a 100 kW DC array, monthly plane-of-array irradiation is [80, 120, 160] kWh/m² and AC generation [6400, 9600, 12000] kWh. Calculate PR = (Eac/Pdc)/(Hpoa/Gstc), with Gstc = 1 kW/m². Plot three monthly PR values. These are three illustrative months, not a complete year.

```{code-cell} python
# Exercise 12: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 12

1. Calculate reference yield from irradiation divided by STC irradiance.
2. Calculate final yield from AC energy divided by DC nameplate.
3. Divide the yields and label the dimensionless chart.

```{code-cell} python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame({"H_kWh_m2": [80, 120, 160],
                   "E_AC_kWh": [6400, 9600, 12000]}, index=["Jan", "Feb", "Mar"])
df["PR"] = (df.E_AC_kWh / 100) / (df.H_kWh_m2 / 1.0)
print(df)
df.PR.plot.bar(ylabel="Performance ratio (dimensionless)", ylim=(0, 1.1),
               title="Synthetic monthly PV performance")
plt.tight_layout()
plt.show()
assert np.allclose(df.PR, [0.8, 0.8, 0.75])
```

**Interpretation and checks:** March has lower normalized yield despite higher energy. Investigate resource, temperature and losses. PR above one is possible under some conditions and conventions; it is a flag to investigate, not an absolute physical impossibility.

::::

(exercise-13)=
## Exercise 13 — Pumped-storage water and electricity budget

**Difficulty:** Medium  
**Python skills:** energy conversion, round-trip efficiency

**Reference:** JRC 2025, Section 2.1, pumped-storage technology [@jrc2025hydro]; JICA, Chapter 3 [@jica2011].

**Task:** Move 100,000 m³ of water through constant head 100 m. Use density 1000 kg/m³, gravity 9.81 m/s², pump efficiency 0.85 and generation efficiency 0.90. Calculate stored gravitational MWh, required pumping MWh, recovered MWh and round-trip efficiency. Ignore other losses.

```{code-cell} python
# Exercise 13: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 13

1. Calculate potential energy in joules; divide by 3.6 billion for MWh.
2. Divide stored energy by pump efficiency for electrical input.
3. Multiply stored energy by generation efficiency and compare input/output.

```{code-cell} python
import math
stored_mwh = 1000 * 100000 * 9.81 * 100 / 3.6e9
pumping_mwh = stored_mwh / 0.85
recovered_mwh = stored_mwh * 0.90
round_trip = recovered_mwh / pumping_mwh
print(f"Stored {stored_mwh:.2f}, input {pumping_mwh:.3f}, recovered {recovered_mwh:.3f} MWh")
print(f"Round trip: {round_trip:.1%}")
assert math.isclose(stored_mwh, 27.25)
assert math.isclose(round_trip, 0.765)
assert pumping_mwh > stored_mwh > recovered_mwh
```

**Interpretation and checks:** Expected: 27.25 MWh stored and 76.5% round-trip efficiency. Efficiency multiplies across stages; it does not add.

::::

(exercise-14)=
## Exercise 14 — Geothermal decline scenarios

**Difficulty:** Medium  
**Python skills:** functions, arrays, scenario plots

**Reference:** Grant and Bixley, Sections 2.5 and 3.4, exploitation and simple models [@grant2011].

**Task:** A plant starts at 5 MW net. Compare 0%, 1% and 3% annual compounded power decline over 20 non-leap operating years, with 95% availability. Plot annual GWh and print cumulative GWh. Treat decline rates as assumptions, not calibrated reservoir forecasts.

```{code-cell} python
# Exercise 14: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 14

1. Create operating years 1 through 20.
2. Use P_y = 5(1−d)^(y−1).
3. Multiply by 8760 hours and availability; convert MWh to GWh.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
years = np.arange(1, 21)
totals = []
for decline in [0.0, 0.01, 0.03]:
    net_mw = 5 * (1 - decline)**(years - 1)
    energy_gwh = net_mw * 8760 * 0.95 / 1000
    totals.append(energy_gwh.sum())
    print(f"Decline {decline:.0%}: {energy_gwh.sum():.2f} GWh over 20 years")
    plt.plot(years, energy_gwh, label=f"{decline:.0%} per year")
plt.xlabel("Operating year")
plt.ylabel("Net energy (GWh/year)")
plt.legend()
plt.tight_layout()
plt.show()
assert np.isclose(totals[0], 832.2)
assert totals[0] > totals[1] > totals[2]
```

**Interpretation and checks:** All scenarios start at 41.61 GWh/year. The zero-decline total is 832.2 GWh. Thermal breakthrough, pressure changes and reinjection cannot be inferred from this assumed exponential curve.

::::

(exercise-15)=
## Exercise 15 — Wind direction as circular data

**Difficulty:** Medium  
**Python skills:** trigonometry, validation, sector counts

**Reference:** Manwell et al., Sections 2.4 and 2.8, directional wind data [@manwell2009]; NumPy [@numpyDocs].

**Task:** Calculate the equal-weight circular mean of [350, 10, 0] degrees, using meteorological directions from north clockwise. Bin them into eight compass sectors centered at 0°, 45°, etc. Reject empty/non-finite data and flag [90, 270] as an undefined mean.

```{code-cell} python
# Exercise 15: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 15

1. Convert angles to radians and average sine and cosine.
2. Check vector magnitude before atan2 and wrap to [0,360).
3. Offset by half a sector width before integer binning.

```{code-cell} python
import numpy as np
def circular_mean(degrees):
    x = np.asarray(degrees, float)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("nonempty finite directions required")
    rad = np.deg2rad(x)
    s, c = np.sin(rad).mean(), np.cos(rad).mean()
    if np.hypot(s, c) < 1e-12:
        raise ValueError("mean direction is undefined")
    return np.round(np.degrees(np.arctan2(s, c)) % 360, 10) % 360
directions = np.array([350, 10, 0])
mean = circular_mean(directions)
sectors = ((directions % 360 + 22.5) // 45).astype(int) % 8
counts = np.bincount(sectors, minlength=8)
print("Mean degrees:", mean, "N NE E SE S SW W NW counts:", counts)
assert mean == 0 and counts.tolist() == [3, 0, 0, 0, 0, 0, 0, 0]
try:
    circular_mean([90, 270])
except ValueError:
    print("Opposite winds have no unique mean direction")
else:
    raise AssertionError("undefined mean accepted")
```

**Interpretation and checks:** Expected: north. The arithmetic mean would be 120°. Equal-weight direction statistics differ from speed- or energy-weighted statistics; state which you use.

::::

(exercise-16)=
## Exercise 16 — Reservoir routing with a mass-balance check

**Difficulty:** Hard  
**Python skills:** stateful loops, conservation, DataFrame

**Reference:** JICA, Chapter 8, generation planning [@jica2011]; IFC, Chapter 7 [@ifc2015].

**Task:** Route six one-hour intervals with inflows [2, 0, 8, 16, 0, 0] m³/s, initial storage 20,000 m³ and maximum storage 50,000 m³. Each interval releases up to 1 m³/s for the environment first, then up to 3 m³/s through a turbine, limited by available water. Spill remaining excess storage. Assume constant net head 20 m, efficiency 0.9, zero evaporation, and within-hour inflow available for release. Report storage, environmental shortage, turbine power and spill; prove water conservation at every step.

```{code-cell} python
# Exercise 16: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 16

1. Add inflow volume to the previous storage.
2. Allocate environmental release, then turbine water; neither may exceed availability.
3. Spill water above capacity and carry the remaining storage forward.
4. Check every volume balance and convert turbine volume back to flow for power.

```{code-cell} python
import numpy as np
import pandas as pd
inflows = [2, 0, 8, 16, 0, 0]
storage, capacity, seconds = 20000.0, 50000.0, 3600.0
rows = []
for hour, inflow in enumerate(inflows):
    previous = storage
    available = previous + inflow * seconds
    environmental = min(1 * seconds, available)
    available -= environmental
    turbine_volume = min(3 * seconds, available)
    available -= turbine_volume
    spill = max(available - capacity, 0)
    storage = available - spill
    power_mw = 1000 * 9.81 * (turbine_volume / seconds) * 20 * 0.9 / 1e6
    assert np.isclose(previous + inflow * seconds,
                      storage + environmental + turbine_volume + spill)
    rows.append([hour, storage, 3600 - environmental, spill, power_mw])
result = pd.DataFrame(rows, columns=["hour", "storage_m3", "environmental_shortage_m3",
                                    "spill_m3", "power_MW"])
print(result)
print(f"Energy {result.power_MW.sum():.5f} MWh")
assert result.storage_m3.between(0, capacity).all()
assert np.isclose(result.spill_m3.sum(), 7600)
assert np.isclose(storage, 21200)
assert np.isclose(result.power_MW.sum(), 3.09996)
```

**Interpretation and checks:** The operating rule prioritizes ecology over generation; it is not an optimal dispatch. Constant head omits the storage–elevation relation. Intra-hour timing, outlet limits and flood rules matter in real routing.

::::

(exercise-17)=
## Exercise 17 — Hybrid supply with a lossy battery

**Difficulty:** Hard  
**Python skills:** dispatch loops, balances, constraints

**Reference:** Wade, Chapters 5 and 9 [@wade2003]; Manwell et al., Sections 10.3 and 10.7 [@manwell2009]; Hu et al., complementary-system chapters [@hu2026].

**Task:** For six one-hour intervals, renewable generation is [0, 2, 5, 6, 1, 0] MW and demand is 2 MW throughout. Start a 4 MWh usable battery empty, with 2 MW AC charge/discharge limits and charge/discharge efficiencies each 0.9. Serve load directly, charge only from surplus, then discharge against deficit. Tabulate state of charge, curtailment and unmet load. Reconcile bus-side energy and stored energy at every interval.

```{code-cell} python
# Exercise 17: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 17

1. Serve demand directly from renewable output.
2. Limit charging by surplus, power and remaining storage divided by charge efficiency.
3. Limit discharge by deficit, power and stored energy times discharge efficiency.
4. Track conversion losses and verify two independent energy balances.

```{code-cell} python
import numpy as np
import pandas as pd
generation = np.array([0, 2, 5, 6, 1, 0], float)
demand = np.full(6, 2.0)
dt, capacity, limit, eta = 1.0, 4.0, 2.0, 0.9
soc = 0.0
rows = []
for g, d in zip(generation, demand):
    initial = soc
    direct = min(g, d)
    surplus, deficit = g - direct, d - direct
    charge = min(surplus, limit, (capacity - soc) / (eta * dt))
    soc += charge * eta * dt
    discharge = min(deficit, limit, soc * eta / dt)
    soc -= discharge * dt / eta
    curtailed, unmet = surplus - charge, deficit - discharge
    loss = charge * dt * (1 - eta) + discharge * dt * (1 / eta - 1)
    assert np.isclose(g + discharge, (d - unmet) + charge + curtailed)
    assert np.isclose(initial + charge * dt, soc + discharge * dt + loss)
    assert -1e-12 <= soc <= capacity + 1e-12
    rows.append([soc, charge, discharge, curtailed, unmet, loss])
result = pd.DataFrame(rows, columns=["SOC_MWh", "charge_MW", "discharge_MW",
                                    "curtailed_MW", "unmet_MW", "loss_MWh"])
print(result.round(4))
assert np.isclose(result.unmet_MW.sum() * dt, 2.0)
assert np.isclose(result.curtailed_MW.sum() * dt, 3.0)
assert np.isclose(soc, 0.2666666667)
```

**Interpretation and checks:** Expected: 2 MWh unmet, 3 MWh curtailed and about 0.267 MWh left in storage. Ending stored energy is not yet delivered electricity. Six synthetic hours cannot establish annual reliability; initial/final storage must be comparable when ranking scenarios.

::::

(exercise-18)=
## Exercise 18 — Wind AEP sensitivity with a Weibull resource

**Difficulty:** Hard  
**Python skills:** probability bins, convergence, scenario analysis

**Reference:** Manwell et al., Sections 2.4–2.5, wind statistics and energy estimation [@manwell2009].

**Task:** Use Weibull CDF F(v)=1−exp[−(v/c)^k], shape k=2 and scales c=7, 8, 9 m/s. Integrate the exercise-8 3 MW curve using bin probabilities from CDF differences and midpoint powers over 0–30 m/s. Compare 0.1 and 0.05 m/s bins. Use 8760 hours and a single combined retained-energy factor 0.90. Report net GWh and capacity factor; explain the omitted tail.

```{code-cell} python
# Exercise 18: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 18

1. Calculate probability mass per bin, not PDF values treated as probabilities.
2. Evaluate the turbine curve at bin midpoints and sum power times probability.
3. Apply annual hours and the combined loss factor once.
4. Refine the grid and compare the answers for numerical convergence.

```{code-cell} python
import numpy as np
def aep(scale, width):
    edges = np.linspace(0, 30, round(30 / width) + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    cdf = 1 - np.exp(-(edges / scale)**2)
    mass = np.diff(cdf)
    power = np.zeros_like(centers)
    ramp = (centers >= 3) & (centers < 12)
    power[ramp] = 3 * (centers[ramp]**3 - 3**3) / (12**3 - 3**3)
    power[(centers >= 12) & (centers < 25)] = 3
    net_mwh = np.sum(power * mass) * 8760 * 0.90
    assert np.isclose(mass.sum() + np.exp(-(30 / scale)**2), 1)
    return net_mwh
results = []
for scale in [7, 8, 9]:
    coarse, fine = aep(scale, 0.1), aep(scale, 0.05)
    relative_change = abs(fine - coarse) / fine
    cf = fine / (3 * 8760)
    results.append(fine)
    print(f"c={scale}: {fine/1000:.3f} GWh; CF={cf:.2%}; grid change={relative_change:.3%}")
    assert relative_change < 0.001 and 0 < cf < 1
assert results[0] < results[1] < results[2]
```

**Interpretation and checks:** The tail above 30 m/s contributes zero because this turbine stops at 25 m/s; do not renormalize the truncated probabilities. These are sensitivity scenarios, not confidence intervals or a measured resource forecast.

::::

(exercise-19)=
## Exercise 19 — Fit a geothermal decline model without future leakage

**Difficulty:** Hard  
**Python skills:** log-linear fit, holdout, baseline comparison

**Reference:** Grant and Bixley, Section 3.4 and model applicability in Section 3.9 [@grant2011]; NumPy [@numpyDocs].

**Task:** Generate ten synthetic annual net powers P_y=5×0.98^(y−1) MW. Fit log(P) = a + b(y−1) using only years 1–7. Predict years 8–10, compare MAE with a fixed last-training-value forecast, and recover annual decline 1−exp(b). Use this noiseless example to learn the procedure, not to claim real forecasting accuracy.

```{code-cell} python
# Exercise 19: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 19

1. Split chronologically before fitting.
2. Fit a line to positive training powers in log space.
3. Transform held-out predictions back to MW.
4. Compare with the last observed training value and discuss unrealistic noiseless accuracy.

```{code-cell} python
import numpy as np
years = np.arange(1, 11)
power = 5 * 0.98**(years - 1)
train_years, test_years = years[:7], years[7:]
train, test = power[:7], power[7:]
if np.any(train <= 0):
    raise ValueError("log model needs strictly positive powers")
slope, intercept = np.polyfit(train_years - 1, np.log(train), 1)
prediction = np.exp(intercept + slope * (test_years - 1))
baseline = np.full(test.shape, train[-1])
model_mae = np.mean(np.abs(test - prediction))
baseline_mae = np.mean(np.abs(test - baseline))
decline = 1 - np.exp(slope)
print(f"Decline={decline:.2%}; model MAE={model_mae:.6f} MW; baseline MAE={baseline_mae:.4f} MW")
assert train_years.max() < test_years.min()
assert np.isclose(decline, 0.02) and model_mae < 1e-10
assert baseline_mae > model_mae
```

**Interpretation and checks:** The fitted decline is 2% and error is nearly zero only because the synthetic data exactly follow the model. Real data require uncertainty, rolling evaluation and reservoir/process knowledge; log transforms also affect error assumptions.

::::

(exercise-20)=
## Exercise 20 — Discounted PV cost with degradation and sensitivity

**Difficulty:** Hard  
**Python skills:** discounting, functions, scenario tables

**Reference:** IFC, Chapter 14, economic analysis [@ifc2015]; IRENA cost methodology [@irena2026]; Wade, Chapter 9 [@wade2003].

**Task:** An illustrative PV project costs €1,000,000 at year 0, has annual O&M €20,000, first-year net AC output 1,500 MWh and 0.5% annual degradation over 25 years. In constant euros, use real discount rates 0%, 3%, 6%, 9%. Calculate LCOE = discounted costs / discounted energy. Put annual O&M and energy at each year end. Omit taxes, financing cash flows, replacements, salvage and decommissioning, and state these limits.

```{code-cell} python
# Exercise 20: write your solution here.
# 1. Enter the given values with units in variable names.
# 2. Calculate the requested quantities.
# 3. Print results, add a check, and explain the assumptions.
```

::::{dropdown} Solution — Exercise 20

1. Generate years 1–25 and compounded energy.
2. Discount each annual cost and energy quantity with the same rate.
3. Add year-0 capital cost without discounting.
4. Compare scenarios and use an undiscounted zero-degradation case as an independent check.

```{code-cell} python
import numpy as np
import pandas as pd
def lcoe(rate, degradation=0.005):
    if not np.isfinite(rate) or rate < 0 or not 0 <= degradation < 1:
        raise ValueError("invalid rate or degradation")
    years = np.arange(1, 26)
    energy = 1500 * (1 - degradation)**(years - 1)
    discount = (1 + rate)**years
    present_cost = 1000000 + np.sum(20000 / discount)
    present_energy = np.sum(energy / discount)
    return present_cost / present_energy
rates = [0.0, 0.03, 0.06, 0.09]
result = pd.DataFrame({"real_rate": rates,
                       "LCOE_EUR_MWh": [lcoe(r) for r in rates]})
print(result.round(3))
# Independent hand check: (1,000,000 + 25*20,000) / (25*1,500).
assert np.isclose(lcoe(0, degradation=0), 40.0)
assert np.all(np.diff(result.LCOE_EUR_MWh) > 0)
```

**Interpretation and checks:** Higher discount rates increase LCOE in this example. LCOE is not a tariff, profit estimate or complete grid-system cost. Inputs are teaching assumptions, not current market quotations; compare projects only with consistent currency, year and system boundaries.

::::

## Before the final project

Choose one calculation, explain its units without code, alter one assumption and predict the effect. Then complete the [hybrid portfolio project](../section8/projectIntro.md). Full publication details are in the [reference catalogue](../references.md).
