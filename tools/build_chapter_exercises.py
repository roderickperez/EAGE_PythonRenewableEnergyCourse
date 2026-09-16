"""Maintain 20 original exercises inside each of the five teaching chapters.

The source chapters are MyST computational notebooks. Each solution is standalone.
Only practice answers belong here; assessment answers remain instructor-local.
"""
from pathlib import Path
from textwrap import dedent
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1] / 'EAGE_PythonRenewableEnergyCourse'
BANK = {key: [] for key in ['general', 'solar', 'hydro', 'wind', 'geothermal']}
REFERENCES = {
    'general': 'Energy accounting: [@jica2011], Chapter 3; Python [@pythonDocs]; arrays and data [@numpyDocs; @pandasDocs].',
    'solar': 'PV conversion and systems: [@foster2010; @wade2003]; temperature model [@pvlibDocs].',
    'hydro': 'Hydropower equations and generation planning: [@jica2011], Chapters 3 and 8; hydrology and environmental flow [@ifc2015], Chapters 7 and 12.',
    'wind': 'Wind resource, aerodynamics and energy estimation: [@manwell2009], Chapters 2–3; turbine operation [@wagner2009].',
    'geothermal': 'Fluid heat, exploitation and simplified models: [@grant2011], Chapters 2–3. Decline rates and efficiencies below are teaching assumptions.',
}

def add(topic, title, prompt, code, interpretation):
    code = dedent(code).strip()
    steps = re.findall(r'^# Step \d+: (.+)$', code, re.M)
    assert len(steps) >= 3, title
    BANK[topic].append(dict(title=title, prompt=prompt, code=code,
                           steps=steps, interpretation=interpretation))

# GENERAL: five easy, ten medium, five hard.
add('general', 'Convert energy units', 'Convert 2.5 MWh to kWh and MJ. Use 1 kWh = 3.6 MJ.', '''
# Step 1: Store the energy in MWh.
energy_mwh = 2.5
# Step 2: Convert to kWh, then MJ.
energy_kwh = energy_mwh * 1000
energy_mj = energy_kwh * 3.6
# Step 3: Print labelled results and check the conversion.
print(energy_kwh, 'kWh;', energy_mj, 'MJ')
assert energy_kwh == 2500 and energy_mj == 9000
''', 'Energy is conserved by a unit conversion: 2,500 kWh = 9,000 MJ.')
add('general', 'Power over a time interval', 'A generator averages 3 MW for 6 hours. Calculate MWh and mean MW from the energy.', '''
# Step 1: Enter interval-average power and duration.
power_mw, hours = 3, 6
# Step 2: Multiply power by elapsed time.
energy_mwh = power_mw * hours
# Step 3: Recover mean power and verify the result.
print(energy_mwh, 'MWh;', energy_mwh / hours, 'MW')
assert energy_mwh == 18
''', '18 MWh is energy; 3 MW is power.')
add('general', 'Daily capacity factor', 'A 5 MW plant delivers 48 MWh in 24 hours. Calculate its daily capacity factor.', '''
# Step 1: Calculate maximum possible energy over this day.
maximum_mwh = 5 * 24
# Step 2: Divide actual energy by that maximum.
capacity_factor = 48 / maximum_mwh
# Step 3: Print a percentage and check bounds.
print(f'{capacity_factor:.1%}')
assert abs(capacity_factor - 0.4) < 1e-12
''', '40% describes this day, not the annual resource or conversion efficiency.')
add('general', 'Conversion losses', 'A device receives 100 MWh and converts 85% to useful energy. Calculate output and losses.', '''
# Step 1: Enter input energy and fractional efficiency.
input_mwh, efficiency = 100, 0.85
# Step 2: Calculate useful energy and the remainder.
output_mwh = input_mwh * efficiency
loss_mwh = input_mwh - output_mwh
# Step 3: Reconcile the energy balance.
print(output_mwh, 'MWh useful;', loss_mwh, 'MWh losses')
assert output_mwh + loss_mwh == input_mwh
''', '85 MWh useful and 15 MWh lost; efficiency is dimensionless.')
add('general', 'Generation shares', 'Solar, hydro, wind and geothermal produce 10, 30, 40 and 20 MWh. Print each source share.', '''
# Step 1: Store source energies in a dictionary.
energy = dict(solar=10, hydro=30, wind=40, geothermal=20)
# Step 2: Sum the energy and divide each source by the total.
total = sum(energy.values())
shares = {name: value / total for name, value in energy.items()}
# Step 3: Print percentages and confirm they sum to one.
for name, share in shares.items():
    print(name, f'{share:.0%}')
assert abs(sum(shares.values()) - 1) < 1e-12
''', 'Shares use an energy denominator, not installed capacity.')
add('general', 'Unequal sampling intervals', 'Average powers [2,4,1] MW last [0.5,1,2] hours. Calculate total energy and time-weighted average power.', '''
import numpy as np
# Step 1: Represent power and duration as matching arrays.
p = np.array([2, 4, 1]); dt = np.array([0.5, 1, 2])
# Step 2: Integrate interval energy and divide by elapsed time.
energy = np.sum(p * dt); mean_power = energy / dt.sum()
# Step 3: Verify energy against a hand calculation.
print(energy, 'MWh;', mean_power, 'MW')
assert energy == 7 and mean_power == 2
''', 'The arithmetic mean of the three powers is inappropriate for unequal durations.')
add('general', 'Read generation from CSV', 'Read this CSV using pandas: source,mwh with rows solar,12; wind,18; hydro,10. Calculate total energy.', '''
from io import StringIO
import pandas as pd
# Step 1: Load the supplied CSV text.
csv = 'source,mwh\\nsolar,12\\nwind,18\\nhydro,10\\n'
df = pd.read_csv(StringIO(csv))
# Step 2: Validate values before aggregating.
assert df.source.is_unique and df.mwh.notna().all() and df.mwh.ge(0).all()
total = df.mwh.sum()
# Step 3: Print the table and checked total.
print(df.to_string(index=False)); print(total, 'MWh')
assert total == 40
''', 'StringIO behaves like a file; replace it with a CSV path for local data.')
add('general', 'Missing generation is not zero', 'For [1, missing,3,2] hourly MW readings, report coverage and observed energy, but do not invent a four-hour total.', '''
import pandas as pd
# Step 1: Preserve the missing measurement.
p = pd.Series([1, None, 3, 2], dtype=float)
# Step 2: Calculate coverage and energy for observed hours only.
coverage = p.notna().mean(); observed_mwh = p.sum(min_count=1)
# Step 3: Label the result and verify it.
print(f'Coverage: {coverage:.0%}; observed energy: {observed_mwh} MWh')
assert coverage == 0.75 and observed_mwh == 6
''', '6 MWh covers three observed hours; the full four-hour total is unknown.')
add('general', 'Exact duplicate readings', 'Times [00,01,01,02] have MW [1,2,2,3]. Remove only exact timestamp/value duplicates and check uniqueness.', '''
import pandas as pd
# Step 1: Build a timestamped table.
df = pd.DataFrame({'time': pd.to_datetime(['2025-01-01 00:00', '2025-01-01 01:00', '2025-01-01 01:00', '2025-01-01 02:00'], utc=True), 'mw': [1,2,2,3]})
# Step 2: Remove exact repeated records, then check for conflicting times.
clean = df.drop_duplicates()
assert clean.time.is_unique
# Step 3: Sum the remaining hourly mean powers.
print(clean.to_string(index=False)); print(clean.mw.sum(), 'MWh')
assert len(clean) == 3 and clean.mw.sum() == 6
''', 'Different values at one timestamp would require investigation, not arbitrary deletion.')
add('general', 'Match supply and demand', 'Hourly renewable supply is [1,5,2] MW and demand [3,3,3] MW. Calculate served, curtailed and unmet energy.', '''
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
''', 'Results are 6, 2 and 3 MWh; total supply alone cannot establish demand coverage.')
add('general', 'Aggregate daily energy', 'Generate 48 consecutive UTC hourly means of 2 MW and aggregate to daily MWh.', '''
import pandas as pd
# Step 1: Create an hourly UTC index and matching mean powers.
p = pd.Series(2.0, index=pd.date_range('2025-01-01', periods=48, freq='h', tz='UTC'))
# Step 2: Multiply by one hour and sum each day.
daily = (p * 1.0).resample('D').sum(min_count=24)
# Step 3: Check complete coverage and print daily energy.
print(daily.rename('MWh'))
assert (daily == 48).all() and daily.sum() == 96
''', 'A complete UTC day has 24 hourly intervals; local daylight-saving days require care.')
add('general', 'Compare fleet capacity factors', 'Plant A: 2 MW and 24 MWh; plant B: 8 MW and 144 MWh over the same 24 hours. Calculate plant and fleet CF.', '''
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
''', 'Fleet CF is capacity-weighted when the time window is the same.')
add('general', 'SQL and pandas agreement', 'Group solar energies [10,12] and wind energies [20,18] MWh by source using both SQL and pandas.', '''
import sqlite3
import pandas as pd
# Step 1: Create a small table of source observations.
df = pd.DataFrame({'source': ['solar','solar','wind','wind'], 'mwh': [10,12,20,18]})
# Step 2: Aggregate independently using SQL and pandas.
with sqlite3.connect(':memory:') as con:
    df.to_sql('energy', con, index=False)
    sql = pd.read_sql_query('SELECT source, SUM(mwh) AS mwh FROM energy GROUP BY source ORDER BY source', con).set_index('source').mwh
expected = df.groupby('source').mwh.sum().sort_index()
# Step 3: Reconcile totals and display the result.
pd.testing.assert_series_equal(sql, expected)
print(sql)
''', 'Solar totals 22 MWh and wind 38 MWh; matching code outputs still requires valid source definitions.')
add('general', 'Forecast benchmark', 'Use monthly MWh [10,12,14,16] as training and [15,17] as test. Compare fixed last-value and training-mean forecast MAE.', '''
import numpy as np
# Step 1: Separate earlier training observations from later test observations.
train = np.array([10,12,14,16]); test = np.array([15,17])
# Step 2: Construct forecasts without reading test values.
last_forecast = np.repeat(train[-1], len(test)); mean_forecast = np.repeat(train.mean(), len(test))
# Step 3: Evaluate both forecasts on the same test window.
last_mae = np.abs(test-last_forecast).mean(); mean_mae = np.abs(test-mean_forecast).mean()
print(last_mae, mean_mae, 'MWh MAE')
assert last_mae == 1 and mean_mae == 3
''', 'Two test months illustrate evaluation; they do not establish long-term forecast skill.')
add('general', 'Plot a source mix', 'Create a labelled bar chart of solar 20, hydro 35, wind 30 and geothermal 15 MWh, and check the total.', '''
import matplotlib.pyplot as plt
# Step 1: Store source labels and comparable energies.
sources = ['Solar','Hydro','Wind','Geothermal']; energy = [20,35,30,15]
# Step 2: Plot values with an explicit energy unit.
fig, ax = plt.subplots(figsize=(6,3)); ax.bar(sources, energy)
ax.set(ylabel='Energy (MWh)', title='Synthetic one-day generation')
fig.tight_layout(); plt.show()
# Step 3: Verify the plotted total.
print(sum(energy), 'MWh'); assert sum(energy) == 100
''', 'A common energy unit permits comparison; this synthetic mix is not a regional statistic.')
add('general', 'Battery dispatch with losses', 'Supply [0,4,4,0] MW meets constant 2 MW demand. Start empty with 3 MWh usable storage, 2 MW charge/discharge limits and 90% efficiency each way. Route four hours and prove energy balance.', '''
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
    assert np.isclose(supply+discharge, direct+discharge+charge+curtailed)
    assert np.isclose(soc-before, charge*0.9-discharge/0.9)
    rows.append((soc, curtailed, unmet))
# Step 3: Check limits and report energy outcomes.
assert all(0 <= row[0] <= 3+1e-12 for row in rows)
print('SOC MWh, curtailment MWh, unmet MWh by hour:', rows)
assert np.isclose(sum(r[2] for r in rows), 2)
''', 'Charging losses consume supply and discharge losses consume stored energy; final stored energy is not delivered energy.')
add('general', 'Discounted energy cost', 'For €1000 initial cost, €20 annual O&M and 100 MWh/year over 10 years, calculate LCOE at real rates 0%,5%,10%, with year-end flows.', '''
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
''', 'Teaching costs are intentionally small. This omits replacements, taxes and end-of-life costs; see IFC Chapter 14 for economic boundaries.')
add('general', 'Strict multi-file alignment', 'Join two CSV-like tables with hourly UTC times: solar [0,1,2] and wind [2,1,0]. Reject duplicate times or mismatched coverage before calculating their sum.', '''
import pandas as pd
import numpy as np
# Step 1: Create separate source tables with their own timestamp columns.
time = pd.date_range('2025-01-01', periods=3, freq='h', tz='UTC')
solar = pd.DataFrame({'time':time, 'solar_mw':[0,1,2]})
wind = pd.DataFrame({'time':time, 'wind_mw':[2,1,0]})
# Step 2: Validate matching coverage and perform a one-to-one join.
assert solar.time.is_unique and wind.time.is_unique
assert solar.time.equals(wind.time)
joined = solar.merge(wind, on='time', validate='one_to_one')
joined['total_mw'] = joined.solar_mw + joined.wind_mw
# Step 3: Test both completeness and the known sum.
assert len(joined)==3 and np.allclose(joined.total_mw, 2)
print(joined.to_string(index=False))
''', 'An inner join alone can silently discard unmatched times; coverage must be checked first.')
add('general', 'Capacity search under a reliability target', 'Wind produces [0,1,2,1] MW per unit; firm geothermal supplies 1 MW and demand is 2 MW each hour. Search 0–4 wind units for coverage at least 80%. Report the smallest feasible count, unmet and curtailed MWh.', '''
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
''', 'No amount of this wind shape fixes the zero-wind hour. A coverage target is not the same as hourly reliability.')
add('general', 'Monte Carlo uncertainty', 'Assume 1000 independent illustrative annual-energy outcomes uniformly distributed between 80 and 120 GWh. Use seed 7; report mean, 5th and 95th percentiles and probability of less than 90 GWh.', '''
import numpy as np
# Step 1: Sample a stated synthetic uncertainty model reproducibly.
rng = np.random.default_rng(7); energy = rng.uniform(80,120,1000)
# Step 2: Summarise the distribution and threshold frequency.
interval = np.quantile(energy,[0.05,0.95]); probability = np.mean(energy<90)
# Step 3: Check physical support and explain the probability convention.
print('Mean GWh:', energy.mean(), '5th/95th GWh:', interval, 'P(E<90):', probability)
assert energy.min()>=80 and energy.max()<=120 and 0<=probability<=1
''', 'These are simulation percentiles under an assumed distribution, not a confidence interval or calibrated project P90.')

# Solar exercises.
add('solar','Panel DC power','At 800 W/m², a 2 m² panel has efficiency 20%. Calculate DC watts.', '''
# Step 1: Store irradiance, area and efficiency.
g, area, eta = 800, 2, 0.20
# Step 2: Multiply incident power by conversion efficiency.
power_w = g*area*eta
# Step 3: Label and verify the result.
print(power_w, 'W DC'); assert power_w == 320
''','320 W assumes constant efficiency and uniform irradiance.')
add('solar','Array nameplate','An array contains 24 modules rated at 400 W DC. Calculate its STC rating in kW.', '''
# Step 1: Enter module count and rating.
count, module_w = 24, 400
# Step 2: Add module ratings and convert to kW.
array_kw = count*module_w/1000
# Step 3: Print and check the nameplate.
print(array_kw,'kW DC'); assert array_kw == 9.6
''','Nameplate describes STC; actual operating output varies.')
add('solar','Daily irradiation','Four hourly mean irradiances are 0,200,600,400 W/m². Calculate four-hour irradiation in kWh/m².', '''
# Step 1: Store hourly means and duration.
irradiance = [0,200,600,400]; hours = 1
# Step 2: Integrate and convert Wh to kWh.
irradiation = sum(irradiance)*hours/1000
# Step 3: Verify and label the four-hour result.
print(irradiation,'kWh/m²'); assert irradiation == 1.2
''','This is a four-hour window, not necessarily a complete day.')
add('solar','Inverter conversion','A PV array delivers 5 kW DC to a 96%-efficient inverter with a 6 kW AC limit. Find AC kW.', '''
# Step 1: Enter the DC input and equipment assumptions.
dc_kw, efficiency, rating_kw = 5, 0.96, 6
# Step 2: Apply conversion efficiency and the output limit.
ac_kw = min(dc_kw*efficiency, rating_kw)
# Step 3: Check the output cannot exceed input or rating.
print(ac_kw,'kW AC'); assert ac_kw == 4.8 and ac_kw <= dc_kw
''','No clipping occurs in this example; efficiency is held constant.')
add('solar','Capacity factor from AC energy','A 10 kW AC plant exports 48 kWh in 24 hours. Calculate AC-based CF and state the denominator.', '''
# Step 1: Set exported energy, AC capacity and duration.
energy_kwh, capacity_kw, hours = 48, 10, 24
# Step 2: Calculate the AC-based capacity factor.
cf = energy_kwh/(capacity_kw*hours)
# Step 3: Print a labelled percentage.
print(f'AC-based CF: {cf:.0%}'); assert cf == 0.2
''','20% uses AC nameplate. A DC-based capacity factor would have a different denominator.')
add('solar','Cell-temperature estimate','At ambient 25°C and irradiance [0,400,800] W/m², use NOCT 45°C and Tc=Ta+G(NOCT−20)/800. Calculate temperatures.', '''
import numpy as np
# Step 1: Store irradiance and temperature parameters.
g = np.array([0,400,800]); ambient = 25; noct = 45
# Step 2: Apply the approximate NOCT relationship.
cell = ambient+g*(noct-20)/800
# Step 3: Verify the 800 W/m² case.
print(cell,'°C'); assert np.allclose(cell,[25,37.5,50])
''','This simple thermal relationship omits wind and mounting effects.')
add('solar','Temperature coefficient','At 1000 W/m², a 10 kW STC array has gamma −0.004/°C. Evaluate DC output at cell temperatures [25,45,65]°C.', '''
import numpy as np
# Step 1: Convert temperature cases to an array.
temperature = np.array([25,45,65]); gamma = -0.004
# Step 2: Apply the temperature multiplier relative to 25°C.
dc_kw = 10*(1+gamma*(temperature-25))
# Step 3: Check direction and expected output.
print(dc_kw,'kW DC'); assert np.allclose(dc_kw,[10,9.2,8.4])
''','Higher cell temperature reduces output for a negative power coefficient.')
add('solar','Clipping budget','Hourly DC output [0,4,8,10] kW passes through a 97% inverter capped at 7 kW AC. Calculate AC kWh and clipping kWh.', '''
import numpy as np
# Step 1: Calculate AC power before applying the rating.
dc = np.array([0,4,8,10]); available_ac = dc*0.97
# Step 2: Clip output and calculate discarded potential AC energy.
ac = np.minimum(available_ac,7); clipping = available_ac-ac
# Step 3: Check energy accounting across one-hour intervals.
print(ac.sum(),'kWh AC;',clipping.sum(),'kWh clipping')
assert np.allclose(ac+clipping,available_ac) and np.isclose(ac.sum(),17.88)
''','Clipping is measured after modeled inverter conversion; it is distinct from inverter losses.')
add('solar','Sequential losses','Start with 1000 kWh DC; apply 3% soiling, 2% wiring and 4% inverter losses sequentially. Compare with simply subtracting 9%.', '''
import numpy as np
# Step 1: Express loss fractions as retained fractions.
retained = 1-np.array([0.03,0.02,0.04])
# Step 2: Multiply the successive retained fractions.
output = 1000*np.prod(retained); additive = 1000*(1-0.09)
# Step 3: Print the difference and check the product.
print(output,'kWh;',output-additive,'kWh difference')
assert np.isclose(output,912.576)
''','Multiplicative losses apply to changing energy bases; simple addition is an approximation.')
add('solar','Performance ratio','A 100 kW DC array exports 12,000 kWh with POA irradiation 150 kWh/m². Use reference irradiance 1 kW/m² to calculate PR.', '''
# Step 1: Calculate final yield in hours from AC energy and DC rating.
final_yield = 12000/100
# Step 2: Calculate reference yield and their ratio.
reference_yield = 150/1; pr = final_yield/reference_yield
# Step 3: Verify and display the dimensionless result.
print(f'PR: {pr:.0%}'); assert pr == 0.8
''','PR normalises for irradiation; it is not cell conversion efficiency.')
add('solar','Specific yield','Two systems generate [4000,9000] kWh from [5,10] kW DC. Calculate kWh/kWp and identify the higher specific yield.', '''
import numpy as np
# Step 1: Store energy and DC nameplate values.
energy = np.array([4000,9000]); capacity = np.array([5,10])
# Step 2: Normalise production by installed capacity.
specific_yield = energy/capacity
# Step 3: Compare the normalised values.
print(specific_yield,'kWh/kWp'); assert np.argmax(specific_yield)==1
''','The periods must match before comparing yields; higher yield does not alone establish lower cost.')
add('solar','Unequal daylight intervals','Average AC powers [1,4,2] kW persist for [0.5,2,1.5] hours. Integrate their energy.', '''
import numpy as np
# Step 1: Store paired power and duration arrays.
power = np.array([1,4,2]); duration = np.array([0.5,2,1.5])
# Step 2: Multiply each interval before summing.
energy = (power*duration).sum()
# Step 3: Verify the hand calculation.
print(energy,'kWh'); assert energy == 11.5
''','A sum of kW samples without duration is not energy for unequal intervals.')
add('solar','Nighttime and invalid data','For readings [0,200,missing,−10] W/m², preserve night zero, mark negative values missing, and report valid fraction.', '''
import pandas as pd
# Step 1: Load readings without replacing missing values.
g = pd.Series([0,200,None,-10],dtype=float)
# Step 2: Flag negatives and mask them.
invalid = g.lt(0); clean = g.mask(invalid)
# Step 3: Check the valid night zero and report coverage.
print(clean, 'Valid fraction:', clean.notna().mean())
assert clean.iloc[0]==0 and clean.notna().mean()==0.5
''','Zero at night is a valid observation; a missing daylight observation cannot be treated as zero.')
add('solar','Compounded degradation','First-year output is 100 MWh. With 0.5% annual degradation, calculate output in years 1–5 and cumulative MWh.', '''
import numpy as np
# Step 1: Index years with no degradation in the first year.
years = np.arange(1,6)
# Step 2: Apply annual retained output multiplicatively.
energy = 100*0.995**(years-1)
# Step 3: Check the first and last years and print the sum.
print(energy,'MWh/year;',energy.sum(),'MWh total')
assert energy[0]==100 and np.isclose(energy[-1],98.0149500625)
''','The rate is an assumption; weather and availability are held constant.')
add('solar','Daily profile plot','Model 24 hourly mean irradiances as max(0,800 sin(pi(h−6)/12)). Use 20 m² and 20% efficiency. Plot DC kW and integrate MWh.', '''
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Generate a labelled synthetic hourly irradiance profile.
h = np.arange(24); g = np.maximum(0,800*np.sin(np.pi*(h-6)/12))
# Step 2: Convert incident radiation to DC power and energy.
dc_kw = g*20*0.2/1000; energy_mwh = dc_kw.sum()/1000
# Step 3: Plot with units and verify the maximum.
fig,ax=plt.subplots(figsize=(6,3)); ax.plot(h,dc_kw)
ax.set(xlabel='UTC hour',ylabel='DC power (kW)',title='Synthetic PV day'); fig.tight_layout(); plt.show()
print(energy_mwh,'MWh'); assert np.isclose(dc_kw.max(),3.2)
''','The irradiance values are treated as interval means; no real location or date is represented.')
add('solar','Size an inverter by clipping target','For hourly DC [0,2,5,9,12,8,3,0] kW and 97% efficiency, search integer AC ratings 5–12 kW for at most 5% clipped potential AC energy.', '''
import numpy as np
# Step 1: Calculate the common available AC profile.
available = np.array([0,2,5,9,12,8,3,0])*0.97
# Step 2: Evaluate each candidate using the same clipping denominator.
trials = [(rating, np.maximum(available-rating,0).sum()/available.sum()) for rating in range(5,13)]
# Step 3: Select the smallest rating that satisfies the target.
best = next(t for t in trials if t[1]<=0.05)
print('Rating kW and clipping fraction:',best)
assert best[0]==10 and all(f>0.05 for r,f in trials if r<best[0])
''','An energy constraint is not a financial optimum; inverter prices and lifetime behavior are omitted.')
add('solar','Temperature and irradiance sensitivity grid','For G=[200,600,1000] W/m² and ambient=[10,25,40]°C, calculate a 3×3 grid of AC power for 10 kW DC, 8 kW AC, NOCT45, gamma−0.004 and eta0.97.', '''
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
''','At fixed irradiance, higher modeled cell temperature cannot increase power here; clipping can flatten sensitivity.')
add('solar','PV and battery self-consumption','PV hourly AC [0,4,6,0] kW meets constant 2 kW load with an initially empty 3 kWh battery, 2 kW limits and 90% efficiencies. Report unmet load, curtailed PV and final SOC.', '''
import numpy as np
# Step 1: Initialise storage and bookkeeping in kWh.
soc=0.; unmet_total=0.; curtailed_total=0.; rows=[]
# Step 2: Route each one-hour interval through direct use, charging and discharge.
for pv in [0,4,6,0]:
    before=soc; direct=min(pv,2)
    charge=min(pv-direct,2,(3-soc)/0.9); discharge=min(2-direct,2,soc*0.9)
    soc+=charge*0.9-discharge/0.9
    unmet=2-direct-discharge; curtailed=pv-direct-charge
    assert np.isclose(before+charge*0.9,soc+discharge/0.9)
    unmet_total+=unmet; curtailed_total+=curtailed; rows.append(soc)
# Step 3: Verify storage limits and show final energy accounting.
print(unmet_total,curtailed_total,soc,'kWh unmet, curtailed, final SOC')
assert np.isclose(unmet_total,2) and all(0<=s<=3+1e-12 for s in rows)
''','Stored energy remaining at the end has not yet served the load. Battery economic sizing needs longer data.')
add('solar','PV LCOE with degradation','Use €1m capital, €20k annual O&M, 1500 MWh first-year AC, 0.5% annual degradation and 25 years. Calculate LCOE at 0%,3%,6% real discount rates.', '''
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
''','See IFC Chapter 14 and IRENA cost methodology. Costs are synthetic; taxes, replacements and salvage are omitted.')
add('solar','Detect underperformance without future leakage','Expected daily energy is [10,20,30,40,50,60] kWh; observed is [9,18,27,36,30,36]. Calibrate an observed/expected ratio on days1–4, then flag holdout days below 80% of calibrated expectation.', '''
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
''','Flags identify a discrepancy, not its cause. Weather-model errors, outages or sensor faults need separate diagnosis.')

# Additional technology banks are defined below.
add('hydro','Net head','Gross head is 40 m and head loss 3 m. Calculate net head.', '''
# Step 1: Enter gross head and hydraulic losses.
gross, loss = 40, 3
# Step 2: Subtract losses before calculating useful head.
net = gross-loss
# Step 3: Check the head is positive and report it.
print(net,'m'); assert net == 37
''','Net head is 37 m; losses reduce rather than increase usable head.')
add('hydro','Electrical power','Use Q=10 m³/s, net head30 m, efficiency0.9, density1000 kg/m³ and g9.81 m/s². Calculate MW.', '''
# Step 1: Store parameters in SI units.
q, head, eta, rho, g = 10,30,0.9,1000,9.81
# Step 2: Convert hydraulic power to electricity and MW.
power = rho*g*q*head*eta/1e6
# Step 3: Check a hand-calculated result.
print(power,'MW'); assert abs(power-2.6487)<1e-12
''','2.6487 MW omits generator clipping.')
add('hydro','Daily hydro energy','A hydro generator averages 2 MW for 24 hours. Calculate MWh and kWh.', '''
# Step 1: Enter average power and duration.
power, hours = 2,24
# Step 2: Integrate and convert energy units.
mwh=power*hours; kwh=mwh*1000
# Step 3: Verify and print both units.
print(mwh,'MWh;',kwh,'kWh'); assert mwh==48 and kwh==48000
''','The power must be an interval average for this calculation.')
add('hydro','Environmental reservation','River flow is 8 m³/s and the environmental reservation is 3 m³/s. Calculate available turbine flow.', '''
# Step 1: Enter river flow and reservation.
river, reserve = 8,3
# Step 2: Reserve water before assigning turbine flow.
usable=max(river-reserve,0)
# Step 3: Check non-negativity and report the result.
print(usable,'m³/s'); assert usable==5
''','This simplified reservation is not a site-specific environmental-flow prescription.')
add('hydro','Water volume','A turbine passes 3 m³/s for 2 hours. Calculate water volume.', '''
# Step 1: Convert hours to seconds.
seconds=2*3600
# Step 2: Integrate the constant flow rate.
volume=3*seconds
# Step 3: Check and label the volume.
print(volume,'m³'); assert volume==21600
''','Flow is volume per second; multiplying by hours without conversion is incorrect.')
add('hydro','Flow to power table','For river flows [2,5,10,20] m³/s, reserve3, use head25 m, eta0.9, and a 2 MW rating. Tabulate usable flow and electrical MW.', '''
import numpy as np
import pandas as pd
# Step 1: Calculate usable flow for each river condition.
river=np.array([2,5,10,20]); usable=np.maximum(river-3,0)
# Step 2: Apply the hydraulic model and generator cap.
power=np.minimum(1000*9.81*usable*25*0.9/1e6,2)
# Step 3: Check dry and clipped cases and print a table.
print(pd.DataFrame({'river_m3s':river,'usable_m3s':usable,'power_mw':power}))
assert power[0]==0 and power[-1]==2
''','Capping power does not mean all available water must pass through the turbine.')
add('hydro','Generator flow limit','Find maximum turbine flow for a 2 MW generator, net head25 m and eta0.9. Compare against available flow12 m³/s.', '''
# Step 1: Rearrange P=rho*g*Q*H*eta for Q.
q_limit=2e6/(1000*9.81*25*0.9)
# Step 2: Limit turbine flow to equipment capacity.
q_turbine=min(12,q_limit); bypass=12-q_turbine
# Step 3: Recalculate power as a check.
power=1000*9.81*q_turbine*25*0.9/1e6
print(q_limit,'m³/s limit;',bypass,'m³/s bypass'); assert abs(power-2)<1e-12
''','Bypass is water not used for generation; environmental flow has already been reserved.')
add('hydro','Quadratic head loss','At Q=[2,4,6] m³/s use gross head30 m and loss=0.1Q² m. Calculate net head and MW at eta0.9.', '''
import numpy as np
# Step 1: Model the assumed flow-dependent loss.
q=np.array([2,4,6]); loss=0.1*q**2
# Step 2: Calculate head before calculating power.
head=30-loss; power=1000*9.81*q*head*0.9/1e6
# Step 3: Check head remains physically valid.
print(head,'m;',power,'MW'); assert np.all(head>0) and np.all(head<30)
''','The loss coefficient is synthetic and needs hydraulic calibration for a real conduit.')
add('hydro','Monthly durations','Mean output is 2 MW in January and February2025. Use pandas calendar month lengths to calculate MWh for each month.', '''
import pandas as pd
# Step 1: Create the two calendar months.
months=pd.date_range('2025-01-01',periods=2,freq='MS')
# Step 2: Multiply average output by actual month hours.
energy=2*24*months.days_in_month.to_numpy()
# Step 3: Verify the January and non-leap February results.
print(energy,'MWh'); assert list(energy)==[1488,1344]
''','Constant monthly mean output does not imply equal monthly energy.')
add('hydro','Flow-duration ranking','Sort daily flows [8,2,10,4] descending and assign exceedance plotting positions rank/(n+1).', '''
import numpy as np
# Step 1: Rank observed daily mean flows from largest to smallest.
flow=np.sort([8,2,10,4])[::-1]
# Step 2: Use the stated plotting-position convention.
exceedance=np.arange(1,len(flow)+1)/(len(flow)+1)
# Step 3: Print paired values and check endpoints.
print(list(zip(exceedance,flow)))
assert np.all(np.diff(flow)<=0) and np.allclose(exceedance,[.2,.4,.6,.8])
''','A duration curve loses chronology and cannot directly route reservoir storage.')
add('hydro','Efficiency sensitivity','At Q8 m³/s and net head40 m, compare efficiencies0.75,0.85,0.95. Print MW and the gain from lowest to highest.', '''
import numpy as np
# Step 1: Define conversion-efficiency scenarios.
eta=np.array([0.75,0.85,0.95])
# Step 2: Keep resource inputs fixed across scenarios.
power=1000*9.81*8*40*eta/1e6
# Step 3: Verify linear sensitivity and report gain.
print(power,'MW; gain:',power[-1]-power[0],'MW')
assert np.allclose(power/power[0],eta/eta[0])
''','Real turbine efficiency varies with operating point; these are comparison assumptions.')
add('hydro','Pumped-storage round trip','Store100000 m³ through100 m head. With pump efficiency0.85 and generation efficiency0.9, find gravitational, input and recovered MWh.', '''
import numpy as np
# Step 1: Convert gravitational joules to MWh.
stored=1000*9.81*100000*100/3.6e9
# Step 2: Account for pumping and generating conversion losses.
input_energy=stored/0.85; output=stored*0.9
# Step 3: Check round-trip efficiency.
print(stored,input_energy,output,'MWh stored, input, output')
assert np.isclose(output/input_energy,0.85*0.9)
''','Storage shifts electricity and consumes net energy; it is not a new primary-energy source.')
add('hydro','Validate resource readings','For river flows [3,−1,missing,5] m³/s, flag invalid readings and report valid fraction without filling gaps.', '''
import pandas as pd
# Step 1: Preserve the measured values and missing marker.
flow=pd.Series([3,-1,None,5],dtype=float)
# Step 2: Replace only physically invalid negatives with missing.
clean=flow.mask(flow.lt(0))
# Step 3: Report coverage rather than a complete-period generation claim.
print(clean,'coverage:',clean.notna().mean()); assert clean.notna().sum()==2
''','Do not infer zero flow from an absent measurement.')
add('hydro','Environmental shortage','Daily flows[1,4,7] m³/s must first provide up to3 m³/s environmental release. Calculate release, shortage and remaining turbine flow.', '''
import numpy as np
# Step 1: Store daily mean flows and the reservation target.
river=np.array([1,4,7]); target=3
# Step 2: Allocate available water before calculating shortages.
release=np.minimum(river,target); shortage=target-release; turbine=river-release
# Step 3: Reconcile water and print all three quantities.
assert np.array_equal(release+turbine,river)
print('Release, shortage, turbine m³/s:',release,shortage,turbine)
assert np.array_equal(shortage,[2,0,0])
''','A shortage means the river cannot meet the target; no negative turbine flow is allowed.')
add('hydro','Plot a power-duration curve','Daily flows[2,4,8,12,16] m³/s drive head20 m, eta0.9 and cap2 MW with no reservation in this example. Plot ranked MW and calculate five-day MWh.', '''
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
''','Sorting preserves total energy for equal intervals but removes the order needed for storage simulation.')
add('hydro','Reservoir routing','Route hourly inflows[2,0,8,16,0,0] m³/s; initial storage20000 m³ and maximum50000. Prioritise1 m³/s environment, then up to3 turbine; spill excess. Use head20 m and eta0.9. Check every water balance.', '''
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
''','Within-hour inflow is immediately available; head is constant and evaporation is omitted.')
add('hydro','Storage-dependent head','Route inflows[2,0,3] m³/s for three hours from20000 m³. Withdraw up to2 m³/s, no spill limit. Use beginning-of-step head H=10+S/10000 m and eta0.9; compare with constant12 m.', '''
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
''','Beginning-of-step head is a numerical approximation; smaller time steps may be needed for rapid level changes.')
add('hydro','Environmental-flow trade-off','For daily river flows[2,5,8,12] m³/s, compare reservations1,3,5 m³/s, head30 m, eta0.9, cap2 MW. Calculate four-day energy and environmental-shortage volume.', '''
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
''','The model compares water allocation, not ecological adequacy or regulatory compliance.')
add('hydro','Search a generator rating','Flows[2,4,8,12] m³/s last24 h each, head30 m and eta0.9. Choose the smallest candidate rating[1,2,3,4] MW that captures95% of uncapped energy.', '''
import numpy as np
# Step 1: Calculate the uncapped electrical resource.
available=1000*9.81*np.array([2,4,8,12])*30*0.9/1e6
# Step 2: Evaluate retained energy for each candidate rating.
results=[(r,np.minimum(available,r).sum()/available.sum()) for r in [1,2,3,4]]
# Step 3: Select and verify the first feasible rating.
best=next(row for row in results if row[1]>=0.95)
print('MW rating, retained energy fraction:',best); assert best[0]==3
''','This is an energy-capture criterion; construction cost and seasonal hydrology are not represented.')
add('hydro','Pumped-storage dispatch and water balance','An upper reservoir starts empty with capacity10000 m³ and head100 m. Pump with1 MW for two hours at85%, then generate for two hours at up to0.7 MW and90%. Track water and electricity.', '''
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
''','The final reservoir need not be empty; recovered/pumped energy over a partial cycle is not full round-trip efficiency.')

add('wind','Rotor swept area','Calculate swept area for a20 m radius rotor using pi r².', '''
import math
# Step 1: Enter the rotor radius in metres.
radius=20
# Step 2: Calculate the swept disk area.
area=math.pi*radius**2
# Step 3: Check against diameter-based area.
print(area,'m²'); assert math.isclose(area,math.pi*(2*radius)**2/4)
''','Use rotor swept area, not blade material area.')
add('wind','Available wind power','Use density1.225 kg/m³, area1000 m² and speed8 m/s. Calculate available wind kW.', '''
# Step 1: Enter resource parameters in SI units.
rho,area,speed=1.225,1000,8
# Step 2: Calculate kinetic-energy flux through the rotor area.
available_kw=0.5*rho*area*speed**3/1000
# Step 3: Check the numerical result.
print(available_kw,'kW'); assert abs(available_kw-313.6)<1e-10
''','313.6 kW is available in the wind, not electrical output.')
add('wind','Cubic speed sensitivity','Compare available power at5 and10 m/s for the same rotor and density. Calculate their ratio.', '''
# Step 1: Store the two speeds.
low,high=5,10
# Step 2: Cancel common factors and compare cubed speeds.
ratio=high**3/low**3
# Step 3: Verify the effect of doubling speed.
print(ratio,'times'); assert ratio==8
''','A real turbine does not keep increasing output cubically above rated speed.')
add('wind','Aerodynamic extraction','Available wind power is500 kW, Cp0.4 and drivetrain efficiency0.95. Find electrical kW and compare Cp with16/27.', '''
# Step 1: Specify power and fractional conversion factors.
available,cp,efficiency=500,0.4,0.95
# Step 2: Apply aerodynamic extraction and drivetrain losses.
electrical=available*cp*efficiency
# Step 3: Check the ideal Betz bound and the result.
print(electrical,'kW'); assert cp<16/27 and electrical==190
''','Betz is an ideal aerodynamic limit; it is not a turbine capacity factor.')
add('wind','Daily turbine capacity factor','A3 MW turbine delivers21.6 MWh in24 h. Calculate daily CF.', '''
# Step 1: Calculate the rated energy for the same interval.
maximum=3*24
# Step 2: Divide measured energy by rated energy.
cf=21.6/maximum
# Step 3: Print the percentage and check bounds.
print(f'{cf:.0%}'); assert abs(cf-0.3)<1e-12
''','30% is a period-specific utilization measure.')
add('wind','Hub-height adjustment','Wind is6 m/s at10 m. Estimate at heights[10,50,100] m using v=vref(z/zref)^0.14.', '''
import numpy as np
# Step 1: Store measurement height, speed and target heights.
height=np.array([10,50,100]); reference=10; speed=6
# Step 2: Apply the assumed power-law shear profile.
hub=speed*(height/reference)**0.14
# Step 3: Verify the reference height is unchanged.
print(hub,'m/s'); assert hub[0]==6 and np.all(np.diff(hub)>0)
''','The exponent is illustrative and not universal across terrain and atmospheric stability.')
add('wind','Density sensitivity','At fixed area1000 m² and8 m/s, compare available kW for densities[1.0,1.225,1.3] kg/m³.', '''
import numpy as np
# Step 1: Store candidate air densities.
rho=np.array([1.0,1.225,1.3])
# Step 2: Keep geometry and speed fixed.
power=0.5*rho*1000*8**3/1000
# Step 3: Check proportionality with density.
print(power,'kW'); assert np.allclose(power/power[0],rho/rho[0])
''','This checks available wind power; manufacturer electrical curves require their specified density treatment.')
add('wind','Piecewise turbine curve','Implement3 MW rated power: zero below3 m/s, 3(v³−3³)/(12³−3³) between3 and12, rated to below25, zero thereafter. Test[0,3,6,12,25].', '''
import numpy as np
# Step 1: Define a vectorised curve with explicit operating boundaries.
def curve(values):
    v=np.asarray(values,dtype=float)
    if np.any(~np.isfinite(v)) or np.any(v<0): raise ValueError('Invalid wind speed')
    return np.where((v>=3)&(v<12),3*(v**3-3**3)/(12**3-3**3),np.where((v>=12)&(v<25),3.,0.))
# Step 2: Evaluate boundary and intermediate speeds.
power=curve([0,3,6,12,25])
# Step 3: Check key thresholds and print MW.
print(power,'MW'); assert np.allclose(power[[0,1,3,4]],[0,0,3,0])
''','Cut-out is a shutdown boundary; the idealised curve omits control hysteresis.')
add('wind','Power of mean versus mean power','Compare the mean of v³ with the cube of mean speed for speeds[4,8] m/s.', '''
import numpy as np
# Step 1: Store equal-duration wind observations.
v=np.array([4,8])
# Step 2: Compare averaging before and after the nonlinear transformation.
mean_cube=np.mean(v**3); cube_mean=np.mean(v)**3
# Step 3: Print the discrepancy and check its direction.
print(mean_cube,cube_mean,'m³/s³'); assert mean_cube==288 and cube_mean==216
''','Using mean speed in a nonlinear power model can bias energy estimates; clipping adds further effects.')
add('wind','Turbulence intensity','For[6,8,10] m/s, calculate population standard deviation divided by mean speed. State ddof=0.', '''
import numpy as np
# Step 1: Define the observations and statistical convention.
v=np.array([6,8,10]); mean=v.mean()
# Step 2: Divide population standard deviation by nonzero mean.
ti=v.std(ddof=0)/mean
# Step 3: Check and print the dimensionless ratio.
print(f'TI: {ti:.2%}'); assert np.isclose(ti,np.sqrt(8/3)/8)
''','Three samples demonstrate the calculation; they do not meet a field measurement campaign specification.')
add('wind','Circular mean direction','Calculate the equal-weight circular mean of[350,10,0] degrees clockwise from north.', '''
import numpy as np
# Step 1: Convert angles to radians before trigonometry.
angle=np.deg2rad([350,10,0])
# Step 2: Average sine and cosine and recover a compass angle.
s=np.sin(angle).mean(); c=np.cos(angle).mean()
mean=np.degrees(np.arctan2(s,c))%360
# Step 3: Verify north modulo a full rotation.
print(mean,'degrees'); assert min(abs(mean),abs(mean-360))<1e-10
''','An arithmetic mean would be misleading across the north/360° boundary.')
add('wind','Compass sector counts','Bin directions[350,0,20,45,90,180,270] into eight sectors centered on N,NE,E,SE,S,SW,W,NW.', '''
import numpy as np
# Step 1: Shift by half a sector so north straddles zero degrees.
direction=np.array([350,0,20,45,90,180,270]); sector=((direction+22.5)%360//45).astype(int)
# Step 2: Count each sector including empty ones.
counts=np.bincount(sector,minlength=8)
# Step 3: Check all records are counted and north has three.
print(dict(zip(['N','NE','E','SE','S','SW','W','NW'],counts)))
assert counts.sum()==7 and counts[0]==3
''','These are frequency counts, not energy-weighted wind-rose sectors.')
add('wind','Combined availability and losses','Gross annual energy is10 GWh. Apply availability0.97 and retained wake energy0.92 sequentially.', '''
# Step 1: Store independent retained-energy assumptions.
gross=10; availability=0.97; wake_retained=0.92
# Step 2: Apply each factor once.
net=gross*availability*wake_retained
# Step 3: Check net energy and show the total loss fraction.
print(net,'GWh; loss fraction:',1-net/gross); assert abs(net-8.924)<1e-12
''','Avoid applying a loss twice when an input power curve or energy estimate is already net of it.')
add('wind','Unequal operating intervals','Mean powers[0.5,2,3] MW last[2,1,0.5] hours. Calculate energy and CF for a3 MW turbine.', '''
import numpy as np
# Step 1: Store aligned power and duration values.
p=np.array([0.5,2,3]); dt=np.array([2,1,0.5])
# Step 2: Integrate and use the same duration in the capacity-factor denominator.
energy=np.sum(p*dt); cf=energy/(3*dt.sum())
# Step 3: Check the hand sum and report results.
print(energy,'MWh;',cf,'CF'); assert energy==4.5 and 0<=cf<=1
''','The energy-weighted time calculation is essential when intervals differ.')
add('wind','Histogram of wind speeds','Plot a histogram of[0,2,4,6,8,10,12,14] m/s with bins[0,5,10,15]. Check counts.', '''
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Specify speeds and bin edges.
speed=np.array([0,2,4,6,8,10,12,14]); edges=[0,5,10,15]
# Step 2: Count observations and plot frequency.
counts,_=np.histogram(speed,bins=edges)
fig,ax=plt.subplots(figsize=(6,3)); ax.hist(speed,bins=edges,edgecolor='black')
ax.set(xlabel='Wind speed (m/s)',ylabel='Observations',title='Synthetic wind sample'); fig.tight_layout(); plt.show()
# Step 3: Check bin counts and completeness.
print(counts); assert np.array_equal(counts,[3,2,3])
''','A frequency histogram uses counts; probability density would have a different vertical-axis unit.')
add('wind','Weibull annual energy and convergence','Use Weibull shape2, scale8 m/s and the3/12/25 m/s,3 MW curve. Integrate using midpoint powers and CDF bin probabilities over0–30 m/s. Compare bin widths0.1 and0.05; apply8760 h and net factor0.9.', '''
import numpy as np
# Step 1: Define the curve and exact bin probabilities from the Weibull CDF.
def aep(step):
    edges=np.arange(0,30+step/2,step); mid=(edges[:-1]+edges[1:])/2
    probability=np.diff(1-np.exp(-(edges/8)**2))
    p=np.where((mid>=3)&(mid<12),3*(mid**3-27)/(1728-27),np.where((mid>=12)&(mid<25),3,0))
    return np.sum(p*probability)*8760*0.9/1000
# Step 2: Refine the integration grid.
coarse,fine=aep(0.1),aep(0.05)
# Step 3: Check convergence and capacity limits.
print(coarse,fine,'GWh; relative difference:',abs(coarse-fine)/fine)
assert abs(coarse-fine)/fine<0.001 and 0<fine<3*8760/1000
''','Probability above30 m/s has zero output under this shutdown curve; its omission does not remove generating energy.')
add('wind','Undefined circular means','Write a direction-mean function rejecting empty/nonfinite data and resultant length below1e−8. Verify[90,270] is undefined and[350,10] is north.', '''
import numpy as np
# Step 1: Validate input angles and compute the mean vector.
def mean_direction(values):
    a=np.asarray(values,dtype=float)
    if a.size==0 or np.any(~np.isfinite(a)): raise ValueError('Need finite directions')
    s=np.sin(np.deg2rad(a)).mean(); c=np.cos(np.deg2rad(a)).mean()
    if np.hypot(s,c)<1e-8: raise ValueError('Undefined mean direction')
    return np.degrees(np.arctan2(s,c))%360
# Step 2: Verify cancellation is explicitly reported.
try: mean_direction([90,270])
except ValueError: print('Opposite directions: undefined mean')
else: raise AssertionError('Cancellation was not detected')
# Step 3: Verify the well-defined northward sample.
answer=mean_direction([350,10]); assert min(answer,360-answer)<1e-8
print(answer,'degrees')
''','A near-zero resultant is not evidence for a meaningful northward average.')
add('wind','Fit shear from multiple heights','Measured speeds at heights[20,40,80] m are generated by6(z/20)^0.2. Fit log(v) against log(z/20) and predict at100 m.', '''
import numpy as np
# Step 1: Build synthetic multi-height observations.
height=np.array([20,40,80]); speed=6*(height/20)**0.2
# Step 2: Estimate exponent and reference speed by log-linear regression.
alpha,intercept=np.polyfit(np.log(height/20),np.log(speed),1)
prediction=np.exp(intercept)*(100/20)**alpha
# Step 3: Recover the known synthetic parameters.
print('Alpha:',alpha,'100 m wind:',prediction,'m/s')
assert np.isclose(alpha,0.2) and np.isclose(np.exp(intercept),6)
''','Three heights provide a fit and consistency check; real profiles require concurrent quality-controlled measurements.')
add('wind','Compare hub heights on energy','At10 m, speeds[3,5,7,9] m/s last6 h each. Compare hub heights50 and100 m using exponent0.14 and the3 MW piecewise curve. Calculate daily MWh, not power from mean speed.', '''
import numpy as np
# Step 1: Define the common measured speeds and a turbine model.
vref=np.array([3,5,7,9])
def curve(v):
    return np.where((v>=3)&(v<12),3*(v**3-27)/(1728-27),np.where((v>=12)&(v<25),3,0))
# Step 2: Adjust each interval to each candidate hub height.
energy={height:float(curve(vref*(height/10)**0.14).sum()*6) for height in [50,100]}
# Step 3: Check rated bounds and compare the designs.
print(energy,'MWh'); assert all(0<=e<=72 for e in energy.values()) and energy[100]>=energy[50]
''','Higher towers change cost and loads. More energy in this sample does not prove economic superiority.')
add('wind','Bootstrap a sample mean power','Observed daily-average powers[0.4,0.8,1.2,0.6,1.5,1.0] MW are treated as independent. With seed11, bootstrap2000 equal-sized samples and calculate a percentile95% interval for mean MW.', '''
import numpy as np
# Step 1: Set observations and a reproducible random generator.
p=np.array([0.4,0.8,1.2,0.6,1.5,1.0]); rng=np.random.default_rng(11)
# Step 2: Resample observations with replacement and average each sample.
means=rng.choice(p,size=(2000,len(p)),replace=True).mean(axis=1)
interval=np.quantile(means,[0.025,0.975])
# Step 3: Check interval order and report the sampling assumptions.
print('Sample mean:',p.mean(),'MW; bootstrap interval:',interval,'MW')
assert interval[0]<interval[1] and means.min()>=p.min() and means.max()<=p.max()
''','Six days are insufficient for a bankable resource estimate. Serial dependence would require a suitable block bootstrap or another model.')

add('geothermal','Temperature at depth','Surface temperature15°C and gradient30°C/km: calculate temperature at2 km with a linear model.', '''
# Step 1: Store depth and gradient with matching kilometre units.
surface,gradient,depth=15,30,2
# Step 2: Add the gradient-induced temperature increase.
temperature=surface+gradient*depth
# Step 3: Check and label the estimate.
print(temperature,'°C'); assert temperature==75
''','A linear conductive gradient is not a prediction of a convective reservoir.')
add('geothermal','Heat flow in produced water','Water flow50 kg/s cools from150 to70°C. With cp4180 J/(kg K), calculate thermal MW.', '''
# Step 1: Calculate the temperature difference.
delta=150-70
# Step 2: Apply sensible heat flow and convert W to MW.
thermal=50*4180*delta/1e6
# Step 3: Check the hand calculation.
print(thermal,'MW thermal'); assert abs(thermal-16.72)<1e-12
''','The model assumes single-phase liquid and constant heat capacity.')
add('geothermal','Gross and net electricity','Thermal power20 MW has gross electrical efficiency12% and parasitic load10% of gross electricity. Calculate gross and net MW.', '''
# Step 1: Convert thermal input into gross electricity.
gross=20*0.12
# Step 2: Subtract the parasitic fraction of gross electricity.
net=gross*(1-0.10)
# Step 3: Check the result and distinguish the two powers.
print(gross,net,'MW gross/net'); assert abs(net-2.16)<1e-12
''','Parasitic fraction here applies to gross electrical power, not thermal input.')
add('geothermal','Availability-adjusted energy','A plant produces3 MW net when available, with95% availability in a365-day year. Calculate GWh.', '''
# Step 1: Calculate expected operating hours.
hours=365*24*0.95
# Step 2: Integrate net power and convert to GWh.
energy=3*hours/1000
# Step 3: Check the numerical result.
print(energy,'GWh'); assert abs(energy-24.966)<1e-12
''','This assumes full net output whenever available.')
add('geothermal','Kelvin and temperature differences','Convert150 and70°C to kelvin, then show that the temperature difference is unchanged.', '''
# Step 1: Store Celsius temperatures.
hot,cold=150,70
# Step 2: Add273.15 to both absolute temperatures.
hot_k,cold_k=hot+273.15,cold+273.15
# Step 3: Check that the difference is80 K.
print(hot_k,cold_k,'K; difference:',hot_k-cold_k,'K')
assert abs((hot_k-cold_k)-(hot-cold))<1e-10
''','Use absolute kelvin temperatures in thermodynamic ratios; Celsius differences are valid for sensible heat.')
add('geothermal','Depth-temperature table','At depths[1,2,3] km, compare gradients25 and35°C/km with surface15°C. Build a pandas table.', '''
import numpy as np
import pandas as pd
# Step 1: Store depths in a common unit.
depth=np.array([1,2,3])
# Step 2: Calculate each gradient scenario separately.
table=pd.DataFrame({f'{g} C_per_km':15+g*depth for g in [25,35]},index=depth)
# Step 3: Print labelled values and check one scenario.
print(table.rename_axis('Depth km')); assert table.iloc[-1,1]==120
''','Gradient uncertainty changes inferred temperature substantially at depth.')
add('geothermal','Compare well flows','For flows[30,50,70] kg/s, temperatures150/70°C, cp4180, efficiency0.12 and parasitic fraction0.1, calculate net MW.', '''
import numpy as np
# Step 1: Store well-flow cases.
flow=np.array([30,50,70])
# Step 2: Apply the same thermal and electrical assumptions.
net=flow*4180*(150-70)*0.12*0.9/1e6
# Step 3: Check proportionality to flow and display power.
print(net,'MW net'); assert np.allclose(net/net[0],flow/flow[0])
''','Pump demand is represented only by a fixed fractional loss in this model.')
add('geothermal','Reinjection temperature sensitivity','At50 kg/s and production160°C, compare reinjection[60,80,100]°C with cp4180, efficiency0.12 and parasitic0.1.', '''
import numpy as np
# Step 1: Store reinjection-temperature alternatives.
reinjection=np.array([60,80,100])
# Step 2: Calculate thermal drawdown and net electrical power.
net=50*4180*(160-reinjection)*0.12*0.9/1e6
# Step 3: Verify the direction of the simplified sensitivity.
print(net,'MW net'); assert np.all(np.diff(net)<0)
''','Lower reinjection temperatures may increase modeled heat recovery but create chemical and reservoir constraints.')
add('geothermal','Multiple well-field totals','Two wells have flows[40,60] kg/s and production[150,170]°C. Reinjection is70°C, cp4180, conversion0.12 and parasitic0.1. Calculate individual and total net MW.', '''
import numpy as np
# Step 1: Align each well flow with its production temperature.
flow=np.array([40,60]); temperature=np.array([150,170])
# Step 2: Calculate and sum net well contributions.
net=flow*4180*(temperature-70)*0.12*0.9/1e6
# Step 3: Compare the total against direct thermal accounting.
print(net,'MW by well;',net.sum(),'MW total')
assert np.isclose(net.sum(),4180*(40*80+60*100)*0.12*0.9/1e6)
''','Adding well powers assumes the shared plant can accept all of the available resource.')
add('geothermal','Parasitic-load scenarios','Gross output is5 MW. Compare parasitic fractions[0.05,0.10,0.20] and calculate net MW and24-hour MWh.', '''
import numpy as np
# Step 1: Store alternative operating-loss assumptions.
fraction=np.array([0.05,0.10,0.20])
# Step 2: Deduct each parasitic fraction from gross output.
net=5*(1-fraction); energy=net*24
# Step 3: Check ordering and label both quantities.
print(net,'MW;',energy,'MWh'); assert np.all(np.diff(net)<0)
''','Parasitic demand includes pumps and auxiliaries; a constant fraction simplifies their operating behavior.')
add('geothermal','Thermodynamic upper bound','Compute Carnot efficiency for hot150°C and cold25°C using kelvin. Compare a12% conversion assumption.', '''
# Step 1: Convert both reservoir temperatures to kelvin.
hot=150+273.15; cold=25+273.15
# Step 2: Calculate the ideal reversible-engine efficiency.
carnot=1-cold/hot
# Step 3: Verify the illustrative actual conversion is below the bound.
print(carnot,'Carnot fraction; assumed actual:',0.12); assert 0<0.12<carnot<1
''','Carnot is an upper bound between fixed-temperature reservoirs, not a practical geothermal cycle model.')
add('geothermal','Conductive heat flux','Use thermal conductivity2.5 W/(m K) and gradient30 K/km. Calculate upward conductive heat-flux magnitude W/m².', '''
# Step 1: Convert the temperature gradient to K per metre.
gradient=30/1000
# Step 2: Apply the magnitude form of Fourier conduction.
flux=2.5*gradient
# Step 3: Check units and numerical value.
print(flux,'W/m²'); assert abs(flux-0.075)<1e-12
''','The magnitude is75 mW/m². Heat flux requires conductivity; a temperature gradient alone is not flux.')
add('geothermal','Compounded net-power decline','A5 MW net plant declines2% per year. Calculate years1–10 and annual GWh at95% availability.', '''
import numpy as np
# Step 1: Start the decline exponent at zero for year1.
years=np.arange(1,11)
# Step 2: Calculate annual net power and availability-adjusted energy.
power=5*0.98**(years-1); energy=power*8760*0.95/1000
# Step 3: Check first-year energy and decreasing power.
print(energy,'GWh/year'); assert np.isclose(energy[0],41.61) and np.all(np.diff(power)<0)
''','A prescribed decline curve is a scenario, not a calibrated reservoir forecast.')
add('geothermal','Invalid well records','Flows[40,−2,50] kg/s and production[150,160,60]°C have reinjection70°C. Retain only finite positive-flow records hotter than reinjection; report exclusions.', '''
import numpy as np
# Step 1: Preserve aligned well observations.
flow=np.array([40,-2,50]); production=np.array([150,160,60])
# Step 2: Construct a physical-validity mask before calculating heat.
valid=np.isfinite(flow)&np.isfinite(production)&(flow>0)&(production>70)
thermal=flow[valid]*4180*(production[valid]-70)/1e6
# Step 3: Report excluded records and verify the retained count.
print('Excluded indices:',np.where(~valid)[0],'; thermal MW:',thermal)
assert valid.sum()==1 and np.isclose(thermal[0],13.376)
''','Excluded values need investigation; silently converting negative heat to positive power would hide errors.')
add('geothermal','Plot decline scenarios','Compare initial5 MW with annual decline0%,1%,3% over20 years and95% availability. Plot annual GWh and print cumulative values.', '''
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
''','Long-run scenarios are sensitive to decline assumptions and intervention choices.')
add('geothermal','Fit decline without future leakage','Create ten annual powers5×0.98^(year−1) MW. Fit log(power) on years1–7 only, predict8–10, and compare MAE against the last training observation.', '''
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
''','Exact recovery occurs because the noiseless observations follow the fitted model by construction.')
add('geothermal','Well field with plant clipping','Three wells have flows[40,50,60] kg/s and production[150,160,170]°C. Use reinjection70°C, cp4180, conversion0.12, parasitic0.1 and a4 MW net plant cap. Compare all operating versus outage of each well.', '''
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
''','Clipping can hide some well loss. This allocation ignores interactions between wells and shared equipment.')
add('geothermal','Net output with pumping penalty','Compare flows20–100 kg/s in steps10. Gross MW=m×4180×80×0.12/1e6 and pump MW=0.00004m². Find the discrete flow maximising net power.', '''
import numpy as np
# Step 1: Define candidate flow rates and a synthetic pump-load model.
flow=np.arange(20,101,10); pump=0.00004*flow**2
# Step 2: Calculate gross and net power consistently in MW.
gross=flow*4180*80*0.12/1e6; net=gross-pump
# Step 3: Select the best candidate and verify the energy subtraction.
best=np.argmax(net)
print('Best flow:',flow[best],'kg/s; net:',net[best],'MW')
assert np.allclose(net+pump,gross) and flow[best]==100
''','The best candidate is the upper search boundary, not an established physical optimum; do not extrapolate the pump model.')
add('geothermal','Lifetime energy with an intervention','Starting at5 MW, power declines3% per year through year10. At the start of year11 restore power to5 MW and then decline3% again through year20. Compare cumulative GWh with uninterrupted decline;95% availability.', '''
import numpy as np
# Step 1: Calculate the uninterrupted reference trajectory.
year=np.arange(1,21); baseline=5*0.97**(year-1)
# Step 2: Reset the age exponent at the intervention year.
age=np.where(year<=10,year-1,year-11)
intervention=5*0.97**age
# Step 3: Integrate the incremental generation and verify the reset.
gain=np.sum(intervention-baseline)*8760*0.95/1000
print(gain,'GWh additional'); assert intervention[10]==5 and gain>0
''','Intervention cost, downtime and reservoir feasibility are omitted; additional energy is not automatically economic benefit.')
add('geothermal','Joint uncertainty in flow and temperature','With seed5 sample2000 independent flows uniform40–60 kg/s and production temperatures uniform140–170°C. Use reinjection70°C, cp4180, eta0.12, parasitic0.1. Report net-power median,5th/95th percentiles and probability above2 MW.', '''
import numpy as np
# Step 1: State the synthetic uncertainty distributions and independence assumption.
rng=np.random.default_rng(5); flow=rng.uniform(40,60,2000); temp=rng.uniform(140,170,2000)
# Step 2: Propagate paired samples through the net-power model.
net=flow*4180*(temp-70)*0.12*0.9/1e6
# Step 3: Summarise outcomes and check the physical range.
print('5th/median/95th MW:',np.quantile(net,[0.05,0.5,0.95]),'P(net>2):',np.mean(net>2))
assert np.all(net>0) and np.all(net<=60*4180*100*0.12*0.9/1e6)
''','The distributions and independence are assumptions, not measured reservoir uncertainty or a confidence interval.')

PATHS = {'general':'section0/introRenewableEnergy.md','solar':'section6/solarEnergy.md',
         'hydro':'section6/hydroelectricEnergy.md','wind':'section6/windEnergy.md',
         'geothermal':'section6/geothermalEnergy.md'}

def render(topic):
    exercises=BANK[topic]
    assert len(exercises)==20, (topic,len(exercises))
    text='## Chapter practice — 20 Python exercises\n\n**5 easy · 10 medium · 5 hard.** Work through the exercises in order. All inputs are synthetic teaching data. Each solution is directly below its question and starts collapsed in the book. Open it after trying your own code. Each solution runs independently; run the full notebook from the first cell when studying the chapter. References identify the underlying concepts rather than copied textbook problems.\n\n'
    for i,e in enumerate(exercises,1):
        e=e.copy()
        e['prompt']=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', e['prompt'])
        e['prompt']=re.sub(r'(?<=\d)(?=(?:MW|GW|kW|W/|m³|m²|m/s|kg|°C|km|h\b))',' ',e['prompt'])
        level='Easy' if i<=5 else 'Medium' if i<=15 else 'Hard'
        text+=f'({topic}-exercise-{i:02})=\n### Exercise {i:02} — {e["title"]}\n\n**Difficulty:** {level}\n\n**Reference:** {REFERENCES[topic]}\n\n**Task:** {e["prompt"]}\n\n```{{code-cell}} python\n# Your solution for {topic} exercise {i:02}.\n```\n\n::::{{dropdown}} Step-by-step answer — {topic.title()} {i:02}\n\n'
        text+='\n'.join(f'{j}. {s}' for j,s in enumerate(e['steps'],1))
        text+=f'\n\n```{{code-cell}} python\n{e["code"]}\n```\n\n**Interpretation:** {e["interpretation"]}\n\n::::\n\n'
    return text

def export_notebook(path):
    """Create a companion .ipynb from the complete MyST chapter, not just exercises."""
    text=path.read_text(encoding='utf-8')
    text=re.sub(r'\A---\n.*?\n---\n','',text,flags=re.S)
    cells=[]; last=0
    for match in re.finditer(r'```\{code-cell\} python\n(.*?)```',text,re.S):
        if text[last:match.start()].strip(): cells.append(nbformat.v4.new_markdown_cell(text[last:match.start()].strip()))
        cells.append(nbformat.v4.new_code_cell(match.group(1).strip()))
        last=match.end()
    if text[last:].strip():cells.append(nbformat.v4.new_markdown_cell(text[last:].strip()))
    # MyST dropdown markup is book-specific. Standard notebooks show clear answer headings.
    for cell in cells:
        if cell.cell_type=='markdown':
            cell.source=re.sub(r'^::::\{dropdown\} (.*)$',r'#### \1',cell.source,flags=re.M)
            cell.source=re.sub(r'^::::\s*$','',cell.source,flags=re.M)
            cell.source=re.sub(r'^\([a-z]+-exercise-\d+\)=\n','',cell.source,flags=re.M)
    # Preserve checked code outputs when only surrounding teaching text changes.
    destination=path.with_suffix('.ipynb')
    if destination.exists():
        previous=nbformat.read(destination,as_version=4)
        old_code={c.source:c for c in previous.cells if c.cell_type=='code'}
        for cell in cells:
            if cell.cell_type=='code' and cell.source in old_code:
                old=old_code[cell.source]
                cell.outputs=old.outputs;cell.execution_count=old.execution_count
    # Supply readable bibliographic details in standard notebook viewers too.
    cited=set()
    for group in re.findall(r'\[@([^\]]+)\]',text):
        cited.update(part.strip().lstrip('@') for part in group.split(';'))
    bibliography=(ROOT/'references.bib').read_text(encoding='utf-8')
    details=['## References used in this notebook','']
    for key,body in re.findall(r'@\w+\{([^,]+),\s*(.*?)\n\}',bibliography,re.S):
        if key not in cited:continue
        def field(name):
            match=re.search(r'\b'+name+r'\s*=\s*\{(.*?)\}\s*[,\n]',body+'\n',re.S|re.I)
            return match.group(1).replace('{','').replace('}','') if match else ''
        details.append(f'- **{key}:** {field("author")}. {field("title")} ({field("year")}). '+field('url'))
    cells.append(nbformat.v4.new_markdown_cell('\n'.join(details)))
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
    nbformat.write(nb,destination)

def main():
    for topic,relative in PATHS.items():
        path=ROOT/relative; text=path.read_text(encoding='utf-8')
        if '## Chapter practice — 20 Python exercises' in text:
            start=text.index('## Chapter practice — 20 Python exercises')
        else:
            start=text.index('## Exercises' if topic=='general' else '## Guided exercises')
        tail=text[text.index('## Common mistakes',start):] if topic!='general' else ''
        text=text[:start]+render(topic)+tail
        if topic=='general':
            text=text.replace('Start Python practice with the [graded workbook](../section7/renewableExercises.md). The short calculations below are preliminary unit checks.','The 20 exercises below progress from unit conversions to portfolio analysis. Study Python basics before attempting the medium and hard problems.')
        download=f'\n[Download this complete chapter as a Jupyter notebook]({path.stem}.ipynb). In standard Jupyter viewers, answer headings and code are visible below each prompt; the book provides collapse controls.\n\n'
        if download not in text:
            text=text.replace('## Chapter practice — 20 Python exercises',download+'## Chapter practice — 20 Python exercises',1)
        path.write_text(text,encoding='utf-8'); export_notebook(path)
        print(topic,len(BANK[topic]),'exercises')
    index='# Chapter exercise guide — 100 Python exercises\n\nThe exercises are embedded directly inside each teaching chapter, with a step-by-step answer below every question. Every chapter has **5 easy, 10 medium and 5 hard** exercises, each with references and executable checks.\n\n| Chapter | Exercises | Open the chapter |\n|---|---:|---|\n'
    for topic,relative in PATHS.items():
        index+=f'| {topic.title()} | 20 | [Study and practice](../{relative}#chapter-practice-20-python-exercises) |\n'
    index+='\nEach chapter also offers a complete downloadable `.ipynb` notebook. In the book, practice answers start collapsed; in ordinary notebook viewers the step-by-step answers appear below their prompts.\n\n## After practice\n\nTake the [25-question test](renewableEnergyTest.md), then complete the [four-source final project](../section8/projectIntro.md) using the five separate input datasets. Test answers and the final-project solution are instructor-only and are not included in the student book.\n'
    (ROOT/'section7/renewableExercises.md').write_text(index,encoding='utf-8')

if __name__=='__main__':main()
