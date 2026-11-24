# Project 5: Final Project — Design and Analyse a Hybrid Renewable Portfolio

## Purpose

Use **one Python notebook** to turn hourly solar, wind, river-flow, geothermal and demand inputs into an analysis of an isolated electricity grid. Start with the five supplied CSVs and carry the same validated hourly data through the technology models, grid balance, plots and scenario comparison. [Download the project starter notebook](projectStarter.ipynb).

## Scenario

A small isolated grid is considering four renewable resources:

- a PV array with a DC rating and an AC inverter limit;
- a wind plant described by cut-in, rated, and cut-out speeds;
- a run-of-river plant with environmental flow and a generator rating;
- a geothermal well field with a gross conversion efficiency and parasitic load.

Analyse one synthetic week at hourly resolution. The synthetic weather and demand make every submission reproducible and avoid implying that the results describe a real site.

:::{iframe} ../../project-map/index.html
:width: 100%
:title: Isometric renewable portfolio and community demand map
:::

## Load the separate input datasets

Download and read each CSV separately before joining on `timestamp`:

- [Solar resource](data/solar.csv): `irradiance_wm2`, `ambient_c`.
- [Wind resource](data/wind.csv): `wind_speed_ms` at hub height.
- [Hydro resource](data/hydro.csv): `river_flow_m3s` before environmental reservation.
- [Geothermal resource](data/geothermal.csv): `mass_flow_kgs`, `production_c`, `reinjection_c`.
- [Electricity demand](data/demand.csv): `demand_mw`.

See the [dataset definitions](data/README.md). Every file contains 168 hourly
interval-average inputs with UTC timestamps, from 1 June 2025 at 00:00 through
7 June at 23:00. Each interval lasts one hour. These are synthetic observations,
not current data for a real location. **Read all five files; do not replace them
with newly generated data or hard-code geothermal output.**

In the repository the files are in `EAGE_PythonRenewableEnergyCourse/section8/data`.
To download them, open each of the five links above and use the browser download/save control; save the raw CSV, not an HTML page. Keep the filenames `solar.csv`, `wind.csv`, `hydro.csv`, `geothermal.csv`, and `demand.csv` together. In Colab, open the left Files panel, choose Upload, and select all five CSVs. In Jupyter, use the file browser Upload control or copy them into your project folder. In the course sandbox's **Workspace** tab, the working directory is `/course/section8` and the files are available automatically at the relative path `data/` (absolute path `/course/section8/data/`); no external download is needed there (the separate **Notebook** tab running JupyterLite only bundles `Start here.ipynb`, so upload the five CSVs there if you use that tab). Select the dataset folder explicitly
with `pathlib.Path`, read each file using `pandas.read_csv`, parse times with
`pandas.to_datetime(..., utc=True)`, and validate before joining. Do not rely on
matching row positions. The first timestamp labels the start of its interval.

### Common model parameters

| Source | Required parameters |
|---|---|
| Solar | 16 MW DC, 13 MW AC, NOCT 45°C, gamma −0.004/°C, inverter efficiency 0.97 |
| Wind | 12 MW rated, cut-in 3 m/s, rated speed 12 m/s, cut-out 25 m/s |
| Hydro | 8 MW rated, net head 42 m, efficiency 0.88, environmental reservation 6 m³/s |
| Geothermal | 3.5 MW **net nameplate**, gross thermal-to-electric efficiency 0.12, parasitic fraction 0.10 of gross electricity |

Use water density 1000 kg/m³, gravity 9.81 m/s² and single-phase water heat
capacity 4180 J/(kg K). Read geothermal flow and both temperatures from its CSV
for every interval; cap the resulting net electrical output at 3.5 MW. Use this
fixed rating for geothermal capacity factor, not the observed maximum.
There is no storage, import or export in the base case: unused surplus is curtailed
and uncovered demand is shortfall.

## Required engineering models

Use these simplified boundaries consistently:

### Solar PV

Basis: [@pvlibDocs] for the linear DC approximation; [Sandia NOCT](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/cell-temperature/noct-cell-temperature/) for the fuller thermal model. Here POA equals effective irradiance and the reduced NOCT equation, constant inverter efficiency and clipping are specified assumptions. $G$ is W/m²; all temperatures are °C; both electrical ratings are MW.

$$T_c=T_a+\frac{G}{800}(NOCT-20)$$

$$P_{DC}=P_{DC,r}\frac{G}{1000}\max\left(0,1+\gamma_P(T_c-25)\right)$$

$$P_{AC}=\min(P_{DC}\eta_{inv},P_{AC,r})$$

### Wind

Basis: [@manwell2009], §§2.5 and 3.2. The piecewise cubic ramp is the stipulated teaching curve; $v$, $v_{in}$ and $v_r$ are m/s, and $P_r$ is MW.

Use a four-region turbine curve: zero below cut-in, a continuous ramp to rated power, rated power until cut-out, and zero at or above cut-out. Do not extend $v^3$ through the rated and shutdown regions.

In the ramp region use $P=P_r(v^3-v_{in}^3)/(v_r^3-v_{in}^3)$.

### Hydropower

Basis: [@jica2011], §3.1.2 and §8.3; [@ifc2015], Chapters 7 and 12. Flows are m³/s, head is m, and efficiency is dimensionless. The hydraulic expression returns W; convert to MW before applying the 8 MW cap. This scenario measures the environmental target in a separate bypass reach upstream of the turbine return, so the reserved water is unavailable to the turbine. A requirement measured downstream of the tailrace may instead include qualifying turbine discharge; actual routing depends on the site's licence.

$$Q_{usable}=\max(Q_{river}-Q_{environmental},0)$$

$$P_h=\min\left(\frac{\eta\rho gQ_{usable}H_{net}}{10^6},P_{rated}\right)\quad\text{(MW)}$$

For the stated bypass-first rule, environmental release is $Q_{env,served}=\min(Q_{river},Q_{environmental})$, and environmental-flow shortfall is $Q_{env,short}=\max(Q_{environmental}-Q_{river},0)$. For example, 4 m³/s of river flow against a 6 m³/s target leaves 2 m³/s of environmental shortfall and no flow for the turbine. This water-flow shortfall is separate from electricity shortfall.

### Geothermal

$$P_{th}=\dot m c_p(T_p-T_r)$$

$$P_{net}=\min\left(\frac{P_{th}\eta_{conv}(1-f_{parasitic})}{10^6},P_{r,net}\right)\quad\text{(MW)}$$

Use a consistent unit before applying the cap: the sensible-heat formula returns W with the stated SI inputs, while $P_{r,net}=3.5$ MW. Divide W by $10^6$ before comparing with 3.5. [@grant2011], Chapters 2–3; the net rating is a specified project assumption.

The geothermal formula is a simplified single-phase sensible-heat model. State that an enthalpy model is needed for flashing or two-phase fluid.

## Basic — load and check the inputs

### Step 1 — Configure the notebook and read the CSVs

Set one dataset directory with `pathlib.Path`. Import pandas, NumPy and Matplotlib. Put the required ratings and assumptions into a parameter dictionary. Use `pd.read_csv` five times (or once in a loop), with a separate DataFrame for each supplied file. Print the file names, column names and units. The five DataFrames are the inputs to Step 2.

### Step 2 — Validate each table and join by UTC timestamp

Parse dates as UTC and sort chronologically. The supplied timestamps include `+00:00`. When adapting the project, establish the source timezone first: `pd.to_datetime(..., utc=True)` treats timezone-naive timestamps as UTC, so it does not convert an unknown local clock time. Localize known local times using the correct timezone and explicitly handle ambiguous or nonexistent daylight-saving times before converting to UTC. Convert required resource and demand columns with `pandas.to_numeric(..., errors="raise")` on copies, then use those converted values for validation, joins and calculations; this prevents numeric text from being concatenated. Use a reusable validation function and report row count, required columns, duplicate timestamps, missing and non-finite values, and physical ranges for each table. Stop before calculating the complete-week metrics if required values are missing or invalid; investigate or apply a documented, justified repair and rerun validation. Identify repaired values and label resulting totals accordingly. Do not silently replace missing observations with zero or present partial sums as a full week. Check that production temperature is at least reinjection temperature and that resource rates and demand are non-negative. Verify the exact 168-hour coverage and one-hour spacing in every table. Join by timestamp with one-to-one validation only after checking that all five timelines match. The output is a single hourly DataFrame with synthetic input values and a unique UTC index.

## Medium — model each source and build the grid balance

### Step 3 — Write and test the unit and technology functions

Write reusable helpers for W to MW and for integrating MW over hours to get MWh. Implement four vectorized technology functions returning MW using the equations above. Document units, reject invalid inputs and enforce equipment limits. Test each function with a hand-calculated input, and include zero-resource, clipped and wind cut-out cases where relevant.

### Step 4 — Calculate hourly power and reconcile supply with demand

Call the four functions on the joined DataFrame from Step 2. Add one power column per source, total renewable power and the supplied demand. Derive the hourly served, curtailed and shortfall columns using the no-storage identities below. Verify the source sum and both balances for every hour. Return this hourly table from a reusable function so the scenario stage can rerun the same calculation with different parameters.

Also record the environmental release and environmental-flow shortfall in m³/s for each interval under the bypass-first assumption. Integrate shortfall flow over each interval to report a total shortfall volume in m³; for an hourly mean, multiply by 3,600 seconds. Keep that result distinct from electricity shortfall in MWh.

At every hour, use the following no-storage accounting convention, treating generation and demand as constant within the hour. The result is an hourly-resolution approximation: hourly averages alone cannot determine exact simultaneous supply and demand within the interval. Each $E$ is energy over that same interval; the basis is conservation of energy [@foster2010], §3.8.1.

$$E_{served}=\min(E_{renewable},E_{demand})$$

$$E_{curtailed}=\max(E_{renewable}-E_{demand},0)$$

$$E_{shortfall}=\max(E_{demand}-E_{renewable},0)$$

For one-hour intervals, the numerical sum of MW samples is MWh. If you change the resolution, multiply by the time-step duration.

## Hard — evaluate the portfolio and compare designs

### Step 5 — Summarise the baseline and make the figures

Use the hourly balance from Step 4 to calculate weekly energy by source, available renewable energy, demand, served energy, curtailment and shortfall in MWh. Report renewable coverage as total served energy divided by total demand. If adapted inputs have zero total demand, report coverage as undefined rather than zero or 100%; explain that the ratio has no denominator, and do not rank scenarios by that value. Calculate each capacity factor against its fixed electrical rating and 168 hours: use the AC rating for solar and the net nameplate for geothermal.

Produce four Matplotlib figures:

1. resource inputs over time;
2. source power and total renewable power over time;
3. demand versus renewable supply, with shortfall or curtailment visible;
4. an energy-by-source or capacity-factor comparison.

Axes, legends, time period, and units are mandatory. Avoid dual axes unless the quantities cannot be compared otherwise and both scales are explicit.

### Step 6 — Run a scenario loop and defend a recommendation

Keep the baseline inputs and parameter dictionary. Choose two parameters to vary, such as PV DC/AC ratio and environmental flow. In a loop or comprehension, construct at least three configurations: baseline, change A only, and change B only. You can also run both changes together. For each configuration, reuse the same joined input data and the functions from Steps 3–5. Compare source energy, coverage, shortfall and curtailment in a table and a scenario plot. Explain which change you would choose under a stated criterion and one limitation of drawing conclusions from a synthetic week.

Include environmental-flow shortfall volume separately in the scenario comparison, especially when changing the reservation. A zero shortfall in the supplied synthetic week is not evidence of compliance at a real site.

## Required validation

- Prove that the hourly index is unique and chronological.
- Assert that every source power is non-negative and does not exceed its rating.
- Reconcile total renewable power with the sum of source columns.
- Reconcile served + curtailed with renewable energy, and served + shortfall with demand.
- Confirm $0\le CF\le1$. When total demand is positive, confirm $0\le$ renewable coverage $\le1$; when total demand is zero, report coverage as undefined.
- Check at least one hand-calculated hour against the function result.

## Submission checklist

Submit one executed Python notebook with the six steps in order, plus the source-energy and grid-balance table, scenario table, four baseline figures, scenario comparison plot and short interpretation. The notebook should recreate the tables and figures from the supplied CSVs when run from top to bottom in a fresh kernel.

## Assessment rubric

| Criterion | Weight |
|---|---:|
| Renewable-energy concepts, equations, and model boundaries | 25% |
| Python functions, readability, and reuse | 20% |
| Units, aggregation, and numerical correctness | 15% |
| Data quality, assertions, and reconciliation | 15% |
| Plots and interpretation | 15% |
| Scenario reasoning and limitations | 10% |

## References and further reading

Hydropower and environmental flow [@ifc2015; @jica2011]; PV performance [@foster2010]; wind and hybrid systems [@manwell2009]; geothermal models [@grant2011]; multi-energy integration [@hu2026].
