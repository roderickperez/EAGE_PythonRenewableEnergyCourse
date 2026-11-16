# Assessment

**60 questions · six topics · 10 questions per topic.** Each topic has 4 easy, 4 medium and 2 hard questions. Answer the theory questions in words and show calculations, units, assumptions and code where requested. The questions use synthetic teaching values and simplified models. Solutions are not published here. Use a fresh notebook and your own work.

For questions referring to the shared SQL schema, assume a SQLite table `generation(country TEXT, source TEXT, year INTEGER, generation_gwh REAL)`. One row is one country–source–year observation; `generation_gwh` may be NULL. Source labels are `Hydro`, `Solar`, `Wind` and `Geothermal`. Do not assume missing observations are zero or add overlapping categories.

## General Python

### Question 01 — Easy

Explain the difference between a Python list, dictionary, and pandas DataFrame; give one use for each.

### Question 02 — Easy

A plant runs at 3 MW for 4 h. Write Python to calculate MWh and label the result.

### Question 03 — Easy

Write a function that converts kWh to MWh and demonstrate it with 2,500 kWh.

### Question 04 — Easy

Explain the difference between a measured zero and a missing value in a generation time series.

### Question 05 — Medium

Given powers `[2, 4, 1]` MW for durations `[1, 0.5, 2]` h, use Python to calculate total MWh and time-weighted mean MW.

### Question 06 — Medium

Create a pandas DataFrame with three timestamps and power values `[2.0, None, 4.0]`. Count missing readings without filling them.

### Question 07 — Medium

Write a function that accepts power and hours, rejects negative inputs with `ValueError`, and returns MWh. Demonstrate one valid and one invalid input.

### Question 08 — Medium

Explain why duplicate timestamps and mixed time zones can invalidate a join of hourly resource and generation data. Describe two checks.

### Question 09 — Hard

Using Python, align hourly generation and demand by timestamp using an outer join, preserve missing values, and report unmatched timestamps. Describe how you would validate energy totals.

### Question 10 — Hard

Design and implement a testable function that calculates energy from interval-average power and unequal durations, rejects invalid inputs, and returns both total MWh and coverage. State your missing-data policy and test boundary cases.

## Python and SQL

### Question 11 — Easy

Explain the roles of a database, a table, SQL, and pandas in an energy-data workflow.

### Question 12 — Easy

Write Python imports for pandas and SQLite, then connect to a local file named `assessment.db`.

### Question 13 — Easy

Write SQL that selects `source` and `generation_gwh` from `generation` for year 2023.

### Question 14 — Easy

Explain the difference between `COUNT(*)` and `COUNT(generation_gwh)` when a measurement is NULL.

### Question 15 — Medium

Create a DataFrame of three fictional country, source, year and GWh records, and write it to a new SQLite table. Explain the implications of `if_exists="replace"`.

### Question 16 — Medium

Use `pd.read_sql_query` and a bound `?` parameter to retrieve one country from `generation`; close the connection.

### Question 17 — Medium

Write SQL to group by source and return row count, non-null measurement count, and average GWh.

### Question 18 — Medium

Describe what `%load_ext sql`, `%sql sqlite:///assessment.db`, and `%%sql` do in a Jupyter notebook and where `%%sql` must appear.

### Question 19 — Hard

Design `sources(source PRIMARY KEY, label)` and a `generation` table with a foreign key, a composite key and an appropriate nonnegative check. Explain NULL and SQLite foreign-key enforcement.

### Question 20 — Hard

A join duplicates some generation records because source labels are not unique. Write SQL checks for duplicate keys and unmatched rows, then propose a corrected join and reconciliation against the original total.

## Hydroelectric energy

### Question 21 — Easy

Define gross head, head loss and net head, and identify the units used for each.

### Question 22 — Easy

Distinguish power in MW from energy in MWh for a hydroelectric plant.

### Question 23 — Easy

Write Python to calculate net head given gross head 42 m and loss 3 m.

### Question 24 — Easy

Explain the difference between a run-of-river plant and pumped storage.

### Question 25 — Medium

Using density 1,000 kg/m³, g 9.81 m/s², flow 8 m³/s, net head 35 m and efficiency 0.85, write Python to calculate electrical MW.

### Question 26 — Medium

For river flows `[1, 5, 9]` m³/s and a 3 m³/s environmental target, calculate release, shortfall, and usable flow for each interval.

### Question 27 — Medium

A reservoir starts at 20,000 m³, receives 4 m³/s and releases 3 m³/s for one hour. Write Python to calculate final storage, stating assumptions.

### Question 28 — Medium

Using the shared `generation` schema below, write SQL to count non-null hydroelectric readings and calculate their average GWh by country.

### Question 29 — Hard

Implement a reservoir routing function with one-hour intervals, environmental release before turbine withdrawal, a maximum storage and spill. Test a low-inflow and an overflow case and verify mass balance.

### Question 30 — Hard

Compare two generator ratings using hourly flows, fixed net head and efficiency. Write a Python approach for clipped power and energy, then describe an SQL query to compare summed energy by rating without counting missing intervals as zero.

## Solar energy

### Question 31 — Easy

Distinguish irradiance from irradiation and give suitable units for each.

### Question 32 — Easy

Explain the role of an inverter in a photovoltaic system.

### Question 33 — Easy

Write Python to convert 750 Wh/m² to kWh/m².

### Question 34 — Easy

Why can a PV module’s rated power differ from its actual output?

### Question 35 — Medium

Compute ideal DC power for irradiance 800 W/m², module area 12 m² and efficiency 20%; show units and Python.

### Question 36 — Medium

A 10 kW DC output passes through an inverter with 96% efficiency and an 8 kW AC cap. Calculate the exported AC power and clipping.

### Question 37 — Medium

Given interval-average AC outputs `[0, 3, 5, 2]` kW for 30 minutes each, calculate kWh and explain the time factor.

### Question 38 — Medium

Using the shared `generation` schema, write SQL for annual non-null solar GWh by country and report the count of measurements used.

### Question 39 — Hard

Write a function for temperature-adjusted PV DC power using stated reference power, irradiance, cell temperature and temperature coefficient. Validate inputs and test reference and hot conditions.

### Question 40 — Hard

Compare two inverter caps using the same hourly DC series, inverter efficiency and missing-value policy. Implement annual energy and clipping calculations, and explain two design considerations beyond energy.

## Wind energy

### Question 41 — Easy

Explain why wind is variable and distinguish wind speed from wind direction.

### Question 42 — Easy

What is turbine rated power, and why is it not the same as annual energy?

### Question 43 — Easy

Write Python to compute rotor swept area for radius 20 m using `math.pi`.

### Question 44 — Easy

Why must cut-in and cut-out speeds be represented in a turbine power curve?

### Question 45 — Medium

Calculate available wind power in kW for density 1.2 kg/m³, swept area 1,000 m² and wind speed 8 m/s using `0.5*rho*A*v**3`.

### Question 46 — Medium

Write a function returning zero below 3 m/s, increasing linearly from 0 MW at 3 m/s to 2 MW at 12 m/s, remaining at 2 MW below 25 m/s, and returning zero at or above 25 m/s. Test all three boundary speeds.

### Question 47 — Medium

Explain why the arithmetic mean of directions 355° and 5° is misleading and describe a circular-mean method.

### Question 48 — Medium

Using the shared `generation` schema, write SQL to count wind records and non-null readings by year.

### Question 49 — Hard

Given hourly wind speed data and a specified power curve, write Python to calculate energy while flagging invalid and missing speeds; report coverage separately.

### Question 50 — Hard

Describe and implement a comparison of two turbine ratings over the same speed series. Check interval duration, cut-out behavior and missing data, and explain why a short sample cannot establish long-term AEP.

## Geothermal energy

### Question 51 — Easy

Distinguish geothermal temperature gradient from heat flux and give units.

### Question 52 — Easy

Explain the difference between direct heat use and electricity generation.

### Question 53 — Easy

Convert 150°C to kelvin in Python.

### Question 54 — Easy

Define gross electrical power, auxiliary load, and net electrical power.

### Question 55 — Medium

Calculate thermal MW for mass flow 20 kg/s and enthalpy drop 500 kJ/kg, stating the conversion.

### Question 56 — Medium

For 10 MW thermal input, 12% gross conversion efficiency and 0.3 MW auxiliary load, calculate gross and net MW.

### Question 57 — Medium

Explain why reinjection and reservoir replenishment matter when estimating long-term production.

### Question 58 — Medium

Using the shared `generation` schema, write SQL for geothermal GWh by country and year, showing row count and non-null measurement count.

### Question 59 — Hard

Write a Python function for net electrical output from mass flow, enthalpy drop, conversion efficiency and auxiliary load. Validate physical input bounds and test zero-flow behavior.

### Question 60 — Hard

Given annual declining flow measurements with gaps, propose and implement a Python workflow to calculate net generation and report missing coverage; explain why extrapolation needs reservoir evidence.
