# Solar

Ten assessment questions: four easy, four medium and two hard. Show reasoning, units and code where requested. Solutions are not published.

For SQL questions, use SQLite table `generation(country TEXT, source TEXT, year INTEGER, generation_gwh REAL)`. Each row represents one country–source–year observation; `generation_gwh` may be NULL. Source labels are `Hydro`, `Solar`, `Wind` and `Geothermal`. Missing observations are not zero.

### Question 31 <span class="difficulty-badge difficulty-easy">Easy</span>

Distinguish irradiance from irradiation and give suitable units for each.

### Question 32 <span class="difficulty-badge difficulty-easy">Easy</span>

Explain the role of an inverter in a photovoltaic system.

### Question 33 <span class="difficulty-badge difficulty-easy">Easy</span>

Write Python to convert 750 Wh/m² to kWh/m².

### Question 34 <span class="difficulty-badge difficulty-easy">Easy</span>

Why can a PV module’s rated power differ from its actual output?

### Question 35 <span class="difficulty-badge difficulty-medium">Medium</span>

Compute ideal DC power for irradiance 800 W/m², module area 12 m² and efficiency 20%; show units and Python.

### Question 36 <span class="difficulty-badge difficulty-medium">Medium</span>

A 10 kW DC output passes through an inverter with 96% efficiency and an 8 kW AC cap. Calculate the exported AC power and clipping.

### Question 37 <span class="difficulty-badge difficulty-medium">Medium</span>

Given interval-average AC outputs `[0, 3, 5, 2]` kW for 30 minutes each, calculate kWh and explain the time factor.

### Question 38 <span class="difficulty-badge difficulty-medium">Medium</span>

Using the shared `generation` schema, write SQL for annual non-null solar GWh by country and report the count of measurements used.

### Question 39 <span class="difficulty-badge difficulty-hard">Hard</span>

Write a function for temperature-adjusted PV DC power using stated reference power, irradiance, cell temperature and temperature coefficient. Validate inputs and test reference and hot conditions.

### Question 40 <span class="difficulty-badge difficulty-hard">Hard</span>

Compare two inverter caps using the same hourly DC series, inverter efficiency and missing-value policy. Implement annual energy and clipping calculations, and explain two design considerations beyond energy.
