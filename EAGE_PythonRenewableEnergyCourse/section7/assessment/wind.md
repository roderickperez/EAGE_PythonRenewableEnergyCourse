# Wind

Ten assessment questions: four easy, four medium and two hard. Show reasoning, units and code where requested. Solutions are not published.

For SQL questions, use SQLite table `generation(country TEXT, source TEXT, year INTEGER, generation_gwh REAL)`. Each row represents one country–source–year observation; `generation_gwh` may be NULL. Source labels are `Hydro`, `Solar`, `Wind` and `Geothermal`. Missing observations are not zero.

### Question 41 <span class="difficulty-badge difficulty-easy">Easy</span>

Explain why wind is variable and distinguish wind speed from wind direction.

### Question 42 <span class="difficulty-badge difficulty-easy">Easy</span>

What is turbine rated power, and why is it not the same as annual energy?

### Question 43 <span class="difficulty-badge difficulty-easy">Easy</span>

Write Python to compute rotor swept area for radius 20 m using `math.pi`.

### Question 44 <span class="difficulty-badge difficulty-easy">Easy</span>

Why must cut-in and cut-out speeds be represented in a turbine power curve?

### Question 45 <span class="difficulty-badge difficulty-medium">Medium</span>

Calculate available wind power in kW for density 1.2 kg/m³, swept area 1,000 m² and wind speed 8 m/s using `0.5*rho*A*v**3`.

### Question 46 <span class="difficulty-badge difficulty-medium">Medium</span>

Write a function returning zero below 3 m/s, increasing linearly from 0 MW at 3 m/s to 2 MW at 12 m/s, remaining at 2 MW below 25 m/s, and returning zero at or above 25 m/s. Test all three boundary speeds.

### Question 47 <span class="difficulty-badge difficulty-medium">Medium</span>

Explain why the arithmetic mean of directions 355° and 5° is misleading and describe a circular-mean method.

### Question 48 <span class="difficulty-badge difficulty-medium">Medium</span>

Using the shared `generation` schema, write SQL to count wind records and non-null readings by year.

### Question 49 <span class="difficulty-badge difficulty-hard">Hard</span>

For hourly wind speeds `[2, 6, 10, None, -1, 26]` m/s, use an illustrative 3 MW curve: zero at or below 3 m/s and at or above 25 m/s; 3 MW from 12 to below 25 m/s; between 3 and 12 m/s use `3 * (v**3 - 3**3) / (12**3 - 3**3)` MW. Treat each valid speed as constant for its one-hour interval. Calculate energy, flag invalid and missing speeds, and report coverage separately.

### Question 50 <span class="difficulty-badge difficulty-hard">Hard</span>

Compare 3 MW and 4 MW versions of the curve in Question 49 using its same six one-hour speed observations and speed thresholds, scaling the ramp and plateau in proportion to rating. Flag invalid and missing values, check cut-out behavior, and report observed-period energy and coverage. These abstract curves are not tied to a rotor diameter. Explain why the short sample cannot establish long-term annual energy production.
