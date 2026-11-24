# Geothermal

Ten assessment questions: four easy, four medium and two hard. Show reasoning, units and code where requested. Solutions are not published.

For SQL questions, use SQLite table `generation(country TEXT, source TEXT, year INTEGER, generation_gwh REAL)`. Each row represents one country–source–year observation; `generation_gwh` may be NULL. Source labels are `Hydro`, `Solar`, `Wind` and `Geothermal`. Missing observations are not zero.

### Question 51 <span class="difficulty-badge difficulty-easy">Easy</span>

Distinguish geothermal temperature gradient from heat flux and give units.

### Question 52 <span class="difficulty-badge difficulty-easy">Easy</span>

Explain the difference between direct heat use and electricity generation.

### Question 53 <span class="difficulty-badge difficulty-easy">Easy</span>

Convert 150°C to kelvin in Python.

### Question 54 <span class="difficulty-badge difficulty-easy">Easy</span>

Define gross electrical power, auxiliary load, and net electrical power.

### Question 55 <span class="difficulty-badge difficulty-medium">Medium</span>

Calculate thermal MW for mass flow 20 kg/s and enthalpy drop 500 kJ/kg, stating the conversion.

### Question 56 <span class="difficulty-badge difficulty-medium">Medium</span>

For 10 MW thermal input, 12% gross conversion efficiency and 0.3 MW auxiliary load, calculate gross and net MW.

### Question 57 <span class="difficulty-badge difficulty-medium">Medium</span>

Explain why reinjection and reservoir replenishment matter when estimating long-term production.

### Question 58 <span class="difficulty-badge difficulty-medium">Medium</span>

Using the shared `generation` schema, write SQL for geothermal GWh by country and year, showing row count and non-null measurement count.

### Question 59 <span class="difficulty-badge difficulty-hard">Hard</span>

Write a Python function for net electrical output from mass flow, enthalpy drop, conversion efficiency and auxiliary load. Validate physical input bounds and test zero-flow behavior.

### Question 60 <span class="difficulty-badge difficulty-hard">Hard</span>

For years 2021–2024 use annual mean production flows `[50, 47, None, 42]` kg/s. Assume a constant 300 kJ/kg enthalpy drop, 12% gross thermal-to-electric efficiency, auxiliaries equal to 10% of gross output, and 8,000 operating hours in each reported year. Treat each supplied flow as representative of those operating hours; leave the missing year unknown. Implement net annual generation and coverage reporting, and explain why extrapolation needs reservoir evidence.
