# Hydroelectric

Ten assessment questions: four easy, four medium and two hard. Show reasoning, units and code where requested. Solutions are not published.

For SQL questions, use SQLite table `generation(country TEXT, source TEXT, year INTEGER, generation_gwh REAL)`. Each row represents one country–source–year observation; `generation_gwh` may be NULL. Source labels are `Hydro`, `Solar`, `Wind` and `Geothermal`. Missing observations are not zero.

### Question 21 <span class="difficulty-badge difficulty-easy">Easy</span>

Define gross head, head loss and net head, and identify the units used for each.

### Question 22 <span class="difficulty-badge difficulty-easy">Easy</span>

Distinguish power in MW from energy in MWh for a hydroelectric plant.

### Question 23 <span class="difficulty-badge difficulty-easy">Easy</span>

Write Python to calculate net head given gross head 42 m and loss 3 m.

### Question 24 <span class="difficulty-badge difficulty-easy">Easy</span>

Explain the difference between a run-of-river plant and pumped storage.

### Question 25 <span class="difficulty-badge difficulty-medium">Medium</span>

Using density 1,000 kg/m³, g 9.81 m/s², flow 8 m³/s, net head 35 m and efficiency 0.85, write Python to calculate electrical MW.

### Question 26 <span class="difficulty-badge difficulty-medium">Medium</span>

For river flows `[1, 5, 9]` m³/s and a 3 m³/s environmental target, calculate release, shortfall, and usable flow for each interval.

### Question 27 <span class="difficulty-badge difficulty-medium">Medium</span>

A reservoir starts at 20,000 m³, receives 4 m³/s and releases 3 m³/s for one hour. Write Python to calculate final storage, stating assumptions.

### Question 28 <span class="difficulty-badge difficulty-medium">Medium</span>

Using the shared `generation` schema below, write SQL to count non-null hydroelectric readings and calculate their average GWh by country.

### Question 29 <span class="difficulty-badge difficulty-hard">Hard</span>

Implement a reservoir routing function with one-hour intervals, environmental release before turbine withdrawal, a maximum storage and spill. Test a low-inflow and an overflow case and verify mass balance.

### Question 30 <span class="difficulty-badge difficulty-hard">Hard</span>

Compare two generator ratings using hourly flows, fixed net head and efficiency. Write a Python approach for clipped power and energy, then describe an SQL query to compare summed energy by rating without counting missing intervals as zero.
