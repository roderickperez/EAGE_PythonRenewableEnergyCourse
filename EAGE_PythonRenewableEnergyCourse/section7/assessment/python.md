# Python

Ten assessment questions: four easy, four medium and two hard. Show reasoning, units and code where requested. Solutions are not published.

### Question 01 <span class="difficulty-badge difficulty-easy">Easy</span>

Explain the difference between a Python list, dictionary, and pandas DataFrame; give one use for each.

### Question 02 <span class="difficulty-badge difficulty-easy">Easy</span>

A plant runs at 3 MW for 4 h. Write Python to calculate MWh and label the result.

### Question 03 <span class="difficulty-badge difficulty-easy">Easy</span>

Write a function that converts kWh to MWh and demonstrate it with 2,500 kWh.

### Question 04 <span class="difficulty-badge difficulty-easy">Easy</span>

Explain the difference between a measured zero and a missing value in a generation time series.

### Question 05 <span class="difficulty-badge difficulty-medium">Medium</span>

Given powers `[2, 4, 1]` MW for durations `[1, 0.5, 2]` h, use Python to calculate total MWh and time-weighted mean MW.

### Question 06 <span class="difficulty-badge difficulty-medium">Medium</span>

Create a pandas DataFrame with three timestamps and power values `[2.0, None, 4.0]`. Count missing readings without filling them.

### Question 07 <span class="difficulty-badge difficulty-medium">Medium</span>

Write a function that accepts power and hours, rejects negative inputs with `ValueError`, and returns MWh. Demonstrate one valid and one invalid input.

### Question 08 <span class="difficulty-badge difficulty-medium">Medium</span>

Explain why duplicate timestamps and mixed time zones can invalidate a join of hourly resource and generation data. Describe two checks.

### Question 09 <span class="difficulty-badge difficulty-hard">Hard</span>

Using Python, align hourly generation and demand by timestamp using an outer join, preserve missing values, and report unmatched timestamps. Describe how you would validate energy totals.

### Question 10 <span class="difficulty-badge difficulty-hard">Hard</span>

Design and implement a testable function that calculates energy from interval-average power and unequal durations, rejects invalid inputs, and returns both total MWh and coverage. State your missing-data policy and test boundary cases.
