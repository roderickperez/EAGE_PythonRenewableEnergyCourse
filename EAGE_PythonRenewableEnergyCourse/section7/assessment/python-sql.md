# Python & SQL

Ten assessment questions: four easy, four medium and two hard. Show reasoning, units and code where requested. Solutions are not published.

For SQL questions, use SQLite table `generation(country TEXT, source TEXT, year INTEGER, generation_gwh REAL)`. Each row represents one country–source–year observation; `generation_gwh` may be NULL. Source labels are `Hydro`, `Solar`, `Wind` and `Geothermal`. Missing observations are not zero.

### Question 11 <span class="difficulty-badge difficulty-easy">Easy</span>

Explain the roles of a database, a table, SQL, and pandas in an energy-data workflow.

### Question 12 <span class="difficulty-badge difficulty-easy">Easy</span>

Write Python imports for pandas and SQLite, then connect to a local file named `assessment.db`.

### Question 13 <span class="difficulty-badge difficulty-easy">Easy</span>

Write SQL that selects `source` and `generation_gwh` from `generation` for year 2023.

### Question 14 <span class="difficulty-badge difficulty-easy">Easy</span>

Explain the difference between `COUNT(*)` and `COUNT(generation_gwh)` when a measurement is NULL.

### Question 15 <span class="difficulty-badge difficulty-medium">Medium</span>

Create a DataFrame of three fictional country, source, year and GWh records, and write it to a new SQLite table. Explain the implications of `if_exists="replace"`.

### Question 16 <span class="difficulty-badge difficulty-medium">Medium</span>

Use `pd.read_sql_query` and a bound `?` parameter to retrieve one country from `generation`; close the connection.

### Question 17 <span class="difficulty-badge difficulty-medium">Medium</span>

Write SQL to group by source and return row count, non-null measurement count, and average GWh.

### Question 18 <span class="difficulty-badge difficulty-medium">Medium</span>

Describe what `%load_ext sql`, `%sql sqlite:///assessment.db`, and `%%sql` do in a Jupyter notebook and where `%%sql` must appear.

### Question 19 <span class="difficulty-badge difficulty-hard">Hard</span>

Design `sources(source PRIMARY KEY, label)` and a `generation` table with a foreign key, a composite key and an appropriate nonnegative check. Explain NULL and SQLite foreign-key enforcement.

### Question 20 <span class="difficulty-badge difficulty-hard">Hard</span>

A join duplicates some generation records because source labels are not unique. Write SQL checks for duplicate keys and unmatched rows, then propose a corrected join and reconciliation against the original total.
