# Python–SQL Quiz

**4 Basic · 8 Medium · 8 Hard.** Try each question before opening its practice solution.

:::{admonition} Quiz 1 — Database and table
:class: note

**Difficulty:** Basic

How is a SQLite database different from a table?

:::

:::{admonition} Solution
:class: tip, dropdown

The database is the managed collection stored in a file; a table is one named collection of rows within it.

:::

:::{admonition} Quiz 2 — Select
:class: note

**Difficulty:** Basic

Write SQL to select `source` and `year` from `generation`.

:::

:::{admonition} Solution
:class: tip, dropdown

`SELECT source, year FROM generation;`

:::

:::{admonition} Quiz 3 — Filter
:class: note

**Difficulty:** Basic

Select all generation rows for 2023.

:::

:::{admonition} Solution
:class: tip, dropdown

`SELECT * FROM generation WHERE year = 2023;`

:::

:::{admonition} Quiz 4 — Count
:class: note

**Difficulty:** Basic

How do you count every row in `generation`?

:::

:::{admonition} Solution
:class: tip, dropdown

`SELECT COUNT(*) FROM generation;`

:::

:::{admonition} Quiz 5 — Pandas import
:class: note

**Difficulty:** Medium

Which Python imports provide pandas and SQLite access?

:::

:::{admonition} Solution
:class: tip, dropdown

`import pandas as pd` and `import sqlite3`.

:::

:::{admonition} Quiz 6 — Connect
:class: note

**Difficulty:** Medium

How do you open or create a local `course.db` SQLite file?

:::

:::{admonition} Solution
:class: tip, dropdown

`connection = sqlite3.connect("course.db")`. Close the connection when finished.

:::

:::{admonition} Quiz 7 — DataFrame to SQL
:class: note

**Difficulty:** Medium

Write DataFrame `df` into a new table called `generation` without its index.

:::

:::{admonition} Solution
:class: tip, dropdown

`df.to_sql("generation", connection, index=False)`; choose `if_exists` deliberately.

:::

:::{admonition} Quiz 8 — Read into pandas
:class: note

**Difficulty:** Medium

How do you load all rows from `generation` into a DataFrame?

:::

:::{admonition} Solution
:class: tip, dropdown

`pd.read_sql_query("SELECT * FROM generation", connection)`.

:::

:::{admonition} Quiz 9 — Sort
:class: note

**Difficulty:** Medium

Select source and GWh, largest GWh first.

:::

:::{admonition} Solution
:class: tip, dropdown

`SELECT source, generation_gwh FROM generation ORDER BY generation_gwh DESC;`

:::

:::{admonition} Quiz 10 — Missing values
:class: note

**Difficulty:** Medium

How do you select unknown generation values?

:::

:::{admonition} Solution
:class: tip, dropdown

`WHERE generation_gwh IS NULL`. SQL NULL is not zero and is not tested with `= NULL`.

:::

:::{admonition} Quiz 11 — Non-null count
:class: note

**Difficulty:** Medium

What is the difference between `COUNT(*)` and `COUNT(generation_gwh)`?

:::

:::{admonition} Solution
:class: tip, dropdown

The first counts rows; the second counts only non-NULL values in that column.

:::

:::{admonition} Quiz 12 — Conversion
:class: note

**Difficulty:** Medium

Write a query converting `generation_gwh` to MWh.

:::

:::{admonition} Solution
:class: tip, dropdown

`SELECT generation_gwh * 1000 AS generation_mwh FROM generation;`

:::

:::{admonition} Quiz 13 — Group
:class: note

**Difficulty:** Hard

How do you count rows for each source?

:::

:::{admonition} Solution
:class: tip, dropdown

`SELECT source, COUNT(*) AS n FROM generation GROUP BY source;`

:::

:::{admonition} Quiz 14 — Parameter binding
:class: note

**Difficulty:** Hard

How can Python safely bind a year in a SQLite query?

:::

:::{admonition} Solution
:class: tip, dropdown

`pd.read_sql_query("SELECT * FROM generation WHERE year = ?", connection, params=(2023,))`. Bind values, not untrusted table names.

:::

:::{admonition} Quiz 15 — Magic commands
:class: note

**Difficulty:** Hard

What do `%load_ext sql` and `%%sql` do in Jupyter?

:::

:::{admonition} Solution
:class: tip, dropdown

The first loads the installed extension. The second, on the first line of a cell, sends the cell body as SQL to the selected connection.

:::

:::{admonition} Quiz 16 — Connection URL
:class: note

**Difficulty:** Hard

What file does `%sql sqlite:///course.db` refer to?

:::

:::{admonition} Solution
:class: tip, dropdown

A SQLite file named `course.db` relative to the kernel working directory. The SQL magic extension must be installed.

:::

:::{admonition} Quiz 17 — Join
:class: note

**Difficulty:** Hard

Join `generation g` to `sources s` on source code.

:::

:::{admonition} Solution
:class: tip, dropdown

`SELECT g.source, s.label FROM generation AS g JOIN sources AS s ON g.source = s.source;`

:::

:::{admonition} Quiz 18 — Duplicate join
:class: note

**Difficulty:** Hard

Why can a join inflate a sum?

:::

:::{admonition} Solution
:class: tip, dropdown

If a matching key occurs multiple times in the joined table, one observation can be duplicated. Check key uniqueness and row counts.

:::

:::{admonition} Quiz 19 — Transactions
:class: note

**Difficulty:** Hard

What should happen when one step of a multi-step database update fails?

:::

:::{admonition} Solution
:class: tip, dropdown

Use a transaction and roll back the group so a partial update is not committed. Validate data and keep backups.

:::

:::{admonition} Quiz 20 — Replace
:class: note

**Difficulty:** Hard

What is the risk of `df.to_sql(..., if_exists="replace")`?

:::

:::{admonition} Solution
:class: tip, dropdown

It drops and recreates the named table, destroying its previous contents and potentially its constraints; use only a disposable tutorial table.

:::

## Further reading

[SQL basics](../../section5/sql-basics.md); [SQL and Python notebook](../../section5/sql-python.ipynb).
