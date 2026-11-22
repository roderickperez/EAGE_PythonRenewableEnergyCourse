# SQL basics

SQL (Structured Query Language) asks a database to create, retrieve and change data. This lesson uses SQLite syntax and a small, fictional table; run its statements in order in a fresh SQLite database, for example with the `sqlite3` command-line client. The separate [Python and SQLite notebook](sql-python.ipynb) shows a related Python workflow using its own `generation_tutorial` table; it does not create this lesson’s `observations` table. SQL keywords are conventionally uppercase, but SQLite accepts lowercase. A semicolon ends a statement.

## Create and insert

```sql
CREATE TABLE observations (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    year INTEGER NOT NULL CHECK (typeof(year) = 'integer'),
    generation_gwh REAL CHECK (generation_gwh IS NULL OR typeof(generation_gwh) IN ('integer', 'real'))
);
INSERT INTO observations (source, year, generation_gwh) VALUES
    ('Solar', 2023, 12.5), ('Wind', 2023, 20.0),
    ('Solar', 2024, 15.0), ('Wind', 2024, NULL);
```

`PRIMARY KEY` identifies a row; `NOT NULL` requires a value. The year check requires an integer storage value; [SQLite's integer affinity](https://www.sqlite.org/datatype3.html#type_affinity) can first convert numeric text such as `'2023'` to an integer. The check does not establish a sensible year range, so choose and validate that separately. Here `NULL` represents an unknown measurement, not zero. The generation `CHECK` accepts NULL or numeric values and rejects nonnumeric text; add `NOT NULL` if measurements must always exist. It does not reject infinity, so validate that measurements are finite before inserting them when finite values are required. Decide separately how to handle missing values; for example, Python’s SQLite binding can store a floating-point NaN as SQL NULL. Negative net generation can be meaningful for storage and other source-defined series, so apply any sign constraint only after checking the provider definition. Run the create statement once, or use a fresh tutorial database.

## Select, filter and sort

```sql
SELECT source, year, generation_gwh FROM observations;
SELECT source, generation_gwh FROM observations
WHERE year = 2023 AND generation_gwh > 10
ORDER BY generation_gwh DESC;
SELECT DISTINCT source FROM observations;
SELECT source, year FROM observations WHERE generation_gwh IS NULL;
```

`SELECT` chooses columns; `FROM` identifies the table; `WHERE` filters rows; `ORDER BY` sorts them. Use `IS NULL`, not `= NULL`. Never rely on a row order without `ORDER BY`.

## Calculate and summarize

```sql
SELECT source, year, generation_gwh * 1000 AS generation_mwh
FROM observations;
SELECT COUNT(*) AS rows, COUNT(generation_gwh) AS measured_rows,
       SUM(generation_gwh) AS total_gwh, AVG(generation_gwh) AS average_gwh
FROM observations;
SELECT source, COUNT(*) AS rows, COUNT(generation_gwh) AS measured_rows,
       AVG(generation_gwh) AS mean_gwh
FROM observations GROUP BY source HAVING COUNT(*) >= 2;
```

`COUNT(*)` includes all four rows; `COUNT(generation_gwh)` counts the three non-null measurements. `SUM` and `AVG` ignore NULL, so report coverage and avoid treating a partial total as complete. In SQLite, `SUM` and `AVG` return NULL when there are no non-NULL inputs, including an all-missing group; a measured zero instead produces a numeric zero. SQLite's `TOTAL` returns 0.0 for no non-NULL inputs, so it cannot by itself distinguish these cases. `GROUP BY` summarizes each source. This `HAVING` condition requires two rows, not two measurements: Wind has two rows but only one measured value. To require at least two measurements instead, use `HAVING COUNT(generation_gwh) >= 2`. Only combine measurements with compatible units, periods and boundaries.

## Modify and join

```sql
UPDATE observations SET generation_gwh = 21.0
WHERE source = 'Wind' AND year = 2024;
CREATE TABLE sources (source TEXT PRIMARY KEY NOT NULL, category TEXT NOT NULL);
INSERT INTO sources VALUES ('Solar', 'Photovoltaic'), ('Wind', 'Onshore');
SELECT o.source, o.year, s.category
FROM observations AS o JOIN sources AS s ON o.source = s.source;
DELETE FROM observations WHERE id = 4;
```

`UPDATE` and `DELETE` change stored records; omitting `WHERE` affects every row. The explicit `NOT NULL` is needed for an ordinary SQLite text primary key; an `INTEGER PRIMARY KEY` has different behavior. A join matches rows using a relationship, so verify key uniqueness to avoid multiplying observations. Use transactions and backups for real data. Parameterize values in Python rather than inserting user input into SQL strings. See the [SQLite language reference](https://www.sqlite.org/lang.html) and [Python sqlite3 documentation](https://docs.python.org/3/library/sqlite3.html).

Next, use the [notebook](sql-python.ipynb) to create and query a database from pandas, then explore the [renewable-energy SQL examples](SQL.md).
