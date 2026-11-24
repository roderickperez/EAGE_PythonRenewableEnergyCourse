# Introduction to databases

A **database** is an organized collection of data; a **database management system (DBMS)** manages storage, retrieval and updates. CSV is a file format, SQL is a language, and SQLite/PostgreSQL are database systems. [PostgreSQL concepts](https://www.postgresql.org/docs/current/tutorial-concepts.html), [SQLite](https://www.sqlite.org/about.html), [Python CSV](https://docs.python.org/3/library/csv.html)

## Structured and unstructured data

Structured data has a predictable organization: a table of monthly generation with country, source, month and GWh is one example. Semi-structured data such as JSON has fields and nesting but records may differ. Unstructured data includes free text, images and audio, whose contents are not naturally a fixed table. These are descriptions of the data, not judgments about its value. A database may store or index several kinds; sensor readings are usually structured even when numerous.

## Why use a database instead of only files?

A CSV or spreadsheet is convenient for sharing, small experiments and transparent inspection. A DBMS becomes useful when many tables must be connected, records updated consistently, multiple users work concurrently, or queries must select small subsets of a larger collection. Keys and constraints help prevent duplicates and inconsistent relationships; transactions group changes. A database also introduces design, maintenance and backup work. It does not replace backups or automatically make incorrect data trustworthy.

| Question | CSV or DataFrame | Database |
|---|---|---|
| Organization | Usually one independent table at a time | Multiple related tables and enforced rules |
| Filtering | Load and filter in Python or spreadsheet software | Query selected rows with SQL, often using indexes |
| Updates | Rewrite or coordinate files and versions | Transactions and controlled updates |
| Collaboration | File sharing and locking need planning | Depends on the DBMS; server systems support concurrent clients |

Pandas DataFrames are in-memory tables for analysis; they complement a persistent database. A database table is also tabular: the distinction is management and operations, not whether data has rows and columns.

## Relational and non-relational systems

Relational systems organize data in tables with relationships and usually support SQL. **SQLite** runs embedded in an application and is useful for a portable local tutorial; **PostgreSQL** is a client-server relational DBMS suited to shared applications and larger deployments. Neither is automatically best for every workload. Non-relational systems include document stores (nested records), key-value stores (lookup by key), graph databases (relationships and traversals), and wide-column stores. Products vary in transactions, schemas and query languages; choose by the actual access pattern and operational needs. PostgreSQL can also store JSON. [PostgreSQL tutorial](https://www.postgresql.org/docs/current/tutorial.html), [SQLite overview](https://www.sqlite.org/about.html).

## Separate data models, deployment and workload

| Question | Examples | Classification |
|---|---|---|
| How are records represented? | Relational, document, key-value, graph | Data model |
| Where/how is the service operated? | Embedded, managed cloud, distributed | Deployment |
| What work does it do? | Operational transactions, analytical queries | Workload |

A relational database can be cloud-hosted and operational at the same time. Cloud, centralized and relational are not mutually exclusive types. Cloud hosting does not automatically guarantee security, low cost or speed. [PostgreSQL architecture](https://www.postgresql.org/docs/current/tutorial-arch.html), [Amazon RDS overview](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html)

## Relational concepts

| Term | Meaning | Example |
|---|---|---|
| Table | Named collection of rows and columns | `generation` |
| Row / grain | One observation at a stated level of detail | One country–source–month |
| Column | Attribute with a defined meaning | `generation_gwh` |
| Primary key | Unique identifier; require non-null key fields | `(geo, source, month)` |
| Foreign key | Reference to a primary/unique parent key | Source in the source table |
| Constraint | Rule enforced on insert/update | Required unit |
| Index | Structure supporting selected lookups | Month index |
| Transaction | Operations committed or rolled back together | Insert a validated download |

Sources: [PostgreSQL constraints](https://www.postgresql.org/docs/current/ddl-constraints.html), [transactions](https://www.postgresql.org/docs/current/tutorial-transactions.html), [SQLite indexes](https://www.sqlite.org/lang_createindex.html).

An electricity-flow diagram is not a database schema: a schema identifies stored fields, keys and relationships. Our original exercise schema is:

```text
energy_source(source PRIMARY KEY, label)
             1
             |
            many
generation(geo, source FOREIGN KEY, month, generation_gwh, status_flag)
PRIMARY KEY (geo, source, month)
```

Immediately after opening each SQLite connection, before beginning a transaction or savepoint, run `PRAGMA foreign_keys = ON` and verify `PRAGMA foreign_keys` returns 1. Changing this setting inside an active transaction has no effect. [SQLite foreign keys](https://www.sqlite.org/foreignkeys.html)

## Missing data and units

SQL `NULL` means missing/unknown, not zero. `SUM` and `AVG` omit null inputs; `COUNT(*)` counts rows while `COUNT(column)` counts non-null values. Report coverage with a total. Keys prevent duplicate records, but cannot establish physical units or prevent double-counting aggregate categories. [SQLite aggregates](https://www.sqlite.org/lang_aggfunc.html)

Historical course exports contain monthly **net electricity generation in GWh**, not consumption. Broad hydro may include pumped-storage output; a selected-category sum is not automatically an official renewable-only or EU-27 total. Keep geography, product, units, flags and period. [Eurostat energy metadata](https://ec.europa.eu/eurostat/web/energy/methodology)

## Other models

Document databases can store nested records and still use schema validation/indexes. Time-series sensor data are often structured and can be stored relationally. “Energy data” does not imply “unstructured data.” [MongoDB modelling](https://www.mongodb.com/docs/manual/data-modeling/)

Hierarchical and network DBMSs are historical navigational data models; object-oriented databases persist objects. They are different from the cloud/operational deployment labels. We use SQLite because it is embedded and requires no separate database server. [IBM database models](https://www.ibm.com/think/topics/database), [SQLite serverless architecture](https://www.sqlite.org/serverless.html)

## Practical route

1. Work through [SQL basics](sql-basics.md) and the [Python and SQLite notebook](sql-python.ipynb).
2. Apply the commands in the [renewable-energy SQL reference](SQL.md) and the [Eurostat SQL/pandas notebook](SQL_Pandas.ipynb).
3. [Download a Eurostat dataset and build SQLite](download-to-database.md), then compare the [worldwide dataset](world-energy-download.md) and [U.S. EIA guide](eia-energy-data.md).

The tutorial explains the source filters, download paths, saved filenames, expected checks and recovery from failed downloads.
