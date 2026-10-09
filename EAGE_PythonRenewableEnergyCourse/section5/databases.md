# Databases for renewable-energy data

A **database** is an organized collection of data; a **database management system (DBMS)** manages storage, retrieval and updates. CSV is a file format, SQL is a language, and SQLite/PostgreSQL are database systems. [PostgreSQL concepts](https://www.postgresql.org/docs/current/tutorial-concepts.html), [SQLite](https://www.sqlite.org/about.html), [Python CSV](https://docs.python.org/3/library/csv.html)

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

Enable SQLite foreign-key checking on every connection using `PRAGMA foreign_keys = ON`. [SQLite foreign keys](https://www.sqlite.org/foreignkeys.html)

## Missing data and units

SQL `NULL` means missing/unknown, not zero. `SUM` and `AVG` omit null inputs; `COUNT(*)` counts rows while `COUNT(column)` counts non-null values. Report coverage with a total. Keys prevent duplicate records, but cannot establish physical units or prevent double-counting aggregate categories. [SQLite aggregates](https://www.sqlite.org/lang_aggfunc.html)

Historical course exports contain monthly **net electricity generation in GWh**, not consumption. Broad hydro may include pumped-storage output; a selected-category sum is not automatically an official renewable-only or EU-27 total. Keep geography, product, units, flags and period. [Eurostat energy metadata](https://ec.europa.eu/eurostat/web/energy/methodology)

## Other models

Document databases can store nested records and still use schema validation/indexes. Time-series sensor data are often structured and can be stored relationally. “Energy data” does not imply “unstructured data.” [MongoDB modelling](https://www.mongodb.com/docs/manual/data-modeling/)

Hierarchical and network DBMSs are historical navigational data models; object-oriented databases persist objects. They are different from the cloud/operational deployment labels. We use SQLite because it is embedded and requires no separate database server. [IBM database models](https://www.ibm.com/think/topics/database), [SQLite serverless architecture](https://www.sqlite.org/serverless.html)

## Practical route

1. [Download a dataset, validate it and build SQLite](download-to-database.md).
2. Study the [SQL command reference](SQL.md).
3. Continue to the [historical Eurostat SQL/pandas notebook](SQL_Pandas.ipynb).

The tutorial explains the source filters, download paths, saved filenames, expected checks and recovery from failed downloads.
