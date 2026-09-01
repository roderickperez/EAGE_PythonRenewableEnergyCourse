# Course Summary

This summary follows the actual course structure and distinguishes core outcomes from optional extensions.

## Section 0 — Renewable-energy context

Energy, power, units, capacity factor, resource variability, and the role of auditable data analysis.

## Section 1 — Python environments and notebooks

Python 3, development environments, Jupyter/Colab, Markdown, and reproducible computational documents.

## Section 2 — Python basics

Variables, numeric operations, functions, lists, tuples, dictionaries, sets, conditions, `for` loops, and selected built-in functions. Function naming is a convention rather than a syntax restriction, and functions without `return` yield `None`.

## Section 3 — Scientific Python libraries

- NumPy arrays, vectorised arithmetic, shape, indexing, and sampling.
- pandas DataFrames, selection, missing values, summaries, and transformations.
- Matplotlib and Seaborn for labelled scientific plots.
- Optional overviews of interactive and geospatial libraries.

## Section 4 — Data quality and time series

Source definitions, units, tidy data, missingness, duplicate keys, coverage, exploratory analysis, monthly decomposition with period 12, stationarity testing, and chronological forecast evaluation. The Eurostat case study is monthly net electricity generation in GWh, not consumption in TJ.

## Section 5 — Databases and SQL

Relational design, primary and foreign keys, SQLite, SQL filtering and aggregation, a country–month–source generation table, and reconciliation of SQL totals against pandas.

## Section 6 — Renewable-energy calculations

| Technology | Principal relationship | Important interpretation |
|---|---|---|
| Hydroelectric | $P=\eta\rho gQH$ | Annual energy uses net rated capacity and capacity factor |
| Solar PV | $P\approx P_{STC}(G/G_{STC})[1+\alpha(T_c-T_{STC})]$ | Irradiance and temperature both affect DC output |
| Wind | $P=\tfrac12\rho A v^3 C_P$ | Expected power requires a speed distribution and turbine power curve |
| Geothermal | $q=-k\,dT/dz$ and $P_{th}=\dot m c_p\Delta T$ | Thermal and electrical power must be kept distinct |

These are simplified engineering models. Site assessment requires technology-specific loss models, uncertainty, environmental constraints, and current project data.

## Section 7 — Quizzes and exercises

Exercises reinforce Python, NumPy, pandas, plotting, and renewable-energy calculations. Learners should run every solution and inspect units and edge cases rather than accepting output solely because code executes.

## Section 8 — Final project

The final project integrates source interpretation, data quality, visualisation, monthly forecasting, SQLite, reconciliation checks, an engineering scenario, and technical communication.

## Core completion checklist

- [x] Explain the difference between power and energy.
- [x] Validate units, grain, date coverage, and missing values before analysis.
- [x] Manipulate arrays and DataFrames without hidden shape assumptions.
- [x] Produce charts with accurate titles, axes, units, and scope.
- [x] Avoid double-counting aggregate and component energy categories.
- [x] Use a chronological holdout and a simple forecast baseline.
- [x] Create and query a relational SQLite database.
- [x] Apply the main hydro, solar, wind, and geothermal equations with stated assumptions.
- [x] Communicate data freshness and limitations.

Deep learning, operational forecasting, power-system optimisation, satellite classification, and bankable resource assessment are optional next steps, not claimed core outcomes of this short course.
