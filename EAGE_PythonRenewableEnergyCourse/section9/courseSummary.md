# Course Summary

This summary follows the actual course structure and distinguishes core outcomes from optional extensions.

## Section 0 — Renewable-energy context

Energy, power, units, capacity factor, resource variability, and the role of auditable data analysis.

## Data quality and time series

Source definitions, units, tidy data, missingness, duplicate keys, coverage, exploratory analysis, monthly decomposition with period 12, stationarity testing, and chronological forecast evaluation. The Eurostat case study is monthly net electricity generation in GWh, not consumption in TJ.

## Databases and SQL

Relational design, primary and foreign keys, SQLite, SQL filtering and aggregation, a live Eurostat download with raw-data provenance, a country–month–source generation table, and reconciliation of SQL totals against pandas.

## Renewable-energy calculations

| Technology | Principal relationship | Important interpretation |
|---|---|---|
| Hydroelectric | $P_e=\eta\rho gQH_{net}$ | Use net head; $E=P_r CF T$ at a consistent electrical boundary [@jica2011], Chapter 3 |
| Solar PV | $P\approx P_{STC}(G/G_{STC})[1+\alpha(T_c-T_{STC})]$ | Irradiance and cell temperature both affect DC output; here $\alpha$ is the fractional power-temperature coefficient [@pvlibDocs] |
| Wind | $P_{rotor}=\tfrac12\rho A v^3 C_P$ | Shaft power precedes drivetrain/generator losses; electrical energy uses the turbine curve [@manwell2009], §§2.5 and 3.2 |
| Geothermal | $q=-k\,dT/dz$ and $P_{th}=\dot m c_p\Delta T$ | Thermal and electrical power must be kept distinct [@grant2011], Chapters 2–3 |

These are simplified engineering models. Site assessment requires technology-specific loss models, uncertainty, environmental constraints, and current project data.

## Renewable-energy quizzes and exercises

Exercises apply prerequisite Python, NumPy, pandas, and plotting knowledge to renewable-energy calculations. Learners should run every solution and inspect units and edge cases rather than accepting output solely because code executes. General Python and library quizzes remain in the separate Python Course.

## Final project

The final project models one synthetic week of a hybrid solar, wind, hydro and geothermal portfolio, reconciles supply and demand, and compares design scenarios. Monthly forecasting and SQLite are separate case studies in Sections 4 and 5.

## Core completion checklist

- [ ] Explain the difference between power and energy.
- [ ] Validate units, grain, date coverage, and missing values before analysis.
- [ ] Apply prerequisite array and DataFrame skills without hidden shape assumptions.
- [ ] Produce renewable-energy charts with accurate titles, axes, units, and scope.
- [ ] Avoid double-counting aggregate and component energy categories.
- [ ] Use a chronological holdout and a simple forecast baseline.
- [ ] Create and query a relational SQLite database.
- [ ] Apply the main hydro, solar, wind, and geothermal equations with stated assumptions.
- [ ] Communicate data freshness and limitations.

Deep learning, operational forecasting, power-system optimisation, satellite classification, and bankable resource assessment are optional next steps, not claimed core outcomes of this short course.
