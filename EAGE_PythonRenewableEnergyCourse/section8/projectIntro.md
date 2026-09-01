# Final Project: Auditable Renewable-Electricity Analysis

## Scenario

Prepare a technical briefing from the bundled Eurostat monthly net-electricity-generation exports. The analysis must be numerically correct, reproducible, and explicit about missing data and the partial 2024 period.

## Deliverables

Submit one executed notebook and a one-page summary containing:

1. A source statement covering the measure, unit, geography, date range, export date, and limitations.
2. A tidy dataset with one row per country, month, and source; preserve missing values.
3. Duplicate-key, missingness, coverage, and negative-value checks.
4. A renewable aggregate comprising combustible renewable, geothermal, hydro, other renewable, solar, and wind. Never add an aggregate to its components.
5. Three correctly labelled visuals: monthly trend, source composition, and a country comparison.
6. A 12-month decomposition and chronological 12-month holdout evaluated against a seasonal-naïve forecast with MAE and RMSE.
7. A SQLite table with a primary key on country, month, and source, plus a parameterised query.
8. One Section 6 engineering calculation with assumptions and clear distinctions among rated power, actual power, and energy.
9. Three evidence-backed findings, two limitations, and one recommendation.

## Required validation

- Prove that the intended grain is unique.
- Reconcile the renewable SQL total with an independent pandas calculation.
- Confirm that the test period is strictly after the training period.
- Do not convert missing observations to zero without source documentation.
- Do not compare seven months of 2024 with a complete year without a partial-period label.

## Suggested notebook structure

1. Goal and source context
2. Setup and parameters
3. Load and validate data
4. Results and figures
5. Forecast baseline
6. Database query
7. Engineering scenario
8. Conclusions and limitations

## Assessment rubric

| Criterion | Weight |
|---|---:|
| Units, taxonomy, and source interpretation | 20% |
| Reproducibility and code quality | 20% |
| Data-quality and reconciliation checks | 20% |
| Visual and statistical integrity | 15% |
| Engineering calculation | 15% |
| Communication and limitations | 10% |

The project is complete only when the notebook runs top to bottom in a fresh kernel.
