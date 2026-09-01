# Final Project: Solution Guide

This is an instructor guide rather than one prescribed answer. The executed notebooks in [Section 4](../section4/timeSeriesEnergyConsumption.ipynb) and [Section 5](../section5/SQL_Pandas.ipynb) are the reference implementations for loading, taxonomy, missing values, time-series validation, SQLite creation, and reconciliation.

## Expected interpretation

- Measure: monthly net electricity generation in GWh.
- Period: January 2016 through July 2024; the final year is partial.
- Renewable set: combustible renewable, geothermal, hydro, other renewable, solar, and wind.
- Fossil total: the non-renewable combustible-fuels aggregate. Do not add it to coal, gas, and oil.
- Regional totals: sum available reporting countries and show coverage.

## Minimum checks

```python
key = ["country", "month", "source"]
assert not generation.duplicated(key).any()
assert generation["month"].max() == pd.Timestamp("2024-07-01")
assert train.index.max() < test.index.min()
```

Investigate negative corrections and retain missing observations as missing.

## Forecast benchmark

For monthly data, use the seasonal-naïve baseline $\hat{y}_t=y_{t-12}$ and evaluate the same future holdout with:

$$MAE=\frac{1}{n}\sum_{t=1}^{n}|y_t-\hat{y}_t|$$

$$RMSE=\sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t-\hat{y}_t)^2}$$

A more complex model is useful only when it materially improves on this baseline without using future information.

## Example engineering calculation

For a hydro plant with 500 MW net rated capacity and a 0.45 capacity factor:

$$E=500\ \text{MW}\times8760\ \text{h}\times0.45
=1{,}971{,}000\ \text{MWh}=1.971\ \text{TWh}$$

This uses rated capacity. Applying capacity factor to an already averaged actual power would count availability twice.

## Evaluation notes

Deduct substantially for labelling generation as consumption, using TJ for GWh data, double-counting aggregate and component sources, replacing unknown values with zero, random train/test shuffling, comparing partial and complete years, missing chart units, or conclusions unsupported by visible evidence.

Accept alternative implementations when their grain, assumptions, formulas, and validation checks are explicit and correct.
