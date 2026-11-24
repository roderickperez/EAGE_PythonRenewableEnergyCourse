# Exploratory analysis of renewable electricity generation

[Open this lesson in Google Colab](https://colab.research.google.com/github/roderickperez/EAGE_PythonRenewableEnergyCourse/blob/main/EAGE_PythonRenewableEnergyCourse/section4/energy-data-analysis.ipynb)

Use Eurostat’s monthly net wind electricity generation for Austria in 2023. Each row measures generation over a calendar month in GWh. A longer month has more time to accumulate energy. Load the twelve values from the [bundled Eurostat extract](../data/examples/eurostat_at_wind_2023.csv).

## Descriptive statistics and time series

Check the monthly index and describe the distribution. Convert GWh to MWh and divide by calendar hours to estimate monthly average power.

```{code-cell} python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

asset = Path("data/examples/eurostat_at_wind_2023.csv")
csv_path = next(candidate / asset for parent in (Path.cwd(), *Path.cwd().parents)
                for candidate in (parent, parent / "EAGE_PythonRenewableEnergyCourse")
                if (candidate / asset).is_file())
observations = pd.read_csv(csv_path)
assert observations["geo"].eq("AT").all() and observations["source"].eq("RA300").all()
assert observations["unit"].eq("GWH").all() and observations["freq"].eq("M").all()
observations["month"] = pd.to_datetime(observations["month"], format="%Y-%m", utc=True)
energy = observations.set_index("month")["generation_gwh"].sort_index()
months = pd.date_range("2023-01-01", periods=12, freq="MS", tz="UTC")
assert energy.index.is_unique and energy.index.equals(months)
assert energy.notna().all() and energy.ge(0).all()
print(energy.describe().round(2))
hours = pd.Series(months.days_in_month * 24, index=months)
mean_power = (1_000 * energy / hours).rename("mean_power_mw")
assert np.isclose((mean_power * hours).sum(), 1_000 * energy.sum())
```

## Seasonal variation

Matplotlib preserves chronological order. Seaborn compares monthly averages by calendar season. The DJF group contains January, February and December within the sample year. Several years are needed to establish a recurring seasonal pattern.

```{code-cell} python
season_order = ["DJF", "MAM", "JJA", "SON"]
season_by_month = dict(zip(range(1, 13), [
    "DJF", "DJF", "MAM", "MAM", "MAM", "JJA",
    "JJA", "JJA", "SON", "SON", "SON", "DJF",
]))
data = mean_power.to_frame()
data["season"] = data.index.month.map(season_by_month)

fig, (series_ax, season_ax) = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
series_ax.plot(energy.index, energy, marker="o")
series_ax.set(ylabel="Generation (GWh)", title="Austria monthly wind generation, 2023 (UTC)")
sns.stripplot(data=data, x="season", y="mean_power_mw", order=season_order,
              jitter=False, ax=season_ax)
season_ax.set(xlabel="Calendar-season group", ylabel="Monthly average power (MW)")
plt.show()
```

## Practice

Find the season with the lowest average power weighted by hours. Set July’s generation to missing and report the share of summer hours covered and the subtotal from the remaining observations.

:::{admonition} Solution
:class: dropdown

```{code-cell} python
season = data["season"]
weighted_mw = 1_000 * energy.groupby(season).sum() / hours.groupby(season).sum()
assert weighted_mw.idxmin() == "JJA"

with_gap = energy.copy()
with_gap.loc["2023-07-01"] = np.nan
summer = season.eq("JJA")
reported = summer & with_gap.notna()
coverage = hours[reported].sum() / hours[summer].sum()
assert np.isclose(coverage, 61 / 92)
print(f"Summer coverage: {coverage:.1%}")
print(f"Reported summer subtotal: {with_gap[summer].sum(min_count=1):.2f} GWh")
```
:::

[Eurostat dataset](https://ec.europa.eu/eurostat/databrowser/view/nrg_cb_pem/default/table?lang=en) · [Longer time-series case study](timeSeriesEnergyConsumption.ipynb)
