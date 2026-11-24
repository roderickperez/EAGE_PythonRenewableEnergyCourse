# Worldwide installed wind capacity

[Open this lesson in Google Colab](https://colab.research.google.com/github/roderickperez/EAGE_PythonRenewableEnergyCourse/blob/main/EAGE_PythonRenewableEnergyCourse/section6/labs/wind-capacity.ipynb)

Installed capacity is the maximum net generating capacity reported in MW at year end. Total wind capacity includes onshore and offshore wind. Generation measures energy produced over a period and is reported in units such as MWh or TWh. Use the wind time-series exercises for energy estimates, where the duration of each power reading matters.

## Compare capacity over time

The example uses the World and regional rows from the Wind energy table in IRENA’s Renewable Capacity Statistics 2026. The sum of the rounded regional figures agrees with the published world total to within 1 MW. The snapshot stays fixed to that edition for reproducibility. See the full table for country figures.

```{code-cell} python
import pandas as pd
import matplotlib.pyplot as plt

world_mw = pd.Series(
    [467316, 514972, 563516, 623076, 733274,
     824193, 903089, 1018224, 1132690, 1291368],
    index=range(2016, 2026), name="World",
)
regional_mw = pd.Series({
    "Africa": 11469, "Asia": 723843, "Central America and Caribbean": 2267,
    "Eurasia": 17749, "Europe": 283935, "Middle East": 1939,
    "North America": 184083, "Oceania": 17686, "South America": 48396,
})
latest = world_mw.iloc[-1]
assert world_mw.index.is_unique and abs(regional_mw.sum() - latest) <= 1
print(f"World, end of 2025: {latest / 1_000:,.1f} GW")
print((100 * regional_mw / latest).round(2).rename("Share of world (%)"))

fig, (trend, regions) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
trend.plot(world_mw.index, world_mw / 1_000, marker="o")
trend.set(xlabel="Year end", ylabel="Capacity (GW)", title="World wind capacity")
(regional_mw.sort_values() / 1_000).plot.barh(ax=regions)
regions.set(xlabel="Capacity (GW)", title="IRENA regions at end of 2025")
plt.show()
```

## Practice

Calculate the net change in world capacity from 2024 to 2025 in GW and as a percentage of the 2024 total. Explain why the net change can differ from gross capacity brought online.

:::{admonition} Solution
:class: dropdown

```{code-cell} python
net_change = world_mw.loc[2025] - world_mw.loc[2024]
growth_pct = 100 * net_change / world_mw.loc[2024]
assert net_change == 158_678 and round(growth_pct, 1) == 14.0
print(f"Net change: {net_change / 1_000:.3f} GW ({growth_pct:.1f}%)")
```

Retirements, changes in reported ratings and revisions to prior-year data can affect this difference.
:::

[IRENA Renewable Capacity Statistics 2026, Wind energy tables](https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2026/Mar/IRENA_DAT_RE_capacity_statistics_2026.pdf) · [Wind workbook, including vector maps](../../section7/workbook/wind.md) · [Wind Data Lab, including time-series calculations](../../section7/renewableEnergyquizzes/eolicenergy.md)
