# Seaborn

Seaborn builds statistical graphics on Matplotlib and works naturally with tidy pandas DataFrames. It does not replace data validation: plot units, missingness, and category definitions must be checked first.

```python
import seaborn as sns

sns.scatterplot(data=df, x="wind_speed_m_s", y="power_mw", hue="season")
```

Use `fill=True` for filled kernel-density plots in current Seaborn releases. Correlation or a fitted trend in a chart does not establish causation.
