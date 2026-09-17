# Seaborn

Seaborn builds statistical graphics on Matplotlib and works naturally with tidy pandas DataFrames. It does not replace data validation: plot units, missingness, and category definitions must be checked first.

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Synthetic interval observations for a reproducible plotting example.
df = pd.DataFrame({"wind_speed_m_s": [3, 5, 7, 9, 4, 6],
                   "power_mw": [0, 0.2, 0.6, 1.3, 0.1, 0.4],
                   "season": ["winter"] * 3 + ["summer"] * 3})
sns.scatterplot(data=df, x="wind_speed_m_s", y="power_mw", hue="season")
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Electrical power (MW)")
plt.show()
```

Use `fill=True` for filled kernel-density plots in current Seaborn releases. Correlation or a fitted trend in a chart does not establish causation.
