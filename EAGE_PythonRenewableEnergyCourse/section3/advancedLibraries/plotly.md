# Plotly

Plotly creates interactive figures with hover labels, zooming, filtering, and HTML export.

```python
import plotly.express as px
import pandas as pd

monthly = pd.DataFrame({"month": ["Jan", "Feb", "Mar"] * 2,
                        "generation_gwh": [1.2, 1.5, 2.1, 3.4, 3.0, 2.8],
                        "source": ["solar"] * 3 + ["wind"] * 3})
# Synthetic monthly totals; these are not measured plant data.
fig = px.line(monthly, x="month", y="generation_gwh", color="source",
              labels={"generation_gwh": "Net generation (GWh)"})
fig.show()
```

Interactivity does not correct misleading scales or definitions. Include units, source, date scope, and accessible colour choices just as for a static chart.
