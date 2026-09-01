# Plotly

Plotly creates interactive figures with hover labels, zooming, filtering, and HTML export.

```python
import plotly.express as px

fig = px.line(monthly, x="month", y="generation_gwh", color="source",
              labels={"generation_gwh": "Net generation (GWh)"})
fig.show()
```

Interactivity does not correct misleading scales or definitions. Include units, source, date scope, and accessible colour choices just as for a static chart.
