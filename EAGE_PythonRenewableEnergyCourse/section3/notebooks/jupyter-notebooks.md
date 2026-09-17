# Jupyter Notebooks

Jupyter Notebook is an interactive document format for combining Python code, explanations, equations, data, and visualizations in one place.

A notebook is useful for:

- exploring renewable-energy datasets step by step;
- recording the code and results of an analysis;
- teaching Python with executable examples; and
- sharing reproducible data-processing workflows.

Notebook files use the `.ipynb` extension and can be opened in JupyterLab, Jupyter Notebook, or compatible editors such as VS Code.

```python
import pandas as pd

# Small synthetic dataset, so the first notebook example needs no external file.
from io import StringIO
energy = pd.read_csv(StringIO("source,energy_mwh\nsolar,12\nwind,18\nhydro,24\n"))
energy.head()
```
