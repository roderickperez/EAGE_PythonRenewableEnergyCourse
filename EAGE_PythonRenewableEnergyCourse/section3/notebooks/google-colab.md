# Google Colab

Google Colab is a hosted notebook environment that runs in a web browser. It is useful when you want to experiment without installing Python and the scientific stack locally.

Colab supports:

- sharing notebooks through Google Drive or GitHub;
- installing additional packages with `pip`;
- connecting to cloud storage; and
- accessing optional GPU or TPU compute.

A typical setup cell is:

```python
!pip install pandas matplotlib

import pandas as pd
import matplotlib.pyplot as plt
```

Keep data paths portable when sharing a Colab notebook, and document any package installation or external files required to reproduce the analysis.
