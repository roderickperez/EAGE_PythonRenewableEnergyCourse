# CSV

CSV is plain text, not a self-describing scientific format. It does not preserve units, data types, missing-value conventions, time zones, or coordinate systems unless these are documented separately.

```python
import pandas as pd
from io import StringIO

# Synthetic CSV text; replace StringIO with your file path for real input.
csv_text = "month,source,generation_mwh\n2025-01-01,solar,12\n2025-02-01,solar,:\n"
df = pd.read_csv(StringIO(csv_text), parse_dates=["month"], na_values=[":", "NA"])
print(df)
```

After loading, inspect shape, columns, types, duplicate keys, date range, and missingness before analysis.
