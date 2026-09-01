# CSV

CSV is plain text, not a self-describing scientific format. It does not preserve units, data types, missing-value conventions, time zones, or coordinate systems unless these are documented separately.

```python
df = pd.read_csv("generation.csv", parse_dates=["month"], na_values=[":", "NA"])
```

After loading, inspect shape, columns, types, duplicate keys, date range, and missingness before analysis.
