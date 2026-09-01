# xarray

xarray provides labelled N-dimensional arrays. A `DataArray` contains values plus named dimensions and coordinates; a `Dataset` contains several aligned variables.

```python
wind_speed = (dataset["u10"] ** 2 + dataset["v10"] ** 2) ** 0.5
monthly = wind_speed.resample(time="MS").mean()
```

Label-aware operations reduce axis mistakes, but coordinate alignment can introduce missing values. Inspect dimensions and coordinates after merges, selections, and resampling.
