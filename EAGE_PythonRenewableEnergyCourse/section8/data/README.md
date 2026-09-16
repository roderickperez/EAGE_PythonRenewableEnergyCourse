# Final-project input data

These five **synthetic** CSV datasets represent the same 168 hourly intervals,
starting 1 June 2025 at 00:00 UTC and ending with the interval starting 7 June at
23:00 UTC. Every record represents an **interval-average** input over the following
hour. The final interval ends on 8 June at 00:00 UTC.

| File | Fields after `timestamp` | Units |
|---|---|---|
| solar.csv | irradiance_wm2, ambient_c | W/m² (plane of array), °C |
| wind.csv | wind_speed_ms | m/s at turbine hub height |
| hydro.csv | river_flow_m3s | m³/s before environmental reservation |
| geothermal.csv | mass_flow_kgs, production_c, reinjection_c | kg/s, °C, °C |
| demand.csv | demand_mw | MW electricity demand |

Read **each file separately** and validate its schema, timestamps, missingness,
finite values and physical ranges before a one-to-one timestamp join. Matching
row numbers alone are not a reliable join key. UTC timestamps include an explicit
offset. Do not apply a second hub-height correction to the wind data.

Values are generated with NumPy seed 2026 by `tools/build_project_datasets.py`.
They contain no calibrated site observations, provider data or real plant results.
CSV values are rounded to eight decimal places; use numerical tolerances in checks.
The files intentionally have complete matching coverage so learners can establish
a reproducible baseline before creating their own missing-data challenges.

Models apply nonlinear functions to interval-average inputs, an approximation
that does not capture within-hour variability. No output-generation column or
worked project solution is included in these input files.
