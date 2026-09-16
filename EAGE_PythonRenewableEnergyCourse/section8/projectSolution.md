---
kernelspec:
  name: python3
  display_name: Python 3
---

# Final Project: Executable Reference Solution

This is one valid implementation, not the only acceptable answer. The data are synthetic and deterministic. The model is suitable for teaching Python and energy balances, not for investment, permitting, dispatch, or detailed plant design.

**References:** [@ifc2015; @foster2010; @manwell2009; @grant2011; @hu2026].

::::{dropdown} Reveal the step-by-step project solution

## 1. Imports and parameters

```{code-cell} python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RHO_WATER = 1000.0
GRAVITY = 9.81
CP_WATER = 4180.0
STEP_HOURS = 1.0

parameters = {
    "solar": {"dc_mw": 16.0, "ac_mw": 13.0, "noct_c": 45.0,
              "gamma": -0.004, "inverter_efficiency": 0.97},
    "wind": {"rated_mw": 12.0, "cut_in": 3.0,
             "rated_speed": 12.0, "cut_out": 25.0},
    "hydro": {"rated_mw": 8.0, "net_head_m": 42.0,
              "efficiency": 0.88, "environmental_flow": 6.0},
    "geothermal": {"mass_flow": 70.0, "production_c": 160.0,
                   "reinjection_c": 70.0, "conversion_efficiency": 0.12,
                   "parasitic_fraction": 0.10},
}
```

## 2. Synthetic hourly inputs

The random generator has a fixed seed. Resource series include daily patterns and variability but are not calibrated to any location.

```{code-cell} python
rng = np.random.default_rng(2026)
time = pd.date_range("2025-06-01", periods=24 * 7, freq="h")
hour = time.hour.to_numpy()
elapsed = np.arange(len(time))

solar_shape = np.maximum(np.sin(np.pi * (hour - 5.5) / 14), 0)
cloud_factor = np.clip(0.82 + rng.normal(0, 0.12, len(time)), 0.35, 1.05)
irradiance_wm2 = 950 * solar_shape * cloud_factor
ambient_c = 19 + 8 * np.sin(2 * np.pi * (hour - 8) / 24)
wind_speed_ms = np.clip(rng.weibull(2.1, len(time)) * 8.0, 0, 32)
river_flow_m3s = np.clip(24 + 3 * np.sin(2 * np.pi * elapsed / (24 * 7))
                          + rng.normal(0, 1.2, len(time)), 0, None)
demand_mw = 23 + 5 * np.sin(2 * np.pi * (hour - 16) / 24) ** 2

portfolio = pd.DataFrame({
    "irradiance_wm2": irradiance_wm2,
    "ambient_c": ambient_c,
    "wind_speed_ms": wind_speed_ms,
    "river_flow_m3s": river_flow_m3s,
    "demand_mw": demand_mw,
}, index=time)
portfolio.head()
```

## 3. Data-quality checks

```{code-cell} python
required_resources = ["irradiance_wm2", "wind_speed_ms", "river_flow_m3s", "demand_mw"]
assert portfolio.index.is_unique
assert portfolio.index.is_monotonic_increasing
assert portfolio.isna().sum().sum() == 0
assert portfolio[required_resources].ge(0).all().all()
assert portfolio.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()

quality_summary = pd.Series({
    "rows": len(portfolio),
    "start": portfolio.index.min(),
    "end": portfolio.index.max(),
    "missing_values": int(portfolio.isna().sum().sum()),
    "duplicate_timestamps": int(portfolio.index.duplicated().sum()),
    "time_step_hours": STEP_HOURS,
})
quality_summary
```

## 4. Renewable-power functions

```{code-cell} python
def solar_power_mw(irradiance, ambient_c, dc_mw, ac_mw, noct_c=45,
                   gamma=-0.004, inverter_efficiency=0.97):
    """Temperature-corrected and inverter-clipped PV AC power in MW."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [irradiance, ambient_c, dc_mw, ac_mw, noct_c, gamma, inverter_efficiency]):
        raise ValueError("model inputs must be finite")
    irradiance = np.asarray(irradiance, dtype=float)
    if np.any(irradiance < 0) or dc_mw < 0 or ac_mw < 0:
        raise ValueError("irradiance and ratings must be non-negative")
    if not 0 < inverter_efficiency <= 1:
        raise ValueError("inverter efficiency must be in (0, 1]")
    cell_c = np.asarray(ambient_c) + irradiance / 800 * (noct_c - 20)
    factor = np.maximum(1 + gamma * (cell_c - 25), 0)
    dc_power = dc_mw * irradiance / 1000 * factor
    return np.minimum(dc_power * inverter_efficiency, ac_mw)

def wind_power_mw(speed, rated_mw, cut_in=3, rated_speed=12, cut_out=25):
    """Four-region simplified turbine/plant power curve in MW."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [speed, rated_mw, cut_in, rated_speed, cut_out]):
        raise ValueError("model inputs must be finite")
    speed = np.asarray(speed, dtype=float)
    if np.any(speed < 0) or rated_mw < 0 or not 0 <= cut_in < rated_speed < cut_out:
        raise ValueError("invalid speed or power-curve thresholds")
    power = np.zeros_like(speed)
    ramp = (speed >= cut_in) & (speed < rated_speed)
    power[ramp] = rated_mw * (speed[ramp]**3 - cut_in**3) / (rated_speed**3 - cut_in**3)
    power[(speed >= rated_speed) & (speed < cut_out)] = rated_mw
    return power

def hydro_power_mw(river_flow, environmental_flow, net_head_m,
                   efficiency, rated_mw):
    """Run-of-river electrical power in MW after environmental-flow reserve."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [river_flow, environmental_flow, net_head_m, efficiency, rated_mw]):
        raise ValueError("model inputs must be finite")
    river_flow = np.asarray(river_flow, dtype=float)
    if np.any(river_flow < 0) or environmental_flow < 0 or net_head_m < 0 or rated_mw < 0:
        raise ValueError("flows and head must be non-negative")
    if not 0 <= efficiency <= 1:
        raise ValueError("efficiency must be between 0 and 1")
    usable_flow = np.maximum(river_flow - environmental_flow, 0)
    hydraulic = efficiency * RHO_WATER * GRAVITY * usable_flow * net_head_m / 1e6
    return np.minimum(hydraulic, rated_mw)

def geothermal_power_mw(mass_flow, production_c, reinjection_c,
                        conversion_efficiency, parasitic_fraction):
    """Net MW from a simplified single-phase sensible-heat model."""
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in [mass_flow, production_c, reinjection_c, conversion_efficiency, parasitic_fraction]):
        raise ValueError("model inputs must be finite")
    mass_flow = np.asarray(mass_flow, dtype=float)
    delta_t = np.asarray(production_c, dtype=float) - reinjection_c
    if np.any(mass_flow < 0) or np.any(delta_t < 0):
        raise ValueError("mass flow and temperature difference must be non-negative")
    if not 0 <= conversion_efficiency <= 1 or not 0 <= parasitic_fraction < 1:
        raise ValueError("invalid efficiency or parasitic fraction")
    thermal_mw = mass_flow * CP_WATER * delta_t / 1e6
    return thermal_mw * conversion_efficiency * (1 - parasitic_fraction)

# Boundary and hand-calculation tests
assert solar_power_mw(0, 25, 16, 13) == 0
assert wind_power_mw([2, 12, 25], 12).tolist() == [0.0, 12.0, 0.0]
assert hydro_power_mw(6, 6, 42, 0.88, 8) == 0
expected_geo = 70 * CP_WATER * (160 - 70) / 1e6 * 0.12 * 0.90
assert np.isclose(geothermal_power_mw(70, 160, 70, 0.12, 0.10), expected_geo)
```

## 5. Calculate hourly generation

```{code-cell} python
solar = parameters["solar"]
wind = parameters["wind"]
hydro = parameters["hydro"]
geo = parameters["geothermal"]

portfolio["solar_mw"] = solar_power_mw(
    portfolio["irradiance_wm2"], portfolio["ambient_c"], **solar
)
portfolio["wind_mw"] = wind_power_mw(portfolio["wind_speed_ms"], **wind)
portfolio["hydro_mw"] = hydro_power_mw(portfolio["river_flow_m3s"], **hydro)
portfolio["geothermal_mw"] = geothermal_power_mw(**geo)

source_columns = ["solar_mw", "wind_mw", "hydro_mw", "geothermal_mw"]
portfolio["renewable_mw"] = portfolio[source_columns].sum(axis=1)
portfolio["served_mw"] = np.minimum(portfolio["renewable_mw"], portfolio["demand_mw"])
portfolio["curtailment_mw"] = np.maximum(portfolio["renewable_mw"] - portfolio["demand_mw"], 0)
portfolio["shortfall_mw"] = np.maximum(portfolio["demand_mw"] - portfolio["renewable_mw"], 0)

ratings_mw = {"solar_mw": solar["ac_mw"], "wind_mw": wind["rated_mw"],
              "hydro_mw": hydro["rated_mw"], "geothermal_mw": expected_geo}
for source, rating in ratings_mw.items():
    assert portfolio[source].between(0, rating + 1e-12).all(), f"{source} violates its power bounds"

assert np.allclose(portfolio["renewable_mw"], portfolio[source_columns].sum(axis=1)), "source sum mismatch"
assert np.allclose(portfolio["served_mw"] + portfolio["curtailment_mw"], portfolio["renewable_mw"]), "renewable balance mismatch"
assert np.allclose(portfolio["served_mw"] + portfolio["shortfall_mw"], portfolio["demand_mw"]), "demand balance mismatch"
portfolio[source_columns + ["renewable_mw", "demand_mw"]].head()
```

## 6. Energy and capacity-factor metrics

```{code-cell} python
energy_mwh = portfolio[source_columns].sum() * STEP_HOURS
capacity_factors = pd.Series({
    source.replace("_mw", ""): energy_mwh[source] / (rating * len(portfolio) * STEP_HOURS)
    for source, rating in ratings_mw.items()
})

demand_energy_mwh = portfolio["demand_mw"].sum() * STEP_HOURS
served_energy_mwh = portfolio["served_mw"].sum() * STEP_HOURS
curtailed_energy_mwh = portfolio["curtailment_mw"].sum() * STEP_HOURS
shortfall_energy_mwh = portfolio["shortfall_mw"].sum() * STEP_HOURS
renewable_coverage = served_energy_mwh / demand_energy_mwh

metrics = pd.Series({
    "renewable_available_mwh": energy_mwh.sum(),
    "demand_mwh": demand_energy_mwh,
    "renewable_served_mwh": served_energy_mwh,
    "curtailed_mwh": curtailed_energy_mwh,
    "shortfall_mwh": shortfall_energy_mwh,
    "renewable_coverage": renewable_coverage,
})
assert capacity_factors.between(0, 1 + 1e-12).all(), f"invalid capacity factors: {capacity_factors.to_dict()}"
assert 0 <= renewable_coverage <= 1, f"invalid renewable coverage: {renewable_coverage}"
print(metrics.round(3))
print("\nCapacity factors:\n", capacity_factors.round(3))
```

## 7. Resource and generation plots

```{code-cell} python
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes[0, 0].plot(portfolio.index, portfolio["irradiance_wm2"], color="goldenrod")
axes[0, 0].set(ylabel="W/m²", title="Solar irradiance")
axes[0, 1].plot(portfolio.index, portfolio["wind_speed_ms"], color="steelblue")
axes[0, 1].set(ylabel="m/s", title="Wind speed")
axes[1, 0].plot(portfolio.index, portfolio["river_flow_m3s"], color="teal")
axes[1, 0].set(ylabel="m³/s", title="River flow")
axes[1, 1].plot(portfolio.index, portfolio["demand_mw"], color="black")
axes[1, 1].set(ylabel="MW", title="Electricity demand")
for ax in axes.flat:
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
```

```{code-cell} python
fig, ax = plt.subplots(figsize=(12, 5))
ax.stackplot(portfolio.index, *[portfolio[column] for column in source_columns],
             labels=[name.replace("_mw", "").title() for name in source_columns], alpha=0.8)
ax.plot(portfolio.index, portfolio["demand_mw"], color="black", linewidth=2, label="Demand")
ax.fill_between(portfolio.index, portfolio["renewable_mw"], portfolio["demand_mw"],
                where=portfolio["demand_mw"] > portfolio["renewable_mw"],
                color="red", alpha=0.2, label="Shortfall")
ax.set(xlabel="Time", ylabel="Power (MW)", title="Hybrid renewable supply and demand")
ax.legend(ncol=3, loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
```

```{code-cell} python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
energy_mwh.rename(index=lambda value: value.replace("_mw", "").title()).plot.bar(
    ax=axes[0], color=["goldenrod", "steelblue", "teal", "firebrick"]
)
axes[0].set(ylabel="Energy (MWh)", title="Energy by source")
capacity_factors.plot.bar(ax=axes[1], color="slateblue", ylim=(0, 1))
axes[1].set(ylabel="Capacity factor", title="One-week capacity factors")
for ax in axes:
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
```

## 8. Scenario comparison

The scenario varies PV DC/AC ratio and environmental flow while holding the synthetic resources constant. Coverage improves with supply, but curtailment can also rise. This illustrates why “install more” is not a complete design criterion.

```{code-cell} python
scenario_rows = []
for dc_ac_ratio in [1.0, 1.2, 1.4]:
    for environmental_flow in [4.0, 6.0, 8.0]:
        solar_scenario = solar_power_mw(
            portfolio["irradiance_wm2"], portfolio["ambient_c"],
            dc_mw=solar["ac_mw"] * dc_ac_ratio, ac_mw=solar["ac_mw"],
            noct_c=solar["noct_c"], gamma=solar["gamma"],
            inverter_efficiency=solar["inverter_efficiency"],
        )
        hydro_scenario = hydro_power_mw(
            portfolio["river_flow_m3s"], environmental_flow,
            hydro["net_head_m"], hydro["efficiency"], hydro["rated_mw"],
        )
        total = solar_scenario + portfolio["wind_mw"] + hydro_scenario + portfolio["geothermal_mw"]
        served = np.minimum(total, portfolio["demand_mw"])
        curtailed = np.maximum(total - portfolio["demand_mw"], 0)
        scenario_rows.append({
            "dc_ac_ratio": dc_ac_ratio,
            "environmental_flow_m3s": environmental_flow,
            "coverage": served.sum() / portfolio["demand_mw"].sum(),
            "curtailment_mwh": curtailed.sum() * STEP_HOURS,
        })

scenarios = pd.DataFrame(scenario_rows)
fig, ax = plt.subplots(figsize=(8, 5))
for environmental_flow, group in scenarios.groupby("environmental_flow_m3s"):
    ax.plot(group["dc_ac_ratio"], group["coverage"], marker="o",
            label=f"Environmental flow {environmental_flow:g} m³/s")
ax.set(xlabel="PV DC/AC ratio", ylabel="Renewable coverage",
       title="Portfolio scenario comparison", ylim=(0, 1.02))
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
scenarios.round(3)
```

## Interpretation and limitations

- Solar varies diurnally, wind is variable, hydro is partly constrained by environmental flow and rating, and the simplified geothermal plant is steady.
- Capacity factor is calculated separately for each technology using its own rating; it is not conversion efficiency.
- Coverage uses renewable energy actually serving demand. Available renewable energy above demand is curtailment, not served energy.
- The week is synthetic and too short for annual resource or reliability claims.
- The PV model omits orientation, shading, detailed inverter curves, and many losses.
- The wind model requires a manufacturer curve and site-specific losses for real AEP.
- The hydro model omits reservoir routing, variable head, turbine-efficiency curves, and regulatory dispatch.
- The geothermal model assumes single-phase sensible heat and fixed reservoir conditions; real systems may require enthalpy and reservoir simulation.

An excellent student submission may use different functions or plots, but the units, physical boundaries, reconciliations, and interpretations must remain correct.

::::
