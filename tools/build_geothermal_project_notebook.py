"""Generate the standalone geothermal field designer teaching notebook."""
from pathlib import Path
import nbformat as nb

model = '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N, DX, H, CP = 10, 200.0, 50.0, 4180.0
def neighbors(i):
    x, y = i % N, i // N
    return ([i - 1] if x else []) + ([i + 1] if x < N - 1 else []) + ([i - N] if y else []) + ([i + N] if y < N - 1 else [])

def simulate(injector=43, producer=46, rate=50., injection=70., initial=160.,
             heat_capacity=2_500_000., efficiency=.12, auxiliary=.10,
             capacity=5., availability=.90, price=75., years=20, days_step=30):
    if (not isinstance(injector, (int, np.integer)) or isinstance(injector, (bool, np.bool_))
            or not isinstance(producer, (int, np.integer)) or isinstance(producer, (bool, np.bool_))
            or injector == producer or not (0 <= injector < 100 and 0 <= producer < 100)):
        raise ValueError("Select distinct integer cells between 0 and 99")
    values = [rate, injection, initial, heat_capacity, efficiency, auxiliary,
              capacity, availability, price, years, days_step]
    if not all(np.isscalar(value) and np.isfinite(value) for value in values):
        raise ValueError("Model inputs must be finite scalars")
    if (rate < 0 or initial <= injection or injection <= -273.15 or heat_capacity <= 0
            or not 0 <= efficiency <= 1 or not 0 <= auxiliary < 1
            or capacity < 0 or not 0 <= availability <= 1 or years <= 0 or days_step <= 0):
        raise ValueError("Check physical ranges, model duration and time step")
    duration_seconds = years * 365 * 86400
    if not np.isfinite(duration_seconds) or min(days_step, 30) * 86400 < duration_seconds / 10000:
        raise ValueError("This teaching model permits at most 10,000 time steps; shorten the duration or increase days_step")
    adjacent = [neighbors(i) for i in range(100)]
    p, b = np.zeros(100), np.zeros(100)
    b[injector], b[producer] = rate, -rate
    for _ in range(12000):
        change = 0.
        for i in range(1, 100):
            value = (b[i] + sum(p[j] for j in adjacent[i])) / len(adjacent[i])
            change = max(change, abs(value - p[i]))
            p[i] = value
        if change < max(1., rate) * 1e-12:
            break
    edges, outgoing = [], np.zeros(100)
    for i in range(100):
        for j in adjacent[i]:
            if j > i:
                q = p[i] - p[j]
                source, target = (i, j) if q >= 0 else (j, i)
                edges.append((source, target, abs(q)))
                outgoing[source] += abs(q)
    outgoing[producer] += rate
    cell_capacity = DX * DX * H * heat_capacity
    stable_seconds = .8 * cell_capacity / (outgoing.max() * CP) if outgoing.max() else float("inf")
    if not np.isfinite(cell_capacity) or cell_capacity <= 0 or stable_seconds < duration_seconds / 10000:
        raise ValueError("The stable time step would require more than 10,000 steps; review flow and heat capacity")
    t = np.full(100, initial, dtype=float)
    elapsed = energy = revenue = 0.
    records = []
    while elapsed < duration_seconds:
        if len(records) >= 10000:
            raise ValueError("This teaching model permits at most 10,000 time steps")
        dt = min(days_step * 86400, stable_seconds, 30 * 86400, duration_seconds - elapsed)
        if elapsed + dt <= elapsed:
            raise ValueError("The time step is too small to advance the simulation clock")
        old, delta = t.copy(), np.zeros(100)
        for source, target, q in edges:
            heat = q * CP * old[source] * dt * availability
            delta[source] -= heat
            delta[target] += heat
        delta[injector] += rate * CP * injection * dt * availability
        delta[producer] -= rate * CP * old[producer] * dt * availability
        net = min(capacity, max(0., rate * CP * (old[producer] - injection) / 1e6 * efficiency * (1 - auxiliary)))
        energy += net * dt / 3600 * availability
        revenue += net * dt / 3600 * availability * price
        t += delta / cell_capacity
        elapsed += dt
        records.append(dict(years=elapsed / (365 * 86400), production_C=t[producer],
                            injection_C=injection, energy_MWh=energy, revenue_USD=revenue))
    return pd.DataFrame(records), t.reshape(N, N)

result, temperature_grid = simulate()
result.tail()
'''
cells = [
 nb.v4.new_markdown_cell('# Geothermal Field Designer: Python project\n\nExplore the browser’s conservative single-layer teaching model. Default inputs match the single-pair explorer, but numerical time steps may differ, so cumulative results need not match exactly. It uses a balanced equal-conductance flow network, constant properties, and no recharge or physical-pressure solution. The results are synthetic and do not predict a real field. Grid IDs here run from 0 to 99; the interface labels them 1 to 100.\n\nThe conversion efficiency is a fixed teaching input, not a plant feasibility calculation. The allowed range 0–1 enforces an energy-accounting boundary; it does not imply that every allowed value, especially 100%, is physically achievable. The model does not specify a heat-rejection temperature or calculate a thermodynamic limit. For a heat engine between two fixed-temperature reservoirs, the [Carnot bound](https://openstax.org/books/college-physics/pages/15-4-carnots-perfect-heat-engine-the-second-law-of-thermodynamics-restated) is 1 minus the cold-to-hot absolute-temperature ratio; real conversion is lower. Production temperature changes here, and injection temperature is not automatically the heat-rejection temperature, so this bound cannot be applied directly without a suitable plant model.'),
 nb.v4.new_code_cell(model),
 nb.v4.new_code_cell('fig, axes = plt.subplots(1, 2, figsize=(12, 4))\naxes[0].plot(result.years, result.production_C, label="Production")\naxes[0].plot(result.years, result.injection_C, label="Injection")\naxes[0].set(xlabel="Simulated years", ylabel="Temperature (°C)")\naxes[0].legend()\naxes[1].plot(result.years, result.energy_MWh)\naxes[1].set(xlabel="Simulated years", ylabel="Net electricity (MWh)")\nplt.tight_layout()'),
 nb.v4.new_code_cell('plt.figure(figsize=(6, 5))\nplt.imshow(temperature_grid, origin="lower", cmap="coolwarm", vmin=70, vmax=160)\nplt.colorbar(label="Reservoir temperature (°C)")\nplt.xlabel("Grid column")\nplt.ylabel("Grid row")\nplt.title("Final simulated reservoir temperature")\nplt.show()'),
 nb.v4.new_markdown_cell('## Investigate\n\nCompare the default with `simulate(producer=44)` and `simulate(producer=48)` at the same time. Repeat with rates of 25 and 75 kg/s. Compare net energy and production temperature. How does the constant-property, closed-boundary model limit your conclusions? Export a run using `result.to_csv("geothermal-results.csv", index=False)`. Browser CSV power is the instantaneous value after the time step; cumulative energy is integrated using the preceding value.'),
 nb.v4.new_markdown_cell('## Compare a single-pair browser export\n\nDownload the CSV from the first, single-pair explorer and place it beside this notebook. The following function recomputes each recorded instantaneous net power and cumulative revenue from the exported settings. A setting change resets this explorer’s history, so the price remains constant within its export. It does not replay unsaved internal integration steps or the separate multi-depth explorer. Call `compare_single_pair_csv("geothermal-field-results.csv")` using the name of your downloaded browser file. The two maximum differences should be close to zero, allowing for numerical rounding.'),
 nb.v4.new_code_cell('''def compare_single_pair_csv(path):
    data = pd.read_csv(path)
    columns = ["configured_depth_count", "total_flow_kg_s", "production_C", "injection_C",
               "conversion_efficiency", "auxiliary_fraction", "capacity_MW", "price_USD_per_MWh",
               "net_power_MW", "energy_MWh", "revenue_USD"]
    if data.empty or not set(columns).issubset(data.columns):
        raise ValueError("Choose a nonempty CSV from the single-pair explorer")
    numeric = data[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all() or not numeric.configured_depth_count.eq(1).all():
        raise ValueError("Expected finite values for one configured depth")
    if ((numeric.total_flow_kg_s < 0).any() or (numeric.capacity_MW < 0).any()
            or (numeric.energy_MWh < 0).any() or (numeric.price_USD_per_MWh < 0).any()
            or (numeric.production_C <= -273.15).any() or (numeric.injection_C <= -273.15).any()
            or not numeric.conversion_efficiency.between(0, 1).all()
            or not numeric.auxiliary_fraction.between(0, 1).all()):
        raise ValueError("Recorded values are outside the model's physical ranges")
    if numeric.price_USD_per_MWh.nunique() != 1:
        raise ValueError("Single-pair export should have one price; check the source file")
    thermal = (numeric.total_flow_kg_s * 4180 * (numeric.production_C - numeric.injection_C) / 1e6).clip(lower=0)
    expected_power = np.minimum(numeric.capacity_MW, thermal * numeric.conversion_efficiency * (1 - numeric.auxiliary_fraction))
    expected_revenue = numeric.energy_MWh * numeric.price_USD_per_MWh
    return pd.Series({"max_power_difference_MW": (expected_power - numeric.net_power_MW).abs().max(),
                      "max_revenue_difference_USD": (expected_revenue - numeric.revenue_USD).abs().max()})
'''),
 nb.v4.new_markdown_cell('## Compare a multi-depth browser export\n\nThe second explorer saves each producer’s flow and temperature as JSON in its CSV. Call `compare_multidepth_csv("geothermal-multifield-results.csv")` after downloading that CSV beside this notebook. The function checks separately clipped useful heat, the common plant cap and project balance for every recorded event. Same-time events remain in their recorded order. It checks sampled values only: it does not replay the multi-depth reservoir or the internal integration of energy and revenue.'),
 nb.v4.new_code_cell('''def compare_multidepth_csv(path):
    import json
    data = pd.read_csv(path)
    columns = ["producer_streams_json", "injection_C", "thermal_MW", "conversion_efficiency",
               "auxiliary_fraction", "capacity_MW", "net_power_MW", "revenue_USD",
               "cumulative_investment_USD", "project_balance_USD"]
    if data.empty or not set(columns).issubset(data.columns):
        raise ValueError("Choose a nonempty CSV from the multi-depth explorer")
    numeric = data[columns[1:]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Expected finite recorded operating and financial values")
    if ((numeric.capacity_MW < 0).any() or (numeric.injection_C <= -273.15).any()
            or not numeric.conversion_efficiency.between(0, 1).all()
            or not numeric.auxiliary_fraction.between(0, 1).all()):
        raise ValueError("Recorded values are outside the model's physical ranges")
    thermal = []
    for streams_json, injection in zip(data.producer_streams_json, numeric.injection_C):
        streams = json.loads(streams_json)
        if not isinstance(streams, list):
            raise ValueError("Expected a list of producer streams")
        contributions = []
        for stream in streams:
            flow, temperature = float(stream["weight"]), float(stream["temperature"])
            if not np.isfinite([flow, temperature]).all() or flow < 0 or temperature <= -273.15:
                raise ValueError("Producer flow and temperature must be finite, with nonnegative flow and temperature above absolute zero")
            contributions.append(max(0., flow * 4180 * (temperature - injection) / 1e6))
        thermal.append(sum(contributions))
    thermal = np.asarray(thermal)
    power = np.minimum(numeric.capacity_MW, thermal * numeric.conversion_efficiency * (1 - numeric.auxiliary_fraction))
    balance = numeric.revenue_USD - numeric.cumulative_investment_USD
    return pd.Series({"max_thermal_difference_MW": np.max(np.abs(thermal - numeric.thermal_MW)),
                      "max_power_difference_MW": np.max(np.abs(power - numeric.net_power_MW)),
                      "max_balance_difference_USD": np.max(np.abs(balance - numeric.project_balance_USD))})
'''),
 nb.v4.new_markdown_cell('## Check time-step sensitivity\n\nA stable explicit step is not necessarily time-step converged. Compare `simulate(years=2, days_step=30)` with smaller steps such as `days_step=5` and `days_step=1`, and compare the final production temperature and cumulative MWh. This teaching implementation rejects settings requiring more than 10,000 steps; shorten the duration or revise the inputs if it reports this limit. Use matching inputs and simulated times. Differences are numerical sensitivity, not proof that either run predicts a real reservoir; sampled browser rows alone cannot reproduce the internal integration exactly.'),
 nb.v4.new_code_cell('near, _ = simulate(producer=44)\nfar, _ = simulate(producer=48)\nplt.plot(near.years, near.production_C, label="Near")\nplt.plot(far.years, far.production_C, label="Far")\nplt.xlabel("Simulated years")\nplt.ylabel("Production temperature (°C)")\nplt.legend()\nplt.show()'),
]
notebook=nb.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
path=Path(__file__).resolve().parents[1]/'EAGE_PythonRenewableEnergyCourse/section8/geothermal-field-designer.ipynb'
nb.write(notebook,path)
