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
    if injector == producer or not (0 <= injector < 100 and 0 <= producer < 100):
        raise ValueError("Select distinct cells between 0 and 99")
    if rate < 0 or initial <= injection or days_step <= 0:
        raise ValueError("Check flow, temperature and step")
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
    t = np.full(100, initial, dtype=float)
    elapsed = energy = revenue = 0.
    records = []
    while elapsed < years * 365 * 86400:
        dt = min(days_step * 86400, stable_seconds, 30 * 86400, years * 365 * 86400 - elapsed)
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
 nb.v4.new_markdown_cell('# Geothermal Field Designer: Python project\n\nReproduce the browser’s conservative single-layer teaching model. Default inputs match the explorer. It uses a balanced equal-conductance flow network, constant properties, and no recharge or physical-pressure solution. The results are synthetic and do not predict a real field. Grid IDs here run from 0 to 99; the interface labels them 1 to 100.'),
 nb.v4.new_code_cell(model),
 nb.v4.new_code_cell('fig, axes = plt.subplots(1, 2, figsize=(12, 4))\naxes[0].plot(result.years, result.production_C, label="Production")\naxes[0].plot(result.years, result.injection_C, label="Injection")\naxes[0].set(xlabel="Simulated years", ylabel="Temperature (°C)")\naxes[0].legend()\naxes[1].plot(result.years, result.energy_MWh)\naxes[1].set(xlabel="Simulated years", ylabel="Net electricity (MWh)")\nplt.tight_layout()'),
 nb.v4.new_code_cell('plt.figure(figsize=(6, 5))\nplt.imshow(temperature_grid, origin="lower", cmap="coolwarm", vmin=70, vmax=160)\nplt.colorbar(label="Reservoir temperature (°C)")\nplt.xlabel("Grid column")\nplt.ylabel("Grid row")\nplt.title("Final simulated reservoir temperature")\nplt.show()'),
 nb.v4.new_markdown_cell('## Investigate\n\nCompare the default with `simulate(producer=44)` and `simulate(producer=48)` at the same time. Repeat with rates of 25 and 75 kg/s. Compare net energy and production temperature. How does the constant-property, closed-boundary model limit your conclusions? Export a run using `result.to_csv("geothermal-results.csv", index=False)`. Browser CSV power is the instantaneous value after the time step; cumulative energy is integrated using the preceding value.'),
 nb.v4.new_code_cell('near, _ = simulate(producer=44)\nfar, _ = simulate(producer=48)\nplt.plot(near.years, near.production_C, label="Near")\nplt.plot(far.years, far.production_C, label="Far")\nplt.xlabel("Simulated years")\nplt.ylabel("Production temperature (°C)")\nplt.legend()\nplt.show()'),
]
notebook=nb.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
path=Path(__file__).resolve().parents[1]/'EAGE_PythonRenewableEnergyCourse/section8/geothermal-field-designer.ipynb'
nb.write(notebook,path)
