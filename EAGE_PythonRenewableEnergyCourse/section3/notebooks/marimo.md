# marimo: a reactive Python notebook

**marimo** (the tool intended by “Marino”) tracks variables defined and read by cells. Running a changed cell normally reruns dependent cells; lazy mode marks them stale until requested. This differs from Jupyter's chosen execution order. [Reactivity](https://docs.marimo.io/guides/reactivity/)

Notebooks are Python `.py` files. Reactivity does not automatically track every external CSV/API change or in-place mutation. Keep explicit provenance and refresh steps; avoid defining the same global name in several cells. [Reactivity](https://docs.marimo.io/guides/reactivity/), [multiple definitions](https://docs.marimo.io/guides/understanding_errors/multiple_definitions/)

## Install and launch

In an activated local environment, run these terminal commands:

```bash
python -m pip install marimo
marimo edit energy_demo.py
```

The editor opens in your browser; Python runs in the environment hosting marimo. This application is separate from the EAGE sandbox worker. [Installation](https://docs.marimo.io/getting_started/installation/), [quickstart](https://docs.marimo.io/getting_started/quickstart/)

## Reactive energy example

Create four cells in the editor:

```text
Cell 1:
import marimo as mo

Cell 2:
power = mo.ui.slider(0, 10, value=2, label="Mean power (MW)")
power

Cell 3:
hours = 3
energy_mwh = power.value * hours

Cell 4:
mo.md(f"Energy in {hours} h: **{energy_mwh:.1f} MWh**")
```

Move the slider from 2 to 5 MW: output changes from 6 to 15 MWh. Keep widget creation and reading `.value` in separate cells for dependency tracking. [Interactive elements](https://docs.marimo.io/guides/interactivity/), [slider API](https://docs.marimo.io/api/inputs/slider/). Physics: original example $E=P\Delta t$ [@jica2011], Chapter 3.

Save; display without the editor using `marimo run energy_demo.py`. [Run as an app](https://docs.marimo.io/guides/apps/)

## Practice

- **Easy:** predict output before moving the slider.
- **Medium:** replace `hours = 3` with a second widget.
- **Challenge:** add a CSV and show its source, retrieval time and units. Explain which changes refresh automatically.
