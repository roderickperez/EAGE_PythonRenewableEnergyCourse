# Jupyter Notebook and JupyterLab

**Jupyter Notebook** is an authoring application; `.ipynb` is the JSON document format for cells, metadata and saved outputs. **JupyterLab** adds a workspace for notebooks, editors and terminals. Neither requires Anaconda. [Notebook introduction](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html), [format](https://nbformat.readthedocs.io/en/latest/format_description.html), [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/user/interface.html)

## First reproducible notebook

1. Follow the [installation guide](../../section1/IDE.md); launch JupyterLab.
2. Create a Python notebook called `energy_first_steps.ipynb`.
3. Add a Markdown cell: “Three synthetic interval-average powers; each lasts one hour.”
4. Run this code cell with Shift+Enter:

```{code-cell} python
powers_mw = [2.0, 3.0, 1.0]
duration_hours = 1.0
energy_mwh = sum(powers_mw) * duration_hours
print(f"Energy: {energy_mwh:.1f} MWh")
assert energy_mwh == 6.0
```

This original example uses $E=\sum_i\bar P_i\Delta t_i$; MW × h gives MWh. Samples must represent interval-average power. [@jica2011], Chapter 3; [Python sum](https://docs.python.org/3/library/functions.html#sum).

5. Add an interpretation, then save.
6. Restart the kernel and run all cells. Confirm 6.0 MWh appears without old variables.
7. Share the notebook with separate inputs. A saved output records an earlier run, not proof that current code reproduces it. [Running code](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Running%20Code.html)

## Hidden state

Classic Jupyter permits cells to run in any chosen order. If A sets `power_mw = 2`, B computes `energy_mwh = power_mw * 3`, and you rerun only A with 5, B's result remains stale. Execution counters record execution order; page position does not enforce it. [Running code](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Running%20Code.html)

Use `Path.cwd()` to inspect the working folder. In IPython, `%pip install package_name` targets the active kernel environment; this is notebook syntax, not ordinary script syntax. [pathlib](https://docs.python.org/3/library/pathlib.html), [IPython %pip](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pip)

## Practice

- **Easy:** change powers and check the result by hand.
- **Medium:** use durations `[0.5, 1, 2]` hours and explain time weighting.
- **Challenge:** demonstrate stale output, then restore a notebook that runs from a fresh kernel.

Next: [download a real dataset](../../section5/download-to-database.md).
