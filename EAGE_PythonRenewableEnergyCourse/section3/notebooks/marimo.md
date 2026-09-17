# Marimo

Marimo is a reactive Python notebook designed to keep code, outputs, and dependencies synchronized. When a variable changes, Marimo updates the cells that depend on it.

Marimo notebooks are stored as Python files, which makes them convenient to version-control and review alongside other source code.

Install and run Marimo with:

```bash
pip install marimo
marimo edit
```

A minimal reactive notebook can contain ordinary Python code and interactive elements:

Run this application in your local Python environment with Marimo installed.
The EAGE sandbox includes this explanation, but its worker does not host a Marimo
web server; use its normal Python workspace for calculations instead.

```python
import marimo

app = marimo.App()

@app.cell
def explore():
    import pandas as pd
    return pd

if __name__ == "__main__":
    app.run()
```

Marimo is a useful option for interactive data applications that need more structure than a traditional notebook.
