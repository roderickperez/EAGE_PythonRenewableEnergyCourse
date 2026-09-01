# Python for Renewable Energy Data Processing

This repository contains the EAGE short course and its source data.

## Reproducible setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/rebuild_course_notebooks.py
python tools/validate_course.py
jupyter book build EAGE_PythonRenewableEnergyCourse --html
```

The Eurostat notebooks use the bundled August 2024 exports for reproducibility. Refresh the exports and record the retrieval date before using their figures as current statistics.

Optional geospatial and deep-learning packages are declared separately in `pyproject.toml`; they are not required to compile the core course.
