# Python for Renewable Energy Data Processing

This repository contains the EAGE short course and its source data.

## Reproducible setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/rebuild_course_notebooks.py
python tools/execute_course_notebooks.py
python tools/validate_course.py
cd EAGE_PythonRenewableEnergyCourse
jupyter book build --html
```

The Eurostat notebooks use the bundled August 2024 exports for reproducibility. Refresh the exports and record the retrieval date before using their figures as current statistics.

Optional geospatial and deep-learning packages are declared separately in `pyproject.toml`; they are not required to compile the core course.

## Updated teaching material

Start with `EAGE_PythonRenewableEnergyCourse/learningGuide.md`. The reference catalogue lists all supplied PDFs, and `section7/renewableExercises.md` contains 5 easy, 10 medium and 5 hard exercises with initially collapsed, step-by-step solutions and individual references.

`python tools/validate_course.py` executes core Markdown cells and tests every new solution independently. Interactive input snippets and optional tools are identified separately. The workbook can be regenerated with `python tools/build_exercise_workbook.py`.

The Eurostat examples report a selected-category sum including broad hydro; it is not a verified renewable-only or EU-27 total. See notebook accounting caveats.
