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

## Teaching material

The general renewable-energy, solar, hydro, wind and geothermal chapters each
contain **20 exercises (5 easy, 10 medium, 5 hard)**. Each question has a referenced,
step-by-step answer directly below it. Practice answers start collapsed in the
book. Complete chapter `.ipynb` companions are supplied alongside the MyST sources.

The question-only test is `section7/renewableEnergyTest.md`. The final project is
`section8/projectIntro.md` and loads five separate CSV files from `section8/data`.

**Instructor-only:** `instructor_private/test_answer_key.md` and
`instructor_private/final_project_solution.md` exist locally, outside the book
source and are excluded from Git. They are not included in the HTML build or
uploaded with commits. Do not move them under the book directory. Existing
technology quizzes remain optional practice with answers; they are not the test.
The earlier project solution is removed from the current student source; old Git
commits still contain the previously published version.

Regenerate practice material with `python tools/build_chapter_exercises.py`,
execute notebooks with `python tools/execute_course_notebooks.py`, and validate
with `python tools/validate_course.py`. Rebuild synthetic input files only when
needed using `python tools/build_project_datasets.py`.

After building HTML, run `python tools/check_student_build.py` before publication.
The Pages workflow runs this check before uploading the student artifact. Clean
generated output when removing a previously published page so stale downloads do
not survive. The pipeline follows the [GitHub Pages workflow guidance](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [MyST base-URL guidance](https://mystmd.org/guide/deployment-github-pages).

The Eurostat examples report a selected-category sum including broad hydro;
it is not a verified renewable-only or EU-27 total. See notebook accounting caveats.
