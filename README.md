# Python for Renewable Energy Data Processing

This repository contains the EAGE short course and its source data.

## Where to find the course files

The main course folder is `EAGE_PythonRenewableEnergyCourse/` inside this repository. Its `section0/` and `section4/` through `section9/` folders contain the lesson text (`.md`) and downloadable Python notebooks (`.ipynb`). The chapter exercises are in `section7/workbook/`, quizzes in `section7/renewableEnergyquizzes/`, and final-project instructions and datasets in `section8/` and `section8/data/`. The course overview is `courseIntro.md`.

The original assessment and its correction evidence are in the repository root: `COURSE_ASSESSMENT_2026-11-18.md`, `COURSE_FINDING_EVIDENCE_INDEX.md`, and `COURSE_ADDITIONAL_FINDING_EVIDENCE.md`. These records describe local changes and their verification.

## Reproducible setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm ci
npm test
npm run build
python tools/rebuild_course_notebooks.py
python tools/execute_course_notebooks.py
python tools/validate_course.py
cd EAGE_PythonRenewableEnergyCourse
jupyter book build --html
cd ..
python tools/stage_course_web_assets.py
python tools/check_student_build.py
```

If running Windows Node from PowerShell while the repository is on a WSL `\\wsl.localhost\...` path, the default npm command shell cannot use that path as its working directory. With PowerShell 7 (`pwsh`) installed, select it for npm scripts in that terminal before running `npm test` or `npm run build`:

```powershell
$env:npm_config_script_shell = (Get-Command pwsh).Source
npm test
```

Alternatively, use Node and npm installed inside WSL and run the commands from a WSL shell.

The Eurostat notebooks use the bundled August 2024 exports for reproducibility. Refresh the exports and record the retrieval date before using their figures as current statistics.

Additional geospatial and deep-learning packages are declared separately in `pyproject.toml`; they are not required to compile the core course.

The course book and sandbox follow the eight-module EAGE outline, starting with
renewable-energy fundamentals and an included Python introduction. The core setup
includes GeoPandas and Folium for the mapping labs. Refer to the course overview
for lesson and exercise links. Each module has an Open in Google Colab link in its introductory lesson or lab. Regenerate the seven new lesson notebooks with python tools/build_module_notebooks.py after editing the lesson sources or their bundled datasets.

The hydroelectric lesson includes a locally bundled Three.js teaching lab. Edit
its source under `web/hydro-lab/`, run `npm test` and `npm run build`, and commit
the generated files under `EAGE_PythonRenewableEnergyCourse/static/hydro-lab/`.
The lab has no runtime CDN or downloaded-model dependency.

## Teaching material

The general renewable-energy, solar, wind and geothermal workbooks each contain
**20 exercises (4 basic, 8 medium, 8 hard)**. The hydropower workbook has **22
exercises (4 basic, 10 medium, 8 hard)**, including two additional worked examples.
Each question has a referenced, step-by-step answer directly below it. Practice answers start collapsed in the
book. Complete chapter `.ipynb` companions are supplied alongside the MyST sources.

Before the exercises, every energy chapter includes detailed physical descriptions,
component and definition tables, equations with symbols and units, assumptions,
operating limits, a worked numerical example and links to the supplied and additional
textbook reading. The generator preserves these theory sections and carries their
full text into the notebook downloads. `tools/validate_theory.py` checks preservation
and the worked-example arithmetic.

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

To check standalone notebook downloads when a host labels `.ipynb` files as JSON,
start `python assessment_support/serve_book_json.py` in one terminal after the
HTML build and staging step, then run `node assessment_support/test_book_notebook_downloads.cjs`
in another. The browser check uses Microsoft Edge by default; set `EAGE_TEST_EDGE`
to another Chromium-based browser executable if needed. Stop the local test server
when finished.

After building HTML, run `python tools/check_student_build.py` before publication.
The Pages workflow runs this check before uploading the student artifact. Clean
generated output when removing a previously published page so stale downloads do
not survive. The pipeline follows the [GitHub Pages workflow guidance](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [MyST base-URL guidance](https://mystmd.org/guide/deployment-github-pages).

The Eurostat examples report a selected-category sum including broad hydro;
it is not a verified renewable-only or EU-27 total. See notebook accounting caveats.

## Browser Python sandbox

Open the [public participant sandbox](https://roderickperez.github.io/EAGE_PythonRenewableEnergyCourse/sandbox/)
or run `./start_sandbox.sh` from this repository and visit `http://localhost:8766`.
It imports the complete book curriculum, including the 102 chapter exercises and
quiz lessons. Assessment and final-project answers remain private; participants
have their own saved coding and written-answer workspaces. See
[sandbox setup and operation](EAGE_pythonSandBox/README.md).
