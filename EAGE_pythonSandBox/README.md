# EAGE renewable-energy Python sandbox

Public link: <https://roderickperez.github.io/EAGE_PythonRenewableEnergyCourse/sandbox/>

The sandbox uses the Jupyter Book as its curriculum source. It includes the full
theory, equations, references, Python examples, 100 chapter exercises and all quiz
pages. Practice answers start collapsed. The 25-question assessment and hybrid
final project contain only questions/requirements, with separate saved coding and
written-answer workspaces. Instructor answer files are never read by the builder.

## Run locally in Ubuntu / WSL

```bash
cd ~/DataScienceProjects/EAGE_PythonRenewableEnergyCourse
./start_sandbox.sh
```

Open <http://localhost:8766>. The script rebuilds the content and serves **only the
sandbox directory**. Stop with Ctrl+C. To use another port: `PORT=8767 ./start_sandbox.sh`.

For a fresh checkout, install the environment first:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./start_sandbox.sh
```

From Windows PowerShell:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'cd ~/DataScienceProjects/EAGE_PythonRenewableEnergyCourse && ./start_sandbox.sh'
```

## Participant workflow

1. Read the lesson, or search for a topic. Use the section selector for long chapters.
2. Choose **Write your solution** for an exercise, assessment question or project
   section. Choose **Edit this code** to copy an example into a separate draft.
3. Cells in a lesson share Python variables. **Include preceding examples** helps
   with setup dependencies. Deliberately invalid syntax, output transcripts and
   API templates remain visible as explanations, without a Run button.
4. Run with the button or Ctrl+Enter. Output, tables and figures appear underneath.
   **Stop** terminates Python; **Restart Python** clears its variables and files.
5. Code, written answers and input values are saved per workspace in this browser.
   **Export all my work** makes a JSON backup for submission or import later.
   **Download .py** exports the current program. This is a practice environment,
   not an invigilated exam or automatic submission/grading service.

Project data are available as `data/solar.csv`, `data/wind.csv`, `data/hydro.csv`,
`data/geothermal.csv` and `data/demand.csv` from the project workspace. Each is the
same 168-hour synthetic dataset used in the book. Uploads are available at
`/uploads/<filename>`; exported Python files can be downloaded using their path.
Restarting clears uploads and generated files, so download those before restarting.

## Runtime and limits

Python runs in a Web Worker using Pyodide **314.0.7**. Terminating the worker keeps
the interface responsive even during an infinite loop. The first run needs network
access to jsDelivr for Python and packages; optional pure-Python packages are loaded
from PyPI. Code and saved answers are not sent to a grading server. Participant
code that explicitly requests an external URL still makes that network request.

Core NumPy, pandas, Matplotlib, SciPy and SQLite examples are supported. Plotly and
Folium output use isolated HTML frames; maps and interactive plots may need their
external tile/script providers. Package versions are pinned where explicitly
installed and otherwise follow the Pyodide distribution. Jupyter magics, desktop
servers such as `marimo edit`, Jupyter widgets, Earth Engine authentication and
unsupported compiled extensions need the book's local/Colab environment. Their
explanations are included; they are not silently presented as browser capabilities.

Validation on 17 September 2026: 133 browser runtime checks passed with zero
failures and 67 figures, including all 100 independent chapter solutions, all quiz
chapters and both longer data notebooks. GeoPandas and Cartopy examples also passed
with the pinned distribution. The local validator executed 333 basic/quiz cells.
Draft restoration and the Stop control were checked in the interface.

The source book is English. The obsolete independent English/Spanish lesson packs
have been removed to prevent their old examples from diverging from this course.
The original standalone sandbox identified Gabriel Westman as its author; this
course integration replaces that implementation while retaining the EAGE branding.

## Maintain and deploy

Edit the book's Markdown/notebook sources, then run:

```bash
.venv/bin/python tools/build_sandbox.py
.venv/bin/python tools/validate_sandbox.py
.venv/bin/python tools/validate_course.py
```

`generated/` is ignored build output, including allowlisted public datasets, images,
notebooks and supplied reference PDFs. The builder reads the public `myst.yml` TOC,
preserves source hashes and copies only participant assets. Never add an instructor
answer document to that TOC. Public practice answers are intentionally downloadable;
collapsing them is a learning aid, not access control.

The existing **deploy-book** GitHub Actions workflow builds both the book and
`/sandbox/` into one Pages artifact. This avoids competing deployments overwriting
the book. It checks curriculum coverage and private-answer exclusion before upload.
The repository's GitHub Pages source is GitHub Actions. A push to `main` or a manual
workflow dispatch rebuilds the public site.

For the actual-browser runtime suite, copy `tools/sandbox_runtime_check.html` to
`EAGE_pythonSandBox/runtime-check.html`, launch the sandbox, and open
`http://localhost:8766/runtime-check.html`. That developer page is excluded from
deployment. It checks the 100 independent public solutions, quiz and basic chapter
code, plotting and project-data access without loading private answers.

Runtime documentation: [Pyodide workers](https://pyodide.org/en/stable/usage/webworker.html),
[package loading](https://pyodide.org/en/stable/usage/loading-packages.html), and
[GitHub Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
