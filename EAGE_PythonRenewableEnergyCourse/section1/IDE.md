# Editors, IDEs and Python environments

An **editor** edits source files; an **integrated development environment (IDE)** also integrates development tools such as debugging and tests. Editors can acquire IDE features through extensions. VS Code, for example, supports Python interpreter selection, execution and debugging. [VS Code Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial)

A **Python interpreter** executes Python. An **environment** selects that interpreter and its packages. A notebook **kernel** is the process holding variables and executing cells. The browser is usually its interface; the EAGE sandbox instead runs Pyodide in a browser worker. [venv](https://docs.python.org/3/library/venv.html), [Jupyter architecture](https://docs.jupyter.org/en/latest/projects/architecture/content-architecture.html), [Pyodide](https://pyodide.org/en/stable/)

## Choose a workflow

| Tool | Execution location | Saved work | Reproducibility check |
|---|---|---|---|
| EAGE sandbox | Browser worker | Exported drafts/files | Stop/reset clears variables and uploaded files |
| Jupyter Notebook / JupyterLab | Local computer or kernel server | `.ipynb` plus separate inputs | Restart and run all |
| Google Colab | Usually a Google-managed VM | Notebook in Drive; runtime files saved separately | Reconnect and rerun setup/download cells |
| marimo | Local/server Python or configured browser deployment | `.py` notebook | Check dependencies and external-data refresh |
| VS Code | Selected local/remote interpreter | Scripts and notebooks | Verify selected environment |

Software sources: [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/user/interface.html), [Colab FAQ](https://research.google.com/colaboratory/faq.html), [marimo](https://docs.marimo.io/guides/reactivity/), [VS Code environments](https://code.visualstudio.com/docs/python/environments). The sandbox row describes this course's implementation.

## Local setup

1. Install a supported Python 3 release compatible with the course requirements from [python.org](https://www.python.org/downloads/).
2. Create a course folder, with `data` and `outputs` subfolders.
3. Open a terminal there and create an environment:

```bash
python -m venv .venv
```

4. Activate with `.venv\Scripts\activate.bat` in Windows Command Prompt, or `source .venv/bin/activate` in macOS/Linux. PowerShell users can invoke `.venv\Scripts\python.exe -m pip ...` directly without activation.
5. In the activated environment:

```bash
python -m pip install jupyterlab ipykernel numpy pandas matplotlib
python -m jupyterlab
```

These are terminal commands, not Python cells. `python -m pip` targets that interpreter. Activation changes command lookup; it does not install packages. [venv](https://docs.python.org/3/library/venv.html), [Jupyter installation](https://jupyter.org/install), [pip](https://pip.pypa.io/en/stable/user_guide/)

## Check your environment

```{code-cell} python
import sys
from pathlib import Path
print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
print("Current folder:", Path.cwd())
```

A missing package may be installed in a different environment. A missing CSV may be in a different working folder. Inspect the paths first. [Import errors](https://docs.python.org/3/library/exceptions.html#ModuleNotFoundError), [pathlib](https://docs.python.org/3/library/pathlib.html)

[Spyder](https://docs.spyder-ide.org/current/index.html) is a scientific IDE with a variable explorer; it does not require Anaconda. [PyCharm](https://www.jetbrains.com/help/pycharm/quick-start-guide.html) integrates Python development and debugging. [Sublime Text](https://www.sublimetext.com/docs/) is an extensible editor.

Anaconda is a distribution; conda manages environments/packages; Jupyter is independent. A `venv` uses the Python installation that creates it. [conda concepts](https://docs.conda.io/projects/conda/en/stable/user-guide/concepts/index.html), [venv](https://docs.python.org/3/library/venv.html)

Continue with [Jupyter](../section3/notebooks/jupyter-notebooks.md), [Colab](../section3/notebooks/google-colab.md), and [marimo](../section3/notebooks/marimo.md).
