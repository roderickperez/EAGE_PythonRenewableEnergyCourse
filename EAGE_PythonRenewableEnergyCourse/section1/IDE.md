# IDE

**Objetives**
* Provide an *overview* of Python IDEs and code editors for beginners and professionals.
---

## Code Editor
A code editor is a tool that is used to write and edit code. They are usually lightweight and can be great for learning. However, once your program gets larger, you need to test and debug your code, that's where IDEs come in.

## IDEs

An **IDE** (Integrated Development Environment) understand your code much better than a text editor. It usually provides features such as build automation, code linting, testing and debugging. This can significantly speed up your work. The downside is that (sometimes) IDEs can be complicated to use.

On Google we can find several IDEs, such as: 

* [Sublime Text](https://www.sublimetext.com/): Sublime Text is a popular code editor that supports many languages including Python. It's fast, highly customizable and has a huge community.
* [Visual Studio Code](https://code.visualstudio.com/): a code editor with Python, notebook, debugging, and environment extensions.
* [Visual Studio Code](https://code.visualstudio.com/): Visual Studio Code (VS Code) is a free and open-source IDE created by Microsoft that can be used for Python development.
* [PyCharm](https://www.jetbrains.com/pycharm/): PyCharm is an IDE for professional developers. It is created by JetBrains, a company known for creating great software development tools.
* [Eclipse](https://www.eclipse.org/): Eclipse is a free and open-source IDE that can be used for Python development.
* [Spyder](https://www.spyder-ide.org/): It is an IDE for Python under Anaconda.
* others.

## Anaconda
According to their website, [Anaconda](https://www.anaconda.com/products/distribution) is a free Python (and also R) distribution (including for commercial use and redistribution). It includes more than 400 of the most popular Python packages for science, math, engineering, and data analysis. The default IDE bundled with Anaconda is Spyder which is just another Python package that can be installed even without Anaconda.

Anaconda provides its own package manager (```conda```) and package repository. But it allows installation of packages from **PyPI** using ```pip``` if the package is not in Anaconda repositories. It is especially good if you are installing on Microsoft Windows as it can easily install packages that would otherwise require you to install C/C++ compilers and libraries if you were using ```pip```. It is certainly an added advantage that **conda**, in addition to being a package manager, is also a virtual environment manager allowing you to install independent development environments and switch from one to the other (similar to virtualenv).

One of the most popular features of Anaconda is that allows to create notebooks, which allow us to write and run code in a single file, combining *cells* of text and code very efficiently in a very interactive way.

## Jupyter

[Jupyter project](https://github.com/jupyter/), or **Jupyter** (**Ju**lia, **Py**thon, **R**), is a free web-based software for interactive computing across multiple programming languages, under Anaconda distribution. You can also call it a web application under Anaconda. For some reseachers, it is their prefered tool to prototype their data analysis algorithms.

### Jypiter Lab
Jupyter Lab ia a web-based application that allows you to lets you collect multiple Jupyter Notebooks under one tab. It is also part of Anaconda distribution and it is a good way to prototype your data analysis algorithms.

## Google CoLab
Google Colaboratory, or "[Colab](https://colab.research.google.com/)", is a product from Google Research that is based on the open source [Jupyter project](https://github.com/jupyter/). It allows anybody to write and execute arbitrary Python code through the browser, and is especially well suited to machine learning, data analysis and education. In other words, Colab is a hosted Jupyter notebook service that requires no setup to use, while providing access free of charge to computing resources including GPUs. The only requirement is that you have a Google account.

Google Colab provides hosted notebooks and, when available, access to CPU, GPU, or TPU runtimes. Accelerator availability, memory, session length, and performance vary by plan, region, workload, and current capacity; do not assume a fixed speed-up or runtime limit.

### Where are my notebooks stored, and can I share them?link
Colab notebooks are stored in Google Drive, or can be loaded from GitHub. Colab notebooks can be shared just as you would with Google Docs or Sheets. Simply click the Share button at the top right of any Colab notebook, or follow these Google Drive [file sharing instructions](https://support.google.com/drive/answer/2494822?co=GENIE.Platform%3DDesktop&hl=en).

## Marimo

[Marimo](https://marimo.io/) is a reactive Python notebook framework designed for reproducible scientific and data workflows. Unlike a classic notebook that executes cells in a linear order, Marimo keeps code, output, and dependencies synchronized so that a change in one variable can update the cells that depend on it automatically.

This makes Marimo especially useful when you are building notebooks that behave more like executable analysis scripts than static teaching examples. It is also a good option for sharing notebooks that should be easier to review in version control because the notebook is stored as a Python file rather than a JSON-based `.ipynb` file.

### Install Marimo

To install Marimo locally:

```bash
pip install marimo
```

To launch it in a browser:

```bash
marimo edit
```

For official documentation and source code, see:

- [Marimo documentation](https://docs.marimo.io/)
- [Marimo GitHub repository](https://github.com/marimo-team/marimo)
- [Marimo installation guide](https://docs.marimo.io/getting_started/installation/)

### Why use Marimo?

- Good for reproducible, reactive workflows.
- Makes code easier to version-control and review in Git.
- Works well for notebook-based data analysis and exploratory programming.
- Can be a strong alternative when you want more structure than a traditional notebook but less overhead than a full app framework.

## Comparative overview of notebook environments

The following table compares the main notebook environments used in this course: Jupyter Notebook, Google Colab, and Marimo.

| Tool | Access model | Setup required | Compute resources | GPU / TPU access | Collaboration | Agent coding support | Typical use |
|---|---|---|---|---|---|---|---|
| Jupyter Notebook | Local app in the browser; files saved on your machine | Download and install Python + Jupyter | Depends on your local machine | Usually only if your local hardware supports it | Good with Git and local sharing | Moderate; works well with VS Code and local editors | Classroom, local research, reproducible analysis |
| Google Colab | Web-based; runs in the browser | No local install required | Cloud resources managed by Google | Often available through free and paid plans | Strong with Google Drive and sharing links | Good, especially with AI-assisted coding environments | Fast prototyping, teaching, cloud experiments |
| Marimo | Local browser app or remote deployment | Install with `pip install marimo` | Depends on local machine or server | Depends on the host environment | Good with Git and code-based notebooks | Strong for code-first workflows and AI-assisted exploratory coding | Reproducible notebooks and reactive data apps |

### Practical recommendation

- Use **Jupyter Notebook** when you want a simple local workflow and full control over your environment.
- Use **Google Colab** when you want a quick browser-based environment with cloud compute and no local installation.
- Use **Marimo** when you want a reactive, code-first notebook workflow that feels more structured and version-control friendly.

## Anatomy of a Google Colab Notebook

* Cells
  - Text
  - Code
* Code Snippets
* Files
* Share
* Runtime
  - None
  - GPU
  - TPU

* Shortcuts
  - UP / DOWN: Move between cells
  - ESC: Exit cell edit mode
  - Shift + Enter: Run cell
  - Ctrl + M + Y: Convert a cell from text to code
  - Ctrl + M + M: Convert a cell from code to text
  - Tools > Keyboard shortcuts...


```{image} ../images/section1/googleColabAnatomy.png
:alt: googleColabAnatomy
:class: bg-primary mb-1
:width: 800px
:align: center
```

## References

- [Jupyter Project](https://github.com/jupyter/)
- [Jupyter Notebook documentation](https://jupyter.org/)
- [JupyterLab documentation](https://jupyterlab.readthedocs.io/en/stable/)
- [Google Colab](https://colab.research.google.com/)
- [Google Drive sharing guide](https://support.google.com/drive/answer/2494822?co=GENIE.Platform%3DDesktop&hl=en)
- [Marimo documentation](https://docs.marimo.io/)
- [Marimo GitHub repository](https://github.com/marimo-team/marimo)
- [Marimo installation guide](https://docs.marimo.io/getting_started/installation/)
