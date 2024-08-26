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
* [Atom](https://atom.io/): Atom is an open-source code editor developed by Github that can be used for Python development (similar Sublime text).
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

Google Colab is very good for neural network training and development, specially since if we have limited computing resources, we can use Colab for development and testing. One of the biggest advantages of Google Colab is that let us select between CPU (Central Processing Units), GPU (Graphical Processing Units (15 -20x CPU) and TPU (Tensor Processing Units - 30x GPU) during the runtime of our code. 

Next table compare the performance of Colab and Jupyter Notebooks.

```{table} Jupyer vs Colab
:name: jupyer-vs-colab-comparison

| Jupyter Notebook | Google Colab |
|---|---|
| Direct access to local file system | Data stored in GDrive |
| Local computation: Uses your local hardware (CPU | GPU) | Remote Computation: 12 GB GPU RAM for up to 12 hours |
| Local Resources | 128 bit FLOPS (floating point number) |
| Local Resources | Unlimited RAM (up to 13 Gb RAM) |
| Local Resources | Unlimited disk space (up to 103 Gb) |
| Install packages locally just once | Re-install packages for each session |
| Considered safer in terms of data security | Allows collaborative work between developers |
```

### Where are my notebooks stored, and can I share them?link
Colab notebooks are stored in Google Drive, or can be loaded from GitHub. Colab notebooks can be shared just as you would with Google Docs or Sheets. Simply click the Share button at the top right of any Colab notebook, or follow these Google Drive [file sharing instructions](https://support.google.com/drive/answer/2494822?co=GENIE.Platform%3DDesktop&hl=en).

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


```{image} ../../images/googleColabAnatomy.png
:alt: googleColabAnatomy
:class: bg-primary mb-1
:width: 800px
:align: center
```
