# Google Colab

Colab provides hosted Jupyter notebooks. A managed runtime executes Python on a remote VM; your browser displays the interface. Accelerator availability and session lifetimes vary. The course needs no GPU and promises no fixed runtime. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

## Start and save

1. Open [Colab](https://colab.research.google.com/) and sign in.
2. Create a notebook or use **File → Upload notebook** for a course `.ipynb`.
3. Save a copy in Drive with a meaningful name.
4. Connect to a CPU runtime and run:

```{code-cell} python
import sys
from pathlib import Path
print("Python:", sys.version.split()[0])
print("Working directory:", Path.cwd())
print("Example energy:", 2.5 * 4, "MWh")
```

This original calculation assumes constant 2.5 MW for four hours [@jica2011], Chapter 3. Inspection functions: [sys](https://docs.python.org/3/library/sys.html), [pathlib](https://docs.python.org/3/library/pathlib.html).

5. If necessary, install packages in a **Colab/IPython command cell**:

```ipython
%pip install pandas matplotlib
```

This targets the active kernel; an imported package may need a runtime restart after a version change. [IPython package magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pip)

## Download and upload data

1. Open the exercise's CSV link. Save the data file, not the surrounding HTML.
2. Upload through Colab's **Files** panel. The usual folder is `/content`; check `Path.cwd()`.
3. Read the exact filename with `pd.read_csv`. Upload all five CSVs for the final project.
4. Export your notebook and download result files before the runtime ends.

Saving a notebook in Drive does not automatically save arbitrary runtime files. Sharing includes saved cells/outputs, but not your running VM or its files/packages. [Colab FAQ: sharing and runtime state](https://research.google.com/colaboratory/faq.html)

The [download-to-database tutorial](../../section5/download-to-database.md) includes a live Eurostat request, filenames, checks and recovery instructions. It needs no API key or Drive mount.

## Practice

- **Easy:** locate the tutorial's downloaded CSV in Files.
- **Medium:** download it to your computer and upload it into a fresh notebook.
- **Challenge:** reproduce the result in a fresh runtime using only your documented setup/download steps.
