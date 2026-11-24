# Python for Renewable Energy Data Processing: An Extensive Online Short Course for Geoscientists and Engineers

```{image} Logo4.png
:alt: Wind turbines beyond a green field
:class: bg-primary mb-1
:width: 800px
:align: center
```

Last technical review: November 24, 2026

Start with the [learning guide](learningGuide.md), use the [Applied Python Workbook](section7/renewableExercises.md), and consult the [reference catalogue](references.md).

## Course Description

This course applies Python to renewable-energy data and engineering. It connects energy datasets, exploratory analysis, SQL, time-series methods, and engineering calculations for hydroelectric, solar, wind, and geothermal systems.
Participants build and apply a Python foundation to load, validate, analyse, visualise, and store real energy data. An optional [book time-series case study](section4/timeSeriesEnergyConsumption.ipynb), outside the browser lesson navigation, introduces reproducible time-series analysis and predictive modelling. It requires the [course repository](https://github.com/roderickperez/EAGE_PythonRenewableEnergyCourse) and its bundled spreadsheets; the notebook alone does not contain the data. The browser course also includes a chronological forecasting exercise in the [Wind Data Lab](section7/renewableEnergyquizzes/eolicenergy.md). Distinguish measured data, engineering estimates, and model forecasts.

Module 2 introduces programming fundamentals, notebooks, NumPy, pandas and plotting. Exercises can run in the course sandbox, in a local Python environment or in a notebook with the required data and packages.

## Python preparation

Start with [Python foundations for renewable energy](section0/python-foundations.md) if you are new to Python or want a refresher. The separate [Python Course](https://roderickperezaltamar.com/private-course/eage-python-course/index.html) provides further practice and may require separate access.

For additional self-study, the official guides cover [Python fundamentals](https://docs.python.org/3/tutorial/) (aimed at people with some programming experience), [using Jupyter notebooks](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html), [NumPy arrays](https://numpy.org/doc/stable/user/quickstart.html), [pandas data analysis](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html), and [Matplotlib plotting](https://matplotlib.org/stable/tutorials/pyplot.html). These guides are supplementary.

## EAGE course outline

| Module | Lessons and practice |
| --- | --- |
| 1. Introduction to renewable energy | [Fundamentals and applications](section0/introRenewableEnergy.md) across hydroelectric, wind, solar and geothermal energy |
| 2. Python programming | [Python foundations](section0/python-foundations.md) and [general Python quiz](section7/renewableEnergyquizzes/generalPython.md) |
| 3. Data processing and analysis | [Eurostat statistics, exploratory analysis, charts and seasonality](section4/energy-data-analysis.md) |
| 4. Advanced data visualization | [Static and interactive choropleths with GeoPandas and Folium](section4/energy-maps.md) |
| 5. Wind energy | [Generation principles](section6/windEnergy.md), [worldwide installed capacity](section6/labs/wind-capacity.md), [wind maps](section7/workbook/wind.md) and [time-series estimates](section7/renewableEnergyquizzes/eolicenergy.md) |
| 6. Hydroelectric energy | [Generation principles](section6/hydroelectricEnergy.md) and [river mapping and flood flow](section6/labs/river-flood-flow.md) |
| 7. Solar energy | [Radiation and PV generation](section6/solarEnergy.md) and [modelled and measured irradiance](section6/labs/solar-irradiance.md) |
| 8. Geothermal energy | [Generation principles](section6/geothermalEnergy.md) and [world gradient map and well temperature log](section6/labs/geothermal-gradient.md) |

Published reference [EAGE course description and syllabus](https://learninggeoscience.org/mod/page/view.php?id=6785).

(how-the-browser-sandbox-was-made)=

## How the Python Sandbox works

The [Python sandbox](https://roderickperez.github.io/EAGE_PythonRenewableEnergyCourse/sandbox/) combines the course's Markdown and Jupyter notebook lessons with an HTML, CSS and JavaScript interface. A build script turns the public course content into the lessons, examples and exercises you see here. The code editor sends your Python to [Pyodide](https://pyodide.org/), the technology that makes it run directly in your browser without installing Python on your computer.

Pyodide brings CPython, the standard Python interpreter, to the browser by compiling it to **WebAssembly** with Emscripten. WebAssembly lets browsers execute compiled code, while Pyodide connects Python to JavaScript. This sandbox uses **Pyodide 314.0.7 (Python 3.14.2)** inside a **Web Worker**, a background browser task. That keeps the interface responsive while Python runs and lets the Stop button terminate a running program. Results return to the page as text, tables and figures.

The first run downloads the runtime and required packages from jsDelivr, so an internet connection is needed. Python executes on your device, and drafts are saved in this browser; export your work to keep a backup. Restarting Python clears runtime variables and uploaded files. Code that requests external data still needs network access and must respect browser access restrictions.

Variables are shared while you work within one lesson. Switching lessons creates a fresh variable namespace, including when you return to an earlier lesson, so rerun its setup. Saved drafts are separate from those variables. Runtime files can persist across lesson switches: if you edit a supplied dataset in place, returning to the lesson does not restore its original contents. Preserve a copy or checksum before editing. **Restart Python** clears the temporary runtime files and reloads supplied course assets when next needed; download any files you want to keep first.

### Libraries available in this Pyodide version

The table below lists **all 286 packages and their versions** in the official [Packages built in Pyodide](https://pyodide.org/en/stable/usage/packages-in-pyodide.html) table for **314.0.7**, checked on **8 October 2026** against the sandbox's [pinned package manifest](https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide-lock.json). The documentation's `stable` link may change with future releases; this table records the version used here.

When troubleshooting version-specific behavior, check the runtime version and consult a matching documentation series, such as the [Python 3.14 tutorial](https://docs.python.org/3.14/tutorial/), [NumPy 2.4 guide](https://numpy.org/doc/2.4/user/) or [Matplotlib 3.10 guide](https://matplotlib.org/3.10.8/users/index.html). These guides may describe a later patch release. A cited library's documentation does not by itself mean that library is installed in the sandbox.

These are packages distributed with Pyodide, loaded when needed rather than all at startup. The sandbox detects imports to load bundled packages. Additional compatible pure-Python wheels can be installed with `micropip`; this sandbox also installs Plotly, Folium, OpenPyXL and Seaborn when their examples need them. Python's standard library is separate from this package table. Package availability does not guarantee every feature works in a browser: desktop interfaces, Jupyter widgets, authenticated services and some native extensions need a local or Colab environment.

<details>
<summary>Show all 286 packages available in Pyodide 314.0.7</summary>

| Library / package | Version |
|---|---|
| affine | 2.4.0 |
| aiohappyeyeballs | 2.6.1 |
| aiohttp | 3.13.5 |
| aiosignal | 1.4.0 |
| altair | 6.0.0 |
| annotated-doc | 0.0.4 |
| annotated-types | 0.7.0 |
| anyio | 4.13.0 |
| argon2-cffi | 23.1.0 |
| argon2-cffi-bindings | 25.1.0 |
| astropy | 7.2.0 |
| astropy_iers_data | 0.2026.4.1.15.5.49 |
| asttokens | 3.0.1 |
| async-timeout | 5.0.1 |
| asyncpg | 0.31.0 |
| atomicwrites | 1.4.1 |
| attrs | 26.1.0 |
| audioop-lts | 0.2.2 |
| b2d | 0.7.4 |
| bcrypt | 5.0.0 |
| beautifulsoup4 | 4.14.3 |
| bilby.cython | 0.5.4 |
| biopython | 1.87 |
| bitarray | 3.8.1 |
| bitstring | 4.4.0 |
| bleach | 6.3.0 |
| bokeh | 3.9.0 |
| boost-histogram | 1.7.1 |
| Bottleneck | 1.6.0 |
| brotli | 1.2.0 |
| cachetools | 7.0.5 |
| Cartopy | 0.25.0 |
| casadi | 3.7.2 |
| cbor-diag | 1.1.2 |
| certifi | 2026.4.22 |
| cffi | 2.0.0 |
| cffi_example | 0.1 |
| cftime | 1.6.5 |
| charset-normalizer | 3.4.7 |
| clarabel | 0.11.1 |
| click | 8.3.1 |
| cligj | 0.7.2 |
| clingo | 5.8.0 |
| cloudpickle | 3.1.2 |
| cmyt | 2.0.2 |
| cobs | 1.2.2 |
| colorspacious | 1.1.2 |
| contourpy | 1.3.3 |
| coolprop | 7.2.0 |
| coverage | 7.13.5 |
| crc32c | 2.8 |
| crcmod | 1.7 |
| cryptography | 47.0.0 |
| cssselect | 1.4.0 |
| cvxpy-base | 1.8.2 |
| cycler | 0.12.1 |
| cysignals | 1.12.3 |
| cytoolz | 1.1.0 |
| decorator | 5.2.1 |
| demes | 0.2.3 |
| deprecated | 1.3.1 |
| deprecation | 2.1.0 |
| diskcache | 5.6.3 |
| distlib | 0.4.0 |
| distro | 1.9.0 |
| dnspython | 2.8.0 |
| docutils | 0.22.4 |
| donfig | 0.8.1.post1 |
| duckdb | 1.5.1 |
| ewah_bool_utils | 1.3.0 |
| exceptiongroup | 1.3.1 |
| executing | 2.2.1 |
| fastapi | 0.136.1 |
| fiona | 1.10.1 |
| fonttools | 4.62.1 |
| freesasa | 2.2.1 |
| frozenlist | 1.8.0 |
| fsspec | 2026.3.0 |
| future | 1.0.0 |
| galpy | 1.11.2 |
| geopandas | 1.1.3 |
| gmpy2 | 2.3.0 |
| google-crc32c | 1.8.0 |
| h11 | 0.16.0 |
| h3 | 4.4.2 |
| h5py | 3.13.0 |
| healpy | 1.19.0 |
| highspy | 1.13.1 |
| html5lib | 1.1 |
| httpcore | 1.0.9 |
| httpx | 0.28.1 |
| idna | 3.11 |
| igraph | 1.0.0 |
| imageio | 2.37.3 |
| iminuit | 2.30.1 |
| iniconfig | 2.3.0 |
| inspice | 1.7.0.5 |
| ipython | 9.12.0 |
| jedi | 0.19.2 |
| Jinja2 | 3.1.6 |
| jiter | 0.13.0 |
| joblib | 1.5.3 |
| jsonpatch | 1.33 |
| jsonpointer | 3.1.1 |
| jsonschema | 4.26.0 |
| jsonschema_specifications | 2025.9.1 |
| kiwisolver | 1.5.0 |
| lakers-python | 0.6.2 |
| lazy_loader | 0.5 |
| lazy-object-proxy | 1.12.0 |
| libcst | 1.8.6 |
| librt | 0.8.1 |
| lightgbm | 4.6.0 |
| logbook | 1.9.2 |
| lxml | 6.1.3 |
| lz4 | 4.4.5 |
| MarkupSafe | 3.0.3 |
| matplotlib | 3.10.8 |
| matplotlib-inline | 0.2.1 |
| memory-allocator | 0.2.0 |
| micropip | 0.11.1 |
| ml_dtypes | 0.5.4 |
| mmh3 | 5.2.1 |
| more-itertools | 11.0.1 |
| mpmath | 1.4.1 |
| msgpack | 1.1.2 |
| msgspec | 0.20.0 |
| msprime | 1.4.1 |
| multidict | 6.7.1 |
| munch | 4.0.0 |
| mypy | 1.19.1 |
| mysqlclient | 2.2.8 |
| narwhals | 2.18.1 |
| ndindex | 1.10.1 |
| netcdf4 | 1.7.4 |
| networkx | 3.6.1 |
| newick | 1.11.0 |
| nh3 | 0.3.4 |
| nlopt | 2.9.1 |
| nltk | 3.9.4 |
| numcodecs | 0.15.1 |
| numpy | 2.4.6 |
| openai | 2.30.0 |
| opencv-python | 4.11.0.86 |
| optlang | 1.9.0 |
| orjson | 3.11.8 |
| packaging | 26.1 |
| pandas | 3.0.2 |
| parso | 0.8.6 |
| patsy | 1.0.2 |
| pcodec | 1.0.1 |
| peewee | 4.0.4 |
| phispy | 5.0.6 |
| pi-heif | 1.3.0 |
| Pillow | 12.2.0 |
| pillow-heif | 1.3.0 |
| pkgconfig | 1.6.0 |
| platformdirs | 4.9.4 |
| pluggy | 1.6.0 |
| ply | 3.11 |
| polars | 1.33.1 |
| prompt_toolkit | 3.0.52 |
| propcache | 0.4.1 |
| protobuf | 7.34.1 |
| psycopg | 3.3.5 |
| psycopg-binary | 3.3.5 |
| psycopg-c | 3.3.5 |
| pure-eval | 0.2.3 |
| py | 1.11.0 |
| pyarrow | 22.0.0 |
| pyclipper | 1.4.0 |
| pycparser | 3.0 |
| pycryptodome | 3.23.0 |
| pydantic | 2.12.5 |
| pydantic_core | 2.41.5 |
| pydoc_data | 1.0.0 |
| pyerfa | 2.0.1.5 |
| pygame-ce | 2.5.7 |
| Pygments | 2.20.0 |
| pyheif | 0.8.0 |
| pyiceberg | 0.11.1 |
| pyinstrument | 5.1.2 |
| pymongo | 4.16.0 |
| pynacl | 1.6.2 |
| pyodide-http | 0.2.2 |
| pyodide-unix-timezones | 1.0.0 |
| pyparsing | 3.3.2 |
| pyproj | 3.7.2 |
| pyroaring | 1.0.4 |
| pyrodigal | 3.7.1 |
| pyrsistent | 0.20.0 |
| pysam | 0.23.0 |
| pyshp | 3.0.3 |
| pytaglib | 3.2.0 |
| pytest | 9.0.2 |
| pytest-asyncio | 0.25.3 |
| pytest-benchmark | 4.0.0 |
| pytest_httpx | 0.36.0 |
| python-calamine | 0.6.2 |
| python-dateutil | 2.9.0.post0 |
| python-flirt | 0.9.10 |
| python-sat | 1.8.dev26 |
| python-solvespace | 3.0.8 |
| pytz | 2026.1.post1 |
| pywavelets | 1.9.0 |
| pyxirr | 0.10.8 |
| pyyaml | 6.0.3 |
| rasterio | 1.5.0 |
| rateslib | 2.7.1 |
| rebound | 4.4.7 |
| reboundx | 4.4.1 |
| referencing | 0.37.0 |
| regex | 2026.3.32 |
| requests | 2.33.1 |
| retrying | 1.4.2 |
| rich | 14.3.3 |
| rpds-py | 0.30.0 |
| ruamel.yaml | 0.19.1 |
| safetensors | 0.7.0 |
| scikit-image | 0.26.0 |
| scikit-learn | 1.8.0 |
| scipy | 1.18.0 |
| screed | 1.1.3 |
| sentencepiece | 0.2.1 |
| setuptools | 82.0.1 |
| shapely | 2.1.2 |
| simplejson | 3.20.2 |
| sisl | 0.16.4 |
| six | 1.17.0 |
| smart-open | 7.5.1 |
| sniffio | 1.3.1 |
| sortedcontainers | 2.4.0 |
| soundfile | 0.12.1 |
| soupsieve | 2.8.3 |
| sourmash | 4.8.14 |
| soxr | 0.5.0.post1 |
| sparseqr | 1.2 |
| sqlalchemy | 2.0.48 |
| stack-data | 0.6.3 |
| starlette | 1.0.0 |
| statsmodels | 0.14.6 |
| strictyaml | 1.7.3 |
| svgwrite | 1.4.3 |
| swiglpk | 5.0.13 |
| sympy | 1.14.0 |
| tblib | 3.2.2 |
| termcolor | 3.3.0 |
| texttable | 1.7.0 |
| texture2ddecoder | 1.0.6 |
| threadpoolctl | 3.6.0 |
| tiktoken | 0.12.0 |
| tomli | 2.4.1 |
| tomli-w | 1.2.0 |
| toolz | 1.1.0 |
| tqdm | 4.67.3 |
| traitlets | 5.14.3 |
| traits | 7.1.0 |
| tree-sitter | 0.23.2 |
| tree-sitter-go | 0.23.3 |
| tree-sitter-java | 0.23.4 |
| tree-sitter-python | 0.23.4 |
| tskit | 1.0.2 |
| typing-extensions | 4.15.0 |
| typing-inspection | 0.4.2 |
| tzdata | 2025.3 |
| ujson | 5.12.0 |
| uncertainties | 3.2.3 |
| unyt | 3.1.0 |
| urllib3 | 2.6.3 |
| vega-datasets | 0.9.0 |
| vrplib | 2.1.0 |
| wcwidth | 0.6.0 |
| webencodings | 0.5.1 |
| wordcloud | 1.9.6 |
| wrapt | 2.1.2 |
| xarray | 2026.2.0 |
| xgboost | 2.1.4 |
| xlrd | 2.0.2 |
| xxhash | 3.6.0 |
| xyzservices | 2026.3.0 |
| yarl | 1.23.0 |
| yt | 4.4.2 |
| zarr | 3.2.1 |
| zengl | 2.7.2 |
| zfpy | 1.0.1 |
| zstandard | 0.25.0 |

</details>

For further reading, see the [Pyodide website](https://pyodide.org/), [package-loading guide](https://pyodide.org/en/stable/usage/loading-packages.html), [Web Worker guide](https://pyodide.org/en/stable/usage/webworker.html), and the course's [sandbox technology references](references.md#sandbox-technology-documentation).

## Objectives

- Apply existing Python, NumPy, pandas, and plotting knowledge to renewable-energy datasets and engineering calculations.
- Explain the limits of descriptive analysis, engineering models, and predictive models, and avoid common problems such as data leakage and invalid train/test splits.
- Apply techniques of analysis and visualization of geoscientific data using the libraries from Python.
- Interpret the output obtained by the prediction models.
- Build and evaluate simple chronological forecast baselines for renewable-energy data. Deep learning is an optional next step.

## Learning outcomes

By the end of the course, participants should be able to:

1. Validate renewable-energy data units, grain, missing values, and category definitions before aggregation.
2. Query and create SQLite databases using reproducible SQL and pandas workflows.
3. Analyse energy time series with appropriate sampling frequencies and rolling windows; the optional book case study adds decomposition and chronological validation.
4. Apply and check the principal equations used in hydroelectric, solar, wind, and geothermal resource calculations.
5. Build clear, correctly labelled plots and interpret their engineering meaning.
6. Compare measured data, engineering estimates, and model forecasts without conflating them.
7. Build and evaluate simple forecast baselines using chronological validation.
8. Communicate assumptions, uncertainty, data freshness, and model limitations.

## Scope and prerequisites

Prior Python experience is optional. General programming concepts, basic descriptive statistics, algebra and familiarity with energy units are helpful. Learners new to Python should work through Module 2 before starting the data and engineering exercises. This is an applied introductory renewable-energy course; it does not replace a full power-systems, resource-assessment, statistics, or deep-learning course.

## Course navigation

Use the course navigation to choose lessons, exercises and projects. Consult your course organizer for session dates.

## Instructor

**Roderick Perez Altamar, Ph.D.**
Geophysical Engineer from the Simón Bolívar University in Venezuela, with a Master's degree in Geology and a Ph.D. in Geophysics from the University of Oklahoma, an MBA from the Universidad de Los Andes, and authored a [2025 University of Vienna Data Science master’s thesis](https://datascience.univie.ac.at/index.php?L=0&id=185018). Roderick is a seismic
interpreter, with more than 15 years of experience in the Oil & Gas industry, where he has developed techniques for characterizing the fragility of these reservoirs. He specialized in the characterization of
YNC in the USA (Barnett Shale, Eagle Ford, Marcellus Shale, Permian Basin, among others), as well as the characterization and economic evaluation of conventional reservoirs in Colombia, Ecuador, Argentina, among others. Roderick is an expert in pre and post-stack seismic inversion, as well as in the application of Machine Learning and Neural Networks in the analysis of geoscientific data.

roderickperezaltamar@gmail.com

---

## Organized by

**European Association of Geoscientists and Engineers (EAGE)**

## Contact Information

[**Maria Paula Bohorquez**](mailto:mtz@eage.org)

Community Manager


### Course contacts

- Alexandra La Rocca: [aca@eage.org](mailto:aca@eage.org)
- Ilaria Meggetto: [imo@eage.org](mailto:imo@eage.org)
- Isabela Osorio: [ioo@eage.org](mailto:ioo@eage.org)
