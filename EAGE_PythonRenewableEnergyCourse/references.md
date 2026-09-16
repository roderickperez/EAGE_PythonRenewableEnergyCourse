# References and reading guide

Reviewed 16 September 2026. This catalogue covers all 10 PDF files in the repository’s `referenceBooks/` folder (9 distinct works). Publication metadata were checked against the supplied title/copyright pages; topic locators use printed chapters or pages, not PDF-viewer page numbers.

The exercises are original teaching adaptations of these concepts, with synthetic numbers; they are not copied textbook problems. Older books support physical principles, while dated reports support market context. No local PDFs are republished by the book build.

## Books and documents supplied with the course

### Geothermal Reservoir Engineering

[@grant2011]

**Local file:** `referenceBooks/(64) Geothermal Reservoir Engineering- Second Edition.pdf`

**Edition / identifier:** Second edition; ISBN 978-0-12-383880-3.

**Use in this course:** Chapters 2–3: conductive and convective systems, simple models and stored heat; Chapters 4–5: well measurements.

### Guideline and Manual for Hydropower Development, Vol. 2: Small Scale Hydropower

[@jica2011]

**Local file:** `referenceBooks/Guideline and Manual for Hydropower Development Vol 2.pdf`

**Edition / identifier:** March 2011; prepared with Electric Power Development Co. and JP Design.

**Use in this course:** Chapter 3: power and energy; Chapters 7–8: flow-duration curves and generation planning.

### Hydroelectric Power: A Guide for Developers and Investors

[@ifc2015]

**Local file:** `referenceBooks/Hydroelectric Power - A Guide for Developers and Investors.pdf`

**Edition / identifier:** Prepared with Fichtner.

**Use in this course:** Chapter 7: hydrology and energy calculations; Chapters 12–14: environmental effects, costs and economics.

### Clean Energy Technology Observatory: Hydropower and Pumped Storage Hydropower in the European Union — 2025 Status Report on Technology Development, Trends, Value Chains and Markets

[@jrc2025hydro]

**Local file:** `referenceBooks/Hydropower and Pumped Storage Hydropower inthe European Union.pdf`

**Edition / identifier:** JRC143929; DOI 10.2760/3389272.

**Use in this course:** Section 2: technology, capacity, generation and costs. Figures retain the report’s observation years.

### Hydropower and Renewable Energies: Powering a Sustainable Future with Storage and Renewables

[@hu2026]

**Local file:** `referenceBooks/Hydropower and Renewable Energies Powering a Sustainable Future with Storage and Renewables.pdf`

**Edition / identifier:** Edited proceedings; Lecture Notes in Civil Engineering 792; DOI 10.1007/978-981-95-4889-7.

**Use in this course:** Multi-energy integration chapters, including capacity allocation (p. 35) and complementary dispatch (p. 65). Advanced reading, not a beginner design manual.

### Solar Photovoltaic Systems: Technical Training Manual

[@wade2003]

**Local file:** `referenceBooks/Solar Photovoltaic Systems Technical Training Maunal.pdf`

**Edition / identifier:** ISBN 92-3-103904-0; filename retains its original spelling.

**Use in this course:** Chapters 2–3: electricity and PV panels; Chapters 5 and 9: batteries and system sizing. Historical equipment guidance requires current manufacturer specifications.

### Solar Energy: Renewable Energy and the Environment

[@foster2010]

**Local file:** `referenceBooks/Solar-energy-Renewable-Energy-and-the-Environment.pdf`

**Edition / identifier:** ISBN 978-1-4200-7566-3.

**Use in this course:** Solar-resource and photovoltaic chapters: irradiation, conversion and system performance. Use current sources for prices and installed capacity.

### Introduction to Wind Energy Systems: Basics, Technology and Operation

[@wagner2009]

**Local file:** `referenceBooks/Wagner et Mathur - 2009 - Introduction to Wind Energy Systems.pdf`

**Edition / identifier:** DOI 10.1007/978-3-642-02023-0.

**Use in this course:** Wind resource, turbine fundamentals and Chapter 7 economics; suitable introductory companion.

### Wind Energy Explained: Theory, Design and Application

[@manwell2009]

**Local file:** `referenceBooks/Wind Energy Explained.pdf`

**Edition / identifier:** Second edition; ISBN 978-0-470-01500-1.

**Use in this course:** Sections 2.3–2.5: height, wind statistics and energy estimation; Section 3.2: Betz limit; Sections 10.3 and 10.7: hybrid systems and storage.

### Duplicate solar PDF

`referenceBooks/Solar-energy-Renewable-Energy-and-the-Environment (1).pdf` is byte-for-byte identical to the Foster, Ghassemi and Cota file above (SHA-256 checked). Both files are preserved, but use one bibliographic entry [@foster2010].

## Current context and official technical documentation

- [IEA Global Energy Review 2026](https://www.iea.org/reports/global-energy-review-2026): estimates for **2025**, not completed 2026 statistics [@iea2026].
- [IRENA Renewable Power Generation Costs in 2025](https://www.irena.org/Publications/2026/Jul/Renewable-Power-Generation-Costs-in-2025): July 2026 edition. Do not substitute global cost averages for a project budget [@irena2026].
- [Eurostat energy database](https://ec.europa.eu/eurostat/web/energy/database): the course retains August 2024 exports as reproducible historical inputs [@eurostat_energy_database].
- [Python tutorial](https://docs.python.org/3/tutorial/) and [built-in functions](https://docs.python.org/3/library/functions.html) [@pythonDocs].
- [NumPy user guide](https://numpy.org/doc/stable/user/) [@numpyDocs].
- [pandas indexing and selection](https://pandas.pydata.org/docs/user_guide/indexing.html) [@pandasDocs].
- [Matplotlib plotting documentation](https://matplotlib.org/stable/users/index.html) [@matplotlibDocs].
- [SQLite SQL language](https://www.sqlite.org/lang.html) [@sqliteDocs].
- [pvlib PVWatts DC model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html) [@pvlibDocs].

## Further reading already cited in the lessons

Wind Energy Handbook [@burton2011wind]; Solar Energy Engineering [@kalogirou2014solar]; Geothermal Power Plants [@diPippo2016geothermal]; Small Hydro Power [@paish2002hydro]. These are additional references, not additional supplied PDF files.

## How to cite an exercise

Give its number and course title, the underlying book/chapter, the code revision used, and any external data’s provider, product/version, retrieval date, units, location and observation period. State all model assumptions. For synthetic data, say so explicitly. See the [100-exercise chapter guide](section7/renewableExercises.md).
