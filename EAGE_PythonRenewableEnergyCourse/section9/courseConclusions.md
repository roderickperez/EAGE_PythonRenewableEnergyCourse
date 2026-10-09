# Course conclusions

You have practised turning renewable-energy questions into Python calculations with explicit assumptions, units, data checks and interpretable results. Use the [completion checklist](courseSummary.md) to identify topics that need another attempt.

## Keep the physical meaning visible

Power is a rate; energy accumulates over time. Capacity factor, conversion efficiency and availability answer different questions. Hydro needs net head and usable flow; wind energy requires a power curve and resource distribution; PV output depends on effective irradiance and cell temperature; geothermal thermal power must be converted to a defined electrical boundary. [@jica2011], Chapter 3; [@manwell2009], §§2.3–2.5; [@foster2010], Chapter 5; [@grant2011], Chapters 2–3.

## Make the work reproducible

Keep the downloaded bytes, provider definitions, filters and retrieval record with the calculation. Restart the kernel and run all cells before sharing a notebook. Distinguish measured data, synthetic exercise inputs and model estimates. [Jupyter notebook workflow](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html); [Eurostat API documentation](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-getting-started).

A useful final submission explains its assumptions, reconciles its balances, labels its figures and reports what cannot be concluded from its data. This is the course's assessment standard; executing without an exception alone is insufficient.

## Continue practising

Reproduce the [download-to-database tutorial](../section5/download-to-database.md), complete the [question-only assessment](../section7/renewableEnergyTest.md), and submit your own [hybrid portfolio project](../section8/projectIntro.md). For deeper physical modelling, follow the technology-specific chapters in the [reference catalogue](../references.md).
