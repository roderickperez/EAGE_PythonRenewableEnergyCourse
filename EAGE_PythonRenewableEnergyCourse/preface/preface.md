# Preface

## Vibe Coding or Vibe Debugging

"Vibe coding" is a new term that gained serious traction in 2025. It refers to the reality that people—even those without formal programming knowledge or experience—can now simply talk to their computers. Through Artificial Intelligence, a fully functional application "magically" appears on your screen. You didn't have to code it, and despite a lack of programming skills (or just sheer laziness), it works perfectly... right up until the moment you have to debug and fix code you never understood in the first place, leaving you staring at AI-generated lines that might as well be hieroglyphs.

Andrej Karpathy (an AI researcher who worked at OpenAI and was Director of AI at Tesla) described “vibe coding” in a [2 February 2025 post](https://x.com/karpathy/status/1886192184808149383) (access may depend on the platform).

```{figure} ../images/preface/A_Karpathy_tweet.png
---
name: AndrejKarpathy_original_vibeCoding_Tweet
alt: Screenshot of Andrej Karpathy's post discussing vibe coding
---
```

In short, vibe coding is the use of artificial intelligence for automated application development. Some integrated development environments (IDEs) now include agentic AI features. These tools can lower the barrier to entry by helping people with little programming experience create a first draft, though useful results still require testing and review. The chart below shows relative search interest in the term over the specified baseline period; it does not measure adoption or establish a continuing trend.

```{figure} ../images/preface/vibe_coding_google_trends_worldwide_2021_2026.png
---
name: preface_vibe_coding_google_trends_20260308
alt: Line chart of worldwide relative search interest in vibe coding through February 2026
---
Google Trends monthly interest over time for "vibe coding" (Worldwide), baseline downloaded on March 8, 2026.
```

The baseline index rose from 3 in February 2025 to 100 in February 2026, with intervening fluctuations. The index is normalized relative search interest for this query, not a count of searches or adopters.

AI assistance can help with some tasks, but its usefulness and reliability vary with the tool, task and review. This does not establish that software developers will become obsolete. The programming profession isn't exactly what most people think it is anyway. Coding heavily involves reading, abstracting, and understanding how each line of instruction shapes the final product.

Usually, the "final" product of vibe coding isn't achieved on the first try. To get there, you need to know what you want, grasp basic programming concepts, and understand how your chosen language or library handles data. You need a clear understanding of functions, variables, and formatting. Therefore, one of our first conclusions is that to get a decent—or even good—product out of vibe coding, you still need to know the fundamentals of software engineering.

Oversimplifying heavily and skipping the deeper technical details, a Large Language Model (LLM) is essentially a massive autocomplete system. Similarly, programming relies on giving a machine a sequence of instructions with a clear, logical structure. At its core, an LLM is replicating existing software patterns. Consequently, vibe coding yields excellent results when someone (likely a human) has already built a similar solution in the past.

One of the main limitations of vibe coding is that the instructions given to the AI agent must be highly precise to achieve the desired outcomes.

Models may lack recent information. Consult official documentation, inspect generated code and execute independent checks. Training schedules vary; there is no universal training duration. Retrieval tools can supplement model knowledge.

Also, in professional environments—especially with large codebases—you will run into context window limits. The context window is the equivalent of the model's short-term memory; in short, it is the maximum amount of text (measured in tokens, which are parts of words or syllables) a model can hold in memory at one time. When input exceeds this capacity, the application may reject the request, truncate earlier content, summarize it, or use another policy. Separately, a model may fail to use information that remains in the context; check the product’s actual behavior. Context limits vary by model and product; verify current specifications instead of assuming a fixed size.

At the end of the day, vibe coding follows the same old rules: if you want to build something important, don't turn off your brain. Read the code and use the LLM as your assistant. We shouldn't let the automatic nature of these tools cause our uniquely human ability—thinking—to atrophy or be forgotten.

Claims that AI would write 90% of code within months were forecasts, not established measurements of software production. This course does not use those predictions as evidence.

Vibe coding is here to stay. We can't bury our heads in the sand; AI assistance can be useful, but its effects on quality and productivity depend on the task, tools, experience, and review effort. Therefore, we must learn to coexist with it without losing our human skills.

The objective of this course is to learn the fundamentals and lay the groundwork to supercharge our skills with AI. Let's not just be copilots or spectators of what's to come—let's be the builders of our own future.

## Summary: The 5 Key Takeaways on Vibe Coding
1. **AI is Lowering the Barrier to Entry**: AI assistance can help people without formal programming experience create application drafts; functional and reliable results still require verification and revision.

2. **Debugging Still Requires Human Logic**: While generating the initial application can feel like magic, fixing or modifying that code later can turn into deciphering hieroglyphs if you don't actually understand what the AI built.

3. **Fundamentals are Non-Negotiable**: To consistently get good results—and to successfully debug—a solid grasp of software engineering basics (like functions, variables, and data structures) remains absolutely essential.

4. **LLMs Have Technical Limitations:** AI models are powerful but constrained. They are limited by their training cutoff dates (meaning they might not know the absolute newest software updates), context window sizes (how much memory they have for a single conversation), and their reliance on highly precise human instructions.

5. **The Programmer's Role is Evolving, Not Ending**: As AI takes over the heavy lifting of writing the actual lines of code, the human role is shifting toward reviewing, steering, and applying critical thinking. We must adapt and use AI as an assistant without letting our analytical skills atrophy.
---

## How to Reproduce and Compare with New Data

Use these steps to download a new Google Trends CSV and plot relative interest. Record the term, region, category, search type, date range and retrieval date:

1. Open Google Trends and configure the query:
	- Link: [Google Trends - vibe coding (Worldwide)](https://trends.google.com/trends/explore?date=2021-03-08%202026-03-08&q=vibe%20coding)
	- Term: `vibe coding`
	- Region: `Worldwide`
	- Time range: retain the baseline window for a like-for-like context, or record any changed window. A changed window may rescale the series.
2. In **Interest over time**, click the download icon and save the CSV.
3. Place the CSV in `EAGE_PythonRenewableEnergyCourse/images/preface/`.
4. From the repository root, set `csv_path` below to the exact new filename and choose a writable output path. In the browser workspace, upload the CSV and set `csv_path` to `/uploads/your-file.csv`; use an output name such as `vibe_coding_comparison.png` and download it with the workspace file tool. The loader recognizes common introductory rows and requires one numeric series. The plot title uses that series label; record the region separately because the CSV may not identify it. It rejects flagged partial periods, `<1`, missing values and unfamiliar layouts rather than silently discarding them; inspect and document an explicit treatment before plotting. An unmarked partial period cannot be detected automatically, so check the export and reporting window.

```python
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

filename = "time_series_Worldwide_20210308-2304_20260308-2304.csv"  # baseline; replace with your new file
csv_candidates = (
    Path("EAGE_PythonRenewableEnergyCourse/images/preface") / filename,
    Path("images/preface") / filename,
    Path("../images/preface") / filename,
    Path("/course/images/preface") / filename,
)
csv_path = next((path for path in csv_candidates if path.is_file()), csv_candidates[0])
out_path = Path("vibe_coding_google_trends_comparison.png")

def load_trends(path):
    """Load one numeric Trends series without silently discarding data rows."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header_row = next((i for i, row in enumerate(csv.reader(stream))
                           if len(row) >= 2 and row[0].strip().casefold()
                           in {"time", "date", "day", "week", "month"}), None)
    if header_row is None:
        raise ValueError("No recognized time-series header; inspect the CSV export")
    table = pd.read_csv(path, skiprows=header_row, dtype=str, keep_default_na=False)
    table.columns = [str(c).strip() for c in table.columns]
    partial_columns = [c for c in table.columns[1:] if c.casefold() == "ispartial"]
    value_columns = [c for c in table.columns[1:] if c not in partial_columns]
    if len(value_columns) != 1 or len(partial_columns) > 1:
        raise ValueError("Expected one query series and at most one isPartial column")
    date_col, value_col = table.columns[0], value_columns[0]
    if partial_columns:
        flags = table[partial_columns[0]].str.strip().str.casefold()
        if not flags.isin(["", "false", "0"]).all():
            raise ValueError("Partial or unrecognized period flags: inspect or exclude those periods explicitly")
    dates = pd.to_datetime(table[date_col], errors="coerce")
    raw_values = table[value_col].str.strip()
    values = pd.to_numeric(raw_values, errors="coerce")
    if dates.isna().any() or values.isna().any() or not values.between(0, 100).all():
        raise ValueError("Invalid dates or values (including <1 or partial markers): inspect them; none were dropped")
    if dates.duplicated().any() or len(table) == 0:
        raise ValueError("Empty series or duplicate dates: inspect the export")
    result = pd.DataFrame({date_col: dates, value_col: values}).sort_values(date_col)
    print(f"Read {len(result)} observations from {result[date_col].min().date()} to {result[date_col].max().date()}; query: {value_col}")
    return result, date_col, value_col

df, date_col, value_col = load_trends(csv_path)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(df[date_col], df[value_col], color="#0b6e4f", linewidth=2.5, marker="o", markersize=3)
ax.set_title(f"Google Trends: {value_col}", fontsize=14, pad=12)
ax.set_xlabel("Time")
ax.set_ylabel("Search Interest (0-100)")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.25)
fig.tight_layout()
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=180)
```

5. Compare the shape and context with the baseline shown above. Google Trends normalizes each query to its own peak, so separately downloaded 0–100 series are not directly comparable as absolute search volumes; use a jointly normalized query or a documented overlap calibration when making quantitative comparisons. Google Trends results are samples and may vary between downloads; treat small differences cautiously.

