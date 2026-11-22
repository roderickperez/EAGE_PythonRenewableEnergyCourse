"""Rebuild the dated Google Trends baseline chart from its checked-in CSV."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parents[1] / "EAGE_PythonRenewableEnergyCourse/images/preface"
data = pd.read_csv(root / "time_series_Worldwide_20210308-2304_20260308-2304.csv")
data["Time"] = pd.to_datetime(data["Time"], errors="raise")
values = pd.to_numeric(data["vibe coding"], errors="raise")
if data.empty or not data["Time"].is_monotonic_increasing:
    raise ValueError("The baseline must contain ordered monthly observations")
fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
ax.plot(data["Time"], values, color="#08734e", linewidth=2.5, marker="o", markersize=3)
last_date, last_value = data["Time"].iloc[-1], values.iloc[-1]
ax.scatter([last_date], [last_value], color="#cb2633", zorder=3)
ax.annotate(f"{last_date:%B %Y}: {last_value:g}", (last_date, last_value),
            xytext=(-8, -20), textcoords="offset points", ha="right", color="#b51d2b", bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
ax.set(title="Google Trends: vibe coding, Worldwide (baseline downloaded 8 March 2026)",
       xlabel="Month", ylabel="Relative search interest (0–100)", ylim=(0, 105))
ax.grid(alpha=.2)
fig.savefig(root / "vibe_coding_google_trends_worldwide_2021_2026.png", dpi=180)
plt.close(fig)
