"""Download a bounded Eurostat series, preserve flags, validate, and save provenance.

Run from the repository root. Public endpoint; no account or API key required.
The source is monthly net electricity generation, not energy consumption.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_cb_pem"


def query_url(geo="AT", year=2023, source="RA300"):
    return BASE + "?" + urlencode({"lang": "en", "freq": "M", "geo": geo,
        "siec": source, "unit": "GWH", "sinceTimePeriod": f"{year}-01",
        "untilTimePeriod": f"{year}-12"})


def decode_series(payload):
    """Decode JSON-stat positional values without assuming time is the last axis."""
    if payload.get("class") != "dataset" or "error" in payload:
        raise ValueError("Expected a Eurostat JSON-stat dataset, not an error/page")
    ids, sizes = payload["id"], payload["size"]
    required = {"freq", "siec", "unit", "geo", "time"}
    if len(ids) != len(required) or len(ids) != len(sizes) or set(ids) != required:
        raise ValueError(f"Unexpected dimensions: {ids}")
    axes = []
    for name, size in zip(ids, sizes):
        index = payload["dimension"][name]["category"]["index"]
        if isinstance(index, dict) and sorted(index.values()) != list(range(size)):
            raise ValueError(f"Invalid category positions: {name}")
        labels = index if isinstance(index, list) else [k for k, _ in sorted(index.items(), key=lambda x: x[1])]
        if len(labels) != size:
            raise ValueError(f"Dimension length mismatch: {name}")
        axes.append(labels)
    def get_at(container, i, default=None):
        return container.get(str(i), default) if isinstance(container, dict) else (container[i] if i < len(container) else default)
    rows = []
    for i, labels in enumerate(itertools.product(*axes)):
        row = dict(zip(ids, labels))
        value = get_at(payload.get("value", {}), i)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("Non-finite or non-numeric generation")
        rows.append({"geo": row["geo"], "source": row["siec"], "month": row["time"],
            "unit": row["unit"], "freq": row["freq"], "generation_gwh": value,
            "status_flag": get_at(payload.get("status", {}), i, "") or ""})
    if any(r["unit"] != "GWH" or r["freq"] != "M" for r in rows):
        raise ValueError("Expected monthly GWh")
    keys = [(r["geo"], r["source"], r["month"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate country/source/month")
    return sorted(rows, key=lambda r: (r["geo"], r["source"], r["month"]))


def download(output, geo="AT", year=2023, source="RA300"):
    url = query_url(geo, year, source)
    request = Request(url, headers={"User-Agent": "EAGE-course-data-tutorial/1.0"})
    with urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"Download returned HTTP {response.status}")
        raw = response.read(5_000_001)
        if len(raw) > 5_000_000:
            raise ValueError("Unexpectedly large download; narrow the query")
    data = json.loads(raw)
    rows = decode_series(data)
    if len(rows) != 12 or {r["month"] for r in rows} != {f"{year}-{m:02}" for m in range(1, 13)}:
        raise ValueError("Expected twelve monthly rows; inspect coverage before proceeding")
    if any(r["geo"] != geo or r["source"] != source for r in rows):
        raise ValueError("Response does not match requested country/source")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_bytes(raw)
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    provenance = {"provider": "Eurostat", "dataset": "nrg_cb_pem", "url": url,
        "title": data["label"], "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "provider_updated": data.get("updated"), "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows), "geo": geo, "source": source, "unit": "GWH", "frequency": "M",
        "source_label": data["dimension"]["siec"]["category"]["label"].get(source),
        "status_labels": data.get("extension", {}).get("status", {}),
        "reuse": "https://ec.europa.eu/eurostat/about-us/policies/copyright"}
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} rows to {output.with_suffix('.csv')}")
    print(f"Observed sum: {sum(r['generation_gwh'] for r in rows if r['generation_gwh'] is not None):.3f} GWh")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Output prefix; defaults to a country/source/year folder under outputs")
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--source", default="RA300")
    args = parser.parse_args()
    output = args.output or Path("outputs") / f"eurostat_{args.geo}_{args.source}_{args.year}"
    download(output, args.geo, args.year, args.source)
