"""Build the Exp 2 DSL primitive library from OSAP SignalDoc (via data_platform).

Emits:
  ../primitives/primitives.json   -- full metadata per primitive (acronym, year, journal,
                                     sample window, description) for analysis
  ../primitives/library_block.txt -- the compact library text embedded verbatim in the
                                     generation prompts (acronym + short description).

Contamination discipline: every primitive is a published predictor with publication
year <= 2016 (OSAP), so the library text contains no post-cutoff information for any
panel model (earliest cutoff: Sept 2021).

data_platform returns polars DataFrames.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl

from _dp import get_dp

OUT = Path(__file__).resolve().parent.parent / "primitives"
OUT.mkdir(exist_ok=True)

MAX_DESC_CHARS = 70  # keep the prompt block small enough for gpt-4-0613's 8k context


def shorten(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= MAX_DESC_CHARS:
        return text
    return text[: MAX_DESC_CHARS - 1].rstrip() + "…"


def main() -> None:
    dp = get_dp()
    doc = dp.signals.open_asset_pricing_doc()
    print("SignalDoc columns:", list(doc.columns))
    print("Cat.Signal counts:\n", doc["Cat.Signal"].value_counts())

    pred = doc.filter(pl.col("Cat.Signal") == "Predictor")
    print(f"Predictors: {len(pred)}")

    records = []
    for r in pred.iter_rows(named=True):
        year_raw = r.get("Year")
        try:
            year = int(year_raw) if year_raw is not None and str(year_raw).strip() else None
        except (TypeError, ValueError):
            year = None
        records.append(
            {
                "acronym": r["Acronym"],
                "authors": str(r.get("Authors") or ""),
                "year": year,
                "journal": str(r.get("Journal") or ""),
                "sample_start": r.get("SampleStartYear"),
                "sample_end": r.get("SampleEndYear"),
                "description": re.sub(r"\s+", " ", str(r.get("LongDescription") or "")).strip(),
            }
        )
    records.sort(key=lambda x: x["acronym"].lower())

    (OUT / "primitives.json").write_text(json.dumps(records, indent=1, default=str))
    print(f"wrote primitives.json ({len(records)} primitives)")

    lines = [f"{r['acronym']}: {shorten(r['description'])}" for r in records]
    block = "\n".join(lines)
    (OUT / "library_block.txt").write_text(block)
    ntok_est = len(block) / 4
    print(f"wrote library_block.txt ({len(lines)} lines, ~{ntok_est:.0f} tokens est.)")

    years = [r["year"] for r in records if r["year"]]
    print(f"publication years: {min(years)}–{max(years)} (must be <= earliest cutoff 2021)")


if __name__ == "__main__":
    main()
