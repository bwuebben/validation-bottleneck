"""Test (a) rework statistics (post-critique revision, 2026-07-25).

Two replacements for the ceiling-limited enrichment ratio, computed from the
already-archived judge output (no new API calls):

1. Match-grade split: post-date share among EXACT matches vs CLOSE matches, per
   model x vintage. Contamination reproduces constructions (exact); convergent
   rediscovery arrives at themes (close/related). A post-date tilt concentrated in
   exact matches survives the compliance objection.
2. Forward publication lag: for each post-date matched proposal, years from the
   roleplay vintage to the counterpart's publication, against the library's own
   forward-publication schedule from that vintage (the null: matching the post-date
   library uniformly).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dp import get_dp  # noqa: E402

EXP1 = Path(__file__).resolve().parent.parent
DERIVED = EXP1 / "derived"


def main() -> None:
    dp = get_dp()
    doc = dp.signals.open_asset_pricing_doc()
    pred = doc.filter(pl.col("Cat.Signal") == "Predictor")
    year = {}
    for r in pred.iter_rows(named=True):
        try:
            year[r["Acronym"]] = int(r["Year"])
        except (TypeError, ValueError):
            pass

    matches = [json.loads(l) for l in (DERIVED / "matches.jsonl").read_text().splitlines()]
    props = {}
    for l in (DERIVED / "proposals.jsonl").read_text().splitlines():
        p = json.loads(l)
        props[(p["file"], p["idx"])] = p

    rows = []
    for m in matches:
        for d in m["matches"]:
            p = props.get((m["file"], d["index"]))
            if p is None:
                continue
            acr = d.get("match")
            rows.append({
                "model": p["model"], "vintage": p["vintage"],
                "strength": d.get("strength"), "acr": acr,
                "pub_year": year.get(acr) if acr else None,
            })
    df = pl.DataFrame(rows)

    # 1. grade split: post-date share among exact vs close matches
    out1 = []
    for (model, vintage, strength), g in df.filter(
            pl.col("strength").is_in(["exact", "close"]) & pl.col("pub_year").is_not_null()
    ).group_by(["model", "vintage", "strength"]):
        post = (g["pub_year"] > vintage).sum()
        out1.append({"model": model, "vintage": vintage, "strength": strength,
                     "n_matched": len(g), "n_post": int(post),
                     "share_post": round(post / len(g), 3)})
    g1 = pl.DataFrame(out1).sort(["model", "vintage", "strength"])
    g1.write_csv(DERIVED / "grade_split.csv")

    # pooled-over-vintage grade split per model
    print("=== post-date share among matches, by grade (pooled vintages) ===")
    for model in sorted(df["model"].unique().to_list()):
        line = [model]
        for strength in ("exact", "close"):
            g = df.filter((pl.col("model") == model) & (pl.col("strength") == strength)
                          & pl.col("pub_year").is_not_null())
            if len(g):
                post = int((g["pub_year"] > g["vintage"]).sum())
                line.append(f"{strength}: {post}/{len(g)} ({100*post/len(g):.0f}%)")
        print("  " + " | ".join(str(x) for x in line))

    # 2. forward publication lag vs library schedule
    lib_years = sorted(year.values())
    out2 = []
    print("\n=== forward publication lag of post-date exact/close matches ===")
    for (model, vintage), g in df.filter(
            pl.col("strength").is_in(["exact", "close"]) & pl.col("pub_year").is_not_null()
    ).group_by(["model", "vintage"]):
        post = g.filter(pl.col("pub_year") > vintage)
        if not len(post):
            continue
        lags = (post["pub_year"] - vintage).to_list()
        lib_fwd = [y - vintage for y in lib_years if y > vintage]
        out2.append({"model": model, "vintage": vintage, "n_post": len(lags),
                     "med_lag": float(np.median(lags)),
                     "lib_med_lag": float(np.median(lib_fwd)),
                     "mean_lag": round(float(np.mean(lags)), 2),
                     "lib_mean_lag": round(float(np.mean(lib_fwd)), 2)})
    g2 = pl.DataFrame(out2).sort(["model", "vintage"])
    g2.write_csv(DERIVED / "publication_lag.csv")
    print(g2)


if __name__ == "__main__":
    main()
