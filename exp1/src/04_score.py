"""Exp 1 scoring — implements ANALYSIS_SPEC.md Stages 2-4 EXACTLY. Run only after
matching is complete. Emits derived/results_a.csv, derived/results_b.txt,
derived/effective_m.csv, derived/audit_sample.jsonl and prints a summary.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

EXP1 = Path(__file__).resolve().parent.parent
DERIVED = EXP1 / "derived"
sys.path.insert(0, str(EXP1.parent / "exp2" / "src"))
from _dp import get_dp  # noqa: E402

MATCHED = {"exact", "close"}


def load_matches() -> dict[str, dict]:
    """proposal_id -> {match, strength}."""
    out = {}
    for line in (DERIVED / "matches.jsonl").read_text().splitlines():
        r = json.loads(line)
        for m in r["matches"]:
            pid = f'{r["file"].split("/")[1]}|{Path(r["file"]).stem}|{m["index"]}'
            out[pid] = {"match": m.get("match"), "strength": m.get("strength", "none"),
                        "reason": m.get("reason", "")}
    return out


def ols_clustered(y, X, clusters):
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    cl = defaultdict(list)
    for i, c in enumerate(clusters):
        cl[c].append(i)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for idx in cl.values():
        Xg, eg = X[idx], e[idx]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    G, n, k = len(cl), len(y), X.shape[1]
    dfc = (G / (G - 1)) * ((n - 1) / (n - k))
    V = dfc * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    return beta, se


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    proposals = [json.loads(l) for l in (DERIVED / "proposals.jsonl").read_text().splitlines()]
    matches = load_matches()
    print(f"{len(proposals)} proposals, {len(matches)} judged")
    missing = [p["proposal_id"] for p in proposals if p["proposal_id"] not in matches]
    if missing:
        print(f"⚠ {len(missing)} proposals without judgments (parse failures) — excluded, flagged")

    dp = get_dp()
    doc = dp.signals.open_asset_pricing_doc()
    pred = doc.filter(pl.col("Cat.Signal") == "Predictor")
    meta = {}
    for r in pred.iter_rows(named=True):
        try:
            yr = int(r["Year"])
        except (TypeError, ValueError):
            yr = None
        try:
            cites = float(r.get("GScholarCites202509") or 0)
        except (TypeError, ValueError):
            cites = 0.0
        meta[r["Acronym"]] = {"year": yr, "cites": cites}

    op = dp.factors.open_asset_pricing("op")
    ls = op.filter(pl.col("port") == "LS").select(["signalname", "date", "ret"])
    print(f"OSAP LS panel: {len(ls)} rows, {ls['signalname'].n_unique()} signals, "
          f"{ls['date'].min()} → {ls['date'].max()}")

    def ann_mean(sig: str, start=None, end=None) -> float | None:
        q = ls.filter(pl.col("signalname") == sig)
        if start is not None:
            q = q.filter(pl.col("date") >= start)
        if end is not None:
            q = q.filter(pl.col("date") <= end)
        if len(q) < 12:
            return None
        return float(q["ret"].mean()) * 12

    import datetime as dt

    # ---------- Stage 2: test (a) ----------
    rows_a = []
    lib_years = [m["year"] for m in meta.values() if m["year"]]
    for model in sorted({p["model"] for p in proposals}):
        for v in sorted({p["vintage"] for p in proposals}):
            ps = [p for p in proposals if p["model"] == model and p["vintage"] == v
                  and p["proposal_id"] in matches]
            n = len(ps)
            post = pre = rel = none = 0
            for p in ps:
                m = matches[p["proposal_id"]]
                if m["strength"] in MATCHED and m["match"] in meta and meta[m["match"]]["year"]:
                    if meta[m["match"]]["year"] > v:
                        post += 1
                    else:
                        pre += 1
                elif m["strength"] == "related":
                    rel += 1
                else:
                    none += 1
            matched = post + pre
            base = sum(1 for y in lib_years if y > v) / len(lib_years)
            share_post_matched = post / matched if matched else np.nan
            rows_a.append({
                "model": model, "vintage": v, "n": n, "matched": matched,
                "post_date": post, "pre_date": pre, "related": rel, "none": none,
                "share_post_of_all": post / n if n else np.nan,
                "share_post_of_matched": share_post_matched,
                "library_base_rate": base,
                "enrichment": share_post_matched / base if matched and base else np.nan,
            })
    dfa = pl.DataFrame(rows_a)
    dfa.write_csv(DERIVED / "results_a.csv")
    print("\n=== TEST (a): regurgitation by vintage ===")
    print(dfa)

    # ---------- Stage 3: test (b) ----------
    counts = defaultdict(int)
    for p in proposals:
        m = matches.get(p["proposal_id"])
        if m and m["strength"] in MATCHED and m["match"] in meta:
            counts[(p["model"], p["vintage"], m["match"])] += 1

    panel = []
    for model in sorted({p["model"] for p in proposals}):
        for v in sorted({p["vintage"] for p in proposals}):
            vstart = dt.date(v, 2, 1)
            vend = dt.date(2024, 12, 31)
            for acro, mm in meta.items():
                if not mm["year"] or mm["year"] <= v:
                    continue
                fut = ann_mean(acro, start=vstart, end=vend)
                if fut is None:
                    continue
                pre_r = ann_mean(acro, end=dt.date(v - 1, 12, 31))
                panel.append({
                    "model": model, "vintage": v, "acronym": acro,
                    "count": counts.get((model, v, acro), 0),
                    "fut": fut,
                    "pre": pre_r if pre_r is not None else 0.0,
                    "pre_missing": 1.0 if pre_r is None else 0.0,
                    "log_cites": np.log1p(mm["cites"]),
                    "pub_year": float(mm["year"]),
                })
    dfb = pl.DataFrame(panel)
    print(f"\n=== TEST (b) panel: {len(dfb)} predictor×vintage×model cells, "
          f"{int(dfb['count'].sum())} matched post-date proposals ===")

    fe_keys = sorted({(r["model"], r["vintage"]) for r in panel})
    fe_idx = {k: i for i, k in enumerate(fe_keys)}
    Xcols = ["fut", "pre", "pre_missing", "log_cites", "pub_year"]
    Xbase = np.array([[r[c] for c in Xcols] for r in panel])
    FE = np.zeros((len(panel), len(fe_keys)))
    for i, r in enumerate(panel):
        FE[i, fe_idx[(r["model"], r["vintage"])]] = 1.0
    X = np.hstack([Xbase, FE])
    y = np.array([float(r["count"]) for r in panel])
    clusters = [r["acronym"] for r in panel]
    beta, se = ols_clustered(y, X, clusters)

    lines = ["=== TEST (b) primary: OLS count ~ fut + controls, model×vintage FE, "
             "SE clustered by predictor ==="]
    for j, c in enumerate(Xcols):
        t = beta[j] / se[j] if se[j] > 0 else np.nan
        lines.append(f"  {c:12s}  beta={beta[j]:+.5f}  se={se[j]:.5f}  t={t:+.2f}")
    lines.append(f"  N={len(y)}  clusters={len(set(clusters))}  FE cells={len(fe_keys)}")

    lines.append("\n=== TEST (b) rank version: Spearman(count, fut) within model×vintage ===")
    zs = []
    for k in fe_keys:
        sub = [r for r in panel if (r["model"], r["vintage"]) == k]
        rho = spearman(np.array([r["count"] for r in sub]),
                       np.array([r["fut"] for r in sub]))
        lines.append(f"  {k[0][:20]:20s} v{k[1]}: rho={rho:+.3f} (n={len(sub)})")
        if not np.isnan(rho) and len(sub) > 3:
            z = np.arctanh(max(min(rho, 0.999), -0.999)) * np.sqrt(len(sub) - 3)
            zs.append(z)
    if zs:
        zc = sum(zs) / np.sqrt(len(zs))
        lines.append(f"  Fisher-combined z = {zc:+.2f}")

    out_b = "\n".join(lines)
    (DERIVED / "results_b.txt").write_text(out_b + "\n")
    print("\n" + out_b)

    # ---------- Stage 4: effective-M ----------
    rows_m = []
    for model in sorted({p["model"] for p in proposals}):
        for v in sorted({p["vintage"] for p in proposals}):
            ids = [(p, matches.get(p["proposal_id"])) for p in proposals
                   if p["model"] == model and p["vintage"] == v]
            macros = [m["match"] for _, m in ids if m and m["strength"] in MATCHED and m["match"]]
            unnames = {p["name"].strip().lower() for p, m in ids
                       if not (m and m["strength"] in MATCHED)}
            cnt = defaultdict(int)
            for a in macros:
                cnt[a] += 1
            top5 = sum(sorted(cnt.values(), reverse=True)[:5]) / len(macros) if macros else np.nan
            rows_m.append({"model": model, "vintage": v,
                           "matched": len(macros), "distinct_matched": len(cnt),
                           "unmatched_distinct_names": len(unnames),
                           "top5_share": top5})
    dfm = pl.DataFrame(rows_m)
    dfm.write_csv(DERIVED / "effective_m.csv")
    print("\n=== Stage 4: effective-M ===")
    print(dfm)

    # ---------- audit sample ----------
    rng = random.Random(7)
    judged = [p for p in proposals if p["proposal_id"] in matches]
    sample = rng.sample(judged, min(40, len(judged)))
    with (DERIVED / "audit_sample.jsonl").open("w") as f:
        for p in sample:
            f.write(json.dumps({**{k: p[k] for k in ("proposal_id", "model", "vintage",
                                                     "name", "construction")},
                                **matches[p["proposal_id"]]}) + "\n")
    print(f"\naudit sample written: {len(sample)} decisions → derived/audit_sample.jsonl")


if __name__ == "__main__":
    main()
