"""Amendment 4: auxiliary-battery positive control (registered 2026-07-25).

Runs the registered Stage-3 aux test mechanics on 14 documented-true implications
(pinned in ANALYSIS_SPEC Amendment 4) over the three registered post-wall windows.
Data machinery mirrors 05_evaluate/06_randomization verbatim.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dp import get_dp  # noqa: E402

EXP2 = Path(__file__).resolve().parent.parent
DERIVED = EXP2 / "derived"
DATA_END = dt.date(2024, 12, 31)
WALLS = {"gpt-4-0613": "2022-04", "gpt-4o-2024-08-06": "2024-01",
         "meta-llama_Llama-3.3-70B-Instruct-Turbo": "2024-03"}

REGIME_ITEMS = [  # (id, primitive, regime, direction)
    ("R1", "Mom12m", "MKT_TRAILING_12M_DOWN", "lower"),
    ("R2", "Mom12m", "VIX_ABOVE_TRAILING_MEDIAN", "lower"),
    ("R3", "STreversal", "VIX_ABOVE_TRAILING_MEDIAN", "higher"),
    ("R4", "Illiquidity", "VIX_ABOVE_TRAILING_MEDIAN", "lower"),
    ("R5", "Size", "VIX_ABOVE_TRAILING_MEDIAN", "lower"),
]
CORR_ITEMS = [  # (id, primA, primB, bound, threshold)
    ("C1", "Mom12m", "BM", "above", 0.3),
    ("C2", "Mom12m", "LRreversal", "above", 0.3),
    ("C3", "BM", "BMdec", "above", 0.3),
    ("C4", "GP", "BM", "above", 0.3),
    ("C5", "Size", "Illiquidity", "above", 0.3),
    ("C6", "Accruals", "AssetGrowth", "above", 0.3),
]
SUB_ITEMS = [("S1", "Mom12m"), ("S2", "BM"), ("S3", "STreversal")]


def month_key(d: dt.date) -> tuple[int, int]:
    return (d.year, d.month)


def prev_month(k):
    return (k[0] - 1, 12) if k[1] == 1 else (k[0], k[1] - 1)


def wall_after(k, wall):
    return k > (int(wall[:4]), int(wall[5:7]))


def main() -> None:
    dp = get_dp()
    op = dp.factors.open_asset_pricing("op")
    ls = op.filter(pl.col("port") == "LS").select(["signalname", "date", "ret"])
    series: dict[str, dict] = {}
    for r in ls.iter_rows(named=True):
        series.setdefault(r["signalname"], {})[month_key(r["date"])] = float(r["ret"])

    vix = dp.macro.fred("VIXCLS")
    vix_m = {}
    for r in vix.iter_rows(named=True):
        if r["VIXCLS"] is not None:
            vix_m[month_key(r["date"])] = float(r["VIXCLS"])
    vix_keys = sorted(vix_m)
    vix_above = {}
    for i, k in enumerate(vix_keys):
        hist = [vix_m[kk] for kk in vix_keys[max(0, i - 59): i + 1]]
        if len(hist) >= 36:
            vix_above[k] = vix_m[k] > float(np.median(hist))

    ff = dp.factors.fama_french("3-factor")
    mkt = {}
    for r in ff.iter_rows(named=True):
        mr = r.get("Mkt-RF"); rf = r.get("RF")
        if mr is not None and rf is not None:
            mkt[month_key(r["date"])] = (float(mr) + float(rf)) / 100.0
    mkt_keys = sorted(mkt)
    mkt_up = {}
    for i, k in enumerate(mkt_keys):
        if i >= 12:
            mkt_up[k] = np.prod([1 + mkt[kk] for kk in mkt_keys[i - 12: i]]) - 1 > 0

    def regime_true(regime, k):
        km1 = prev_month(k)
        if regime.startswith("VIX"):
            v = vix_above.get(km1)
            return None if v is None else (v if regime == "VIX_ABOVE_TRAILING_MEDIAN" else not v)
        u = mkt_up.get(k)
        return None if u is None else (u if regime == "MKT_TRAILING_12M_UP" else not u)

    rows = []
    for model, wall in WALLS.items():
        def post_of(prim):
            return {k: v for k, v in series[prim].items()
                    if wall_after(k, wall) and dt.date(k[0], k[1], 1) <= DATA_END}

        for iid, prim, regime, direction in REGIME_ITEMS:
            post = post_of(prim)
            ins, outs = [], []
            for k, v in post.items():
                rt = regime_true(regime, k)
                if rt is True:
                    ins.append(v)
                elif rt is False:
                    outs.append(v)
            if len(ins) >= 6 and len(outs) >= 6:
                diff = float(np.mean(ins) - np.mean(outs))
                ok = (diff > 0) == (direction == "higher")
                rows.append({"model": model, "id": iid, "type": "regime",
                             "tested": True, "pass": bool(ok), "detail": round(diff, 3)})
            else:
                rows.append({"model": model, "id": iid, "type": "regime",
                             "tested": False, "pass": None,
                             "detail": f"in={len(ins)},out={len(outs)}"})

        for iid, pa, pb, bound, thr in CORR_ITEMS:
            A, B = post_of(pa), post_of(pb)
            joint = sorted(set(A) & set(B))
            if len(joint) >= 12:
                c = abs(float(np.corrcoef([A[k] for k in joint],
                                          [B[k] for k in joint])[0, 1]))
                ok = (c < thr) == (bound == "below")
                rows.append({"model": model, "id": iid, "type": "corr",
                             "tested": True, "pass": bool(ok), "detail": round(c, 3)})
            else:
                rows.append({"model": model, "id": iid, "type": "corr",
                             "tested": False, "pass": None, "detail": f"joint={len(joint)}"})

        for iid, prim in SUB_ITEMS:
            post = post_of(prim)
            ks = sorted(post)
            half = len(ks) // 2
            if half >= 3:
                m1 = float(np.mean([post[k] for k in ks[:half]]))
                m2 = float(np.mean([post[k] for k in ks[half:]]))
                rows.append({"model": model, "id": iid, "type": "subperiod",
                             "tested": True, "pass": bool(m1 > 0 and m2 > 0),
                             "detail": f"{m1:.2f}/{m2:.2f}"})

    df = pl.DataFrame(rows)
    df.write_csv(DERIVED / "aux_positive_control.csv")
    tested = df.filter(pl.col("tested"))
    print(tested)
    for typ in ("regime", "corr", "subperiod"):
        sub = tested.filter(pl.col("type") == typ)
        n, p = len(sub), int(sub["pass"].sum())
        print(f"{typ:10s}: {p}/{n} pass ({100*p/max(n,1):.0f}%)")
    n, p = len(tested), int(tested["pass"].sum())
    print(f"{'TOTAL':10s}: {p}/{n} pass ({100*p/max(n,1):.0f}%)")


if __name__ == "__main__":
    main()
