"""Grammar-null histogram data for the paper's model-vs-monkey figure.

Re-runs the registered Amendment 3 randomization pool (R=10,000 uniform draws from the
DSL, seed 42) and, in addition to the summary statistics already written by
06_randomization.py, exports the bootstrapped best-of-M distribution as a histogram per
model so the figure is reproducible from committed data.

Adds nothing to the registered analysis: same pool, same seed, same windows, same
pipeline. Output: derived/grammar_null_hist.json
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random
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
MIN_MONTHS = {"gpt-4-0613": 24, "gpt-4o-2024-08-06": 9,
              "meta-llama_Llama-3.3-70B-Instruct-Turbo": 8}
M_MODEL = {"gpt-4-0613": 330, "gpt-4o-2024-08-06": 318,
           "meta-llama_Llama-3.3-70B-Instruct-Turbo": 294}
REGIMES = ["VIX_ABOVE_TRAILING_MEDIAN", "VIX_BELOW_TRAILING_MEDIAN",
           "TERM_SPREAD_POSITIVE", "TERM_SPREAD_NEGATIVE",
           "MKT_TRAILING_12M_UP", "MKT_TRAILING_12M_DOWN"]
WEIGHTS = [-1.0, -0.5, 0.5, 1.0]
R_POOL, B_BOOT, SEED = 10_000, 10_000, 42
BINS = np.arange(1.0, 6.01, 0.25)


def month_key(d: dt.date) -> tuple[int, int]:
    return (d.year, d.month)


def prev_month(k):
    return (k[0] - 1, 12) if k[1] == 1 else (k[0], k[1] - 1)


def wall_after(k, wall: str) -> bool:
    return k > (int(wall[:4]), int(wall[5:7]))


def main() -> None:
    dp = get_dp()

    op = dp.factors.open_asset_pricing("op")
    ls = op.filter(pl.col("port") == "LS").select(["signalname", "date", "ret"])
    series: dict[str, dict] = {}
    for r in ls.iter_rows(named=True):
        series.setdefault(r["signalname"], {})[month_key(r["date"])] = float(r["ret"])
    primitives = sorted(series)

    vix = dp.macro.fred("VIXCLS")
    vix_m = {}
    for r in vix.iter_rows(named=True):
        if r["VIXCLS"] is not None:
            vix_m[month_key(r["date"])] = float(r["VIXCLS"])
    vk = sorted(vix_m)
    vix_above = {}
    for i, k in enumerate(vk):
        h = [vix_m[x] for x in vk[max(0, i - 59): i + 1]]
        if len(h) >= 36:
            vix_above[k] = vix_m[k] > float(np.median(h))

    tr = dp.macro.treasury_rates().sort("date")
    tr_rows = [(r["date"], float(r["rate_10y"]) - float(r["rate_3m"]))
               for r in tr.iter_rows(named=True)
               if r["rate_10y"] is not None and r["rate_3m"] is not None]
    spread_pos = {}
    for k in {month_key(d) for d, _ in tr_rows}:
        eom = [s for d, s in tr_rows if month_key(d) <= k]
        if eom:
            last = [s for d, s in tr_rows if month_key(d) == k]
            spread_pos[k] = (last[-1] if last else eom[-1]) > 0

    ff = dp.factors.fama_french("3-factor")
    mkt = {}
    for r in ff.iter_rows(named=True):
        a, b = r.get("Mkt-RF"), r.get("RF")
        if a is not None and b is not None:
            mkt[month_key(r["date"])] = (float(a) + float(b)) / 100.0
    mkeys = sorted(mkt)
    mkt_up = {}
    for i, k in enumerate(mkeys):
        if i >= 12:
            mkt_up[k] = np.prod([1 + mkt[x] for x in mkeys[i - 12: i]]) - 1 > 0

    def regime_true(regime, k):
        km1 = prev_month(k)
        if regime.startswith("VIX"):
            v = vix_above.get(km1)
            return None if v is None else (
                v if regime == "VIX_ABOVE_TRAILING_MEDIAN" else not v)
        if regime.startswith("TERM"):
            s = spread_pos.get(km1)
            return None if s is None else (
                s if regime == "TERM_SPREAD_POSITIVE" else not s)
        u = mkt_up.get(k)
        return None if u is None else (
            u if regime == "MKT_TRAILING_12M_UP" else not u)

    def strategy_series(terms, regime, vt):
        keys = None
        for p, _ in terms:
            s = set(series.get(p, {}))
            keys = s if keys is None else keys & s
        gated = {}
        for k in sorted(keys or []):
            v = sum(w * series[p][k] for p, w in terms)
            if regime != "NONE":
                rt = regime_true(regime, k)
                if rt is None:
                    continue
                v = v if rt else 0.0
            gated[k] = v
        if not vt:
            return gated
        ks = sorted(gated)
        out = {}
        target = 10.0 / math.sqrt(12)
        for i, k in enumerate(ks):
            h = [gated[x] for x in ks[max(0, i - 36): i]]
            s = 1.0
            if len(h) >= 24:
                sd = float(np.std(h, ddof=1))
                if sd > 1e-9:
                    s = min(target / sd, 5.0)
            out[k] = gated[k] * s
        return out

    rng = random.Random(SEED)
    pool = []
    for _ in range(R_POOL):
        kk = rng.choice([2, 3, 4])
        prims = rng.sample(primitives, kk)
        terms = tuple((p, rng.choice(WEIGHTS)) for p in prims)
        pool.append((terms, rng.choice(REGIMES + ["NONE"]), rng.random() < 0.5))
    all_series = [strategy_series(t, rg, vt) for (t, rg, vt) in pool]

    perf = pl.read_csv(DERIVED / "results_perf.csv")
    out = {"spec": {"R": R_POOL, "B": B_BOOT, "seed": SEED,
                    "bin_edges": [round(float(b), 4) for b in BINS]}, "models": {}}
    nprng = np.random.default_rng(SEED)
    for model, wall in WALLS.items():
        window = [k for k in sorted({x for s in series.values() for x in s})
                  if wall_after(k, wall) and dt.date(k[0], k[1], 1) <= DATA_END]
        nwin = len(window)
        ts = []
        for strat in all_series:
            post = {k: v for k, v in strat.items()
                    if wall_after(k, wall) and dt.date(k[0], k[1], 1) <= DATA_END}
            if len(post) < MIN_MONTHS[model] or len(post) < 0.8 * nwin:
                continue
            r = np.array([post[k] for k in sorted(post)])
            sd = float(r.std(ddof=1))
            ts.append(float(r.mean()) / (sd / math.sqrt(len(r))) if sd > 0 else 0.0)
        ts = np.array(ts)
        M = M_MODEL[model]
        maxima = ts[nprng.integers(0, len(ts), size=(B_BOOT, M))].max(axis=1)
        counts, _ = np.histogram(maxima, bins=BINS)
        sub = perf.filter((pl.col("model") == model) & pl.col("t").is_not_null())
        realized = float(sub["t"].max())
        # empirical CDF of the best-of-M null on a fine grid: the reported percentile
        # reads directly off this curve at t = realized max.
        grid = np.arange(1.5, 5.501, 0.05)
        cdf = [(round(float(g), 3), round(float((maxima < g).mean() * 100), 3))
               for g in grid]
        out["models"][model] = {
            "M": M, "pool_evaluable": int(len(ts)),
            "hist_counts": [int(c) for c in counts],
            "cdf_grid": cdf,
            "null_max_p50": float(np.percentile(maxima, 50)),
            "realized_max_t": round(realized, 3),
            "realized_percentile": round(float((maxima < realized).mean() * 100), 2),
        }
        print(f"{model}: null p50 {np.percentile(maxima,50):.2f} "
              f"realized {realized:.2f} pct {100*(maxima<realized).mean():.2f}")

    (DERIVED / "grammar_null_hist.json").write_text(json.dumps(out, indent=1))
    print("wrote derived/grammar_null_hist.json")


if __name__ == "__main__":
    main()
