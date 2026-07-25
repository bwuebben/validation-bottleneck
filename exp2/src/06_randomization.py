"""Amendment 3: grammar randomization test (registered 2026-07-25 before running).

Draw R=10,000 expressions uniformly from the registered DSL grammar (seed 42), evaluate
each on the identical post-wall pipeline as 05_evaluate, and per model bootstrap the
null distribution of max-of-M t-statistics (B=10,000, M = registered distinct-M).
Data machinery below mirrors 05_evaluate.py verbatim (same sources, same rules).
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
R_POOL = 10_000
B_BOOT = 10_000
SEED = 42


def month_key(d: dt.date) -> tuple[int, int]:
    return (d.year, d.month)


def prev_month(k: tuple[int, int]) -> tuple[int, int]:
    return (k[0] - 1, 12) if k[1] == 1 else (k[0], k[1] - 1)


def wall_after(k: tuple[int, int], wall: str) -> bool:
    wy, wm = int(wall[:4]), int(wall[5:7])
    return k > (wy, wm)


def main() -> None:
    dp = get_dp()

    op = dp.factors.open_asset_pricing("op")
    ls = op.filter(pl.col("port") == "LS").select(["signalname", "date", "ret"])
    series: dict[str, dict] = {}
    for r in ls.iter_rows(named=True):
        series.setdefault(r["signalname"], {})[month_key(r["date"])] = float(r["ret"])
    primitives = sorted(series)
    print(f"{len(primitives)} primitives loaded")

    vix = dp.macro.fred("VIXCLS")
    vix_m: dict[tuple, float] = {}
    for r in vix.iter_rows(named=True):
        if r["VIXCLS"] is not None:
            vix_m[month_key(r["date"])] = float(r["VIXCLS"])
    vix_keys = sorted(vix_m)
    vix_above: dict[tuple, bool] = {}
    for i, k in enumerate(vix_keys):
        hist = [vix_m[kk] for kk in vix_keys[max(0, i - 59): i + 1]]
        if len(hist) >= 36:
            vix_above[k] = vix_m[k] > float(np.median(hist))

    tr = dp.macro.treasury_rates().sort("date")
    tr_rows = [(r["date"], float(r["rate_10y"]) - float(r["rate_3m"]))
               for r in tr.iter_rows(named=True)
               if r["rate_10y"] is not None and r["rate_3m"] is not None]
    spread_pos: dict[tuple, bool] = {}
    for k in {month_key(d) for d, _ in tr_rows}:
        eom = [s for d, s in tr_rows if month_key(d) <= k]
        if eom:
            last = [s for d, s in tr_rows if month_key(d) == k]
            spread_pos[k] = (last[-1] if last else eom[-1]) > 0

    ff = dp.factors.fama_french("3-factor")
    mkt: dict[tuple, float] = {}
    for r in ff.iter_rows(named=True):
        mr = r.get("Mkt-RF"); rf = r.get("RF")
        if mr is not None and rf is not None:
            mkt[month_key(r["date"])] = (float(mr) + float(rf)) / 100.0
    mkt_keys = sorted(mkt)
    mkt_up: dict[tuple, bool] = {}
    for i, k in enumerate(mkt_keys):
        if i >= 12:
            cum = np.prod([1 + mkt[kk] for kk in mkt_keys[i - 12: i]]) - 1
            mkt_up[k] = cum > 0

    def regime_true(regime: str, k: tuple) -> bool | None:
        km1 = prev_month(k)
        if regime.startswith("VIX"):
            v = vix_above.get(km1)
            if v is None:
                return None
            return v if regime == "VIX_ABOVE_TRAILING_MEDIAN" else not v
        if regime.startswith("TERM"):
            s = spread_pos.get(km1)
            if s is None:
                return None
            return s if regime == "TERM_SPREAD_POSITIVE" else not s
        u = mkt_up.get(k)
        if u is None:
            return None
        return u if regime == "MKT_TRAILING_12M_UP" else not u

    def strategy_series(terms, regime, vt) -> dict[tuple, float]:
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
            hist = [gated[kk] for kk in ks[max(0, i - 36): i]]
            s = 1.0
            if len(hist) >= 24:
                sd = float(np.std(hist, ddof=1))
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
        regime = rng.choice(REGIMES + ["NONE"])
        vt = rng.random() < 0.5
        pool.append((terms, regime, vt))
    print(f"pool drawn: {len(pool)} expressions (seed {SEED})")

    all_series = [strategy_series(t, rg, vt) for (t, rg, vt) in pool]

    out = {"spec": {"R": R_POOL, "B": B_BOOT, "seed": SEED}, "models": {}}
    nprng = np.random.default_rng(SEED)
    for model, wall in WALLS.items():
        window = [k for k in sorted({kk for s in series.values() for kk in s})
                  if wall_after(k, wall) and dt.date(k[0], k[1], 1) <= DATA_END]
        nwin = len(window)
        ts = []
        excluded = 0
        for strat in all_series:
            post = {k: v for k, v in strat.items() if wall_after(k, wall)
                    and dt.date(k[0], k[1], 1) <= DATA_END}
            if len(post) < MIN_MONTHS[model] or len(post) < 0.8 * nwin:
                excluded += 1
                continue
            r = np.array([post[k] for k in sorted(post)])
            sd = float(r.std(ddof=1))
            ts.append(float(r.mean()) / (sd / math.sqrt(len(r))) if sd > 0 else 0.0)
        ts = np.array(ts)
        M = M_MODEL[model]
        idx = nprng.integers(0, len(ts), size=(B_BOOT, M))
        maxima = ts[idx].max(axis=1)
        out["models"][model] = {
            "pool_evaluable": int(len(ts)), "pool_excluded": int(excluded),
            "pool_t_mean": float(ts.mean()), "pool_t_sd": float(ts.std(ddof=1)),
            "pool_t_max": float(ts.max()), "M": M,
            "null_max_mean": float(maxima.mean()),
            "null_max_p50": float(np.percentile(maxima, 50)),
            "null_max_p90": float(np.percentile(maxima, 90)),
            "null_max_p95": float(np.percentile(maxima, 95)),
            "null_max_p99": float(np.percentile(maxima, 99)),
        }
        print(f"{model}: pool {len(ts)} evaluable ({excluded} excluded), "
              f"null max-of-{M}: mean {maxima.mean():.2f}, p50 "
              f"{np.percentile(maxima,50):.2f}, p95 {np.percentile(maxima,95):.2f}")

    perf = pl.read_csv(DERIVED / "results_perf.csv")
    for model, wall in WALLS.items():
        sub = perf.filter((pl.col("model") == model) & pl.col("t").is_not_null())
        if not len(sub):
            continue
        mx = float(sub["t"].max())
        window = [k for k in sorted({kk for s in series.values() for kk in s})
                  if wall_after(k, wall) and dt.date(k[0], k[1], 1) <= DATA_END]
        ts = []
        for strat in all_series:
            post = {k: v for k, v in strat.items() if wall_after(k, wall)
                    and dt.date(k[0], k[1], 1) <= DATA_END}
            if len(post) < MIN_MONTHS[model] or len(post) < 0.8 * len(window):
                continue
            r = np.array([post[k] for k in sorted(post)])
            sd = float(r.std(ddof=1))
            ts.append(float(r.mean()) / (sd / math.sqrt(len(r))) if sd > 0 else 0.0)
        ts = np.array(ts)
        rng2 = np.random.default_rng(SEED + 1)
        maxima = ts[rng2.integers(0, len(ts), size=(B_BOOT, M_MODEL[model]))].max(axis=1)
        m = out["models"][model]
        m["realized_max_t"] = mx
        m["realized_max_percentile"] = float((maxima < mx).mean() * 100)
        print(f"{model}: realized max t={mx:.2f} -> percentile "
              f"{m['realized_max_percentile']:.1f} of null "
              f"(null p50={np.percentile(maxima,50):.2f})")

    (DERIVED / "randomization_test.json").write_text(json.dumps(out, indent=1))
    print("wrote derived/randomization_test.json")


if __name__ == "__main__":
    main()
