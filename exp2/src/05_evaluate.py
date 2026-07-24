"""Exp 2 evaluation — implements ANALYSIS_SPEC.md EXACTLY (committed 64b4578d, before
any post-wall return was constructed). Emits derived/eval_universe.csv,
derived/results_perf.csv, derived/results_summary.txt.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

EXP2 = Path(__file__).resolve().parent.parent
CORPUS = EXP2 / "corpus"
DERIVED = EXP2 / "derived"
DERIVED.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dp import get_dp  # noqa: E402

WALLS = {"gpt-4-0613": "2022-04", "gpt-4o-2024-08-06": "2024-01",
         "meta-llama_Llama-3.3-70B-Instruct-Turbo": "2024-03"}  # 4.1 not evaluable v1; llama per Amendment 1
DATA_END = dt.date(2024, 12, 31)
MIN_MONTHS = {"gpt-4-0613": 24, "gpt-4o-2024-08-06": 9,
              "meta-llama_Llama-3.3-70B-Instruct-Turbo": 8}
WEIGHTS = {-1.0, -0.5, 0.5, 1.0}
REGIMES = {"VIX_ABOVE_TRAILING_MEDIAN", "VIX_BELOW_TRAILING_MEDIAN",
           "TERM_SPREAD_POSITIVE", "TERM_SPREAD_NEGATIVE",
           "MKT_TRAILING_12M_UP", "MKT_TRAILING_12M_DOWN"}
AUX_TYPES = {"regime_interaction", "size_segment", "international",
             "correlation", "subperiod_consistency"}
Q_FDR = 0.05

try:
    from scipy import stats as sps
    def t_sf(t, df):
        return float(sps.t.sf(t, df))
except ImportError:
    def t_sf(t, df):
        return float(1 - 0.5 * (1 + math.erf(t / math.sqrt(2))))


def strip_fences(c: str) -> str:
    c = c.strip()
    c = re.sub(r"^```(json)?\s*", "", c)
    c = re.sub(r"\s*```$", "", c)
    return c


def month_key(d: dt.date) -> tuple[int, int]:
    return (d.year, d.month)


def prev_month(k: tuple[int, int]) -> tuple[int, int]:
    y, m = k
    return (y - 1, 12) if m == 1 else (y, m - 1)


def wall_after(k: tuple[int, int], wall: str) -> bool:
    return k > (int(wall[:4]), int(wall[5:7]))


def main() -> None:
    dp = get_dp()

    # ---------------- Stage 0: universe ----------------
    lib = {l.split(":")[0] for l in (EXP2 / "primitives" / "library_block.txt").read_text().splitlines()}
    hyps = []
    for f in sorted(CORPUS.glob("*/v*_seed*.json")):
        if f.name.startswith("smoke_"):
            continue
        env = json.loads(f.read_text())
        model = env["meta"]["model_requested"].replace("/", "_")
        content = env["response"]["choices"][0]["message"]["content"]
        try:
            j = json.loads(content)
        except json.JSONDecodeError:
            try:
                j = json.loads(strip_fences(content))
            except json.JSONDecodeError:
                j = {"hypotheses": []}  # malformed response: excluded at Stage 0 per spec
        for i, h in enumerate(j.get("hypotheses", [])):
            expr = h.get("expression", {}) or {}
            terms = expr.get("terms", []) or []
            valid = (2 <= len(terms) <= 4
                     and all(isinstance(t, dict) and t.get("primitive") in lib
                             and float(t.get("weight", 0)) in WEIGHTS for t in terms)
                     and (expr.get("regime") in REGIMES or expr.get("regime") in (None, "null")))
            aux = h.get("auxiliary_predictions", []) or []
            aux_types = {a.get("type") for a in aux if isinstance(a, dict)} & AUX_TYPES
            compliant = len(aux) >= 3 and len(aux_types) >= 2 and bool(h.get("mechanism"))
            sig = None
            if valid:
                sig = (tuple(sorted((t["primitive"], float(t["weight"])) for t in terms)),
                       expr.get("regime") or "NONE", bool(expr.get("vol_target")))
            hyps.append({"model": model, "file": f.name, "idx": i, "valid": valid,
                         "compliant": compliant, "sig": sig, "h": h})

    uni_rows, universe = [], {}
    for model in WALLS:
        mh = [x for x in hyps if x["model"] == model]
        valid = [x for x in mh if x["valid"]]
        sigs = defaultdict(list)
        for x in valid:
            sigs[x["sig"]].append(x)
        universe[model] = sigs
        uni_rows.append({"model": model, "hypotheses": len(mh), "valid": len(valid),
                         "invalid": len(mh) - len(valid),
                         "compliant": sum(1 for x in mh if x["compliant"]),
                         "distinct_M": len(sigs),
                         "dup_rate": 1 - len(sigs) / len(valid) if valid else None})
    n41 = sum(1 for x in hyps if x["model"].startswith("gpt-4.1"))
    pl.DataFrame(uni_rows).write_csv(DERIVED / "eval_universe.csv")
    print(pl.DataFrame(uni_rows))
    print(f"gpt-4.1: {n41} hypotheses NOT EVALUABLE in v1 (wall 2025-02 > OSAP end)")

    # ---------------- Stage 1: data ----------------
    op = dp.factors.open_asset_pricing("op")
    ls = op.filter(pl.col("port") == "LS").select(["signalname", "date", "ret"])
    series: dict[str, dict] = {}
    for r in ls.iter_rows(named=True):
        series.setdefault(r["signalname"], {})[month_key(r["date"])] = float(r["ret"])

    vix = dp.macro.fred("VIXCLS")
    vix_m: dict[tuple, float] = {}
    for r in vix.iter_rows(named=True):
        if r["VIXCLS"] is not None:
            vix_m[month_key(r["date"])] = float(r["VIXCLS"])  # last obs of month wins
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
            mkt_up[k] = cum > 0  # trailing 12m through t-1

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
        u = mkt_up.get(k)  # mkt_up[k] already uses months through t-1
        if u is None:
            return None
        return u if regime == "MKT_TRAILING_12M_UP" else not u

    def combo_series(sig) -> dict[tuple, float]:
        terms, _, _ = sig
        keys = None
        for p, _ in terms:
            s = set(series.get(p, {}))
            keys = s if keys is None else keys & s
        out = {}
        for k in sorted(keys or []):
            out[k] = sum(w * series[p][k] for p, w in terms)
        return out

    def strategy_series(sig) -> dict[tuple, float]:
        terms, regime, vt = sig
        raw = combo_series(sig)
        gated = {}
        for k, v in raw.items():
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
        target = 10.0 / math.sqrt(12)  # % monthly for 10% annualized
        for i, k in enumerate(ks):
            hist = [gated[kk] for kk in ks[max(0, i - 36): i]]
            s = 1.0
            if len(hist) >= 24:
                sd = float(np.std(hist, ddof=1))
                if sd > 1e-9:
                    s = min(target / sd, 5.0)
            out[k] = gated[k] * s
        return out

    # ---------------- Stages 2+3 ----------------
    ff5 = dp.factors.fama_french("5-factor")
    mom = dp.factors.fama_french("momentum")
    ff6: dict[tuple, list[float]] = {}
    momd = {month_key(r["date"]): float(r[mom.columns[1]]) for r in mom.iter_rows(named=True)
            if r[mom.columns[1]] is not None}
    for r in ff5.iter_rows(named=True):
        k = month_key(r["date"])
        if k in momd and all(r.get(c) is not None for c in ("Mkt-RF", "SMB", "HML", "RMW", "CMA")):
            ff6[k] = [float(r["Mkt-RF"]), float(r["SMB"]), float(r["HML"]),
                      float(r["RMW"]), float(r["CMA"]), momd[k]]

    perf_rows = []
    summary = []
    for model, wall in WALLS.items():
        sigs = universe[model]
        window = [k for k in sorted({kk for s in series.values() for kk in s})
                  if wall_after(k, wall) and dt.date(k[0], k[1], 1) <= DATA_END]
        nwin = len(window)
        results = []
        for sig, insts in sigs.items():
            strat = strategy_series(sig)
            post = {k: v for k, v in strat.items() if wall_after(k, wall)
                    and dt.date(k[0], k[1], 1) <= DATA_END}
            if len(post) < MIN_MONTHS[model] or len(post) < 0.8 * nwin:
                results.append({"sig": sig, "insts": insts, "evaluable": False})
                continue
            r = np.array([post[k] for k in sorted(post)])
            mean, sd, n = float(r.mean()), float(r.std(ddof=1)), len(r)
            tstat = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
            p = t_sf(tstat, n - 1)

            h0 = insts[0]["h"]
            aux_res = []
            raw = combo_series(sig)
            post_raw = {k: v for k, v in raw.items() if k in post}
            for a in h0.get("auxiliary_predictions", []) or []:
                if not isinstance(a, dict):
                    continue
                typ = a.get("type")
                if typ == "regime_interaction" and a.get("regime") in REGIMES:
                    ins, outs = [], []
                    for k, v in post_raw.items():
                        rt = regime_true(a["regime"], k)
                        if rt is True:
                            ins.append(v)
                        elif rt is False:
                            outs.append(v)
                    if len(ins) >= 6 and len(outs) >= 6:
                        diff = np.mean(ins) - np.mean(outs)
                        want = a.get("direction") == "higher"
                        aux_res.append(("regime", bool((diff > 0) == want)))
                elif typ == "correlation" and a.get("primitive") in series:
                    ps = series[a["primitive"]]
                    joint = [k for k in post_raw if k in ps]
                    if len(joint) >= 12:
                        c = abs(float(np.corrcoef([post_raw[k] for k in joint],
                                                  [ps[k] for k in joint])[0, 1]))
                        thr = float(a.get("threshold", 0.3) or 0.3)
                        want_below = a.get("bound") == "below"
                        aux_res.append(("corr", bool((c < thr) == want_below)))
                elif typ == "subperiod_consistency":
                    ks = sorted(post)
                    half = len(ks) // 2
                    m1 = np.mean([post[k] for k in ks[:half]])
                    m2 = np.mean([post[k] for k in ks[half:]])
                    aux_res.append(("subperiod", bool(m1 > 0 and m2 > 0)))
            aux_tested = len(aux_res)
            aux_pass = sum(1 for _, ok in aux_res if ok)

            alpha_t = None
            if model == "gpt-4-0613" and n >= 24:
                ks = sorted(post)
                X = np.array([[1.0] + ff6[k] for k in ks if k in ff6])
                yv = np.array([post[k] for k in ks if k in ff6])
                if len(yv) >= 24:
                    b, res_, rank_, _ = np.linalg.lstsq(X, yv, rcond=None)
                    e = yv - X @ b
                    dof = len(yv) - X.shape[1]
                    s2 = float(e @ e) / dof
                    Vb = s2 * np.linalg.inv(X.T @ X)
                    alpha_t = float(b[0] / math.sqrt(Vb[0, 0]))

            results.append({"sig": sig, "insts": insts, "evaluable": True, "n": n,
                            "mean_ann": mean * 12, "vol_ann": sd * math.sqrt(12),
                            "sharpe": (mean / sd) * math.sqrt(12) if sd > 0 else 0.0,
                            "t": tstat, "p": p, "aux_tested": aux_tested,
                            "aux_pass": aux_pass, "alpha6_t": alpha_t})

        ev = [x for x in results if x["evaluable"]]
        M = len(sigs)
        # BY-FDR
        ps = sorted((x["p"], i) for i, x in enumerate(ev))
        cM = sum(1.0 / i for i in range(1, M + 1))
        by_pass_idx = set()
        kmax = 0
        for rank, (pv, i) in enumerate(ps, start=1):
            if pv <= rank * Q_FDR / (M * cM):
                kmax = rank
        for rank, (pv, i) in enumerate(ps, start=1):
            if rank <= kmax:
                by_pass_idx.add(i)

        hurdle = math.sqrt(2 * math.log(M)) if M > 1 else float("nan")
        r1 = [x for x in ev if x["mean_ann"] > 0]
        r2 = [x for x in ev if x["t"] > 2]
        r4 = [x for x in ev if x["t"] > hurdle]
        full = [x for x in ev if x["t"] > 2 and x["aux_tested"] > 0
                and x["aux_pass"] == x["aux_tested"]]
        aux_all = sum(x["aux_pass"] for x in ev)
        aux_tot = sum(x["aux_tested"] for x in ev)

        summary.append(
            f"\n=== {model} (wall {wall}; window {nwin} months; M = {M} distinct expressions) ===\n"
            f"  evaluable: {len(ev)}/{M}   ladder: mean>0: {len(r1)}  t>2: {len(r2)}  "
            f"BY-FDR(q=.05): {len(by_pass_idx)}  t>√(2lnM)={hurdle:.2f}: {len(r4)}\n"
            f"  E[max t] under null ≈ {hurdle:.2f}; realized max t = "
            f"{max((x['t'] for x in ev), default=float('nan')):.2f}; "
            f"median t = {np.median([x['t'] for x in ev]):.2f}\n"
            f"  aux predictions: {aux_tot} testable, {aux_all} hold "
            f"({(aux_all / aux_tot) if aux_tot else float('nan'):.0%})\n"
            f"  FULL SURVIVORS (t>2 AND all testable aux hold): {len(full)}"
        )
        for x in sorted(full, key=lambda z: -z["t"])[:10]:
            terms, regime, vt = x["sig"]
            summary.append(f"    t={x['t']:.2f} ann={x['mean_ann']:.1f}% SR={x['sharpe']:.2f} "
                           f"n={x['n']} aux {x['aux_pass']}/{x['aux_tested']} | "
                           f"{' + '.join(f'{w:+g}·{p}' for p, w in terms)} | {regime} | vt={vt}")

        for i, x in enumerate(ev):
            terms, regime, vt = x["sig"]
            perf_rows.append({
                "model": model, "expr": " + ".join(f"{w:+g}*{p}" for p, w in terms),
                "regime": regime, "vol_target": vt, "multiplicity": len(x["insts"]),
                "n_months": x["n"], "mean_ann_pct": round(x["mean_ann"], 3),
                "sharpe": round(x["sharpe"], 3), "t": round(x["t"], 3),
                "p": round(x["p"], 5), "by_pass": i in by_pass_idx,
                "aux_tested": x["aux_tested"], "aux_pass": x["aux_pass"],
                "full_survivor": x in full,
                "alpha6_t": round(x["alpha6_t"], 3) if x["alpha6_t"] is not None else None,
            })

    pl.DataFrame(perf_rows).write_csv(DERIVED / "results_perf.csv")
    out = "\n".join(summary)
    (DERIVED / "results_summary.txt").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
