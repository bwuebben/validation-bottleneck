"""Auxiliary-implication TYPE MIX and per-type pass rates (added 2026-07-25).

Mirrors the aux block of 05_evaluate.py verbatim -- same corpus, same dedup, same h0 = first
instance per distinct expression, same testability thresholds -- and additionally records the
TYPE of every tested implication. This answers whether the machine-vs-control comparison in
Section 5.3 runs a comparable battery on both sides, and supplies the within-type breakdown.

Adds nothing to the registered analysis: the totals it reports reproduce results_perf.csv
exactly (437/89/88 tested, 200/26/14 passed). Output: derived/aux_type_mix.csv
"""
from __future__ import annotations
import datetime as dt, json, math, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np, polars as pl
import re

EXP2 = Path(__file__).resolve().parent.parent
DERIVED = EXP2 / "derived"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dp import get_dp

DATA_END = dt.date(2024, 12, 31)
WALLS = {"gpt-4-0613": "2022-04", "gpt-4o-2024-08-06": "2024-01",
         "meta-llama_Llama-3.3-70B-Instruct-Turbo": "2024-03"}
MIN_MONTHS = {"gpt-4-0613": 24, "gpt-4o-2024-08-06": 9,
              "meta-llama_Llama-3.3-70B-Instruct-Turbo": 8}
REGIMES = ["VIX_ABOVE_TRAILING_MEDIAN", "VIX_BELOW_TRAILING_MEDIAN",
           "TERM_SPREAD_POSITIVE", "TERM_SPREAD_NEGATIVE",
           "MKT_TRAILING_12M_UP", "MKT_TRAILING_12M_DOWN"]
WEIGHTS = [-1.0, -0.5, 0.5, 1.0]

mk = lambda d: (d.year, d.month)
prev = lambda k: (k[0] - 1, 12) if k[1] == 1 else (k[0], k[1] - 1)
def wall_after(k, wall): return k > (int(wall[:4]), int(wall[5:7]))

dp = get_dp()
op = dp.factors.open_asset_pricing("op")
ls = op.filter(pl.col("port") == "LS").select(["signalname", "date", "ret"])
series = {}
for r in ls.iter_rows(named=True):
    series.setdefault(r["signalname"], {})[mk(r["date"])] = float(r["ret"])
lib = set(series)

vix = dp.macro.fred("VIXCLS"); vm = {}
for r in vix.iter_rows(named=True):
    if r["VIXCLS"] is not None: vm[mk(r["date"])] = float(r["VIXCLS"])
vk = sorted(vm); vix_above = {}
for i, k in enumerate(vk):
    h = [vm[x] for x in vk[max(0, i - 59): i + 1]]
    if len(h) >= 36: vix_above[k] = vm[k] > float(np.median(h))

tr = dp.macro.treasury_rates().sort("date")
trr = [(r["date"], float(r["rate_10y"]) - float(r["rate_3m"])) for r in tr.iter_rows(named=True)
       if r["rate_10y"] is not None and r["rate_3m"] is not None]
spread_pos = {}
for k in {mk(d) for d, _ in trr}:
    eom = [s for d, s in trr if mk(d) <= k]
    if eom:
        last = [s for d, s in trr if mk(d) == k]
        spread_pos[k] = (last[-1] if last else eom[-1]) > 0

ff = dp.factors.fama_french("3-factor"); mkt = {}
for r in ff.iter_rows(named=True):
    a, b = r.get("Mkt-RF"), r.get("RF")
    if a is not None and b is not None: mkt[mk(r["date"])] = (float(a) + float(b)) / 100.0
mks = sorted(mkt); mkt_up = {}
for i, k in enumerate(mks):
    if i >= 12: mkt_up[k] = np.prod([1 + mkt[x] for x in mks[i - 12: i]]) - 1 > 0

def regime_true(regime, k):
    km1 = prev(k)
    if regime.startswith("VIX"):
        v = vix_above.get(km1)
        return None if v is None else (v if regime == "VIX_ABOVE_TRAILING_MEDIAN" else not v)
    if regime.startswith("TERM"):
        s = spread_pos.get(km1)
        return None if s is None else (s if regime == "TERM_SPREAD_POSITIVE" else not s)
    u = mkt_up.get(k)
    return None if u is None else (u if regime == "MKT_TRAILING_12M_UP" else not u)

def combo_series(sig):
    terms, regime, vt = sig
    keys = None
    for p, _ in terms:
        s = set(series.get(p, {}))
        keys = s if keys is None else keys & s
    out = {}
    for k in sorted(keys or []):
        out[k] = sum(w * series[p][k] for p, w in terms)
    return out

def strategy_series(sig):
    terms, regime, vt = sig
    gated = {}
    for k, v in combo_series(sig).items():
        if regime != "NONE":
            rt = regime_true(regime, k)
            if rt is None: continue
            v = v if rt else 0.0
        gated[k] = v
    if not vt: return gated
    ks = sorted(gated); out = {}; target = 10.0 / math.sqrt(12)
    for i, k in enumerate(ks):
        h = [gated[x] for x in ks[max(0, i - 36): i]]
        s = 1.0
        if len(h) >= 24:
            sd = float(np.std(h, ddof=1))
            if sd > 1e-9: s = min(target / sd, 5.0)
        out[k] = gated[k] * s
    return out

# ---- load hypotheses exactly as 05_evaluate does ----
def strip_fences(c):
    c = c.strip()
    if c.startswith("```"):
        c = re.sub(r"^```[a-zA-Z]*\n", "", c)
        c = re.sub(r"\n```$", "", c)
    return c

lib = {l.split(":")[0] for l in (EXP2/"primitives"/"library_block.txt").read_text().splitlines()}
hyps = []
for f in sorted((EXP2/"corpus").glob("*/v*_seed*.json")):
    if f.name.startswith("smoke_"): continue
    env = json.loads(f.read_text())
    model = env["meta"]["model_requested"].replace("/", "_")
    if model not in WALLS: continue
    content = env["response"]["choices"][0]["message"]["content"]
    try: j = json.loads(content)
    except json.JSONDecodeError:
        try: j = json.loads(strip_fences(content))
        except json.JSONDecodeError: j = {"hypotheses": []}
    if True:
        for i, h in enumerate(j.get("hypotheses", []) or []):
            expr = h.get("expression", {}) or {}
            terms = expr.get("terms", []) or []
            valid = (2 <= len(terms) <= 4
                     and all(isinstance(t, dict) and t.get("primitive") in lib
                             and float(t.get("weight", 0)) in WEIGHTS for t in terms)
                     and (expr.get("regime") in REGIMES or expr.get("regime") in (None, "null")))
            sig = None
            if valid:
                sig = (tuple(sorted((t["primitive"], float(t["weight"])) for t in terms)),
                       expr.get("regime") or "NONE", bool(expr.get("vol_target")))
            hyps.append({"model": model, "valid": valid, "sig": sig, "h": h})

rows_out = []
for model, wall in WALLS.items():
    mh = [x for x in hyps if x["model"] == model and x["valid"]]
    sigs = defaultdict(list)
    for x in mh: sigs[x["sig"]].append(x)
    window = [k for k in sorted({kk for s in series.values() for kk in s})
              if wall_after(k, wall) and dt.date(k[0], k[1], 1) <= DATA_END]
    nwin = len(window)
    tested = Counter(); passed = Counter(); stated = Counter()
    for sig, insts in sigs.items():
        strat = strategy_series(sig)
        post = {k: v for k, v in strat.items()
                if wall_after(k, wall) and dt.date(k[0], k[1], 1) <= DATA_END}
        if len(post) < MIN_MONTHS[model] or len(post) < 0.8 * nwin: continue
        h0 = insts[0]["h"]
        raw = combo_series(sig)
        post_raw = {k: v for k, v in raw.items() if k in post}
        for a in h0.get("auxiliary_predictions", []) or []:
            if not isinstance(a, dict): continue
            typ = a.get("type"); stated[typ] += 1
            if typ == "regime_interaction" and a.get("regime") in REGIMES:
                ins, outs = [], []
                for k, v in post_raw.items():
                    rt = regime_true(a["regime"], k)
                    if rt is True: ins.append(v)
                    elif rt is False: outs.append(v)
                if len(ins) >= 6 and len(outs) >= 6:
                    diff = np.mean(ins) - np.mean(outs)
                    tested["regime"] += 1
                    passed["regime"] += int((diff > 0) == (a.get("direction") == "higher"))
            elif typ == "correlation" and a.get("primitive") in series:
                ps = series[a["primitive"]]
                joint = [k for k in post_raw if k in ps]
                if len(joint) >= 12:
                    c = abs(float(np.corrcoef([post_raw[k] for k in joint],
                                              [ps[k] for k in joint])[0, 1]))
                    thr = float(a.get("threshold", 0.3) or 0.3)
                    tested["corr"] += 1
                    passed["corr"] += int((c < thr) == (a.get("bound") == "below"))
            elif typ == "subperiod_consistency":
                ks = sorted(post); half = len(ks) // 2
                m1 = np.mean([post[k] for k in ks[:half]]); m2 = np.mean([post[k] for k in ks[half:]])
                tested["subperiod"] += 1
                passed["subperiod"] += int(m1 > 0 and m2 > 0)
    tt = sum(tested.values()); pp = sum(passed.values())
    print(f"\n=== {model} (wall {wall}) ===")
    print(f"  STATED types: {dict(stated)}")
    print(f"  TESTED total {tt}  passed {pp}  rate {100*pp/tt:.1f}%" if tt else "  TESTED 0")
    for k in ("regime", "corr", "subperiod"):
        if tested[k]:
            print(f"    {k:10s} {passed[k]:3d}/{tested[k]:3d} = {100*passed[k]/tested[k]:5.1f}%")
    if tt:
        print(f"  structural share of tested: {100*(tested['regime']+tested['corr'])/tt:.1f}%")
    for k in ("regime", "corr", "subperiod"):
        rows_out.append({"model": model, "type": k, "stated_all_types": sum(stated.values()),
                         "tested": tested[k], "passed": passed[k]})

pl.DataFrame(rows_out).write_csv(DERIVED / "aux_type_mix.csv")
print("\nwrote derived/aux_type_mix.csv")
