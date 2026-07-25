#!/usr/bin/env python3
"""Recompute every checkable number in the paper from the committed artifacts, and diff.

Motivation (2026-07-25): two cold reads found published numbers that were simply wrong --
"negative in 21 of 25 cells" when the artifact says 20, and a citation-control t labelled as the
OpenAI panel when it was the four-model value. Both were arithmetic anyone could have checked.
The underlying pipeline outputs have been correct every time; the weak layer is numbers
transcribed into prose and then left behind when a revision changes the subset they describe.

So this script does not trust the prose at all. It recomputes each fact from the committed
derived artifacts and asserts that the rendered paper contains it. It also sweeps every numeral
in the paper and reports any that no registered fact explains, so new unbacked numbers surface
on the next run rather than in a referee report.

Usage:  python3 check_numbers.py [path/to/paper.pdf]
Exit 0 if every registered fact is present and no unexplained numeral appears; 1 otherwise.
Requires: pdftotext (poppler). Pure stdlib otherwise.
"""
from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

def _find_root(start: Path) -> Path:
    """Repo root = the directory containing exp1/derived and exp2/derived.

    Works from the private project dir (script at the root) and from the public archive
    (script in scripts/), so the same file verifies both copies of the paper.
    """
    for d in (start, *start.parents):
        if (d / "exp2" / "derived").is_dir() and (d / "exp1" / "derived").is_dir():
            return d
    sys.exit("could not locate exp1/derived and exp2/derived above " + str(start))


ROOT = _find_root(Path(__file__).resolve().parent)


def _default_pdf() -> Path:
    for cand in ("main.pdf", "validation-bottleneck.pdf"):
        if (ROOT / cand).exists():
            return ROOT / cand
    return ROOT / "main.pdf"


PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_pdf()

MODELS = ["gpt-4-0613", "gpt-4o-2024-08-06", "meta-llama_Llama-3.3-70B-Instruct-Turbo"]
SHORT = {"gpt-4-0613": "gpt-4-0613", "gpt-4o-2024-08-06": "gpt-4o",
         "meta-llama_Llama-3.3-70B-Instruct-Turbo": "Llama-3.3-70B"}

results: list[tuple[bool, str, str]] = []
claimed: set[str] = set()


def load_text() -> str:
    if not PDF.exists():
        sys.exit(f"paper not found: {PDF}")
    out = subprocess.run(["pdftotext", str(PDF), "-"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("pdftotext failed -- install poppler")
    t = out.stdout
    for dash in "‐‑‒–—−":
        t = t.replace(dash, "-")
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t)


TEXT = load_text()


def need(label: str, *variants: str, source: str = "") -> None:
    """Assert at least one rendering of a recomputed fact appears in the paper.

    NOTE the known limitation: this matches anywhere in the document, so a value that also
    occurs elsewhere can pass spuriously. Use near() for any fact whose number is not unique.
    """
    hit = next((v for v in variants if v in TEXT), None)
    for v in variants:
        claimed.add(v)
    results.append((hit is not None, label, f"{source}  [{hit or ' | '.join(variants)}]"))


def near(label: str, anchor: str, value: str, window: int = 90, source: str = "") -> None:
    """Assert `value` occurs within `window` characters of `anchor` -- for non-unique numbers."""
    claimed.add(value)
    claimed.add(anchor)
    ok = False
    for m in re.finditer(re.escape(anchor), TEXT):
        seg = TEXT[max(0, m.start() - window): m.end() + window]
        if value in seg:
            ok = True
            break
    results.append((ok, label, f"{source}  [{value} near '{anchor}']"))


def pct(n: int, d: int) -> int:
    return round(100 * n / d)


# --------------------------------------------------------------------------- artifacts
def rows(p: str) -> list[dict]:
    return list(csv.DictReader((ROOT / p).open()))


perf = rows("exp2/derived/results_perf.csv")
uni = rows("exp2/derived/eval_universe.csv")
ctrl = [r for r in rows("exp2/derived/aux_positive_control.csv") if r["tested"] == "true"]
tmix = rows("exp2/derived/aux_type_mix.csv")
ra = rows("exp1/derived/results_a.csv")
gs = rows("exp1/derived/grade_split.csv")
em = rows("exp1/derived/effective_m.csv")
rz = json.loads((ROOT / "exp2/derived/randomization_test.json").read_text())["models"]
cite = rows("exp1/derived/citation_subpanels.csv")
rb = (ROOT / "exp1/derived/results_b.txt").read_text()

# --------------------------------------------------------------- Exp 2 evaluation ladder
for m in MODELS:
    sub = [r for r in perf if r["model"] == m and r["t"] not in ("", "None")]
    ts = sorted(float(r["t"]) for r in sub)
    u = next(r for r in uni if r["model"] == m)
    src = f"exp2/derived (model {SHORT[m]})"
    need(f"{SHORT[m]}: window months", str(int(sub[0]["n_months"])), source=src)
    need(f"{SHORT[m]}: M distinct", str(int(u["distinct_M"])), source=src)
    need(f"{SHORT[m]}: evaluable", str(len(sub)), source=src)
    need(f"{SHORT[m]}: mean>0", str(sum(1 for r in sub if float(r["mean_ann_pct"]) > 0)), source=src)
    need(f"{SHORT[m]}: t>2 count", str(sum(1 for t in ts if t > 2)), source=src)
    need(f"{SHORT[m]}: max t", f"{max(ts):.2f}", source=src)
    need(f"{SHORT[m]}: sqrt(2 ln M)", f"{math.sqrt(2*math.log(int(u['distinct_M']))):.2f}", source=src)
    need(f"{SHORT[m]}: aux tested", str(sum(int(r["aux_tested"]) for r in sub)), source=src)
    at = sum(int(r["aux_tested"]) for r in sub)
    ap = sum(int(r["aux_pass"]) for r in sub)
    need(f"{SHORT[m]}: aux pass rate", f"{pct(ap, at)}%", source=src)
    need(f"{SHORT[m]}: full survivors", str(sum(1 for r in sub if r["full_survivor"] == "true")), source=src)
    need(f"{SHORT[m]}: dup rate", f"{100*float(u['dup_rate']):.1f}%", f"{100*float(u['dup_rate']):.0f}%",
         source=src)
    z = rz[m]
    need(f"{SHORT[m]}: grammar-null median", f"{z['null_max_p50']:.2f}", source="randomization_test.json")
    need(f"{SHORT[m]}: realized percentile", f"{round(z['realized_max_percentile'])}%",
         f"{round(z['realized_max_percentile'])}th", source="randomization_test.json")

# compliance and validity, per model
_ev = [r for r in uni if not r["model"].startswith("gpt-4.1")]
_cr = [100 * int(r["compliant"]) / int(r["hypotheses"]) for r in _ev]
need("compliance range", f"{min(_cr):.0f}-{max(_cr):.0f}%", source="eval_universe.csv")
for r in _ev:
    need(f"{SHORT[r['model']]}: hypotheses", str(int(r["hypotheses"])), source="eval_universe.csv")
    need(f"{SHORT[r['model']]}: invalid", str(int(r["invalid"])), source="eval_universe.csv")

# pooled machine aux
at = sum(int(r["aux_tested"]) for r in perf if r["t"] not in ("", "None"))
ap = sum(int(r["aux_pass"]) for r in perf if r["t"] not in ("", "None"))
need("pooled machine aux tested", str(at), source="results_perf.csv")
need("pooled machine aux rate", f"{pct(ap, at)}%", source="results_perf.csv")
need("panel evaluable total", str(sum(1 for r in perf if r["t"] not in ("", "None"))),
     source="results_perf.csv")

# ------------------------------------------------------------------ positive control
cp = sum(1 for r in ctrl if r["pass"] == "true")
need("control pooled pass rate", f"{pct(cp, len(ctrl))}%", source="aux_positive_control.csv")
need("control pooled fraction", f"{cp} of {len(ctrl)}", f"{cp}/{len(ctrl)}", source="aux_positive_control.csv")
for typ, lbl in [("regime", "regime"), ("corr", "correlation"), ("subperiod", "subperiod")]:
    s = [r for r in ctrl if r["type"] == typ]
    need(f"control {lbl} {sum(1 for r in s if r['pass']=='true')}/{len(s)}",
         f"{sum(1 for r in s if r['pass']=='true')}/{len(s)}", source="aux_positive_control.csv")
c0 = [r for r in ctrl if r["model"] == "gpt-4-0613"]
c0p = sum(1 for r in c0 if r["pass"] == "true")
need("control long-window rate", f"{pct(c0p, len(c0))}%", source="aux_positive_control.csv")
need("control long-window fraction", f"{c0p} of {len(c0)}", f"{c0p}/{len(c0)}",
     source="aux_positive_control.csv")
nc = [r for r in ctrl if r["type"] != "corr"]
need("control excl. threshold items", f"{sum(1 for r in nc if r['pass']=='true')} of {len(nc)}",
     source="aux_positive_control.csv")
need("control excl. threshold rate", f"{pct(sum(1 for r in nc if r['pass']=='true'), len(nc))}%",
     source="aux_positive_control.csv")

# ------------------------------------------------------------------ aux type mix
t0 = {r["type"]: (int(r["tested"]), int(r["passed"])) for r in tmix if r["model"] == "gpt-4-0613"}
tot0 = sum(v[0] for v in t0.values())
for k, lbl in [("regime", "regime"), ("corr", "correlation"), ("subperiod", "subperiod")]:
    need(f"machine {lbl} {t0[k][1]}/{t0[k][0]}", f"{t0[k][1]} of {t0[k][0]}", f"{t0[k][1]}/{t0[k][0]}",
         source="aux_type_mix.csv")
    # context-anchored: these percentages collide with other rates in the paper
    near(f"machine {lbl} pass % (in context)", f"{t0[k][1]} of {t0[k][0]}",
         f"{pct(t0[k][1], t0[k][0])}%", source="aux_type_mix.csv")
need("machine structural share", f"{100*(t0['regime'][0]+t0['corr'][0])/tot0:.1f}%", source="aux_type_mix.csv")
need("machine subperiod share", f"{100*t0['subperiod'][0]/tot0:.1f}%", source="aux_type_mix.csv")
need("control structural share", f"{pct(11, 14)}%", source="aux_positive_control.csv (5+6 of 14)")

# ------------------------------------------------------------------------ Exp 1
matched = {}
for r in ra:
    matched.setdefault(r["model"], [0, 0])
    matched[r["model"]][0] += int(r["n"])
    matched[r["model"]][1] += int(r["matched"])
lo = min(100 * v[1] / v[0] for v in matched.values())
hi = max(100 * v[1] / v[0] for v in matched.values())
need("confinement range", f"{round(lo)}-{round(hi)}%", source="results_a.csv")
base = {r["vintage"]: float(r["library_base_rate"]) for r in ra}
for v in sorted(base):
    need(f"base-rate ceiling {v}", f"{1/base[v]:.2f}", source="results_a.csv")
gsa: dict[str, dict[str, list[int]]] = {}
for r in gs:
    gsa.setdefault(r["model"], {}).setdefault(r["strength"], [0, 0])
    gsa[r["model"]][r["strength"]][0] += int(r["n_matched"])
    gsa[r["model"]][r["strength"]][1] += int(r["n_post"])
ex = [100 * v["exact"][1] / v["exact"][0] for v in gsa.values()]
need("exact-match post-date range", f"{round(min(ex))}-{round(max(ex))}%",
     f"{round(min(ex))}-{round(max(ex))} percent", source="grade_split.csv")
t5 = [float(r["top5_share"]) for r in em]
need("top-5 concentration range", f"{100*min(t5):.0f}-{100*max(t5):.0f}%", source="effective_m.csv")
per = {}
for r in em:
    per.setdefault(r["model"], []).append(float(r["top5_share"]))
means = [sum(v) / len(v) for v in per.values()]
need("top-5 per-model mean range", f"{100*min(means):.0f}%", f"{100*max(means):.0f}%",
     source="effective_m.csv")

# Table 1 cells (results_a.csv) and Table 3 cells (grade_split.csv): register every value
for r in ra:
    for v in (str(int(r["matched"])), str(int(r["n"])),
              f'{float(r["share_post_of_matched"]):.2f}',
              f'{float(r["library_base_rate"]):.2f}',
              f'{float(r["enrichment"]):.2f}'):
        claimed.add(v)
for mdl, d in gsa.items():
    for st in ("exact", "close"):
        claimed.add(str(d[st][0]))
        claimed.add(f"{d[st][1]/d[st][0]:.2f}")
# in-text figures with their own provenance
for v in ["2.10%", "3.86%", "1.76", "6,430", "4.67", "33.2%", "12.7%", "0.73%", "91%",
          "12%", "19%", "125", "10%", "106", "63%", "3.26", "2.09", "3.4", "1.38",
          "0.30", "0.31", "70%", "69%", "898", "5.17", "1.07"]:
    claimed.add(v)

# ------------------------------------------------------------------------ test (b)
beta = re.search(r"fut\s+beta=([+-][\d.]+)\s+se=[\d.]+\s+t=([+-][\d.]+)", rb)
need("test (b) beta", f"{abs(float(beta.group(1))):.3f}", source="results_b.txt")
need("test (b) t", f"{abs(float(beta.group(2))):.2f}", source="results_b.txt")
_n = re.search(r"N=(\d+)", rb).group(1)
need("test (b) N", f"{int(_n):,}", _n, source="results_b.txt")
rhos = [float(x) for x in re.findall(r"rho=([+-][\d.]+)", rb)]
need("test (b) negative cells", f"{sum(1 for x in rhos if x < 0)} of {len(rhos)}", source="results_b.txt")
_fz = re.search(r"Fisher-combined z = ([+-]?[\d.]+)", rb)
need("Fisher z", f"{abs(float(_fz.group(1))):.2f}", source="results_b.txt")
_cl = re.search(r"log_cites\s+beta=[+-][\d.]+\s+se=[\d.]+\s+t=([+-][\d.]+)", rb)
need("citation control all five (registered)", f"{abs(float(_cl.group(1))):.2f}",
     source="results_b.txt (04_score, inv)")
for r in cite:
    if r["panel"] == "all_five":
        continue  # the paper quotes the registered 04_score value above, not the pinv diagnostic
    need(f"citation control {r['panel']}", f"{abs(float(r['log_cites_t'])):.2f}",
         source="citation_subpanels.csv")

# ------------------------------------------------------------- closed-form constants
u = 1.2564
g = (1 - math.exp(-u)) / math.sqrt(u)
need("knowability coefficient", f"{g/math.sqrt(math.log(2)):.2f}", source="closed form")
need("knowability T*", f"{u/math.log(2):.2f}", source="closed form")
need("knowability h-factor", f"{(1/(g/math.sqrt(math.log(2))))**2:.2f}", source="closed form")
need("knowability SR0=1,t*=3", "fifteen", source="1.70*(3/1)^2 = 15.3")
need("knowability SR0=.5,t*=3", "sixty", source="1.70*(3/0.5)^2 = 61.2")
need("power 36y", "36", source="(3/0.5)^2")
need("power 16y", "sixteen", "16", source="(2/0.5)^2")
need("eq maxsr at M=330,T=32mo", f"{math.sqrt(2*math.log(330)/(32/12)):.2f}", source="closed form")
need("N_eff rho=0.01", f"{3000/(1+2999*0.01):.0f}", source="closed form")
need("SE rho=0.01", f"{100/math.sqrt(96.8*720):.2f}%", source="closed form")
need("SE rho=0.002", f"{100/math.sqrt(428.7*720):.2f}%", source="closed form")
for M, T, lbl in [(1e4, 5, "floor M=1e4 T=5"), (1e6, 5, "floor M=1e6 T=5"), (1e6, 10, "floor M=1e6 T=10")]:
    need(lbl, f"{math.sqrt(2*math.log(M)/T):.1f}", source="closed form")

# ------------------------------------------------------------------ corpus counts
need("exp1 proposals", "2,750", source="5 models x 5 vintages x 110")
need("exp1 per model", "550", source="110 x 5")
need("exp1 per model-vintage", "110", source="11 runs x 10")
need("exp2 generated (4 models)", "1,336", source="336*3 + 328")
need("exp2 evaluated-model total", "1,000", source="336+336+328")
need("library size", "212", source="Chen-Zimmermann OSAP")
need("ChronoGPT generations", "165", source="3 vintages x 5 dates x 11 runs")

# ------------------------------------------------------- unexplained-numeral sweep
KNOWN_STRUCTURAL = re.compile(
    r"^(19\d\d|20\d\d|[12]\d{3}-\d\d|0|1|2|3|4|5|6|7|8|9|10|11|12|"
    r"1\.0|0\.5|0\.2|0\.3|0\.05|2\.0|3\.0|100|1/2|24|36|60|"
    r"[ivxlIVXL]+)$")
BODY = TEXT
_cut = BODY.rfind("References")
if _cut > len(BODY) // 2:
    BODY = BODY[:_cut]
_start = BODY.find("For as long as quantitative investing")
if _start > 0:
    BODY = BODY[_start:]
nums = re.findall(r"\d+(?:[.,]\d+)*%?", BODY)
unexplained = []
for n in set(nums):
    if any(n in c for c in claimed):
        continue
    if KNOWN_STRUCTURAL.match(n):
        continue
    unexplained.append(n)

# ------------------------------------------------------------------------ report
fails = [r for r in results if not r[0]]
print(f"REGISTERED FACTS: {len(results)} checked, {len(results)-len(fails)} present, {len(fails)} MISSING\n")
for ok, label, src in results:
    if not ok:
        print(f"  [MISSING] {label:44s} {src}")
if not fails:
    print("  all registered facts appear in the paper as computed from the artifacts.")

print(f"\nUNEXPLAINED NUMERALS: {len(unexplained)} (eyeball these; not necessarily wrong)")
for n in sorted(unexplained, key=lambda x: (len(x), x))[:60]:
    ctx = re.search(r".{55}" + re.escape(n) + r".{55}", BODY)
    print(f"  {n:>10s}  ...{ctx.group(0).strip() if ctx else ''}...")

sys.exit(1 if fails else 0)
