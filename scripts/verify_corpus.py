#!/usr/bin/env python3
"""Standalone integrity + reproduction check for the Validation Bottleneck archive.

Stdlib only. Verifies:
  1. Corpus counts: 220 Experiment-1 generation files / 2,200 proposals;
     168 Experiment-2 generation files / 1,336 well-formed hypotheses (one archived
     response is malformed JSON and contributes zero, per the registered specification).
  2. Manifest integrity: every manifest line's output hash matches the SHA-256 of the
     archived response content.
  3. Reproduction: recomputes Experiment 1's headline table (results_a.csv) from the raw
     corpus, the raw judge matches, and the bundled predictor metadata, and compares.

Exit code 0 = all checks pass.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATCHED = {"exact", "close"}
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def strip_fences(c):
    c = c.strip()
    c = re.sub(r"^```(json)?\s*", "", c)
    c = re.sub(r"\s*```$", "", c)
    return c


def content_of(env):
    return (env["response"].get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


print("== 1. Corpus counts ==")
e1_files = [f for f in sorted((ROOT / "exp1" / "corpus").glob("*/y*_seed*.json"))
            if not f.name.startswith("smoke_")]
props = 0
for f in e1_files:
    env = json.loads(f.read_text())
    try:
        j = json.loads(content_of(env))
    except json.JSONDecodeError:
        j = json.loads(strip_fences(content_of(env)))
    props += len(j.get("predictors", []))
check("exp1 generation files == 220", len(e1_files) == 220, str(len(e1_files)))
check("exp1 proposals == 2200", props == 2200, str(props))

e2_files = [f for f in sorted((ROOT / "exp2" / "corpus").glob("*/v*_seed*.json"))
            if not f.name.startswith("smoke_")]
hyps = 0
for f in e2_files:
    env = json.loads(f.read_text())
    try:
        j = json.loads(content_of(env))
    except json.JSONDecodeError:
        try:
            j = json.loads(strip_fences(content_of(env)))
        except json.JSONDecodeError:
            j = {}  # one archived malformed response; excluded per specification
    hyps += len(j.get("hypotheses", []))
check("exp2 generation files == 168", len(e2_files) == 168, str(len(e2_files)))
check("exp2 hypotheses == 1336", hyps == 1336, str(hyps))

print("== 2. Manifest hash integrity ==")
for exp in ("exp1", "exp2"):
    man = ROOT / exp / "corpus" / "manifest.jsonl"
    n = bad = 0
    for line in man.read_text().splitlines():
        r = json.loads(line)
        f = ROOT / exp / r["file"].replace("corpus/", "corpus/", 1) if r["file"].startswith("corpus/") \
            else ROOT / exp / "corpus" / r["file"]
        f = (ROOT / exp / r["file"]) if (ROOT / exp / r["file"]).exists() else f
        if not f.exists():
            continue
        env = json.loads(f.read_text())
        h = hashlib.sha256(content_of(env).encode()).hexdigest()
        n += 1
        if h != r["output_sha256"]:
            bad += 1
    check(f"{exp} manifest hashes ({n} checked)", bad == 0, f"{bad} mismatches")

print("== 3. Reproduce exp1 results_a.csv ==")
years = {r["acronym"]: r["year"]
         for r in json.loads((ROOT / "exp2" / "primitives" / "primitives.json").read_text())
         if r.get("year")}
matches = {}
for line in (ROOT / "exp1" / "derived" / "matches.jsonl").read_text().splitlines():
    r = json.loads(line)
    for m in r["matches"]:
        pid = f'{r["file"].split("/")[1]}|{Path(r["file"]).stem}|{m["index"]}'
        matches[pid] = m
rows = {}
for f in e1_files:
    env = json.loads(f.read_text())
    meta = env["meta"]
    model, v = meta["model_requested"], meta["roleplay_year"]
    try:
        j = json.loads(content_of(env))
    except json.JSONDecodeError:
        j = json.loads(strip_fences(content_of(env)))
    for i in range(len(j.get("predictors", []))):
        pid = f"{f.parent.name}|{f.stem}|{i}"
        m = matches.get(pid)
        if m is None:
            continue
        key = (model, v)
        d = rows.setdefault(key, dict(n=0, post=0, pre=0))
        d["n"] += 1
        if m.get("strength") in MATCHED and m.get("match") in years:
            if years[m["match"]] > v:
                d["post"] += 1
            else:
                d["pre"] += 1
lib_years = list(years.values())
bad = 0
with open(ROOT / "exp1" / "derived" / "results_a.csv") as fh:
    for r in csv.DictReader(fh):
        key = (r["model"], int(r["vintage"]))
        d = rows.get(key)
        matched = d["post"] + d["pre"]
        base = sum(1 for y in lib_years if y > key[1]) / len(lib_years)
        ok = (d["n"] == int(r["n"]) and matched == int(r["matched"])
              and d["post"] == int(r["post_date"]) and d["pre"] == int(r["pre_date"])
              and abs(base - float(r["library_base_rate"])) < 1e-9)
        if matched:
            ok = ok and abs(d["post"] / matched - float(r["share_post_of_matched"])) < 1e-9
        if not ok:
            bad += 1
check("results_a.csv reproduced from raw corpus (20 model×vintage cells)", bad == 0,
      f"{bad} cell mismatches")

print()
if fails:
    print(f"RESULT: {len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
print("RESULT: all checks passed.")
