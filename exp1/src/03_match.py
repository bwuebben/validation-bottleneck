"""LLM-judge matching of Exp 1 proposals against the 212 OSAP predictors.

Per ANALYSIS_SPEC.md Stage 1: judge = gpt-4.1-2025-04-14, temp 0, seed 42, one call per
generation file (10 proposals). Match on construction/idea, not name or fame. Raw judge
responses archived under corpus_matching/; parsed matches to derived/matches.jsonl.

Usage:  python 03_match.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

EXP1 = Path(__file__).resolve().parent.parent
EXP2 = EXP1.parent / "exp2"
DERIVED = EXP1 / "derived"
MATCH_DIR = EXP1 / "corpus_matching"
MATCH_DIR.mkdir(exist_ok=True)
API_URL = "https://api.openai.com/v1/chat/completions"

JUDGE_MODEL = "gpt-4.1-2025-04-14"
TIMEOUT = 180

SYSTEM = (
    "You are an expert in the empirical asset-pricing literature on cross-sectional "
    "return predictors. You respond with valid JSON only, no markdown fences."
)

TASK = """Below is a library of 212 published cross-sectional return predictors (acronym: description), followed by 10 proposed predictors.

For EACH proposal, decide whether its CONSTRUCTION matches a library predictor. Judge on the construction and underlying signal — NOT on the name, and NOT on how famous the predictor is.
- "exact": essentially the same construction.
- "close": the same core signal/idea with different implementation details.
- "related": shares a theme or data source but is a different signal.
- "none": no meaningful counterpart.
For "exact"/"close"/"related", give the single best-matching acronym; for "none", null.

Output JSON only: {"matches": [{"index": 0, "match": "<ACRONYM>"|null, "strength": "exact"|"close"|"related"|"none", "reason": "<15 words max>"}, ... exactly 10 items]}

LIBRARY:
{LIBRARY}

PROPOSALS:
{PROPOSALS}"""


def load_key() -> str:
    import os
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not key:
        env = EXP2 / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                name, _, val = line.strip().partition("=")
                if name in ("OPENAI_API_KEY", "OPENAI_KEY") and val:
                    key = val.strip().strip('"').strip("'")
    if not key:
        sys.exit("No OPENAI_API_KEY / OPENAI_KEY found.")
    return key


def call(key: str, user: str) -> dict:
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 2000,
    }
    delay = 5.0
    for _ in range(6):
        r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"},
                          json=payload, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        sys.exit(f"API error {r.status_code}: {r.text[:300]}")
    sys.exit("Exhausted retries.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    library = (EXP2 / "primitives" / "library_block.txt").read_text()
    proposals = [json.loads(l) for l in (DERIVED / "proposals.jsonl").read_text().splitlines()]
    by_file: dict[str, list[dict]] = defaultdict(list)
    for p in proposals:
        by_file[p["file"]].append(p)
    files = sorted(by_file)
    if args.limit:
        files = files[: args.limit]
    print(f"{len(proposals)} proposals in {len(files)} files; judge={JUDGE_MODEL}")
    if args.dry_run:
        return

    key = load_key()
    outfile = DERIVED / "matches.jsonl"
    done = set()
    if outfile.exists():
        for l in outfile.read_text().splitlines():
            done.add(json.loads(l)["file"])

    for i, fn in enumerate(files):
        if fn in done:
            continue
        ps = sorted(by_file[fn], key=lambda p: p["idx"])
        ptxt = "\n".join(
            f'{p["idx"]}. name: {p["name"]} | construction: {p["construction"]} | rationale: {p["rationale"]}'
            for p in ps
        )
        user = TASK.replace("{LIBRARY}", library).replace("{PROPOSALS}", ptxt)
        resp = call(key, user)
        content = resp["choices"][0]["message"]["content"]

        raw_name = fn.replace("/", "_").replace(".json", "") + ".judge.json"
        (MATCH_DIR / raw_name).write_text(json.dumps({
            "meta": {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "judge_model": JUDGE_MODEL, "source_file": fn,
                     "prompt_sha256": hashlib.sha256(user.encode()).hexdigest()},
            "response": resp}, indent=1))

        try:
            j = json.loads(content)
            matches = j["matches"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  !! parse failure for {fn}: {e} — raw archived, continuing")
            continue

        with outfile.open("a") as f:
            f.write(json.dumps({"file": fn, "judge_model": JUDGE_MODEL,
                                "matches": matches}) + "\n")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(files)} files judged")
        time.sleep(0.3)
    print("matching complete →", outfile)


if __name__ == "__main__":
    main()
