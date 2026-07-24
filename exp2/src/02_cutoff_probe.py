"""Effective-cutoff probe battery — run against each API model BEFORE retirement.

Three families (see ../prompts/probe_battery.json):
  A. index_recall  — GJY-style date-only recall: S&P 500 monthly direction, 2020-01..2023-12,
                     with logprobs (LAP = P(higher)+P(lower); collapse locates the wall).
  B. event_recall  — dated events straddling Sept 2021 (pre-cutoff controls should be known;
                     post-cutoff events should not be).
  C. self_report   — the model's own claimed cutoff (known to be unreliable; recorded anyway).

Usage:  python 02_cutoff_probe.py --model gpt-4-0613 [--dry-run]
Archives to corpus/<model>/probes.jsonl (one line per probe, full response incl. logprobs).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

EXP2 = Path(__file__).resolve().parent.parent
API_URL = "https://api.openai.com/v1/chat/completions"
TIMEOUT = 120

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
        sys.exit("No OPENAI_API_KEY / OPENAI_KEY in environment or exp2/.env.")
    return key


def call(key: str, model: str, question: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 5,
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
        if r.status_code == 400 and "logprobs" in r.text:
            payload.pop("logprobs", None)
            payload.pop("top_logprobs", None)
            continue
        sys.exit(f"API error {r.status_code}: {r.text[:300]}")
    sys.exit("Exhausted retries.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    battery = json.loads((EXP2 / "prompts" / "probe_battery.json").read_text())
    probes = battery["probes"]
    print(f"battery v{battery['version']}: {len(probes)} probes, model={args.model}")
    if args.dry_run:
        for p in probes[:3]:
            print(" ", p["id"], "→", p["question"][:80])
        print("dry run — no calls.")
        return

    key = load_key()
    outdir = EXP2 / "corpus" / args.model.replace("/", "_")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "probes.jsonl"
    done_ids = set()
    if outfile.exists():
        for line in outfile.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["probe_id"])
            except (json.JSONDecodeError, KeyError):
                pass

    with outfile.open("a") as f:
        for i, p in enumerate(probes):
            if p["id"] in done_ids:
                continue
            resp = call(key, args.model, p["question"], p.get("max_tokens", 16))
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
            f.write(json.dumps({
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "probe_id": p["id"],
                "family": p["family"],
                "question_sha256": hashlib.sha256(p["question"].encode()).hexdigest(),
                "model_requested": args.model,
                "model_returned": resp.get("model"),
                "system_fingerprint": resp.get("system_fingerprint"),
                "answer": content,
                "response": resp,
            }) + "\n")
            f.flush()
            print(f"  {i + 1}/{len(probes)} {p['id']}: {content[:60]!r}")
            time.sleep(0.5)
    print(f"probes archived → {outfile}")


if __name__ == "__main__":
    main()
