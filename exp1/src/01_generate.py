"""Exp 1 (roleplay leak) generation harness — OpenAI, fully archived.

Same archival discipline as exp2 (see ../README.md). Key comes from exp2/.env.

Usage:
  python 01_generate.py --model gpt-4-0613 [--smoke] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

HARNESS_VERSION = "1.0.0"
EXP1 = Path(__file__).resolve().parent.parent
EXP2 = EXP1.parent / "exp2"
CORPUS = EXP1 / "corpus"
API_URL = "https://api.openai.com/v1/chat/completions"

YEARS = [1990, 1995, 2000, 2005, 2010]
GREEDY = {"temperature": 0.0, "seed": 42}
SAMPLED = [{"temperature": 0.8, "seed": s} for s in range(1, 11)]
MAX_TOKENS = 2500
TIMEOUT = 300
MAX_RETRIES = 6

SYSTEM = (
    "You are a quantitative equity researcher. You respond with valid JSON only, "
    "conforming exactly to the schema requested — no markdown fences, no commentary "
    "outside the JSON."
)


def load_key() -> str:
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


def build_prompt(year: int) -> tuple[str, str]:
    task = (EXP1 / "prompts" / "task_roleplay.txt").read_text()
    user = task.replace("{YEAR}", str(year))
    sha = hashlib.sha256((SYSTEM + "\x00" + user).encode()).hexdigest()
    return user, sha


def call_openai(key: str, model: str, user: str, temperature: float,
                seed: int, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    delay = 5.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=TIMEOUT,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 529):
                print(f"    HTTP {r.status_code}, retry {attempt}/{MAX_RETRIES} in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            sys.exit(f"API error {r.status_code}: {r.text[:500]}")
        except requests.RequestException as e:
            print(f"    network error ({e}), retry {attempt}/{MAX_RETRIES} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    sys.exit("Exhausted retries.")


def archive(model: str, year: int, seed: int, temperature: float,
            prompt_sha: str, resp: dict, smoke: bool) -> Path:
    outdir = CORPUS / model.replace("/", "_")
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = "smoke_" if smoke else ""
    fname = f"{prefix}y{year}_seed{seed:02d}_T{temperature:g}.json"
    meta = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_requested": model,
        "roleplay_year": year,
        "prompt_sha256": prompt_sha,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "harness_version": HARNESS_VERSION,
        "smoke": smoke,
    }
    path = outdir / fname
    path.write_text(json.dumps({"meta": meta, "response": resp}, indent=1))

    content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    manifest_line = {
        **meta,
        "file": str(path.relative_to(EXP1)),
        "model_returned": resp.get("model"),
        "system_fingerprint": resp.get("system_fingerprint"),
        "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
        "usage": resp.get("usage"),
        "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    with (CORPUS / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(manifest_line) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = [GREEDY] + SAMPLED
    plan = [(y, r) for y in YEARS for r in runs]
    print(f"model={args.model}  years={YEARS}  calls={'1 (smoke)' if args.smoke else len(plan)}")

    if args.dry_run:
        user, sha = build_prompt(YEARS[0])
        print(f"prompt_sha256({YEARS[0]})={sha[:16]}…  user_chars={len(user)}")
        return

    key = load_key()

    if args.smoke:
        global MAX_TOKENS
        MAX_TOKENS = 300
        user, sha = build_prompt(YEARS[0])
        resp = call_openai(key, args.model, user, 0.0, 42, MAX_TOKENS)
        path = archive(args.model, YEARS[0], 42, 0.0, sha, resp, smoke=True)
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        print(f"smoke OK → {path}")
        print("first 200 chars:", content[:200])
        return

    done = 0
    for year, run in plan:
        t, s = run["temperature"], run["seed"]
        outdir = CORPUS / args.model.replace("/", "_")
        fname = f"y{year}_seed{s:02d}_T{t:g}.json"
        if (outdir / fname).exists():
            done += 1
            continue
        user, sha = build_prompt(year)
        print(f"  call {done + 1}/{len(plan)}: y{year} seed={s} T={t}")
        resp = call_openai(key, args.model, user, t, s, MAX_TOKENS)
        archive(args.model, year, s, t, sha, resp, smoke=False)
        done += 1
        time.sleep(1.0)
    print(f"complete: {done}/{len(plan)} archived under corpus/{args.model}/")


if __name__ == "__main__":
    main()
