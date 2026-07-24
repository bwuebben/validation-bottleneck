"""Exp 2 generation harness — OpenAI chat completions, fully archived.

The archived corpus (raw responses + call metadata) IS the deliverable; see ../README.md.

Usage:
  python 01_generate.py --list-models
  python 01_generate.py --model gpt-4-0613 --smoke
  python 01_generate.py --model gpt-4-0613 [--variant v1_baseline|v2_diversity] [--dry-run]

API key: env OPENAI_API_KEY, else exp2/.env (gitignored).
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
EXP2 = Path(__file__).resolve().parent.parent
PROMPTS = EXP2 / "prompts"
CORPUS = EXP2 / "corpus"
API_URL = "https://api.openai.com/v1/chat/completions"

VARIANTS = ["v1_baseline", "v2_diversity"]
GREEDY = {"temperature": 0.0, "seed": 42}
SAMPLED = [{"temperature": 0.8, "seed": s} for s in range(1, 21)]
MAX_TOKENS = 4800
TIMEOUT = 300
MAX_RETRIES = 6


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
        sys.exit("No OPENAI_API_KEY / OPENAI_KEY in environment or exp2/.env — cannot call the API.")
    return key


def build_prompt(variant: str) -> tuple[str, str, str]:
    """Return (system_text, user_text, prompt_sha256). Hash covers both messages."""
    system = (PROMPTS / "system.txt").read_text()
    task = (PROMPTS / f"task_{variant}.txt").read_text()
    library = (EXP2 / "primitives" / "library_block.txt").read_text()
    user = task.replace("{LIBRARY}", library)
    sha = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()
    return system, user, sha


def call_openai(key: str, model: str, system: str, user: str,
                temperature: float, seed: int, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
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


def archive(model: str, variant: str, seed: int, temperature: float,
            prompt_sha: str, resp: dict, smoke: bool) -> Path:
    outdir = CORPUS / model.replace("/", "_")
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = "smoke_" if smoke else ""
    fname = f"{prefix}{variant}_seed{seed:02d}_T{temperature:g}.json"
    meta = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_requested": model,
        "prompt_variant": variant,
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
        "file": str(path.relative_to(EXP2)),
        "model_returned": resp.get("model"),
        "system_fingerprint": resp.get("system_fingerprint"),
        "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
        "usage": resp.get("usage"),
        "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "output_chars": len(content),
    }
    with (CORPUS / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(manifest_line) + "\n")
    return path


def list_models(key: str) -> None:
    r = requests.get("https://api.openai.com/v1/models",
                     headers={"Authorization": f"Bearer {key}"}, timeout=60)
    r.raise_for_status()
    ids = sorted(m["id"] for m in r.json()["data"])
    watch = [m for m in ids if any(t in m for t in ("gpt-4-0613", "gpt-4o-2024", "gpt-4.1", "gpt-4-turbo"))]
    print(f"{len(ids)} models served. Panel-relevant:")
    for m in watch:
        print("  ", m)
    if "gpt-4-0613" not in ids and "gpt-4" not in ids:
        print("  !! gpt-4-0613 NOT in list — check deprecation status immediately.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--smoke", action="store_true", help="single cheap call (max_tokens=300)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts, print plan, no API calls")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        list_models(load_key())
        return

    if not args.model:
        sys.exit("--model required (or --list-models)")

    variants = [args.variant] if args.variant else VARIANTS
    runs = [GREEDY] + SAMPLED

    plan = [(v, r) for v in variants for r in runs]
    system, _, _ = build_prompt(variants[0])
    print(f"model={args.model}  variants={variants}  calls={'1 (smoke)' if args.smoke else len(plan)}")

    if args.dry_run:
        for v in variants:
            _, user, sha = build_prompt(v)
            print(f"  {v}: prompt_sha256={sha[:16]}…  user_chars={len(user)} (~{len(user)//4} tokens)")
        print("dry run — no API calls made.")
        return

    key = load_key()

    if args.smoke:
        v = variants[0]
        system, user, sha = build_prompt(v)
        global MAX_TOKENS
        MAX_TOKENS = 300
        resp = call_openai(key, args.model, system, user, 0.0, 42, MAX_TOKENS)
        path = archive(args.model, v, 42, 0.0, sha, resp, smoke=True)
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        print(f"smoke OK → {path}")
        print(f"model_returned={resp.get('model')}  fingerprint={resp.get('system_fingerprint')}")
        print("first 200 chars:", content[:200])
        return

    done = 0
    for v in variants:
        system, user, sha = build_prompt(v)
        for run in runs:
            t, s = run["temperature"], run["seed"]
            outdir = CORPUS / args.model.replace("/", "_")
            fname = f"{v}_seed{s:02d}_T{t:g}.json"
            if (outdir / fname).exists():
                print(f"  skip (exists): {v} seed={s} T={t}")
                done += 1
                continue
            print(f"  call {done + 1}/{len(plan)}: {v} seed={s} T={t}")
            resp = call_openai(key, args.model, system, user, t, s, MAX_TOKENS)
            archive(args.model, v, s, t, sha, resp, smoke=False)
            done += 1
            time.sleep(1.0)
    print(f"complete: {done}/{len(plan)} archived under corpus/{args.model}/")


if __name__ == "__main__":
    main()
