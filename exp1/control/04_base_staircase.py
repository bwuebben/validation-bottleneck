"""Dated-fact staircase on BASE ChronoGPT vintages (no instruction tuning).

The instruct models' SFT corpus is filtered to pre-2000 content, anchoring short-answer
distributions and confounding the staircase (Trump ≤ 0.10 even in 2019–2024 vintages).
Base models carry the pretraining vintage without that confound. Cloze prompts, forced
choice scored on the FIRST TOKEN only (fixes the multi-token length artifact that favored
"Rio de Janeiro").

Usage: venv/bin/python 04_base_staircase.py --vintages 19991231,...
Output: derived/base_staircase.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
import json
import sys
from pathlib import Path

import tiktoken
import torch
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
OUT = HERE / "derived"
OUT.mkdir(exist_ok=True)
OUTFILE = OUT / "base_staircase.jsonl"

FACTS = [
    {"id": "president",
     "cloze": "The President of the United States is",
     "candidates": ["Clinton", "Bush", "Obama", "Trump", "Biden"]},
    {"id": "olympics",
     "cloze": "The most recent Summer Olympic Games were held in",
     "candidates": ["Atlanta", "Sydney", "Athens", "Beijing", "London",
                    "Rio", "Tokyo", "Paris"]},
]


def load_module(repo: str):
    path = hf_hub_download(repo_id=repo, filename="ChronoGPT_inference.py")
    spec = importlib.util.spec_from_file_location("chronogpt_base", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chronogpt_base"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintages", required=True)
    args = ap.parse_args()
    vintages = args.vintages.split(",")

    import os
    device = torch.device("cpu" if os.environ.get("FORCE_CPU") else ("mps" if torch.backends.mps.is_available() else "cpu"))
    tok = tiktoken.get_encoding("gpt2")
    base = f"manelalab/chrono-gpt-v1-{vintages[0]}"
    mod = load_module(base)
    config = torch.load(hf_hub_download(repo_id=base, filename="config.pt"),
                        map_location="cpu")
    model = mod.ChronoGPT(**config).to(device).half().eval()
    print(f"base model built on {device}", flush=True)

    done = set()
    if OUTFILE.exists():
        for line in OUTFILE.read_text().splitlines():
            r = json.loads(line)
            done.add((r["vintage"], r["fact"]))

    for vint in vintages:
        repo = f"manelalab/chrono-gpt-v1-{vint}"
        state = torch.load(hf_hub_download(repo_id=repo, filename="pytorch_model.bin"),
                           map_location="cpu")
        with torch.no_grad():
            model.load_state_dict(state)
        del state
        gc.collect()
        print(f"=== {repo} ===", flush=True)

        with OUTFILE.open("a") as f:
            for fact in FACTS:
                if (vint, fact["id"]) in done:
                    continue
                pids = tok.encode(fact["cloze"], allowed_special={"<|endoftext|>"})
                ids = torch.tensor([pids], device=device)
                with torch.no_grad():
                    logits = model(ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
                first = {c: tok.encode(" " + c)[0] for c in fact["candidates"]}
                lps = {c: lp[t].item() for c, t in first.items()}
                mx = max(lps.values())
                ps = {c: torch.exp(torch.tensor(v - mx)).item() for c, v in lps.items()}
                z = sum(ps.values())
                ps = {c: round(v / z, 4) for c, v in ps.items()}
                top = max(ps, key=ps.get)
                f.write(json.dumps({
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "vintage": vint, "fact": fact["id"], "top": top, "probs": ps,
                }) + "\n")
                f.flush()
                print(f"  {fact['id']}: {top} {ps[top]:.2f}", flush=True)
    print("base staircase complete →", OUTFILE)


if __name__ == "__main__":
    main()
