"""ChronoGPT LAP staircase — instrument certification for the wall-setting probe.

For each chronologically consistent vintage (manelalab/chrono-gpt-instruct-v1-YYYY1231) and
each month, ask the index-direction question from the exp2 probe battery and score a
FORCED-CHOICE readout: sequence logprob of each candidate ("higher"/"lower"/"unknown")
appended to the model's own Alpaca-style prompt format. Normalized LAP =
(p_high + p_low) / Σ. Forced choice is robust to weak instruction-following at 1.5B — it
reads what is in the training corpus, not the model's chat manners. Expected: recall
collapses at each vintage's own wall — a staircase certifying the instrument that froze
exp2/WALLS.md.

Plumbing per the authors' tutorial (LinyingLyu/ChronoGPT): custom nanoGPT-class model in
the repo's ChronoGPT_instruct.py; tiktoken gpt2 tokenizer; one model object, per-vintage
state_dict swaps.

Usage:
  venv/bin/python 01_lap_staircase.py --vintages 20201231 --start 2019-06 --end 2022-06
"""
from __future__ import annotations

import argparse
import calendar
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
OUTFILE = OUT / "lap_staircase.jsonl"

SYSTEM = """You are ChronoGPT, a large language model trained by ManelaLab at WashU.
    Below is an instruction that describes a task.
    Write a response that appropriately completes the request."""
CANDIDATES = ["higher", "lower", "unknown"]


def question(y: int, m: int) -> str:
    return (f"In {calendar.month_name[m]} {y}, did the S&P 500 index end the month higher "
            f"or lower than it ended the previous month? Answer with exactly one word: "
            f"higher, lower, or unknown.")


def months(start: str, end: str):
    y, m = int(start[:4]), int(start[5:7])
    ye, me = int(end[:4]), int(end[5:7])
    while (y, m) <= (ye, me):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def load_module(repo: str):
    path = hf_hub_download(repo_id=repo, filename="ChronoGPT_instruct.py")
    spec = importlib.util.spec_from_file_location("chronogpt_instruct", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chronogpt_instruct"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintages", required=True, help="comma-separated YYYYMMDD suffixes")
    ap.add_argument("--start", default="1998-01")
    ap.add_argument("--end", default="2024-12")
    args = ap.parse_args()
    vintages = args.vintages.split(",")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = tiktoken.get_encoding("gpt2")
    base_repo = f"manelalab/chrono-gpt-instruct-v1-{vintages[0]}"
    mod = load_module(base_repo)
    config = torch.load(hf_hub_download(repo_id=base_repo, filename="config.pt"),
                        map_location="cpu")
    model = mod.ChronoGPT(**config).to(device).half().eval()
    print(f"model built on {device}; vintages: {vintages}", flush=True)

    done = set()
    if OUTFILE.exists():
        for line in OUTFILE.read_text().splitlines():
            r = json.loads(line)
            done.add((r["vintage"], r["month"]))

    cand_ids = {c: tok.encode(" " + c) for c in CANDIDATES}

    for vint in vintages:
        repo = f"manelalab/chrono-gpt-instruct-v1-{vint}"
        bin_path = hf_hub_download(repo_id=repo, filename="pytorch_model.bin")
        state = torch.load(bin_path, map_location="cpu")
        with torch.no_grad():
            model.load_state_dict(state)
        del state
        gc.collect()
        print(f"=== {repo} loaded ===", flush=True)

        with OUTFILE.open("a") as f:
            for y, m in months(args.start, args.end):
                key = f"{y}-{m:02d}"
                if (vint, key) in done:
                    continue
                prompt = (f"\n\n### Instruction:\n{SYSTEM}\n{question(y, m)}"
                          f"\n\n### Input:\n### Response:\n")
                pids = tok.encode(prompt, allowed_special={"<|endoftext|>"})
                lps = {}
                for c, cids in cand_ids.items():
                    ids = torch.tensor([pids + cids], device=device)
                    with torch.no_grad():
                        logits = model(ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    logprobs = torch.log_softmax(logits[0].float(), dim=-1)
                    lp = sum(logprobs[len(pids) - 1 + i, t].item()
                             for i, t in enumerate(cids))
                    lps[c] = lp
                mx = max(lps.values())
                ps = {c: torch.exp(torch.tensor(v - mx)).item() for c, v in lps.items()}
                z = sum(ps.values())
                ps = {c: v / z for c, v in ps.items()}
                f.write(json.dumps({
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "vintage": vint, "month": key,
                    "p_higher": round(ps["higher"], 4), "p_lower": round(ps["lower"], 4),
                    "p_unknown": round(ps["unknown"], 4),
                    "lap_forced": round(ps["higher"] + ps["lower"], 4),
                }) + "\n")
                f.flush()
        print(f"  {vint} complete", flush=True)
    print("staircase run complete →", OUTFILE)


if __name__ == "__main__":
    main()
