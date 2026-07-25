"""ChronoGPT dated-fact staircase — size-robust wall certification.

The index-direction probe is flat at 1.5B (small models never memorized monthly index
paths; see 01_lap_staircase.py smoke finding). This battery uses iconic dated facts that
any English corpus contains, scored by forced choice over candidate answers; the winning
candidate should step at known dates across vintages, certifying that each vintage's
knowledge ends at its wall.

Facts: (1) sitting U.S. president (steps 2001, 2009, 2017, 2021);
       (2) most recent Summer Olympics host city (steps 1996, 2000, 2004, ..., 2024).

Usage: venv/bin/python 02_dated_facts.py --vintages 19991231,20041231,...
Output: derived/dated_facts.jsonl
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
OUTFILE = OUT / "dated_facts.jsonl"

SYSTEM = """You are ChronoGPT, a large language model trained by ManelaLab at WashU.
    Below is an instruction that describes a task.
    Write a response that appropriately completes the request."""

FACTS = [
    {"id": "president",
     "question": "Who is the current President of the United States? Answer with the last name only.",
     "candidates": ["Clinton", "Bush", "Obama", "Trump", "Biden"]},
    {"id": "olympics",
     "question": "In which city were the most recent Summer Olympic Games held? Answer with the city name only.",
     "candidates": ["Atlanta", "Sydney", "Athens", "Beijing", "London",
                    "Rio de Janeiro", "Tokyo", "Paris"]},
]


def load_module(repo: str):
    path = hf_hub_download(repo_id=repo, filename="ChronoGPT_instruct.py")
    spec = importlib.util.spec_from_file_location("chronogpt_instruct", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chronogpt_instruct"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintages", required=True)
    args = ap.parse_args()
    vintages = args.vintages.split(",")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = tiktoken.get_encoding("gpt2")
    base = f"manelalab/chrono-gpt-instruct-v1-{vintages[0]}"
    mod = load_module(base)
    config = torch.load(hf_hub_download(repo_id=base, filename="config.pt"),
                        map_location="cpu")
    model = mod.ChronoGPT(**config).to(device).half().eval()
    print(f"model built on {device}", flush=True)

    done = set()
    if OUTFILE.exists():
        for line in OUTFILE.read_text().splitlines():
            r = json.loads(line)
            done.add((r["vintage"], r["fact"]))

    for vint in vintages:
        repo = f"manelalab/chrono-gpt-instruct-v1-{vint}"
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
                prompt = (f"\n\n### Instruction:\n{SYSTEM}\n{fact['question']}"
                          f"\n\n### Input:\n### Response:\n")
                pids = tok.encode(prompt, allowed_special={"<|endoftext|>"})
                lps = {}
                for cand in fact["candidates"]:
                    cids = tok.encode(" " + cand)
                    ids = torch.tensor([pids + cids], device=device)
                    with torch.no_grad():
                        logits = model(ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    lp = torch.log_softmax(logits[0].float(), dim=-1)
                    lps[cand] = sum(lp[len(pids) - 1 + i, t].item()
                                    for i, t in enumerate(cids)) / len(cids)
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
    print("dated-fact staircase complete →", OUTFILE)


if __name__ == "__main__":
    main()
