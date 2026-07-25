"""ChronoGPT roleplay-generation control arm — BATCHED local generation.

The blind baseline for Exp 1 test (b): a vintage-τ model roleplaying year Y ≥ τ is blind
by construction to all post-τ literature and returns, so its proposal distribution cannot
tilt toward post-τ winners. Runs the SAME roleplay prompt as the frontier arm (verbatim,
for instrument comparability), wrapped in ChronoGPT's Alpaca format; output parsing is
lenient downstream (1.5B JSON is imperfect; raw text is archived either way).

Batching: the 10 sampled runs of a vintage-year share one prompt and generate as a single
batch (greedy run separately at B=1), with per-row end-of-text tracking.

Usage: venv/bin/python 03_generate_control.py --vintages 19991231,20041231,20091231
Output: corpus/chrono-{vintage}/y{year}.json (11 generations per file + metadata)
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
EXP1 = HERE.parent
CORPUS = HERE / "corpus"
CORPUS.mkdir(parents=True, exist_ok=True)

SYSTEM = """You are ChronoGPT, a large language model trained by ManelaLab at WashU.
    Below is an instruction that describes a task.
    Write a response that appropriately completes the request."""
YEARS = [1990, 1995, 2000, 2005, 2010]
N_SAMPLED = 10
MAX_NEW = 800
CTX = 1792
EOS = 50256


def load_module(repo: str):
    path = hf_hub_download(repo_id=repo, filename="ChronoGPT_instruct.py")
    spec = importlib.util.spec_from_file_location("chronogpt_instruct", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chronogpt_instruct"] = mod
    spec.loader.exec_module(mod)
    return mod


def batched_generate(model, device, prompt_ids, batch: int, temperature: float,
                     seed: int) -> list[list[int]]:
    """Fully tensorized: no per-step host syncs (the naive loop's per-row .item()
    calls stall MPS); done-check synced once per 64 steps; decode after the loop."""
    torch.manual_seed(seed)
    ids = torch.tensor([prompt_ids] * batch, device=device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    eos = torch.full((batch,), EOS, dtype=ids.dtype, device=device)
    n0 = ids.shape[1]
    for step in range(MAX_NEW):
        if ids.shape[1] >= CTX:
            break
        with torch.no_grad():
            logits = model(ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        last = logits[:, -1, :].float()
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(last / temperature, dim=-1), 1).squeeze(1)
        else:
            nxt = last.argmax(dim=-1)
        nxt = torch.where(done, eos, nxt)
        done = done | (nxt == EOS)
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        if step % 64 == 63 and bool(done.all().item()):
            break
    rows = ids[:, n0:].cpu().tolist()
    out = []
    for row in rows:
        toks = []
        for t in row:
            if t == EOS:
                break
            toks.append(t)
        out.append(toks)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintages", required=True)
    args = ap.parse_args()
    vintages = args.vintages.split(",")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    tok = tiktoken.get_encoding("gpt2")
    task = (EXP1 / "prompts" / "task_roleplay.txt").read_text()
    base = f"manelalab/chrono-gpt-instruct-v1-{vintages[0]}"
    mod = load_module(base)
    config = torch.load(hf_hub_download(repo_id=base, filename="config.pt"),
                        map_location="cpu")
    model = mod.ChronoGPT(**config).to(device).half().eval()
    print(f"model built on {device}", flush=True)

    for vint in vintages:
        repo = f"manelalab/chrono-gpt-instruct-v1-{vint}"
        state = torch.load(hf_hub_download(repo_id=repo, filename="pytorch_model.bin"),
                           map_location="cpu")
        with torch.no_grad():
            model.load_state_dict(state)
        del state
        gc.collect()
        outdir = CORPUS / f"chrono-{vint}"
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"=== {repo} ===", flush=True)

        for year in YEARS:
            outfile = outdir / f"y{year}.json"
            if outfile.exists():
                continue
            user = task.replace("{YEAR}", str(year))
            prompt = f"\n\n### Instruction:\n{SYSTEM}\n{user}\n\n### Input:\n### Response:\n"
            pids = tok.encode(prompt, allowed_special={"<|endoftext|>"})
            t0 = dt.datetime.now(dt.timezone.utc)
            greedy = batched_generate(model, device, pids, 1, 0.0, 42)
            sampled = batched_generate(model, device, pids, N_SAMPLED, 0.8, year)
            gens = []
            for i, toks in enumerate(greedy + sampled):
                text = tok.decode(toks).split("<|endoftext|>")[0]
                gens.append({"run": "greedy" if i == 0 else f"sampled_{i}",
                             "temperature": 0.0 if i == 0 else 0.8,
                             "n_tokens": len(toks), "text": text})
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_text(json.dumps({
                "meta": {"utc": t0.isoformat(), "vintage": vint, "roleplay_year": year,
                         "model": repo, "prompt_tokens": len(pids),
                         "max_new": MAX_NEW, "seed_greedy": 42, "seed_sampled": year},
                "generations": gens}, indent=1))
            dt_s = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
            print(f"  y{year}: 11 generations in {dt_s:.0f}s", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
    print("control generation complete", flush=True)


if __name__ == "__main__":
    main()
