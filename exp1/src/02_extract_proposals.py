"""Flatten the registered Exp 1 corpus into proposals.jsonl (one line per proposal)."""
from __future__ import annotations

import json
import re
from pathlib import Path

EXP1 = Path(__file__).resolve().parent.parent
CORPUS = EXP1 / "corpus"
OUT = EXP1 / "derived"
OUT.mkdir(exist_ok=True)


def strip_fences(c: str) -> str:
    c = c.strip()
    c = re.sub(r"^```(json)?\s*", "", c)
    c = re.sub(r"\s*```$", "", c)
    return c


def main() -> None:
    rows = []
    for f in sorted(CORPUS.glob("*/y*_seed*.json")):
        if f.name.startswith("smoke_"):
            continue
        env = json.loads(f.read_text())
        meta = env["meta"]
        resp = env["response"]
        if isinstance(resp.get("content"), list):  # anthropic shape
            content = "".join(b.get("text", "") for b in resp["content"]
                              if b.get("type") == "text")
        else:
            content = resp["choices"][0]["message"]["content"]
        try:
            j = json.loads(content)
        except json.JSONDecodeError:
            j = json.loads(strip_fences(content))
        for i, p in enumerate(j["predictors"]):
            rows.append(
                {
                    "proposal_id": f"{f.parent.name}|{f.stem}|{i}",
                    "model": meta["model_requested"],
                    "vintage": meta["roleplay_year"],
                    "seed": meta["seed"],
                    "temperature": meta["temperature"],
                    "file": str(f.relative_to(EXP1)),
                    "idx": i,
                    "name": str(p.get("name", "")),
                    "construction": str(p.get("construction", "")),
                    "rationale": str(p.get("rationale", "")),
                    "higher_returns_for": str(p.get("higher_returns_for", "")),
                }
            )
    out = OUT / "proposals.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {out.relative_to(EXP1)}: {len(rows)} proposals")


if __name__ == "__main__":
    main()
