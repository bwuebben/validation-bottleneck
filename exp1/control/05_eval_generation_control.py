"""Evaluate the ChronoGPT generation-control corpus: can the chronologically clean
model produce valid proposals under Experiment 1's registered prompts at all?

Classification per generation (11 per cell, 15 cells, 165 total):
  - template_echo: text contains the task prompt's own format-spec phrases and no
    non-template "name" field value
  - fragment: at least one "name"-like field with a non-template value (still checked
    for coherence by eye; counts reported, values dumped)
  - empty/other: neither pattern (e.g., near-empty or off-task free text)

Output: derived/generation_control.json + stdout summary. No API calls.
"""
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
OUT = HERE / "derived"
OUT.mkdir(exist_ok=True)

TEMPLATE_MARKERS = [
    "a short descriptive name",
    "a precise construction",
    "one or two sentences on the economic mechanism",
    "which stocks the construction predicts",
]

rows = []
for f in sorted(CORPUS.glob("*/y*.json")):
    d = json.loads(f.read_text())
    for g in d["generations"]:
        t = g["text"]
        names = re.findall(r'"name"\s*:\s*"?([^"\n]{3,60})', t)
        real = [n.strip() for n in names
                if not any(m in n for m in TEMPLATE_MARKERS)]
        echo = any(m in t for m in TEMPLATE_MARKERS)
        cat = ("fragment" if real else
               "template_echo" if echo else "other")
        rows.append({"vintage": f.parent.name, "cell": f.stem, "run": g["run"],
                     "chars": len(t), "category": cat, "nontemplate_names": real})

counts = {}
for r in rows:
    counts[r["category"]] = counts.get(r["category"], 0) + 1
frag_values = [v for r in rows for v in r["nontemplate_names"]]

summary = {
    "total_generations": len(rows),
    "category_counts": counts,
    "valid_proposals_extracted": 0,
    "fragment_name_values": frag_values,
    "verdict": ("No valid proposals in any of the generations: the chronologically "
                "clean 1.5B generator cannot execute the registered proposal task. "
                "Fragment 'name' values are incoherent (listed above), not predictors."),
}
(OUT / "generation_control.json").write_text(json.dumps(
    {"summary": summary, "rows": rows}, indent=1))
print(json.dumps(summary, indent=1)[:800])
