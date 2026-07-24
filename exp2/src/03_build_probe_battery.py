"""Generate ../prompts/probe_battery.json (static artifact; rerun only to change design).

Families:
  A. index_recall (48): GJY-style date-only recall of S&P 500 monthly direction,
     2020-01..2023-12 — brackets the Sept-2021 gpt-4-0613 cutoff on both sides.
  B. event_recall (12): dated events straddling the cutoff — 4 pre-cutoff controls
     (model should know) + 8 post-cutoff (model should not).
  C. self_report (2): the model's own claimed cutoff (unreliable; recorded anyway).
"""
from __future__ import annotations

import calendar
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "prompts" / "probe_battery.json"

probes = []

for year in range(2020, 2025):
    for month in range(1, 13):
        label = f"{calendar.month_name[month]} {year}"
        probes.append({
            "id": f"idx_{year}{month:02d}",
            "family": "index_recall",
            "question": (
                f"In {label}, did the S&P 500 index end the month higher or lower than it "
                f"ended the previous month? Answer with exactly one word: higher, lower, or unknown."
            ),
            "max_tokens": 3,
        })

EVENTS = [
    # pre-cutoff controls (before Sept 2021)
    ("ev_gamestop", "In which month and year did the GameStop short squeeze occur? Answer with month and year only."),
    ("ev_archegos", "In which month and year did Archegos Capital Management collapse? Answer with month and year only."),
    ("ev_coinbase", "In which month and year did Coinbase list on Nasdaq via direct listing? Answer with month and year only."),
    ("ev_evergrande", "Which large Chinese property developer's debt crisis dominated financial news in September 2021? Answer with the company name only."),
    # post-cutoff (after Sept 2021)
    ("ev_ukraine", "In which month and year did Russia begin its full-scale invasion of Ukraine? Answer with month and year only, or 'unknown'."),
    ("ev_twitter", "Which person completed an acquisition of Twitter in October 2022? Answer with the name only, or 'unknown'."),
    ("ev_ftx", "Which cryptocurrency exchange filed for bankruptcy in November 2022? Answer with the company name only, or 'unknown'."),
    ("ev_ukpm", "Who became UK Prime Minister in October 2022? Answer with the name only, or 'unknown'."),
    ("ev_svb", "In which month and year did Silicon Valley Bank fail? Answer with month and year only, or 'unknown'."),
    ("ev_cs", "Which bank agreed to acquire Credit Suisse in March 2023? Answer with the bank name only, or 'unknown'."),
    ("ev_chatgpt", "In which month and year was ChatGPT first released to the public? Answer with month and year only, or 'unknown'."),
    ("ev_fed75", "In which month and year did the Federal Reserve deliver its first 75 basis point rate hike since 1994? Answer with month and year only, or 'unknown'."),
]
for pid, q in EVENTS:
    probes.append({"id": pid, "family": "event_recall", "question": q, "max_tokens": 20})

probes.append({
    "id": "self_cutoff_1", "family": "self_report",
    "question": "What is your training data cutoff? Answer with a month and year only.",
    "max_tokens": 12,
})
probes.append({
    "id": "self_cutoff_2", "family": "self_report",
    "question": "What is the most recent date of real-world events you have knowledge of? Answer with a month and year only.",
    "max_tokens": 12,
})

OUT.write_text(json.dumps({"version": "1.1", "probes": probes}, indent=1))
print(f"wrote {OUT.name}: {len(probes)} probes "
      f"({sum(1 for p in probes if p['family'] == 'index_recall')} index_recall, "
      f"{sum(1 for p in probes if p['family'] == 'event_recall')} event_recall, "
      f"{sum(1 for p in probes if p['family'] == 'self_report')} self_report)")
