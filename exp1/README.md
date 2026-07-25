# Exp 1 — The Roleplay Leak: GENERATION STAGE

*Created 2026-07-23. The paper's demolition evidence (REBUILD Pillar 4, Tier B milestone):
prompt current models to roleplay a past date and propose "novel" cross-sectional predictors;
measure (a) regurgitation of the post-date literature and (b) the sharper test — proposals
overweighted toward factors in proportion to their post-date performance (selection on
future returns, which appears in NO prior paper). General-ML precedent that prompted
cutoffs leak on causally-related knowledge: EMNLP 2025 "Can Prompts Rewind Time?"
(LITSWEEP B18). gpt-4-0613 runs now for the same 2026-10-23 retirement reason as Exp 2.*

## Design

- **Roleplay vintages:** January 1990, 1995, 2000, 2005, 2010 — a dose-response curve
  (leak rate and future-performance tilt as a function of how much post-date literature
  the model has read).
- **Models:** gpt-4-0613 (deadline), gpt-4o-2024-08-06, gpt-4.1-2025-04-14; open-weight
  panel later.
- **Sampling:** per model × vintage: 1 greedy (T=0, seed 42) + 10 sampled (T=0.8, seeds
  1–10) = 55 calls/model. K=10 predictors per call → up to 550 proposals per model.
- **The prompt deliberately roleplays a past date** — the leak is the measurement (the
  inverse of Exp 2's discipline, by design). It does NOT mention OSAP or any predictor
  library: proposals must be free recall, matched offline afterward.

## Measurement (later stage, after corpus commit)

1. **Match** each proposal to the 212 OSAP predictors (SignalDoc definitions) — LLM-judge
   matching with a human-audited subsample; unmatched proposals classified separately.
2. **(a) Regurgitation:** fraction of proposals matching predictors first PUBLISHED after
   the roleplay date (SignalDoc `Year`), by vintage and model.
3. **(b) Selection on future returns:** among matched proposals, regress proposal frequency
   on the predictor's realized post-vintage long-short performance (OSAP monthly returns) —
   controlling for pre-vintage performance and citation counts (GScholarCites in
   SignalDoc). A positive future-performance tilt is knowledge the model cannot have at
   the roleplay date. THE novel test.
4. Effective-M of proposals per vintage (dedup) — mode-collapse read.

## Planned extension: the vintage-model CONTROL ARM (deadline-free, BJW 2026-07-23)

Run the same roleplay prompts on **ChronoGPT annual vintages** (manelalab on HuggingFace,
~1999–2024; 1.5B, local inference on the M5). A vintage-2000 model roleplaying 2000 cannot
regurgitate post-2000 literature or tilt toward future winners BY CONSTRUCTION — its
proposal distribution is the leakage-free baseline for test (b): the frontier models'
post-date-performance tilt is measured against a control whose tilt must be exactly zero.
State the capability confound plainly (1.5B proposals are crude; the comparison is about
the TILT, not proposal quality). Bonus runs: (i) the 74-probe LAP battery across all
vintages — the LAP collapse should staircase exactly at each vintage's wall, certifying
the instrument used to freeze exp2's walls; (ii) Cheng-et-al dated_data verification on
open weights per FEASIBILITY §1.5. Also check release status of Kelly–Malamud–Schwab–Xu's
4B point-in-time LM (1T chronologically filtered tokens) as a stronger vintage generator.
The capability–validity tradeoff itself (clean XOR capable) is a paper point — engage
ChronoLLM's "retrain vintage models" route as the opposite-route competitor per LITSWEEP.

## Archival

Same schema as exp2: `corpus/<model>/y<YEAR>_seed<NN>_T<temp>.json` + `corpus/manifest.jsonl`;
raw responses never edited; corpus commit = registration of the proposal set before any
matching/scoring is run.

## Running

```bash
cd exp1/src
PY=~/git/data_platform/.venv/bin/python
$PY 01_generate.py --model gpt-4-0613 --smoke
$PY 01_generate.py --model gpt-4-0613
$PY 01_generate.py --model gpt-4o-2024-08-06
$PY 01_generate.py --model gpt-4.1-2025-04-14
```
Key: exp2/.env (shared loader). Cost ≈ $10 total, dominated by gpt-4-0613.
