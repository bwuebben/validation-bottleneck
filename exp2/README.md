# Exp 2 — Cutoff-Vintage Forward Test: GENERATION STAGE

*Created 2026-07-23. Queue-jumped per BJW decision: **gpt-4-0613 (Sept 2021 cutoff, the
longest-OOS generator) retires from the OpenAI API 2026-10-23** — generation, archival, and
effective-cutoff probing must complete before then. Design per `../REBUILD.md` (Pillar 4) and
`../FEASIBILITY.md` (§1 model panel, §2.6 portfolio-level DSL verdict).*

## What this stage produces

The **archived generation corpus**: every raw model response, with full call metadata,
committed to git. The commit hash is the **pre-registration timestamp** — no evaluation
against post-cutoff data may run until the corpus is committed. The corpus outlives the
models (the APIs are non-deterministic and the snapshots get retired; the archive is the
reproducibility artifact — FEASIBILITY §1.4).

## DSL v1.0 (portfolio-level)

A hypothesis is a composite over the **212 OSAP predictor long-short monthly return series**
(`primitives/primitives.json`, built from `dp.signals.open_asset_pricing_doc()`; publication
years 1973–2016, so the library text is pre-cutoff for every panel model):

```
expression := { terms: [{weight, primitive}] (2..4 terms),
                regime: R | null,
                vol_target: bool }
weight ∈ {-1.0, -0.5, +0.5, +1.0}
primitive ∈ {212 OSAP acronyms}
R ∈ {VIX_ABOVE_TRAILING_MEDIAN, VIX_BELOW_TRAILING_MEDIAN,
     TERM_SPREAD_POSITIVE, TERM_SPREAD_NEGATIVE,
     MKT_TRAILING_12M_UP, MKT_TRAILING_12M_DOWN}
```

**Evaluation semantics (ex-ante only, fixed now, before any generation):** composite return
in month t = Σᵢ wᵢ·LSᵢ,t, held only if the regime indicator was TRUE at the end of month
t−1 (else zero), then (if vol_target) scaled to 10% annualized using trailing 36-month
volatility measured through month t−1. Regime data: VIX (trailing median through t−1),
10y–3m Treasury term spread sign at t−1, market trailing-12m return sign at t−1 — all via
`dp.macro.*` / `dp.factors.fama_french`.

**M is exact.** The grammar's cardinality is computable in closed form (choose 2–4 of 212
primitives × 4 weights each × 7 regime states × 2 vol flags); the paper reports it. Every
generated hypothesis is one draw from this enumerable space — multiple-testing accounting
is exact for the first time (REBUILD Pillar 2).

**Auxiliary predictions (protocol safeguard #3):** each hypothesis must carry ≥3
falsifiable auxiliary predictions implied by its single mechanism, ≥2 distinct types, from
the vocabulary: `regime_interaction`, `size_segment` (via JKP size-cap factors),
`international` (via JKP regions), `correlation`, `subperiod_consistency`. All testable
with `dp.factors.*` — no vendor data.

## Model panel (this run: the October-deadline assets)

| Model | Cutoff | OOS window | Status |
|---|---|---|---|
| gpt-4-0613 | Sept 2021 | ~4.75y | **retires 2026-10-23 — this run's reason** |
| gpt-4o-2024-08-06 | Oct 2023 | ~2.7y | at-risk (no posted date) |
| gpt-4.1 | Jun 2024 | ~2y | at-risk |

Open-weight backbone (Llama-2-70B, Llama-3.1-70B, Qwen2.5-72B, OLMo 2-32B) runs later via
hosted inference — permanent weights, no deadline. This Mac (M5, 24GB) cannot host 70B
locally.

## Sampling policy (FEASIBILITY §1.4)

Per model × per prompt variant: **1 greedy run** (temperature 0, seed 42) + **20 sampled
runs** (temperature 0.8, seeds 1–20). Two prompt variants: `v1_baseline`,
`v2_diversity` (anti-mode-collapse instruction — the cross-sample proposal distribution is
itself the effective-M measurement). 42 calls/model; K=8 hypotheses per call →
target ≤ 336 hypothesis-draws per model before dedup.

Seeds are best-effort on OpenAI; `system_fingerprint` recorded per call. max_tokens=4800
(fits gpt-4-0613's 8k context with the ~3k prompt). No `response_format` anywhere —
gpt-4-0613 doesn't support it, and parameters are held identical across models for
comparability.

## Contamination discipline for the prompts

- No dates, events, or facts after Sept 2021 anywhere in system/task text.
- Library = OSAP acronym + short description only (publication years ≤2016).
- No "pretend it is year X" roleplay — that is Exp 1's (invalid-by-design) treatment; here
  the wall is the model's actual training cutoff and evaluation is strictly post-cutoff.

## Archival schema

`corpus/<model>/<variant>_seed<NN>_T<temp>.json` — envelope:
`{meta: {utc, model_requested, prompt_variant, prompt_sha256, seed, temperature,
max_tokens, harness_version}, response: <full API response incl. system_fingerprint,
usage, choices>}`. Plus `corpus/manifest.jsonl`, one line per call. Raw responses are
never edited; parsing/repair happens downstream at evaluation time.

## Running

```bash
# API key: put OPENAI_API_KEY=sk-... in exp2/.env (gitignored via repo .env* rule)
cd exp2/src
PY=~/git/data_platform/.venv/bin/python
$PY 00_build_primitives.py          # done 2026-07-23 (rerun only on OSAP release change)
$PY 01_generate.py --list-models    # verify gpt-4-0613 still served
$PY 01_generate.py --model gpt-4-0613 --smoke   # 1 cheap auth/format check
$PY 01_generate.py --model gpt-4-0613           # full 42-call run
$PY 01_generate.py --model gpt-4o-2024-08-06
$PY 01_generate.py --model gpt-4.1
$PY 02_cutoff_probe.py --model gpt-4-0613       # 62-probe effective-cutoff battery
```

Cost estimate: gpt-4-0613 ≈ $15–20; 4o/4.1 well under $10 combined. Probes: cents.

Analysis stages, in order (all read the archived corpus; none calls a model API):

```bash
$PY 04_freeze_walls.py              # walls from probe data alone, committed before results
$PY 05_evaluate.py                  # the evaluation ladder -> derived/results_perf.csv
$PY 06_randomization.py             # Amendment 3: grammar randomization null (seed 42)
$PY 07_aux_positive_control.py      # Amendment 4: documented-true auxiliary positive control
$PY 08_grammar_null_hist.py         # best-of-M null CDF for the paper's model-vs-monkey figure
```

`08_grammar_null_hist.py` re-runs the Amendment 3 pool and adds nothing to the registered
analysis: same pool, same seed, same windows, same pipeline. It exists so the figure is
reproducible from committed data. It reproduces the archived pool exactly (8,099 / 7,749 /
7,749 evaluable) and the archived null medians to four decimals; bootstrap percentiles agree
with `randomization_test.json` to within resampling noise, leaving the reported 9th / 58th /
16th percentiles unchanged.

## After this stage (not this run)

1. Commit corpus (= registration). 2. Effective-cutoff verification write-up (probe battery
here + Cheng et al. `dated_data` on the open-weight panel later). 3. Evaluation harness:
composite construction + auxiliary tests via `dp.factors` on post-cutoff months only.
4. Open-weight generation via hosted APIs. 5. Effective-M analysis across the 20-sample
draws (dedup at the expression level; Si–Yang–Hashimoto comparison).
