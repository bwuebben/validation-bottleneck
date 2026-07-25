# Exp 1 — ANALYSIS SPEC (pre-registered before any scoring)

*Written 2026-07-23, AFTER the proposal corpus was registered (558bd284) and BEFORE any
matching output or return data is examined. The git commit of this file freezes the tests.*

## Stage 1 — Matching (LLM-judge)

- Judge: **gpt-4.1-2025-04-14**, temperature 0, seed 42. One call per generation file
  (10 proposals per call). Judge receives the 212-line OSAP library (acronym: description)
  and the 10 proposals (name + construction + rationale), and returns per proposal:
  `{"match": <ACRONYM>|null, "strength": "exact"|"close"|"related"|"none", "reason": ...}`.
  Instructions: match on **construction/idea**, not on name or fame; "exact" = same
  construction; "close" = same core signal, different implementation details; "related" =
  same theme, different signal. Raw judge responses archived (`corpus_matching/`).
- **Matched** below means strength ∈ {exact, close}.
- Audit: 40 RNG-sampled (seed 7) judge decisions reserved for BJW eyeball before the
  results are treated as final; disagreement rate reported.

## Stage 2 — Test (a): regurgitation by vintage (descriptive)

For each (model, vintage v): shares of the 110 proposals that are (i) matched to a
predictor with SignalDoc publication Year > v ("post-date match" — the leak), (ii) matched
with Year ≤ v (claimed-novel-but-already-published), (iii) related-only, (iv) unmatched.
**Enrichment ratio** = post-date matched share ÷ library base rate (fraction of the 212
with Year > v). Report by model × vintage (dose-response across vintages).

## Stage 3 — Test (b): selection on future returns (THE novel test)

- Unit: OSAP predictor j × vintage v × model m, restricted to predictors with Year_j > v
  (the post-date pool), INCLUDING zeros (post-date predictors never proposed).
- `count_jvm` = number of proposals (of the 110) matched (exact/close) to j.
- **Primary future-performance regressor:** `fut_jv` = annualized mean OSAP long-short
  return of j over **Feb(v) → Dec 2024** (dp.factors OSAP monthly LS).
- Controls: `pre_jv` = annualized mean LS return over available months strictly before
  Jan(v) (0 with a missing-dummy if none); `log(1+GScholarCites202509)`; publication year.
- **Primary specification:** pooled OLS `count_jvm ~ fut_jv + pre_jv + pre_missing +
  log_cites + pub_year`, with model × vintage fixed effects, HC1 robust SEs clustered by
  predictor. **The claim tested: β(fut) > 0 — proposal frequency loads on returns realized
  strictly after the roleplay date.** Secondary: Poisson with the same RHS; rank version:
  Spearman(count, fut) within (model, vintage), combined by Fisher.
- Robustness (pre-listed, no others without flagging): fut window Feb(v) → Dec of
  publication year (pre-publication-only future returns); matched = exact-only;
  greedy-run-only subsample; per-model estimates.

## Stage 4 — Effective-M (descriptive)

Within (model, vintage): distinct matched acronyms / total matched proposals; plus
name-level dedup rate of unmatched proposals (case-folded exact name match). Report the
count distribution's concentration (top-5 share).

## Interpretation discipline

Test (a) alone cannot separate leak from convergent rediscovery of *ideas*; test (b) can —
future *returns* are not derivable from pre-date information. The vintage-model control
arm (ChronoGPT, README) later provides the zero-by-construction baseline. Negative/near-
zero β(fut) with tight CIs is a publishable finding and will be reported identically.

---

## Audit closure (2026-07-24)

Stage 1's reserved audit executed: the RNG(7) 40-decision sample (regenerated over all
2,200 judged proposals, four models) was reviewed by the author. Disagreement rate: 0/40.
The judge's category-boundary wobble on customer-concentration items (related vs none)
falls entirely within the unmatched pool and does not affect any scored quantity.
**Audit PASSED — the Experiment 1 results are final.**

---

## Amendment 2 (2026-07-25) — third-lineage arm: Anthropic claude-opus-4-8

Registered BEFORE any claude-opus-4-8 generation call (the commit of this amendment
precedes the first request; manifest timestamps verify).

- **Model:** `claude-opus-4-8` (Anthropic Messages API) — a third independent training
  lineage after OpenAI and Meta. **Experiment 1 only:** the model's reported training
  cutoff (2025) leaves no evaluable post-cutoff factor-data window, so no Experiment 2
  universe is generated for it.
- **Prompts:** the registered prompts verbatim (same `task_roleplay.txt`, same SYSTEM;
  `prompt_sha256` unchanged).
- **Sampling deviation (flagged):** this API accepts neither `temperature` nor `seed`
  (removed on this model family); thinking is disabled by omission. The registered
  1-greedy + 10-seeded-samples design becomes **11 runs at provider-default sampling**.
  File labels keep the `seed00–seed10` slot as run indices only; `meta` records
  `temperature: null` and `sampling: provider_default`. The analysis pools all 110
  proposals per vintage, unchanged.
- **Cutoff wall:** not used by any Experiment-1 statistic (post-date pools are defined by
  SignalDoc publication year vs. roleplay vintage). For the panel table the vendor-reported
  cutoff (2025) is recorded; it postdates the latest roleplay vintage (2010) by ≥15 years.
  No probe battery is run for this arm (logprobs are unavailable regardless; the
  answer-based fallback would apply if a wall were ever needed).
- **Judge and tests:** unchanged — gpt-4.1-2025-04-14, temperature 0, same instructions;
  tests (a), (b), and effective-M identical. Supplemental audit: 10 RNG(seed 7) judge
  decisions from this arm reserved for author review, reported alongside the original 40.
- **Interpretation registered up front:** a positive or less-negative tilt for a
  2025-cutoff model does not indicate foresight — post-2010 factor performance through
  2024 lies inside its training corpus, so any tilt improvement measures better
  exploitation of the in-corpus performance record. The arm is a capability
  dose-response on the selection variable, not a forecast test.

### Amendment 2 — audit closure (2026-07-25)

The 10-row RNG(7) supplemental sample from the claude-opus-4-8 arm
(`derived/audit_sample_claude.jsonl`) was reviewed by the author: **PASSED, 0/10
disagreement** ("opus audit pass is good"). Five-model results are final.
