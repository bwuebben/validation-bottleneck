# Exp 2 — EVALUATION SPEC (pre-registered before any post-wall return is touched)

*Written 2026-07-24, after the corpus (1d833680) and walls (b16a69c9) were committed and
BEFORE any composite return is constructed. The git commit of this file freezes the tests.
Registered corpus: 1,008 hypotheses (336/model). Walls: gpt-4-0613 → 2022-04, gpt-4o →
2024-01, gpt-4.1 → 2025-02. Evaluation uses months STRICTLY AFTER each wall, through
2024-12 (OSAP data end).*

## Stage 0 — Universe, compliance, dedup (no return data)

- Parse all raw responses (fence-strip where needed). **Drop** any hypothesis whose
  expression references a primitive outside the 212, has weights outside {±1, ±0.5}, or
  has <2 or >4 terms (unevaluable as specified).
- **Compliance flags** (reported, not exclusions): ≥3 auxiliary predictions with ≥2
  distinct types from the vocabulary; well-formed regime; mechanism present.
- **Dedup / exact M:** expression signature = sorted (primitive, weight) pairs + regime +
  vol_target. Evaluation universe = **distinct signatures per model** (multiplicity
  recorded — the effective-M measurement). M_model = distinct count = the multiple-testing
  denominator.
- **gpt-4.1 is NOT EVALUABLE in v1**: its wall (2025-02) exceeds the OSAP data end
  (2024-12). Deferred to the next OSAP release; reported as such. v1 evaluates gpt-4-0613
  (post-wall window 2022-05 → 2024-12, 32 months) and gpt-4o (2024-02 → 2024-12, 11 months).

## Stage 1 — Composite construction (semantics frozen in README §DSL; parameters here)

- Composite monthly return r_t = Σᵢ wᵢ·LSᵢ,t using OSAP `op` long-short legs
  (`dp.factors.open_asset_pricing('op')`, port='LS', ret in monthly %).
- **Regime gate** (indicator measured at month-end t−1; if false, r_t = 0 — the LS book is
  self-financing, so flat = 0 excess):
  - VIX_ABOVE/BELOW_TRAILING_MEDIAN: month-end VIXCLS (last daily obs of the month,
    `dp.macro.fred('VIXCLS')`) vs the trailing **60-month** median of month-end values
    through t−1 (min 36 obs); strictly above = ABOVE, ties → BELOW.
  - TERM_SPREAD_POSITIVE/NEGATIVE: (rate_10y − rate_3m) at the last observation ≤
    month-end t−1 (`dp.macro.treasury_rates()`); positive = > 0.
  - MKT_TRAILING_12M_UP/DOWN: cumulative (Mkt−RF + RF) over months t−12…t−1 > 0
    (`dp.factors.fama_french('3-factor')`, % units).
- **Vol targeting** (if vol_target): scale month t by s_t = 10%/√12 ÷ trailing 36-month
  std of the gated composite through t−1 (min 24 obs, else s_t = 1); **s_t capped at 5**.
- Data sufficiency: a hypothesis is evaluable if all its primitives jointly cover ≥80% of
  the post-wall window and ≥24 months (0613) / ≥9 months (4o); composite months = months
  with all primitives present.

## Stage 2 — Primary performance test (post-wall months only)

Per distinct expression, as signed by the generating model: n, annualized mean and vol,
Sharpe, one-sided t = mean/(sd/√n), p from t(n−1). **The ladder** (survival counts
reported at each rung, per model):
1. mean > 0;
2. t > 2;
3. **Benjamini–Yekutieli FDR q = 0.05** within model across its M distinct expressions
   (dependence-robust, per REBUILD's Storey fix);
4. reference: t > √(2 ln M_model) (the extreme-value hurdle; also quote E[max t] ≈
   √(2 ln M) under the null for the realized M and window).
Secondary (0613 only, ≥24-month expressions): 6-factor alpha t (FF5 + momentum,
`dp.factors.fama_french`), post-wall regression.

## Stage 3 — Auxiliary-prediction falsification (the protocol's teeth)

Testable subset in v1 (size_segment and international require a JKP mapping — **not
testable in v1**, stated; counted as untestable, not as failures):
- regime_interaction: mean composite return in regime-true vs regime-false post-wall
  months (≥6 months in each leg, else untestable); prediction holds iff the point
  difference has the predicted sign.
- correlation: post-wall |corr(composite, primitive LS)| (≥12 joint months) vs the stated
  bound/threshold.
- subperiod_consistency: mean > 0 in BOTH halves of the post-wall window.
**Full survival** = rung 2 (t > 2) AND every testable auxiliary prediction holds.
Also reported: aux pass-rates unconditionally (do mechanisms' implications hold at all?).

## Stage 4 — Outputs

`derived/eval_universe.csv`, `derived/results_perf.csv`, `derived/results_summary.txt`:
per-model ladder table, t-stat distribution vs the √(2 ln M) reference, aux pass-rates,
full-survivor list (if any) with expressions and mechanisms verbatim. Either outcome is
the paper's Evidence section: survivors = the protocol demonstrated on the only valid
window; zero survivors = the validation bottleneck, measured. No other tests without a
flagged spec amendment.
