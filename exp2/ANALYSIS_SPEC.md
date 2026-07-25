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

---

## Amendment 1 — open-weight replication arm (2026-07-24, flagged per the Stage 4 rule;
committed BEFORE any post-wall return for this model was touched)

Model added: **meta-llama/Llama-3.3-70B-Instruct-Turbo** (Together serving; weights
permanently on HuggingFace; reported knowledge cutoff December 2023 per Meta's model card;
independent training lineage — the panel's one-vendor limitation mitigant). Wall frozen
from probe data under the UNCHANGED rule, using the pre-built answer-based fallback (the
provider returns no logprobs): **WALL = 2024-03**; OSAP window 2024-04 → 2024-12
(9 months); MIN_MONTHS = 8. All Stage 0–4 rules apply unchanged; the one malformed-JSON
generation call (1 of 42) is excluded at Stage 0 as an invalid response. For Experiment 1,
the same judge, matching protocol, and tests (a)/(b) extend to this model (Fisher
combination now over 20 model×vintage cells). No other changes.

---

## Amendment 3 (2026-07-25) — grammar randomization test (registered before running)

Purpose: the noise benchmark for the realized maximum t. The √(2lnM) expression used in
the paper is the leading-order extreme-value term, not E[max]; and composite t-statistics
are cross-sectionally dependent, which no closed form absorbs. The correct null is the
**grammar monkey**: does the model's selection of M expressions beat M uniform draws from
the same grammar, evaluated on the identical window and pipeline?

- **Pool:** R = 10,000 expressions drawn uniformly from the registered grammar, seed 42:
  k ~ U{2,3,4}; k distinct primitives uniform without replacement from the 212; each
  weight iid U{−1, −0.5, +0.5, +1}; regime ~ U{six regimes, NONE} (7 equally likely);
  vol_target ~ Bernoulli(1/2). No dedup (draws are iid; a monkey may repeat itself).
- **Evaluation:** the registered Stage-1/Stage-2 pipeline verbatim (same gating, vol
  targeting, sufficiency: ≥ MIN_MONTHS and ≥80% of the model's post-wall window; draws
  failing sufficiency for a model are excluded from that model's pool, count reported).
- **Null distribution of the maximum:** per model, B = 10,000 resamples with replacement
  of size M_model (registered distinct-M: 330 / 318 / 294) from the model's pool
  t-statistics; the resample maxima form the null. Report the realized max t, the null's
  mean and 50/90/95/99 percentiles, and the realized max's percentile in the null.
- **Role:** descriptive benchmark replacing the √(2lnM)-vs-max comparison in the text;
  the BY-FDR stage remains the confirmatory multiplicity control. Registered reading:
  a realized max at or below the null median indicates the generator's ranking carries
  no exploitable information relative to uniform sampling from its own grammar.

---

## Amendment 4 (2026-07-25) — auxiliary-battery positive control (registered before running)

Purpose: establish whether the auxiliary test types have power on the short post-wall
windows, by running the identical test mechanics on implications documented as true in
the published literature. Pinned ex ante; the list below is frozen at this commit.

**Test mechanics:** identical to the registered Stage-3 aux code (05_evaluate):
regime_interaction = sign of (mean in-regime − mean out) on post-wall raw LS months,
≥6 obs per side; correlation = |corr| vs 0.3 on ≥12 joint post-wall months, bound
above/below; subperiod_consistency = both halves of the post-wall window positive.
Windows: all three registered model windows (2022-04, 2024-01, 2024-03 walls → Dec 2024).

**Pinned implications (OSAP signalnames; direction; source):**
Regime interactions:
 R1 Mom12m lower in MKT_TRAILING_12M_DOWN — momentum follows up-markets, crashes after
    down-markets (Cooper–Gutierrez–Hameed 2004; Daniel–Moskowitz 2016)
 R2 Mom12m lower in VIX_ABOVE_TRAILING_MEDIAN — momentum weak in high-vol states
    (Barroso–Santa-Clara 2015)
 R3 STreversal higher in VIX_ABOVE_TRAILING_MEDIAN — reversal = liquidity provision,
    paid more in stress (Nagel 2012)
 R4 Illiquidity lower in VIX_ABOVE_TRAILING_MEDIAN — illiquid-minus-liquid realized
    returns fall in liquidity crises (Acharya–Pedersen 2005; Amihud 2002)
 R5 Size lower in VIX_ABOVE_TRAILING_MEDIAN — small caps underperform in
    flight-to-quality states
Correlations (|corr| ≥ 0.3 unless noted):
 C1 Mom12m × BM — strong negative value–momentum correlation
    (Asness–Moskowitz–Pedersen 2013)
 C2 Mom12m × LRreversal — long-run reversal is value-like/anti-momentum
    (De Bondt–Thaler 1985; AMP 2013)
 C3 BM × BMdec — near-duplicate constructions of one signal (structural)
 C4 GP × BM — profitability negatively correlated with value (Novy-Marx 2013)
 C5 Size × Illiquidity — size and illiquidity intertwined (Amihud 2002)
 C6 Accruals × AssetGrowth — accruals/investment kinship (Fairfield et al. 2003;
    Cooper–Gulen–Schill 2008)
Subperiod consistency (both halves positive — deliberately included although true
premia are underpowered here; the per-type pass-rate decomposition is the point):
 S1 Mom12m   S2 BM   S3 STreversal

**Registered reading:** per-type pass rates of documented-true implications on the same
windows. High structural-type rates with low subperiod rates would establish that the
battery's power resides in structure-testing implications; uniform ~50% rates would
establish the battery is uninformative at these horizons and Section 3.2's protocol
claim must be weakened accordingly. Either outcome is reported.
