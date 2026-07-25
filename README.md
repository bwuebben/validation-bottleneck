# The Validation Bottleneck: Alpha Discovery When Hypotheses Are Free

Replication archive for:

> Wuebben, B. J. (2026). *The Validation Bottleneck: Alpha Discovery When Hypotheses Are
> Free.* Working paper. (`validation-bottleneck.pdf` in this repository.)

The paper argues that large language models have made investment hypotheses effectively
free, moving the binding constraint of quantitative research to out-of-sample time the
generating model has provably not seen: the model's (verified) training cutoff is the only
valid evaluation boundary. Two pre-registered experiments on public data supply the
evidence. In Experiment 1, five models spanning three training lineages (OpenAI, Meta,
Anthropic), asked to roleplay past dates (1990–2010) and propose "novel" cross-sectional
predictors, return that literature anyway — 26–55% of their "novel" proposals match
predictors already in the library, i.e. exactly what they were told to avoid. Convergent
rediscovery could explain that; it cannot explain the sign: proposal frequency is
*negatively* related to post-date factor performance (Fisher-combined z = −3.88, negative in
20 of 25 model-vintage cells — four of five for every one of the five models), which is
information no truthful roleplayer could hold. The
models reproduce prominence, not returns. In Experiment 2, models with recall-verified cutoffs
generated 1,336 composite factors across four models — 1,000 of them evaluable — inside an
enumerable grammar (making the trial count exact; M = 330 for the longest-window model)
and were evaluated strictly after their cutoffs: nothing survives Benjamini–Yekutieli
false-discovery control; against 10,000 uniform draws from the same grammar the best
machine-selected composite lands at the 9th/58th/16th percentile of the best-of-M null;
and the models' mechanism-implied auxiliary predictions hold at 39% pooled (240/614) against
70% (14/20) for implications already documented as true put through the same battery — or 46%
(200/437) against 86% (12/14) restricted to the one window on which the structural test types
can run. Both comparisons are reported in the paper; the gap is wider on the like-for-like one.

## Repository layout

```
validation-bottleneck.pdf     the paper
exp1/                         Experiment 1 — the roleplay leak
  ANALYSIS_SPEC.md            pre-registered analysis specification
  prompts/                    the roleplay prompt (verbatim)
  src/                        generation, extraction, matching, scoring code
  corpus/                     every raw model response, with full call metadata
  corpus_matching/            every raw judge response
  derived/                    proposals, matches, results tables, audit sample
exp2/                         Experiment 2 — the cutoff-vintage forward test
  ANALYSIS_SPEC.md            pre-registered evaluation specification
  prompts/                    generation prompts + the 74-probe recall battery
  src/                        generation, probing, wall-freezing, evaluation code
  corpus/                     every raw model response and probe result
  primitives/                 the 212-predictor library metadata used in prompts
  derived/                    evaluation universe and results
  WALLS.md                    the frozen per-model evaluation walls (probe-derived)
scripts/verify_corpus.py      standalone integrity + reproduction check (stdlib only)
scripts/check_numbers.py      recomputes every number in the paper and diffs it against the PDF
```

## Registration discipline

The experiments were registered by construction, in this order: (1) generation corpora
archived and committed; (2) evaluation walls frozen from recall-probe data alone
(`exp2/WALLS.md`); (3) analysis specifications committed (`ANALYSIS_SPEC.md` in each
experiment); (4) only then was any performance data examined. Raw model responses are
never edited; every archived response carries its request parameters, timestamps, and
content hashes in the corpus manifests. This repository is a faithful snapshot of that
archive.

## Verify

No dependencies beyond the Python standard library:

```bash
python3 scripts/verify_corpus.py
```

This checks corpus integrity (file counts, proposal/hypothesis counts, manifest content
hashes) and independently recomputes Experiment 1's headline table
(`exp1/derived/results_a.csv`) from the raw corpus, the raw judge output, and the bundled
predictor metadata.

A second check verifies the *paper* rather than the data (requires `pdftotext`):

```bash
python3 scripts/check_numbers.py
```

It recomputes 105 published quantities from the committed derived artifacts — the evaluation
ladder, the grammar-null percentiles, both auxiliary accountings and their per-type breakdowns,
Experiment 1's tables, the test-(b) regression, the closed-form constants and every corpus
count — and asserts that each appears in the compiled PDF as computed. It then sweeps every
remaining numeral in the body and lists those no registered fact explains, so an unbacked number
introduced by a future revision surfaces on the next run. Exit status is non-zero if anything is
missing.

The check exists because it was needed: two rounds of cold reading found published figures that
were simply wrong — a count of negative cells reported as 21 when the artifact says 20, and a
regression t-statistic labelled as one model subset when it came from another. Both were
checkable arithmetic. One known limitation is documented in the script: presence is tested
against the whole document, so a value that also occurs elsewhere can pass spuriously; facts
whose numbers are not unique are anchored to nearby text instead.

Full re-scoring of Experiment 2 additionally requires the public factor data: the Open
Source Asset Pricing portfolio returns (openassetpricing.com), the Ken French data library,
and FRED series VIXCLS. The scoring code in `exp*/src/` is included verbatim as run; it
reads those sources through a local data layer (`_dp.py`) that any user can replace with
direct downloads from the providers above.

## Data credit

- Chen, A. Y., and T. Zimmermann (2022). "Open Source Cross-Sectional Asset Pricing."
  *Critical Finance Review* 11(2), 207–264. https://www.openassetpricing.com
- He, S., L. Lv, A. Manela, and J. Wu (2025). "Chronologically Consistent Large
  Language Models" (ChronoGPT vintage models, https://huggingface.co/manelalab) —
  used for the vintage-model control arm in `exp1/control/`.
- Kenneth French Data Library.
- FRED (Federal Reserve Economic Data), series VIXCLS; U.S. Treasury yields.

## License

Code: MIT (see `LICENSE`). Paper and text: CC BY 4.0.
