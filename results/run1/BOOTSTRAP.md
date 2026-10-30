# Bootstrap confidence intervals — run1

Source: `errors_baseline.csv`, `errors_finetuned.csv`, `results.json` in this folder. Script: `scripts/bootstrap_ci.py run1`.

## Method
- Corpus WER/CER = sum(S+D+I) / sum(reference length) after the same normalization as `train.py`; point estimates are asserted equal to `results.json` (exact float equality).
- 10,000 bootstrap resamples, seed 42, 95% percentile intervals. The difference is baseline minus fine-tuned (positive = fine-tuning lower error), computed on the same resamples for both systems.

## Deviation from a per-utterance paired bootstrap (read this)
run1's CSVs were each sorted by their own WER and have no index column, so utterances cannot be matched by row position. They were matched by identical reference text. 8 reference texts occur twice (16 rows, different clips with the same text), and which baseline clip pairs with which fine-tuned clip is unknown for those rows. Each such group is therefore pooled into one resampling unit.
The bootstrap resamples **628 units** drawn from 636 utterances (8 units contain 2 clips). Point estimates are unaffected (they equal `results.json`); the intervals differ slightly from an utterance-level bootstrap.

## Results (run1)

| metric | baseline [95% CI] | fine-tuned [95% CI] | difference baseline - fine-tuned [95% CI] |
|---|---|---|---|
| WER | 17.81% [16.17%, 19.70%] | 11.04% [9.69%, 12.58%] | +6.77 pp [+5.80 pp, +7.81 pp] |
| CER | 7.16% [5.93%, 8.70%] | 4.61% [3.63%, 5.85%] | +2.55 pp [+2.04 pp, +3.12 pp] |

## Not computed
- **Medical-term error and drug-term error: no CI for run1.** The run1 CSVs contain no entity labels and `probe.json` holds entities for 3 test rows only. See run2 (`out/run2/BOOTSTRAP.md`) for term-error intervals.
- Intervals reflect resampling of the test utterances only. They do not capture training randomness (single seed per run) or the speaker-overlap issue (about 1 clip per `session_id`).
