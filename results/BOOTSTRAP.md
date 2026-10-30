# Bootstrap confidence intervals — run2 (canonical run)

Source: instrumented `errors_<system>.csv` (original test order, per-utterance edit counts and term hits) and `results.json` in this folder. Script: `scripts/bootstrap_ci.py run2`.

## Method
- Paired bootstrap over the 636 test utterances: each of 10,000 resamples draws utterances with replacement (seed 42) and applies the **same** draw to both systems. 95% percentile intervals.
- Corpus rates: WER = word edits / reference words; CER = character edits / reference characters; term error = missed matched entity terms / matched entity terms (exact whole-phrase match after normalization, as in `train.py`).
- Point estimates are asserted equal to `results.json` (exact for WER, CER and overall term error; drug term error equals the 4-decimal value stored there; term counts also asserted).
- Difference = first system minus second system, in percentage points (positive = second system has lower error).
- Terms per metric (baseline CSV): 1885 matched terms overall, 416 drug terms.

## Baseline vs fine-tuned (LoRA)

| metric | baseline [95% CI] | fine-tuned [95% CI] | difference baseline - fine-tuned [95% CI] |
|---|---|---|---|
| WER | 17.81% [16.20%, 19.58%] | 11.35% [9.87%, 13.02%] | +6.46 pp [+5.36 pp, +7.62 pp] |
| CER | 7.16% [5.95%, 8.63%] | 4.93% [3.72%, 6.40%] | +2.23 pp [+1.24 pp, +3.00 pp] |
| Term error (all matched terms) | 35.44% [32.76%, 38.28%] | 27.75% [25.19%, 30.48%] | +7.69 pp [+5.97 pp, +9.43 pp] |
| Term error (drugs) | 72.36% [68.26%, 76.53%] | 48.80% [43.90%, 53.72%] | +23.56 pp [+18.78 pp, +28.43 pp] |

## Additional: baseline vs baseline + vocabulary prompt (no training)

| metric | baseline [95% CI] | baseline+prompt [95% CI] | difference baseline - baseline+prompt [95% CI] |
|---|---|---|---|
| WER | 17.81% [16.20%, 19.58%] | 16.98% [15.39%, 18.74%] | +0.84 pp [+0.32 pp, +1.39 pp] |
| CER | 7.16% [5.95%, 8.63%] | 6.82% [5.61%, 8.28%] | +0.34 pp [+0.13 pp, +0.56 pp] |
| Term error (all matched terms) | 35.44% [32.76%, 38.28%] | 34.75% [32.11%, 37.56%] | +0.69 pp [-0.22 pp, +1.62 pp] |
| Term error (drugs) | 72.36% [68.26%, 76.53%] | 69.47% [65.33%, 73.77%] | +2.88 pp [+0.45 pp, +5.46 pp] |

## Limits
- Intervals reflect resampling of the test utterances only. They do not capture training randomness (one seed per run; compare with run1 for a replicate) or the speaker-overlap issue (about 1 clip per `session_id`).
- Term-error intervals treat each utterance as a unit; terms within an utterance are not independent.
- Per-utterance count self-check recorded by `train.py` (`counts_check`): {'baseline': True, 'baseline_prompted': True, 'finetuned': True, 'finetuned_prompted': True}
