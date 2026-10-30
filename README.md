# Whisper-small + LoRA on clinician speech: does it reduce medical-term errors?

Fine-tuning `openai/whisper-small` with LoRA on the Eka Care medical ASR dataset (English subset), evaluated against the
zero-shot model with WER, CER and a medical-term error rate built from the dataset's entity labels, with bootstrap
confidence intervals. Everything runs as a script kernel on Kaggle (Tesla T4).

**Research question:** does LoRA fine-tuning reduce WER and *medical-term* errors, and does training-free vocabulary
prompting give a cheaper gain?

> **Read the Limitations section.** Each run is a single training seed, the split behaves close to an utterance-level split,
> and the corti experiment is exploratory and not a claim.

## Results (run2, the canonical run)

636 held-out utterances, 1,885 matched medical terms. Differences are baseline minus fine-tuned (positive = fine-tuning lowers
error), with paired-bootstrap 95% confidence intervals (10,000 resamples over utterances, seed 42). Source:
[`results/results.json`](results/results.json), [`results/BOOTSTRAP.md`](results/BOOTSTRAP.md).

| Metric | Baseline (zero-shot) | Fine-tuned (LoRA) | Difference (95% CI) |
|---|---|---|---|
| WER | 17.81% | 11.35% | +6.46 pp (+5.36 to +7.62) |
| CER | 7.16% | 4.93% | +2.23 pp (+1.24 to +3.00) |
| Medical-term error (1,885 terms) | 35.44% | 27.75% | +7.69 pp (+5.97 to +9.43) |
| Drug-term error (416 terms) | 72.36% | 48.80% | +23.56 pp (+18.78 to +28.43) |

Term error by category (run2; no confidence intervals computed per category except drugs above):

| Category (terms) | Baseline | Fine-tuned |
|---|---|---|
| drugs (416) | 72.36% | 48.80% |
| clinical_findings (1,017) | 23.70% | 19.57% |
| diagnostics (182) | 41.76% | 40.11% |
| misc_medical (220) | 14.09% | 12.73% |
| advices (50) | 38.00% | 40.00% |

`advices` got **worse** after fine-tuning (and in the replicate); with only 50 terms it is also noisy. It is not claimed as an improvement.

### Vocabulary prompting (no training)

A prompt built from the 25 most frequent drug-category terms in the training set. Baseline vs baseline + prompt (paired bootstrap):

| Metric | Baseline | Baseline + prompt | Difference (95% CI) |
|---|---|---|---|
| WER | 17.81% | 16.98% | +0.84 pp (+0.32 to +1.39) |
| CER | 7.16% | 6.82% | +0.34 pp (+0.13 to +0.56) |
| Medical-term error | 35.44% | 34.75% | +0.69 pp (-0.22 to +1.62) |
| Drug-term error | 72.36% | 69.47% | +2.88 pp (+0.45 to +5.46) |

Prompting gave a small WER gain and **no measurable overall medical-term gain** (the interval includes 0). With the fine-tuned
model, adding the prompt did not help: WER 11.84% with the prompt vs 11.35% without (no interval computed). The prompt vocabulary is noisy
(it includes drug classes and company names), so this is a weak test of prompting, not a verdict on it.

## Replicate (run1)

A second run with the same config and seed 42 (the baseline is deterministic, so baseline numbers are identical). Run1's per-utterance
outputs had no index column, so its intervals use a cluster bootstrap that pools 8 pairs of identical-text clips
(see [`results/run1/BOOTSTRAP.md`](results/run1/BOOTSTRAP.md)); run1 has no term-error intervals.

| Fine-tuned model | run1 (replicate) | run2 (canonical) |
|---|---|---|
| WER | 11.04% | 11.35% |
| CER | 4.61% | 4.93% |
| Medical-term error | 28.33% | 27.75% |
| Drug-term error | 50.96% | 48.80% |
| WER difference vs baseline (95% CI) | +6.77 pp (+5.80 to +7.81) | +6.46 pp (+5.36 to +7.62) |

The two runs differ by a few tenths of a point on WER and CER and by up to about 2 points on drug-term error. I have not investigated the
cause. Run2 was fixed as the canonical run before it was launched, regardless of how it compared with run1.

Note: the raw `results/run1/results.json` and `RESULTS.md` carry a split label written by the earlier script revision that
overstated the split (it implied speaker-level separation, which the data does not support; see Limitations). I left the raw files unedited.

## Dataset, split, config

- **Data:** [`ekacare/eka-medical-asr-evaluation-dataset`](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset),
  `en` subset, 3,619 clips (none removed by a 0.3 to 30 s duration filter). Transcripts carry medical-entity labels
  (drugs, clinical findings, diagnostics, advices, misc). The dataset card lists the **MIT** license.
- **Split:** 704 of 3,524 `session_id`s held out (20%), then test texts that also appear in train were dropped:
  **2,901 train / 636 test** utterances. `session_id` is about one per clip, so this is effectively an utterance-level split and
  **speaker overlap is not ruled out**.
- **Model and training:** `openai/whisper-small`, LoRA r=32, alpha=64, dropout 0.05 on `q_proj`/`v_proj`
  (3,538,944 trainable parameters, 1.44% of the model), learning rate 1e-3, 50 warmup steps, 600 steps, effective batch 16
  (8 x 2 accumulation), fp16, seed 42. Training took 1,208 s on a Tesla T4; training loss fell from 1.524 to 0.1454.
- **Decoding:** greedy, `language="en"`, `max_new_tokens=160`.
- **Metrics:** text is lowercased, hyphens/slashes become spaces and other punctuation is removed before scoring. WER/CER are
  corpus-level (summed edits over summed reference length). Medical-term error is the share of labeled terms (only those
  literally present in the reference) that do not appear as a whole phrase in the normalized hypothesis.
- **Environment (both runs):** torch 2.11.0+cu128, transformers 5.22.0, peft 0.21.2, datasets 5.1.0. Libraries are installed unpinned at kernel start.

## Failure analysis (run2)

Full write-up with example rows: [`results/ERROR_ANALYSIS.md`](results/ERROR_ANALYSIS.md). It is based on transcripts only; **no audio was reviewed**.

- 218 of 636 utterances are exact matches after normalization (baseline: 124); 55 are fully wrong, WER >= 1 (baseline: 118).
- 51 of those 55 have 2 reference words or fewer; such short clips are 127 of the 636 test utterances.
- Of the 30 worst transcripts, 19 are rare drug names or clinical terms (13 drug, 6 clinical); the rest are abbreviations, spelling
  variants such as "thirstless" vs "thirst less", hallucination, a code-switched Hindi sentence, a suspected label error and a company name.
- Only 3 of 256 long clips (>= 20 words) failed (WER >= 0.5), all by fluent hallucination or dropped output.

## Exploratory: zero-shot transfer to `corti/med-term` (not a claim)

Run2 also scored the baseline and the fine-tuned model on the English subset of `corti/med-term` (200 clips, 568 matched terms). The fine-tuned model was
not trained on it. The dataset is **fully synthetic TTS dictation**, not clinician speech.

| Metric | Baseline | Fine-tuned |
|---|---|---|
| WER | 14.96% | 10.21% |
| CER | 10.70% | 5.39% |
| Medical-term error | 40.14% | 42.43% |

The result is mixed: WER and CER fell while term error **rose**, and I have not investigated why. There are no confidence intervals, because per-utterance
corti outputs were not saved. Possible confound, partly confirmed: the fine-tuned model was merged and cast to fp16, while the baseline was loaded without an
explicit cast, so its actual dtype at inference is unconfirmed; the fine-tuned corti path also differs from the main evaluation path. Treat this as a
smoke test only.

Corti data is used for evaluation only and is not redistributed. Dataset: Corti ASR Evaluation Dataset Licence (CDLA-Permissive-2.0 plus a use-restrictions
addendum that forbids training on the data). Citation, as given on the dataset card: Nix et al., "Symphony for Speech-to-Text: Supporting Real-Time Medical
Voice Interfaces", arXiv:2605.16545, 2026. https://doi.org/10.48550/arXiv.2605.16545

## Limitations

- **Single seed per run.** Each run trains once with seed 42; I have one replicate. The bootstrap intervals resample test utterances only and do not capture training randomness.
- **Utterance-level split behavior.** About one clip per `session_id` (3,524 sessions for 3,619 clips), so held-out sessions do not guarantee held-out speakers. Only exact-text duplicates between test and train were removed; near-duplicate phrasing may remain.
- **Noisy prompt vocabulary.** The top-25 "drug" terms include drug classes and company names, so the prompting experiment is a weak test.
- **No audio reviewed.** The error categories come from transcripts; I make no claim about noise, accent or speaker effects. The accent of the English subset was not verified.
- **Exact-match term metric.** Spelling or spacing variants count as errors, and there is no clinical-severity weighting, so term error can overstate clinical harm.
- **Whisper-small only.** No larger Whisper model, wav2vec2 or other baseline was compared.
- **`advices` category.** Only 50 terms; it got worse after fine-tuning and is too small to interpret.
- **Possible memorization.** Training loss steps down sharply at epoch boundaries; the held-out numbers above are what count.
- **Corti experiment** is exploratory and unconfirmed (synthetic data, no intervals, possible dtype confound, term error worse).

## Reproduce

Requires a phone-verified Kaggle account (GPU and Internet are needed) and Windows PowerShell. Credentials are never stored in this repo.

```powershell
pip install kaggle jiwer pandas numpy
kaggle auth login                                   # or place credentials per the Kaggle CLI docs
$env:KAGGLE_USERNAME = "<your-kaggle-username>"
.\scripts\kaggle_run.ps1 -Stage probe               # data columns, split sizes, entity categories
.\scripts\kaggle_run.ps1 -Stage baseline            # zero-shot numbers
.\scripts\kaggle_run.ps1 -Stage all -Steps 600 -Corti   # LoRA fine-tune + eval (+ exploratory corti), downloads to out\all
Rename-Item out\all out\run2                        # the analysis scripts read out\run1 / out\run2
python scripts\make_report.py out\run2\results.json # results table + claim guard
python scripts\bootstrap_ci.py run2                 # paired bootstrap CIs -> out\run2\BOOTSTRAP.md
python scripts\compare_runs.py out\run1\results.json out\run2\results.json   # needs a second run saved as out\run1
python scripts\check_readme.py                      # checks README numbers against results/ (or out/ if present)
```

Drop `-Corti` to skip the corti evaluation. Results will vary a little from run to run (see the replicate above). `out/` is git-ignored;
`results/` holds the files these numbers come from.

## Repository layout

```
kaggle/train.py        Kaggle script kernel: probe | baseline | all (config block at the top)
scripts/               kaggle_run.ps1 (push/poll/download), make_report.py, bootstrap_ci.py, compare_runs.py, check_readme.py
results/               run2 (canonical): RESULTS.md, results.json, BOOTSTRAP.md, ERROR_ANALYSIS.md; run1/ replicate
docs/                  ERROR_ANALYSIS.md (same file as results/ERROR_ANALYSIS.md)
```

## License and data

The Eka Care dataset is MIT-licensed per its dataset card; no dataset audio or transcripts are included in this repository. The corti dataset terms are
described above. This repository does not yet include a license file for the code.
