# Error analysis — fine-tuned Whisper-small (LoRA) on Eka Care `en` (run2, canonical)

Source: `out/run2/errors_finetuned_sorted.csv` (636 test utterances, sorted by per-utterance WER, worst first; `idx` = row in the
original test order). Run: LoRA r=32, lr 1e-3, 600 steps, seed 42, single run. Aggregate numbers come from `out/run2/results.json`.

## How this was done (and its limits)
- I read the 30 worst rows and assigned each to one category **from the text only**. I did NOT listen to the audio, so causes
  like "noise", "accent" or "mumbled" are unverified and not claimed.
- The top-30 list is **length-biased**: ranking by per-utterance WER puts 1-word utterances first (one wrong word = WER 1.0 or more).
  Median reference length is 6 words overall but **1 word** in the worst 30. The aggregate counts below cover all 636 rows.
- Categories are my judgement; one annotator, no second reader. Small-sample counts, not a measured taxonomy.

## Aggregate facts (all 636 test utterances, run2)
| Fact | Value |
|---|---|
| Utterances with WER = 0 (exact match after normalization) | 218 (baseline: 124) |
| Utterances with WER >= 1 (fully wrong) | 55 (baseline: 118) |
| ...of those 55, utterances with <= 2 reference words | 51 (37 are single words) |
| Test utterances with <= 2 reference words | 127 of 636 (about 20%) |
| Long utterances (>= 20 words) | 256, of which only 3 have WER >= 0.5 |

After fine-tuning, almost all complete failures are isolated 1–2 word terms (mostly drug and clinical-term names). Long,
sentence-like utterances are mostly fine, with 3 catastrophic exceptions (category 5).

## Categorization of the 30 worst utterances (run2)
| # | Category | Count | `idx` values |
|---|---|---|---|
| 1 | Drug names (generic and brand), phonetic near-misses | 13 | 360, 16, 38, 50, 51, 99, 105, 110, 156, 161, 175, 180, 194 |
| 2 | Non-drug clinical / diagnostic terms and phrases | 6 | 27, 39, 78, 102, 103, 165 |
| 3 | Abbreviations | 3 | 367, 40, 124 |
| 4 | Spacing / spelling variant of the same word | 3 | 297, 101, 159 |
| 5 | Hallucination / loop / dropped output | 2 | 131, 411 |
| 6 | Hindi–English code-switching | 1 | 361 |
| 7 | Suspected reference (label) error | 1 | 76 |
| 8 | Company name | 1 | 55 |

### 1. Drug names (13 of 30)
Rare drug names are the largest single group; drugs are also the weakest term category overall
(term error 72.36% baseline -> 48.80% fine-tuned in run2, still the highest of any category).
- REF `arteether` -> HYP `rt etha`
- REF `fradiomycin` -> HYP `radiomycin`
- REF `levodropropizine` -> HYP `legotropropeazine`
- REF `clarithromycin` -> HYP `chlarythromesan`
- REF `aminophylline` -> HYP `amenofiline`
- Near-misses a letter or two off, counted as a full word error: `ramosetron` -> `ramostrone`, `lamotrigine` -> `lamotrogen`,
  `acetaminophen` -> `acetaminophil`, `Oxetacaine` -> `oxetacale`, `Waxrid` -> `vaxrid`, `Chericof` -> `jericoff`.

### 2. Non-drug clinical terms and phrases (6 of 30)
- REF `tetanus toxoid` -> HYP `titanist oxide`
- REF `feeling anxious` -> HYP `phylin anoxias`
- REF `dog bite` -> HYP `dopide`
- Spelling-level near-misses: `lymphadenopathy` -> `limphadenopathy`, `orthopnea` -> `arthopnea`.
- REF `chronic pruritis` -> HYP `protic pruritus`: the second word is the standard spelling and the reference spells it `pruritis`.

### 3. Abbreviations (3 of 30)
- REF `dmt2` -> HYP `dmt 2`; REF `high tg` -> HYP `ytg`; REF `gej tumor` -> HYP `geast humer`.

### 4. Spacing / spelling variants (3 of 30)
Arguably not recognition failures; the model produced the same sounds with different orthography.
- REF `thirstless` -> HYP `thirst less`
- REF `venous stasis` -> HYP `venostasis`
- REF `Huminsulin R 100Iu Cartridge` -> HYP `Human Insulin R 100 IU Cartridge`
Exact-match metrics (WER, term error) count these as errors. This is a metric limitation, not necessarily a model limitation.

### 5. Hallucination / loop / dropped output (2 of 30 here; 3 long-clip failures in total)
- idx 131: REF `impulsiveness` (one word) -> HYP `epulchrackle dimples dimples epulchrackle dimples epulchrackle ...` (a repetition
  loop, WER 23.0). This one single clip contributes an extreme per-utterance WER.
- idx 411: REF (33 words) `A decision was made to first treat the hepatitis infection and initiate liver protection therapy with a combination of
  Entecavir ...` -> HYP `Galaxy S5-L Tablet is a combination of two medicines, both of which help to control the blood pressure ...` (fluent,
  unrelated content).
- The other two long-clip failures (WER >= 0.5 on clips of >= 20 words) are not in the top 30 rows listed above but are in the aggregate:
  idx 621 (REF `Insulin Lispro is a fast acting insulin analogue ...`, 40 words -> HYP `Recording`) and idx 474 (REF `Fexozit 120mg Tablet is
  an anti-allergy medicine ...` -> HYP `Galaxy S5 Tablet is a combination of two medicines, both of which help to control the blood pressure ...`).
Cause (audio quality, silence, decoding) is not investigated; I did not listen to these clips.

### 6. Hindi–English code-switching (1 of 30)
- idx 361: REF `jab ana hoga apko call kar dungi` -> HYP `When you will have to come, I will call you.`
The reference is romanized Hindi; the model returned an English paraphrase instead of a transcription. Decoding is forced to
`language="en"`, so this looks like translation behaviour; I have not tested that explanation.

### 7. Suspected reference (label) error (1 of 30)
- idx 76: REF `abendazole` -> HYP `albendazole`. `albendazole` is the standard drug name, so the reference looks misspelled and the
  hypothesis appears to be the correct word. This is my judgement from the text, not verified against the audio. One row only.

### 8. Company name (1 of 30)
- idx 55: REF `Zee Laboratories` -> HYP `z laboratory's`.

## Top 3 failure patterns (run2)
1. **Rare drug and clinical-term names**: 19 of the worst 30 (13 drug, 6 clinical); drug-term error is still 48.80% after fine-tuning.
2. **Isolated 1–2 word utterances** dominate full failures: 51 of the 55 utterances with WER >= 1 have <= 2 reference words,
   and such clips are about 20% of the test set.
3. **Rare catastrophic outputs on long clips** (fluent hallucination, dropped output): 3 of 256 long clips have WER >= 0.5.

## Relationship to run1 (replicate)
Run1 (`out/run1/errors_finetuned.csv`) was analysed earlier with the same method; I compared the two runs on these measured points only:
- 17 of the 30 worst references in run2 also appear in run1's worst 30.
- Fully-wrong utterances (WER >= 1): 53 in run1, 55 in run2; of those, 48 (run1) and 51 (run2) have <= 2 reference words.
- Long clips with WER >= 0.5: 3 of 256 in both runs. The `Insulin Lispro` clip became `Recording` in both runs, and the `Entecavir` clip
  produced an unrelated fluent "tablet ... blood pressure" text in both (`Turkish Tablet ...` in run1, `Galaxy S5-L Tablet ...` in run2).
- Category counts for the worst 30 are not identical (run1 had 13 drug and 9 clinical rows; run2 has 13 and 6). I did not otherwise compare the runs.

## What this analysis does NOT show
- No audio was reviewed, so no claim about noise, accent, mumbling or speaker effects.
- No significance testing on the categories; the top-30 slice is not representative of the average utterance.
- Orthographic variants (category 4) and near-misses mean WER and exact-match term error overstate clinical error; no
  clinical-severity weighting was done.
