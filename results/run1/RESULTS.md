# Results (ekacare/eka-medical-asr-evaluation-dataset, session-disjoint + text-dedupe)

train utts: 2901 | test utts: 636 | base: openai/whisper-small
config: {'lora_r': 32, 'lr': 0.001, 'steps': 600, 'eff_batch': 16, 'n_train': 2901}

entity categories in test: {'clinical_findings': 397, 'drugs': 321, 'diagnostics': 90, 'advices': 44, 'misc_medical': 93}
sessions: 3524 | test utts with >=1 entity: 635

| metric | baseline | baseline_prompted | finetuned | finetuned_prompted |
|---|---|---|---|---|
| wer | 17.81% | 16.98% | 11.04% | 11.47% |
| cer | 7.16% | 6.82% | 4.61% | 4.99% |
| term_error_rate | 35.44% | 34.75% | 28.33% | 28.86% |
| term_err_advices | 38.00% | 40.00% | 42.00% | 36.00% |
| term_err_clinical_findings | 23.70% | 23.99% | 20.35% | 20.75% |
| term_err_diagnostics | 41.76% | 39.56% | 37.91% | 41.21% |
| term_err_drugs | 72.36% | 69.47% | 50.96% | 51.44% |
| term_err_misc_medical | 14.09% | 13.64% | 11.36% | 11.82% |
| n_terms | 1885 | 1885 | 1885 | 1885 |

## Claim guard (resume bullets may only state CLAIMABLE items)

- fine-tune vs baseline wer: 17.81% -> 11.04% (+38.0% relative reduction) [CLAIMABLE]
- fine-tune vs baseline cer: 7.16% -> 4.61% (+35.6% relative reduction) [CLAIMABLE]
- fine-tune vs baseline term_error_rate: 35.44% -> 28.33% (+20.1% relative reduction) [CLAIMABLE]
- fine-tune vs baseline term_err_advices: 38.00% -> 42.00% (-10.5% relative reduction) [DO NOT CLAIM (no improvement)]
- fine-tune vs baseline term_err_clinical_findings: 23.70% -> 20.35% (+14.1% relative reduction) [CLAIMABLE]
- fine-tune vs baseline term_err_diagnostics: 41.76% -> 37.91% (+9.2% relative reduction) [CLAIMABLE]
- fine-tune vs baseline term_err_drugs: 72.36% -> 50.96% (+29.6% relative reduction) [CLAIMABLE]
- fine-tune vs baseline term_err_misc_medical: 14.09% -> 11.36% (+19.4% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline wer: 17.81% -> 16.98% (+4.7% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline cer: 7.16% -> 6.82% (+4.7% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline term_error_rate: 35.44% -> 34.75% (+1.9% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline term_err_advices: 38.00% -> 40.00% (-5.3% relative reduction) [DO NOT CLAIM (no improvement)]
- prompt-bias (no training) vs baseline term_err_clinical_findings: 23.70% -> 23.99% (-1.2% relative reduction) [DO NOT CLAIM (no improvement)]
- prompt-bias (no training) vs baseline term_err_diagnostics: 41.76% -> 39.56% (+5.3% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline term_err_drugs: 72.36% -> 69.47% (+4.0% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline term_err_misc_medical: 14.09% -> 13.64% (+3.2% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline wer: 17.81% -> 11.47% (+35.6% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline cer: 7.16% -> 4.99% (+30.3% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_error_rate: 35.44% -> 28.86% (+18.6% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_err_advices: 38.00% -> 36.00% (+5.3% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_err_clinical_findings: 23.70% -> 20.75% (+12.4% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_err_diagnostics: 41.76% -> 41.21% (+1.3% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_err_drugs: 72.36% -> 51.44% (+28.9% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_err_misc_medical: 14.09% -> 11.82% (+16.1% relative reduction) [CLAIMABLE]
