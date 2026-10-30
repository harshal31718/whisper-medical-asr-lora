# Results (ekacare/eka-medical-asr-evaluation-dataset, held-out split, text-deduplicated; speaker overlap not ruled out (session_id is ~1 per clip))

train utts: 2901 | test utts: 636 | base: openai/whisper-small
config: {'lora_r': 32, 'lr': 0.001, 'steps': 600, 'eff_batch': 16, 'n_train': 2901}

entity categories in test: {'clinical_findings': 397, 'drugs': 321, 'diagnostics': 90, 'advices': 44, 'misc_medical': 93}
sessions: 3524 | test utts with >=1 entity: 635

| metric | baseline | baseline_prompted | finetuned | finetuned_prompted |
|---|---|---|---|---|
| wer | 17.81% | 16.98% | 11.35% | 11.84% |
| cer | 7.16% | 6.82% | 4.93% | 5.41% |
| term_error_rate | 35.44% | 34.75% | 27.75% | 28.33% |
| term_err_advices | 38.00% | 40.00% | 40.00% | 42.00% |
| term_err_clinical_findings | 23.70% | 23.99% | 19.57% | 20.35% |
| term_err_diagnostics | 41.76% | 39.56% | 40.11% | 39.01% |
| term_err_drugs | 72.36% | 69.47% | 48.80% | 49.28% |
| term_err_misc_medical | 14.09% | 13.64% | 12.73% | 13.64% |
| n_terms | 1885 | 1885 | 1885 | 1885 |

## Claim guard (resume bullets may only state CLAIMABLE items)

- fine-tune vs baseline wer: 17.81% -> 11.35% (+36.3% relative reduction) [CLAIMABLE]
- fine-tune vs baseline cer: 7.16% -> 4.93% (+31.1% relative reduction) [CLAIMABLE]
- fine-tune vs baseline term_error_rate: 35.44% -> 27.75% (+21.7% relative reduction) [CLAIMABLE]
- fine-tune vs baseline term_err_advices: 38.00% -> 40.00% (-5.3% relative reduction) [DO NOT CLAIM (no improvement)]
- fine-tune vs baseline term_err_clinical_findings: 23.70% -> 19.57% (+17.4% relative reduction) [CLAIMABLE]
- fine-tune vs baseline term_err_diagnostics: 41.76% -> 40.11% (+4.0% relative reduction) [CLAIMABLE]
- fine-tune vs baseline term_err_drugs: 72.36% -> 48.80% (+32.6% relative reduction) [CLAIMABLE]
- fine-tune vs baseline term_err_misc_medical: 14.09% -> 12.73% (+9.7% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline wer: 17.81% -> 16.98% (+4.7% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline cer: 7.16% -> 6.82% (+4.7% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline term_error_rate: 35.44% -> 34.75% (+1.9% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline term_err_advices: 38.00% -> 40.00% (-5.3% relative reduction) [DO NOT CLAIM (no improvement)]
- prompt-bias (no training) vs baseline term_err_clinical_findings: 23.70% -> 23.99% (-1.2% relative reduction) [DO NOT CLAIM (no improvement)]
- prompt-bias (no training) vs baseline term_err_diagnostics: 41.76% -> 39.56% (+5.3% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline term_err_drugs: 72.36% -> 69.47% (+4.0% relative reduction) [CLAIMABLE]
- prompt-bias (no training) vs baseline term_err_misc_medical: 14.09% -> 13.64% (+3.2% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline wer: 17.81% -> 11.84% (+33.5% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline cer: 7.16% -> 5.41% (+24.4% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_error_rate: 35.44% -> 28.33% (+20.1% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_err_advices: 38.00% -> 42.00% (-10.5% relative reduction) [DO NOT CLAIM (no improvement)]
- fine-tune+prompt vs baseline term_err_clinical_findings: 23.70% -> 20.35% (+14.1% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_err_diagnostics: 41.76% -> 39.01% (+6.6% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_err_drugs: 72.36% -> 49.28% (+31.9% relative reduction) [CLAIMABLE]
- fine-tune+prompt vs baseline term_err_misc_medical: 14.09% -> 13.64% (+3.2% relative reduction) [CLAIMABLE]
- corti zero-shot transfer wer: 14.96% -> 10.21% [CLAIMABLE]
- corti zero-shot transfer cer: 10.70% -> 5.39% [CLAIMABLE]
- corti zero-shot transfer term_error_rate: 40.14% -> 42.43% [DO NOT CLAIM]
