# %% [markdown]
# # Whisper-small LoRA fine-tune on Eka Care medical ASR data (held-out split, text-deduplicated)
# Run on Kaggle (GPU T4, Internet ON). Each "# %%" block is one notebook cell.
# Outputs: results.json + errors_*.csv  (these are the ONLY source of numbers for your resume).

"""
Whisper-small LoRA fine-tune on Eka Care medical ASR (held-out split by session_id, ~1 clip per session; text-deduplicated).
Kaggle SCRIPT kernel. Edit the USER CONFIG block, push with scripts/kaggle_run.*, read out/results.json.

STAGE:
  "probe"    -> load + inspect data, write probe.json, exit  (cheap sanity gate)
  "baseline" -> probe + zero-shot baseline eval, exit
  "all"      -> baseline + LoRA fine-tune + eval (+ optional corti cross-eval)
Numbers on the resume must come ONLY from results.json produced by this script.
"""
# ---------------- USER CONFIG (edit before each push) ----------------
STAGE = "all"          # "probe" | "baseline" | "all"
MAX_STEPS = 600          # ~1 hr on a T4; 400 if short on time
TEST_FRAC = 0.2          # fraction of SESSIONS held out
RUN_CORTI = True        # optional zero-shot cross-dataset check (needs license/gating check)
LR, LORA_R = 1e-3, 32
# ---------------------------------------------------------------------
import os, sys, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # force ONE gpu (T4x2 would silently change effective batch size)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "peft", "datasets",
                "accelerate", "jiwer", "librosa", "soundfile"], check=True)
# Kaggle image ships torchao 0.10; new peft raises ImportError on it (<0.16). We don't use torchao -> remove it.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
try:                                                # optional HF token (only if the dataset is gated)
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    login(UserSecretsClient().get_secret("HF_TOKEN"))
    print("HF login ok")
except Exception as e:
    print("HF token not used:", str(e)[:80])

import os, re, json, random, time, ast
import numpy as np, pandas as pd, torch, jiwer
from collections import Counter
from dataclasses import dataclass
from datasets import load_dataset, Audio
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from peft import LoraConfig, get_peft_model

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out"; os.makedirs(OUT, exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BASE = "openai/whisper-small"
import transformers, peft, datasets as _ds
json.dump(dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__, datasets=_ds.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", seed=SEED, stage=STAGE),
          open(f"{OUT}/env.json", "w"), indent=1)

# ---- CONFIG ----------------------------------------------------------------
DATASET_ID, SUBSET = "ekacare/eka-medical-asr-evaluation-dataset", "en"   # 3.62k rows, test split only
# -----------------------------------------------------------------------------

# %% [markdown]
# ## 1. Metrics

# %%
def norm(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[-/]", " ", t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def wer_cer(refs, hyps):
    r = [norm(x) for x in refs]; h = [norm(x) for x in hyps]
    k = [i for i, x in enumerate(r) if x]
    r = [r[i] for i in k]; h = [h[i] for i in k]
    return jiwer.wer(r, h), jiwer.cer(r, h)

def term_error_rate(hyps, term_lists):
    """Fraction of reference medical terms NOT present (whole-phrase) in the hypothesis."""
    total = miss = 0
    for hyp, terms in zip(hyps, term_lists):
        hn = " " + norm(hyp) + " "
        for t in terms:
            total += 1
            miss += (" " + norm(t) + " ") not in hn
    return miss / max(total, 1), total

def extract_entities(cell, ref_text):
    """Robustly pull entity strings out of whatever shape `medical_entities` has
    (JSON string / dict of lists / list). Keep only strings actually present in the reference."""
    if isinstance(cell, str):
        for loader in (json.loads, ast.literal_eval):
            try: cell = loader(cell); break
            except Exception: pass
    out = {}
    def walk(x, cat="all"):
        # Eka format: [text, category, [[start, end]]] triples
        if isinstance(x, (list, tuple)) and len(x) == 3 and isinstance(x[0], str) \
                and isinstance(x[1], str) and isinstance(x[2], (list, tuple)):
            if len(x[0].strip()) >= 3: out.setdefault(x[1], []).append(x[0].strip())
            return
        if isinstance(x, dict):
            for k, v in x.items(): walk(v, str(k) if cat == "all" else cat)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v, cat)
        elif isinstance(x, str) and len(x.strip()) >= 3:
            out.setdefault(cat, []).append(x.strip())
    walk(cell)
    rn = " " + norm(ref_text) + " "
    return {c: sorted({t for t in ts if (" " + norm(t) + " ") in rn}) for c, ts in out.items()}

# %% [markdown]
# ## 2. Load Eka medical ASR data, split by SESSION (no leakage)

# %%
raw = load_dataset(DATASET_ID, SUBSET)
full = raw[list(raw.keys())[0]]
print(full)                         # CHECK: columns should include audio, text, session_id, medical_entities
print(full[0]["medical_entities"], "|", full[0]["text"])

# input_columns=... avoids decoding audio during filtering (much faster)
full = full.filter(lambda d: d is not None and 0.3 < float(d) <= 30, input_columns=["duration"])
sids = [str(s) for s in full["session_id"]]
sess = sorted(set(sids)); random.Random(SEED).shuffle(sess)
print("distinct sessions:", len(sess))
assert len(sess) >= 5, "Too few sessions for a session split - paste me the first-cell output and I'll adapt the split."
test_sess = set(sess[: max(2, int(len(sess) * TEST_FRAC))])
is_test = np.array([s in test_sess for s in sids])
tr_idx, te_idx = np.where(~is_test)[0], np.where(is_test)[0]

# drop test utterances whose normalized text also appears in train (duplicate-phrase leakage)
all_texts = [norm(t) for t in full["text"]]          # column access: no audio decode
train_texts = {all_texts[i] for i in tr_idx}
te_idx = np.array([i for i in te_idx if all_texts[i] not in train_texts])
print(f"sessions: {len(sess)} (held out {len(test_sess)}) | train {len(tr_idx)} | test {len(te_idx)} (after dedupe)")

train = full.select(tr_idx.tolist()).cast_column("audio", Audio(sampling_rate=16000))
test = full.select(te_idx.tolist()).cast_column("audio", Audio(sampling_rate=16000))
TEXT = "text"
test_ents = [extract_entities(c, t) for c, t in zip(test["medical_entities"], test[TEXT])]
print("test utterances with >=1 matched entity:", sum(1 for e in test_ents if any(e.values())))
print("entity categories:", Counter(c for e in test_ents for c in e))

# Prompt-biasing vocabulary mined from TRAIN entities only (no test leakage)
cnt = Counter()
for c, t in zip(train["medical_entities"], train[TEXT]):
    for cat, ts in extract_entities(c, t).items():
        if "drug" in cat.lower(): cnt.update(ts)
MED_PROMPT = "Medical dictation: " + ", ".join(w for w, _ in cnt.most_common(25)) + "."
print(MED_PROMPT)
_dur = np.array(full["duration"], dtype=float)
json.dump(dict(columns=full.column_names, n_rows_after_filter=len(full), n_sessions=len(sess), n_train=len(train), n_test=len(test),
               entity_categories=dict(Counter(c for e in test_ents for c in e)),
               test_with_entities=sum(1 for e in test_ents if any(e.values())),
               duration_s=dict(mean=float(_dur.mean()), p50=float(np.median(_dur)), max=float(_dur.max())),
               sample_texts=list(test[TEXT][:5]), sample_entities=[str(x) for x in test["medical_entities"][:3]],
               prompt=MED_PROMPT, test_ents=test_ents),   # test_ents: matched entities for ALL test rows, in test order
          open(f"{OUT}/probe.json", "w"), indent=1)
if STAGE == "probe":
    print("PROBE DONE"); sys.exit(0)

# %% [markdown]
# ## 3. Evaluation helpers

# %%
processor = WhisperProcessor.from_pretrained(BASE, language="english", task="transcribe")
tok, fe = processor.tokenizer, processor.feature_extractor

@torch.no_grad()
def transcribe(model, audios, bs=16, prompt=None):
    model.eval(); outs = []
    kw = dict(language="en", task="transcribe", max_new_tokens=160, num_beams=1)
    if prompt: kw["prompt_ids"] = torch.tensor(tok.get_prompt_ids(prompt)).to(DEV)
    for i in range(0, len(audios), bs):
        feats = fe(audios[i:i + bs], sampling_rate=16000, return_tensors="pt").input_features.to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            ids = model.generate(input_features=feats, **kw)
        txt = tok.batch_decode(ids, skip_special_tokens=True)
        if prompt: txt = [t.replace(prompt, "").strip() if t.strip().startswith(prompt[:20]) else t for t in txt]
        outs += [t.strip() for t in txt]
    return outs

def edit_counts(ref, hyp):
    """Per-utterance word/char edit counts after norm(). Empty normalized refs count 0 (wer_cer() skips them too)."""
    rn, hn = norm(ref), norm(hyp)
    if not rn: return dict(n_ref_words=0, n_sub=0, n_del=0, n_ins=0, n_ref_chars=0, char_errors=0)
    w, c = jiwer.process_words(rn, hn), jiwer.process_characters(rn, hn)
    return dict(n_ref_words=w.hits + w.substitutions + w.deletions, n_sub=w.substitutions, n_del=w.deletions,
                n_ins=w.insertions, n_ref_chars=c.hits + c.substitutions + c.deletions,
                char_errors=c.substitutions + c.deletions + c.insertions)

def term_hits(hyp, ents):
    """[[category, term, found], ...] for every matched entity; same normalization/matching as term_error_rate()."""
    hn = " " + norm(hyp) + " "
    return [[cat, t, (" " + norm(t) + " ") in hn] for cat, ts in ents.items() for t in ts]

def evaluate(model, name, prompt=None):
    t0 = time.time()
    hyps = transcribe(model, [x["array"] for x in test["audio"]], prompt=prompt)
    w, c = wer_cer(test[TEXT], hyps)
    res = {"wer": w, "cer": c, "n_utts": len(hyps)}
    allt = [sum(e.values(), []) for e in test_ents]
    res["term_error_rate"], res["n_terms"] = term_error_rate(hyps, allt)
    for cat in sorted({c for e in test_ents for c in e}):
        r, n = term_error_rate(hyps, [e.get(cat, []) for e in test_ents])
        if n >= 30: res[f"term_err_{cat}"] = round(r, 4); res[f"n_{cat}"] = n
    res["eval_sec"] = round(time.time() - t0)
    # per-utterance table in ORIGINAL test order (idx = row in `test`); the sorted copy is for error analysis
    df = pd.DataFrame([dict(idx=i, ref=ref_i, hyp=hyp_i, wer=jiwer.wer(norm(ref_i) or "x", norm(hyp_i)),
                            **edit_counts(ref_i, hyp_i), terms_json=json.dumps(term_hits(hyp_i, test_ents[i])))
                       for i, (ref_i, hyp_i) in enumerate(zip(test[TEXT], hyps))])
    # the per-utterance counts must reproduce the corpus numbers above (recorded, not fatal)
    wer_chk = (df.n_sub.sum() + df.n_del.sum() + df.n_ins.sum()) / df.n_ref_words.sum()
    cer_chk = df.char_errors.sum() / df.n_ref_chars.sum()
    hits = [h for tj in df.terms_json for h in json.loads(tj)]
    res["counts_check"] = bool(abs(wer_chk - w) < 1e-12 and abs(cer_chk - c) < 1e-12 and len(hits) == res["n_terms"]
                               and abs(sum(1 for h in hits if not h[2]) / max(len(hits), 1) - res["term_error_rate"]) < 1e-12)
    print(name, json.dumps(res, indent=1))
    df.to_csv(f"{OUT}/errors_{name}.csv", index=False)
    df.sort_values("wer", ascending=False, kind="stable").to_csv(f"{OUT}/errors_{name}_sorted.csv", index=False)
    return res

# %% [markdown]
# ## 4. Baseline (zero-shot) - run this FIRST and save the number

# %%
base_model = WhisperForConditionalGeneration.from_pretrained(BASE).to(DEV)
results = {"dataset": DATASET_ID, "split": "held-out split, text-deduplicated; speaker overlap not ruled out (session_id is ~1 per clip)", "base": BASE, "n_train": len(train), "n_test": len(test)}
results["baseline"] = evaluate(base_model, "baseline")
results["baseline_prompted"] = evaluate(base_model, "baseline_prompted", prompt=MED_PROMPT)
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
if STAGE == "baseline":
    print("BASELINE DONE"); sys.exit(0)

# %% [markdown]
# ## 5. LoRA fine-tune

# %%
del base_model; import gc; gc.collect(); torch.cuda.empty_cache()   # free GPU before training

def prep(batch):
    a = batch["audio"]
    batch["input_features"] = fe(a["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = tok(batch[TEXT]).input_ids
    return batch

keep_short = lambda x: len(x["audio"]["array"]) < 30 * 16000
train_p = train.filter(keep_short).map(prep, remove_columns=train.column_names)

@dataclass
class Collator:
    def __call__(self, feats):
        inp = fe.pad([{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
        lab = tok.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        if (labels[:, 0] == model.config.decoder_start_token_id).all():
            labels = labels[:, 1:]
        inp["labels"] = labels
        return inp

model = WhisperForConditionalGeneration.from_pretrained(BASE)
model.config.forced_decoder_ids = None; model.config.suppress_tokens = []
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                                         target_modules=["q_proj", "v_proj"], bias="none"))
model.print_trainable_parameters()

args = Seq2SeqTrainingArguments(
    output_dir=f"{OUT}/lora", per_device_train_batch_size=8, gradient_accumulation_steps=2,
    learning_rate=LR, warmup_steps=50, max_steps=MAX_STEPS, fp16=(DEV == "cuda"),
    logging_steps=25, save_steps=MAX_STEPS, remove_unused_columns=False,
    label_names=["labels"], report_to="none", seed=SEED)
Seq2SeqTrainer(model=model, args=args, train_dataset=train_p, data_collator=Collator()).train()
model.save_pretrained(f"{OUT}/lora_adapter")

# %% [markdown]
# ## 6. Evaluate fine-tuned model

# %%
model = model.to(DEV)
results["finetuned"] = evaluate(model, "finetuned")
results["finetuned_prompted"] = evaluate(model, "finetuned_prompted", prompt=MED_PROMPT)
results["config"] = dict(lora_r=LORA_R, lr=LR, steps=MAX_STEPS, eff_batch=16, n_train=len(train_p))
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
print(json.dumps(results, indent=1))

# %% [markdown]
# ## 7. Inference speed (for a latency line on the resume, only if you measure it)

# %%
import time as _t
merged = model.merge_and_unload().to(DEV).half().eval()
clip = [test[0]["audio"]["array"]]
for _ in range(3): transcribe(merged, clip)
t = _t.time(); [transcribe(merged, clip) for _ in range(10)]
dur = len(clip[0]) / 16000
print(f"avg latency {(_t.time()-t)/10:.2f}s for {dur:.1f}s audio -> RTF {((_t.time()-t)/10)/dur:.3f}")

# %% [markdown]
# ## 8. OPTIONAL: zero-shot cross-dataset check on corti/med-term (English, 200 dictation utterances)
# Run only if time allows. Different speakers + dictation style = a second, unseen test set.
# Check license/gating on the dataset page before citing it. Clips may exceed 30 s, so we use chunked inference.

# %%
if not RUN_CORTI:
    print("ALL DONE (corti skipped)"); sys.exit(0)
from transformers import pipeline

ct = load_dataset("corti/med-term", "en", split="test").cast_column("audio", Audio(sampling_rate=16000))
print(ct); print(ct[0]["transcription"], "|", ct[0]["medical_terms"])
ct_refs = ct["transcription"]
ct_terms = [sum(extract_entities(m, r).values(), []) for m, r in zip(ct["medical_terms"], ct_refs)]
print("utterances with matched terms:", sum(1 for t in ct_terms if t))

def eval_corti(m, name):
    asr = pipeline("automatic-speech-recognition", model=m, tokenizer=tok, feature_extractor=fe,
                   chunk_length_s=30, stride_length_s=5, batch_size=8, device=0 if DEV == "cuda" else -1,
                   torch_dtype=torch.float16 if DEV == "cuda" else torch.float32)
    hyps = [o["text"].strip() for o in asr([x["array"] for x in ct["audio"]],
                                           generate_kwargs={"language": "en", "task": "transcribe"})]
    w, c = wer_cer(ct_refs, hyps); mt, n = term_error_rate(hyps, ct_terms)
    r = {"wer": w, "cer": c, "term_error_rate": mt, "n_terms": n, "n_utts": len(hyps)}
    print(name, json.dumps(r, indent=1)); return r

results["corti_baseline"] = eval_corti(WhisperForConditionalGeneration.from_pretrained(BASE).to(DEV), "corti_baseline")
results["corti_finetuned"] = eval_corti(merged, "corti_finetuned")
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
print("ALL DONE")
