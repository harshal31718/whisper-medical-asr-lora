"""Turn out/<stage>/results.json into a results table + claim guard for the resume.

Usage:  python scripts/make_report.py out/all/results.json  [out/all/probe.json]
Writes RESULTS.md next to results.json and prints it.
Rule enforced here: a bullet may claim an improvement ONLY if this script says CLAIMABLE.
"""
import json, sys, pathlib

path = pathlib.Path(sys.argv[1])
r = json.loads(path.read_text())
probe_p = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else path.with_name("probe.json")
probe = json.loads(probe_p.read_text()) if probe_p.exists() else {}

VARIANTS = ["baseline", "baseline_prompted", "finetuned", "finetuned_prompted"]
present = [v for v in VARIANTS if v in r]
metrics = ["wer", "cer", "term_error_rate"] + sorted(
    {k for v in present for k in r[v] if k.startswith("term_err_")})
pct = lambda x: f"{100 * x:.2f}%" if isinstance(x, (int, float)) else "-"

lines = [f"# Results ({r.get('dataset','?')}, {r.get('split','?')})", "",
         f"train utts: {r.get('n_train','?')} | test utts: {r.get('n_test','?')} | base: {r.get('base','?')}",
         f"config: {r.get('config','(baseline only)')}", ""]
if probe:
    lines += [f"entity categories in test: {probe.get('entity_categories')}",
              f"sessions: {probe.get('n_sessions')} | test utts with >=1 entity: {probe.get('test_with_entities')}", ""]
lines += ["| metric | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
for m in metrics:
    lines.append(f"| {m} | " + " | ".join(pct(r[v].get(m)) for v in present) + " |")
lines.append("| n_terms | " + " | ".join(str(r[v].get("n_terms", "-")) for v in present) + " |")

guard = ["", "## Claim guard (resume bullets may only state CLAIMABLE items)", ""]
def cmp(a, b, label):
    if a not in r or b not in r: return
    for m in metrics:
        x, y = r[a].get(m), r[b].get(m)
        if x is None or y is None or x == 0: continue
        rel = 100 * (x - y) / x
        tag = "CLAIMABLE" if y < x else "DO NOT CLAIM (no improvement)"
        guard.append(f"- {label} {m}: {pct(x)} -> {pct(y)} ({rel:+.1f}% relative reduction) [{tag}]")
cmp("baseline", "finetuned", "fine-tune vs baseline")
cmp("baseline", "baseline_prompted", "prompt-bias (no training) vs baseline")
cmp("baseline", "finetuned_prompted", "fine-tune+prompt vs baseline")
if "corti_baseline" in r and "corti_finetuned" in r:
    for m in ("wer", "cer", "term_error_rate"):
        x, y = r["corti_baseline"][m], r["corti_finetuned"][m]
        guard.append(f"- corti zero-shot transfer {m}: {pct(x)} -> {pct(y)} [{'CLAIMABLE' if y < x else 'DO NOT CLAIM'}]")
if "finetuned" in r and "baseline" in r and r["finetuned"]["wer"] >= r["baseline"]["wer"]:
    guard += ["", "!! Fine-tuning did NOT reduce WER. Do not write an improvement bullet. "
              "Use the Tier B bullets in docs/RESUME_BULLETS.md and explain why in the interview."]
out = "\n".join(lines + guard) + "\n"
path.with_name("RESULTS.md").write_text(out)
print(out)
