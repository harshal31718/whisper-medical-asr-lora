"""Check the numbers in README.md against the run result files and print any mismatch.

Usage:  python scripts/check_readme.py [path/to/README.md]

Sources (run2 = canonical, run1 = replicate): out/run2 and out/run1 if present, otherwise results/ and results/run1.
Checks:
  1. every README table row is compared with results.json and BOOTSTRAP.md (exact strings);
  2. every two-decimal number and every comma-formatted count in the README must occur in the sources;
  3. if out/run2 exists: counts from the error CSVs, the training log, probe.json and env.json, and that results/ copies
     are byte-identical to their out/ sources.
Anything that cannot be checked because a source file is missing is listed as UNVERIFIED (it is not a pass).
Exit code 1 if any mismatch is found.
"""
import hashlib
import json
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN2 = ROOT / "out" / "run2" if (ROOT / "out" / "run2" / "results.json").exists() else ROOT / "results"
RUN1 = ROOT / "out" / "run1" if (ROOT / "out" / "run1" / "results.json").exists() else ROOT / "results" / "run1"
ROW = re.compile(r"^\| (?P<m>[^|]+?) \| (?P<a>[\d.]+)% \[[^\]]*\] \| (?P<b>[\d.]+)% \[[^\]]*\] \| "
                 r"(?P<d>[+-][\d.]+) pp \[(?P<lo>[+-][\d.]+) pp, (?P<hi>[+-][\d.]+) pp\] \|$")
TWO_DEC = re.compile(r"(?<![\w.])[+-]?\d+\.\d{2}(?![\d.])")
COMMA_INT = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\d,])")

mismatches: list[str] = []
unverified: list[str] = []


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bootstrap(path: pathlib.Path) -> list[dict]:
    """Tables of the BOOTSTRAP.md: metric -> dict(a, b, d, lo, hi) (strings exactly as printed)."""
    tables: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| metric |"):
            tables.append({})
        elif tables and (m := ROW.match(line)):
            tables[-1][m["m"]] = m.groupdict()
    return tables


def sections(readme: str) -> dict[str, str]:
    parts = re.split(r"^#{2,3} (.+)$", readme, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def section(secs: dict[str, str], prefix: str) -> str:
    for title, text in secs.items():
        if title.startswith(prefix):
            return text
    mismatches.append(f"README has no section starting with '{prefix}'")
    return ""


def expect_row(text: str, label: str, tokens: list[str], where: str) -> None:
    line = next((ln for ln in text.splitlines() if ln.startswith(f"| {label}")), None)
    if line is None:
        mismatches.append(f"[{where}] row '{label}' not found")
        return
    for token in tokens:
        if token not in line:
            mismatches.append(f"[{where}] row '{label}': expected '{token}' in: {line}")


def diff_text(entry: dict) -> str:
    return f"{entry['d']} pp ({entry['lo']} to {entry['hi']})"


def check_tables(secs: dict[str, str], r2: dict, r1: dict, bs2: list[dict], bs1: list[dict]) -> None:
    base, fine, bprompt = r2["baseline"], r2["finetuned"], r2["baseline_prompted"]
    names = {"WER": ("WER", "wer"), "CER": ("CER", "cer"), "Medical-term error (1,885 terms)": ("Term error (all matched terms)", "term_error_rate"),
             "Drug-term error (416 terms)": ("Term error (drugs)", "term_err_drugs")}
    main = section(secs, "Results (run2")
    for label, (bname, key) in names.items():
        entry = bs2[0][bname]
        expect_row(main, label, [pct(base[key]), pct(fine[key]), diff_text(entry)], "run2 table")
        if f"{entry['a']}%" != pct(base[key]) or f"{entry['b']}%" != pct(fine[key]):
            mismatches.append(f"BOOTSTRAP.md and results.json disagree for {bname}: {entry['a']}%/{entry['b']}% vs {pct(base[key])}/{pct(fine[key])}")
    for cat in ("drugs", "clinical_findings", "diagnostics", "misc_medical", "advices"):
        expect_row(main, f"{cat} ({fine['n_' + cat]:,})", [pct(base[f"term_err_{cat}"]), pct(fine[f"term_err_{cat}"])], "category table")

    prompt = section(secs, "Vocabulary prompting")
    for label, bname, key in (("WER", "WER", "wer"), ("CER", "CER", "cer"), ("Medical-term error", "Term error (all matched terms)", "term_error_rate"),
                              ("Drug-term error", "Term error (drugs)", "term_err_drugs")):
        expect_row(prompt, label, [pct(base[key]), pct(bprompt[key]), diff_text(bs2[1][bname])], "prompting table")
    for token in (pct(r2["finetuned_prompted"]["wer"]), pct(fine["wer"])):
        if token not in prompt:
            mismatches.append(f"[prompting text] expected '{token}'")

    rep = section(secs, "Replicate (run1)")
    for label, key in (("WER", "wer"), ("CER", "cer"), ("Medical-term error", "term_error_rate"), ("Drug-term error", "term_err_drugs")):
        expect_row(rep, label, [pct(r1["finetuned"][key]), pct(fine[key])], "replicate table")
    expect_row(rep, "WER difference", [diff_text(bs1[0]["WER"]), diff_text(bs2[0]["WER"])], "replicate table")
    if any(r1["baseline"][k] != base[k] for k in ("wer", "cer", "term_error_rate", "term_err_drugs")):
        mismatches.append("run1 and run2 baselines differ, but the README says they are identical")

    corti = section(secs, "Exploratory")
    cb, cf = r2["corti_baseline"], r2["corti_finetuned"]
    for label, key in (("WER", "wer"), ("CER", "cer"), ("Medical-term error", "term_error_rate")):
        expect_row(corti, label, [pct(cb[key]), pct(cf[key])], "corti table")
    for token in (f"{cb['n_utts']} clips", f"{cb['n_terms']} matched terms"):
        if token not in corti:
            mismatches.append(f"[corti text] expected '{token}'")


def allowed_numbers(r2: dict, r1: dict) -> tuple[set[str], set[str]]:
    decimals: set[str] = set()
    for result in (r2, r1):
        for system in result.values():
            if isinstance(system, dict):
                decimals |= {f"{100 * v:.2f}" for k, v in system.items()
                             if isinstance(v, float) and not k.startswith("n_") and k != "eval_sec"}
    for path in (RUN2 / "BOOTSTRAP.md", RUN1 / "BOOTSTRAP.md"):
        decimals |= {m.lstrip("+") for m in TWO_DEC.findall(path.read_text(encoding="utf-8"))}
    ints = {f"{v:,}" for result in (r2, r1) for system in [result, *[s for s in result.values() if isinstance(s, dict)]]
            for k, v in system.items() if isinstance(v, int) and not isinstance(v, bool) and v >= 1000}
    for path in (RUN2 / "BOOTSTRAP.md", RUN1 / "BOOTSTRAP.md"):        # e.g. the number of bootstrap resamples
        ints |= set(COMMA_INT.findall(path.read_text(encoding="utf-8")))
    return decimals, ints


def check_train_config(readme: str) -> set[str]:
    """README hyperparameters must match kaggle/train.py. Returns the decimals that come from train.py (e.g. dropout)."""
    src = (ROOT / "kaggle" / "train.py").read_text(encoding="utf-8")
    grab = lambda pattern: re.search(pattern, src, re.M)
    rank, steps = grab(r"^LR, LORA_R = ([\de.\-]+), (\d+)"), grab(r"^MAX_STEPS = (\d+)")
    dropout, warmup = grab(r"lora_dropout=([\d.]+)"), grab(r"warmup_steps=(\d+)")
    batch, new_tokens = grab(r"per_device_train_batch_size=(\d+), gradient_accumulation_steps=(\d+)"), grab(r"max_new_tokens=(\d+)")
    if not all((rank, steps, dropout, warmup, batch, new_tokens)):
        unverified.append("hyperparameters (could not parse kaggle/train.py)")
        return set()
    expected = [f"r={rank[2]}", f"alpha={2 * int(rank[2])}", f"dropout {dropout[1]}", f"{steps[1]} steps", f"{warmup[1]} warmup steps",
                f"({batch[1]} x {batch[2]} accumulation)", f"max_new_tokens={new_tokens[1]}", f"learning rate {rank[1]}"]
    mismatches.extend(f"[config] expected '{t}' (from kaggle/train.py)" for t in expected if t not in readme)
    return {dropout[1]}


def strip_code(readme: str) -> str:
    return re.sub(r"```.*?```", "", readme, flags=re.S)


def check_numbers(readme: str, decimals: set[str], ints: set[str], extra_dec: set[str], extra_int: set[str]) -> None:
    text = strip_code(readme)
    for token in TWO_DEC.findall(text):
        if token.lstrip("+") not in decimals | extra_dec:
            mismatches.append(f"two-decimal number not found in any source: {token}")
    for token in COMMA_INT.findall(text):
        if token not in ints | extra_int:
            mismatches.append(f"count not found in any source: {token}")


def error_counts(csv_dir: pathlib.Path, name: str) -> dict:
    df = pd.read_csv(csv_dir / f"errors_{name}_sorted.csv", keep_default_na=False, dtype={"ref": str, "hyp": str, "terms_json": str})
    words = df.ref.str.split().str.len()
    bad = df[df.wer >= 1]
    long_ = df[words >= 20]
    return dict(n=len(df), exact=int((df.wer == 0).sum()), bad=len(bad), short_bad=int((words[bad.index] <= 2).sum()),
                short_all=int((words <= 2).sum()), long_n=len(long_), long_fail=int((long_.wer >= 0.5).sum()))


def check_out_files(readme: str, secs: dict[str, str]) -> tuple[set[str], set[str]]:
    """Checks that need out/run2 (error CSVs, log, probe.json, env.json). Returns extra allowed decimals/ints."""
    extra_dec: set[str] = set()
    extra_int: set[str] = set()
    if not (RUN2 / "errors_finetuned_sorted.csv").exists():
        unverified.append("failure-analysis counts, training loss/time, trainable parameters, environment versions, split sizes (need out/run2)")
        return extra_dec, extra_int
    fine, base = error_counts(RUN2, "finetuned"), error_counts(RUN2, "baseline")
    fa = section(secs, "Failure analysis")
    for token in (f"{fine['exact']} of {fine['n']} utterances are exact matches", f"(baseline: {base['exact']})",
                  f"{fine['bad']} are fully wrong", f"(baseline: {base['bad']})", f"{fine['short_bad']} of those {fine['bad']}",
                  f"{fine['short_all']} of the {fine['n']}", f"{fine['long_fail']} of {fine['long_n']} long clips"):
        if token not in fa:
            mismatches.append(f"[failure analysis] expected '{token}'")
    log = "".join(e["data"] for e in json.loads((RUN2 / "whisper-medasr-lora.log").read_text(encoding="utf-8")))
    params = re.search(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+) \|\| trainable%: ([\d.]+)", log)
    runtime = re.search(r"'train_runtime': '(\d+)'", log)
    losses = re.findall(r"\{'loss': '([\d.]+)'", log)
    extra_dec.add(f"{float(params[3]):.2f}")
    extra_int |= {params[1], f"{int(runtime[1]):,}"}
    for token in (params[1], f"{float(params[3]):.2f}%", f"{int(runtime[1]):,} s", f"from {losses[0]} to {losses[-1]}"):
        if token not in readme:
            mismatches.append(f"[config] expected '{token}' (from training log)")
    probe, env = load_json(RUN2 / "probe.json"), load_json(RUN2 / "env.json")
    extra_int |= {f"{probe['n_rows_after_filter']:,}", f"{probe['n_sessions']:,}"}
    for token in (f"{probe['n_rows_after_filter']:,} clips", f"{probe['n_sessions']:,}", f"{probe['n_train']:,} train / {probe['n_test']} test",
                  f"torch {env['torch']}", f"transformers {env['transformers']}", f"peft {env['peft']}", f"datasets {env['datasets']}"):
        if token not in readme:
            mismatches.append(f"[config] expected '{token}' (from probe.json / env.json)")
    return extra_dec, extra_int


def check_copies() -> None:
    pairs = [(ROOT / "results" / n, RUN2 / n) for n in ("results.json", "BOOTSTRAP.md", "RESULTS.md")]
    pairs.append((ROOT / "results" / "ERROR_ANALYSIS.md", ROOT / "docs" / "ERROR_ANALYSIS.md"))
    pairs += [(ROOT / "results" / "run1" / n, RUN1 / n) for n in ("results.json", "BOOTSTRAP.md", "RESULTS.md")]
    for copy, source in pairs:
        if not copy.exists() or not source.exists() or copy == source:
            continue
        if hashlib.sha256(copy.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            mismatches.append(f"{copy.relative_to(ROOT)} differs from {source.relative_to(ROOT)}")


def main() -> None:
    readme_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    r2, r1 = load_json(RUN2 / "results.json"), load_json(RUN1 / "results.json")
    bs2, bs1 = parse_bootstrap(RUN2 / "BOOTSTRAP.md"), parse_bootstrap(RUN1 / "BOOTSTRAP.md")
    secs = sections(readme)
    check_tables(secs, r2, r1, bs2, bs1)
    extra_dec, extra_int = check_out_files(readme, secs)
    extra_dec |= check_train_config(readme)
    decimals, ints = allowed_numbers(r2, r1)
    check_numbers(readme, decimals, ints, extra_dec, extra_int)
    check_copies()
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"README: {readme_path.name} | run2 source: {RUN2.relative_to(ROOT)} | run1 source: {RUN1.relative_to(ROOT)}")
    for item in unverified:
        print("UNVERIFIED:", item)
    if mismatches:
        print(f"{len(mismatches)} MISMATCH(ES):")
        for item in mismatches:
            print(" -", item)
        sys.exit(1)
    print("OK: no mismatches.")


if __name__ == "__main__":
    main()
