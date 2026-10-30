"""Bootstrap confidence intervals for the Whisper medical-ASR runs.

Usage:  python scripts/bootstrap_ci.py run1     (legacy CSVs: cluster bootstrap, WER/CER only)
        python scripts/bootstrap_ci.py run2     (instrumented CSVs: paired bootstrap over idx, WER/CER/term error)

run1 (CSVs sorted by WER, no index column):
    Rows are matched across systems by identical reference text. Eight reference texts occur twice (16 rows), and those
    clips cannot be paired individually, so every identical-text group is ONE resampling unit (cluster bootstrap).

run2 (CSVs in original test order with idx, edit counts and per-utterance term hits):
    Plain paired bootstrap over utterances: the same resampled idx is applied to both systems.

Corpus rates = sum(errors) / sum(reference units): WER over words, CER over characters, term error over matched entities,
the same definitions as train.py. Point estimates are asserted equal to results.json. 10,000 resamples, seed 42,
95% percentile intervals. The difference is system A minus system B (positive = B has lower error).
"""
import json
import pathlib
import re
import sys

import jiwer
import numpy as np
import pandas as pd

N_BOOT, SEED = 10_000, 42
CSV_KW = dict(keep_default_na=False, dtype={"ref": str, "hyp": str, "terms_json": str})


def norm(text: str) -> str:
    """Identical to norm() in kaggle/train.py."""
    t = str(text).lower()
    t = re.sub(r"[-/]", " ", t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def edit_counts(ref: str, hyp: str) -> dict:
    """Per-utterance word and character edit counts after norm(). Empty normalized references are skipped, as in wer_cer()."""
    rn, hn = norm(ref), norm(hyp)
    if not rn:
        return dict(w_err=0, w_len=0, c_err=0, c_len=0)
    w, c = jiwer.process_words(rn, hn), jiwer.process_characters(rn, hn)
    return dict(
        w_err=w.substitutions + w.deletions + w.insertions, w_len=w.hits + w.substitutions + w.deletions,
        c_err=c.substitutions + c.deletions + c.insertions, c_len=c.hits + c.substitutions + c.deletions,
    )


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def resample_indices(n_units: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n_units, size=(N_BOOT, n_units))


def corpus_rate(err: np.ndarray, length: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return err[idx].sum(axis=1) / length[idx].sum(axis=1)


def summarize(name: str, err_a, len_a, err_b, len_b, idx) -> dict:
    """Point estimates plus bootstrap intervals for one metric; units are rows (paired) or clusters."""
    point_a, point_b = err_a.sum() / len_a.sum(), err_b.sum() / len_b.sum()
    boot_a, boot_b = corpus_rate(err_a, len_a, idx), corpus_rate(err_b, len_b, idx)
    return dict(metric=name, a=point_a, b=point_b, diff=point_a - point_b,
                a_ci=percentile_ci(boot_a), b_ci=percentile_ci(boot_b), diff_ci=percentile_ci(boot_a - boot_b))


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def pp(x: float) -> str:
    return f"{100 * x:+.2f} pp"


def table(title_a: str, title_b: str, rows: list[dict]) -> list[str]:
    lines = [f"| metric | {title_a} [95% CI] | {title_b} [95% CI] | difference {title_a} - {title_b} [95% CI] |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['metric']} | {pct(r['a'])} [{pct(r['a_ci'][0])}, {pct(r['a_ci'][1])}] | "
                     f"{pct(r['b'])} [{pct(r['b_ci'][0])}, {pct(r['b_ci'][1])}] | "
                     f"{pp(r['diff'])} [{pp(r['diff_ci'][0])}, {pp(r['diff_ci'][1])}] |")
    return lines


# ----------------------------------------------------------------------------------------------- run1
def run1(out_dir: pathlib.Path) -> dict:
    base = pd.read_csv(out_dir / "errors_baseline.csv", **CSV_KW)
    fine = pd.read_csv(out_dir / "errors_finetuned.csv", **CSV_KW)
    results = json.loads((out_dir / "results.json").read_text())

    assert len(base) == len(fine), "row counts differ"
    assert sorted(base.ref) == sorted(fine.ref), "the two CSVs do not contain the same reference texts"

    for df in (base, fine):
        counts = pd.DataFrame([edit_counts(r, h) for r, h in zip(df.ref, df.hyp)])
        df[counts.columns] = counts

    for key, df in (("baseline", base), ("finetuned", fine)):      # exact equality with results.json
        wer, cer = df.w_err.sum() / df.w_len.sum(), df.c_err.sum() / df.c_len.sum()
        assert wer == results[key]["wer"], f"{key} WER {wer!r} != results.json {results[key]['wer']!r}"
        assert cer == results[key]["cer"], f"{key} CER {cer!r} != results.json {results[key]['cer']!r}"

    cols = ["w_err", "w_len", "c_err", "c_len"]
    grp_b, grp_f = base.groupby("ref", sort=True)[cols].sum(), fine.groupby("ref", sort=True)[cols].sum()
    assert list(grp_b.index) == list(grp_f.index), "cluster keys differ"
    assert (grp_b.w_len.values == grp_f.w_len.values).all() and (grp_b.c_len.values == grp_f.c_len.values).all(), \
        "reference lengths differ within a cluster"

    sizes = base.groupby("ref").size()
    info = dict(n_rows=len(base), n_units=len(grp_b), n_pooled_groups=int((sizes > 1).sum()),
                n_pooled_rows=int(sizes[sizes > 1].sum()))
    idx = resample_indices(len(grp_b))
    rows = [
        summarize("WER", grp_b.w_err.values, grp_b.w_len.values, grp_f.w_err.values, grp_f.w_len.values, idx),
        summarize("CER", grp_b.c_err.values, grp_b.c_len.values, grp_f.c_err.values, grp_f.c_len.values, idx),
    ]
    for row, key in zip(rows, ("wer", "cer")):
        assert row["a"] == results["baseline"][key] and row["b"] == results["finetuned"][key]
    return dict(info=info, rows=rows)


def write_run1_markdown(out_dir: pathlib.Path, res: dict) -> str:
    info = res["info"]
    lines = [
        "# Bootstrap confidence intervals — run1", "",
        "Source: `errors_baseline.csv`, `errors_finetuned.csv`, `results.json` in this folder. Script: `scripts/bootstrap_ci.py run1`.",
        "", "## Method",
        f"- Corpus WER/CER = sum(S+D+I) / sum(reference length) after the same normalization as `train.py`; point estimates are "
        "asserted equal to `results.json` (exact float equality).",
        f"- {N_BOOT:,} bootstrap resamples, seed {SEED}, 95% percentile intervals. The difference is baseline minus fine-tuned "
        "(positive = fine-tuning lower error), computed on the same resamples for both systems.",
        "", "## Deviation from a per-utterance paired bootstrap (read this)",
        "run1's CSVs were each sorted by their own WER and have no index column, so utterances cannot be matched by row position. "
        f"They were matched by identical reference text. {info['n_pooled_groups']} reference texts occur twice "
        f"({info['n_pooled_rows']} rows, different clips with the same text), and which baseline clip pairs with which "
        "fine-tuned clip is unknown for those rows. Each such group is therefore pooled into one resampling unit.",
        f"The bootstrap resamples **{info['n_units']} units** drawn from {info['n_rows']} utterances "
        f"({info['n_pooled_groups']} units contain 2 clips). Point estimates are unaffected (they equal `results.json`); the "
        "intervals differ slightly from an utterance-level bootstrap.",
        "", "## Results (run1)", "",
    ]
    lines += table("baseline", "fine-tuned", res["rows"])
    lines += ["", "## Not computed",
              "- **Medical-term error and drug-term error: no CI for run1.** The run1 CSVs contain no entity labels and "
              "`probe.json` holds entities for 3 test rows only. See run2 (`out/run2/BOOTSTRAP.md`) for term-error intervals.",
              "- Intervals reflect resampling of the test utterances only. They do not capture training randomness "
              "(single seed per run) or the speaker-overlap issue (about 1 clip per `session_id`).", ""]
    text = "\n".join(lines)
    (out_dir / "BOOTSTRAP.md").write_text(text, encoding="utf-8")
    return text


# ----------------------------------------------------------------------------------------------- run2
def load_instrumented(out_dir: pathlib.Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(out_dir / f"errors_{name}.csv", **CSV_KW)
    assert list(df.idx) == list(range(len(df))), f"errors_{name}.csv is not in original test order"
    return df


def term_arrays(df: pd.DataFrame, category: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-utterance (missed, total) matched-term counts, optionally for one entity category."""
    missed, total = np.zeros(len(df)), np.zeros(len(df))
    for i, tj in enumerate(df.terms_json):
        for cat, _term, found in json.loads(tj):
            if category is None or cat == category:
                total[i] += 1
                missed[i] += not found
    return missed, total


def metric_arrays(df: pd.DataFrame) -> dict:
    """metric name -> (error array, length array) per utterance."""
    out = {
        "WER": ((df.n_sub + df.n_del + df.n_ins).to_numpy(float), df.n_ref_words.to_numpy(float)),
        "CER": (df.char_errors.to_numpy(float), df.n_ref_chars.to_numpy(float)),
        "Term error (all matched terms)": term_arrays(df),
        "Term error (drugs)": term_arrays(df, "drugs"),
    }
    return out


def check_against_results(name: str, arrays: dict, results: dict) -> None:
    ref = results[name]
    point = {m: e.sum() / n.sum() for m, (e, n) in arrays.items()}
    assert point["WER"] == ref["wer"], f"{name} WER {point['WER']!r} != results.json {ref['wer']!r}"
    assert point["CER"] == ref["cer"], f"{name} CER {point['CER']!r} != results.json {ref['cer']!r}"
    assert point["Term error (all matched terms)"] == ref["term_error_rate"], f"{name} term error mismatch"
    assert round(point["Term error (drugs)"], 4) == ref["term_err_drugs"], f"{name} drug term error mismatch"
    assert int(arrays["Term error (all matched terms)"][1].sum()) == ref["n_terms"], f"{name} n_terms mismatch"
    assert int(arrays["Term error (drugs)"][1].sum()) == ref["n_drugs"], f"{name} n_drugs mismatch"


def compare(name_a: str, name_b: str, out_dir: pathlib.Path, results: dict, idx: np.ndarray) -> list[dict]:
    df_a, df_b = load_instrumented(out_dir, name_a), load_instrumented(out_dir, name_b)
    assert len(df_a) == len(df_b) and (df_a.ref.values == df_b.ref.values).all(), "systems are not row-aligned by reference"
    arr_a, arr_b = metric_arrays(df_a), metric_arrays(df_b)
    check_against_results(name_a, arr_a, results)
    check_against_results(name_b, arr_b, results)
    rows = []
    for m in arr_a:
        (ea, la), (eb, lb) = arr_a[m], arr_b[m]
        assert (la == lb).all(), f"{m}: reference unit counts differ between systems"
        rows.append(summarize(m, ea, la, eb, lb, idx))
    return rows


def run2(out_dir: pathlib.Path) -> dict:
    results = json.loads((out_dir / "results.json").read_text())
    n = len(load_instrumented(out_dir, "baseline"))
    idx = resample_indices(n)
    main = compare("baseline", "finetuned", out_dir, results, idx)
    extra = compare("baseline", "baseline_prompted", out_dir, results, idx)
    n_terms = {m: int(t.sum()) for m, (_, t) in metric_arrays(load_instrumented(out_dir, "baseline")).items()}
    return dict(n=n, main=main, prompt=extra, n_terms=n_terms, counts_check={k: results[k].get("counts_check")
                for k in ("baseline", "baseline_prompted", "finetuned", "finetuned_prompted") if k in results})


def write_run2_markdown(out_dir: pathlib.Path, res: dict) -> str:
    lines = [
        "# Bootstrap confidence intervals — run2 (canonical run)", "",
        "Source: instrumented `errors_<system>.csv` (original test order, per-utterance edit counts and term hits) and "
        "`results.json` in this folder. Script: `scripts/bootstrap_ci.py run2`.", "",
        "## Method",
        f"- Paired bootstrap over the {res['n']} test utterances: each of {N_BOOT:,} resamples draws utterances with replacement "
        f"(seed {SEED}) and applies the **same** draw to both systems. 95% percentile intervals.",
        "- Corpus rates: WER = word edits / reference words; CER = character edits / reference characters; term error = "
        "missed matched entity terms / matched entity terms (exact whole-phrase match after normalization, as in `train.py`).",
        "- Point estimates are asserted equal to `results.json` (exact for WER, CER and overall term error; drug term error "
        "equals the 4-decimal value stored there; term counts also asserted).",
        "- Difference = first system minus second system, in percentage points (positive = second system has lower error).",
        f"- Terms per metric (baseline CSV): {res['n_terms']['Term error (all matched terms)']} matched terms overall, "
        f"{res['n_terms']['Term error (drugs)']} drug terms.", "",
        "## Baseline vs fine-tuned (LoRA)", "",
    ]
    lines += table("baseline", "fine-tuned", res["main"])
    lines += ["", "## Additional: baseline vs baseline + vocabulary prompt (no training)", ""]
    lines += table("baseline", "baseline+prompt", res["prompt"])
    lines += ["", "## Limits",
              "- Intervals reflect resampling of the test utterances only. They do not capture training randomness "
              "(one seed per run; compare with run1 for a replicate) or the speaker-overlap issue (about 1 clip per `session_id`).",
              "- Term-error intervals treat each utterance as a unit; terms within an utterance are not independent.",
              f"- Per-utterance count self-check recorded by `train.py` (`counts_check`): {res['counts_check']}", ""]
    text = "\n".join(lines)
    (out_dir / "BOOTSTRAP.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    modes = {"run1": (run1, write_run1_markdown), "run2": (run2, write_run2_markdown)}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        sys.exit(__doc__)
    compute, write = modes[sys.argv[1]]
    out_dir = pathlib.Path("out") / sys.argv[1]
    sys.stdout.reconfigure(encoding="utf-8")
    print(write(out_dir, compute(out_dir)))


if __name__ == "__main__":
    main()
