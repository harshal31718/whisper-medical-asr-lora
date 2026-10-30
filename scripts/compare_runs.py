"""Side-by-side table of two runs' results.json (values copied programmatically, never typed).

Usage:  python scripts/compare_runs.py out/run1/results.json out/run2/results.json
"""
import json
import pathlib
import sys

METRICS = [("WER", "wer"), ("CER", "cer"), ("Term error (all)", "term_error_rate"), ("Term error (drugs)", "term_err_drugs"),
           ("Term error (advices)", "term_err_advices")]
SYSTEMS = ["baseline", "baseline_prompted", "finetuned", "finetuned_prompted"]
CORTI = ["corti_baseline", "corti_finetuned"]


def pct(value) -> str:
    return f"{100 * value:.2f}%" if isinstance(value, (int, float)) else "-"


def delta(a, b) -> str:
    return f"{100 * (b - a):+.2f} pp" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    r1, r2 = (json.loads(pathlib.Path(p).read_text()) for p in sys.argv[1:])
    lines = ["| system | metric | run1 | run2 (canonical) | run2 - run1 |", "|---|---|---|---|---|"]
    for system in SYSTEMS + CORTI:
        for label, key in METRICS:
            if system in CORTI and key not in ("wer", "cer", "term_error_rate"):
                continue
            a, b = r1.get(system, {}).get(key), r2.get(system, {}).get(key)
            if a is None and b is None:
                continue
            lines.append(f"| {system} | {label} | {pct(a)} | {pct(b)} | {delta(a, b)} |")
    lines.append(f"| - | n_terms / n_test (run1; run2) | {r1['baseline']['n_terms']} / {r1['n_test']} | "
                 f"{r2['baseline']['n_terms']} / {r2['n_test']} | - |")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
