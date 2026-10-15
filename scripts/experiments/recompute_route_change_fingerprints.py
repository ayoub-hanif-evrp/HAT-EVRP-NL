"""Recompute e2e route-change rates with vehicle-order-invariant fingerprints.

Does not re-run expensive routing. Reads existing end_to_end CSVs, normalizes
fingerprints by sorting route segments, and writes a versioned output directory.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_io import make_run_directory  # noqa: E402
from charging import route_sequence_fingerprint  # noqa: E402


def normalize_fp(fp: str) -> str:
    if not fp:
        return ""
    sequences = []
    for part in fp.split("|"):
        if not part:
            continue
        sequences.append(tuple(int(x) for x in part.split("-") if x != ""))
    return route_sequence_fingerprint(sequences)


def main() -> None:
    src = ROOT / "results" / "raw" / "nonlinear_charging" / "end_to_end_charging_sensitivity.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    run_dir = make_run_directory(
        ROOT / "results" / "raw" / "nonlinear_charging_reanalysis",
        "fingerprint_invariant",
    )
    # Normalize fingerprints in a copy
    for r in rows:
        r["route_fingerprint_raw"] = r.get("route_fingerprint", "")
        r["route_fingerprint"] = normalize_fp(r.get("route_fingerprint", ""))

    out_e2e = run_dir / "end_to_end_charging_sensitivity_invariant.csv"
    fieldnames = list(rows[0].keys())
    with out_e2e.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_inst: dict[tuple, dict] = {}
    for r in rows:
        by_inst.setdefault((r["size"], r["instance"]), {})[r["charging_model"]] = r
    changes = []
    for (size, name), d in sorted(by_inst.items()):
        if "TRUE-NL" not in d:
            continue
        base = d["TRUE-NL"]["route_fingerprint"]
        for model in ("LINEAR-SECANT", "LINEAR-EARLY-RATE"):
            if model not in d:
                continue
            fp = d[model]["route_fingerprint"]
            changes.append(
                {
                    "size": size,
                    "instance": name,
                    "charging_model": model,
                    "route_changed": int(fp != base and fp != "" and base != ""),
                    "true_nl_feasible": d["TRUE-NL"]["feasible_under_true_physics"],
                    "linear_true_feasible": d[model]["feasible_under_true_physics"],
                    "true_nl_full_h": d["TRUE-NL"]["full_h_true"],
                    "linear_full_h_true": d[model]["full_h_true"],
                }
            )
    out_chg = run_dir / "end_to_end_decision_changes_invariant.csv"
    with out_chg.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0].keys()))
        writer.writeheader()
        writer.writerows(changes)

    # Summary rates
    summary_path = run_dir / "route_change_summary.md"
    lines = ["# Vehicle-order-invariant route-change rates", ""]
    lines.append("| size | model | changed | n | rate |")
    lines.append("|---:|---|---:|---:|---:|")
    for size in sorted({int(c["size"]) for c in changes}):
        for model in ("LINEAR-SECANT", "LINEAR-EARLY-RATE"):
            xs = [c for c in changes if int(c["size"]) == size and c["charging_model"] == model]
            n = len(xs)
            ch = sum(int(c["route_changed"]) for c in xs)
            lines.append(f"| {size} | {model} | {ch} | {n} | {100*ch/n:.1f}% |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {run_dir}")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
