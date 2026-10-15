"""Layout-aware paired differences and descriptive confidence intervals.

Treat customer layouts (instances), not instance×seed rows, as the primary
paired units unless the caller explicitly requests seed-level pairing.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_layout_deltas(
    rows: list[dict],
    *,
    method_a: str,
    method_b: str,
    objective_col: str = "objective_h",
    instance_col: str = "instance",
    method_col: str = "method",
    seed_col: str | None = "seed",
) -> dict:
    """Return layout-mean (B − A) deltas with descriptive SE over layouts.

    Negative mean => method_b better when lower objective is better.
    """
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = row[method_col]
        if method not in {method_a, method_b}:
            continue
        if row.get("feasible") not in (None, "", "1", "True", True, 1):
            # Keep feasible-only when a feasibility flag exists.
            if str(row.get("feasible", "1")) in {"0", "False", "false"}:
                continue
        by[row[instance_col]][method].append(float(row[objective_col]))

    deltas: list[float] = []
    for _inst, methods in by.items():
        if method_a not in methods or method_b not in methods:
            continue
        # Seed-average within layout, then pair.
        a = _mean(methods[method_a])
        b = _mean(methods[method_b])
        deltas.append(b - a)

    n = len(deltas)
    mean = _mean(deltas)
    se = _stdev(deltas) / math.sqrt(n) if n > 1 else 0.0
    wins = sum(1 for d in deltas if d < -1e-9)
    losses = sum(1 for d in deltas if d > 1e-9)
    ties = n - wins - losses
    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_layouts": n,
        "mean_delta": mean,
        "std_delta": _stdev(deltas),
        "se_delta": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "wins_b": wins,
        "ties": ties,
        "losses_b": losses,
        "note": "Descriptive layout-SE CI; not a claim of independent instance×seed samples.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--a", required=True, help="Baseline method key")
    parser.add_argument("--b", required=True, help="Challenger method key")
    parser.add_argument("--objective", default="objective_h")
    parser.add_argument("--method-col", default="method")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv_path.open(encoding="utf-8")))
    out = paired_layout_deltas(
        rows,
        method_a=args.a,
        method_b=args.b,
        objective_col=args.objective,
        method_col=args.method_col,
    )
    for key, value in out.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
