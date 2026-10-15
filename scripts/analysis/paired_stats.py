"""Proper paired statistical comparisons for HAT-EVRP-NL results.

Pairing rules
-------------
1. If both ``instance`` and ``seed`` columns exist, pair on (instance, seed).
2. Otherwise pair on instance only.
3. Duplicate keys for the same method raise (or are reported) rather than
   silently averaged away.
4. Feasibility is analyzed separately from objective quality.
5. Infeasible outcomes are **not** dropped before pairing unless the caller
   explicitly requests feasible-only objective analysis — and then unmatched
   partners are reported.

Confidence intervals from the paired bootstrap are descriptive unless a formal
hypothesis test is requested; independence assumptions are stated explicitly.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _is_feasible(row: dict) -> bool | None:
    if "feasible" not in row or row["feasible"] in ("", None):
        return None
    return str(row["feasible"]).lower() in {"1", "true", "yes"}


def _pair_key(row: dict, instance_col: str, seed_col: str | None) -> tuple:
    inst = row[instance_col]
    if seed_col and seed_col in row and row[seed_col] not in ("", None):
        return (inst, str(row[seed_col]))
    return (inst,)


def paired_comparison(
    rows: list[dict],
    *,
    method_a: str,
    method_b: str,
    objective_col: str = "objective_h",
    instance_col: str = "instance",
    method_col: str = "method",
    seed_col: str | None = "seed",
    feasible_only_objective: bool = False,
    bootstrap: int = 2000,
    rng_seed: int = 0,
) -> dict:
    """Paired B−A analysis with explicit match/mismatch accounting.

    Sampling unit: each matched pair key (instance[, seed]).
    Negative mean_delta => method_b better when lower objective is better.
    """
    buckets: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = row.get(method_col)
        if method not in {method_a, method_b}:
            continue
        key = _pair_key(row, instance_col, seed_col)
        buckets[key][method].append(row)

    duplicates = 0
    unmatched_a = 0
    unmatched_b = 0
    matched_keys: list[tuple] = []
    for key, methods in buckets.items():
        na = len(methods.get(method_a, []))
        nb = len(methods.get(method_b, []))
        if na > 1 or nb > 1:
            duplicates += max(0, na - 1) + max(0, nb - 1)
        if na == 0 and nb > 0:
            unmatched_b += 1
        elif nb == 0 and na > 0:
            unmatched_a += 1
        elif na >= 1 and nb >= 1:
            matched_keys.append(key)

    # Feasibility paired outcomes (use first observation if duplicates).
    both_feas = both_infeas = a_only = b_only = 0
    for key in matched_keys:
        ra = buckets[key][method_a][0]
        rb = buckets[key][method_b][0]
        fa, fb = _is_feasible(ra), _is_feasible(rb)
        if fa is None or fb is None:
            continue
        if fa and fb:
            both_feas += 1
        elif (not fa) and (not fb):
            both_infeas += 1
        elif fa and not fb:
            a_only += 1
        else:
            b_only += 1

    # Objective deltas on matched pairs.
    deltas: list[float] = []
    skipped_infeas = 0
    missing_obj = 0
    per_seed: dict[str, list[float]] = defaultdict(list)
    for key in matched_keys:
        ra = buckets[key][method_a][0]
        rb = buckets[key][method_b][0]
        fa, fb = _is_feasible(ra), _is_feasible(rb)
        if feasible_only_objective:
            if fa is False or fb is False:
                skipped_infeas += 1
                continue
        try:
            a = float(ra[objective_col])
            b = float(rb[objective_col])
        except (KeyError, TypeError, ValueError):
            missing_obj += 1
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            missing_obj += 1
            continue
        d = b - a
        deltas.append(d)
        if len(key) > 1:
            per_seed[str(key[1])].append(d)

    n = len(deltas)
    mean = _mean(deltas)
    se = _stdev(deltas) / math.sqrt(n) if n > 1 else 0.0
    wins = sum(1 for d in deltas if d < -1e-9)
    losses = sum(1 for d in deltas if d > 1e-9)
    ties = n - wins - losses

    # Paired bootstrap CI (percentile).
    ci_low = ci_high = float("nan")
    if n >= 2 and bootstrap > 0:
        rng = random.Random(rng_seed)
        boots = []
        for _ in range(bootstrap):
            sample = [deltas[rng.randrange(n)] for _ in range(n)]
            boots.append(_mean(sample))
        boots.sort()
        lo_i = int(0.025 * (bootstrap - 1))
        hi_i = int(0.975 * (bootstrap - 1))
        ci_low, ci_high = boots[lo_i], boots[hi_i]

    seed_summary = {
        seed: {
            "n": len(vals),
            "mean_delta": _mean(vals),
            "wins": sum(1 for d in vals if d < -1e-9),
            "ties": sum(1 for d in vals if abs(d) <= 1e-9),
            "losses": sum(1 for d in vals if d > 1e-9),
        }
        for seed, vals in sorted(per_seed.items())
    }

    unit = "instance×seed" if seed_col else "instance"
    return {
        "method_a": method_a,
        "method_b": method_b,
        "pairing_unit": unit,
        "n_matched_pairs": len(matched_keys),
        "n_objective_pairs": n,
        "unmatched_a_only": unmatched_a,
        "unmatched_b_only": unmatched_b,
        "duplicate_extra_rows": duplicates,
        "skipped_infeasible_pairs": skipped_infeas,
        "missing_objective_pairs": missing_obj,
        "feasibility": {
            "both_feasible": both_feas,
            "both_infeasible": both_infeas,
            "only_a_feasible": a_only,
            "only_b_feasible": b_only,
        },
        "mean_delta": mean,
        "std_delta": _stdev(deltas),
        "se_delta": se,
        "ci95_normal_low": mean - 1.96 * se,
        "ci95_normal_high": mean + 1.96 * se,
        "ci95_bootstrap_low": ci_low,
        "ci95_bootstrap_high": ci_high,
        "wins_b_better": wins,
        "ties": ties,
        "losses_b_worse": losses,
        "per_seed": seed_summary,
        "feasible_only_objective": feasible_only_objective,
        "note": (
            f"Paired on {unit}. Bootstrap CI is descriptive percentile CI under "
            f"resampling of matched pairs; pairs are not independent layouts if "
            f"the same instance appears under multiple seeds."
        ),
    }


# Backward-compatible alias
def paired_layout_deltas(*args, **kwargs):
    kwargs["feasible_only_objective"] = kwargs.get("feasible_only_objective", True)
    return paired_comparison(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--objective", default="objective_h")
    parser.add_argument("--method-col", default="method")
    parser.add_argument("--seed-col", default="seed")
    parser.add_argument("--feasible-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv_path.open(encoding="utf-8")))
    out = paired_comparison(
        rows,
        method_a=args.a,
        method_b=args.b,
        objective_col=args.objective,
        method_col=args.method_col,
        seed_col=args.seed_col or None,
        feasible_only_objective=args.feasible_only,
        bootstrap=args.bootstrap,
    )
    for key, value in out.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
