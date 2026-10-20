"""Compare execution-matched classical search with the completed neural matrix.

Reads results/raw/classical_matched_v1/official.csv and the official neural
evaluation CSVs. Infeasible classical rows keep a blank objective, so this
summary reports feasibility separately from the mean of feasible objectives.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSICAL = ROOT / "results" / "raw" / "classical_matched_v1" / "official.csv"
NEURAL = ROOT / "results" / "raw" / "bc_rl_full_v1" / "eval"
OUT = ROOT / "results" / "tables" / "classical_matched_v1_summary.csv"

MODELS = {"H2": "RT", "H4": "RT + DeepSets"}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    if not CLASSICAL.exists():
        raise SystemExit(f"missing {CLASSICAL}")
    neural_by_seed: dict[tuple, list[float]] = defaultdict(list)
    for path in NEURAL.glob("H*_seed*.csv"):
        model, seed = path.stem.split("_seed")
        if model not in MODELS:
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["size"]), row["protocol"], model, row["role"], row["instance"])
                neural_by_seed[key].append(float(row["objective_h"]))
    neural_instance = {key: mean(vals) for key, vals in neural_by_seed.items()}

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    with CLASSICAL.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["budget_kind"], int(row["size"]), row["budget_protocol"], row["method"])].append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "budget_kind",
        "size",
        "protocol",
        "method",
        "n",
        "feasible",
        "budget_value",
        "mean_executions",
        "mean_requests",
        "mean_cache_hits",
        "mean_runtime_s",
        "mean_feasible_objective_h",
        "rt_bc_mean_h",
        "rt_bc_on_classical_feasible_h",
        "rt_rl_mean_h",
        "hat_bc_mean_h",
        "hat_bc_on_classical_feasible_h",
        "hat_rl_mean_h",
    ]
    rows_out = []
    for (kind, size, protocol, method), items in sorted(grouped.items()):
        if len(items) != 20:
            continue
        feasible = [item for item in items if item["feasible"] == "1" and item["objective_h"]]
        names = {item["instance"] for item in feasible}

        def neural_mean(model: str, role: str, only: set[str] | None = None) -> float:
            vals = [
                neural_instance[(size, protocol, model, role, item["instance"])]
                for item in items
                if (only is None or item["instance"] in only)
                and (size, protocol, model, role, item["instance"]) in neural_instance
            ]
            return mean(vals)

        rows_out.append(
            {
                "budget_kind": kind,
                "size": size,
                "protocol": protocol,
                "method": method,
                "n": len(items),
                "feasible": len(feasible),
                "budget_value": items[0]["budget_value"],
                "mean_executions": f"{mean([float(i['oracle_calls']) for i in items]):.2f}",
                "mean_requests": f"{mean([float(i['oracle_requests']) for i in items]):.2f}",
                "mean_cache_hits": f"{mean([float(i['oracle_cache_hits']) for i in items]):.2f}",
                "mean_runtime_s": f"{mean([float(i['runtime_s']) for i in items]):.4f}",
                "mean_feasible_objective_h": f"{mean([float(i['objective_h']) for i in feasible]):.4f}" if feasible else "",
                "rt_bc_mean_h": f"{neural_mean('H2', 'bc_best'):.4f}",
                "rt_bc_on_classical_feasible_h": f"{neural_mean('H2', 'bc_best', names):.4f}" if names else "",
                "rt_rl_mean_h": f"{neural_mean('H2', 'rl_best'):.4f}",
                "hat_bc_mean_h": f"{neural_mean('H4', 'bc_best'):.4f}",
                "hat_bc_on_classical_feasible_h": f"{neural_mean('H4', 'bc_best', names):.4f}" if names else "",
                "hat_rl_mean_h": f"{neural_mean('H4', 'rl_best'):.4f}",
            }
        )
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {OUT} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
