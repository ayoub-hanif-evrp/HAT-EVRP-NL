"""Recount nonlinear-charging rows from instance files, not the summary."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw" / "nonlinear_charging"
OUT = ROOT / "results" / "tables" / "final" / "nonlinear_instance_audit.csv"


def main() -> None:
    specs = (
        (RAW / "fixed_route_charging_comparison.csv", "fixed_route", "plan_"),
        (RAW / "end_to_end_charging_sensitivity.csv", "end_to_end", ""),
    )
    rows_out = []
    for path, experiment, prefix in specs:
        groups = defaultdict(lambda: defaultdict(int))
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if experiment == "fixed_route" and not row["method"].startswith(prefix):
                    continue
                if experiment == "fixed_route" and row["experiment"] != "fixed_route":
                    continue
                key = (experiment, row["size"], row["charging_model"])
                groups[key]["n"] += 1
                groups[key]["planner"] += int(row["feasible_under_planner"])
                groups[key]["true"] += int(row["feasible_under_true_physics"])
                groups[key]["duration"] += int(row["duration_violation"] or 0)
                groups[key]["soc"] += int(row.get("soc_fail") or 0)
                reason = row.get("failure_reason") or ""
                if "missing" in reason:
                    groups[key]["missing"] += 1
        for (experiment_name, size, model), counts in sorted(groups.items()):
            rows_out.append(
                {
                    "experiment": experiment_name,
                    "size": size,
                    "charging_model": model,
                    "n_instances": counts["n"],
                    "planner_feasible": counts["planner"],
                    "true_feasible": counts["true"],
                    "duration_violations": counts["duration"],
                    "soc_failures": counts["soc"],
                    "missing_plan": counts["missing"],
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    for row in rows_out:
        print(row)


if __name__ == "__main__":
    main()
