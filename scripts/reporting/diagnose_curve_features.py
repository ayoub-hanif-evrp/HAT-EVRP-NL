"""Diagnostic for static charging-curve features. No retraining."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import cumulative_charging_time
from data import generate_synthetic, instance_files, parse_instance

DATA = ROOT / "dataset" / "montoya-et-al-2017"
OUT = ROOT / "results" / "tables" / "final" / "curve_feature_diagnostic.csv"
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
LATE = (0.5, 0.85, 0.95, 1.0)


def _features(instance, fractions):
    rows = []
    for station in instance.stations:
        node = instance.nodes[station]
        curve = instance.curves[node.station_type]
        full = max(curve.times[-1], 1e-9)
        values = [
            cumulative_charging_time(curve, fraction * instance.battery_wh) / full
            for fraction in fractions
        ]
        late_hours = [
            cumulative_charging_time(curve, fraction * instance.battery_wh)
            for fraction in LATE
        ]
        rows.append(
            {
                "instance": instance.name,
                "station_type": node.station_type,
                "grid": ",".join(f"{value:.6f}" for value in values),
                "f25": values[0],
                "f50": values[1],
                "f75": values[2],
                "f100": values[3],
                "hours_50": late_hours[0],
                "hours_85": late_hours[1],
                "hours_95": late_hours[2],
                "hours_100": late_hours[3],
                "share_of_time_in_last_15pct_energy": (late_hours[3] - late_hours[1]) / late_hours[3],
            }
        )
    return rows


def main() -> None:
    official = []
    for path in instance_files(DATA):
        instance = parse_instance(path)
        if len(instance.customers) in (10, 20, 40):
            official.append(instance)
    synthetic = [
        generate_synthetic(10, 11_100_000 + index, generator_version="v2")
        for index in range(40)
    ]
    rows = []
    for group, instances in (("official", official), ("synthetic_v2_train_prefix", synthetic)):
        for instance in instances:
            for row in _features(instance, FRACTIONS):
                row["group"] = group
                rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_type = defaultdict(set)
    for row in rows:
        if row["group"] == "official":
            by_type[row["station_type"]].add(row["grid"])
    summary = {
        "official_instances": len(official),
        "synthetic_instances": len(synthetic),
        "distinct_feature_vectors_by_type_on_official": {
            key: sorted(value) for key, value in by_type.items()
        },
        "breakpoints_wh": [0, 13600, 15200, 16000],
        "sampled_fractions": list(FRACTIONS),
        "fractions_on_first_segment": [0.25, 0.5, 0.75],
        "note": (
            "0.75Q is still on the first linear piece because the first kink is at 0.85Q. "
            "F(Q) includes the slow tail, so F(0.75Q)/F(Q) is below 0.75. "
            "If each technology has one feature vector, the station-type one-hot already identifies it."
        ),
    }
    summary_path = OUT.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
