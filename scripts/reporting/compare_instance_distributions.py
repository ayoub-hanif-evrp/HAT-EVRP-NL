"""Compare training, official, and slow-charger instance distributions. No training."""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import SyntheticConfig, generate_synthetic, instance_files, parse_instance

OUT = ROOT / "results" / "tables" / "final" / "distribution_coverage.csv"


def _holdout(index: int):
    config = SyntheticConfig(
        geometry="clustered",
        station_placement="pmedian_region",
        density=0.2,
        tech_pool=("slow",),
        cluster_sigma=6.0,
        n_clusters=3,
        station_count=2,
        depot_bias="center",
        layout_family="compact",
        region_span=120.0,
    )
    return generate_synthetic(10, 21_300_000 + index, config=config, generator_version="v2")


def _profile(instance):
    customers = [instance.nodes[index] for index in instance.customers]
    depot = instance.nodes[instance.depot]
    station_nodes = [instance.nodes[index] for index in instance.stations]
    types = [node.station_type for node in station_nodes]
    distances = [
        math.hypot(node.x - depot.x, node.y - depot.y) for node in customers
    ]
    energies = [instance.energy_wh[instance.depot][node.id] for node in customers]
    xs = [node.x for node in customers]
    ys = [node.y for node in customers]
    return {
        "n_customers": len(customers),
        "n_stations": len(station_nodes),
        "frac_slow": types.count("slow") / len(types),
        "frac_normal": types.count("normal") / len(types),
        "frac_fast": types.count("fast") / len(types),
        "slow_only": int(set(types) == {"slow"}),
        "depot_customer_km": sum(distances) / len(distances),
        "span_km": max(max(xs) - min(xs), max(ys) - min(ys)),
        "mean_depot_energy_wh": sum(energies) / len(energies),
        "max_depot_energy_over_battery": max(energies) / instance.battery_wh,
        "service_h": sum(instance.service_times[index] for index in instance.customers),
        "tmax_h": instance.max_route_hours,
    }


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def main() -> None:
    cache = pickle.loads(
        (ROOT / "results" / "raw" / "bc_rl_full_v1" / "teacher_cache_n10_train500_val100.pkl").read_bytes()
    )
    groups = {
        "teacher_cache_train": [pair[0] for pair in cache["train"]],
        "teacher_cache_validation": [pair[0] for pair in cache["val"]],
        "raw_v2_seed_block_train200": [
            generate_synthetic(10, 11_100_000 + index, generator_version="v2")
            for index in range(200)
        ],
        "raw_v2_seed_block_val100": [
            generate_synthetic(10, 11_200_000 + index, generator_version="v2")
            for index in range(100)
        ],
        "slow_holdout": [_holdout(index) for index in range(20)],
        "official_n10": [],
        "official_n20": [],
        "official_n40": [],
    }
    for path in instance_files(ROOT / "dataset" / "montoya-et-al-2017"):
        instance = parse_instance(path)
        size = len(instance.customers)
        if size in (10, 20, 40):
            groups[f"official_n{size}"].append(instance)
    summary = []
    for name, instances in groups.items():
        profiles = [_profile(instance) for instance in instances]
        summary.append(
            {
                "group": name,
                "n": len(profiles),
                "mean_stations": round(_mean(profiles, "n_stations"), 3),
                "frac_slow_stations": round(_mean(profiles, "frac_slow"), 3),
                "frac_normal_stations": round(_mean(profiles, "frac_normal"), 3),
                "frac_fast_stations": round(_mean(profiles, "frac_fast"), 3),
                "slow_only_instance_rate": round(_mean(profiles, "slow_only"), 3),
                "mean_depot_customer_km": round(_mean(profiles, "depot_customer_km"), 3),
                "mean_span_km": round(_mean(profiles, "span_km"), 3),
                "mean_depot_energy_wh": round(_mean(profiles, "mean_depot_energy_wh"), 1),
                "mean_max_depot_energy_over_battery": round(
                    _mean(profiles, "max_depot_energy_over_battery"), 3
                ),
                "service_h": profiles[0]["service_h"],
                "tmax_h": profiles[0]["tmax_h"],
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
