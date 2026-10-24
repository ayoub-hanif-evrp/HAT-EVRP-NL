"""Create disjoint synthetic splits with no teacher-feasibility filter.

The protocol file is written before the instances. Every generated instance is
kept. Hashes are stored before any policy evaluation.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import ChargingCurve, Instance, SyntheticConfig, generate_synthetic

OUT = ROOT / "results" / "raw" / "final_autonomous_study" / "splits"
PROTOCOL = {
    "name": "unbiased_synthetic_splits_v1",
    "parent_commit": "789af01b5c338fe814c470a46c31e78e070f39e6",
    "size": 10,
    "teacher_filter": False,
    "feasibility_rule": "independent nonlinear checker at evaluation time; infeasible decodes use penalized duration n*T_max",
    "penalized_cap": "n_customers * max_route_hours",
    "not_excluded": "instances on which a teacher fails are retained",
    "historical_not_used_as_confirmation": [
        "teacher-cache validation seeds at and after 11200000",
        "official Montoya instances",
        "slow holdout seeds 21300000-21300019",
    ],
    "splits": [
        {"name": "dev_v2", "role": "development_validation", "generator": "v2", "seed_start": 31_000_000, "n": 100, "sample_seed0": 950_000},
        {"name": "confirm_v2", "role": "locked_final_confirmation", "generator": "v2", "seed_start": 32_000_000, "n": 100, "sample_seed0": 960_000},
        {"name": "shift_v2c", "role": "distribution_shift", "generator": "v2c", "seed_start": 33_000_000, "n": 40, "sample_seed0": 961_000},
        {
            "name": "shift_slow",
            "role": "distribution_shift",
            "generator": "v2",
            "seed_start": 34_000_000,
            "n": 40,
            "sample_seed0": 962_000,
            "config": {
                "geometry": "clustered",
                "station_placement": "pmedian_region",
                "density": 0.2,
                "tech_pool": ["slow"],
                "cluster_sigma": 6.0,
                "n_clusters": 3,
                "station_count": 2,
                "depot_bias": "center",
                "layout_family": "compact",
                "region_span": 120.0,
            },
        },
        {
            "name": "shift_geo",
            "role": "distribution_shift",
            "generator": "v2",
            "seed_start": 35_000_000,
            "n": 40,
            "sample_seed0": 963_000,
            "config": {
                "geometry": "clustered",
                "station_placement": "pmedian_region",
                "density": 0.3,
                "tech_pool": ["fast", "normal", "slow"],
                "cluster_sigma": 8.0,
                "n_clusters": 4,
                "station_count": 3,
                "depot_bias": "offset",
                "layout_family": "elongated",
                "region_span": 140.0,
            },
        },
        {"name": "curve_probe", "role": "curve_shape_probe", "generator": "v2", "seed_start": 36_000_000, "n": 24, "sample_seed0": 964_000, "curve_perturbation": True},
    ],
}


def _config(spec: dict) -> SyntheticConfig | None:
    raw = spec.get("config")
    if raw is None:
        return None
    payload = dict(raw)
    payload["tech_pool"] = tuple(payload["tech_pool"])
    return SyntheticConfig(**payload)


def _instance_hash(instance: Instance) -> str:
    payload = {
        "name": instance.name,
        "nodes": [
            [node.id, node.kind, round(node.x, 6), round(node.y, 6), node.station_type]
            for node in instance.nodes
        ],
        "curves": {
            key: {"times": list(curve.times), "energies": list(curve.energies)}
            for key, curve in sorted(instance.curves.items())
        },
        "battery_wh": instance.battery_wh,
        "max_route_hours": instance.max_route_hours,
        "service_times": list(instance.service_times),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _perturb_curves(instance: Instance, seed: int) -> Instance:
    """Vary breakpoint slopes while keeping station-type labels and geometry."""
    rng = random.Random(seed + 17)
    curves = {}
    for name, curve in instance.curves.items():
        widths = [curve.energies[i + 1] - curve.energies[i] for i in range(len(curve.energies) - 1)]
        base_dt = [curve.times[i + 1] - curve.times[i] for i in range(len(curve.times) - 1)]
        chosen = None
        for _ in range(200):
            multipliers = [rng.uniform(0.55, 1.7) for _ in widths]
            rates = [base_dt[i] * multipliers[i] / widths[i] for i in range(len(widths))]
            if all(rates[i] < rates[i + 1] for i in range(len(rates) - 1)):
                chosen = multipliers
                break
        if chosen is None:
            chosen = [1.0, 1.15, 1.4]
        times = [0.0]
        for delta, multiplier in zip(base_dt, chosen):
            times.append(times[-1] + delta * multiplier)
        curves[name] = ChargingCurve(name, tuple(times), tuple(curve.energies))
    return Instance(
        name=instance.name + "_curved",
        path=instance.path,
        nodes=instance.nodes,
        depot=instance.depot,
        customers=instance.customers,
        stations=instance.stations,
        service_times=instance.service_times,
        speed_kmph=instance.speed_kmph,
        consumption_wh_per_km=instance.consumption_wh_per_km,
        battery_wh=instance.battery_wh,
        max_route_hours=instance.max_route_hours,
        decimals=instance.decimals,
        curves=curves,
        distances_km=instance.distances_km,
        travel_hours=instance.travel_hours,
        energy_wh=instance.energy_wh,
    )


def _assert_disjoint(splits: dict[str, list[Instance]]) -> None:
    seen = {}
    for name, instances in splits.items():
        for instance in instances:
            if instance.name in seen:
                raise SystemExit(f"duplicate instance {instance.name} in {seen[instance.name]} and {name}")
            seen[instance.name] = name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved != PROTOCOL:
            raise SystemExit("split protocol already exists and does not match this script")
    else:
        protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    splits = {}
    rows = []
    for spec in PROTOCOL["splits"]:
        instances = []
        for offset in range(spec["n"]):
            seed = spec["seed_start"] + offset
            instance = generate_synthetic(
                PROTOCOL["size"],
                seed,
                config=_config(spec),
                generator_version=spec["generator"],
            )
            instances.append(instance)
            rows.append(
                {
                    "split": spec["name"],
                    "variant": "base",
                    "index": offset,
                    "seed": seed,
                    "name": instance.name,
                    "sha256": _instance_hash(instance),
                }
            )
            if spec.get("curve_perturbation"):
                perturbed = _perturb_curves(instance, seed)
                instances.append(perturbed)
                rows.append(
                    {
                        "split": spec["name"],
                        "variant": "perturbed_curves",
                        "index": offset,
                        "seed": seed,
                        "name": perturbed.name,
                        "sha256": _instance_hash(perturbed),
                    }
                )
        splits[spec["name"]] = instances
    _assert_disjoint({key: value for key, value in splits.items() if key != "curve_probe"})
    dev = [instance for instance in splits["dev_v2"]]
    selection = {
        "instances": dev,
        "sample_seed0": 950_000,
        "n_samples": 3,
        "split": "dev_v2",
        "note": "Checkpoint selection for the objective-aligned run uses only this split.",
    }
    (OUT / "instances.pkl").write_bytes(pickle.dumps({"protocol": PROTOCOL, "splits": splits}))
    (OUT / "selection_dev.pkl").write_bytes(pickle.dumps(selection))
    hash_path = OUT / "hashes.csv"
    import csv
    with hash_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print({"splits": {key: len(value) for key, value in splits.items()}, "hashes": len(rows)})


if __name__ == "__main__":
    main()
