"""Regenerate locked BC/RL routes and split duration into travel, service, and charging.

Evaluation only. The published objective is compared with the regenerated
objective. A mismatch stops the run. Rows are appended, so a stopped run resumes.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import decompose_solution, route_sequence_fingerprint
from data import parse_instance
from train import decode, load_policy

sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from eval_bc_rl_official import instances_of_size

OUT = ROOT / "results" / "raw" / "bc_rl_route_decomposition_v1"
SAMPLES = ROOT / "results" / "raw" / "bc_rl_full_v1" / "eval_stochastic" / "official_samples.csv"
RUNS = ROOT / "results" / "raw" / "bc_rl_full_v1" / "runs"
MODELS = ("H2", "H4")
SIZES = {10, 20}
TOLERANCE_H = 1e-4
SAMPLE_SEED0 = 910_000
FIELDS = [
    "model",
    "seed",
    "role",
    "size",
    "instance_index",
    "instance",
    "protocol",
    "trial",
    "published_objective_h",
    "regenerated_objective_h",
    "objective_abs_error_h",
    "feasible",
    "identity_ok",
    "fingerprint",
    "n_routes",
    "n_charges",
    "station_types",
    "travel_h",
    "service_h",
    "charging_h",
    "full_h",
    "energy_charged_wh",
    "oracle_calls",
    "runtime_s",
]
CHARGE_FIELDS = [
    "model",
    "seed",
    "role",
    "size",
    "instance",
    "protocol",
    "trial",
    "route_index",
    "station_node",
    "station_type",
    "soc_in_wh",
    "energy_wh",
    "soc_out_wh",
    "charge_h",
]


def _done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["model"], row["seed"], row["role"], row["size"], row["instance"], row["protocol"], row["trial"])
            for row in csv.DictReader(handle)
        }


def _charges(instance, plans) -> list[dict]:
    events = []
    for route_index, plan in enumerate(plans):
        if not plan.feasible or plan.stops is None:
            continue
        soc = instance.battery_wh
        stops = plan.stops
        for index, (node, amount) in enumerate(stops):
            if amount is not None and instance.nodes[node].kind == 2 and float(amount) > 1e-7:
                added = float(amount)
                q_out = min(soc + added, instance.battery_wh)
                curve = instance.curves[instance.nodes[node].station_type]
                from charging import charging_time

                events.append(
                    {
                        "route_index": route_index,
                        "station_node": node,
                        "station_type": instance.nodes[node].station_type,
                        "soc_in_wh": soc,
                        "energy_wh": q_out - soc,
                        "soc_out_wh": q_out,
                        "charge_h": charging_time(curve, soc, q_out),
                    }
                )
                soc = q_out
            if index + 1 < len(stops):
                soc = max(0.0, soc - instance.energy_wh[node][stops[index + 1][0]])
    return events


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    solution_path = OUT / "solutions.csv"
    charge_path = OUT / "charges.csv"
    done = _done(solution_path)
    new_solution = not solution_path.exists()
    new_charge = not charge_path.exists()
    paths = {size: instances_of_size(ROOT / "dataset", size) for size in SIZES}
    policies: dict[tuple, object] = {}
    mismatches = []
    with SAMPLES.open(encoding="utf-8", newline="") as source, solution_path.open(
        "a", encoding="utf-8", newline=""
    ) as solution_handle, charge_path.open("a", encoding="utf-8", newline="") as charge_handle:
        solution_writer = csv.DictWriter(solution_handle, fieldnames=FIELDS)
        charge_writer = csv.DictWriter(charge_handle, fieldnames=CHARGE_FIELDS)
        if new_solution:
            solution_writer.writeheader()
        if new_charge:
            charge_writer.writeheader()
        for row in csv.DictReader(source):
            if row["model"] not in MODELS or int(row["size"]) not in SIZES:
                continue
            if row["protocol"] not in {"greedy", "sample"}:
                continue
            key = (
                row["model"],
                row["seed"],
                row["role"],
                row["size"],
                row["instance"],
                row["protocol"],
                row["trial"],
            )
            if key in done:
                continue
            size = int(row["size"])
            index = int(row["instance_index"])
            instance_path = paths[size][index]
            instance = parse_instance(instance_path)
            if instance.name != row["instance"]:
                raise SystemExit(f"instance order mismatch at {row['instance']}")
            policy_key = (row["model"], row["seed"], row["role"])
            if policy_key not in policies:
                filename = "bc_best.pt" if row["role"] == "bc" else "rl_best.pt"
                policies[policy_key] = load_policy(RUNS / f"{row['model']}_seed{row['seed']}" / filename)
            seed = 0 if row["protocol"] == "greedy" else SAMPLE_SEED0 + index * 16 + int(row["trial"])
            with torch.no_grad():
                result = decode(
                    policies[policy_key],
                    instance,
                    sample=row["protocol"] == "sample",
                    seed=seed,
                )
            published = float(row["objective_h"]) if row["objective_h"] else float("nan")
            regenerated = float(result["objective"]) if result.get("feasible") else float("inf")
            error = abs(regenerated - published)
            if error > TOLERANCE_H:
                mismatches.append({**{name: key[i] for i, name in enumerate(("model", "seed", "role", "size", "instance", "protocol", "trial"))}, "published": published, "regenerated": regenerated})
                print({"mismatch": mismatches[-1]}, flush=True)
                break
            part = decompose_solution(instance, result["plans"])
            if not part.get("identity_ok"):
                raise SystemExit(f"decomposition identity failed for {key}: {part.get('reason')}")
            customers = [
                tuple(node for node, _amount in plan.stops if instance.nodes[node].kind == 1)
                for plan in result["plans"]
                if plan.stops is not None
            ]
            solution_writer.writerow(
                {
                    "model": row["model"],
                    "seed": row["seed"],
                    "role": row["role"],
                    "size": row["size"],
                    "instance_index": index,
                    "instance": row["instance"],
                    "protocol": row["protocol"],
                    "trial": row["trial"],
                    "published_objective_h": published,
                    "regenerated_objective_h": regenerated,
                    "objective_abs_error_h": error,
                    "feasible": int(bool(part["feasible"])),
                    "identity_ok": int(bool(part["identity_ok"])),
                    "fingerprint": route_sequence_fingerprint(customers),
                    "n_routes": part["n_routes"],
                    "n_charges": part["n_charges"],
                    "station_types": "|".join(part["station_types"]),
                    "travel_h": part["travel_h"],
                    "service_h": part["service_h"],
                    "charging_h": part["charging_h"],
                    "full_h": part["full_h"],
                    "energy_charged_wh": part["energy_charged_wh"],
                    "oracle_calls": result.get("oracle_calls", ""),
                    "runtime_s": result.get("runtime", ""),
                }
            )
            for event in _charges(instance, result["plans"]):
                charge_writer.writerow(
                    {
                        "model": row["model"],
                        "seed": row["seed"],
                        "role": row["role"],
                        "size": row["size"],
                        "instance": row["instance"],
                        "protocol": row["protocol"],
                        "trial": row["trial"],
                        **event,
                    }
                )
            solution_handle.flush()
            charge_handle.flush()
            done.add(key)
        if not mismatches:
            print({"wrote": str(solution_path), "rows": len(done)}, flush=True)
    if mismatches:
        mismatch_path = OUT / "objective_mismatches.csv"
        with mismatch_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mismatches[0]))
            writer.writeheader()
            writer.writerows(mismatches)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
