"""Phase 1–2 diagnostic: can neural-guided construction beat BC greedy?

The protocol files are written before any decode. Training is not started here.
A later training stage is allowed only when the frozen 1% development rule passes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import FixedRouteOracle, decompose_solution
from data import Instance, SyntheticConfig, generate_synthetic
from train import decode, load_policy

STUDY = ROOT / "results" / "raw" / "algorithmic_superiority_v1"
SPLIT_DIR = STUDY / "splits"
OUT = STUDY / "guided_diagnostic"
PARENT = "5fa3e382f6cb408fd35edf10714be693cb1bad0c"

SPLIT_PROTOCOL = {
    "name": "algorithmic_superiority_splits_v1",
    "parent_commit": PARENT,
    "size": 10,
    "teacher_filter": False,
    "coverability": "an instance is physically coverable when every customer has a feasible dedicated route under FixedRouteOracle",
    "quality_subset": "physically coverable instances, defined before any policy decode",
    "all_instances_reported": True,
    "historical_not_reused": [
        "seeds at and after 11200000 in the teacher cache",
        "unfiltered seeds 31000000-36000000",
        "slow holdout 21300000-21300019",
        "official Montoya instances",
    ],
    "splits": [
        {"name": "dev_v2", "role": "development", "generator": "v2", "seed_start": 41_100_000, "n": 40, "evaluated_in_diagnostic": True},
        {"name": "confirm_v2", "role": "confirmation_locked", "generator": "v2", "seed_start": 42_000_000, "n": 40, "evaluated_in_diagnostic": False},
        {"name": "shift_v2c", "role": "distribution_shift", "generator": "v2c", "seed_start": 43_000_000, "n": 24, "evaluated_in_diagnostic": False},
        {
            "name": "shift_slow",
            "role": "distribution_shift",
            "generator": "v2",
            "seed_start": 44_000_000,
            "n": 24,
            "evaluated_in_diagnostic": False,
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
            "seed_start": 45_000_000,
            "n": 24,
            "evaluated_in_diagnostic": False,
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
    ],
}

EVAL_PROTOCOL = {
    "name": "guided_decoder_diagnostic_v1",
    "parent_commit": PARENT,
    "checkpoints": {
        "bc": "results/raw/bc_rl_full_v1/runs/H2_seed{seed}/bc_best.pt",
        "rl": "results/raw/bc_rl_full_v1/runs/H2_seed{seed}/rl_best.pt",
    },
    "seeds": [42, 43, 44, 45, 46],
    "architecture": "H2",
    "n_samples": 8,
    "sample_seed0": 970_000,
    "temperature": 1.0,
    "cost_lambda_control": 0.0,
    "cost_lambda_positive": 4.0,
    "cost_lambda_note": "One predeclared positive coefficient. No coefficient search.",
    "decoder": (
        "Limited-discrepancy neural construction. At one decision on the greedy "
        "trace, replace the greedy action by the next legal action in policy rank, "
        "then finish with greedy decoding. Rank alternatives by policy probability. "
        "Keep the completed feasible plan with the lowest exact nonlinear duration. "
        "No local search, ruin-and-repair, insertion, or destroy operators."
    ),
    "budgets": {
        "narrow": {
            "max_deviations": 4,
            "alternatives": ["customer_rank2"],
            "max_solver_executions": 400,
        },
        "wide": {
            "max_deviations": 8,
            "alternatives": ["customer_rank2", "route_rank2"],
            "max_solver_executions": 1200,
        },
    },
    "practical_threshold_relative": 0.01,
    "threshold_source": "Predeclared before this development evaluation. Not revised after confirmation.",
    "continue_to_training_only_if": (
        "On physically coverable development instances, the best wide-budget guided "
        "route, taking the better of the BC and RL decoders on each instance, has "
        "paired seed-mean duration at least 1% below BC greedy, on at least 4 of 5 "
        "seeds, with a two-sided Student-t 95% interval entirely below zero. "
        "Otherwise stop and do not train."
    ),
    "penalized_cap": "n_customers * max_route_hours",
    "local_search": False,
    "training_in_this_script": False,
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


def _write_json(path: Path, payload: dict) -> None:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != payload:
            raise SystemExit(f"{path} already exists and does not match this script")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_splits() -> dict:
    _write_json(SPLIT_DIR / "protocol.json", SPLIT_PROTOCOL)
    blob_path = SPLIT_DIR / "instances.pkl"
    if blob_path.exists():
        return pickle.loads(blob_path.read_bytes())
    splits = {}
    rows = []
    for spec in SPLIT_PROTOCOL["splits"]:
        instances = []
        for offset in range(spec["n"]):
            seed = spec["seed_start"] + offset
            instance = generate_synthetic(
                SPLIT_PROTOCOL["size"],
                seed,
                config=_config(spec),
                generator_version=spec["generator"],
            )
            oracle = FixedRouteOracle(instance)
            singletons = sum(
                1 for customer in instance.customers if oracle.solve((customer,)).feasible
            )
            digest = _instance_hash(instance)
            rows.append(
                {
                    "split": spec["name"],
                    "instance_index": offset,
                    "name": instance.name,
                    "seed": seed,
                    "sha256": digest,
                    "feasible_singletons": singletons,
                    "n_customers": len(instance.customers),
                    "physically_coverable": int(singletons == len(instance.customers)),
                }
            )
            instances.append(instance)
        splits[spec["name"]] = instances
    blob = {"protocol": SPLIT_PROTOCOL, "splits": splits}
    blob_path.write_bytes(pickle.dumps(blob))
    with (SPLIT_DIR / "hashes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return blob


def _fingerprint(result: dict) -> str:
    return "|".join("-".join(str(node) for node in route) for route in (result.get("routes") or ()))


def _hours(result: dict, instance: Instance) -> float:
    if result["feasible"]:
        return float(result["objective"])
    return float(len(instance.customers) * instance.max_route_hours)


def _parts(instance: Instance, result: dict) -> dict:
    if not result["feasible"]:
        return {"travel_h": "", "service_h": "", "charging_h": ""}
    parts = decompose_solution(instance, result["plans"])
    return {
        "travel_h": parts["travel_h"],
        "service_h": parts["service_h"],
        "charging_h": parts["charging_h"],
    }


def _record_decode(policy, instance, oracle, **kwargs) -> dict:
    with torch.no_grad():
        result = decode(policy, instance, oracle=oracle, local_passes=0, ruin_iterations=0, **kwargs)
    return result


def _choices(step: dict, alternatives: list[str]) -> list[tuple[str, dict]]:
    choices = []
    if "customer_rank2" in alternatives and len(step["customer_ranked"]) > 1:
        alternative = step["customer_ranked"][1]
        if alternative != step["customer_chosen"]:
            choices.append(("customer_rank2", {"route": step["route_chosen"], "customer": alternative}))
    if "route_rank2" in alternatives and len(step["route_ranked"]) > 1:
        alternative = step["route_ranked"][1]
        if alternative != step["route_chosen"]:
            choices.append(("route_rank2", {"route": alternative, "customer": None}))
    return choices


def _better(candidate: dict, incumbent: dict) -> bool:
    if not candidate["feasible"]:
        return False
    if not incumbent["feasible"]:
        return True
    return float(candidate["objective"]) < float(incumbent["objective"]) - 1e-8


def guided_search(policy, instance) -> tuple[dict, dict, dict, list, list[dict]]:
    """Run the narrow budget first, then the extra wide-budget deviations on the same cache."""
    oracle = FixedRouteOracle(instance)
    trace: list = []
    greedy = _record_decode(policy, instance, oracle, sample=False, seed=0, trace=trace, cost_lambda=0.0)
    greedy["solver_executions"] = oracle.solver_executions
    narrow_budget = EVAL_PROTOCOL["budgets"]["narrow"]
    wide_budget = EVAL_PROTOCOL["budgets"]["wide"]
    order: list[tuple[int, str, dict, str]] = []
    for decision, step in enumerate(trace[: int(narrow_budget["max_deviations"])]):
        for name, choice in _choices(step, narrow_budget["alternatives"]):
            order.append((decision, name, choice, "narrow"))
    for decision, step in enumerate(trace[: int(wide_budget["max_deviations"])]):
        for name, choice in _choices(step, wide_budget["alternatives"]):
            if decision < int(narrow_budget["max_deviations"]) and name in narrow_budget["alternatives"]:
                continue
            order.append((decision, name, choice, "wide"))
    best = {"narrow": greedy, "wide": greedy}
    expansions = {"narrow": 0, "wide": 0}
    attempts = []
    narrow_executions = oracle.solver_executions
    for decision, name, choice, phase in order:
        if oracle.solver_executions >= int(wide_budget["max_solver_executions"]):
            break
        counts_for_narrow = phase == "narrow" and oracle.solver_executions < int(narrow_budget["max_solver_executions"])
        forced = [
            {"route": previous["route_chosen"], "customer": previous["customer_chosen"]}
            for previous in trace[:decision]
        ]
        forced.append(choice)
        result = _record_decode(
            policy,
            instance,
            oracle,
            sample=False,
            seed=0,
            forced_actions=forced,
            cost_lambda=0.0,
        )
        expansions["wide"] += 1
        if counts_for_narrow:
            expansions["narrow"] += 1
            narrow_executions = oracle.solver_executions
        delta = ""
        improved = 0
        if result["feasible"] and greedy["feasible"]:
            delta = float(result["objective"]) - float(greedy["objective"])
            improved = int(delta < -1e-8)
        if counts_for_narrow and _better(result, best["narrow"]):
            best["narrow"] = result
        if _better(result, best["wide"]):
            best["wide"] = result
        attempts.append(
            {
                "decision": decision,
                "alternative": name,
                "phase": phase,
                "delta_h": delta,
                "improved": improved,
                "feasible": int(bool(result["feasible"])),
                "fingerprint": _fingerprint(result),
                "executions_after": oracle.solver_executions,
            }
        )
    finished = {}
    for phase in ("narrow", "wide"):
        chosen = dict(best[phase])
        chosen["guided_expansions"] = expansions[phase]
        chosen["guided_executions"] = oracle.solver_executions if phase == "wide" else narrow_executions
        finished[phase] = chosen
    return greedy, finished["narrow"], finished["wide"], trace, attempts


def _sample_block(policy, instance, index: int, greedy_objective: float | None) -> dict:
    oracle = FixedRouteOracle(instance)
    fingerprints = []
    best = None
    entropies = []
    n_better = 0
    n_feasible = 0
    for trial in range(EVAL_PROTOCOL["n_samples"]):
        result = _record_decode(
            policy,
            instance,
            oracle,
            sample=True,
            seed=EVAL_PROTOCOL["sample_seed0"] + index * 16 + trial,
            temperature=EVAL_PROTOCOL["temperature"],
            cost_lambda=0.0,
        )
        fingerprints.append(_fingerprint(result))
        entropies.append(float(result["entropy"]))
        if result["feasible"]:
            n_feasible += 1
            if best is None or float(result["objective"]) < float(best["objective"]):
                best = result
            if greedy_objective is not None and float(result["objective"]) < greedy_objective - 1e-8:
                n_better += 1
    unique = len(set(fingerprints))
    return {
        "sample_unique": unique,
        "sample_duplicate_fraction": 1.0 - unique / EVAL_PROTOCOL["n_samples"],
        "sample_better_count": n_better,
        "sample_feasible_count": n_feasible,
        "sample_best_h": "" if best is None else float(best["objective"]),
        "sample_entropy": sum(entropies) / len(entropies),
        "sample_executions": oracle.solver_executions,
    }


def _policy_row(seed: int, index: int, instance: Instance, digest: str, coverable: int, singletons: int, policy_name: str, policy) -> tuple[dict, list, list]:
    started = time.perf_counter()
    greedy, narrow, wide, trace, attempts = guided_search(policy, instance)
    search_runtime = time.perf_counter() - started
    parts = _parts(instance, greedy)
    lam_oracle = FixedRouteOracle(instance)
    lam_started = time.perf_counter()
    lam = _record_decode(
        policy,
        instance,
        lam_oracle,
        sample=False,
        seed=0,
        cost_lambda=EVAL_PROTOCOL["cost_lambda_positive"],
    )
    greedy_raw = float(greedy["objective"]) if greedy["feasible"] else None
    samples = _sample_block(policy, instance, index, greedy_raw)
    row = {
        "seed": seed,
        "instance_index": index,
        "instance": instance.name,
        "sha256": digest,
        "physically_coverable": coverable,
        "feasible_singletons": singletons,
        "policy": policy_name,
        "greedy_feasible": int(bool(greedy["feasible"])),
        "greedy_h": _hours(greedy, instance),
        "greedy_raw_h": "" if greedy_raw is None else greedy_raw,
        "greedy_travel_h": parts["travel_h"],
        "greedy_service_h": parts["service_h"],
        "greedy_charging_h": parts["charging_h"],
        "greedy_n_routes": greedy["n_routes"],
        "greedy_runtime_s": search_runtime,
        "greedy_executions": greedy.get("solver_executions", ""),
        "greedy_entropy": greedy["entropy"],
        "greedy_fingerprint": _fingerprint(greedy),
        "sample_unique": samples["sample_unique"],
        "sample_duplicate_fraction": samples["sample_duplicate_fraction"],
        "sample_better_count": samples["sample_better_count"],
        "sample_feasible_count": samples["sample_feasible_count"],
        "sample_best_h": samples["sample_best_h"],
        "sample_entropy": samples["sample_entropy"],
        "sample_executions": samples["sample_executions"],
        "lambda4_feasible": int(bool(lam["feasible"])),
        "lambda4_h": _hours(lam, instance),
        "lambda4_fingerprint": _fingerprint(lam),
        "lambda4_executions": lam_oracle.solver_executions,
        "lambda4_runtime_s": time.perf_counter() - lam_started,
        "lambda4_same_as_greedy": int(_fingerprint(lam) == _fingerprint(greedy)),
    }
    if greedy_raw is not None and samples["sample_best_h"] != "":
        row["sample_best_delta_h"] = float(samples["sample_best_h"]) - greedy_raw
    else:
        row["sample_best_delta_h"] = ""
    attributions = []
    for budget_name, best in (("narrow", narrow), ("wide", wide)):
        guided_parts = _parts(instance, best)
        row[f"{budget_name}_feasible"] = int(bool(best["feasible"]))
        row[f"{budget_name}_h"] = _hours(best, instance)
        row[f"{budget_name}_fingerprint"] = _fingerprint(best)
        row[f"{budget_name}_executions"] = best.get("guided_executions", "")
        row[f"{budget_name}_runtime_s"] = search_runtime
        row[f"{budget_name}_expansions"] = best.get("guided_expansions", 0)
        row[f"{budget_name}_travel_h"] = guided_parts["travel_h"]
        row[f"{budget_name}_charging_h"] = guided_parts["charging_h"]
        row[f"{budget_name}_improved"] = int(greedy["feasible"] and best["feasible"] and _better(best, greedy))
        for attempt in attempts:
            if budget_name == "narrow" and attempt["phase"] != "narrow":
                continue
            attributions.append(
                {
                    "seed": seed,
                    "policy": policy_name,
                    "instance_index": index,
                    "budget": budget_name,
                    "decision": attempt["decision"],
                    "alternative": attempt["alternative"],
                    "delta_h": attempt["delta_h"],
                    "improved": attempt["improved"],
                    "feasible": attempt["feasible"],
                    "fingerprint": attempt["fingerprint"],
                }
            )
    return row, trace, attributions


def main() -> int:
    blob = ensure_splits()
    _write_json(OUT / "frozen_protocol.json", EVAL_PROTOCOL)
    hashes = {
        (row["split"], row["name"]): row
        for row in csv.DictReader((SPLIT_DIR / "hashes.csv").open(encoding="utf-8"))
    }
    instances = blob["splits"]["dev_v2"]
    fields_path = OUT / "evaluations.csv"
    attr_path = OUT / "deviations.csv"
    div_path = OUT / "divergences.csv"
    done = set()
    if fields_path.exists():
        with fields_path.open(encoding="utf-8", newline="") as handle:
            done = {(row["seed"], row["instance_index"]) for row in csv.DictReader(handle)}
    started = time.perf_counter()
    policies = {
        seed: {
            name: load_policy(ROOT / pattern.format(seed=seed))
            for name, pattern in EVAL_PROTOCOL["checkpoints"].items()
        }
        for seed in EVAL_PROTOCOL["seeds"]
    }
    with fields_path.open("a", encoding="utf-8", newline="") as handle, attr_path.open(
        "a", encoding="utf-8", newline=""
    ) as attr_handle, div_path.open("a", encoding="utf-8", newline="") as div_handle:
        writer = None
        attr_writer = None
        div_writer = None
        for seed in EVAL_PROTOCOL["seeds"]:
            for index, instance in enumerate(instances):
                key = (str(seed), str(index))
                if key in done:
                    continue
                meta = hashes[("dev_v2", instance.name)]
                if meta["sha256"] != _instance_hash(instance):
                    raise SystemExit(f"hash mismatch for {instance.name}")
                traces = {}
                seed_attributions = []
                rows = []
                for policy_name in ("bc", "rl"):
                    row, trace, attributions = _policy_row(
                        seed,
                        index,
                        instance,
                        meta["sha256"],
                        int(meta["physically_coverable"]),
                        int(meta["feasible_singletons"]),
                        policy_name,
                        policies[seed][policy_name],
                    )
                    rows.append(row)
                    traces[policy_name] = trace
                    seed_attributions.extend(attributions)
                limit = min(len(traces["bc"]), len(traces["rl"]))
                first = ""
                for step in range(limit):
                    if (
                        traces["bc"][step]["route_chosen"] != traces["rl"][step]["route_chosen"]
                        or traces["bc"][step]["customer_chosen"] != traces["rl"][step]["customer_chosen"]
                    ):
                        first = step
                        break
                if first == "" and (
                    len(traces["bc"]) != len(traces["rl"])
                    or rows[0]["greedy_fingerprint"] != rows[1]["greedy_fingerprint"]
                ):
                    first = limit
                divergence = {
                    "seed": seed,
                    "instance_index": index,
                    "same_fingerprint": int(rows[0]["greedy_fingerprint"] == rows[1]["greedy_fingerprint"]),
                    "first_decision": first,
                    "bc_entropy": rows[0]["greedy_entropy"],
                    "rl_entropy": rows[1]["greedy_entropy"],
                    "bc_decisions": len(traces["bc"]),
                    "rl_decisions": len(traces["rl"]),
                }
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    if not done:
                        writer.writeheader()
                    attr_fields = [
                        "seed", "policy", "instance_index", "budget", "decision",
                        "alternative", "delta_h", "improved", "feasible", "fingerprint",
                    ]
                    attr_writer = csv.DictWriter(attr_handle, fieldnames=attr_fields)
                    if not done:
                        attr_writer.writeheader()
                    div_writer = csv.DictWriter(div_handle, fieldnames=list(divergence))
                    if not done:
                        div_writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                for item in seed_attributions:
                    attr_writer.writerow(item)
                div_writer.writerow(divergence)
                handle.flush()
                attr_handle.flush()
                div_handle.flush()
                print(
                    f"seed {seed} instance {index} coverable {meta['physically_coverable']} "
                    f"bc {float(rows[0]['greedy_h']):.3f} wide {float(rows[0]['wide_h']):.3f} "
                    f"rl {float(rows[1]['greedy_h']):.3f} wide {float(rows[1]['wide_h']):.3f}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    (OUT / "status.json").write_text(
        json.dumps({"completed": True, "elapsed_s": elapsed, "rows_skipped_at_start": len(done)}, indent=2),
        encoding="utf-8",
    )
    print({"completed": True, "elapsed_s": round(elapsed, 3)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
