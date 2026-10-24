"""Confirm the best-of-K zero gradient and compare action probabilities.

This script does not train. It reads the frozen objective-aligned logs and
scores the saved H2 checkpoints on the coverable development instances.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bc_rl import leave_one_out_baselines, reinforce_loss, return_to_go_loss, reward_value
from train import decode, load_policy

OUT = ROOT / "results" / "raw" / "reinforce_diagnosis_v1"
BOK = ROOT / "results" / "raw" / "final_autonomous_study" / "objective_aligned" / "runs"
LOO = ROOT / "results" / "raw" / "bc_rl_full_v1" / "runs"
DEV = ROOT / "results" / "raw" / "algorithmic_superiority_v1"


def _rows(path: Path):
    return [row for row in csv.DictReader(path.open(encoding="utf-8")) if row["stage"] == "rl"]


def summarize_logs() -> dict:
    rows = []
    for seed in range(42, 47):
        for row in _rows(BOK / f"H2_seed{seed}" / "reward.csv"):
            row["seed"] = seed
            rows.append(row)
    grad = np.array([float(row["grad_norm"]) for row in rows])
    zero = grad == 0
    oracle = np.array([float(row["oracle_calls"]) for row in rows])
    entropy = np.array([float(row["entropy"]) for row in rows])
    useful = int((~zero).sum())
    summary = {
        "best_of_k_updates": len(rows),
        "zero_gradient_updates": useful and int(zero.sum()),
        "zero_gradient_percent": float(100 * zero.mean()),
        "updates_with_a_sample_better_than_greedy": useful,
        "useful_updates_per_1000_solver_executions": float(1000 * useful / oracle.sum()),
        "solver_executions": int(oracle.sum()),
        "entropy_mean": float(entropy.mean()),
        "entropy_first_50": float(entropy[:50].mean()),
        "entropy_last_50": float(entropy[-50:].mean()),
        "zero_gradient_by_seed": {},
    }
    # The expression above stores the wrong zero count when useful is 0. Set it directly.
    summary["zero_gradient_updates"] = int(zero.sum())
    logged_adv = np.array([
        float(row["advantage_abs"]) if row["advantage_abs"] else math.nan for row in rows
    ])
    summary["advantage_abs_on_nonzero_updates_mean"] = float(np.nanmean(logged_adv))
    summary["advantage_abs_on_nonzero_updates_std"] = float(np.nanstd(logged_adv))
    summary["advantage_missing_exactly_on_zero_gradient"] = bool(np.isnan(logged_adv).sum() == zero.sum())
    mean_beats = 0
    comparable = 0
    for row in rows:
        if not row["mean_reward"] or not row["greedy_reward"]:
            continue
        comparable += 1
        if float(row["mean_reward"]) > float(row["greedy_reward"]) + 1e-12:
            mean_beats += 1
    summary["nonzero_updates_whose_mean_sample_beats_greedy"] = mean_beats
    summary["nonzero_updates"] = comparable
    original = {}
    for seed in range(42, 47):
        loo = _rows(LOO / f"H2_seed{seed}" / "reward.csv")
        values = np.array([float(row["grad_norm"]) for row in loo])
        original[str(seed)] = {
            "updates": len(values),
            "zero_gradient_updates": int((np.abs(values) < 1e-12).sum()),
            "mean_grad_norm": float(values.mean()),
        }
        seed_rows = [row for row in rows if row["seed"] == seed]
        seed_grad = np.array([float(row["grad_norm"]) for row in seed_rows])
        summary["zero_gradient_by_seed"][str(seed)] = {
            "zero_percent": float(100 * (seed_grad == 0).mean()),
            "early_100_zero_percent": float(100 * (seed_grad[:100] == 0).mean()),
            "late_100_zero_percent": float(100 * (seed_grad[-100:] == 0).mean()),
        }
    summary["original_loo"] = original
    return summary


def _head_norms(model) -> dict:
    groups = {"route": 0.0, "customer": 0.0, "other": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().pow(2).sum())
        if "route_score" in name:
            groups["route"] += value
        elif "customer_" in name:
            groups["customer"] += value
        else:
            groups["other"] += value
    return {key: math.sqrt(value) for key, value in groups.items()}


def one_batch_gradients() -> dict:
    blob = pickle.loads((DEV / "splits" / "instances.pkl").read_bytes())
    hashes = list(csv.DictReader((DEV / "splits" / "hashes.csv").open(encoding="utf-8")))
    coverable = [
        blob["splits"]["dev_v2"][int(row["instance_index"])]
        for row in hashes
        if row["split"] == "dev_v2" and row["physically_coverable"] == "1"
    ][:2]
    policy = load_policy(ROOT / "results/raw/bc_rl_full_v1/runs/H2_seed42/bc_best.pt")
    report = {}
    for name, loss_of in (("loo", None), ("rtg", None)):
        policy = load_policy(ROOT / "results/raw/bc_rl_full_v1/runs/H2_seed42/bc_best.pt")
        policy.train()
        before = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        losses = []
        for instance in coverable:
            sampled = [
                decode(policy, instance, sample=True, seed=1000 + trial + (0 if name == "loo" else 50))
                for trial in range(4)
            ]
            if name == "loo":
                rewards = [
                    reward_value(row, n_customers=len(instance.customers), max_route_hours=instance.max_route_hours)
                    for row in sampled
                ]
                baselines = leave_one_out_baselines(rewards)
                losses.append(
                    reinforce_loss(
                        [row["log_prob"] for row in sampled],
                        [reward - baseline for reward, baseline in zip(rewards, baselines)],
                        [None] * len(sampled),
                    )
                )
            else:
                losses.append(
                    return_to_go_loss(
                        sampled,
                        n_customers=len(instance.customers),
                        max_route_hours=instance.max_route_hours,
                    )
                )
        optimizer = torch.optim.Adam(policy.parameters(), lr=2e-5)
        optimizer.zero_grad()
        loss = torch.stack(losses).mean()
        loss.backward()
        heads = _head_norms(policy)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        delta = 0.0
        for key, value in policy.state_dict().items():
            delta += float((value.detach() - before[key]).pow(2).sum())
        report[name] = {"loss": float(loss.detach()), "grad_norm": heads, "parameter_l2_change": math.sqrt(delta)}
    # Telescoping error on one feasible greedy decode.
    policy.eval()
    with torch.no_grad():
        result = decode(policy, coverable[0], sample=False, seed=0)
    built = sum(float(plan.duration_hours) for plan in result["plans"])
    report["telescope"] = {
        "feasible": bool(result["feasible"]),
        "sum_marginal_h": sum(result["marginal_hours"]),
        "sum_plan_h": built,
        "objective_h": result["objective"] if result["feasible"] else None,
        "abs_error_h": abs(sum(result["marginal_hours"]) - (result["objective"] if result["feasible"] else built)),
    }
    return report


def credit() -> dict:
    blob = pickle.loads((DEV / "splits" / "instances.pkl").read_bytes())
    instances = blob["splits"]["dev_v2"]
    hashes = {
        int(row["instance_index"]): row
        for row in csv.DictReader((DEV / "splits" / "hashes.csv").open(encoding="utf-8"))
        if row["split"] == "dev_v2"
    }
    improved = {
        (row["seed"], row["policy"], row["instance_index"], row["decision"], row["alternative"])
        for row in csv.DictReader((DEV / "guided_diagnostic" / "deviations.csv").open(encoding="utf-8"))
        if row["budget"] == "wide" and row["improved"] == "1"
    }
    records = []
    for seed in range(42, 47):
        policies = {
            "bc": load_policy(ROOT / f"results/raw/bc_rl_full_v1/runs/H2_seed{seed}/bc_best.pt"),
            "rl": load_policy(ROOT / f"results/raw/bc_rl_full_v1/runs/H2_seed{seed}/rl_best.pt"),
        }
        for index, instance in enumerate(instances):
            if hashes[index]["physically_coverable"] != "1":
                continue
            traces = {}
            for name, policy in policies.items():
                trace = []
                with torch.no_grad():
                    decode(policy, instance, sample=False, seed=0, trace=trace)
                traces[name] = trace
            limit = min(len(traces["bc"]), len(traces["rl"]))
            for step in range(limit):
                for name in ("bc", "rl"):
                    item = traces[name][step]
                    customer_gain = ("customer_rank2" in item and False)
                    improved_customer = (
                        f"{seed}",
                        name,
                        str(index),
                        str(step),
                        "customer_rank2",
                    ) in improved
                    improved_route = (f"{seed}", name, str(index), str(step), "route_rank2") in improved
                    if not improved_customer and not improved_route:
                        continue
                    if improved_customer and len(item["customer_ranked_log_prob"]) > 1:
                        records.append(
                            {
                                "seed": seed,
                                "policy": name,
                                "head": "customer",
                                "chosen_log_prob": item["customer_ranked_log_prob"][0],
                                "better_log_prob": item["customer_ranked_log_prob"][1],
                            }
                        )
                    if improved_route and len(item["route_ranked_log_prob"]) > 1:
                        records.append(
                            {
                                "seed": seed,
                                "policy": name,
                                "head": "route",
                                "chosen_log_prob": item["route_ranked_log_prob"][0],
                                "better_log_prob": item["route_ranked_log_prob"][1],
                            }
                        )
                    customer_gain = customer_gain
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "better_action_logprobs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "policy", "head", "chosen_log_prob", "better_log_prob"])
        writer.writeheader()
        writer.writerows(records)
    summary = {"n": len(records)}
    for head in ("customer", "route"):
        for policy in ("bc", "rl"):
            subset = [row for row in records if row["head"] == head and row["policy"] == policy]
            if not subset:
                continue
            summary[f"{policy}_{head}_better_log_prob_mean"] = float(np.mean([row["better_log_prob"] for row in subset]))
            summary[f"{policy}_{head}_chosen_log_prob_mean"] = float(np.mean([row["chosen_log_prob"] for row in subset]))
    # Paired RL minus BC on the same improving customer actions is not identical across policies.
    # Compare each policy's probability of its own improving second action.
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"logs": summarize_logs(), "one_batch": one_batch_gradients(), "credit": credit()}
    (OUT / "gradient_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
