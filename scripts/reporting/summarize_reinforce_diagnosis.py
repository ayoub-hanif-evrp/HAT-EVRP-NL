"""Paired validation and confirmation summary for the return-to-go correction."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw" / "reinforce_diagnosis_v1"
RUNS = RAW / "runs"
CONFIRM = RAW / "confirmation" / "evaluations.csv"
TABLE = ROOT / "results" / "tables" / "final" / "reinforce_diagnosis_paired.csv"
FIGURE = ROOT / "results" / "final" / "figures" / "fig_reinforce_diagnosis.png"
T_CRIT = float(stats.t.ppf(0.975, 4))
SEEDS = [42, 43, 44, 45, 46]


def _paired(deltas: list[float], left: list[float]) -> dict:
    array = np.asarray(deltas, dtype=float)
    mean = float(array.mean())
    half = T_CRIT * float(array.std(ddof=1)) / math.sqrt(len(array))
    nonzero = [value for value in deltas if value != 0.0]
    sign_p = (
        float(stats.binomtest(sum(value < 0 for value in nonzero), len(nonzero), 0.5, alternative="two-sided").pvalue)
        if nonzero
        else 1.0
    )
    wilcoxon_p = (
        1.0
        if np.allclose(array, 0.0)
        else float(stats.wilcoxon(array, alternative="two-sided", method="exact").pvalue)
    )
    relatives = [delta / base for delta, base in zip(deltas, left)]
    return {
        "mean_h": mean,
        "ci_low_h": mean - half,
        "ci_high_h": mean + half,
        "n_negative": int(sum(value < 0 for value in deltas)),
        "mean_relative": sum(relatives) / len(relatives),
        "t_p": float(stats.ttest_1samp(array, 0.0).pvalue),
        "sign_p": sign_p,
        "wilcoxon_p": wilcoxon_p,
        "deltas_h": "|".join(f"{value:.6f}" for value in deltas),
    }


def _selected(path: Path, stage: str) -> dict:
    rows = [row for row in csv.DictReader(path.open(encoding="utf-8")) if row["stage"] == stage]
    return max(rows, key=lambda row: (float(row["feasible_rate"]), -float(row["penalized_mean_h"])))


def _validation() -> dict:
    out = {}
    for seed in SEEDS:
        loo_rows = list(csv.DictReader((RUNS / f"H2_seed{seed}_loo" / "training.csv").open(encoding="utf-8")))
        rtg_rows = list(csv.DictReader((RUNS / f"H2_seed{seed}_rtg" / "training.csv").open(encoding="utf-8")))
        bc = next(row for row in loo_rows if row["stage"] == "bc_best")
        loo = _selected(RUNS / f"H2_seed{seed}_loo" / "training.csv", "rl")
        rtg = _selected(RUNS / f"H2_seed{seed}_rtg" / "training.csv", "rl")
        out[seed] = {"bc": bc, "loo": loo, "rtg": rtg, "loo_trace": [row for row in loo_rows if row["stage"] == "rl"], "rtg_trace": [row for row in rtg_rows if row["stage"] == "rl"]}
    return out


def _confirm_tables() -> dict:
    grouped: dict[tuple, list] = {}
    for row in csv.DictReader(CONFIRM.open(encoding="utf-8")):
        grouped.setdefault((row["candidate"], int(row["seed"]), int(row["instance_index"])), []).append(row)
    by_seed: dict[str, dict] = {}
    for candidate in ("bc", "loo", "rtg"):
        by_seed[candidate] = {}
        for seed in SEEDS:
            items = []
            for index in range(40):
                rows = grouped[(candidate, seed, index)]
                greedy = next(row for row in rows if row["protocol"] == "greedy")
                samples = [row for row in rows if row["protocol"].startswith("sample")]
                penalized = [float(row["penalized_h"]) for row in [greedy, *samples]]
                items.append(
                    {
                        "coverable": greedy["physically_coverable"] == "1",
                        "feasible": greedy["feasible"] == "1",
                        "greedy": float(greedy["penalized_h"]),
                        "anchored": min(penalized),
                        "sample_mean": sum(float(row["penalized_h"]) for row in samples) / len(samples),
                        "travel": float(greedy["travel_h"]) if greedy["travel_h"] else None,
                        "service": float(greedy["service_h"]) if greedy["service_h"] else None,
                        "charging": float(greedy["charging_h"]) if greedy["charging_h"] else None,
                        "runtime": float(greedy["runtime_s"]),
                        "solvers": int(float(greedy["oracle_solver_executions"])),
                        "bundle_solvers": sum(int(float(row["oracle_solver_executions"])) for row in [greedy, *samples]),
                        "bundle_runtime": sum(float(row["runtime_s"]) for row in [greedy, *samples]),
                    }
                )
            by_seed[candidate][seed] = items
    return by_seed


def _mean_metric(items: list[dict], key: str, coverable_only: bool) -> float:
    chosen = [item for item in items if (not coverable_only or item["coverable"])]
    return sum(item[key] for item in chosen) / len(chosen)


def _add(rows: list[dict], **payload) -> None:
    stats_row = _paired(payload.pop("deltas"), payload.pop("left_values"))
    stats_row.update(payload)
    rows.append(stats_row)


def _parameter_shift() -> dict:
    shifts = {}
    for seed in SEEDS:
        bc = torch.load(RUNS / f"H2_seed{seed}_loo" / "bc_best.pt", map_location="cpu", weights_only=False)
        bc_state = bc["state_dict"] if isinstance(bc, dict) and "state_dict" in bc else bc
        for mode in ("loo", "rtg"):
            rl = torch.load(RUNS / f"H2_seed{seed}_{mode}" / "rl_best.pt", map_location="cpu", weights_only=False)
            rl_state = rl["state_dict"] if isinstance(rl, dict) and "state_dict" in rl else rl
            groups = {"route": 0.0, "customer": 0.0, "other": 0.0}
            for key, value in rl_state.items():
                delta = (value.float() - bc_state[key].float()).pow(2).sum().item()
                if "route_score" in key:
                    groups["route"] += delta
                elif "customer_" in key:
                    groups["customer"] += delta
                else:
                    groups["other"] += delta
            shifts[f"{seed}_{mode}"] = {name: math.sqrt(total) for name, total in groups.items()}
            shifts[f"{seed}_{mode}"]["total"] = math.sqrt(sum(total for total in groups.values()))
    return shifts


def main() -> None:
    validation = _validation()
    confirm = _confirm_tables()
    table = []
    for metric in ("greedy_penalized_mean_h", "anchored_penalized_mean_h", "feasible_mean_h"):
        for comparison, left_name, right_name in (
            ("loo_minus_bc", "bc", "loo"),
            ("rtg_minus_bc", "bc", "rtg"),
            ("rtg_minus_loo", "loo", "rtg"),
        ):
            left_values = [float(validation[seed][left_name][metric]) for seed in SEEDS]
            right_values = [float(validation[seed][right_name][metric]) for seed in SEEDS]
            _add(
                table,
                deltas=[right - left for left, right in zip(left_values, right_values)],
                left_values=left_values,
                split="validation",
                subset="all_instances",
                metric=metric,
                comparison=comparison,
                n_instances=40,
                left_h=sum(left_values) / len(left_values),
                right_h=sum(right_values) / len(right_values),
            )
    for subset, coverable_only in (("all_instances", False), ("coverable", True)):
        n_instances = 40 if not coverable_only else sum(item["coverable"] for item in confirm["bc"][42])
        for metric in ("greedy", "anchored", "sample_mean"):
            for comparison, left_name, right_name in (
                ("loo_minus_bc", "bc", "loo"),
                ("rtg_minus_bc", "bc", "rtg"),
                ("rtg_minus_loo", "loo", "rtg"),
            ):
                left_values = [_mean_metric(confirm[left_name][seed], metric, coverable_only) for seed in SEEDS]
                right_values = [_mean_metric(confirm[right_name][seed], metric, coverable_only) for seed in SEEDS]
                _add(
                    table,
                    deltas=[right - left for left, right in zip(left_values, right_values)],
                    left_values=left_values,
                    split="confirmation",
                    subset=subset,
                    metric=metric,
                    comparison=comparison,
                    n_instances=n_instances,
                    left_h=sum(left_values) / len(left_values),
                    right_h=sum(right_values) / len(right_values),
                )
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    logs = {}
    for mode in ("loo", "rtg"):
        norms, entropies, solvers, early, late = [], [], [], [], []
        for seed in SEEDS:
            rows = list(csv.DictReader((RUNS / f"H2_seed{seed}_{mode}" / "reward.csv").open(encoding="utf-8")))
            grad = [float(row["grad_norm"]) for row in rows]
            ent = [float(row["entropy"]) for row in rows]
            norms.extend(grad)
            entropies.append(ent)
            solvers.append(sum(int(float(row["oracle_calls"])) for row in rows))
            early.append(ent[0])
            late.append(ent[-1])
        logs[mode] = {
            "updates": len(norms),
            "zero_gradient_updates": int(sum(value == 0.0 for value in norms)),
            "mean_grad_norm": sum(norms) / len(norms),
            "entropy_first": sum(early) / len(early),
            "entropy_last": sum(late) / len(late),
            "solver_executions": sum(solvers),
        }
    credit = list(csv.DictReader((RAW / "better_action_logprobs.csv").open(encoding="utf-8")))
    credit_summary = {}
    for policy in ("bc", "rl"):
        for head in ("customer", "route"):
            chosen = [float(row["chosen_log_prob"]) for row in credit if row["policy"] == policy and row["head"] == head]
            better = [float(row["better_log_prob"]) for row in credit if row["policy"] == policy and row["head"] == head]
            credit_summary[f"{policy}_{head}"] = {
                "n": len(chosen),
                "chosen_mean": sum(chosen) / len(chosen),
                "better_mean": sum(better) / len(better),
                "better_below_exp_neg2": sum(value < -2.0 for value in better) / len(better),
            }
    elapsed = {}
    for seed in SEEDS:
        for mode in ("loo", "rtg"):
            rows = list(csv.DictReader((RUNS / f"H2_seed{seed}_{mode}" / "training.csv").open(encoding="utf-8")))
            elapsed[f"{seed}_{mode}"] = max(float(row["elapsed_s"]) for row in rows if row["elapsed_s"])
    feasibility = {}
    for candidate in ("bc", "loo", "rtg"):
        items = [item for seed in SEEDS for item in confirm[candidate][seed]]
        coverable = [item for item in items if item["coverable"]]
        feasibility[candidate] = {
            "greedy_feasible_rate_all": sum(item["feasible"] for item in items) / len(items),
            "greedy_feasible_rate_coverable": sum(item["feasible"] for item in coverable) / len(coverable),
            "mean_greedy_runtime_s": sum(item["runtime"] for item in items) / len(items),
            "mean_greedy_solvers": sum(item["solvers"] for item in items) / len(items),
            "mean_bundle_runtime_s": sum(item["bundle_runtime"] for item in items) / len(items),
            "mean_bundle_solvers": sum(item["bundle_solvers"] for item in items) / len(items),
        }
        for part in ("travel", "service", "charging"):
            values = [item[part] for item in coverable if item[part] is not None]
            feasibility[candidate][f"coverable_{part}_h"] = sum(values) / len(values)
    summary = {
        "validation_traces": {
            str(seed): {
                mode: [
                    {
                        "step": int(row["step"]),
                        "greedy_h": float(row["greedy_penalized_mean_h"]),
                        "anchored_h": float(row["anchored_penalized_mean_h"]),
                        "feasible_mean_h": float(row["feasible_mean_h"]),
                    }
                    for row in validation[seed][f"{mode}_trace"]
                ]
                for mode in ("loo", "rtg")
            }
            for seed in SEEDS
        },
        "logs": logs,
        "credit": credit_summary,
        "parameter_l2_from_bc": _parameter_shift(),
        "elapsed_s": elapsed,
        "confirmation_operations": feasibility,
        "coverable_confirmation_instances": sum(item["coverable"] for item in confirm["bc"][42]),
    }
    (RAW / "confirmation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _figure(table)
    print({"table": str(TABLE), "rows": len(table), "coverable": summary["coverable_confirmation_instances"]})


def _figure(table: list[dict]) -> None:
    wanted = [
        ("validation", "all_instances", "greedy_penalized_mean_h", "rtg_minus_bc", "Validation\nRTG − BC"),
        ("validation", "all_instances", "greedy_penalized_mean_h", "loo_minus_bc", "Validation\nLOO − BC"),
        ("confirmation", "all_instances", "greedy", "rtg_minus_bc", "Confirmation\nRTG − BC"),
        ("confirmation", "coverable", "greedy", "rtg_minus_bc", "Coverable\nRTG − BC"),
        ("confirmation", "all_instances", "anchored", "rtg_minus_bc", "Greedy+3\nRTG − BC"),
        ("confirmation", "all_instances", "greedy", "rtg_minus_loo", "Confirmation\nRTG − LOO"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=200)
    for index, (split, subset, metric, comparison, label) in enumerate(wanted):
        row = next(
            item
            for item in table
            if item["split"] == split and item["subset"] == subset and item["metric"] == metric and item["comparison"] == comparison
        )
        mean = float(row["mean_h"])
        low = float(row["ci_low_h"])
        high = float(row["ci_high_h"])
        ax.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="o", color="#1f4e79", capsize=4, markersize=6)
        ax.set_xticks(range(len(wanted)))
        ax.set_xticklabels([item[-1] for item in wanted])
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_ylabel("Paired penalized duration (h)")
    ax.set_title("Returned routes after matched BC, original REINFORCE, and return-to-go")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)


if __name__ == "__main__":
    main()
