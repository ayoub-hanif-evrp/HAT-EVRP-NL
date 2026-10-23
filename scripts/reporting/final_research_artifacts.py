"""Tables and figures for the locked BC versus RL construction study.

Reads saved audit rows, the original official greedy evaluations, and, when
present, the locked stochastic official evaluation. Does not train or decode.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presentation_names import METHOD_COLORS, method_label

AUDIT = ROOT / "results" / "raw" / "inference_audit_v1" / "checkpoint_summary.csv"
P0_DIR = ROOT / "results" / "raw" / "bc_rl_full_v1" / "eval"
SAMPLES = ROOT / "results" / "raw" / "bc_rl_full_v1" / "eval_stochastic" / "official_samples.csv"
RUNS = ROOT / "results" / "raw" / "bc_rl_full_v1" / "runs"
CHARGING = ROOT / "results" / "raw" / "nonlinear_charging" / "charging_study_summary.csv"
BKS = ROOT / "results" / "literature_bks_froger2021.csv"
OUT_FIG = ROOT / "results" / "final" / "figures"
OUT_TAB = ROOT / "results" / "tables" / "final"
T_CRIT = 2.7764451051977987
MODELS = ("H1", "H2", "H3", "H4")
PROTOCOLS = (
    ("greedy_h", "Greedy"),
    ("sampled_mean_h", "Mean of 4 samples"),
    ("best_of_4_h", "Best of 4"),
)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _ci(deltas: list[float]) -> tuple[float, float, float, float]:
    mean = _mean(deltas)
    var = sum((delta - mean) ** 2 for delta in deltas) / (len(deltas) - 1)
    sd = math.sqrt(var)
    half = T_CRIT * math.sqrt(var / len(deltas))
    return mean, sd, mean - half, mean + half


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validation_pairs() -> list[dict]:
    rows = _rows(AUDIT)
    bc = {
        (row["model"], int(float(row["seed"]))): row
        for row in rows
        if row["experiment"] == "original" and row["role"] == "bc"
    }
    paired = []
    for row in rows:
        if row["role"] != "rl":
            continue
        base = bc[(row["model"], int(float(row["seed"])))]
        for key, _label in PROTOCOLS:
            bc_h = float(base[key])
            rl_h = float(row[key])
            paired.append(
                {
                    "experiment": row["experiment"],
                    "model": row["model"],
                    "seed": int(float(row["seed"])),
                    "protocol": key,
                    "bc_h": round(bc_h, 4),
                    "rl_h": round(rl_h, 4),
                    "delta_h": round(rl_h - bc_h, 4),
                    "bc_feasible": base["greedy_feasible"] if key == "greedy_h" else base[
                        "sampled_feasible" if key == "sampled_mean_h" else "best_of_4_feasible"
                    ],
                    "rl_feasible": row["greedy_feasible"] if key == "greedy_h" else row[
                        "sampled_feasible" if key == "sampled_mean_h" else "best_of_4_feasible"
                    ],
                    "bc_runtime_s": round(float(base["greedy_runtime_s" if key == "greedy_h" else "sample_runtime_s"]), 3),
                    "rl_runtime_s": round(float(row["greedy_runtime_s" if key == "greedy_h" else "sample_runtime_s"]), 3),
                    "bc_oracle_calls": base["greedy_oracle_calls" if key == "greedy_h" else "sample_oracle_calls"],
                    "rl_oracle_calls": row["greedy_oracle_calls" if key == "greedy_h" else "sample_oracle_calls"],
                }
            )
    return paired


def validation_summary(paired: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in paired:
        grouped[(row["experiment"], row["model"], row["protocol"])].append(row)
    out = []
    for (experiment, model, protocol), rows in sorted(grouped.items()):
        deltas = [float(row["delta_h"]) for row in rows]
        mean, sd, low, high = _ci(deltas)
        bc = _mean([float(row["bc_h"]) for row in rows])
        rl = _mean([float(row["rl_h"]) for row in rows])
        out.append(
            {
                "split": "synthetic_validation",
                "experiment": experiment,
                "model": method_label(model),
                "model_key": model,
                "protocol": protocol,
                "n_seeds": len(rows),
                "bc_h": round(bc, 4),
                "rl_h": round(rl, 4),
                "delta_h": round(mean, 4),
                "relative_pct": round(100.0 * mean / bc, 3),
                "seeds_better": sum(delta < -1e-4 for delta in deltas),
                "seed_sd_h": round(sd, 4),
                "ci_low_h": round(low, 4),
                "ci_high_h": round(high, 4),
                "bc_feasible": round(_mean([float(row["bc_feasible"]) for row in rows]), 4),
                "rl_feasible": round(_mean([float(row["rl_feasible"]) for row in rows]), 4),
            }
        )
    return out


def p0_instance_rows() -> list[dict]:
    bks = {row["instance"]: float(row["bks_h"]) for row in _rows(BKS)}
    collected = []
    for path in sorted(P0_DIR.glob("H*_seed*.csv")):
        model, _seed_token = path.stem.split("_seed")
        seed = int(_seed_token)
        for row in _rows(path):
            if row["protocol"] != "P0":
                continue
            objective = float(row["objective_h"]) if row["objective_h"] else math.nan
            lit = float(row["objective_lit_h"]) if row["objective_lit_h"] else math.nan
            reference = bks.get(row["instance"])
            gap = (lit - reference) / reference * 100.0 if reference and math.isfinite(lit) else math.nan
            collected.append(
                {
                    "model": model,
                    "seed": seed,
                    "role": "bc" if row["role"] == "bc_best" else "rl",
                    "size": int(row["size"]),
                    "instance": row["instance"],
                    "feasible": int(row["feasible"]),
                    "objective_h": objective,
                    "objective_lit_h": lit,
                    "gap_pct": gap,
                    "runtime_s": float(row["runtime_s"] or 0),
                    "oracle_calls": float(row["oracle_calls"] or 0),
                }
            )
    return collected


def _seed_means(rows: list[dict], value: str) -> dict[tuple, float]:
    grouped = defaultdict(list)
    for row in rows:
        if math.isfinite(float(row[value])):
            grouped[(row["model"], row["seed"], row["role"], row["size"])].append(float(row[value]))
    return {key: _mean(values) for key, values in grouped.items()}


def greedy_official_summary(instance_rows: list[dict]) -> list[dict]:
    means = _seed_means(instance_rows, "objective_h")
    gaps = _seed_means(instance_rows, "gap_pct")
    runtime = _seed_means(instance_rows, "runtime_s")
    calls = defaultdict(list)
    feasible = defaultdict(list)
    for row in instance_rows:
        key = (row["model"], row["seed"], row["role"], row["size"])
        calls[key].append(row["oracle_calls"])
        feasible[key].append(row["feasible"])
    out = []
    for model in MODELS:
        sizes = sorted({row["size"] for row in instance_rows if row["model"] == model})
        for size in sizes:
            bc_seeds = []
            rl_seeds = []
            for seed in (42, 43, 44, 45, 46):
                bc_seeds.append(means[(model, seed, "bc", size)])
                rl_seeds.append(means[(model, seed, "rl", size)])
            deltas = [rl - bc for bc, rl in zip(bc_seeds, rl_seeds)]
            mean, sd, low, high = _ci(deltas)
            bc = _mean(bc_seeds)
            out.append(
                {
                    "split": "official_greedy",
                    "model": method_label(model),
                    "model_key": model,
                    "size": size,
                    "bc_h": round(bc, 4),
                    "rl_h": round(_mean(rl_seeds), 4),
                    "delta_h": round(mean, 4),
                    "relative_pct": round(100.0 * mean / bc, 3),
                    "seeds_better": sum(delta < -1e-4 for delta in deltas),
                    "seed_sd_h": round(sd, 4),
                    "ci_low_h": round(low, 4),
                    "ci_high_h": round(high, 4),
                    "bc_feasible": round(_mean([_mean(feasible[(model, seed, "bc", size)]) for seed in (42, 43, 44, 45, 46)]), 4),
                    "rl_feasible": round(_mean([_mean(feasible[(model, seed, "rl", size)]) for seed in (42, 43, 44, 45, 46)]), 4),
                    "bc_runtime_s": round(_mean([runtime[(model, seed, "bc", size)] for seed in (42, 43, 44, 45, 46)]) * 20, 3),
                    "rl_runtime_s": round(_mean([runtime[(model, seed, "rl", size)] for seed in (42, 43, 44, 45, 46)]) * 20, 3),
                    "bc_oracle_calls": round(_mean([sum(calls[(model, seed, "bc", size)]) for seed in (42, 43, 44, 45, 46)]), 1),
                    "rl_oracle_calls": round(_mean([sum(calls[(model, seed, "rl", size)]) for seed in (42, 43, 44, 45, 46)]), 1),
                    "bc_gap_pct": "" if (model, 42, "bc", size) not in gaps else round(_mean([gaps[(model, seed, "bc", size)] for seed in (42, 43, 44, 45, 46)]), 3),
                    "rl_gap_pct": "" if (model, 42, "rl", size) not in gaps else round(_mean([gaps[(model, seed, "rl", size)] for seed in (42, 43, 44, 45, 46)]), 3),
                }
            )
    return out


def stochastic_official_summary() -> list[dict]:
    if not SAMPLES.exists():
        return []
    raw = _rows(SAMPLES)
    by_instance = defaultdict(list)
    for row in raw:
        if row["protocol"] != "sample":
            continue
        by_instance[(row["model"], int(row["seed"]), row["role"], int(row["size"]), row["instance"])].append(row)
    seed_values = defaultdict(lambda: {"mean": [], "best": [], "feasible": [], "runtime": [], "calls": []})
    for (model, seed, role, size, _instance), trials in by_instance.items():
        feasible = [float(trial["objective_h"]) for trial in trials if int(trial["feasible"]) == 1 and trial["objective_h"]]
        if len(feasible) != len(trials):
            raise RuntimeError(f"infeasible official sample at {model} seed {seed} n={size}")
        seed_values[(model, seed, role, size)]["mean"].append(_mean(feasible))
        seed_values[(model, seed, role, size)]["best"].append(min(feasible))
        seed_values[(model, seed, role, size)]["feasible"].append(1.0)
        seed_values[(model, seed, role, size)]["runtime"].append(sum(float(trial["runtime_s"]) for trial in trials))
        seed_values[(model, seed, role, size)]["calls"].append(sum(float(trial["oracle_calls"]) for trial in trials))
    out = []
    sizes = sorted({key[3] for key in seed_values})
    for model in MODELS:
        for size in sizes:
            complete = all(
                len(seed_values[(model, seed, role, size)]["mean"]) == 20
                for seed in (42, 43, 44, 45, 46)
                for role in ("bc", "rl")
            )
            if not complete:
                continue
            for protocol, field in (("sampled_mean_h", "mean"), ("best_of_4_h", "best")):
                deltas = []
                bc_values = []
                rl_values = []
                for seed in (42, 43, 44, 45, 46):
                    bc_h = _mean(seed_values[(model, seed, "bc", size)][field])
                    rl_h = _mean(seed_values[(model, seed, "rl", size)][field])
                    bc_values.append(bc_h)
                    rl_values.append(rl_h)
                    deltas.append(rl_h - bc_h)
                mean, sd, low, high = _ci(deltas)
                bc = _mean(bc_values)
                out.append(
                    {
                        "split": "official_sampled",
                        "model": method_label(model),
                        "model_key": model,
                        "size": size,
                        "protocol": protocol,
                        "bc_h": round(bc, 4),
                        "rl_h": round(_mean(rl_values), 4),
                        "delta_h": round(mean, 4),
                        "relative_pct": round(100.0 * mean / bc, 3),
                        "seeds_better": sum(delta < -1e-4 for delta in deltas),
                        "seed_sd_h": round(sd, 4),
                        "ci_low_h": round(low, 4),
                        "ci_high_h": round(high, 4),
                        "bc_runtime_s": round(
                            _mean([sum(seed_values[(model, seed, "bc", size)]["runtime"]) for seed in (42, 43, 44, 45, 46)]),
                            3,
                        ),
                        "rl_runtime_s": round(
                            _mean([sum(seed_values[(model, seed, "rl", size)]["runtime"]) for seed in (42, 43, 44, 45, 46)]),
                            3,
                        ),
                        "bc_oracle_calls": round(
                            _mean([sum(seed_values[(model, seed, "bc", size)]["calls"]) for seed in (42, 43, 44, 45, 46)]),
                            1,
                        ),
                        "rl_oracle_calls": round(
                            _mean([sum(seed_values[(model, seed, "rl", size)]["calls"]) for seed in (42, 43, 44, 45, 46)]),
                            1,
                        ),
                        "feasible_rate": 1.0,
                    }
                )
    return out


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def fig_validation_protocols(summary: list[dict]) -> None:
    _style()
    rows = [row for row in summary if row["experiment"] == "original"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    protocols = [key for key, _label in PROTOCOLS]
    labels = [label for _key, label in PROTOCOLS]
    width = 0.18
    x = np.arange(len(protocols))
    for offset, model in enumerate(MODELS):
        chosen = [next(row for row in rows if row["model_key"] == model and row["protocol"] == protocol) for protocol in protocols]
        y = [row["delta_h"] for row in chosen]
        yerr = np.array([[row["delta_h"] - row["ci_low_h"], row["ci_high_h"] - row["delta_h"]] for row in chosen]).T
        ax.bar(x + (offset - 1.5) * width, y, width, yerr=yerr, color=METHOD_COLORS[model], label=method_label(model, short=True), capsize=2)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RL − BC duration (h)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_validation_protocols.png")
    plt.close(fig)


def fig_seed_effects(paired: list[dict]) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, model in zip(axes, ("H2", "H4")):
        for index, (protocol, label) in enumerate(PROTOCOLS):
            deltas = [
                row["delta_h"]
                for row in paired
                if row["experiment"] == "original" and row["model"] == model and row["protocol"] == protocol
            ]
            ax.scatter(np.full(len(deltas), index), deltas, color=METHOD_COLORS[model], s=28, zorder=3)
            ax.scatter([index], [_mean(deltas)], marker="D", color="#222222", s=26, zorder=4)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(range(len(PROTOCOLS)), [label for _key, label in PROTOCOLS], rotation=15, ha="right")
        ax.set_title(method_label(model, short=True))
    axes[0].set_ylabel("RL − BC duration (h)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_validation_seed_effects.png")
    plt.close(fig)


def fig_learning_curves() -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, stage, title in (
        (axes[0], "bc", "Behavior cloning"),
        (axes[1], "rl", "REINFORCE"),
    ):
        for model in ("H2", "H4"):
            by_step = defaultdict(list)
            for seed in (42, 43, 44, 45, 46):
                path = RUNS / f"{model}_seed{seed}" / "training.csv"
                for row in _rows(path):
                    if row["stage"] == stage:
                        by_step[int(row["step"])].append(float(row["penalized_mean_h"]))
            steps = sorted(by_step)
            means = [_mean(by_step[step]) for step in steps]
            ax.plot(steps, means, color=METHOD_COLORS[model], label=method_label(model, short=True))
        ax.set_xlabel("Update")
        ax.set_title(title)
    axes[0].set_ylabel("Validation duration (h)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_validation_learning_curves.png")
    plt.close(fig)


def fig_generalization(official: list[dict]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for model, style in (("H2", "-"), ("H4", "--")):
        rows = [row for row in official if row["model_key"] == model]
        sizes = [row["size"] for row in rows]
        ax.plot(sizes, [row["bc_h"] for row in rows], style, color=METHOD_COLORS[model], marker="o", label=f"{method_label(model, short=True)} BC")
        ax.plot(sizes, [row["rl_h"] for row in rows], style, color=METHOD_COLORS[model], marker="o", alpha=0.45, label=f"{method_label(model, short=True)} RL")
    ax.set_xticks([10, 20, 40, 80])
    ax.set_xlabel("Customers")
    ax.set_ylabel("Greedy duration (h)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_greedy_generalization.png")
    plt.close(fig)


def fig_nonlinear() -> None:
    _style()
    rows = [
        row
        for row in _rows(CHARGING)
        if row["experiment"] == "fixed_route" and row["method"].startswith("plan_")
    ]
    labels = {"LINEAR-EARLY-RATE": "Linear early rate", "LINEAR-SECANT": "Linear secant", "TRUE-NL": "Exact nonlinear"}
    colors = {"LINEAR-EARLY-RATE": "#E69F00", "LINEAR-SECANT": "#D55E00", "TRUE-NL": "#0072B2"}
    models = ["LINEAR-EARLY-RATE", "LINEAR-SECANT", "TRUE-NL"]
    sizes = sorted({int(row["size"]) for row in rows})
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.25
    x = np.arange(len(sizes))
    for offset, model in enumerate(models):
        chosen = []
        for size in sizes:
            row = next(item for item in rows if int(item["size"]) == size and item["charging_model"] == model)
            chosen.append(100.0 * int(row["feasible_under_true_physics"]) / int(row["n_instances"]))
        ax.bar(x + (offset - 1) * width, chosen, width, color=colors[model], label=labels[model])
    ax.set_xticks(x, [str(size) for size in sizes])
    ax.set_xlabel("Customers")
    ax.set_ylabel("Feasible under true charging (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_nonlinear_feasibility.png")
    plt.close(fig)


def fig_official_sampled(rows: list[dict]) -> None:
    if not rows:
        return
    _style()
    protocols = ["sampled_mean_h", "best_of_4_h"]
    labels = ["Mean of 4 samples", "Best of 4"]
    sizes = sorted({row["size"] for row in rows})
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.5 * len(sizes), 3.8), sharey=True)
    if len(sizes) == 1:
        axes = [axes]
    width = 0.18
    for ax, size in zip(axes, sizes):
        x = np.arange(len(protocols))
        for offset, model in enumerate(MODELS):
            chosen = [next(row for row in rows if row["model_key"] == model and row["size"] == size and row["protocol"] == protocol) for protocol in protocols]
            y = [row["delta_h"] for row in chosen]
            yerr = np.array([[row["delta_h"] - row["ci_low_h"], row["ci_high_h"] - row["delta_h"]] for row in chosen]).T
            ax.bar(x + (offset - 1.5) * width, y, width, yerr=yerr, color=METHOD_COLORS[model], label=method_label(model, short=True), capsize=2)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_title(f"n = {size}")
    axes[0].set_ylabel("RL − BC duration (h)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=4, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_FIG / "fig_official_sampled.png")
    plt.close(fig)


def fig_rt_deepsets(official: list[dict], sampled: list[dict]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    n10 = [row for row in official if row["size"] == 10]
    x = np.arange(len(MODELS))
    bc = [next(row["bc_h"] for row in n10 if row["model_key"] == model) for model in MODELS]
    rl = [next(row["rl_h"] for row in n10 if row["model_key"] == model) for model in MODELS]
    ax.bar(x - 0.16, bc, 0.32, color="#0072B2", label="BC greedy")
    ax.bar(x + 0.16, rl, 0.32, color="#D55E00", label="RL greedy")
    ax.set_xticks(x, [method_label(model, short=True) for model in MODELS])
    ax.set_ylabel("Official n=10 duration (h)")
    ax.legend(frameon=False)
    if sampled:
        pass
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_architecture_n10_greedy.png")
    plt.close(fig)


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    paired = validation_pairs()
    summary = validation_summary(paired)
    official = greedy_official_summary(p0_instance_rows())
    sampled = stochastic_official_summary()
    _write(OUT_TAB / "validation_seed_pairs.csv", paired)
    _write(OUT_TAB / "validation_summary.csv", summary)
    _write(OUT_TAB / "official_greedy_summary.csv", official)
    if sampled:
        _write(OUT_TAB / "official_sampled_summary.csv", sampled)
    fig_validation_protocols(summary)
    fig_seed_effects(paired)
    fig_learning_curves()
    fig_generalization(official)
    fig_nonlinear()
    fig_official_sampled(sampled)
    fig_rt_deepsets(official, sampled)
    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "validation_checkpoints": 60,
        "official_sampled_present": bool(sampled),
        "figures": sorted(path.name for path in OUT_FIG.glob("*.png")),
        "tables": sorted(path.name for path in OUT_TAB.glob("*.csv")),
    }
    (OUT_TAB / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
