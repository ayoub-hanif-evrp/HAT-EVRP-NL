"""Main-paper neural figures from the completed BC→RL matrix.

Sources are results/raw/bc_rl_full_v1 only. Cycle-3 logs are not read.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presentation_names import method_color, method_label  # noqa: E402
from reporting_paths import FIGURES, LITERATURE_BKS, RAW, SUPP_FIGURES  # noqa: E402

DPI = 300
FULL = RAW / "bc_rl_full_v1"
MODELS = ("H1", "H2", "H3", "H4")
SEEDS = (42, 43, 44, 45, 46)
PROTOCOLS = (("P0", "Construction"), ("P1", "One-shot LS"), ("P2", "Hybrid insert"))


def fig_save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("wrote", path)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(xs) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _stdev(xs) -> float:
    if len(xs) < 2:
        return 0.0
    return float(np.std(xs, ddof=1))


_EVAL: dict[tuple[str, int], list[dict]] = {}


def eval_rows(model: str, seed: int) -> list[dict]:
    key = (model, seed)
    if key not in _EVAL:
        _EVAL[key] = _read(FULL / "eval" / f"{model}_seed{seed}.csv")
    return _EVAL[key]


def subset(model: str, seed: int, role: str, size: int, protocol: str) -> list[dict]:
    return [
        row
        for row in eval_rows(model, seed)
        if row["role"] == role and int(row["size"]) == size and row["protocol"] == protocol
    ]


def seed_quality(model: str, seed: int, role: str, size: int, protocol: str) -> dict:
    rows = subset(model, seed, role, size, protocol)
    ok = [float(row["objective_h"]) for row in rows if row["feasible"] == "1" and row["objective_h"]]
    return {
        "mean_h": _mean(ok),
        "n": len(rows),
        "n_feasible": len(ok),
        "runtime_s": _mean([float(row["runtime_s"]) for row in rows]) if rows else float("nan"),
        "oracle_calls": _mean([float(row["oracle_calls"]) for row in rows]) if rows else float("nan"),
    }


def model_seed_means(role: str, size: int, protocol: str) -> dict[str, list[float]]:
    return {
        model: [seed_quality(model, seed, role, size, protocol)["mean_h"] for seed in SEEDS]
        for model in MODELS
    }


def coverage(size: int, protocol: str = "P0", role: str = "bc_best") -> int:
    rows = subset("H2", 42, role, size, protocol)
    return int(rows[0]["n_available"]) if rows else 0


def make_fig12() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), sharey=True)
    for ax, role, title in (
        (axes[0], "bc_best", "Behavior cloning"),
        (axes[1], "rl_best", "REINFORCE"),
    ):
        for i, model in enumerate(MODELS):
            vals = model_seed_means(role, 10, "P0")[model]
            color = method_color(model)
            ax.scatter(vals, np.full(len(vals), i), s=28, color=color, zorder=3)
            ax.scatter([_mean(vals)], [i], marker="D", s=42, color=color, edgecolor="k", zorder=4)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([method_label(m, short=True) for m in MODELS])
        ax.set_xlabel("Mean full duration (h)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.25)
    axes[0].invert_yaxis()
    fig.suptitle("Official n=10 construction (P0), five seeds; diamond = seed-mean", fontsize=10)
    fig_save(FIGURES / "fig12_neural_architecture_ablation.png")


def make_fig13() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bc = model_seed_means("bc_best", 10, "P0")
    rl = model_seed_means("rl_best", 10, "P0")
    for i, model in enumerate(MODELS):
        deltas = [r - b for b, r in zip(bc[model], rl[model])]
        color = method_color(model)
        ax.scatter(np.full(len(deltas), i) + (np.linspace(-0.12, 0.12, len(deltas))), deltas, s=28, color=color, zorder=3)
        ax.scatter([i], [_mean(deltas)], marker="D", s=46, color=color, edgecolor="k", zorder=4)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([method_label(m, wrap=True) for m in MODELS])
    ax.set_ylabel("RL − BC mean full duration (h)\nnegative = RL better")
    ax.set_title("Paired official n=10 construction change, one point per seed")
    ax.grid(True, axis="y", alpha=0.25)
    fig_save(FIGURES / "fig13_history_deepsets_paired_effect.png")


def _bks() -> dict[str, float]:
    return {row["instance"]: float(row["bks_h"]) for row in _read(LITERATURE_BKS)}


def gap_seed_means(role: str, size: int, protocol: str) -> dict[str, list[float]]:
    bks = _bks()
    out = {model: [] for model in MODELS}
    for model in MODELS:
        for seed in SEEDS:
            gaps = []
            for row in subset(model, seed, role, size, protocol):
                if row["feasible"] != "1" or not row["objective_lit_h"] or row["instance"] not in bks:
                    continue
                lit = float(row["objective_lit_h"])
                base = bks[row["instance"]]
                gaps.append(100.0 * (lit - base) / base)
            if gaps:
                out[model].append(_mean(gaps))
    return out


def make_fig14() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.6), sharey=True)
    for ax, (protocol, name) in zip(axes, PROTOCOLS):
        for i, model in enumerate(MODELS):
            for role, marker in (("bc_best", "o"), ("rl_best", "s")):
                vals = gap_seed_means(role, 10, protocol)[model]
                ax.scatter(
                    np.full(len(vals), i) + (-0.12 if role == "bc_best" else 0.12),
                    vals,
                    marker=marker,
                    s=22,
                    color=method_color(model),
                    zorder=3,
                )
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([method_label(m, short=True) for m in MODELS], rotation=25, ha="right")
        ax.set_title(name)
        ax.grid(True, axis="y", alpha=0.25)
    low, high = axes[0].get_ylim()
    axes[0].set_ylim(0, high if high > low else 1)
    axes[0].set_ylabel("Mean literature gap (%)")
    fig.suptitle("Official n=10 gap to Froger Table 13; circles = BC, squares = RL", fontsize=10)
    fig_save(FIGURES / "fig14_literature_bks_gap.png")


def make_fig15() -> None:
    sizes = (10, 20, 40, 80)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.5))
    for model in MODELS:
        color = method_color(model)
        for role, ls in (("bc_best", "-"), ("rl_best", "--")):
            ys = [_mean(model_seed_means(role, size, "P0")[model]) for size in sizes]
            axes[0].plot(sizes, ys, color=color, ls=ls, lw=1.6, marker="o", ms=3.5)
        for role, marker in (("bc_best", "o"), ("rl_best", "s")):
            rates = []
            for size in sizes:
                cells = [seed_quality(model, seed, role, size, "P0") for seed in SEEDS]
                rates.append(100.0 * sum(c["n_feasible"] for c in cells) / sum(c["n"] for c in cells))
            shift = (MODELS.index(model) - 1.5) * 1.5
            axes[2].plot([s + shift for s in sizes], rates, color=color, marker=marker, ms=3.5, lw=1.0)
    for role, marker, label in (("bc_best", "o", "BC"), ("rl_best", "s", "RL")):
        deltas = []
        for size in sizes:
            h2 = _mean(model_seed_means(role, size, "P0")["H2"])
            h4 = _mean(model_seed_means(role, size, "P0")["H4"])
            deltas.append(h4 - h2)
        axes[1].plot(sizes, deltas, marker=marker, color="#333333", lw=1.5, label=label)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].legend(frameon=True, fontsize=7)
    axes[0].set_ylabel("Mean full duration (h)")
    axes[1].set_ylabel("(RT+DeepSets) − RT (h)")
    axes[2].set_ylabel("Feasible instances (%)")
    axes[2].set_ylim(-5, 105)
    for ax, title in zip(axes, ("P0 duration", "Architecture difference", "Feasibility")):
        ax.set_xlabel("Customers")
        ax.set_xticks(sizes)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color=method_color(m), lw=2, label=method_label(m, short=True)) for m in MODELS
    ]
    handles.append(plt.Line2D([0], [0], color="#333", ls="--", label="RL (dashed in A)"))
    axes[0].legend(handles=handles, fontsize=6.5, frameon=True, loc="upper left")
    fig.suptitle("Size transfer of the n=10 policy under construction only", fontsize=10)
    fig_save(FIGURES / "fig15_generalization_scalability.png")


def make_fig16() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), sharex=True, sharey=True)
    ylabels = [name for _, name in PROTOCOLS]
    observed = []
    for ax, role, title in ((axes[0], "bc_best", "Behavior cloning"), (axes[1], "rl_best", "REINFORCE")):
        for i, (protocol, _) in enumerate(PROTOCOLS):
            for j, model in enumerate(MODELS):
                val = _mean(model_seed_means(role, 10, protocol)[model])
                observed.append(val)
                y = i + (j - 1.5) * 0.18
                ax.scatter([val], [y], s=32, color=method_color(model), zorder=3)
        ax.set_yticks(range(len(PROTOCOLS)))
        ax.set_yticklabels(ylabels)
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.25)
        ax.invert_yaxis()
    lo, hi = min(observed), max(observed)
    pad = max(0.15, 0.08 * (hi - lo))
    for ax in axes:
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel("Mean full duration (h)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=method_color(m), markersize=6, label=method_label(m, short=True))
        for m in MODELS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Official n=10 protocols as categorical alternatives", fontsize=10, y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = FIGURES / "fig16_search_improvement_stages.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def make_fig17() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    markers = {"P0": "o", "P1": "s", "P2": "D"}
    for model in MODELS:
        for protocol, _name in PROTOCOLS:
            for role in ("bc_best", "rl_best"):
                cells = [seed_quality(model, seed, role, 10, protocol) for seed in SEEDS]
                x = _mean([c["oracle_calls"] for c in cells])
                y = _mean([c["mean_h"] for c in cells])
                ax.scatter(
                    [x],
                    [y],
                    marker=markers[protocol],
                    s=36,
                    color=method_color(model),
                    edgecolor="k" if role == "rl_best" else "none",
                    linewidths=0.6,
                    zorder=3,
                )
    ax.set_xlabel("Mean frvcpy oracle calls per instance")
    ax.set_ylabel("Mean full duration (h)")
    ax.set_title("Official n=10 quality and oracle calls")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=method_color(m), label=method_label(m, short=True))
        for m in MODELS
    ]
    handles += [
        plt.Line2D([0], [0], marker="o", color="#333", ls="none", label="Construction"),
        plt.Line2D([0], [0], marker="s", color="#333", ls="none", label="One-shot LS"),
        plt.Line2D([0], [0], marker="D", color="#333", ls="none", label="Hybrid insert"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="k", label="RL edge"),
    ]
    ax.legend(handles=handles, fontsize=6.5, frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig_save(FIGURES / "fig17_quality_vs_compute.png")


def make_fig18() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.4), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODELS):
        color = method_color(model)
        for seed in SEEDS:
            rows = [
                row
                for row in _read(FULL / "runs" / f"{model}_seed{seed}" / "training.csv")
                if row["stage"] == "bc"
            ]
            steps = [int(row["step"]) for row in rows]
            vals = [float(row["validation_objective"]) for row in rows]
            ax.plot(steps, vals, color=color, alpha=0.28, lw=0.9)
            best_i = int(np.argmin(vals))
            ax.scatter([steps[best_i]], [vals[best_i]], marker="*", s=36, color=color, zorder=4)
        ax.set_title(method_label(model, short=True))
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("BC evaluation step")
    axes[1, 1].set_xlabel("BC evaluation step")
    axes[0, 0].set_ylabel("Validation full duration (h)")
    axes[1, 0].set_ylabel("Validation full duration (h)")
    fig.suptitle("Behavior-cloning validation on the shared synthetic set; star = best logged point", fontsize=10)
    fig_save(FIGURES / "fig18_behavior_cloning_learning_curves.png")


def make_fig19() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 3.5), sharey=True)
    for ax, model in zip(axes, MODELS):
        bc = model_seed_means("bc_best", 10, "P0")[model]
        rl = model_seed_means("rl_best", 10, "P0")[model]
        color = method_color(model)
        for b, r in zip(bc, rl):
            ax.plot([0, 1], [b, r], color=color, lw=1.2, marker="o", ms=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["BC", "RL"])
        ax.set_title(method_label(model, short=True))
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Official n=10 mean full duration (h)")
    fig.suptitle("Construction-only checkpoint pair, one line per seed", fontsize=10)
    fig_save(FIGURES / "fig19_bc_vs_rl_checkpoint_quality.png")


def make_fig20() -> None:
    window = 5
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODELS):
        color = method_color(model)
        smoothed = []
        for seed in SEEDS:
            rows = _read(FULL / "runs" / f"{model}_seed{seed}" / "reward.csv")
            steps = [int(row["step"]) for row in rows]
            reward = [float(row["mean_reward"]) for row in rows]
            ax.plot(steps, reward, color=color, alpha=0.2, lw=0.7)
            smooth = [float(np.median(reward[max(0, i - window + 1) : i + 1])) for i in range(len(reward))]
            smoothed.append(smooth)
        n = min(len(s) for s in smoothed)
        ax.plot(steps[:n], np.mean([s[:n] for s in smoothed], axis=0), color=color, lw=2.0)
        ax.set_title(method_label(model, short=True))
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("RL step")
    axes[1, 1].set_xlabel("RL step")
    axes[0, 0].set_ylabel("Normalized reward")
    axes[1, 0].set_ylabel("Normalized reward")
    fig.suptitle("REINFORCE sampled reward; faint = seeds, bold = mean rolling median", fontsize=10)
    fig_save(FIGURES / "fig20_reinforce_reward_dynamics.png")


def make_fig21() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2), sharex=True)
    panels = (
        (axes[0, 0], "grad_norm", "Gradient norm"),
        (axes[0, 1], "entropy", "Policy entropy"),
        (axes[1, 0], "loss", "Policy-gradient loss"),
        (axes[1, 1], "infeasible", "Infeasible rollout fraction"),
    )
    for model in MODELS:
        color = method_color(model)
        series = {key: [] for key, _, _ in ((p[1], None, None) for p in panels)}
        steps = None
        for seed in SEEDS:
            rows = _read(FULL / "runs" / f"{model}_seed{seed}" / "reward.csv")
            steps = [int(row["step"]) for row in rows]
            series["grad_norm"].append([float(row["grad_norm"]) for row in rows])
            series["entropy"].append([float(row["entropy"]) for row in rows])
            series["loss"].append([float(row["loss"]) for row in rows])
            series["infeasible"].append(
                [float(row["n_infeasible_rollouts"]) / float(row["n_rollouts"]) for row in rows]
            )
        for ax, key, title in panels:
            arr = np.array(series[key])
            ax.plot(steps, arr.mean(axis=0), color=color, lw=1.5, label=method_label(model, short=True))
            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=6.5, frameon=True)
    axes[1, 0].set_xlabel("RL step")
    axes[1, 1].set_xlabel("RL step")
    fig.suptitle("RL training diagnostics, mean across five seeds", fontsize=10)
    fig_save(SUPP_FIGURES / "fig21_rl_training_stability.png")


def caption_numbers() -> None:
    print("n80", coverage(80))
    for role in ("bc_best", "rl_best"):
        for protocol, _ in PROTOCOLS:
            means = {m: _mean(model_seed_means(role, 10, protocol)[m]) for m in MODELS}
            print(role, protocol, {m: round(v, 3) for m, v in means.items()})
    for model in MODELS:
        wins = 0
        bc = model_seed_means("bc_best", 10, "P0")[model]
        rl = model_seed_means("rl_best", 10, "P0")[model]
        for b, r in zip(bc, rl):
            if r < b:
                wins += 1
        print(model, "rl_better_seeds", wins, "bc", round(_mean(bc), 3), "rl", round(_mean(rl), 3))


if __name__ == "__main__":
    for fn in (
        make_fig12,
        make_fig13,
        make_fig14,
        make_fig15,
        make_fig16,
        make_fig17,
        make_fig18,
        make_fig19,
        make_fig20,
        make_fig21,
    ):
        fn()
    caption_numbers()
