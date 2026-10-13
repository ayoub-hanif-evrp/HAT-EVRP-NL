"""Neural-learning figures and tables (PNG / Markdown) grounded in training logs."""

from __future__ import annotations

import csv
import math
import statistics
import subprocess
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

from presentation_names import (  # noqa: E402
    method_label,
    search_label,
    stage_label,
    wrap_label,
)
from reporting_paths import FIGURES, RAW, RESULTS, TABLES  # noqa: E402

DPI = 300
DERIVED = RAW / "training_logs" / "derived"
LOG_DIR = RAW / "training_logs"


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(xs) if xs else float("nan")


def _std(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_table(name: str, title: str, meta: dict, headers: list[str], rows: list[list], notes: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    lines = [
        f"# {title}",
        "",
        f"- Objective convention: {meta.get('objective', 'n/a')}",
        f"- Units: {meta.get('units', 'n/a')}",
        f"- Benchmark / data: {meta.get('size', 'n/a')}",
        f"- Instance / seed count: {meta.get('n_instances', 'n/a')}",
        f"- Methods / checkpoints: {meta.get('methods', 'n/a')}",
        f"- Protocol: {meta.get('protocol', 'n/a')}",
        f"- Seeds: {meta.get('seeds', 'n/a')}",
        f"- Source raw: `{meta.get('source', 'n/a')}`",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.extend(["", "## Interpretation", "", notes, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def fig_save(path: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def ensure_derived() -> None:
    script = ROOT / "scripts" / "reporting" / "analyze_training_logs.py"
    subprocess.check_call([sys.executable, str(script), "--version", "c3"], cwd=str(ROOT))


def make_fig_supervision_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.1, 1.1, 1.4, 1.2, "Synthetic\nE-VRP-NL\ninstance"),
        (1.7, 1.1, 1.5, 1.2, "Greedy / LS\nteacher +\nfrvcpy"),
        (3.4, 1.1, 1.4, 1.2, "Behavior\ncloning\n(CE)"),
        (5.0, 1.1, 1.5, 1.2, "Pretrained\nhierarchical\npolicy"),
        (6.7, 1.1, 1.5, 1.2, "REINFORCE\nrollouts"),
        (8.4, 1.8, 1.5, 1.0, "frvcpy\n(non-diff)"),
        (8.4, 0.5, 1.5, 1.0, "Reward /\nbaseline /\nadvantage"),
        (10.1, 1.1, 1.6, 1.2, "Best synth\nvalidation\ncheckpoint"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor="#e8f1fb", edgecolor="#1f4e79", lw=1.3)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)
    for x0, x1, y in [
        (1.5, 1.7, 1.7),
        (3.2, 3.4, 1.7),
        (4.8, 5.0, 1.7),
        (6.5, 6.7, 1.7),
        (8.2, 8.4, 2.3),
        (8.2, 8.4, 1.0),
        (9.9, 10.1, 1.7),
    ]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title(
        "Training supervision flow (frvcpy is algorithmic, not trained)", fontsize=11
    )
    fig_save(FIGURES / "fig11_training_supervision_flow.png")


def make_fig_learning_curves() -> None:
    path = DERIVED / "learning_curves_long_c3.csv"
    rows = _read(path)
    if not rows:
        print("WARN: missing learning curve CSV")
        return
    models = ["H1", "H2", "H3", "H4"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=False)
    for ax, model in zip(axes.ravel(), models):
        sub = [r for r in rows if r["model"] == model]
        # BC validation mean ± std over seeds
        by_step = defaultdict(list)
        for r in sub:
            if r["stage"] == "bc":
                by_step[int(r["step"])].append(float(r["validation_objective"]))
        steps = sorted(by_step)
        means = [_mean(by_step[s]) for s in steps]
        stds = [_std(by_step[s]) for s in steps]
        ax.plot(steps, means, "-o", ms=3, color="#4c78a8", label="BC val (mean±std)")
        ax.fill_between(
            steps,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color="#4c78a8",
            alpha=0.2,
        )
        # RL endpoints per seed
        rl = [r for r in sub if r["stage"] in ("rl", "rl_pomo", "rl_repair")]
        if rl:
            xs = [int(r["step"]) + 400 for r in rl]  # offset for visual stage break
            # Use actual logged step; mark stage boundary at 400
            xs = [400 + int(r["step"]) for r in rl]
            ys = [float(r["validation_objective"]) for r in rl]
            ax.scatter(xs, ys, color="#e15759", s=40, zorder=5, label="RL endpoint (per seed)")
        # Best checkpoint markers
        best_pts = {}
        for r in sub:
            key = (r["seed"], r["overall_best_stage"], r["overall_best_step"])
            best_pts[r["seed"]] = r
        for seed, r in best_pts.items():
            stage = r["overall_best_stage"]
            step = int(float(r["overall_best_step"]))
            x = step if stage == "bc" else 400 + step
            ax.axvline(x, color="#59a14f", ls="--", alpha=0.35)
        ax.axvline(400, color="gray", ls=":", lw=1)
        ax.set_title(method_label(model, short=True), fontsize=9)
        ax.set_xlabel("Training step (BC 1–400; RL offset after 400)")
        ax.set_ylabel("Synth validation full duration (h)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Cycle-3 learning curves on independent synthetic validation "
        "(n=10 train; 12 val instances; seeds 42–44)",
        fontsize=11,
    )
    fig_save(FIGURES / "fig12_learning_curves_h1h4.png")

    # Separate BC-loss panel (not shared with RL loss)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model, color in (("H2", "#4c78a8"), ("H4", "#e15759")):
        sub = [r for r in rows if r["model"] == model and r["stage"] == "bc" and r["bc_loss"] != ""]
        by_step = defaultdict(list)
        for r in sub:
            by_step[int(r["step"])].append(float(r["bc_loss"]))
        steps = sorted(by_step)
        axes[0].plot(steps, [_mean(by_step[s]) for s in steps], "-o", ms=3, color=color, label=model)
    axes[0].set_title("Behavior-cloning cross-entropy (mean over seeds)")
    axes[0].set_xlabel("BC step")
    axes[0].set_ylabel("BC loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for model, color in (("H2", "#4c78a8"), ("H4", "#e15759")):
        sub = [r for r in rows if r["model"] == model and r["rl_loss"] != ""]
        if not sub:
            continue
        axes[1].scatter(
            [model] * len(sub),
            [float(r["rl_loss"]) for r in sub],
            color=color,
            s=50,
            label=method_label(model, short=True),
        )
    axes[1].set_title("RL policy-gradient loss at logged endpoint (separate scale)")
    axes[1].set_ylabel("RL loss (not comparable to BC CE)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Cycle-3 losses: BC CE vs RL PG (separate panels; not comparable)", fontsize=11)
    fig_save(FIGURES / "fig12b_bc_and_rl_losses_separate.png")


def make_fig_rl_effect() -> None:
    rows = _read(DERIVED / "bc_vs_rl_per_seed_c3.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    models = ["H1", "H2", "H3", "H4"]
    x = np.arange(len(models))
    width = 0.35
    bc_means, bc_stds, rl_means, rl_stds = [], [], [], []
    for m in models:
        g = [r for r in rows if r["model"] == m]
        bc_means.append(_mean([float(r["bc_best_val_h"]) for r in g]))
        bc_stds.append(_std([float(r["bc_best_val_h"]) for r in g]))
        rl_means.append(_mean([float(r["rl_end_val_h"]) for r in g]))
        rl_stds.append(_std([float(r["rl_end_val_h"]) for r in g]))
    ax.bar(x - width / 2, bc_means, width, yerr=bc_stds, capsize=3, label="Best BC validation", color="#4c78a8")
    ax.bar(x + width / 2, rl_means, width, yerr=rl_stds, capsize=3, label="RL endpoint validation", color="#e15759")
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(method_label(m, short=True), 18) for m in models], fontsize=7)
    ax.set_ylabel("Synthetic validation full duration (h)")
    ax.set_title("RL effect after BC (Cycle-3, seeds 42–44; lower is better)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate H4 instability
    ax.annotate(
        "H4 RL endpoint unstable\n(checkpoint keeps best BC)",
        xy=(3 + width / 2, rl_means[3]),
        xytext=(2.2, max(rl_means) * 0.85),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#333"),
    )
    fig_save(FIGURES / "fig13_rl_effect_h2_h4.png")


def make_fig_reward_unavailable() -> None:
    """Skip placeholder when real reward dynamics exist; otherwise archive-only notice."""
    real = FIGURES / "fig_main07_reward_dynamics.png"
    reward_csvs = list((RAW / "reward_learning").glob("reward_*_rl_from_bc_rq4.csv")) if (RAW / "reward_learning").exists() else []
    if real.exists() or reward_csvs:
        # Real RQ4 reward figure supersedes the historical placeholder.
        placeholder = FIGURES / "fig14_reward_dynamics_unavailable.png"
        if placeholder.exists():
            dest = RESULTS / "archive" / "figures_legacy" / placeholder.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            placeholder.replace(dest)
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "Reward dynamics figure unavailable\n\n"
        "Historical Cycle-3 training CSVs do not record sampled rewards,\n"
        "baselines, advantages, or infeasible-rollout counts.\n\n"
        "Run scripts/experiments/run_bc_rl_controlled.py for real reward logs.\n"
        "Do not reconstruct rewards from validation objectives.",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#999"),
    )
    ax.set_title("FIGURE D — Reward dynamics (data not present in historical logs)")
    fig_save(FIGURES / "fig14_reward_dynamics_unavailable.png")


def make_fig_bc_vs_rl() -> None:
    rows = _read(DERIVED / "bc_vs_rl_aggregate_c3.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    models = [r["model"] for r in rows]
    deltas = [float(r["mean_rl_minus_bc_best_h"]) for r in rows]
    errs = [float(r["std_rl_minus_bc_best_h"]) for r in rows]
    colors = ["#59a14f" if d < 0 else "#e15759" for d in deltas]
    y = np.arange(len(models))
    ax.barh(y, deltas, xerr=errs, color=colors, capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([method_label(m, short=True) for m in models], fontsize=8)
    ax.set_xlabel("RL endpoint − best BC validation (h); negative = RL better")
    ax.set_title("BC vs RL on synthetic validation (Cycle-3; mean±std over 3 seeds)")
    ax.grid(True, axis="x", alpha=0.3)
    fig_save(FIGURES / "fig15_bc_vs_rl_ablation.png")


def make_fig_hierarchical_policy() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 1.3, 1.8, 1.6, "Open EV\nembeddings\n(+ DeepSets)"),
        (2.3, 1.3, 1.8, 1.6, "Select route:\nexisting EV\nor NEW_ROUTE"),
        (4.4, 1.3, 1.8, 1.6, "Select next\ncustomer\n(masked pointer)"),
        (6.5, 1.3, 1.8, 1.6, "Append /\nbest feasible\ninsertion"),
        (8.6, 2.2, 2.0, 1.2, "frvcpy FRVCP-NL\n(charging plan)"),
        (8.6, 0.6, 2.0, 1.2, "Python validator\nSOC / duration"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor="#eef6ee", edgecolor="#2b6b2b", lw=1.3)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    for x0, x1 in [(2.0, 2.3), (4.1, 4.4), (6.2, 6.5), (8.3, 8.6)]:
        ax.annotate("", xy=(x1, 2.1), xytext=(x0, 2.1), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title(
        "Hierarchical decoding: route choice → customer choice → charging (frvcpy)",
        fontsize=11,
    )
    fig_save(FIGURES / "fig16_hierarchical_policy.png")


def make_fig_history_deepsets() -> None:
    # Use FINAL_n10_fairls means
    path = RAW / "cycle3" / "FINAL_n10_fairls.csv"
    rows = _read(path)
    if not rows:
        return
    order = ["H1", "H2", "H3", "H4"]
    means = []
    labels = []
    for m in order:
        match = [r for r in rows if r["method"] == m]
        if match:
            means.append(float(match[0]["mean_h"]))
            labels.append(wrap_label(method_label(m, short=True), 20))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = np.arange(len(means))
    ax.hlines(y, 19.5, means, color="#ccc")
    ax.plot(means, y, "o", color="#4c78a8", ms=10)
    for yi, m in zip(y, means):
        ax.text(m + 0.02, yi, f"{m:.3f} h", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean full duration (h) — n=10 fair-LS Cycle-3 (60 instance-seeds)")
    ax.set_title("Architecture contribution under matched fair-LS (not DeepSets superiority claim)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(19.5, max(means) + 0.35)
    fig_save(FIGURES / "fig17_history_deepsets_contribution.png")


def write_learning_tables() -> None:
    from train import VARIANTS

    # table10 training configuration
    rows = []
    for key in ("H1", "H2", "H3", "H4"):
        cfg = VARIANTS[key]
        rows.append(
            [
                method_label(key, short=True),
                key,
                "yes" if cfg.history else "no",
                "yes" if cfg.deepsets else "no",
                cfg.d_model,
                cfg.layers,
                cfg.heads,
                "400 BC",
                "standard REINFORCE",
                "40",
                "120 synth n=10",
                "12 synth n=10",
                "42,43,44",
                "Adam lr=default",
                "best (feas, −val obj) over logged steps",
            ]
        )
    _write_table(
        "table10_training_configuration.md",
        "Training configuration for Cycle-3 H1–H4 checkpoints",
        {
            "objective": "Synthetic validation full duration for checkpoint selection",
            "units": "hours / steps",
            "size": "n=10 synthetic train/val",
            "n_instances": "120 train / 12 val per seed",
            "methods": "H1–H4 Cycle-3",
            "protocol": "BC then standard REINFORCE (pomo_k=0); generator v2 defaults",
            "seeds": "42–44",
            "source": "results/archive/reports/notes.md + train.py defaults + training_*_c3.csv",
        },
        [
            "Display name",
            "Key",
            "History",
            "DeepSets",
            "Hidden",
            "Layers",
            "Heads",
            "BC steps",
            "RL type",
            "RL steps",
            "Train inst.",
            "Val inst.",
            "Seeds",
            "Optimizer",
            "Checkpoint rule",
        ],
        rows,
        "Insertion/POMO/repair flags were **not** used for Cycle-3 primary checkpoints. "
        "Post-Cycle-3 `perf_*` runs are separate warm-start experiments.",
    )

    _write_table(
        "table11_reward_functions.md",
        "Reward definitions used in train.py",
        {
            "objective": "Training reward (not the publication BKS gap)",
            "units": "hours / dimensionless penalty",
            "size": "any decode size",
            "n_instances": "n/a",
            "methods": "standard RL / POMO / repair RL",
            "protocol": "as implemented in train.py",
            "seeds": "n/a",
            "source": "train.py (reinforce / pomo_reinforce_step / repair_reinforce_step)",
        },
        [
            "Stage",
            "Feasible reward",
            "Infeasible penalty",
            "Baseline",
            "Loss",
            "Used for Cycle-3 H2/H4?",
        ],
        [
            [
                "Behavior cloning",
                "N/A (cross-entropy on teacher actions)",
                "N/A",
                "N/A",
                "masked hierarchical CE",
                "Yes (400 steps)",
            ],
            [
                "Standard REINFORCE",
                "R = −Z_full",
                "R = −100",
                "greedy rollout reward",
                "L = −(R−b)·Σ log π",
                "Yes (40 steps)",
            ],
            [
                "POMO REINFORCE",
                "R = −Z_full",
                "R = −100",
                "mean of K rollout rewards",
                "mean of −(R−b)·log π",
                "No (Cycle-3 pomo_k=0)",
            ],
            [
                "Repair REINFORCE",
                "R = Z_before − Z_after",
                "R = −50",
                "mean repair-rollout reward",
                "mean of −(R−b)·log π",
                "No (Cycle-3 repair_rl_steps=0)",
            ],
        ],
        "Z_full = travel + service + nonlinear charging. Service is constant for complete "
        "feasible coverage, so maximizing R over feasible solutions is equivalent to minimizing "
        "the literature travel+charging objective. Fixed penalties −100/−50 were not re-tuned "
        "by size; all-infeasible repair batches yield zero gradient when advantages vanish.",
    )

    agg = _read(DERIVED / "bc_vs_rl_aggregate_c3.csv")
    per = _read(DERIVED / "bc_vs_rl_per_seed_c3.csv")
    table_rows = []
    for r in agg:
        table_rows.append(
            [
                method_label(r["model"], short=True),
                f"{float(r['mean_bc_best_val_h']):.4f} ± {float(r['std_bc_best_val_h']):.4f}",
                f"{float(r['mean_rl_end_val_h']):.4f} ± {float(r['std_rl_end_val_h']):.4f}",
                f"{float(r['mean_rl_minus_bc_best_h']):+.4f} ± {float(r['std_rl_minus_bc_best_h']):.4f}",
                f"{r['seeds_rl_improved']}/{r['n_seeds']}",
                f"{r['seeds_deployed_from_bc']}/{r['n_seeds']} BC; "
                f"{r['seeds_deployed_from_rl']}/{r['n_seeds']} RL",
                "unavailable (historical)",
            ]
        )
    _write_table(
        "table12_bc_vs_rl.md",
        "Behavior cloning vs REINFORCE on synthetic validation (Cycle-3)",
        {
            "objective": "Synthetic validation full duration (h)",
            "units": "hours",
            "size": "n=10 synthetic",
            "n_instances": "12 validation instances × 3 seeds",
            "methods": "H1–H4",
            "protocol": "Log-based: best BC step vs RL endpoint; checkpoint keeps best overall",
            "seeds": "42–44",
            "source": "results/raw/training_logs/derived/bc_vs_rl_aggregate_c3.csv",
        },
        [
            "Method",
            "Best BC val h",
            "RL endpoint val h",
            "Δ (RL−BC)",
            "Seeds RL improved",
            "Likely deployed stage",
            "Sampled reward traces",
        ],
        table_rows,
        "Across three Cycle-3 seeds, RL **endpoint** validation beats best-BC on only "
        "**1/3 seeds for H2** and **0/3 for H4** (and H1/H3). Mean RL endpoints are worse "
        "because several seeds collapse. Checkpoint selection still retains BC weights when "
        "RL worsens validation — so deployed H4 (and often H1/H3) are BC-selected. "
        "Do **not** claim RL improves HAT-EVRP-NL. Separate BC-only `.pt` files were not "
        "saved; official Montoya BC-only vs RL weight evaluation remains unavailable "
        "without a controlled retrain.",
    )

    diag_rows = []
    for r in per:
        if r["model"] not in ("H2", "H4"):
            continue
        diag_rows.append(
            [
                method_label(r["model"], short=True),
                r["seed"],
                f"{float(r['bc_end_entropy']):.3f}" if r["bc_end_entropy"] != "" else "",
                f"{float(r['rl_end_entropy']):.3f}" if r["rl_end_entropy"] != "" else "",
                f"{float(r['rl_end_loss']):.2f}" if r["rl_end_loss"] != "" else "",
                f"{float(r['bc_best_val_h']):.4f}",
                f"{float(r['rl_end_val_h']):.4f}",
                r["likely_deployed_stage"],
                "unstable" if float(r["rl_end_val_h"]) > float(r["bc_best_val_h"]) + 1.0 else "stable/mild",
            ]
        )
    _write_table(
        "table13_training_diagnostics.md",
        "Training diagnostics (Cycle-3 H2/H4 per seed)",
        {
            "objective": "Synthetic validation + logged losses/entropy",
            "units": "hours / nats",
            "size": "n=10 synthetic",
            "n_instances": "12 val instances",
            "methods": "H2, H4",
            "protocol": "Cycle-3 logs; RL only logged at endpoint (eval_every=40, rl_steps=40)",
            "seeds": "42–44",
            "source": "results/raw/training_logs/derived/bc_vs_rl_per_seed_c3.csv",
        },
        [
            "Method",
            "Seed",
            "BC-end entropy",
            "RL-end entropy",
            "RL-end PG loss",
            "BC-best val h",
            "RL-end val h",
            "Deployed stage (log)",
            "RL stability",
        ],
        diag_rows,
        "Entropy and RL loss are diagnostics only. Sampled reward variance was not logged "
        "historically. H4 seed-42 RL endpoint 31.02 h vs BC-best ~15.71 h is a clear instability.",
    )


def write_neural_contribution_table() -> None:
    """Construction-only vs post-search — keep search gains separate from model gains."""
    fair = _read(RAW / "cycle3" / "FINAL_n10_fairls.csv")
    equal = _read(RAW / "cycle3" / "FINAL_n10_equal_construction.csv")
    rows_out = []
    if fair and equal:
        fair_by = {r["method"]: r for r in fair}
        eq_by = {r["method"]: r for r in equal}
        for m in ("H0_strong", "H1", "H2", "H3", "H4"):
            if m not in fair_by or m not in eq_by:
                continue
            f = fair_by[m]
            e = eq_by[m]
            rows_out.append(
                [
                    method_label(m, short=True),
                    m,
                    f"{float(e['mean_h']):.4f}",
                    f"{float(f['mean_h']):.4f}",
                    f"{float(f['mean_h']) - float(e['mean_h']):+.4f}",
                    e.get("feasible", ""),
                    f.get("feasible", ""),
                    "equal construction vs fair-LS; n=10 Cycle-3",
                ]
            )
    pipe = _read(RAW / "regression_recovery" / "RECOVERY_n40_recovered_pipeline.csv")
    if pipe:
        for method in ("H2", "H4"):
            cons = [
                float(r["objective_h"])
                for r in pipe
                if r["method"] == method and r["stage"] == "construct" and r["feasible"] == "1"
            ]
            hyb = [
                float(r["objective_h"])
                for r in pipe
                if r["method"] == method and r["stage"] == "hybrid_insert" and r["feasible"] == "1"
            ]
            if cons and hyb:
                rows_out.append(
                    [
                        method_label(method, short=True),
                        method,
                        f"{_mean(cons):.4f}",
                        f"{_mean(hyb):.4f}",
                        f"{_mean(hyb) - _mean(cons):+.4f}",
                        f"{len(cons)}/20",
                        f"{len(hyb)}/20",
                        "n=40 recovered: construct vs hybrid_insert (search gain)",
                    ]
                )
    _write_table(
        "table14_neural_vs_search_contribution.md",
        "Neural construction vs post-search contribution",
        {
            "objective": "Full duration",
            "units": "hours",
            "size": "n=10 fair/equal; n=40 recovered",
            "n_instances": "20 (protocols labeled per row)",
            "methods": "H0_strong–H4 / H2–H4",
            "protocol": "Do not pool rows across protocols",
            "seeds": "Cycle-3 multi-seed n10; n40 seed 42",
            "source": "FINAL_n10_equal_construction + FINAL_n10_fairls + RECOVERY pipeline",
        },
        [
            "Model",
            "Key",
            "Construction-oriented mean h",
            "After search mean h",
            "Δ (search − construction)",
            "Feasible (constr.)",
            "Feasible (search)",
            "Protocol note",
        ],
        rows_out,
        "Negative Δ means search improved the objective. Architecture comparisons should use "
        "matched construction or matched search rows — never attribute hybrid-search gains to DeepSets.",
    )


def write_learning_report() -> None:
    agg = _read(DERIVED / "bc_vs_rl_aggregate_c3.csv")
    lines = [
        "# Learning Results Report",
        "",
        "This report summarizes **what the training logs actually show** about neural learning,",
        "without fabricating reward traces or claiming RL gains without evidence.",
        "",
        "## Cycle-3 protocol (primary checkpoints)",
        "",
        "- Independent synthetic generator (`v2`), n=10.",
        "- Teacher: greedy/local-search routes with `frvcpy` charging.",
        "- 400 behavior-cloning steps; 40 standard REINFORCE steps (`pomo_k=0`, no repair RL).",
        "- 120 train / 12 validation synthetic instances; seeds 42–44.",
        "- Checkpoint = best logged `(feasible_rate, −validation_objective)` — **not** the final RL iterate.",
        "",
        "## Reward (implementation)",
        "",
        "- Standard RL: `R = -Z_full` if feasible else `-100`; baseline = greedy rollout.",
        "- Loss: `L = -(R - baseline) * sum(log π)`.",
        "- `Z_full = travel + service + nonlinear charging`.",
        "- Historical logs **do not** store sampled reward distributions.",
        "",
        "## BC vs RL (synthetic validation)",
        "",
        "| Model | Mean BC-best h | Mean RL-end h | Mean Δ | Seeds RL helped | Deployed from BC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in agg:
        lines.append(
            f"| {r['model']} | {float(r['mean_bc_best_val_h']):.4f} | "
            f"{float(r['mean_rl_end_val_h']):.4f} | "
            f"{float(r['mean_rl_minus_bc_best_h']):+.4f} | "
            f"{r['seeds_rl_improved']}/{r['n_seeds']} | "
            f"{r['seeds_deployed_from_bc']}/{r['n_seeds']} |"
        )
    lines.extend(
        [
            "",
            "### Key conclusions",
            "",
            "1. Across seeds 42–44, RL **endpoint** beats best-BC validation on **1/3 H2 seeds** "
            "and **0/3 H4 seeds**. Mean RL endpoints are pulled up by collapses.",
            "2. **H4 seed 42** is a clear instability (BC-best ≈15.71 h → RL-end ≈31.02 h). "
            "Checkpoint selection retains BC weights — published H4 is **not** an RL-improved model.",
            "3. **Do not claim that RL improves HAT-EVRP-NL** from Cycle-3 alone. Even H2’s "
            "single-seed RL gain is not a robust multi-seed result.",
            "4. Separate BC-only `.pt` files were not saved; official Montoya BC-only vs RL "
            "weight ablation remains **unavailable** without a controlled retrain.",
            "5. Sampled reward traces are **unavailable** in historical logs (fig14).",
            "",
            "## Figures",
            "",
            "- `fig11_training_supervision_flow.png`",
            "- `fig12_learning_curves_h1h4.png`",
            "- `fig13_rl_effect_h2_h4.png`",
            "- `fig_main07_reward_dynamics.png` (real RQ4 logs; placeholder fig14 archived)",
            "- `fig15_bc_vs_rl_ablation.png`",
            "- `fig16_hierarchical_policy.png`",
            "- `fig17_history_deepsets_contribution.png`",
            "",
            "## Raw sources",
            "",
            "- `results/raw/training_logs/training_H*_n10_seed*_c3.csv`",
            "- `results/raw/training_logs/derived/`",
            "",
        ]
    )
    path = RESULTS / "LEARNING_RESULTS_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def build_all_learning() -> None:
    ensure_derived()
    make_fig_supervision_flow()
    make_fig_learning_curves()
    make_fig_rl_effect()
    make_fig_reward_unavailable()
    make_fig_bc_vs_rl()
    make_fig_hierarchical_policy()
    make_fig_history_deepsets()
    write_learning_tables()
    write_neural_contribution_table()
    write_learning_report()


if __name__ == "__main__":
    build_all_learning()
