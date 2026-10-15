"""Generate nonlinear-charging and reward figures/tables from experiment CSVs."""

from __future__ import annotations

import csv
import math
import statistics
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

from presentation_names import method_label, wrap_label  # noqa: E402
from reporting_paths import FIGURES, RAW, RESULTS, TABLES  # noqa: E402

DPI = 300
NL = RAW / "nonlinear_charging"
RL = RAW / "reward_learning"


def _read(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(xs) if xs else float("nan")


def fig_save(path: Path):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print("wrote", path)


def write_table(name, title, meta, headers, rows, notes):
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    lines = [
        f"# {title}",
        "",
        *[f"- {k}: {v}" for k, v in meta.items()],
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.extend(["", "## Interpretation", "", notes, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path)


def make_curve_comparison_figure():
    from charging_surrogates import linear_early_rate_curve, linear_secant_curve
    from data import instance_files, parse_instance

    inst = None
    for path in instance_files(ROOT / "dataset"):
        cand = parse_instance(path)
        if set(cand.curves) >= {"slow", "normal", "fast"}:
            inst = cand
            break
    if inst is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    colors = {"TRUE-NL": "#4c78a8", "LINEAR-SECANT": "#f28e2b", "LINEAR-EARLY-RATE": "#e15759"}
    for ax, stype in zip(axes, ("slow", "normal", "fast")):
        true = inst.curves[stype]
        sec = linear_secant_curve(true)
        early = linear_early_rate_curve(true)
        ax.plot(true.times, [e / 1000 for e in true.energies], "o-", color=colors["TRUE-NL"], label="TRUE-NL")
        ax.plot(sec.times, [e / 1000 for e in sec.energies], "s--", color=colors["LINEAR-SECANT"], label="LINEAR-SECANT")
        ax.plot(early.times, [e / 1000 for e in early.energies], "^--", color=colors["LINEAR-EARLY-RATE"], label="LINEAR-EARLY-RATE")
        ax.set_title(f"{stype} charger")
        ax.set_xlabel("Charging time (h)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Battery energy (kWh)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Nonlinear vs linear charging models ({inst.name})", fontsize=11)
    fig_save(FIGURES / "fig_main02_nonlinear_vs_linear_curves.png")


def make_feasibility_figure():
    rows = _read(NL / "fixed_route_charging_comparison.csv")
    if not rows:
        print("WARN: missing fixed_route CSV")
        return
    # plan_* methods only
    plan_rows = [r for r in rows if r["method"].startswith("plan_")]
    by = defaultdict(lambda: {"n": 0, "feas": 0})
    for r in plan_rows:
        key = (int(r["size"]), r["charging_model"])
        by[key]["n"] += 1
        by[key]["feas"] += int(r["feasible_under_true_physics"])
    sizes = sorted({k[0] for k in by})
    models = ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    x = np.arange(len(sizes))
    width = 0.25
    for i, model in enumerate(models):
        rates = []
        for s in sizes:
            d = by[(s, model)]
            rates.append(100.0 * d["feas"] / d["n"] if d["n"] else 0.0)
        ax.bar(x + (i - 1) * width, rates, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={s}" for s in sizes])
    ax.set_ylabel("% instances feasible under TRUE physics")
    ax.set_title("Fixed-route plans: feasibility when re-evaluated on Montoya nonlinear charging")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig_save(FIGURES / "fig_main03_linear_plan_true_feasibility.png")


def make_charge_time_paired_figure():
    rows = _read(NL / "fixed_route_charging_comparison.csv")
    if not rows:
        return
    # Paired charge hours where BOTH true-nl and linear plan are feasible under true physics
    by_inst = defaultdict(dict)
    for r in rows:
        if r["method"].startswith("plan_") and r["feasible_under_true_physics"] == "1":
            by_inst[(r["size"], r["instance"])][r["charging_model"]] = float(r["true_charging_h"])
    deltas = {"LINEAR-SECANT": [], "LINEAR-EARLY-RATE": []}
    for key, d in by_inst.items():
        if "TRUE-NL" not in d:
            continue
        for model in deltas:
            if model in d:
                deltas[model].append(d[model] - d["TRUE-NL"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data = [deltas[m] for m in ("LINEAR-SECANT", "LINEAR-EARLY-RATE") if deltas[m]]
    labels = [m for m in ("LINEAR-SECANT", "LINEAR-EARLY-RATE") if deltas[m]]
    if not data:
        ax.text(0.5, 0.5, "No paired feasible cases for charge-time deltas", ha="center")
    else:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel("True charging h (linear plan − TRUE-NL plan)")
    ax.set_title("Paired true nonlinear charging time on matched customer sequences")
    ax.grid(True, axis="y", alpha=0.3)
    fig_save(FIGURES / "fig_main03b_paired_charging_time_deltas.png")


def make_objective_decomposition_figure():
    rows = _read(NL / "fixed_route_charging_comparison.csv")
    if not rows:
        return
    # TRUE-NL feasible plans only — travel / service / charge stack by size
    by_size = defaultdict(list)
    for r in rows:
        if r["method"] == "plan_TRUE-NL" and r["feasible_under_true_physics"] == "1":
            by_size[int(r["size"])].append(
                (
                    float(r["travel_h"]),
                    float(r["service_h"]),
                    float(r["true_charging_h"]),
                )
            )
    if not by_size:
        return
    sizes = sorted(by_size)
    travel = [_mean([t for t, s, c in by_size[n]]) for n in sizes]
    service = [_mean([s for t, s, c in by_size[n]]) for n in sizes]
    charge = [_mean([c for t, s, c in by_size[n]]) for n in sizes]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = np.arange(len(sizes))
    ax.bar(x, travel, label="Travel", color="#4c78a8")
    ax.bar(x, service, bottom=travel, label="Service", color="#72b7b2")
    ax.bar(
        x,
        charge,
        bottom=[a + b for a, b in zip(travel, service)],
        label="Nonlinear charging",
        color="#e15759",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in sizes])
    ax.set_ylabel("Mean hours (feasible TRUE-NL fixed-route plans)")
    ax.set_title("Full objective decomposition under Montoya nonlinear charging")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig_save(FIGURES / "fig_main04_objective_decomposition_nl.png")


def make_e2e_feasibility_figure():
    rows = _read(NL / "end_to_end_charging_sensitivity.csv")
    if not rows:
        return
    by = defaultdict(lambda: {"n": 0, "feas": 0})
    for r in rows:
        key = (int(r["size"]), r["charging_model"])
        by[key]["n"] += 1
        by[key]["feas"] += int(r["feasible_under_true_physics"])
    sizes = sorted({k[0] for k in by})
    models = ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    x = np.arange(len(sizes))
    width = 0.25
    for i, model in enumerate(models):
        rates = [
            100.0 * by[(s, model)]["feas"] / by[(s, model)]["n"] if by[(s, model)]["n"] else 0.0
            for s in sizes
        ]
        ax.bar(x + (i - 1) * width, rates, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={s}" for s in sizes])
    ax.set_ylabel("% instances feasible under TRUE physics")
    ax.set_title("End-to-end routing: feasibility when charging model is misspecified")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig_save(FIGURES / "fig_main03c_e2e_true_feasibility.png")


def write_table15():
    summary = _read(NL / "charging_study_summary.csv")
    detail = _read(NL / "fixed_route_charging_comparison.csv") + _read(
        NL / "end_to_end_charging_sensitivity.csv"
    )
    if not summary:
        write_table(
            "table15_nonlinear_vs_linear_charging.md",
            "Nonlinear vs linear charging under true physics",
            {"status": "unavailable — run run_nonlinear_charging_study.py"},
            ["Status"],
            [["Experiment not yet executed"]],
            "Required for RQ3.",
        )
        return

    def _agg_runtime(experiment, size, model, method):
        xs = []
        for r in detail:
            if r.get("experiment") != experiment:
                continue
            if int(r["size"]) != int(size):
                continue
            if r.get("charging_model") != model:
                continue
            # fixed-route has method; e2e rows may use charging_model only
            r_method = r.get("method") or "e2e"
            if r_method != method:
                continue
            xs.append(r)
        if not xs:
            return "n/a", "n/a"
        rts = [float(r["runtime_s"]) for r in xs if r.get("runtime_s") not in ("", None)]
        orcs = [
            float(r["oracle_solver_executions"])
            for r in xs
            if r.get("oracle_solver_executions") not in ("", None)
        ]
        mean_rt = f"{_mean(rts):.4f}" if rts else "n/a"
        mean_orc = f"{_mean(orcs):.2f}" if orcs else "n/a"
        return mean_rt, mean_orc

    rows = []
    for r in summary:
        mch = r["mean_true_charging_h_feasible_only"]
        mfull = r["mean_full_h_true_feasible_only"]
        men = r["mean_energy_kwh_feasible_only"]
        mchg = r.get("mean_n_charges_feasible_only", "")
        mean_rt, mean_orc = _agg_runtime(
            r["experiment"], r["size"], r["charging_model"], r["method"]
        )
        rows.append(
            [
                r["size"],
                r["charging_model"],
                r["method"],
                r["n_instances"],
                f"{r['feasible_under_true_physics']}/{r['n_instances']}",
                r["duration_violations"],
                f"{float(mch):.4f}" if mch not in ("", None) else "n/a",
                f"{float(mfull):.4f}" if mfull not in ("", None) else "n/a",
                f"{float(men):.3f}" if men not in ("", None) else "n/a",
                f"{float(mchg):.2f}" if mchg not in ("", None) else "n/a",
                mean_rt,
                mean_orc,
            ]
        )
    write_table(
        "table15_nonlinear_vs_linear_charging.md",
        "Nonlinear vs linear charging sensitivity (true-physics evaluation)",
        {
            "Objective convention": "All reported full/charging hours use TRUE Montoya nonlinear physics",
            "Units": "hours / kWh; runtime seconds; oracle = frvcpy FixedRouteOracle executions",
            "Protocol": "Fixed-route: matched H2+LS sequences; E2E: same H2 weights, different planning curves",
            "Source raw": "results/raw/nonlinear_charging/",
            "Seeds": "42",
            "Checkpoint": "H2_n10_seed42_c3.pt",
            "Coverage": "20 official instances each for n=10,20,40",
        },
        [
            "Benchmark size",
            "Charging model",
            "Method",
            "Instances",
            "Feasible under true physics",
            "Duration violations",
            "Mean true charging h (feas only)",
            "Mean realized full h (feas only)",
            "Charged energy kWh (feas only)",
            "Charger visits (feas only)",
            "Mean runtime s",
            "Mean oracle executions",
        ],
        rows,
        "Do not rank methods by feasible-only means when feasibility rates differ. "
        "`repair_TRUE-NL_after_*` is a separate repaired method, not the original linear plan. "
        "LINEAR-EARLY-RATE often plans under the surrogate but fails true-physics duration checks; "
        "LINEAR-SECANT often fails already under the planner on larger instances.",
    )


def make_reward_figures():
    reward_files = list(RL.glob("reward_*_rl_from_bc_rq4.csv"))
    if not reward_files:
        print("WARN: no reward CSVs yet; skipping reward dynamics figure")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for path in sorted(reward_files):
        rows = _read(path)
        parts = path.stem.split("_")
        model = parts[1]
        seed = parts[3].replace("seed", "")
        steps = [int(r["step"]) for r in rows]
        rewards = [float(r["mean_sampled_reward"]) for r in rows if r.get("mean_sampled_reward") not in ("", None)]
        feas = [
            100 * float(r["rollout_feasible"])
            for r in rows
            if r.get("rollout_feasible") not in ("", None)
        ]
        adv = [float(r["mean_advantage"]) for r in rows if r.get("mean_advantage") not in ("", None)]
        label = f"{model} s{seed}"
        if rewards:
            axes[0].plot(steps[: len(rewards)], rewards, alpha=0.75, label=label)
        if feas:
            axes[1].plot(steps[: len(feas)], feas, alpha=0.75, label=label)
        if adv:
            axes[2].plot(steps[: len(adv)], adv, alpha=0.75, label=label)
    axes[0].set_title("Sampled reward vs RL step")
    axes[0].set_xlabel("RL step")
    axes[0].set_ylabel("R (−Z_full or −100)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_title("Rollout feasibility")
    axes[1].set_xlabel("RL step")
    axes[1].set_ylabel("% feasible (logged rollout)")
    axes[1].set_ylim(-5, 110)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_title("Advantage (R − greedy baseline)")
    axes[2].set_xlabel("RL step")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Controlled RL reward diagnostics (warm-start from BC-only)", fontsize=11)
    fig_save(FIGURES / "fig_main07_reward_dynamics.png")

    # Paired BC vs RL official construction means
    summary = _read(RL / "bc_vs_rl_official_n10_summary.csv")
    if summary:
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        keys = []
        bc_vals = []
        rl_vals = []
        for model in ("H2", "H4"):
            for seed in ("42", "43", "44"):
                bc = next(
                    (
                        r
                        for r in summary
                        if r["model"] == model and r["seed"] == seed and r["tag"] == "bc_only_rq4"
                    ),
                    None,
                )
                rl = next(
                    (
                        r
                        for r in summary
                        if r["model"] == model and r["seed"] == seed and r["tag"] == "rl_from_bc_rq4"
                    ),
                    None,
                )
                if bc and rl:
                    keys.append(f"{model}\ns{seed}")
                    bc_vals.append(float(bc["mean_full_h"]))
                    rl_vals.append(float(rl["mean_full_h"]))
        x = np.arange(len(keys))
        w = 0.35
        ax.bar(x - w / 2, bc_vals, w, label="BC-only", color="#4c78a8")
        ax.bar(x + w / 2, rl_vals, w, label="RL-from-BC", color="#e15759")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=8)
        ax.set_ylabel("Mean full h (official n=10, construction-only)")
        ax.set_title("Paired BC-only vs RL checkpoints (controlled RQ4)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig_save(FIGURES / "fig_main07b_bc_vs_rl_official.png")

    placeholder = FIGURES / "fig14_reward_dynamics_unavailable.png"
    if placeholder.exists():
        dest = RESULTS / "archive" / "figures_legacy" / "fig14_reward_dynamics_unavailable.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        placeholder.replace(dest)
        print("archived placeholder", dest)


def write_bc_rl_table():
    summary = _read(RL / "bc_vs_rl_official_n10_summary.csv")
    hist = _read(RAW / "training_logs" / "derived" / "bc_vs_rl_aggregate_c3.csv")
    rows = []
    if summary:
        for r in summary:
            rows.append(
                [
                    method_label(r["model"]),
                    r["seed"],
                    r["tag"],
                    f"{r['n_feasible']}/{r['n']}",
                    f"{float(r['mean_full_h']):.4f}",
                    "official n10 construction-only",
                ]
            )
    if hist:
        for r in hist:
            rows.append(
                [
                    method_label(r["model"]),
                    "42-44",
                    "c3_log_bc_best",
                    "synth val",
                    f"{float(r['mean_bc_best_val_h']):.4f}",
                    "historical Cycle-3 log (not controlled ckpt)",
                ]
            )
            rows.append(
                [
                    method_label(r["model"]),
                    "42-44",
                    "c3_log_rl_end",
                    "synth val",
                    f"{float(r['mean_rl_end_val_h']):.4f}",
                    f"RL improved seeds: {r['seeds_rl_improved']}/{r['n_seeds']}",
                ]
            )
    write_table(
        "table06_bc_vs_rl.md",
        "Behavior cloning vs reinforcement learning",
        {
            "Objective": "Full duration (h)",
            "Source": "results/raw/reward_learning + training_logs/derived",
            "Note": "Controlled RQ4 checkpoints are separate from Cycle-3 *_c3.pt",
        },
        ["Method", "Seed", "Stage/tag", "Feasibility", "Mean h", "Protocol"],
        rows if rows else [["unavailable", "", "", "", "", "run run_bc_rl_controlled.py"]],
        "Do not claim RL improves History+DeepSets unless controlled official/synth comparisons show it. "
        "Historical logs alone are not equivalent to saved BC-only vs RL checkpoints.",
    )


def make_schedule_comparison_figure():
    import json

    sched = NL / "schedules"
    true_p = sched / "tc0c40s5cf4_TRUE-NL.json"
    early_p = sched / "tc0c40s5cf4_LINEAR-EARLY-RATE_infeasible_under_true.json"
    if not true_p.exists():
        return
    true = json.loads(true_p.read_text(encoding="utf-8"))
    early = json.loads(early_p.read_text(encoding="utf-8")) if early_p.exists() else None

    def summarize(data):
        lines = [
            f"instance: {data.get('instance')}",
            f"model: {data.get('model')}",
            f"n_routes: {len(data.get('sequences') or [])}",
        ]
        for i, seq in enumerate((data.get("sequences") or [])[:6]):
            lines.append(f"route {i + 1} customers: {seq}")
        stops = data.get("stops") or []
        n_ch = sum(1 for route in stops for _node, dq in route if dq is not None)
        lines.append(f"charger visits in plan: {n_ch}")
        if data.get("full_h") is not None:
            lines.append(f"full_h (true): {float(data['full_h']):.3f}")
        return "\n".join(lines)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, data, title in (
        (axes[0], true, "TRUE-NL plan (feasible under true physics)"),
        (axes[1], early, "LINEAR-EARLY-RATE plan (infeasible under true physics)"),
    ):
        ax.axis("off")
        if not data:
            ax.text(0.5, 0.5, "missing schedule JSON", ha="center")
            continue
        ax.text(
            0.02,
            0.98,
            title + "\n\n" + summarize(data),
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
        )
    fig.suptitle("Representative fixed-route schedules: nonlinear vs linear planning", fontsize=11)
    fig_save(FIGURES / "fig_main03d_schedule_comparison_example.png")


def main():
    """Tables only. Main-paper figures are built by publication_figures.py."""
    write_table15()
    write_bc_rl_table()
    print(
        "charging_and_reward_figures: wrote Markdown tables only; "
        "run scripts/reporting/publication_figures.py for main PNGs"
    )


if __name__ == "__main__":
    main()
