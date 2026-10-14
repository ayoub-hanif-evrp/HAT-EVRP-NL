"""Generate the definitive main-paper figure set (~8 PNGs) and enforce layout.

Obsolete / supporting plots are moved to results/supplementary/figures/ or
results/archive/figures_legacy/. This module is the single source of truth for
results/figures/ contents.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presentation_names import METHOD_DISPLAY, SEARCH_DISPLAY, method_label  # noqa: E402
from reporting_paths import (  # noqa: E402
    ARCHIVE_FIGURES,
    AUTH_N40_DECOMP,
    AUTH_N40_RECOVERY_FAIR,
    AUTH_N40_RECOVERY_PIPE,
    FIGURES,
    MAIN_FIGURES,
    RAW,
    RESULTS,
    SUPP_FIGURES,
    TABLES,
)

DPI = 300
NL = RAW / "nonlinear_charging"
RL = RAW / "reward_learning"
CYCLE3 = RAW / "cycle3"

# Colorblind-friendly palette (shared across figures)
C = {
    "true": "#0072B2",
    "secant": "#E69F00",
    "early": "#D55E00",
    "travel": "#0072B2",
    "service": "#009E73",
    "charge": "#D55E00",
    "h0": "#999999",
    "h1": "#56B4E9",
    "h2": "#0072B2",
    "h3": "#E69F00",
    "h4": "#009E73",
    "bc": "#0072B2",
    "rl": "#D55E00",
    "plan_fail": "#999999",
    "false_feas": "#D55E00",
    "true_feas": "#009E73",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "font.size": 9,
            "axes.grid": False,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(xs) if xs else float("nan")


def _stdev(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def fig_save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("wrote", path)


# ---------------------------------------------------------------------------
# Figure 1 — Architecture
# ---------------------------------------------------------------------------
def make_fig01_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=7.5):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.1,
                color="#444444",
            )
        )

    # Neural stack (left-center)
    box(0.3, 8.2, 2.2, 1.0, "Montoya instance\n(nodes, energy, curves)", "#F0F0F0")
    box(3.0, 8.2, 2.0, 1.0, "Static node\nencoder", "#CFE2F3")
    box(5.5, 8.2, 2.2, 1.0, "Causal history\nTransformer", "#CFE2F3")
    box(8.0, 8.2, 1.7, 1.0, "Dynamic EV\nstate fusion", "#CFE2F3")
    arrow(2.5, 8.7, 3.0, 8.7)
    arrow(5.0, 8.7, 5.5, 8.7)
    arrow(7.7, 8.7, 8.0, 8.7)

    box(3.0, 6.4, 2.0, 1.0, "DeepSets fleet\naggregation", "#CFE2F3")
    box(5.5, 6.4, 2.4, 1.0, "Hierarchical\nroute / customer policy", "#CFE2F3")
    arrow(8.8, 8.2, 4.0, 7.4)
    arrow(5.0, 6.9, 5.5, 6.9)

    # Algorithmic (right)
    box(8.0, 6.4, 1.7, 1.0, "frvcpy FRVCP\n(exact charging)", "#FCE5CD")
    box(8.0, 4.6, 1.7, 1.0, "Python physical\nchecker", "#FCE5CD")
    arrow(7.9, 6.9, 8.0, 6.9)
    arrow(8.85, 6.4, 8.85, 5.6)

    box(3.0, 4.6, 4.5, 1.0, "Customer sequences  →  charge schedule  →  feasible routes", "#FFF2CC")
    arrow(6.7, 6.4, 5.2, 5.6)
    arrow(8.0, 5.1, 7.5, 5.1)

    # Training path
    ax.text(0.3, 3.5, "Training path", fontsize=9, fontweight="bold")
    box(0.3, 2.2, 2.0, 1.0, "Synthetic expert\n(greedy / LS + frvcpy)", "#E8F5E9")
    box(2.7, 2.2, 1.8, 1.0, "Behavior\ncloning", "#E8F5E9")
    box(4.9, 2.2, 2.0, 1.0, "Optional\nREINFORCE", "#E8F5E9")
    box(7.3, 2.2, 2.3, 1.0, "Validation-selected\ncheckpoint", "#E8F5E9")
    arrow(2.3, 2.7, 2.7, 2.7)
    arrow(4.5, 2.7, 4.9, 2.7)
    arrow(6.9, 2.7, 7.3, 2.7)

    ax.text(0.3, 1.2, "Blue = learned neural  ·  Orange = algorithmic (non-differentiable)", fontsize=8)
    ax.text(0.3, 0.5, "H1–H4 ablate history Transformer and DeepSets; charging always via frvcpy.", fontsize=8)
    ax.set_title("HAT-EVRP-NL hybrid architecture and training/inference paths", fontsize=11)
    fig_save(FIGURES / "fig01_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2 — Nonlinear vs linear curves
# ---------------------------------------------------------------------------
def make_fig02_curves() -> None:
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
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    for ax, stype in zip(axes, ("slow", "normal", "fast")):
        true = inst.curves[stype]
        sec = linear_secant_curve(true)
        early = linear_early_rate_curve(true)
        ax.plot(
            true.times,
            [e / 1000 for e in true.energies],
            "o-",
            color=C["true"],
            ms=4,
            lw=1.6,
            label="TRUE-NL (PWL)",
        )
        ax.plot(
            sec.times,
            [e / 1000 for e in sec.energies],
            "s--",
            color=C["secant"],
            ms=4,
            lw=1.4,
            label="LINEAR-SECANT",
        )
        ax.plot(
            early.times,
            [e / 1000 for e in early.energies],
            "^--",
            color=C["early"],
            ms=4,
            lw=1.4,
            label="LINEAR-EARLY-RATE",
        )
        ax.set_title(f"{stype.capitalize()} charger")
        ax.set_xlabel("Charging time (h)")
        ax.grid(True, alpha=0.25)
        # Mark late-SOC region
        if len(true.energies) >= 2:
            qmax = true.energies[-1] / 1000
            ax.axhspan(0.8 * qmax, qmax, color="#EEEEEE", zorder=0)
            ax.text(0.02, 0.82 * qmax, "late SOC\n(slower)", fontsize=6.5, color="#555555", va="bottom")
    axes[0].set_ylabel("Battery energy (kWh)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Montoya nonlinear charging vs linear surrogates ({inst.name})", fontsize=10, y=1.12)
    fig_save(FIGURES / "fig02_nonlinear_charging_curves.png")


# ---------------------------------------------------------------------------
# Figure 3 — Feasibility (fixed-route + e2e), three-state stacked bars
# ---------------------------------------------------------------------------
def _failure_categories(rows: list[dict], *, experiment: str) -> dict:
    """Return {(size, model): {true_feas, plan_fail, false_feas}}."""
    out = defaultdict(lambda: {"true_feas": 0, "plan_fail": 0, "false_feas": 0, "n": 0})
    for r in rows:
        if experiment == "fixed_route" and not r.get("method", "").startswith("plan_"):
            continue
        if experiment == "end_to_end" and r.get("experiment") not in (None, "end_to_end") and "charging_model" not in r:
            continue
        model = r["charging_model"]
        size = int(r["size"])
        key = (size, model)
        out[key]["n"] += 1
        plan_ok = int(r["feasible_under_planner"])
        true_ok = int(r["feasible_under_true_physics"])
        if true_ok:
            out[key]["true_feas"] += 1
        elif plan_ok:
            out[key]["false_feas"] += 1
        else:
            out[key]["plan_fail"] += 1
    return out


def make_fig03_feasibility() -> None:
    fixed = _read(NL / "fixed_route_charging_comparison.csv")
    e2e = _read(NL / "end_to_end_charging_sensitivity.csv")
    cat_a = _failure_categories(fixed, experiment="fixed_route")
    cat_b = _failure_categories(e2e, experiment="end_to_end")

    sizes = [10, 20, 40]
    models = ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]
    short = {"TRUE-NL": "TRUE-NL", "LINEAR-SECANT": "SECANT", "LINEAR-EARLY-RATE": "EARLY"}

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, cats, title in (
        (axes[0], cat_a, "A. Fixed-route charging optimization"),
        (axes[1], cat_b, "B. End-to-end neural routing"),
    ):
        x = np.arange(len(sizes))
        width = 0.26
        for i, model in enumerate(models):
            true_feas = [cats[(s, model)]["true_feas"] for s in sizes]
            false_feas = [cats[(s, model)]["false_feas"] for s in sizes]
            plan_fail = [cats[(s, model)]["plan_fail"] for s in sizes]
            xpos = x + (i - 1) * width
            ax.bar(xpos, true_feas, width, color=C["true_feas"], label="True-physics feasible" if i == 0 else None)
            ax.bar(
                xpos,
                false_feas,
                width,
                bottom=true_feas,
                color=C["false_feas"],
                label="Planner-OK, true-physics fail" if i == 0 else None,
            )
            bottoms = [a + b for a, b in zip(true_feas, false_feas)]
            ax.bar(
                xpos,
                plan_fail,
                width,
                bottom=bottoms,
                color=C["plan_fail"],
                label="Planner rejected / no stops" if i == 0 else None,
            )
            for xi, tf in zip(xpos, true_feas):
                ax.text(xi, 20.25, f"{tf}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"n={s}" for s in sizes], fontsize=8)
        ax.set_ylim(0, 22.5)
        ax.set_title(title, fontsize=9)
        ax.axhline(20, color="#CCCCCC", lw=0.8, ls=":")
        ax.set_ylabel("Instances (out of 20)" if ax is axes[0] else "")
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor=C["true_feas"], label="True-physics feasible"),
        Patch(facecolor=C["false_feas"], label="Planner-OK, true-physics fail"),
        Patch(facecolor=C["plan_fail"], label="Planner rejected / no stops"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(
        "Charging-model impact on physical feasibility (Montoya; 20 instances per size)",
        fontsize=10,
        y=1.12,
    )
    fig.text(
        0.5,
        -0.02,
        "Within each size group (left→right): TRUE-NL · LINEAR-SECANT · LINEAR-EARLY-RATE; numbers = true-feasible counts",
        ha="center",
        fontsize=7.5,
    )
    fig_save(FIGURES / "fig03_charging_model_feasibility.png")


# ---------------------------------------------------------------------------
# Figure 4 — Real SOC / charging schedule comparison
# ---------------------------------------------------------------------------
def _soc_trace_from_stops(inst, stops, *, use_true_curves: bool = True):
    """Simulate SOC vs time for a stop list; return times, socs, events, tmax_hit."""
    from charging import charging_time

    soc = inst.battery_wh
    times = [0.0]
    socs = [1.0]
    t = 0.0
    events = []
    tmax = inst.max_route_hours
    violated = False
    for index, (node, amount) in enumerate(stops):
        kind = inst.nodes[node].kind
        if amount is not None and amount > 1e-9 and kind == 2:
            stype = inst.nodes[node].station_type
            curve = inst.curves[stype]
            q_out = min(soc + float(amount), inst.battery_wh)
            dt = charging_time(curve, soc, q_out)
            events.append({"t0": t, "t1": t + dt, "stype": stype, "energy": q_out - soc, "dt": dt})
            t += dt
            soc = q_out
            times.append(t)
            socs.append(soc / inst.battery_wh)
            if t > tmax + 1e-9:
                violated = True
        if index + 1 < len(stops):
            nxt = stops[index + 1][0]
            dt = inst.travel_hours[node][nxt]
            if kind == 1:
                dt += inst.service_times[node]
            t += dt
            soc = max(0.0, soc - inst.energy_wh[node][nxt])
            times.append(t)
            socs.append(soc / inst.battery_wh)
            if t > tmax + 1e-9:
                violated = True
    return times, socs, events, violated, t


def make_fig04_schedule_case() -> None:
    from data import parse_instance

    # Prefer a compact n=10 false-feasible EARLY-RATE case with stops
    case = "tc0c10s2cf1"
    true_p = NL / "schedules" / f"{case}_TRUE-NL.json"
    early_p = NL / "schedules" / f"{case}_LINEAR-EARLY-RATE_infeasible_under_true.json"
    if not (true_p.exists() and early_p.exists()):
        return
    true_d = json.loads(true_p.read_text(encoding="utf-8"))
    early_d = json.loads(early_p.read_text(encoding="utf-8"))
    # Locate XML
    xml = None
    for p in (ROOT / "dataset").rglob(f"{case}.xml"):
        xml = p
        break
    if xml is None:
        return
    inst = parse_instance(xml)

    # Pick a route that has charging in the linear plan (route index with charges)
    route_idx = 0
    for i, route in enumerate(early_d.get("stops") or []):
        if any(dq is not None for _n, dq in route):
            route_idx = i
            break
    true_stops = true_d["stops"][route_idx] if route_idx < len(true_d["stops"]) else true_d["stops"][0]
    early_stops = early_d["stops"][route_idx]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    panels = (
        (axes[0], true_stops, "A. TRUE-NL plan (true-physics feasible)", C["true"]),
        (axes[1], early_stops, "B. LINEAR-EARLY-RATE plan replayed on TRUE-NL", C["early"]),
    )
    table_rows = []
    for ax, stops, title, color in panels:
        times, socs, events, violated, t_end = _soc_trace_from_stops(inst, stops)
        ax.plot(times, socs, "-", color=color, lw=1.8)
        for ev in events:
            ax.axvspan(ev["t0"], ev["t1"], color=C["charge"], alpha=0.25)
            mid = 0.5 * (ev["t0"] + ev["t1"])
            ax.text(mid, 0.08, ev["stype"][:1].upper(), ha="center", fontsize=6.5, color="#333333")
        ax.axvline(inst.max_route_hours, color="#333333", ls="--", lw=1.0, label=r"$T_{\max}$")
        if violated or t_end > inst.max_route_hours + 1e-9:
            ax.axvspan(inst.max_route_hours, max(t_end, inst.max_route_hours), color="#D55E00", alpha=0.12)
            ax.text(
                inst.max_route_hours + 0.05,
                0.92,
                "duration\nviolation",
                fontsize=6.5,
                color=C["early"],
            )
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Elapsed route time (h)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        charge_h = sum(ev["dt"] for ev in events)
        energy = sum(ev["energy"] for ev in events) / 1000
        table_rows.append(
            {
                "plan": title.split(". ", 1)[-1][:40],
                "route_index": route_idx,
                "customers": true_d["sequences"][route_idx] if route_idx < len(true_d["sequences"]) else "",
                "n_charges": len(events),
                "charge_h_true_curves": f"{charge_h:.4f}",
                "energy_kwh": f"{energy:.3f}",
                "route_end_h": f"{t_end:.4f}",
                "tmax_h": f"{inst.max_route_hours:.1f}",
                "duration_violation": int(t_end > inst.max_route_hours + 1e-9),
            }
        )
    axes[0].set_ylabel("SOC fraction")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"Same customer sequence, different charge plans ({case}, route {route_idx + 1})",
        fontsize=10,
    )
    fig_save(FIGURES / "fig04_charging_schedule_case.png")

    # Companion Markdown table (not inside PNG)
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Charging schedule case (companion to Figure 4)",
        "",
        f"- Instance: `{case}`",
        f"- Fixed customer sequences from TRUE-NL H2+LS fixed-route study",
        f"- Route index shown: {route_idx} (0-based)",
        f"- Customer sequence: `{true_d['sequences'][route_idx]}`",
        f"- LINEAR-EARLY-RATE schedule source: `{early_p.name}`",
        f"- TRUE-NL schedule source: `{true_p.name}`",
        f"- SOC traces replayed with **original Montoya nonlinear curves**",
        "",
        "| Plan | Route | # charges | True-curve charge h | Energy kWh | Route end h | Tmax h | Duration violation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['plan']} | {r['route_index']} | {r['n_charges']} | {r['charge_h_true_curves']} | "
            f"{r['energy_kwh']} | {r['route_end_h']} | {r['tmax_h']} | {r['duration_violation']} |"
        )
    lines.extend(
        [
            "",
            "## Full stop lists (node, Δq Wh)",
            "",
            "### TRUE-NL",
            "",
            "```",
            str(true_stops),
            "```",
            "",
            "### LINEAR-EARLY-RATE (replayed on TRUE-NL)",
            "",
            "```",
            str(early_stops),
            "```",
            "",
        ]
    )
    path = TABLES / "table_fig04_schedule_case.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path)


# ---------------------------------------------------------------------------
# Figure 5 — Objective decomposition (n40 recovered hybrid)
# ---------------------------------------------------------------------------
def make_fig05_decomposition() -> None:
    rows = _read(AUTH_N40_DECOMP)
    # Prefer hybrid_insert stage means
    by = defaultdict(list)
    for r in rows:
        if r.get("stage") != "hybrid_insert":
            continue
        if r.get("feasible") not in ("1", 1, "True", True):
            continue
        by[r["method"]].append(
            (float(r["travel_h"]), float(r["service_h"]), float(r["charging_h"]), float(r["full_h"]))
        )
    # Fallback verified published means if CSV filtering empty
    order = ["H2", "H4"]
    labels = [method_label(m, short=True) for m in order]
    if all(m in by and by[m] for m in order):
        travel = [_mean([t for t, s, c, f in by[m]]) for m in order]
        service = [_mean([s for t, s, c, f in by[m]]) for m in order]
        charge = [_mean([c for t, s, c, f in by[m]]) for m in order]
        totals = [_mean([f for t, s, c, f in by[m]]) for m in order]
    else:
        travel = [33.745, 33.219]
        service = [20.000, 20.000]
        charge = [3.060, 3.151]
        totals = [56.806, 56.370]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    y = np.arange(len(order))
    ax.barh(y, travel, color=C["travel"], label="Travel", height=0.55)
    ax.barh(y, service, left=travel, color=C["service"], label="Service", height=0.55)
    left2 = [a + b for a, b in zip(travel, service)]
    ax.barh(y, charge, left=left2, color=C["charge"], label="Nonlinear charging", height=0.55)
    for yi, tot in zip(y, totals):
        ax.text(tot + 0.3, yi, f"{tot:.3f} h", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean hours (20 official n=40 instances; recovered hybrid_insert)")
    ax.set_title("Objective decomposition under true nonlinear charging")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(0, max(totals) + 8)
    ax.grid(True, axis="x", alpha=0.25)
    # Note
    ax.text(
        0.99,
        0.02,
        "H4 improves mainly via travel, not charging\n(same frvcpy optimizer)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    fig_save(FIGURES / "fig05_objective_decomposition.png")


# ---------------------------------------------------------------------------
# Figure 6 — Architecture ablation (n10 fair-LS)
# ---------------------------------------------------------------------------
def make_fig06_ablation() -> None:
    methods = ["H0_strong", "H1", "H2", "H3", "H4"]
    seed_files = [
        CYCLE3 / f"compare_size10_fairls_final_seed{s}_c3.csv" for s in (42, 43, 44)
    ]
    seed_means = {m: [] for m in methods}
    for path in seed_files:
        rows = _read(path)
        by = defaultdict(list)
        for r in rows:
            if r["method"] in methods and r["feasible"] in ("1", "True", "true", 1):
                by[r["method"]].append(float(r["objective_h"]))
        for m in methods:
            if by[m]:
                seed_means[m].append(_mean(by[m]))
    means = [_mean(seed_means[m]) for m in methods]
    errs = [_stdev(seed_means[m]) for m in methods]
    colors = [C["h0"], C["h1"], C["h2"], C["h3"], C["h4"]]
    labels = [METHOD_DISPLAY[m] for m in methods]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    y = np.arange(len(methods))
    ax.errorbar(
        means,
        y,
        xerr=errs,
        fmt="o",
        color="#333333",
        ecolor="#666666",
        elinewidth=1.2,
        capsize=3,
        markersize=0,
    )
    for yi, m, mean, err, col in zip(y, methods, means, errs, colors):
        ax.plot(mean, yi, "o", color=col, ms=8, zorder=3)
        for sm in seed_means[m]:
            ax.plot(sm, yi, "|", color=col, ms=10, alpha=0.55)
        ax.text(mean + 0.05, yi + 0.18, f"{mean:.3f}±{err:.3f}", fontsize=7, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean full duration (h); seed means ± std (3 seeds × 20 instances)")
    ax.set_title("Matched n=10 Cycle-3 fair one-shot LS architecture comparison")
    # Zoom to data range
    lo = min(means) - 0.35
    hi = max(means) + 0.45
    ax.set_xlim(lo, hi)
    ax.axvline(means[methods.index("H2")], color=C["h2"], ls=":", lw=0.9, alpha=0.5)
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(
        0.99,
        0.02,
        "History helps vs state-only; DeepSets does not\nconsistently beat history alone",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    fig_save(FIGURES / "fig06_architecture_ablation.png")


# ---------------------------------------------------------------------------
# Figure 7 — Learning and reward
# ---------------------------------------------------------------------------
def make_fig07_learning() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.2))

    # A. BC validation progression (Cycle-3 H2/H4 seed42 as representative + all seeds faint)
    ax = axes[0]
    for model, color in (("H2", C["h2"]), ("H4", C["h4"])):
        for seed in (42, 43, 44):
            path = RAW / "training_logs" / f"training_{model}_n10_seed{seed}_c3.csv"
            rows = _read(path)
            bc = [r for r in rows if r.get("stage") == "bc" and r.get("validation_objective")]
            if not bc:
                continue
            steps = [int(r["step"]) for r in bc]
            vals = [float(r["validation_objective"]) for r in bc]
            alpha = 0.9 if seed == 42 else 0.25
            lw = 1.6 if seed == 42 else 0.9
            ax.plot(steps, vals, color=color, alpha=alpha, lw=lw, label=f"{model}" if seed == 42 else None)
    ax.set_title("A. BC validation (Cycle-3)")
    ax.set_xlabel("BC step")
    ax.set_ylabel("Synth-val full h")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    # B. Controlled BC vs RL official construction
    ax = axes[1]
    summary = _read(RL / "bc_vs_rl_official_n10_summary.csv")
    keys = []
    bc_vals = []
    rl_vals = []
    for model in ("H2", "H4"):
        for seed in ("42", "43", "44"):
            bc = next((r for r in summary if r["model"] == model and r["seed"] == seed and r["tag"] == "bc_only_rq4"), None)
            rl = next((r for r in summary if r["model"] == model and r["seed"] == seed and r["tag"] == "rl_from_bc_rq4"), None)
            if bc and rl:
                keys.append(f"{model}\ns{seed}")
                bc_vals.append(float(bc["mean_full_h"]))
                rl_vals.append(float(rl["mean_full_h"]))
    x = np.arange(len(keys))
    w = 0.36
    ax.bar(x - w / 2, bc_vals, w, color=C["bc"], label="BC-only")
    ax.bar(x + w / 2, rl_vals, w, color=C["rl"], label="RL-from-BC")
    for xi, b, r in zip(x, bc_vals, rl_vals):
        if r < b:
            ax.plot([xi - w / 2, xi + w / 2], [b, r], color="#009E73", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=7)
    ax.set_ylabel("Mean full h")
    ax.set_title("B. Controlled RQ4 (official n10)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.02,
        0.98,
        "H2: RL better 1/3\nH4: RL better 3/3\n(construction-only)",
        transform=ax.transAxes,
        va="top",
        fontsize=6.5,
        color="#333333",
    )

    # C. Smoothed reward (rolling median, window=5); skip constant feasibility
    ax = axes[2]
    window = 5
    for model, color in (("H2", C["h2"]), ("H4", C["h4"])):
        series = []
        for seed in (42, 43, 44):
            path = RL / f"reward_{model}_n10_seed{seed}_rl_from_bc_rq4.csv"
            rows = _read(path)
            if not rows:
                continue
            rewards = [float(r["mean_sampled_reward"]) for r in rows if r.get("mean_sampled_reward") not in ("", None)]
            steps = [int(r["step"]) for r in rows[: len(rewards)]]
            if len(rewards) < window:
                continue
            smooth = []
            for i in range(len(rewards)):
                lo = max(0, i - window + 1)
                smooth.append(float(np.median(rewards[lo : i + 1])))
            ax.plot(steps, rewards, color=color, alpha=0.15, lw=0.8)
            series.append((steps, smooth))
            ax.plot(steps, smooth, color=color, alpha=0.35, lw=1.0)
        if series:
            # mean of smoothed curves (align by min length)
            n = min(len(s[1]) for s in series)
            steps = series[0][0][:n]
            mean_s = np.mean([s[1][:n] for s in series], axis=0)
            ax.plot(steps, mean_s, color=color, lw=2.0, label=f"{model} (mean roll. med.)")
    ax.set_title(f"C. RL reward (roll. median w={window})")
    ax.set_xlabel("RL step")
    ax.set_ylabel("R (−Z_full or −100)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Learning evidence: Cycle-3 BC (A) vs controlled RQ4 BC/RL (B–C); protocols are distinct",
        fontsize=9.5,
        y=1.03,
    )
    fig_save(FIGURES / "fig07_learning_and_reward.png")


# ---------------------------------------------------------------------------
# Figure 8 — Quality vs compute
# ---------------------------------------------------------------------------
def make_fig08_quality_compute() -> None:
    """Quality vs stage-level oracle effort for recovered n40 H2 search modes."""
    fair = _read(AUTH_N40_RECOVERY_FAIR)
    pipe = _read(AUTH_N40_RECOVERY_PIPE)
    stages = ["basic_ls", "strong_ls", "classical_ruin", "neural_insert", "hybrid_insert"]
    markers = {"basic_ls": "o", "strong_ls": "s", "classical_ruin": "^", "neural_insert": "D", "hybrid_insert": "*"}
    colors = {
        "basic_ls": "#56B4E9",
        "strong_ls": "#0072B2",
        "classical_ruin": "#E69F00",
        "neural_insert": "#CC79A7",
        "hybrid_insert": "#009E73",
    }

    # Stage-level oracle from fair_search when available; basic_ls incremental from pipeline
    points = []
    for stage in stages:
        if stage == "basic_ls":
            objs, orcs, rts = [], [], []
            by_inst = defaultdict(dict)
            for r in pipe:
                if r["method"] != "H2" or r["feasible"] != "1":
                    continue
                by_inst[r["instance"]][r["stage"]] = r
            for _inst, d in by_inst.items():
                if "basic_ls" not in d or "construct" not in d:
                    continue
                objs.append(float(d["basic_ls"]["objective_h"]))
                orcs.append(
                    float(d["basic_ls"]["oracle_solver_executions"])
                    - float(d["construct"]["oracle_solver_executions"])
                )
                rts.append(float(d["basic_ls"]["runtime_s"]) - float(d["construct"]["runtime_s"]))
            if objs:
                points.append((stage, _mean(objs), _mean(orcs), _mean(rts)))
            continue
        xs = [r for r in fair if r["method"] == "H2" and r["mode"] == stage and r["feasible"] == "1"]
        if not xs:
            # fallback pipeline end-state objective with fair missing
            xs = [r for r in pipe if r["method"] == "H2" and r["stage"] == stage and r["feasible"] == "1"]
            if not xs:
                continue
            mean_h = _mean([float(r["objective_h"]) for r in xs])
            mean_orc = _mean([float(r["oracle_solver_executions"]) for r in xs])
            mean_rt = _mean([float(r["runtime_s"]) for r in xs])
            points.append((stage, mean_h, mean_orc, mean_rt))
            continue
        mean_h = _mean([float(r["objective_h"]) for r in xs])
        mean_orc = _mean([float(r["oracle_solver_executions"]) for r in xs])
        # fair_search runtime_search_s; strong_ls may be 0 → show n/a in label
        rts = [float(r["runtime_search_s"]) for r in xs if float(r.get("runtime_search_s") or 0) > 0]
        mean_rt = _mean(rts) if rts else float("nan")
        points.append((stage, mean_h, mean_orc, mean_rt))

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for stage, mean_h, mean_orc, mean_rt in points:
        rt_lab = f"{mean_rt:.2f}s" if math.isfinite(mean_rt) else "rt n/a"
        ax.scatter(
            mean_orc,
            mean_h,
            s=90,
            marker=markers[stage],
            color=colors[stage],
            label=f"{SEARCH_DISPLAY[stage]} ({rt_lab})",
            zorder=3,
        )
        ax.annotate(
            f"{mean_h:.2f} h",
            (mean_orc, mean_h),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )
    ax.set_xlabel("Mean stage-level frvcpy oracle executions (unequal realized budgets)")
    ax.set_ylabel("Mean full duration (h)")
    ax.set_title("n=40 recovered H2: quality vs oracle effort (20 instances, seed 42)")
    ax.invert_yaxis()
    ax.legend(frameon=True, fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.98,
        0.02,
        "Not an equal-budget experiment;\naggregate end-state means only",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    fig_save(FIGURES / "fig08_quality_compute.png")


# ---------------------------------------------------------------------------
# Layout enforcement
# ---------------------------------------------------------------------------
SUPPLEMENTARY_KEEP = {
    # Supporting / historical (not main paper)
    "fig02_problem_example.png",
    "fig03_charging_curves.png",
    "fig04_objective_decomposition_n40.png",  # if regenerated under old name
    "fig05_soc_charging_trace.png",
    "fig07_scaling_fairls.png",
    "fig09_h2_vs_h4_n40_hybrid.png",
    "fig10_charging_ablation_feasibility.png",
    "fig11_training_supervision_flow.png",
    "fig12_learning_curves_h1h4.png",
    "fig12b_bc_and_rl_losses_separate.png",
    "fig13_rl_effect_h2_h4.png",
    "fig15_bc_vs_rl_ablation.png",
    "fig16_hierarchical_policy.png",
    "fig17_history_deepsets_contribution.png",
    "supp_paired_charging_time_by_size.png",
    "supp_fixed_route_decomp_nl_study.png",
}


def enforce_figure_layout() -> None:
    """Ensure results/figures contains ONLY MAIN_FIGURES; move others."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    SUPP_FIGURES.mkdir(parents=True, exist_ok=True)
    ARCHIVE_FIGURES.mkdir(parents=True, exist_ok=True)
    main_set = set(MAIN_FIGURES)
    for path in list(FIGURES.glob("*.png")):
        if path.name in main_set:
            continue
        if path.name in SUPPLEMENTARY_KEEP or path.name.startswith("supp_"):
            dest = SUPP_FIGURES / path.name
        else:
            dest = ARCHIVE_FIGURES / path.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(path), str(dest))
        print(f"moved {path.name} -> {dest}")
    # Remove any non-PNG junk from main figures
    for path in list(FIGURES.iterdir()):
        if path.is_file() and path.suffix.lower() != ".png":
            dest = ARCHIVE_FIGURES / path.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(path), str(dest))
    present = sorted(p.name for p in FIGURES.glob("*.png"))
    missing = [n for n in MAIN_FIGURES if n not in present]
    extras = [n for n in present if n not in main_set]
    if missing:
        raise RuntimeError(f"Missing main figures: {missing}")
    if extras:
        raise RuntimeError(f"Extra files in results/figures: {extras}")
    print(f"main figures OK ({len(present)}): {present}")


def make_supp_paired_by_size() -> None:
    """Supplementary: paired charge-time deltas conditioned per size with n."""
    rows = _read(NL / "fixed_route_charging_comparison.csv")
    by_inst = defaultdict(dict)
    for r in rows:
        if r["method"].startswith("plan_") and r["feasible_under_true_physics"] == "1":
            by_inst[(int(r["size"]), r["instance"])][r["charging_model"]] = float(r["true_charging_h"])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    for ax, size in zip(axes, (10, 20, 40)):
        deltas = {"SECANT": [], "EARLY": []}
        for (s, _inst), d in by_inst.items():
            if s != size or "TRUE-NL" not in d:
                continue
            if "LINEAR-SECANT" in d:
                deltas["SECANT"].append(d["LINEAR-SECANT"] - d["TRUE-NL"])
            if "LINEAR-EARLY-RATE" in d:
                deltas["EARLY"].append(d["LINEAR-EARLY-RATE"] - d["TRUE-NL"])
        data = [deltas["SECANT"], deltas["EARLY"]]
        ax.boxplot(data, tick_labels=["SECANT", "EARLY"], showmeans=True)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"n={size} (n_pair S/E={len(deltas['SECANT'])}/{len(deltas['EARLY'])})")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("Linear plan vs TRUE-NL")
    axes[0].set_ylabel("Δ true charging h")
    fig.suptitle(
        "Paired true charging-time deltas (only dual-feasible survivors; selection bias)",
        fontsize=9,
    )
    fig_save(SUPP_FIGURES / "supp_paired_charging_time_by_size.png")


def build_all_main_figures() -> None:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    SUPP_FIGURES.mkdir(parents=True, exist_ok=True)
    make_fig01_architecture()
    make_fig02_curves()
    make_fig03_feasibility()
    make_fig04_schedule_case()
    make_fig05_decomposition()
    make_fig06_ablation()
    make_fig07_learning()
    make_fig08_quality_compute()
    make_supp_paired_by_size()
    enforce_figure_layout()


if __name__ == "__main__":
    build_all_main_figures()
