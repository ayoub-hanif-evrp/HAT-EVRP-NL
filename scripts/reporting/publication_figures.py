"""Generate the full main-paper figure set (~21 PNGs) and enforce layout.

Single source of truth for results/figures/. Supporting plots go to
results/supplementary/figures/; superseded files to archive/figures_legacy/.
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presentation_names import (  # noqa: E402
    METHOD_DISPLAY,
    METHOD_DISPLAY_NAMES,
    SEARCH_DISPLAY,
    method_color,
    method_label,
    pair_label,
    search_label,
)
from reporting_paths import (  # noqa: E402
    ARCHIVE_FIGURES,
    AUTH_BKS_GAPS,
    AUTH_N40_DECOMP,
    AUTH_N40_RECOVERY_FAIR,
    AUTH_N40_RECOVERY_PIPE,
    CHECKPOINTS,
    DATASET,
    FIGURES,
    LITERATURE_BKS,
    MAIN_FIGURES,
    NL_RAW,
    RAW,
    RESULTS,
    RL_RAW,
    SOLUTIONS,
    SUPP_FIGURES,
    SUPPLEMENTARY_FIGURES,
    TABLES,
    TRAIN_LOGS,
)

DPI = 300
CYCLE3 = RAW / "cycle3"

# Colorblind-friendly shared palette
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
    "plan_fail": "#BBBBBB",
    "false_feas": "#D55E00",
    "true_feas": "#009E73",
    "depot": "#000000",
    "cust": "#56B4E9",
    "fast": "#D55E00",
    "normal": "#E69F00",
    "slow": "#0072B2",
}
MC = {k: method_color(k) for k in ("H0", "H0_strong", "H1", "H2", "H3", "H4", "H4_open")}


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


def fig_save(path: Path, *, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("wrote", path)


# ---------------------------------------------------------------------------
# FIG 01 — Architecture
# ---------------------------------------------------------------------------
def make_fig01() -> None:
    """Architecture schematic with generous vertical gaps (no arrows through headings)."""
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(1.9, 12.6)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=7.0):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                lw=1.0,
                ec="#333",
                fc=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.1,
                color="#444",
                connectionstyle="arc3,rad=0",
            )
        )

    # --- A. Neural inference ---
    ax.text(0.25, 12.15, "A. Learned neural inference", fontsize=9, fontweight="bold")
    box(0.25, 10.55, 1.85, 1.15, "Montoya\nnode features", "#F0F0F0")
    box(2.35, 10.55, 1.85, 1.15, "Static node\nencoder", "#CFE2F3")
    box(4.45, 10.55, 1.95, 1.15, "Causal route\nhistory Trans.", "#CFE2F3")
    box(6.65, 10.55, 1.85, 1.15, "Dynamic EV\nstate fusion", "#CFE2F3")
    box(8.75, 10.55, 1.95, 1.15, "DeepSets fleet\n(optional)", "#CFE2F3")
    box(10.9, 10.55, 1.2, 1.15, "Hier.\npointer", "#CFE2F3", fs=6.5)
    for x1, x2 in [(2.1, 2.35), (4.2, 4.45), (6.4, 6.65), (8.5, 8.75), (10.7, 10.9)]:
        arrow(x1, 11.12, x2, 11.12)

    box(1.0, 8.85, 2.6, 1.05, "Customer sequences\n(learned routing)", "#FFF2CC")
    arrow(11.5, 10.55, 11.5, 9.4)
    arrow(11.5, 9.4, 3.6, 9.4)
    arrow(2.3, 8.85, 2.3, 7.55)

    # B title only above checker + validated columns (x≥4.3); arrow corridor is x≈2.3.
    ax.text(
        4.3,
        8.15,
        "B. Algorithmic charging & validation\n(non-differentiable)",
        fontsize=8.5,
        fontweight="bold",
        va="center",
        ha="left",
        linespacing=1.15,
    )
    box(0.55, 6.2, 3.4, 1.2, "frvcpy fixed-route\nnonlinear charging (FRVCP-NL)", "#FCE5CD")
    box(4.25, 6.2, 3.4, 1.2, "Independent Python checker\nSOC · duration · coverage", "#FCE5CD")
    box(7.95, 6.2, 3.85, 1.2, "Validated solution\nroutes + charge plan + Z_full", "#E8F5E9")
    arrow(3.95, 6.8, 4.25, 6.8)
    arrow(7.65, 6.8, 7.95, 6.8)

    # --- C. Training (clear gap below B boxes) ---
    ax.text(0.25, 5.55, "C. Training (weights only; never through frvcpy)", fontsize=9, fontweight="bold")
    box(0.25, 4.0, 2.6, 1.1, "Synthetic experts\n(greedy/LS + frvcpy)", "#E8F5E9")
    box(3.2, 4.0, 2.3, 1.1, "Behavior\ncloning", "#E8F5E9")
    box(5.85, 4.0, 2.5, 1.1, "Optional\nREINFORCE", "#E8F5E9")
    box(8.7, 4.0, 3.1, 1.1, "Validation-selected\ncheckpoint", "#E8F5E9")
    for x1, x2 in [(2.85, 3.2), (5.5, 5.85), (8.35, 8.7)]:
        arrow(x1, 4.55, x2, 4.55)

    # Compact color key only (interpretations belong in the manuscript caption).
    ax.text(
        0.25,
        2.6,
        "Blue = learned neural · Orange = algorithmic / non-differentiable · Green = data & outputs",
        fontsize=7.5,
    )
    ax.set_title("HAT-EVRP-NL hybrid architecture", fontsize=11, pad=6)
    fig_save(FIGURES / "fig01_hat_evrp_nl_architecture.png", tight=False)


# ---------------------------------------------------------------------------
# FIG 02 — Benchmark geography
# ---------------------------------------------------------------------------
def make_fig02() -> None:
    from data import parse_instance

    # Official Montoya location families tc0/tc1/tc2 (n=20 representatives)
    picks = [
        ("tc0c20s3cf2", "Family tc0 (n=20)"),
        ("tc1c20s3cf1", "Family tc1 (n=20)"),
        ("tc2c20s3cf0", "Family tc2 (n=20)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0))
    st_color = {"fast": C["fast"], "normal": C["normal"], "slow": C["slow"]}
    for ax, (name, title) in zip(axes, picks):
        path = next((DATASET).rglob(f"{name}.xml"), None)
        if path is None:
            ax.set_title(f"{title}\n(missing)")
            ax.axis("off")
            continue
        inst = parse_instance(path)
        # customers
        cx = [inst.nodes[i].x for i in inst.customers]
        cy = [inst.nodes[i].y for i in inst.customers]
        ax.scatter(cx, cy, c=C["cust"], s=18, label="Customer", zorder=2)
        # depot
        d = inst.nodes[0]
        ax.scatter([d.x], [d.y], c=C["depot"], s=70, marker="s", label="Depot", zorder=3)
        # stations by type
        seen = set()
        for s in inst.stations:
            st = inst.nodes[s].station_type
            lab = f"{st} station" if st not in seen else None
            seen.add(st)
            ax.scatter(
                [inst.nodes[s].x],
                [inst.nodes[s].y],
                c=st_color.get(st, "#333"),
                s=55,
                marker="^",
                label=lab,
                zorder=3,
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
    # Explicit full legend (all station types)
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["cust"], markersize=7, label="Customer"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C["depot"], markersize=7, label="Depot"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C["fast"], markersize=8, label="Fast station"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C["normal"], markersize=8, label="Normal station"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C["slow"], markersize=8, label="Slow station"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Montoya E-VRP-NL spatial layouts (official XML coordinates)", fontsize=10, y=1.14)
    fig_save(FIGURES / "fig02_montoya_instance_layouts.png")


# ---------------------------------------------------------------------------
# FIG 03 — Solved routes from JSON
# ---------------------------------------------------------------------------
def make_fig03() -> None:
    from data import parse_instance

    sol_path = SOLUTIONS / "H2_tc2c40s5cf3_hybrid.json"
    if not sol_path.exists():
        return
    sol = json.loads(sol_path.read_text(encoding="utf-8"))
    xml = next(DATASET.rglob(f"{sol['instance']}.xml"), None)
    if xml is None:
        return
    inst = parse_instance(xml)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    # background nodes
    ax.scatter(
        [inst.nodes[i].x for i in inst.customers],
        [inst.nodes[i].y for i in inst.customers],
        c=C["cust"],
        s=22,
        zorder=2,
        label="Customer",
    )
    ax.scatter([inst.nodes[0].x], [inst.nodes[0].y], c="k", s=80, marker="s", zorder=4, label="Depot")
    for s in inst.stations:
        st = inst.nodes[s].station_type
        ax.scatter(
            [inst.nodes[s].x],
            [inst.nodes[s].y],
            c={"fast": C["fast"], "normal": C["normal"], "slow": C["slow"]}.get(st, "#333"),
            s=50,
            marker="^",
            zorder=3,
            label=f"{st} station",
        )
    # dedupe legend later
    cmap = plt.cm.tab10
    for ri, stops in enumerate(sol["stops"]):
        xs = [inst.nodes[n].x for n, _ in stops]
        ys = [inst.nodes[n].y for n, _ in stops]
        color = cmap(ri % 10)
        ax.plot(xs, ys, "-", color=color, lw=1.6, alpha=0.9, label=f"EV route {ri + 1}")
        # mark charge visits
        for n, dq in stops:
            if dq is not None:
                ax.scatter([inst.nodes[n].x], [inst.nodes[n].y], s=90, facecolors="none", edgecolors=color, lw=1.5, zorder=5)
    # unique legend
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Validated hybrid solution: {sol['instance']} "
        f"({method_label('H2', short=True)}, {sol['full_h']:.2f} h)"
    )
    ax.grid(True, alpha=0.2)
    fig_save(FIGURES / "fig03_feasible_ev_routes.png")


# ---------------------------------------------------------------------------
# FIG 04 — Nonlinear curves
# ---------------------------------------------------------------------------
def make_fig04() -> None:
    from charging_surrogates import linear_early_rate_curve, linear_secant_curve
    from data import instance_files, parse_instance

    inst = None
    for prefer in ("c10", "c20", "c40", ""):
        for path in instance_files(DATASET):
            if prefer and prefer not in path.name:
                continue
            cand = parse_instance(path)
            if set(cand.curves) >= {"slow", "normal", "fast"}:
                inst = cand
                break
        if inst is not None:
            break
    if inst is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True)
    for ax, stype in zip(axes, ("slow", "normal", "fast")):
        true = inst.curves[stype]
        sec = linear_secant_curve(true)
        early = linear_early_rate_curve(true)
        ax.plot(true.times, [e / 1000 for e in true.energies], "o-", color=C["true"], ms=4, lw=1.6, label="TRUE-NL")
        ax.plot(sec.times, [e / 1000 for e in sec.energies], "s--", color=C["secant"], ms=4, lw=1.3, label="LINEAR-SECANT")
        ax.plot(early.times, [e / 1000 for e in early.energies], "^--", color=C["early"], ms=4, lw=1.3, label="LINEAR-EARLY-RATE")
        qmax = true.energies[-1] / 1000
        ax.axhspan(0.8 * qmax, qmax, color="#F2F2F2", zorder=0)
        ax.set_title(f"{stype.capitalize()}")
        ax.set_xlabel("Charging time (h)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Battery energy (kWh)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Montoya nonlinear charging vs linear surrogates ({inst.name})", fontsize=10, y=1.14)
    fig_save(FIGURES / "fig04_nonlinear_charging_curves.png")


# ---------------------------------------------------------------------------
# FIG 05 — Marginal charging time vs SOC
# ---------------------------------------------------------------------------
def make_fig05() -> None:
    from data import instance_files, parse_instance

    inst = None
    for prefer in ("c10", "c20", "c40", ""):
        for path in instance_files(DATASET):
            if prefer and prefer not in path.name:
                continue
            cand = parse_instance(path)
            if set(cand.curves) >= {"slow", "normal", "fast"}:
                inst = cand
                break
        if inst is not None:
            break
    if inst is None:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    colors = {"slow": C["slow"], "normal": C["normal"], "fast": C["fast"]}
    for stype in ("slow", "normal", "fast"):
        curve = inst.curves[stype]
        qmax = curve.energies[-1]
        xs, ys = [], []
        for i in range(len(curve.times) - 1):
            de_kwh = (curve.energies[i + 1] - curve.energies[i]) / 1000.0
            dt_min = (curve.times[i + 1] - curve.times[i]) * 60.0
            if de_kwh <= 1e-12:
                continue
            soc0 = curve.energies[i] / qmax
            soc1 = curve.energies[i + 1] / qmax
            slope = dt_min / de_kwh
            ax.hlines(slope, soc0, soc1, colors=colors[stype], lw=2.2)
            xs.append(0.5 * (soc0 + soc1))
            ys.append(slope)
        ax.plot(xs, ys, "o", color=colors[stype], ms=5, label=stype)
    ax.set_xlabel("SOC fraction (segment midpoint)")
    ax.set_ylabel("Marginal charging time (min / kWh)")
    ax.set_title(f"PWL charging slope vs SOC ({inst.name})")
    ax.legend(title="Technology", loc="upper left", frameon=True)
    ax.grid(True, alpha=0.25)
    fig_save(FIGURES / "fig05_charging_time_vs_soc.png")


# ---------------------------------------------------------------------------
# Feasibility helpers
# ---------------------------------------------------------------------------
def _feas_categories(rows: list[dict], *, fixed: bool) -> dict:
    out = defaultdict(lambda: {"true_feas": 0, "plan_fail": 0, "false_feas": 0, "n": 0})
    for r in rows:
        if fixed and not str(r.get("method", "")).startswith("plan_"):
            continue
        key = (int(r["size"]), r["charging_model"])
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


def _plot_feas_panel(ax, cats, title: str) -> None:
    """Single-axis feasibility panel (used when a shared axis is preferred)."""
    _plot_feas_grouped(ax, cats, title)


def _plot_feas_grouped(ax, cats, title: str) -> None:
    """Compact grouped bars: short model labels + bold size labels underneath."""
    sizes = [10, 20, 40]
    models = ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]
    model_short = {"TRUE-NL": "TRUE-NL", "LINEAR-SECANT": "Secant", "LINEAR-EARLY-RATE": "Early"}
    xpos_all = []
    model_labels = []
    group_centers = []
    width = 0.82
    idx = 0.0
    for s in sizes:
        group_x = []
        for model in models:
            tf = cats[(s, model)]["true_feas"]
            ff = cats[(s, model)]["false_feas"]
            pf = cats[(s, model)]["plan_fail"]
            ax.bar(idx, tf, width, color=C["true_feas"], edgecolor="white", lw=0.4)
            ax.bar(idx, ff, width, bottom=tf, color=C["false_feas"], edgecolor="white", lw=0.4)
            ax.bar(idx, pf, width, bottom=tf + ff, color=C["plan_fail"], edgecolor="white", lw=0.4)
            ax.text(idx, min(20.45, tf + ff + pf + 0.35), str(tf), ha="center", va="bottom", fontsize=7, fontweight="bold")
            xpos_all.append(idx)
            model_labels.append(model_short[model])
            group_x.append(idx)
            idx += 1.05
        group_centers.append((0.5 * (group_x[0] + group_x[-1]), s))
        idx += 0.55
    ax.set_xticks(xpos_all)
    ax.set_xticklabels(model_labels, fontsize=6.5)
    ax.set_ylim(0, 23.5)
    ax.axhline(20, color="#CCC", ls=":", lw=0.8)
    ax.set_ylabel("Instances (of 20)")
    ax.set_title(title, fontsize=9)
    # Draw size labels in a dedicated band below ticks (figure coordinates via axes).
    ax.set_xlabel("")
    for xc, s in group_centers:
        ax.text(xc, -2.6, f"n={s}", ha="center", va="top", fontsize=8, fontweight="bold", clip_on=False)
    ax.tick_params(axis="x", length=3, pad=1)


def _plot_feas_facets(axes, cats, title_prefix: str) -> None:
    """One small panel per size — clearest labeling for manuscript columns."""
    sizes = [10, 20, 40]
    models = ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]
    model_short = {"TRUE-NL": "TRUE-NL", "LINEAR-SECANT": "Secant", "LINEAR-EARLY-RATE": "Early"}
    for ax, s in zip(axes, sizes):
        x = np.arange(len(models))
        for i, model in enumerate(models):
            tf = cats[(s, model)]["true_feas"]
            ff = cats[(s, model)]["false_feas"]
            pf = cats[(s, model)]["plan_fail"]
            ax.bar(i, tf, 0.7, color=C["true_feas"], edgecolor="white", lw=0.4)
            ax.bar(i, ff, 0.7, bottom=tf, color=C["false_feas"], edgecolor="white", lw=0.4)
            ax.bar(i, pf, 0.7, bottom=tf + ff, color=C["plan_fail"], edgecolor="white", lw=0.4)
            ax.text(i, 20.4, str(tf), ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([model_short[m] for m in models], fontsize=7)
        ax.set_ylim(0, 23.2)
        ax.axhline(20, color="#CCC", ls=":", lw=0.8)
        ax.set_title(f"{title_prefix}n={s}", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Instances (of 20)")
        else:
            ax.set_yticklabels([])


def make_fig06() -> None:
    cats = _feas_categories(_read(NL_RAW / "fixed_route_charging_comparison.csv"), fixed=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), sharey=True)
    _plot_feas_facets(axes, cats, "")
    legend_elems = [
        Patch(fc=C["true_feas"], label="True-physics feasible"),
        Patch(fc=C["false_feas"], label="Planner-OK, true-physics fail"),
        Patch(fc=C["plan_fail"], label="Planner rejected / no stops"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Fixed-route feasibility under true Montoya physics (matched sequences)", fontsize=10, y=1.14)
    fig.subplots_adjust(top=0.82, wspace=0.12)
    fig_save(FIGURES / "fig06_fixed_route_charging_feasibility.png", tight=False)


def make_fig07() -> None:
    cats = _feas_categories(_read(NL_RAW / "end_to_end_charging_sensitivity.csv"), fixed=False)
    changes = _read(NL_RAW / "end_to_end_decision_changes.csv")
    fig = plt.figure(figsize=(9.4, 3.5))
    # Extra gap between feasibility facets and route-change panel.
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.0], wspace=0.38)
    gs_a = gs[0].subgridspec(1, 3, wspace=0.18)
    axes = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
    _plot_feas_facets(axes, cats, "")
    for i, s in enumerate((10, 20, 40)):
        axes[i].set_title(f"n={s}", fontsize=9)
    axes[0].set_ylabel("Instances (of 20)")

    ax = fig.add_subplot(gs[1])
    sizes = [10, 20, 40]
    x = np.arange(len(sizes))
    w = 0.35
    for i, model, col in ((0, "LINEAR-SECANT", C["secant"]), (1, "LINEAR-EARLY-RATE", C["early"])):
        rates = []
        for s in sizes:
            xs = [r for r in changes if int(r["size"]) == s and r["charging_model"] == model]
            rates.append(100.0 * sum(int(r["route_changed"]) for r in xs) / len(xs) if xs else 0)
        ax.bar(x + (i - 0.5) * w, rates, w, color=col, label=model.replace("LINEAR-", ""))
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={s}" for s in sizes])
    # Put ylabel on the right to avoid colliding with the n=40 panel.
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel("% instances with changed routes")
    ax.set_ylim(0, 105)
    ax.set_title("Route sequences vs TRUE-NL", fontsize=9)
    ax.legend(loc="upper right", frameon=True, fontsize=6.5)
    ax.grid(True, axis="y", alpha=0.25)

    legend_elems = [
        Patch(fc=C["true_feas"], label="True-physics feasible"),
        Patch(fc=C["false_feas"], label="Planner-OK, true-physics fail"),
        Patch(fc=C["plan_fail"], label="Planner rejected / no stops"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.42, 1.06))
    fig.suptitle(
        f"End-to-end charging-law misspecification "
        f"(frozen {method_label('H2', short=True)} weights)",
        fontsize=10,
        y=1.12,
    )
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.07, right=0.92)
    fig_save(FIGURES / "fig07_end_to_end_charging_feasibility.png", tight=False)


# ---------------------------------------------------------------------------
# FIG 08 — SOC schedule
# ---------------------------------------------------------------------------
def _soc_trace(inst, stops):
    from charging import charging_time

    soc = inst.battery_wh
    times = [0.0]
    socs = [1.0]
    t = 0.0
    events = []
    for index, (node, amount) in enumerate(stops):
        kind = inst.nodes[node].kind
        if amount is not None and amount > 1e-9 and kind == 2:
            curve = inst.curves[inst.nodes[node].station_type]
            q_out = min(soc + float(amount), inst.battery_wh)
            dt = charging_time(curve, soc, q_out)
            events.append({"t0": t, "t1": t + dt, "stype": inst.nodes[node].station_type, "dt": dt, "e": q_out - soc})
            t += dt
            soc = q_out
            times.append(t)
            socs.append(soc / inst.battery_wh)
        if index + 1 < len(stops):
            nxt = stops[index + 1][0]
            dt = inst.travel_hours[node][nxt]
            if kind == 1:
                dt += inst.service_times[node]
            t += dt
            soc = max(0.0, soc - inst.energy_wh[node][nxt])
            times.append(t)
            socs.append(soc / inst.battery_wh)
    return times, socs, events, t


def make_fig08() -> None:
    from data import parse_instance

    case = "tc0c10s2cf1"
    true_d = json.loads((NL_RAW / "schedules" / f"{case}_TRUE-NL.json").read_text(encoding="utf-8"))
    early_d = json.loads(
        (NL_RAW / "schedules" / f"{case}_LINEAR-EARLY-RATE_infeasible_under_true.json").read_text(encoding="utf-8")
    )
    xml = next(DATASET.rglob(f"{case}.xml"))
    inst = parse_instance(xml)
    route_idx = next(i for i, r in enumerate(early_d["stops"]) if any(dq is not None for _n, dq in r))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, stops, title, color in (
        (axes[0], true_d["stops"][route_idx], "A. TRUE-NL plan (feasible)", C["true"]),
        (axes[1], early_d["stops"][route_idx], "B. EARLY-RATE plan on TRUE-NL", C["early"]),
    ):
        times, socs, events, t_end = _soc_trace(inst, stops)
        ax.plot(times, socs, "-", color=color, lw=1.8)
        for ev in events:
            ax.axvspan(ev["t0"], ev["t1"], color=C["charge"], alpha=0.28)
            ax.text(0.5 * (ev["t0"] + ev["t1"]), 0.08, ev["stype"][0].upper(), ha="center", fontsize=6.5)
        ax.axvline(inst.max_route_hours, color="#333", ls="--", lw=1.0)
        if t_end > inst.max_route_hours + 1e-9:
            ax.axvspan(inst.max_route_hours, t_end, color=C["early"], alpha=0.12)
            ax.text(
                0.98,
                0.92,
                f"duration violation\n+{t_end - inst.max_route_hours:.2f} h",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=C["early"],
                bbox=dict(fc="white", ec=C["early"], alpha=0.92, pad=2),
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Elapsed route time (h)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("SOC fraction")
    axes[0].plot([], [], "--", color="#333", label=r"$T_{\max}$")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"Same customer sequence, different charge plans ({case}, route {route_idx + 1})", fontsize=10)
    fig_save(FIGURES / "fig08_soc_charging_schedule.png")
    # companion table
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_fig08_schedule_case.md").write_text(
        (TABLES / "table_fig04_schedule_case.md").read_text(encoding="utf-8")
        if (TABLES / "table_fig04_schedule_case.md").exists()
        else f"# Schedule case `{case}` route {route_idx}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# FIG 09 — Estimation error
# ---------------------------------------------------------------------------
def make_fig09() -> None:
    rows = _read(NL_RAW / "fixed_route_charging_comparison.csv")
    # Subset: planner produced stops (nominal charge available) — disclose selection
    by = defaultdict(list)
    for r in rows:
        if not r["method"].startswith("plan_"):
            continue
        if r["charging_model"] == "TRUE-NL":
            continue
        nom = r.get("nominal_planner_charge_h")
        true = r.get("true_charging_h")
        if nom in ("", None) or true in ("", None):
            continue
        # includes cases later true-infeasible if true_charging_h still computed
        err = float(nom) - float(true)  # predicted - realized; negative => underestimate
        by[(int(r["size"]), r["charging_model"])].append(err)

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.2), sharey=True)
    for ax, size in zip(axes, (10, 20, 40)):
        data = []
        labs = []
        for model, short in (("LINEAR-SECANT", "SECANT"), ("LINEAR-EARLY-RATE", "EARLY")):
            xs = by.get((size, model), [])
            data.append(xs)
            labs.append(f"{short}\nn={len(xs)}")
        if any(data):
            bp = ax.boxplot(data, tick_labels=labs, showmeans=True, patch_artist=True)
            for patch, col in zip(bp["boxes"], [C["secant"], C["early"]]):
                patch.set_facecolor(col)
                patch.set_alpha(0.45)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"n={size}")
        ax.set_xlabel("Surrogate")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Planner charge h − true charge h")
    fig.suptitle(
        "Charging-time estimation error (plans with recorded nominal & true charge hours)",
        fontsize=10,
    )
    fig_save(FIGURES / "fig09_linear_charging_estimation_error.png")


# ---------------------------------------------------------------------------
# FIG 10 — Station usage
# ---------------------------------------------------------------------------
def make_fig10() -> None:
    rows = _read(AUTH_N40_DECOMP)
    counts = {"H2": defaultdict(int), "H4": defaultdict(int)}
    energy = {"H2": [], "H4": []}
    charge_h = {"H2": [], "H4": []}
    for r in rows:
        if r.get("stage") != "hybrid_insert" or r.get("feasible") != "1":
            continue
        m = r["method"]
        if m not in counts:
            continue
        energy[m].append(float(r["energy_charged_kwh"]))
        charge_h[m].append(float(r["charging_h"]))
        for st in (r.get("station_types") or "").split("|"):
            if st:
                counts[m][st] += 1
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2))
    techs = ["slow", "normal", "fast"]
    x = np.arange(len(techs))
    w = 0.35
    ax = axes[0]
    for i, m, col in ((0, "H2", C["h2"]), (1, "H4", C["h4"])):
        vals = [counts[m].get(t, 0) for t in techs]
        ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=method_label(m, short=True))
    ax.set_xticks(x)
    ax.set_xticklabels(techs)
    ax.set_ylabel("Total visits (20 instances)")
    ax.set_title("A. Charger visits")
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    labs = [method_label(m, short=True) for m in ("H2", "H4")]
    e_means = [_mean(energy[m]) for m in ("H2", "H4")]
    ax.bar(labs, e_means, color=[C["h2"], C["h4"]], width=0.55)
    for lab, v in zip(labs, e_means):
        ax.text(lab, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("Mean energy charged (kWh)")
    ax.set_title("B. Energy charged")
    ax.set_ylim(0, max(e_means) * 1.18)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[2]
    c_means = [_mean(charge_h[m]) for m in ("H2", "H4")]
    ax.bar(labs, c_means, color=[C["h2"], C["h4"]], width=0.55)
    for lab, v in zip(labs, c_means):
        ax.text(lab, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Mean charging duration (h)")
    ax.set_title("C. Nonlinear charge time")
    ax.set_ylim(0, max(c_means) * 1.25)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("n=40 recovered hybrid_insert: frvcpy charging choices (TRUE-NL)", fontsize=10, y=1.02)
    fig_save(FIGURES / "fig10_station_usage_and_energy.png")


# ---------------------------------------------------------------------------
# FIG 11 — Objective decomposition
# ---------------------------------------------------------------------------
def make_fig11() -> None:
    rows = _read(AUTH_N40_DECOMP)
    by = defaultdict(list)
    for r in rows:
        if r.get("stage") == "hybrid_insert" and r.get("feasible") == "1":
            by[r["method"]].append(
                (float(r["travel_h"]), float(r["service_h"]), float(r["charging_h"]), float(r["full_h"]))
            )
    order = ["H2", "H4"]
    if all(by[m] for m in order):
        travel = [_mean([t for t, s, c, f in by[m]]) for m in order]
        service = [_mean([s for t, s, c, f in by[m]]) for m in order]
        charge = [_mean([c for t, s, c, f in by[m]]) for m in order]
        totals = [_mean([f for t, s, c, f in by[m]]) for m in order]
    else:
        travel, service, charge, totals = [33.745, 33.219], [20.0, 20.0], [3.060, 3.151], [56.806, 56.370]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    y = np.arange(len(order))
    ax.barh(y, travel, color=C["travel"], height=0.55, label="Travel")
    ax.barh(y, service, left=travel, color=C["service"], height=0.55, label="Service")
    left2 = [a + b for a, b in zip(travel, service)]
    ax.barh(y, charge, left=left2, color=C["charge"], height=0.55, label="Nonlinear charging")
    for yi, tot, tr, sv, ch in zip(y, totals, travel, service, charge):
        ax.text(tot + 0.35, yi, f"{tot:.3f} h", va="center", fontsize=8)
        ax.text(tr / 2, yi, f"{tr:.2f}", va="center", ha="center", fontsize=6.5, color="white")
        ax.text(tr + sv / 2, yi, f"{sv:.1f}", va="center", ha="center", fontsize=6.5, color="white")
        ax.text(tr + sv + ch / 2, yi, f"{ch:.2f}", va="center", ha="center", fontsize=6.5, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([method_label(m, wrap=True) for m in order])
    ax.set_xlabel("Mean hours (20 official n=40; recovered hybrid insert; Z_full)")
    ax.set_xlim(0, max(totals) + 8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, borderaxespad=0.0)
    ax.set_title("Objective decomposition under true nonlinear charging", fontsize=10, pad=22)
    ax.grid(True, axis="x", alpha=0.25)
    fig.subplots_adjust(top=0.82)
    fig_save(FIGURES / "fig11_objective_decomposition.png")


# ---------------------------------------------------------------------------
# FIG 12-21 — completed BC then RL matrix (bc_rl_full_v1)
# ---------------------------------------------------------------------------
def _full_figures():
    import bc_rl_full_figures as full
    return full


def make_fig12() -> None:
    _full_figures().make_fig12()


def make_fig13() -> None:
    _full_figures().make_fig13()


def make_fig14() -> None:
    _full_figures().make_fig14()


def make_fig15() -> None:
    _full_figures().make_fig15()


def make_fig16() -> None:
    _full_figures().make_fig16()


def make_fig17() -> None:
    _full_figures().make_fig17()


def make_fig18() -> None:
    _full_figures().make_fig18()


def make_fig19() -> None:
    _full_figures().make_fig19()


def make_fig20() -> None:
    _full_figures().make_fig20()


def make_fig21() -> None:
    _full_figures().make_fig21()


# ---------------------------------------------------------------------------
# Layout + contact sheet + catalog
# ---------------------------------------------------------------------------
def enforce_figure_layout() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SUPP_FIGURES.mkdir(parents=True, exist_ok=True)
    ARCHIVE_FIGURES.mkdir(parents=True, exist_ok=True)
    main_set = set(MAIN_FIGURES)
    supp_set = set(SUPPLEMENTARY_FIGURES)
    # Keep supplementary diagnostics out of the main-paper directory.
    for name in supp_set:
        main_path = FIGURES / name
        if main_path.exists():
            dest = SUPP_FIGURES / name
            if dest.exists():
                dest.unlink()
            shutil.move(str(main_path), str(dest))
            print(f"moved {name} -> supplementary")
    for path in list(FIGURES.glob("*.png")):
        if path.name in main_set:
            continue
        dest = ARCHIVE_FIGURES / path.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(path), str(dest))
        print(f"moved {path.name} -> {dest}")
    present = sorted(p.name for p in FIGURES.glob("*.png"))
    missing = [n for n in MAIN_FIGURES if n not in present]
    extras = [n for n in present if n not in main_set]
    if missing:
        raise RuntimeError(f"Missing main figures: {missing}")
    if extras:
        raise RuntimeError(f"Extra files in results/figures: {extras}")
    for name in supp_set:
        if not (SUPP_FIGURES / name).exists():
            raise RuntimeError(f"Missing supplementary figure: {name}")
    print(f"main figures OK ({len(present)}); supplementary OK ({len(supp_set)})")


def make_contact_sheet() -> None:
    from PIL import Image

    paths = [FIGURES / n for n in MAIN_FIGURES if (FIGURES / n).exists()]
    if not paths:
        return
    cols = 4
    rows = math.ceil(len(paths) / cols)
    thumb_w, thumb_h = 320, 220
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (255, 255, 255))
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb_w - 10, thumb_h - 28))
        x = (i % cols) * thumb_w + (thumb_w - im.width) // 2
        y = (i // cols) * thumb_h + 18
        sheet.paste(im, (x, y))
    out = SUPP_FIGURES / "figure_contact_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, dpi=(150, 150))
    print("wrote", out)


def build_all_main_figures() -> None:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    makers = [
        make_fig01,
        make_fig02,
        make_fig03,
        make_fig04,
        make_fig05,
        make_fig06,
        make_fig07,
        make_fig08,
        make_fig09,
        make_fig10,
        make_fig11,
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
    ]
    for fn in makers:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — surface per-figure failures
            print(f"ERROR in {fn.__name__}: {exc}")
            raise
    enforce_figure_layout()
    try:
        make_contact_sheet()
    except Exception as exc:  # noqa: BLE001
        print("WARN contact sheet:", exc)


if __name__ == "__main__":
    build_all_main_figures()
