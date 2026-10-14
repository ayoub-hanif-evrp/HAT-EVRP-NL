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

from presentation_names import METHOD_DISPLAY, SEARCH_DISPLAY, method_label  # noqa: E402
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
MC = {"H0_strong": C["h0"], "H1": C["h1"], "H2": C["h2"], "H3": C["h3"], "H4": C["h4"]}


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
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=7.2):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                lw=1.0, ec="#333", fc=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.05, color="#444")
        )

    ax.text(0.2, 10.5, "A. Inference path", fontsize=9, fontweight="bold")
    box(0.2, 9.0, 2.0, 1.1, "Montoya XML\nfeatures", "#F0F0F0")
    box(2.5, 9.0, 1.9, 1.1, "Static node\nencoder", "#CFE2F3")
    box(4.7, 9.0, 2.1, 1.1, "Causal history\nTransformer", "#CFE2F3")
    box(7.1, 9.0, 2.0, 1.1, "Dynamic EV\nstate fusion", "#CFE2F3")
    box(9.4, 9.0, 2.3, 1.1, "DeepSets fleet\n(optional)", "#CFE2F3")
    for a, b in [(2.2, 2.5), (4.4, 4.7), (6.8, 7.1), (9.1, 9.4)]:
        arrow(a, 9.55, b, 9.55)

    box(3.5, 7.1, 2.6, 1.1, "Hierarchical\nroute/customer pointer", "#CFE2F3")
    box(6.5, 7.1, 2.4, 1.1, "Customer sequences\n(learned)", "#FFF2CC")
    arrow(8.1, 9.0, 4.8, 8.2)
    arrow(10.5, 9.0, 7.8, 8.2)
    arrow(6.1, 7.65, 6.5, 7.65)

    ax.text(0.2, 6.4, "B. Exact charging + validation (not learned)", fontsize=9, fontweight="bold")
    box(0.2, 4.9, 3.2, 1.2, "frvcpy FixedRouteOracle\nnonlinear charging (exact FRVCP)", "#FCE5CD")
    box(3.8, 4.9, 3.0, 1.2, "Python physical checker\nSOC / duration / coverage", "#FCE5CD")
    box(7.2, 4.9, 4.5, 1.2, "Validated solution\nroutes + charge schedule + Z_full", "#E8F5E9")
    arrow(5.5, 7.1, 1.8, 6.1)
    arrow(3.4, 5.5, 3.8, 5.5)
    arrow(6.8, 5.5, 7.2, 5.5)

    ax.text(0.2, 4.2, "C. Training path", fontsize=9, fontweight="bold")
    box(0.2, 2.7, 2.5, 1.1, "Synthetic experts\n(greedy/LS + frvcpy)", "#E8F5E9")
    box(3.1, 2.7, 2.2, 1.1, "Behavior\ncloning", "#E8F5E9")
    box(5.7, 2.7, 2.4, 1.1, "Optional\nREINFORCE", "#E8F5E9")
    box(8.5, 2.7, 3.2, 1.1, "Validation-selected\ncheckpoint (.pt)", "#E8F5E9")
    for a, b in [(2.7, 3.1), (5.3, 5.7), (8.1, 8.5)]:
        arrow(a, 3.25, b, 3.25)

    ax.text(0.2, 1.6, "Blue = learned neural  ·  Orange = algorithmic / non-differentiable  ·  Green = data & outputs", fontsize=7.5)
    ax.text(0.2, 0.85, "H1–H4 ablate history Transformer and DeepSets. Charging gradients never flow through frvcpy.", fontsize=7.5)
    ax.set_title("HAT-EVRP-NL hybrid architecture", fontsize=11, pad=8)
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
    ax.set_title(f"Validated hybrid solution: {sol['instance']} (H2, {sol['full_h']:.2f} h)")
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
    fig.text(
        0.5,
        -0.02,
        "Higher SOC ⇒ higher min/kWh on concave Montoya PWL curves (XML segment slopes).",
        ha="center",
        fontsize=7.5,
        color="#444",
    )
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
    sizes = [10, 20, 40]
    models = ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]
    x = np.arange(len(sizes))
    width = 0.26
    for i, model in enumerate(models):
        tf = [cats[(s, model)]["true_feas"] for s in sizes]
        ff = [cats[(s, model)]["false_feas"] for s in sizes]
        pf = [cats[(s, model)]["plan_fail"] for s in sizes]
        xpos = x + (i - 1) * width
        ax.bar(xpos, tf, width, color=C["true_feas"])
        ax.bar(xpos, ff, width, bottom=tf, color=C["false_feas"])
        ax.bar(xpos, pf, width, bottom=[a + b for a, b in zip(tf, ff)], color=C["plan_fail"])
        for xi, v in zip(xpos, tf):
            ax.text(xi, 20.3, str(v), ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={s}" for s in sizes])
    ax.set_ylim(0, 22.8)
    ax.axhline(20, color="#CCC", ls=":", lw=0.8)
    ax.set_ylabel("Instances (of 20)")
    ax.set_title(title, fontsize=9)
    ax.text(0.5, -0.18, "Bars L→R: TRUE-NL · SECANT · EARLY", transform=ax.transAxes, ha="center", fontsize=7)


def make_fig06() -> None:
    cats = _feas_categories(_read(NL_RAW / "fixed_route_charging_comparison.csv"), fixed=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    _plot_feas_panel(ax, cats, "Fixed-route charging optimization (matched sequences)")
    legend_elems = [
        Patch(fc=C["true_feas"], label="True-physics feasible"),
        Patch(fc=C["false_feas"], label="Planner-OK, true-physics fail"),
        Patch(fc=C["plan_fail"], label="Planner rejected / no stops"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Fixed-route feasibility under true Montoya physics", fontsize=10, y=1.12)
    fig_save(FIGURES / "fig06_fixed_route_charging_feasibility.png")


def make_fig07() -> None:
    cats = _feas_categories(_read(NL_RAW / "end_to_end_charging_sensitivity.csv"), fixed=False)
    changes = _read(NL_RAW / "end_to_end_decision_changes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), gridspec_kw={"width_ratios": [1.35, 1.0]})
    _plot_feas_panel(axes[0], cats, "A. True-physics feasibility")
    # route change rates
    ax = axes[1]
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
    ax.set_ylabel("% instances with changed routes")
    ax.set_ylim(0, 100)
    ax.set_title("B. Route sequences vs TRUE-NL", fontsize=9)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.25)
    legend_elems = [
        Patch(fc=C["true_feas"], label="True-physics feasible"),
        Patch(fc=C["false_feas"], label="Planner-OK, true-physics fail"),
        Patch(fc=C["plan_fail"], label="Planner rejected / no stops"),
    ]
    fig.legend(handles=legend_elems, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.05, 1.08))
    fig.suptitle("End-to-end routing: charging-law misspecification (frozen H2 weights)", fontsize=10, y=1.14)
    fig_save(FIGURES / "fig07_end_to_end_charging_feasibility.png")


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
    fig.text(
        0.5,
        -0.04,
        "Negative = linear planner underestimates true nonlinear charging time. Sample sizes = paired recorded hours (not all 20).",
        ha="center",
        fontsize=7.5,
    )
    fig_save(FIGURES / "fig09_linear_charging_estimation_error.png")


# ---------------------------------------------------------------------------
# FIG 10 — Station usage
# ---------------------------------------------------------------------------
def make_fig10() -> None:
    rows = _read(AUTH_N40_DECOMP)
    # hybrid_insert feasible H2/H4
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
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    # visits by tech
    ax = axes[0]
    techs = ["slow", "normal", "fast"]
    x = np.arange(len(techs))
    w = 0.35
    for i, m, col in ((0, "H2", C["h2"]), (1, "H4", C["h4"])):
        vals = [counts[m].get(t, 0) for t in techs]
        ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=method_label(m, short=True))
    ax.set_xticks(x)
    ax.set_xticklabels(techs)
    ax.set_ylabel("Total charger visits (20 instances)")
    ax.set_title("A. Visits by technology")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    # energy / charge hours
    ax = axes[1]
    labs = [method_label(m, short=True) for m in ("H2", "H4")]
    e_means = [_mean(energy[m]) for m in ("H2", "H4")]
    c_means = [_mean(charge_h[m]) for m in ("H2", "H4")]
    x = np.arange(2)
    ax.bar(x - 0.18, e_means, 0.35, color="#56B4E9", label="Energy (kWh)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, c_means, 0.35, color=C["charge"], label="Charge time (h)")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel("Mean energy charged (kWh)")
    ax2.set_ylabel("Mean nonlinear charge h")
    ax.set_title("B. Energy and charging duration")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)
    fig.suptitle("n=40 recovered hybrid_insert: frvcpy charging choices (TRUE-NL)", fontsize=10)
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
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(order))
    ax.barh(y, travel, color=C["travel"], height=0.5, label="Travel")
    ax.barh(y, service, left=travel, color=C["service"], height=0.5, label="Service")
    left2 = [a + b for a, b in zip(travel, service)]
    ax.barh(y, charge, left=left2, color=C["charge"], height=0.5, label="Nonlinear charging")
    for yi, tot in zip(y, totals):
        ax.text(tot + 0.4, yi, f"{tot:.3f} h", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([method_label(m, short=True) for m in order])
    ax.set_xlabel("Mean hours (20 official n=40; recovered hybrid_insert)")
    ax.set_xlim(0, max(totals) + 10)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=False)
    ax.set_title("Objective decomposition under true nonlinear charging")
    ax.grid(True, axis="x", alpha=0.25)
    fig.text(
        0.01,
        -0.02,
        "H4 improves mainly via travel, not charging (same frvcpy optimizer).",
        fontsize=7.5,
        color="#444",
    )
    fig_save(FIGURES / "fig11_objective_decomposition.png")


# ---------------------------------------------------------------------------
# FIG 12 — Architecture ablation
# ---------------------------------------------------------------------------
def make_fig12() -> None:
    methods = ["H0_strong", "H1", "H2", "H3", "H4"]
    seed_means = {m: [] for m in methods}
    for seed in (42, 43, 44):
        rows = _read(CYCLE3 / f"compare_size10_fairls_final_seed{seed}_c3.csv")
        by = defaultdict(list)
        for r in rows:
            if r["method"] in methods and r["feasible"] in ("1", "True", 1):
                by[r["method"]].append(float(r["objective_h"]))
        for m in methods:
            if by[m]:
                seed_means[m].append(_mean(by[m]))
    means = [_mean(seed_means[m]) for m in methods]
    errs = [_stdev(seed_means[m]) for m in methods]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    y = np.arange(len(methods))
    for yi, m, mean, err in zip(y, methods, means, errs):
        ax.errorbar(mean, yi, xerr=err, fmt="o", color=MC[m], ecolor="#666", elinewidth=1.1, capsize=3, ms=8)
        for sm in seed_means[m]:
            ax.plot(sm, yi, "|", color=MC[m], ms=9, alpha=0.5)
        # labels to the right of the error bar tip, clear of markers
        ax.text(mean + err + 0.04, yi, f"{mean:.3f}±{err:.3f}", va="center", fontsize=7, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_DISPLAY[m] for m in methods])
    ax.set_xlabel("Mean full duration (h); seed means ± std (3 seeds × 20 instances)")
    lo = min(m - e for m, e in zip(means, errs)) - 0.15
    hi = max(m + e for m, e in zip(means, errs)) + 0.55
    ax.set_xlim(lo, hi)
    ax.axvline(means[methods.index("H2")], color=C["h2"], ls=":", lw=0.9, alpha=0.5)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("Matched n=10 Cycle-3 fair one-shot LS")
    fig.text(
        0.5,
        -0.02,
        "History helps vs state-only; DeepSets does not consistently beat history alone.",
        ha="center",
        fontsize=7.5,
        color="#444",
    )
    fig_save(FIGURES / "fig12_neural_architecture_ablation.png")


# ---------------------------------------------------------------------------
# FIG 13 — Paired effects
# ---------------------------------------------------------------------------
def make_fig13() -> None:
    # Pair by (instance, seed): H2-H1, H3-H1, H4-H2
    pairs = {"H2−H1": [], "H3−H1": [], "H4−H2": []}
    for seed in (42, 43, 44):
        rows = _read(CYCLE3 / f"compare_size10_fairls_final_seed{seed}_c3.csv")
        by = defaultdict(dict)
        for r in rows:
            if r["feasible"] in ("1", "True", 1) and r["method"] in {"H1", "H2", "H3", "H4"}:
                by[r["instance"]][r["method"]] = float(r["objective_h"])
        for inst, d in by.items():
            if "H1" in d and "H2" in d:
                pairs["H2−H1"].append(d["H2"] - d["H1"])
            if "H1" in d and "H3" in d:
                pairs["H3−H1"].append(d["H3"] - d["H1"])
            if "H2" in d and "H4" in d:
                pairs["H4−H2"].append(d["H4"] - d["H2"])
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    keys = list(pairs)
    data = [pairs[k] for k in keys]
    bp = ax.boxplot(data, tick_labels=[f"{k}\nn={len(pairs[k])}" for k in keys], showmeans=True, patch_artist=True)
    for patch, col in zip(bp["boxes"], [C["h2"], C["h3"], C["h4"]]):
        patch.set_facecolor(col)
        patch.set_alpha(0.4)
    ax.axhline(0, color="k", lw=1.0)
    # mean±CI-ish: mean±1.96*se with note that pairs are instance×seed (not independent layouts)
    for i, k in enumerate(keys):
        xs = pairs[k]
        m = _mean(xs)
        se = _stdev(xs) / math.sqrt(len(xs)) if xs else 0
        ax.errorbar(i + 1, m, yerr=1.96 * se, fmt="D", color="#222", ms=5, capsize=3, zorder=4)
    ax.set_ylabel("Δ full duration (h); negative = improvement")
    ax.set_title("Paired architecture effects (n=10 fair-LS; instance×seed pairs)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.text(
        0.5,
        -0.03,
        "Diamonds: mean ± 1.96·SE over pairs (correlated across seeds; not independent layouts).",
        ha="center",
        fontsize=7.5,
    )
    fig_save(FIGURES / "fig13_history_deepsets_paired_effect.png")


# ---------------------------------------------------------------------------
# FIG 14 — BKS gaps
# ---------------------------------------------------------------------------
def make_fig14() -> None:
    rows = _read(AUTH_BKS_GAPS)
    # Also show n20/n40 if available from other corrected sources — use n10 primary
    methods = ["H0_strong", "H1", "H2", "H3", "H4"]
    gaps = []
    for m in methods:
        r = next((x for x in rows if x["method"] == m), None)
        gaps.append(float(r["mean_gap_lit_pct"]) if r else float("nan"))
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    y = np.arange(len(methods))
    ax.barh(y, gaps, color=[MC[m] for m in methods], height=0.55)
    for yi, g in zip(y, gaps):
        ax.text(g + 0.08, yi, f"{g:.2f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_DISPLAY[m] for m in methods])
    ax.set_xlabel("Mean literature gap %  [(Z_lit − BKS)/BKS]")
    ax.set_title("n=10 fair-LS vs Froger Table 13 BKS (Z_lit = full − service)")
    ax.set_xlim(0, max(gaps) + 1.5)
    ax.grid(True, axis="x", alpha=0.25)
    fig.text(
        0.01,
        -0.02,
        "Protocol: Cycle-3 fair-LS; 20 instances; lit objective excludes fixed service. Source: IMPROVEMENT_bks_gaps_corrected.csv",
        fontsize=7,
        color="#444",
    )
    fig_save(FIGURES / "fig14_literature_bks_gap.png")


# ---------------------------------------------------------------------------
# FIG 15 — Generalization
# ---------------------------------------------------------------------------
def make_fig15() -> None:
    # From FINAL tables / fairls aggregates
    final = {
        10: _read(CYCLE3 / "FINAL_n10_fairls.csv"),
        20: _read(CYCLE3 / "FINAL_n20_fairls.csv"),
        40: _read(CYCLE3 / "FINAL_n40_fairls.csv"),
        80: _read(CYCLE3 / "FINAL_n80_fairls.csv"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))
    ax = axes[0]
    for m, col in (("H2", C["h2"]), ("H4", C["h4"])):
        xs, ys = [], []
        for n in (10, 20, 40, 80):
            row = next((r for r in final[n] if r.get("method") == m), None)
            if not row:
                continue
            xs.append(n)
            ys.append(float(row["mean_h"]))
        ax.plot(xs, ys, "o-", color=col, lw=1.6, label=method_label(m, short=True))
    ax.set_xlabel("Test size n (models trained on synthetic n=10)")
    ax.set_ylabel("Mean full duration (h)")
    ax.set_title("A. Fair-LS duration vs size")
    ax.legend(frameon=True, fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.text(0.98, 0.02, "n=80: 3 instances only", transform=ax.transAxes, ha="right", fontsize=7, color="#D55E00")

    ax = axes[1]
    # runtime / oracle if present
    for m, col in (("H2", C["h2"]), ("H4", C["h4"])):
        xs, ys = [], []
        for n in (10, 20, 40, 80):
            row = next((r for r in final[n] if r.get("method") == m), None)
            if not row or row.get("mean_runtime_s") in ("", None):
                continue
            xs.append(n)
            ys.append(float(row["mean_runtime_s"]))
        if xs:
            ax.plot(xs, ys, "s-", color=col, lw=1.6, label=method_label(m, short=True))
    ax.set_xlabel("Test size n")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("B. Fair-LS runtime vs size")
    ax.legend(frameon=True, fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.suptitle("Generalization of n10-trained checkpoints (fair-LS transfer)", fontsize=10)
    fig_save(FIGURES / "fig15_generalization_scalability.png")


# ---------------------------------------------------------------------------
# FIG 16 — Search stages
# ---------------------------------------------------------------------------
def make_fig16() -> None:
    rows = _read(AUTH_N40_RECOVERY_PIPE)
    stages = ["construct", "basic_ls", "strong_ls", "classical_ruin", "neural_insert", "hybrid_insert"]
    # Independent branches from same construction — show as dots not a chain implying dominance
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for m, marker in (("H2", "o"), ("H4", "s")):
        means = []
        for st in stages:
            xs = [float(r["objective_h"]) for r in rows if r["method"] == m and r["stage"] == st and r["feasible"] == "1"]
            means.append(_mean(xs) if xs else float("nan"))
        ax.plot(range(len(stages)), means, marker=marker, lw=1.4, color=MC[m], label=method_label(m, short=True))
        for i, v in enumerate(means):
            if math.isfinite(v):
                ax.text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=6.5, color=MC[m])
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([SEARCH_DISPLAY.get(s, s) for s in stages], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Mean full duration (h)")
    ax.set_title("n=40 recovered pipeline stages (20 instances; independent search modes)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.25)
    ax.invert_yaxis()
    fig.text(
        0.5,
        -0.08,
        "Modes are alternative post-construction searches (not a forced sequential chain). Lower is better.",
        ha="center",
        fontsize=7.5,
    )
    fig_save(FIGURES / "fig16_search_improvement_stages.png")


# ---------------------------------------------------------------------------
# FIG 17 — Quality vs compute
# ---------------------------------------------------------------------------
def make_fig17() -> None:
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
    points = []
    # basic_ls incremental from pipeline
    by_inst = defaultdict(dict)
    for r in pipe:
        if r["method"] == "H2" and r["feasible"] == "1":
            by_inst[r["instance"]][r["stage"]] = r
    objs, orcs, rts = [], [], []
    for d in by_inst.values():
        if "basic_ls" in d and "construct" in d:
            objs.append(float(d["basic_ls"]["objective_h"]))
            orcs.append(float(d["basic_ls"]["oracle_solver_executions"]) - float(d["construct"]["oracle_solver_executions"]))
            rts.append(float(d["basic_ls"]["runtime_s"]) - float(d["construct"]["runtime_s"]))
    if objs:
        points.append(("basic_ls", _mean(objs), _mean(orcs), _mean(rts)))
    for stage in stages:
        if stage == "basic_ls":
            continue
        xs = [r for r in fair if r["method"] == "H2" and r["mode"] == stage and r["feasible"] == "1"]
        if not xs:
            continue
        mean_h = _mean([float(r["objective_h"]) for r in xs])
        mean_orc = _mean([float(r["oracle_solver_executions"]) for r in xs])
        rts = [float(r["runtime_search_s"]) for r in xs if float(r.get("runtime_search_s") or 0) > 0]
        points.append((stage, mean_h, mean_orc, _mean(rts) if rts else float("nan")))

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for stage, mean_h, mean_orc, mean_rt in points:
        rt_lab = f"{mean_rt:.2f}s" if math.isfinite(mean_rt) else "rt n/a"
        ax.scatter(mean_orc, mean_h, s=100, marker=markers[stage], color=colors[stage], zorder=3, label=f"{SEARCH_DISPLAY[stage]} ({rt_lab})")
        ax.annotate(f"{mean_h:.2f}", (mean_orc, mean_h), textcoords="offset points", xytext=(7, 5), fontsize=7)
    ax.set_xlabel("Mean stage-level frvcpy oracle executions")
    ax.set_ylabel("Mean full duration (h)")
    ax.set_title("n=40 H2: quality vs stage-level compute (20 inst., seed 42)")
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.text(0.5, -0.02, "Unequal realized budgets — not an equal-budget experiment.", ha="center", fontsize=7.5, color="#444")
    fig_save(FIGURES / "fig17_quality_vs_compute.png")


# ---------------------------------------------------------------------------
# FIG 18 — BC learning curves
# ---------------------------------------------------------------------------
def make_fig18() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for model, color in (("H1", C["h1"]), ("H2", C["h2"]), ("H3", C["h3"]), ("H4", C["h4"])):
        for seed in (42, 43, 44):
            rows = _read(TRAIN_LOGS / f"training_{model}_n10_seed{seed}_c3.csv")
            bc = [r for r in rows if r.get("stage") == "bc" and r.get("validation_objective")]
            if not bc:
                continue
            steps = [int(r["step"]) for r in bc]
            vals = [float(r["validation_objective"]) for r in bc]
            ax.plot(steps, vals, "o-", color=color, alpha=0.25 if seed != 42 else 0.95, lw=1.2 if seed == 42 else 0.8, ms=3.5, label=model if seed == 42 else None)
            # mark best
            if seed == 42:
                bi = int(np.argmin(vals))
                ax.scatter([steps[bi]], [vals[bi]], marker="*", s=80, color=color, zorder=4)
    ax.set_xlabel("BC evaluation step")
    ax.set_ylabel("Synthetic-validation full h")
    ax.set_title("Behavior-cloning validation (Cycle-3; markers = logged eval points)")
    ax.legend(title="Seed 42 bold; ★ = best logged", frameon=True, fontsize=7)
    ax.grid(True, alpha=0.25)
    fig_save(FIGURES / "fig18_behavior_cloning_learning_curves.png")


# ---------------------------------------------------------------------------
# FIG 19 — BC vs RL paired
# ---------------------------------------------------------------------------
def make_fig19() -> None:
    summary = _read(RL_RAW / "bc_vs_rl_official_n10_summary.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = 0
    yticks = []
    ylabels = []
    for model in ("H2", "H4"):
        for seed in ("42", "43", "44"):
            bc = next(r for r in summary if r["model"] == model and r["seed"] == seed and r["tag"] == "bc_only_rq4")
            rl = next(r for r in summary if r["model"] == model and r["seed"] == seed and r["tag"] == "rl_from_bc_rq4")
            b, r = float(bc["mean_full_h"]), float(rl["mean_full_h"])
            col = "#009E73" if r < b else "#D55E00"
            ax.plot([b, r], [y, y], "-", color=col, lw=2.0)
            ax.plot(b, y, "o", color=C["bc"], ms=8)
            ax.plot(r, y, "s", color=C["rl"], ms=7)
            ax.text(max(b, r) + 0.15, y, f"Δ={r - b:+.2f}", va="center", fontsize=7, color=col)
            yticks.append(y)
            ylabels.append(f"{model} s{seed}")
            y += 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Mean full duration (h); official n=10 construction-only, 20/20 feasible")
    ax.set_title("Controlled RQ4: BC-only → RL-from-BC (paired slopes)")
    ax.plot([], [], "o", color=C["bc"], label="BC-only")
    ax.plot([], [], "s", color=C["rl"], label="RL-from-BC")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.text(
        0.5,
        -0.03,
        "H2: RL better 1/3 seeds · H4: RL better 3/3 · Distinct from Cycle-3 synth-val RL instability.",
        ha="center",
        fontsize=7.5,
    )
    fig_save(FIGURES / "fig19_bc_vs_rl_checkpoint_quality.png")


# ---------------------------------------------------------------------------
# FIG 20 — Reward dynamics
# ---------------------------------------------------------------------------
def make_fig20() -> None:
    window = 5
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), sharey=True)
    for ax, model, color in ((axes[0], "H2", C["h2"]), (axes[1], "H4", C["h4"])):
        series = []
        for seed in (42, 43, 44):
            rows = _read(RL_RAW / f"reward_{model}_n10_seed{seed}_rl_from_bc_rq4.csv")
            if not rows:
                continue
            steps = [int(r["step"]) for r in rows]
            rew = [float(r["mean_sampled_reward"]) for r in rows]
            base = [float(r["baseline_reward"]) for r in rows]
            ax.plot(steps, rew, color=color, alpha=0.2, lw=0.8)
            ax.plot(steps, base, color="#999", alpha=0.2, lw=0.8, ls="--")
            smooth = [float(np.median(rew[max(0, i - window + 1) : i + 1])) for i in range(len(rew))]
            series.append((steps, smooth, base))
        if series:
            n = min(len(s[1]) for s in series)
            steps = series[0][0][:n]
            ax.plot(steps, np.mean([s[1][:n] for s in series], axis=0), color=color, lw=2.2, label="Reward (roll. med.)")
            ax.plot(steps, np.mean([s[2][:n] for s in series], axis=0), color="#666", lw=1.5, ls="--", label="Greedy baseline")
        ax.set_title(model)
        ax.set_xlabel("RL step")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("R (−Z_full or −100)")
    fig.suptitle(f"REINFORCE reward dynamics (rolling median w={window}; faint = seeds)", fontsize=10)
    fig.text(0.5, -0.03, "R = −Z_full if feasible; R = −100 if infeasible. Source: reward_*_rl_from_bc_rq4.csv", ha="center", fontsize=7.5)
    fig_save(FIGURES / "fig20_reinforce_reward_dynamics.png")


# ---------------------------------------------------------------------------
# FIG 21 — RL stability diagnostics
# ---------------------------------------------------------------------------
def make_fig21() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2), sharex=True)
    for model, color in (("H2", C["h2"]), ("H4", C["h4"])):
        # aggregate mean across seeds
        mats = {"grad_norm": [], "mean_advantage": [], "entropy": [], "loss": []}
        steps_ref = None
        for seed in (42, 43, 44):
            rows = _read(RL_RAW / f"reward_{model}_n10_seed{seed}_rl_from_bc_rq4.csv")
            if not rows:
                continue
            steps_ref = [int(r["step"]) for r in rows]
            mats["grad_norm"].append([float(r["grad_norm"]) for r in rows])
            mats["mean_advantage"].append([float(r["mean_advantage"]) for r in rows])
            mats["entropy"].append([float(r["entropy"]) for r in rows])
            mats["loss"].append([float(r["loss"]) for r in rows])
        if not steps_ref:
            continue
        panels = [
            (axes[0, 0], "grad_norm", "Gradient norm"),
            (axes[0, 1], "mean_advantage", "Advantage (R − baseline)"),
            (axes[1, 0], "entropy", "Policy entropy"),
            (axes[1, 1], "loss", "Policy-gradient loss"),
        ]
        for ax, key, title in panels:
            arr = np.array(mats[key])
            mean = arr.mean(axis=0)
            ax.plot(steps_ref, mean, color=color, lw=1.8, label=model)
            ax.fill_between(steps_ref, arr.min(axis=0), arr.max(axis=0), color=color, alpha=0.12)
            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=True, fontsize=7)
    axes[1, 0].set_xlabel("RL step")
    axes[1, 1].set_xlabel("RL step")
    fig.suptitle("RL training diagnostics (mean across seeds; band = min–max)", fontsize=10)
    fig.text(
        0.5,
        -0.02,
        "Diagnostic only — co-occurrence of large grads with unstable rewards is correlational, not proven causal.",
        ha="center",
        fontsize=7.5,
        color="#444",
    )
    fig_save(FIGURES / "fig21_rl_training_stability.png")


# ---------------------------------------------------------------------------
# Layout + contact sheet + catalog
# ---------------------------------------------------------------------------
def enforce_figure_layout() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SUPP_FIGURES.mkdir(parents=True, exist_ok=True)
    ARCHIVE_FIGURES.mkdir(parents=True, exist_ok=True)
    main_set = set(MAIN_FIGURES)
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
    print(f"main figures OK ({len(present)})")


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
