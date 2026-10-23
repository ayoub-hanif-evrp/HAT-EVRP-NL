"""Figures A, B, and E from saved charging curves, plans, and schedules.

Figure D is written only after the frozen-checkpoint decomposition exists.
No curve or route is invented.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging import charging_time, check_route
from charging_surrogates import instance_with_charging_model, linear_early_rate_curve, linear_secant_curve
from data import instance_files, parse_instance

OUT_FIG = ROOT / "results" / "final" / "figures"
OUT_TAB = ROOT / "results" / "tables" / "final"
SCHEDULES = ROOT / "results" / "raw" / "nonlinear_charging" / "schedules"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 200,
        }
    )


def _instance_by_name(name: str):
    for path in instance_files(ROOT / "dataset"):
        if path.stem == name or name in path.name:
            instance = parse_instance(path)
            if instance.name == name:
                return instance
    raise SystemExit(f"no benchmark file named {name}")


def figure_a() -> None:
    instance = None
    for path in instance_files(ROOT / "dataset"):
        candidate = parse_instance(path)
        if {"slow", "normal", "fast"} <= set(candidate.curves) and (
            instance is None or len(candidate.customers) < len(instance.customers)
        ):
            instance = candidate
    if instance is None:
        raise SystemExit("no instance contains slow, normal, and fast curves")
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    colors = {"slow": "#0072B2", "normal": "#009E73", "fast": "#D55E00"}
    rows = []
    for name in ("slow", "normal", "fast"):
        curve = instance.curves[name]
        energy = np.array(curve.energies) / 1000.0
        ax.plot(energy, curve.times, color=colors[name], marker="o", label=name)
        secant = linear_secant_curve(curve)
        early = linear_early_rate_curve(curve)
        ax.plot(np.array(secant.energies) / 1000.0, secant.times, color=colors[name], linestyle="--", linewidth=1.0)
        ax.plot(np.array(early.energies) / 1000.0, early.times, color=colors[name], linestyle=":", linewidth=1.0)
        for q_wh, t_h in zip(curve.energies, curve.times):
            rows.append({"instance": instance.name, "station_type": name, "energy_kwh": q_wh / 1000.0, "cumulative_h": t_h})
    ax.plot([], [], color="#444444", linestyle="--", label="Secant")
    ax.plot([], [], color="#444444", linestyle=":", label="Early rate")
    ax.set_xlabel("State of charge (kWh)")
    ax.set_ylabel("Cumulative charging time (h)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / "fig_charging_curves.png")
    plt.close(fig)
    _write(OUT_TAB / "fig_charging_curves.csv", rows)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure_b() -> dict:
    chosen = None
    for path in sorted(SCHEDULES.glob("*_infeasible_under_true.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        instance = _instance_by_name(saved["instance"])
        linear = instance_with_charging_model(instance, saved["model"])
        for route_index, raw_stops in enumerate(saved["stops"]):
            stops = tuple((int(node), None if amount is None else float(amount)) for node, amount in raw_stops)
            true_check = check_route(instance, stops)
            linear_check = check_route(linear, stops)
            if linear_check.feasible and (not true_check.feasible) and "exceeds" in true_check.reason:
                chosen = (saved, instance, linear, route_index, stops, true_check, linear_check)
                break
        if chosen:
            break
    if chosen is None:
        raise SystemExit("no saved linear plan is feasible under the surrogate and infeasible by duration under true physics")
    saved, instance, linear, route_index, stops, true_check, linear_check = chosen
    soc = instance.battery_wh
    charges = []
    for index, (node, amount) in enumerate(stops):
        if amount is not None and instance.nodes[node].kind == 2 and float(amount) > 1e-7:
            q_out = min(soc + float(amount), instance.battery_wh)
            station = instance.nodes[node].station_type
            charges.append(
                {
                    "station_type": station,
                    "soc_in_kwh": soc / 1000.0,
                    "soc_out_kwh": q_out / 1000.0,
                    "true_h": charging_time(instance.curves[station], soc, q_out),
                    "linear_h": charging_time(linear.curves[station], soc, q_out),
                }
            )
            soc = q_out
        if index + 1 < len(stops):
            soc = max(0.0, soc - instance.energy_wh[node][stops[index + 1][0]])
    record = {
        "instance": instance.name,
        "charging_model": saved["model"],
        "route_index": route_index,
        "linear_duration_h": linear_check.duration_hours,
        "true_duration_h": true_check.duration_hours,
        "duration_limit_h": instance.max_route_hours,
        "linear_charging_h": linear_check.charging_hours,
        "true_charging_h": true_check.charging_hours,
        "true_reason": true_check.reason,
    }
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    axes[0].bar(
        ["Linear estimate", "True nonlinear", "Duration limit"],
        [record["linear_duration_h"], record["true_duration_h"], record["duration_limit_h"]],
        color=["#E69F00", "#0072B2", "#444444"],
    )
    axes[0].set_ylim(9.90, 10.20)
    axes[0].set_ylabel("Route duration (h)")
    axes[1].bar(
        ["Linear estimate", "True nonlinear"],
        [record["linear_charging_h"], record["true_charging_h"]],
        color=["#E69F00", "#0072B2"],
    )
    axes[1].set_ylabel("Charging time on the route (h)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_linear_plan_infeasible.png")
    plt.close(fig)
    _write(OUT_TAB / "fig_linear_plan_infeasible.csv", [record])
    _write(OUT_TAB / "fig_linear_plan_charges.csv", charges)
    return record


def figure_e() -> None:
    path = SCHEDULES / "tc0c10s2cf1_TRUE-NL.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    instance = _instance_by_name(saved["instance"])
    route_index = None
    stops = None
    for index, raw_stops in enumerate(saved["stops"]):
        if any(amount is not None for _node, amount in raw_stops):
            route_index = index
            stops = tuple((int(node), None if amount is None else float(amount)) for node, amount in raw_stops)
            break
    if stops is None:
        raise SystemExit("saved TRUE-NL schedule has no charging stop")
    check = check_route(instance, stops)
    if not check.feasible:
        raise SystemExit(f"saved TRUE-NL route is not feasible: {check.reason}")
    soc = instance.battery_wh
    clock = 0.0
    times = [0.0]
    levels = [soc / 1000.0]
    events = []
    for index, (node, amount) in enumerate(stops):
        if amount is not None and instance.nodes[node].kind == 2 and float(amount) > 1e-7:
            q_out = min(soc + float(amount), instance.battery_wh)
            station = instance.nodes[node].station_type
            added_h = charging_time(instance.curves[station], soc, q_out)
            events.append(
                {
                    "route_index": route_index,
                    "station_type": station,
                    "arrival_soc_kwh": soc / 1000.0,
                    "departure_soc_kwh": q_out / 1000.0,
                    "energy_kwh": (q_out - soc) / 1000.0,
                    "charge_h": added_h,
                    "clock_h": clock,
                }
            )
            times.extend([clock, clock + added_h])
            levels.extend([soc / 1000.0, q_out / 1000.0])
            clock += added_h
            soc = q_out
        if index + 1 < len(stops):
            nxt = stops[index + 1][0]
            hop = instance.travel_hours[node][nxt]
            times.extend([clock, clock + hop])
            levels.extend([soc / 1000.0, max(0.0, soc - instance.energy_wh[node][nxt]) / 1000.0])
            clock += hop
            if instance.nodes[nxt].kind == 1:
                clock += instance.service_times[nxt]
            soc = max(0.0, soc - instance.energy_wh[node][nxt])
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(times, levels, color="#0072B2")
    for event in events:
        ax.scatter([event["clock_h"]], [event["arrival_soc_kwh"]], color="#D55E00", zorder=3)
    ax.set_xlabel("Time along the route (h)")
    ax.set_ylabel("State of charge (kWh)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_soc_schedule.png")
    plt.close(fig)
    _write(OUT_TAB / "fig_soc_schedule.csv", events)


def figure_d() -> None:
    table = ROOT / "results" / "tables" / "final" / "route_decomposition_summary.csv"
    if not table.exists():
        return
    with table.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["protocol"] == "sample" and row["component"] in {"travel_h", "charging_h", "full_h"}
        ]
    if len(rows) != 12:
        raise SystemExit(f"expected 12 sampled component rows, found {len(rows)}")
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    order = (("travel_h", "Travel", "#0072B2"), ("charging_h", "Charging", "#D55E00"), ("full_h", "Full", "#009E73"))
    for ax, model, title in (
        (axes[0], "H2", "Route Transformer"),
        (axes[1], "H4", "RT + DeepSets"),
    ):
        labels = []
        values = []
        errors = []
        colors = []
        for size in ("10", "20"):
            for component, label, color in order:
                row = next(item for item in rows if item["model"] == model and item["size"] == size and item["component"] == component)
                delta = float(row["delta"])
                labels.append(f"n={size}\n{label}")
                values.append(delta)
                errors.append((delta - float(row["ci_low"]), float(row["ci_high"]) - delta))
                colors.append(color)
        yerr = np.array(errors).T
        ax.bar(range(len(values)), values, color=colors, yerr=yerr, capsize=2)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(title)
    axes[0].set_ylabel("RL − BC, mean of four samples (h)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_rl_duration_components.png")
    plt.close(fig)


def main() -> None:
    figure_a()
    record = figure_b()
    figure_e()
    figure_d()
    print({"figure_b": record})


if __name__ == "__main__":
    main()
