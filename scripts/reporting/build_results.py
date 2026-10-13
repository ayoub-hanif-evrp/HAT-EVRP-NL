"""Build publication PNG figures and Markdown tables from authoritative raw data.

Usage:
  python scripts/reporting/build_results.py --all
  python scripts/reporting/build_results.py --all --skip-decomp   # reuse existing decomp CSV
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presentation_names import method_label, search_label, wrap_label  # noqa: E402
from reporting_paths import (  # noqa: E402
    ARCHIVE,
    AUTH_N40_DECOMP,
    AUTH_N40_RECOVERY_PIPE,
    CHECKPOINTS,
    DATASET,
    FIGURES,
    LITERATURE_BKS,
    RAW,
    RESULTS,
    TABLES,
)

SEED = 42
DPI = 300


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(xs) if xs else float("nan")


def _read_csv(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find(*names: str) -> Path | None:
    search_roots = [
        RESULTS,
        RAW / "cycle3",
        RAW / "regression_recovery",
        RAW / "insertion_repair",
        RAW / "cost_aware",
        RAW / "generator_research",
        RAW / "misc",
        RAW / "neural_repair",
        RAW / "references",
    ]
    for name in names:
        for base in search_roots:
            cand = base / name
            if cand.exists():
                return cand
        matches = list(RAW.rglob(name))
        if matches:
            return matches[0]
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_table(name: str, title: str, meta: dict, headers: list[str], rows: list[list], notes: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    lines = [
        f"# {title}",
        "",
        f"- Objective convention: {meta.get('objective', 'n/a')}",
        f"- Units: {meta.get('units', 'hours')}",
        f"- Benchmark size: {meta.get('size', 'n/a')}",
        f"- Instance count: {meta.get('n_instances', 'n/a')}",
        f"- Methods / checkpoints: {meta.get('methods', 'n/a')}",
        f"- Search protocol: {meta.get('protocol', 'n/a')}",
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
    return path


def fig_save(path: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def make_architecture_png() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    boxes = [
        (0.15, 1.5, 1.55, 1.5, "Static\nnode encoding"),
        (1.9, 1.5, 1.7, 1.5, "Causal EV\nhistory\nTransformer"),
        (3.8, 1.5, 1.65, 1.5, "Dynamic\nroute-state\nfusion"),
        (5.65, 1.5, 1.55, 1.5, "DeepSets\nfleet\naggregation"),
        (7.4, 1.5, 1.55, 1.5, "Hierarchical\nroute /\ncustomer decoder"),
        (9.15, 2.35, 1.5, 1.0, "frvcpy\nFRVCP-NL"),
        (9.15, 0.85, 1.5, 1.0, "Independent\nPython check"),
        (10.85, 1.5, 1.0, 1.5, "Feasible\nRoutePlan"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y), w, h, facecolor="#e8f1fb", edgecolor="#1f4e79", linewidth=1.4
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)
    for x0, x1, y in [
        (1.7, 1.9, 2.25),
        (3.6, 3.8, 2.25),
        (5.45, 5.65, 2.25),
        (7.2, 7.4, 2.25),
        (8.95, 9.15, 2.85),
        (8.95, 9.15, 1.35),
        (10.65, 10.85, 2.25),
    ]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title(
        "HAT-EVRP-NL: hybrid neural routing + exact nonlinear charging (frvcpy)",
        fontsize=11,
    )
    fig_save(FIGURES / "fig01_architecture.png")


def make_charging_curves() -> None:
    from data import instance_files, parse_instance

    inst = None
    for path in instance_files(DATASET):
        candidate = parse_instance(path)
        if set(candidate.curves) >= {"slow", "normal", "fast"}:
            inst = candidate
            break
    if inst is None:
        print("WARN: no XML with slow/normal/fast curves; skipping fig03")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"slow": "#4c78a8", "normal": "#f28e2b", "fast": "#e15759"}
    for stype in ("slow", "normal", "fast"):
        curve = inst.curves[stype]
        ax.plot(
            curve.times,
            [e / 1000.0 for e in curve.energies],
            marker="o",
            color=colors[stype],
            label=f"{stype} (XML breakpoints)",
        )
    ax.set_xlabel("Charging time (h)")
    ax.set_ylabel("Battery energy (kWh)")
    ax.set_title(f"Montoya nonlinear charging curves ({inst.name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_save(FIGURES / "fig03_charging_curves.png")


def _row_objective(row: dict) -> float | None:
    for key in ("mean_h", "objective_h"):
        if row.get(key) not in (None, ""):
            return float(row[key])
    return None


def make_main_quality_and_scale() -> None:
    n10 = _find("FINAL_n10_fairls.csv")
    # Prefer per-instance multi-seed CSV for uncertainty.
    paired_paths = [
        _find(f"compare_size10_fairls_final_seed{s}_c3.csv") for s in (42, 43, 44)
    ]
    paired_paths = [p for p in paired_paths if p]
    if paired_paths:
        by = defaultdict(list)
        for path in paired_paths:
            for r in _read_csv(path):
                if r.get("objective_h") and r.get("method") in (
                    "H0_strong",
                    "H1",
                    "H2",
                    "H3",
                    "H4",
                ):
                    by[r["method"]].append(float(r["objective_h"]))
        methods = [m for m in ("H0_strong", "H1", "H2", "H3", "H4") if m in by]
        means = [_mean(by[m]) for m in methods]
        # Layout-aware SE approximation: std over 60 instance-seeds / sqrt(n)
        errs = [
            (statistics.stdev(by[m]) / math.sqrt(len(by[m]))) if len(by[m]) > 1 else 0.0
            for m in methods
        ]
        if methods:
            fig, ax = plt.subplots(figsize=(8.8, 5.0))
            y = np.arange(len(methods))
            ax.errorbar(means, y, xerr=errs, fmt="o", color="#4c78a8", capsize=3, ms=8)
            ax.set_yticks(y)
            ax.set_yticklabels(
                [wrap_label(method_label(m, short=True), 28) for m in methods],
                fontsize=8,
            )
            ax.set_xlabel("Mean full duration (h) ± SE over instance-seeds")
            ax.set_title(
                "n=10 fair-LS Cycle-3 — matched best-of-10 + one-shot basic LS "
                "(20 instances × 3 seeds)"
            )
            ax.grid(True, axis="x", alpha=0.3)
            lo = min(means) - 0.4
            ax.set_xlim(lo, max(means) + 0.4)
            fig_save(FIGURES / "fig06_main_benchmark_n10_fairls.png")
    elif n10:
        rows = _read_csv(n10)
        methods, means = [], []
        for m in ("H0_strong", "H1", "H2", "H3", "H4"):
            match = [r for r in rows if r.get("method") == m]
            if match and _row_objective(match[0]) is not None:
                methods.append(m)
                means.append(_row_objective(match[0]))
        if methods:
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            y = np.arange(len(methods))
            ax.plot(means, y, "o", color="#4c78a8", ms=9)
            ax.set_yticks(y)
            ax.set_yticklabels(
                [wrap_label(method_label(m, short=True), 28) for m in methods],
                fontsize=8,
            )
            ax.set_xlabel("Mean full duration (h)")
            ax.set_title("n=10 fair-LS Cycle-3 (matched protocol)")
            ax.grid(True, axis="x", alpha=0.3)
            fig_save(FIGURES / "fig06_main_benchmark_n10_fairls.png")

    size_means = {"H2": {}, "H4": {}}
    size_n = {}
    for size, name in (
        (10, "FINAL_n10_fairls.csv"),
        (20, "FINAL_n20_fairls.csv"),
        (40, "FINAL_n40_fairls.csv"),
        (80, "FINAL_n80_fairls.csv"),
    ):
        path = _find(name)
        if not path:
            continue
        rows = _read_csv(path)
        for method in ("H2", "H4"):
            match = [r for r in rows if r.get("method") == method]
            if match and _row_objective(match[0]) is not None:
                size_means[method][size] = _row_objective(match[0])
                size_n[size] = match[0].get("feasible", "")
    sizes = sorted(set(size_means["H2"]) | set(size_means["H4"]))
    if sizes:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for method, style in (("H2", "o-"), ("H4", "s-")):
            xs = [s for s in sizes if s in size_means[method]]
            ys = [size_means[method][s] for s in xs]
            ax.plot(
                xs,
                ys,
                style,
                label=method_label(method, short=True),
            )
        for s in sizes:
            note = size_n.get(s, "")
            ax.annotate(
                str(note),
                (s, size_means["H2"].get(s, size_means["H4"].get(s, 0))),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color="#666",
            )
        ax.set_xlabel("Evaluation customers (training size fixed at n=10)")
        ax.set_ylabel("Mean full duration (h)")
        ax.set_title(
            "Scaling under Cycle-3 fair-LS transfer (annotate feasible counts; n=80 is 3 instances)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig_save(FIGURES / "fig07_scaling_fairls.png")


def make_search_and_h2h4() -> None:
    pipe = _find("RECOVERY_n40_recovered_pipeline.csv")
    if not pipe:
        return
    rows = _read_csv(pipe)
    stages = ["basic_ls", "strong_ls", "classical_ruin", "neural_insert", "hybrid_insert"]
    # Dual panel: quality + oracle executions
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x = np.arange(len(stages))
    width = 0.35
    for i, method in enumerate(("H2", "H4")):
        means, execs = [], []
        for stage in stages:
            group = [
                r
                for r in rows
                if r["method"] == method and r["stage"] == stage and r["feasible"] == "1"
            ]
            means.append(_mean([float(r["objective_h"]) for r in group]))
            execs.append(
                _mean(
                    [
                        float(r["oracle_solver_executions"])
                        for r in group
                        if r.get("oracle_solver_executions") not in ("", None)
                    ]
                )
            )
        ax0.bar(x + (i - 0.5) * width, means, width, label=method_label(method, short=True))
        ax1.bar(x + (i - 0.5) * width, execs, width, label=method_label(method, short=True))
    labels = [wrap_label(search_label(s, short=True), 16) for s in stages]
    for ax, ylab, title in (
        (ax0, "Mean full duration (h)", "Solution quality"),
        (ax1, "Mean frvcpy solver executions", "Compute cost (unequal)"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        "n=40 recovered ordinary λ=0 starts — matched starts, unequal realized search compute",
        fontsize=11,
    )
    fig_save(FIGURES / "fig08_search_ablation_n40_recovered.png")

    by_inst = defaultdict(dict)
    for r in rows:
        if r["stage"] == "hybrid_insert" and r["feasible"] == "1":
            by_inst[r["instance"]][r["method"]] = float(r["objective_h"])
    paired = [
        (inst, d["H2"], d["H4"], d["H4"] - d["H2"])
        for inst, d in sorted(by_inst.items())
        if "H2" in d and "H4" in d
    ]
    if paired:
        deltas = [d for *_, d in paired]
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        ax.errorbar(
            [_mean(deltas)],
            [0],
            xerr=[statistics.stdev(deltas) / math.sqrt(len(deltas))],
            fmt="o",
            color="#e15759",
            capsize=4,
            ms=8,
            label="mean H4−H2 ± SE",
        )
        ax.scatter(deltas, np.linspace(-0.3, 0.3, len(deltas)), alpha=0.55, s=28, color="#59a14f")
        ax.axvline(0, color="black", lw=1)
        wins = sum(1 for d in deltas if d < -1e-9)
        losses = sum(1 for d in deltas if d > 1e-9)
        ties = len(deltas) - wins - losses
        ax.set_yticks([])
        ax.set_xlabel(
            "Full duration difference (h): History+DeepSets (H4) − History (H2)"
        )
        ax.set_title(
            f"n=40 Classical+Neural Insertion — paired (W/T/L={wins}/{ties}/{losses}); "
            f"single seed 42; exploratory"
        )
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
        fig_save(FIGURES / "fig09_h2_vs_h4_n40_hybrid.png")


def make_objective_decomposition_figure(decomp_rows: list[dict]) -> None:
    hybrid = [
        r
        for r in decomp_rows
        if r.get("stage") == "hybrid_insert" and str(r.get("feasible")) in ("1", "True")
    ]
    methods = []
    travel = []
    service = []
    charge = []
    title = "n=40 objective decomposition (recovered hybrid; charging inside duration)"
    if hybrid:
        by = defaultdict(list)
        for r in hybrid:
            by[r["method"]].append(r)
        methods = sorted(by)
        travel = [_mean([float(r["travel_h"]) for r in by[m]]) for m in methods]
        service = [_mean([float(r["service_h"]) for r in by[m]]) for m in methods]
        charge = [_mean([float(r["charging_h"]) for r in by[m]]) for m in methods]
    else:
        # Fallback: Cycle-3 fair-LS aggregates already store mean_charge_h.
        n40 = _find("FINAL_n40_fairls.csv")
        if not n40:
            print("WARN: no decomposition source; skipping fig04")
            return
        rows = _read_csv(n40)
        title = "n=40 objective decomposition (Cycle-3 fair-LS; charging inside duration)"
        for m in ("H2", "H4"):
            match = [r for r in rows if r.get("method") == m]
            if not match:
                continue
            full = float(match[0]["mean_h"])
            ch = float(match[0]["mean_charge_h"])
            # Official n=40 complete coverage ⇒ 20.0 h total service.
            svc = 20.0
            methods.append(m)
            service.append(svc)
            charge.append(ch)
            travel.append(full - svc - ch)
    if not methods:
        print("WARN: empty decomposition; skipping fig04")
        return
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x, travel, label="Travel", color="#4c78a8")
    ax.bar(x, service, bottom=travel, label="Service (fixed)", color="#f28e2b")
    bottom2 = [a + b for a, b in zip(travel, service)]
    ax.bar(x, charge, bottom=bottom2, label="Nonlinear charging", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(method_label(m, short=True), 18) for m in methods], fontsize=8)
    ax.set_ylabel("Mean hours")
    ax.set_title(title)
    ax.legend()
    fig_save(FIGURES / "fig04_objective_decomposition_n40.png")


def make_direct_charge_figure() -> None:
    path = _find("direct_charge_size10_seed42_c3.csv")
    if not path:
        print("WARN: missing direct_charge CSV; skipping fig10")
        return
    rows = _read_csv(path)
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    methods = [m for m in ("hybrid_H4", "direct_CHARGE", "direct_plus_repair") if m in by]
    feas = [
        100.0 * sum(1 for r in by[m] if str(r.get("feasible")) in ("1", "True")) / max(len(by[m]), 1)
        for m in methods
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(methods, feas, color=["#4c78a8", "#e15759", "#f28e2b"][: len(methods)])
    ax.set_ylabel("Feasibility (%)")
    ax.set_title("Nonlinear charging ablation (n=10): frvcpy vs learned CHARGE")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(feas):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center")
    fig_save(FIGURES / "fig10_charging_ablation_feasibility.png")


def _construct_ordinary(policy, inst, samples: int = 3):
    from train import decode

    attempts = [
        decode(policy, inst, sample=i > 0, seed=SEED * 1000 + i, cost_lambda=0.0)
        for i in range(samples)
    ]
    feas = [a for a in attempts if a["feasible"]]
    if not feas:
        return None
    return min(feas, key=lambda a: a["objective"])


def regenerate_n40_decomposition() -> list[dict]:
    """Re-run recovered ordinary λ=0 + hybrid and store true charging hours."""
    import torch
    from charging import FixedRouteOracle, check_solution, decompose_solution
    from data import instance_files, parse_instance
    from evaluate import _refine_best_once
    from train import hybrid_lns, load_policy

    out_path = AUTH_N40_DECOMP
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pipe_ref = {
        (r["instance"], r["method"], r["stage"]): float(r["objective_h"])
        for r in _read_csv(AUTH_N40_RECOVERY_PIPE)
        if r.get("objective_h") and r.get("feasible") == "1"
    }
    policies = {
        "H2": load_policy(CHECKPOINTS / "H2_n10_seed42_c3.pt"),
        "H4": load_policy(CHECKPOINTS / "H4_n10_seed42_c3.pt"),
    }
    instances = [
        parse_instance(p)
        for p in instance_files(DATASET)
        if len(parse_instance(p).customers) == 40
    ]
    rows: list[dict] = []
    solutions_dir = RAW / "regression_recovery" / "solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        for method, policy in policies.items():
            for inst in instances:
                best = _construct_ordinary(policy, inst)
                if best is None:
                    continue
                start_routes = [tuple(r) for r in best["routes"]]
                start_plans = list(best["plans"])

                r1, p1, _, _, _ = _refine_best_once(
                    inst, list(start_routes), list(start_plans), 3, 4000, "basic"
                )
                valid, _ = check_solution(inst, p1)
                obj1 = sum(p.duration_hours for p in p1) if valid else float("inf")
                if obj1 > best["objective"] + 1e-10:
                    r1, p1, obj1 = start_routes, start_plans, best["objective"]

                r2, p2, _, _, _ = _refine_best_once(
                    inst, list(start_routes), list(start_plans), 3, 4000, "strong"
                )
                valid, _ = check_solution(inst, p2)
                obj2 = sum(p.duration_hours for p in p2) if valid else float("inf")
                if obj2 > best["objective"] + 1e-10:
                    r2, p2, obj2 = start_routes, start_plans, best["objective"]

                oracle = FixedRouteOracle(inst)
                rh, ph, _, _ = hybrid_lns(
                    policy,
                    inst,
                    list(r2),
                    list(p2),
                    max_iterations=6,
                    seed=SEED + 91,
                    cost_lambda=4.0,
                    classical_iterations=6,
                    classical_budget=600,
                    repair_insertion=True,
                    oracle=oracle,
                )
                valid, _ = check_solution(inst, ph)
                objh = sum(p.duration_hours for p in ph) if valid else float("inf")
                if objh > obj2 + 1e-10:
                    rh, ph, objh = r2, p2, obj2

                for stage, plans, obj in (("basic_ls", p1, obj1), ("hybrid_insert", ph, objh)):
                    decomp = decompose_solution(inst, plans)
                    ref = pipe_ref.get((inst.name, method, stage))
                    match = (
                        ref is not None
                        and decomp["feasible"]
                        and abs(decomp["full_h"] - ref) <= 1e-3
                    )
                    row = {
                        "instance": inst.name,
                        "method": method,
                        "stage": stage,
                        "feasible": int(decomp["feasible"]),
                        "travel_h": decomp["travel_h"] if decomp["feasible"] else "",
                        "service_h": decomp["service_h"] if decomp["feasible"] else "",
                        "charging_h": decomp["charging_h"] if decomp["feasible"] else "",
                        "full_h": decomp["full_h"] if decomp["feasible"] else "",
                        "lit_h": decomp["lit_h"] if decomp["feasible"] else "",
                        "energy_charged_kwh": (
                            decomp["energy_charged_wh"] / 1000.0 if decomp["feasible"] else ""
                        ),
                        "n_routes": decomp["n_routes"],
                        "n_charges": decomp["n_charges"],
                        "charging_share": decomp["charging_share"] if decomp["feasible"] else "",
                        "station_types": (
                            "|".join(decomp["station_types"]) if decomp["feasible"] else ""
                        ),
                        "identity_ok": int(decomp.get("identity_ok", False)),
                        "pipeline_ref_h": ref if ref is not None else "",
                        "matches_pipeline_1e3": int(bool(match)),
                        "oracle_solver_executions": (
                            oracle.solver_executions if stage == "hybrid_insert" else ""
                        ),
                    }
                    rows.append(row)
                    print(
                        f"decomp {method} {stage} {inst.name} full={row['full_h']} "
                        f"charge={row['charging_h']} match={match}",
                        flush=True,
                    )
                    if stage == "hybrid_insert" and decomp["feasible"]:
                        payload = {
                            "instance": inst.name,
                            "method": method,
                            "stage": stage,
                            "full_h": decomp["full_h"],
                            "routes": [list(r) for r in rh],
                            "stops": [
                                [[int(n), None if a is None else float(a)] for n, a in p.stops]
                                for p in ph
                            ],
                        }
                        (solutions_dir / f"{method}_{inst.name}_hybrid.json").write_text(
                            json.dumps(payload), encoding="utf-8"
                        )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    matched = sum(int(r["matches_pipeline_1e3"]) for r in rows)
    print(
        f"wrote {out_path} ({len(rows)} rows, {matched}/{len(rows)} match pipeline ±1e-3; "
        f"{time.perf_counter()-t0:.1f}s)"
    )
    return rows


def make_routing_example() -> None:
    import torch
    from data import instance_files, parse_instance
    from evaluate import _refine_best_once
    from train import load_policy

    paths = [p for p in instance_files(DATASET) if "c10" in p.name]
    if not paths:
        return
    inst = parse_instance(paths[0])
    policy = load_policy(CHECKPOINTS / "H2_n10_seed42_c3.pt")
    with torch.no_grad():
        best = _construct_ordinary(policy, inst)
    if best is None:
        return
    routes, plans, _, _, _ = _refine_best_once(
        inst, list(best["routes"]), list(best["plans"]), 3, 4000, "basic"
    )
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    depot_plotted = False
    cust_plotted = False
    st_plotted = False
    for node in inst.nodes:
        if node.kind == 0:
            ax.scatter(
                node.x,
                node.y,
                c="black",
                s=90,
                marker="s",
                label="depot" if not depot_plotted else None,
                zorder=5,
            )
            depot_plotted = True
        elif node.kind == 1:
            ax.scatter(
                node.x,
                node.y,
                c="#4c78a8",
                s=36,
                label="customer" if not cust_plotted else None,
                zorder=3,
            )
            cust_plotted = True
        else:
            ax.scatter(
                node.x,
                node.y,
                c="#e15759",
                s=60,
                marker="^",
                label="station" if not st_plotted else None,
                zorder=4,
            )
            st_plotted = True
    colors = plt.cm.tab10.colors
    for i, plan in enumerate(plans):
        xs = [inst.nodes[n].x for n, _ in plan.stops]
        ys = [inst.nodes[n].y for n, _ in plan.stops]
        ax.plot(
            xs,
            ys,
            "-",
            color=colors[i % 10],
            lw=1.8,
            label=f"route {i+1} ({plan.duration_hours:.2f} h)",
        )
    ax.set_title(f"Feasible EV routes: {inst.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal")
    fig_save(FIGURES / "fig02_problem_example.png")


def make_soc_trace() -> None:
    import torch
    from charging import charging_time
    from data import instance_files, parse_instance
    from evaluate import _refine_best_once
    from train import load_policy

    paths = [p for p in instance_files(DATASET) if "c10" in p.name]
    inst = parse_instance(paths[0])
    policy = load_policy(CHECKPOINTS / "H2_n10_seed42_c3.pt")
    with torch.no_grad():
        best = _construct_ordinary(policy, inst)
    if best is None:
        return
    _, plans, _, _, _ = _refine_best_once(
        inst, list(best["routes"]), list(best["plans"]), 3, 4000, "basic"
    )
    plan = max(plans, key=lambda p: p.duration_hours)
    soc = inst.battery_wh
    times = [0.0]
    socs = [1.0]
    t = 0.0
    events = []
    for index, (node, amount) in enumerate(plan.stops):
        kind = inst.nodes[node].kind
        if amount is not None and amount > 1e-9 and kind == 2:
            curve = inst.curves[inst.nodes[node].station_type]
            q_out = min(soc + amount, inst.battery_wh)
            dt = charging_time(curve, soc, q_out)
            events.append((t, t + dt, inst.nodes[node].station_type, q_out - soc))
            t += dt
            soc = q_out
            times.append(t)
            socs.append(soc / inst.battery_wh)
        if index + 1 < len(plan.stops):
            nxt = plan.stops[index + 1][0]
            dt = inst.travel_hours[node][nxt]
            if kind == 1:
                dt += inst.service_times[node]
            t += dt
            soc = max(0.0, soc - inst.energy_wh[node][nxt])
            times.append(t)
            socs.append(soc / inst.battery_wh)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(times, socs, "-", color="#4c78a8", lw=2, label="SOC")
    labeled = False
    for t0, t1, st, _e in events:
        ax.axvspan(
            t0,
            t1,
            color="#59a14f",
            alpha=0.35,
            label=f"charging ({st})" if not labeled else None,
        )
        labeled = True
    ax.set_xlabel("Route time (h)")
    ax.set_ylabel("SOC fraction")
    ax.set_title(f"SOC and nonlinear charging events ({inst.name})")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_save(FIGURES / "fig05_soc_charging_trace.png")


def build_tables(decomp_rows: list[dict]) -> None:
    from data import instance_files, parse_instance

    files = instance_files(DATASET)
    by_n: dict[int, list] = defaultdict(list)
    for path in files:
        inst = parse_instance(path)
        by_n[len(inst.customers)].append(inst)
    sample = parse_instance(files[0])
    # Derive speed/consumption from travel/energy if available via notes (40 km/h, 125 Wh/km).
    table01_rows = []
    for n in sorted(by_n):
        insts = by_n[n]
        n_stations = [sum(1 for node in i.nodes if node.kind == 2) for i in insts]
        families = sorted({i.name[:3] for i in insts})
        charger_types = sorted({t for i in insts for t in i.curves})
        svc = [
            sum(i.service_times[c] for c in i.customers) / max(1, len(i.customers))
            for i in insts
        ]
        table01_rows.append(
            [
                n,
                len(insts),
                f"{min(n_stations)}–{max(n_stations)}",
                ",".join(families),
                ",".join(charger_types),
                "40 km/h (Montoya)",
                "125 Wh/km (Montoya)",
                f"{sample.battery_wh/1000:.1f}",
                f"{sample.max_route_hours:.1f}",
                f"{_mean(svc):.2f}",
            ]
        )
    write_table(
        "table01_benchmark_overview.md",
        "Montoya E-VRP-NL benchmark overview",
        {
            "objective": "Full = travel+service+nonlinear charging; lit = travel+charging",
            "units": "hours / Wh / km/h",
            "size": "10–320 customers",
            "n_instances": sum(len(v) for v in by_n.values()),
            "methods": "dataset only",
            "protocol": "official XML (read-only)",
            "seeds": "n/a",
            "source": "dataset/montoya-et-al-2017",
        },
        [
            "Customers",
            "Instances",
            "Stations (range)",
            "Layout families",
            "Charger types",
            "Speed",
            "Consumption",
            "Battery kWh",
            "Route limit h",
            "Mean service h/cust",
        ],
        table01_rows,
        "Official Montoya testbed retained read-only. Speed/consumption follow the published "
        "Montoya parameterization used by the XML instances.",
    )

    n10 = _find("FINAL_n10_fairls.csv")
    if n10:
        rows = _read_csv(n10)
        table_rows = []
        for m in ("H0", "H0_strong", "H1", "H2", "H3", "H4"):
            match = [r for r in rows if r.get("method") == m]
            if match and _row_objective(match[0]) is not None:
                feas = match[0].get("feasible", "60/60")
                table_rows.append(
                    [
                        method_label(m, short=True),
                        m,
                        f"{_row_objective(match[0]):.4f}",
                        feas,
                        "Cycle-3 fair-LS",
                    ]
                )
        write_table(
            "table03_main_models.md",
            "Main model comparison (n=10 fair-LS Cycle-3)",
            {
                "objective": "Full duration (travel+service+charging)",
                "units": "hours",
                "size": 10,
                "n_instances": "20 instances × up to 3 seeds (see feasible column)",
                "methods": "H0–H4 Cycle-3 n10 checkpoints",
                "protocol": "best-of-10 constructions + one-shot basic LS",
                "seeds": "42–44 aggregated where present",
                "source": str(n10.relative_to(ROOT)).replace("\\", "/"),
            },
            ["Display name", "Key", "Mean full h", "Feasible", "Protocol"],
            table_rows,
            "History Transformer Routing (H2) is the strongest efficient neural ablation under "
            "matched fair-LS; HAT-EVRP-NL (H4) is the full architecture without a consistent "
            "multi-seed DeepSets win. Do not mix with hybrid-search rows.",
        )

    pipe = _find("RECOVERY_n40_recovered_pipeline.csv")
    if pipe:
        rows = _read_csv(pipe)
        table_rows = []
        for method in ("H2", "H4"):
            for stage in ("basic_ls", "strong_ls", "classical_ruin", "neural_insert", "hybrid_insert"):
                objs = [
                    float(r["objective_h"])
                    for r in rows
                    if r["method"] == method and r["stage"] == stage and r["feasible"] == "1"
                ]
                if objs:
                    table_rows.append(
                        [
                            method_label(method, short=True),
                            method,
                            search_label(stage, short=True),
                            stage,
                            f"{_mean(objs):.4f}",
                            len(objs),
                        ]
                    )
        write_table(
            "table04_search_ablation.md",
            "Search ablation on recovered n=40 constructions",
            {
                "objective": "Full duration",
                "units": "hours",
                "size": 40,
                "n_instances": 20,
                "methods": "History Transformer (H2) / HAT-EVRP-NL (H4) Cycle-3",
                "protocol": "ordinary λ=0 starts; matched starts; unequal realized search compute",
                "seeds": "42",
                "source": str(pipe.relative_to(ROOT)).replace("\\", "/"),
            },
            ["Model", "Key", "Search stage", "Stage key", "Mean full h", "N"],
            table_rows,
            "Classical + Neural Insertion Repair beats historical fair-LS. "
            "Not an equal-oracle-budget experiment.",
        )

        by_inst = defaultdict(dict)
        for r in rows:
            if r["stage"] == "hybrid_insert" and r["feasible"] == "1":
                by_inst[r["instance"]][r["method"]] = float(r["objective_h"])
        paired = [
            (inst, d["H2"], d["H4"], d["H4"] - d["H2"])
            for inst, d in sorted(by_inst.items())
            if "H2" in d and "H4" in d
        ]
        wins = sum(1 for *_, d in paired if d < -1e-9)
        losses = sum(1 for *_, d in paired if d > 1e-9)
        ties = len(paired) - wins - losses
        write_table(
            "table06_h2_vs_h4.md",
            "H2 vs H4 under recovered hybrid_insert (n=40)",
            {
                "objective": "Full duration",
                "units": "hours",
                "size": 40,
                "n_instances": len(paired),
                "methods": "H2 vs H4 Cycle-3, identical protocol",
                "protocol": "ordinary λ=0 + strong LS + hybrid_insert",
                "seeds": "42 (single seed)",
                "source": str(pipe.relative_to(ROOT)).replace("\\", "/"),
            },
            [
                "Instance",
                "History Transformer (H2) h",
                "HAT-EVRP-NL (H4) h",
                "H4−H2",
            ],
            [[a, f"{b:.4f}", f"{c:.4f}", f"{d:+.4f}"] for a, b, c, d in paired],
            f"Wins(H4 better)/ties/losses = {wins}/{ties}/{losses}. "
            f"Mean H4−H2 = {_mean([d for *_, d in paired]):+.4f} h. "
            "Single-seed recovered hybrid; not a general DeepSets claim.",
        )

    hybrid = [
        r
        for r in decomp_rows
        if r.get("stage") == "hybrid_insert" and str(r.get("feasible")) in ("1", "True")
    ]
    if hybrid:
        by = defaultdict(list)
        for r in hybrid:
            by[r["method"]].append(r)
        rows_out = []
        for m, group in sorted(by.items()):
            rows_out.append(
                [
                    method_label(m, short=True),
                    m,
                    f"{_mean([float(r['travel_h']) for r in group]):.4f}",
                    f"{_mean([float(r['service_h']) for r in group]):.4f}",
                    f"{_mean([float(r['charging_h']) for r in group]):.4f}",
                    f"{_mean([float(r['full_h']) for r in group]):.4f}",
                    f"{_mean([float(r['lit_h']) for r in group]):.4f}",
                    f"{_mean([float(r['n_charges']) for r in group]):.2f}",
                    f"{_mean([float(r['charging_share']) for r in group])*100:.1f}%",
                ]
            )
        write_table(
            "table02_objective_and_charging.md",
            "Objective decomposition with nonlinear charging (n=40 hybrid)",
            {
                "objective": "FULL=travel+service+charge; LIT=travel+charge",
                "units": "hours",
                "size": 40,
                "n_instances": 20,
                "methods": "H2/H4 Cycle-3 recovered hybrid",
                "protocol": "ordinary λ=0 + strong LS + hybrid_insert",
                "seeds": "42",
                "source": "results/raw/regression_recovery/RECOVERY_n40_objective_decomposition.csv",
            },
            [
                "Model",
                "Key",
                "Travel h",
                "Service h",
                "Nonlinear charge h",
                "Full h",
                "Lit h",
                "Mean charge visits",
                "Charge share",
            ],
            rows_out,
            "Charging hours come from exact F_s(q_out)-F_s(q_in) on validated RoutePlan.stops. "
            "frvcpy optimizes the fixed-route charging subproblem only. "
            "H4’s better full duration is driven mainly by lower travel despite slightly higher mean charging.",
        )
        write_table(
            "table07_charging_analysis.md",
            "Nonlinear charging analysis (n=40 recovered hybrid)",
            {
                "objective": "Charging component of full duration",
                "units": "hours / kWh",
                "size": 40,
                "n_instances": 20,
                "methods": "H2/H4",
                "protocol": "same as table02",
                "seeds": "42",
                "source": "RECOVERY_n40_objective_decomposition.csv",
            },
            ["Method", "Mean charge h", "Mean energy kWh", "Mean visits", "Identity OK rate"],
            [
                [
                    m,
                    f"{_mean([float(r['charging_h']) for r in g]):.4f}",
                    f"{_mean([float(r['energy_charged_kwh']) for r in g if r['energy_charged_kwh'] != '']):.3f}",
                    f"{_mean([float(r['n_charges']) for r in g]):.2f}",
                    f"{_mean([float(r['identity_ok']) for r in g])*100:.0f}%",
                ]
                for m, g in sorted(by.items())
            ],
            "Identity OK means travel+service+charge matches full duration within 1e-6 h.",
        )
    else:
        write_table(
            "table02_objective_and_charging.md",
            "Objective decomposition with nonlinear charging (n=40 hybrid)",
            {
                "objective": "FULL=travel+service+charge; LIT=travel+charge",
                "units": "hours",
                "size": 40,
                "n_instances": 20,
                "methods": "H2/H4",
                "protocol": "unavailable until decomp re-exec",
                "seeds": "42",
                "source": "missing RECOVERY_n40_objective_decomposition.csv",
            },
            ["Status"],
            [["Charging hours unavailable — stop counts alone cannot recover durations."]],
            "Re-run: python scripts/reporting/build_results.py --all",
        )
        write_table(
            "table07_charging_analysis.md",
            "Nonlinear charging analysis (n=40 recovered hybrid)",
            {
                "objective": "Charging component of full duration",
                "units": "hours / kWh",
                "size": 40,
                "n_instances": "n/a",
                "methods": "H2/H4",
                "protocol": "unavailable",
                "seeds": "42",
                "source": "missing",
            },
            ["Status"],
            [["Unavailable until objective decomposition CSV is regenerated."]],
            "Do not invent charging hours from visit counts.",
        )

    corr = _find("IMPROVEMENT_bks_gaps_corrected.csv")
    if corr and LITERATURE_BKS.exists():
        rows = _read_csv(corr)
        # Summary CSV: one row per method; column `n` = instances per method (typically 20).
        n_inst = int(float(rows[0]["n"])) if rows and rows[0].get("n") else 20
        pretty = []
        for r in rows:
            key = r.get("method", "")
            pretty.append(
                [
                    method_label(key, short=True),
                    key,
                    f"{float(r['mean_gap_lit_pct']):.2f}",
                    f"{float(r['median_gap_lit_pct']):.2f}",
                    f"{float(r['mean_gap_full_pct']):.2f}",
                    f"{float(r['median_gap_full_pct']):.2f}",
                    int(float(r["n"])),
                ]
            )
        write_table(
            "table08_bks_comparison.md",
            "Literature-compatible BKS gaps (corrected service convention)",
            {
                "objective": "Lit gap uses (full − Σ service) vs Froger Table 13 travel+charging BKS",
                "units": "% gap",
                "size": "n=10",
                "n_instances": f"{n_inst} official instances per method "
                f"({len(rows)} method-summary rows — not {len(rows)} instances)",
                "methods": ", ".join(r["method"] for r in rows),
                "protocol": "fair-LS Cycle-3; service-time correction applied",
                "seeds": "see IMPROVEMENT report / FINAL fair-LS aggregates",
                "source": str(corr.relative_to(ROOT)).replace("\\", "/"),
            },
            [
                "Display name",
                "Key",
                "Mean lit gap %",
                "Median lit gap %",
                "Mean naive gap % (full vs BKS)",
                "Median naive gap %",
                "N instances",
            ],
            pretty,
            "Cite **lit gap** columns only. Naive gaps (~43%) wrongly compare full duration to "
            f"literature BKS. Each row summarizes {n_inst} benchmark instances.",
        )

    rows_out = []
    for size, name in (
        (10, "FINAL_n10_fairls.csv"),
        (20, "FINAL_n20_fairls.csv"),
        (40, "FINAL_n40_fairls.csv"),
        (80, "FINAL_n80_fairls.csv"),
    ):
        path = _find(name)
        if not path:
            continue
        rows = _read_csv(path)
        for method in ("H2", "H4"):
            match = [r for r in rows if r.get("method") == method]
            if match and _row_objective(match[0]) is not None:
                rows_out.append(
                    [
                        size,
                        method,
                        f"{_row_objective(match[0]):.4f}",
                        match[0].get("feasible", ""),
                        "fair-LS from10",
                    ]
                )
    if pipe:
        rows = _read_csv(pipe)
        for method in ("H2", "H4"):
            objs = [
                float(r["objective_h"])
                for r in rows
                if r["method"] == method and r["stage"] == "hybrid_insert" and r["feasible"] == "1"
            ]
            if objs:
                rows_out.append([40, method, f"{_mean(objs):.4f}", f"{len(objs)}/20", "recovered hybrid"])
    write_table(
        "table05_scalability.md",
        "Scalability across problem sizes",
        {
            "objective": "Full duration",
            "units": "hours",
            "size": "10/20/40/80",
            "n_instances": 20,
            "methods": "H2/H4",
            "protocol": "protocol labeled per row — do not pool blindly",
            "seeds": "42 (+ multi-seed n10 where noted)",
            "source": "FINAL_n*_fairls + RECOVERY pipeline",
        },
        ["n", "Method", "Mean full h", "Feasible", "Protocol"],
        rows_out,
        "Training size is n=10 for all rows. Test size varies. Recovered hybrid is a different search protocol than fair-LS.",
    )

    write_table(
        "table09_reproducibility.md",
        "Reproducibility checklist for strongest n40 result",
        {
            "objective": "Full duration",
            "units": "hours",
            "size": 40,
            "n_instances": 20,
            "methods": "H4_c3 hybrid_insert (best mean)",
            "protocol": "ordinary λ=0, 3 samples, strong LS, classical ruin 6/600 + neural insert 6",
            "seeds": "42",
            "source": "scripts/experiments/run_regression_recovery.py",
        },
        ["Item", "Value"],
        [
            ["Checkpoint", "checkpoints/H4_n10_seed42_c3.pt"],
            ["Construction", "decode sample=(i>0), cost_lambda=0, no force_first"],
            ["Samples", "3"],
            ["LS", "strong, passes=3, budget=4000"],
            ["Hybrid", "ruin iters=6 budget=600 + neural_insert iters=6 repair_insertion=True"],
            ["Command", "python scripts/experiments/run_regression_recovery.py --stage pipeline"],
            ["Mean objective", "56.3700 h (see RECOVERY pipeline)"],
            ["Reporting", "python scripts/reporting/build_results.py --all"],
        ],
        "Preserve best incumbent; never accept a worse repair. Single seed — exploratory.",
    )


def write_index() -> None:
    lines = [
        "# Results index (publication layer)",
        "",
        "This directory separates **presentation** (`figures/`, `tables/`) from **evidence** (`raw/`, `archive/`).",
        "",
        "## Current strongest supported results",
        "",
        "| Setting | Configuration | Mean full h | Notes |",
        "|---|---|---:|---|",
        "| n=40 | H4 Cycle-3 + ordinary λ=0 + hybrid_insert | **56.370** | Beats hist fair-LS 57.506; single seed |",
        "| n=40 | H2 Cycle-3 + ordinary λ=0 + hybrid_insert | **56.806** | Beats hist fair-LS 57.173 |",
        "| n=10 | H2 Cycle-3 fair-LS | **19.813** | Multi-seed Cycle-3 baseline |",
        "",
        "## Do not cite as current best",
        "",
        "- INSERTION forced+λ4 n40 hybrid (~63.8 / 66.9 h): construction regression.",
        "- Naive ~43% BKS gaps: wrong objective convention (full vs lit).",
        "",
        "## Important limitations",
        "",
        "- H4’s better n40 mean in the recovered single-seed experiment does **not** establish a reliable general DeepSets advantage.",
        "- Equal wall-clock neural-search superiority has **not** been established.",
        "- Official benchmark evaluation was revisited repeatedly; treat recent search results as exploratory.",
        "- `frvcpy` is exact for the **fixed-route** charging subproblem, not a proof of global VRP optimality.",
        "",
        "## Figures (PNG only)",
        "",
    ]
    for p in sorted(FIGURES.glob("*.png")):
        lines.append(f"- [`figures/{p.name}`](figures/{p.name})")
    lines.extend(["", "## Tables (Markdown)", ""])
    for p in sorted(TABLES.glob("*.md")):
        lines.append(f"- [`tables/{p.name}`](tables/{p.name})")
    lines.extend(
        [
            "",
            "## Raw evidence",
            "",
            "- `raw/cycle3/` — Cycle-3 fair-LS CSVs",
            "- `raw/regression_recovery/` — recovered construction + hybrid",
            "- `raw/insertion_repair/` — invalid forced-start insertion runs (kept as evidence)",
            "- `archive/reports/` — historical Markdown reports",
            "",
            "## Key reports",
            "",
            "- [`REGRESSION_RECOVERY_REPORT.md`](REGRESSION_RECOVERY_REPORT.md)",
            "- [`FINAL_REPORT.md`](FINAL_REPORT.md) (Cycle-3 baseline)",
            "- [`INSERTION_REPAIR_REPORT.md`](INSERTION_REPAIR_REPORT.md)",
            "- [`LEARNING_RESULTS_REPORT.md`](LEARNING_RESULTS_REPORT.md)",
            "- [`../docs/TRAINING_AND_REWARD_METHODOLOGY.md`](../docs/TRAINING_AND_REWARD_METHODOLOGY.md)",
            "",
            "## Regenerate",
            "",
            "```bash",
            "python scripts/reporting/build_results.py --all --skip-decomp",
            "```",
            "",
        ]
    )
    (RESULTS / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote results/README.md")


def checkpoint_manifest() -> None:
    rows = []
    for path in sorted(CHECKPOINTS.glob("*.pt")):
        name = path.name
        role = "experimental"
        if "_c3.pt" in name:
            role = "primary" if name.startswith(("H2_", "H4_")) else "ablation"
        elif "CHARGE" in name:
            role = "ablation"
        elif any(x in name for x in ("perf_insert", "perf_cont", "perf_repair", "perf_v2")):
            role = "historical"
        elif name.endswith("_c2.pt"):
            role = "historical"
        rows.append(
            {
                "checkpoint": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
            }
        )
    if not rows:
        return
    out = CHECKPOINTS / "manifests" / "checkpoint_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} checkpoints)")


def dataset_hashes() -> None:
    rows = []
    for path in sorted(DATASET.rglob("*.xml")):
        rows.append(
            {
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
        )
    out = RAW / "references" / "dataset_xml_sha256.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} xml)")


def archive_legacy_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    legacy_dir = ARCHIVE / "figures_legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    keep = {
        "fig01_architecture.png",
        "fig02_problem_example.png",
        "fig03_charging_curves.png",
        "fig04_objective_decomposition_n40.png",
        "fig05_soc_charging_trace.png",
        "fig06_main_benchmark_n10_fairls.png",
        "fig07_scaling_fairls.png",
        "fig08_search_ablation_n40_recovered.png",
        "fig09_h2_vs_h4_n40_hybrid.png",
        "fig10_charging_ablation_feasibility.png",
        "fig11_training_supervision_flow.png",
        "fig12_learning_curves_h1h4.png",
        "fig12b_bc_and_rl_losses_separate.png",
        "fig13_rl_effect_h2_h4.png",
        "fig15_bc_vs_rl_ablation.png",
        "fig16_hierarchical_policy.png",
        "fig17_history_deepsets_contribution.png",
        # RQ3 / RQ4 main-paper figures
        "fig_main02_nonlinear_vs_linear_curves.png",
        "fig_main03_linear_plan_true_feasibility.png",
        "fig_main03b_paired_charging_time_deltas.png",
        "fig_main03c_e2e_true_feasibility.png",
        "fig_main03d_schedule_comparison_example.png",
        "fig_main04_objective_decomposition_nl.png",
        "fig_main07_reward_dynamics.png",
        "fig_main07b_bc_vs_rl_official.png",
    }
    for path in list(FIGURES.iterdir()):
        if path.name in keep:
            continue
        if path.suffix.lower() in {".png", ".svg", ".pdf", ".eps"}:
            dest = legacy_dir / path.name
            if dest.exists():
                dest = legacy_dir / f"moved_{path.name}"
            path.replace(dest)
            print(f"archived legacy figure -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--skip-decomp", action="store_true")
    parser.add_argument("--decomp-only", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.decomp_only:
        parser.error("pass --all or --decomp-only")

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    decomp_rows: list[dict] = []
    if args.decomp_only or (args.all and not args.skip_decomp):
        decomp_rows = regenerate_n40_decomposition()
    elif AUTH_N40_DECOMP.exists():
        decomp_rows = _read_csv(AUTH_N40_DECOMP)

    if args.decomp_only:
        return

    make_architecture_png()
    make_routing_example()
    make_charging_curves()
    make_objective_decomposition_figure(decomp_rows)
    make_soc_trace()
    make_main_quality_and_scale()
    make_search_and_h2h4()
    make_direct_charge_figure()
    build_tables(decomp_rows)
    reporting_dir = str(ROOT / "scripts" / "reporting")
    if reporting_dir not in sys.path:
        sys.path.insert(0, reporting_dir)
    from learning_presentation import build_all_learning

    build_all_learning()
    from charging_and_reward_figures import main as build_charging_and_reward

    build_charging_and_reward()
    write_index()
    checkpoint_manifest()
    dataset_hashes()
    archive_legacy_figures()


if __name__ == "__main__":
    main()
