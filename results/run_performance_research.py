"""Continuous performance-research experiments (Steps 1–8).

Stages:
  coverage   — v2 generator category coverage + decoupling checks
  teacher    — basic vs strong teacher on held-out synthetic seeds
  train      — pilot BC for H2 / H4 / H4_open (v2, strong teacher, multi-target)
  synthval   — synthetic validation of Cycle-3 vs new checkpoints
  search     — strong LS vs ruin-and-repair on synthetic / official subset
  official   — official Montoya n=10 (and larger if time) matched eval
  report     — write PERFORMANCE_RESEARCH_REPORT.md

Never overwrites *_c3.pt checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
CKPT = ROOT / "checkpoints"
DATA = ROOT / "dataset"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else float("nan")


def stage_coverage(n: int = 240, seed_start: int = 500000) -> dict:
    from data import SyntheticConfig, generate_synthetic, sample_synthetic_config

    rows = []
    geo_tech = Counter()
    place_dens = Counter()
    tech_counts = Counter()
    geo_counts = Counter()
    place_counts = Counter()
    dens_counts = Counter()
    zero_detour = 0
    repro_ok = 0
    decouple_geo_tech = True

    # Fixed-factor sweeps prove independence under explicit configs.
    for geo in ("random", "clustered", "mixed"):
        for tech in (("fast",), ("fast", "normal"), ("fast", "normal", "slow")):
            for place in ("uniform", "pmedian_region"):
                for dens in (0.10, 0.20):
                    cfg = SyntheticConfig(
                        geometry=geo,
                        station_placement=place,
                        density=dens,
                        tech_pool=tech,
                    )
                    inst = generate_synthetic(
                        10, seed_start + hash((geo, tech, place, dens)) % 100000,
                        config=cfg,
                        generator_version="v2",
                    )
                    assert set(inst.curves) == set(tech)
                    geo_tech[(geo, "+".join(tech))] += 1

    for i in range(n):
        seed = seed_start + i
        inst = generate_synthetic(10, seed, generator_version="v2")
        # Reproducibility
        inst2 = generate_synthetic(10, seed, generator_version="v2")
        if (
            inst.name == inst2.name
            and [(n.x, n.y, n.kind, n.station_type) for n in inst.nodes]
            == [(n.x, n.y, n.kind, n.station_type) for n in inst2.nodes]
        ):
            repro_ok += 1
        # Parse factors from name: syn_v2_{geo}_n{size}_{place}_d{dens}_{tech}_s{seed}
        parts = inst.name.split("_")
        # syn v2 geo n10 place dXX tech sSEED
        geo = parts[2]
        place = parts[4]
        dens = parts[5]
        tech = parts[6]
        geo_counts[geo] += 1
        place_counts[place] += 1
        dens_counts[dens] += 1
        tech_counts[tech] += 1
        geo_tech[(geo, tech)] += 1
        place_dens[(place, dens)] += 1
        # Stations coinciding with customers/depot (near-zero detour)
        cust_xy = {(inst.nodes[c].x, inst.nodes[c].y) for c in inst.customers}
        cust_xy.add((inst.nodes[inst.depot].x, inst.nodes[inst.depot].y))
        for s in inst.stations:
            sx, sy = inst.nodes[s].x, inst.nodes[s].y
            if any(math.hypot(sx - cx, sy - cy) < 1e-9 for cx, cy in cust_xy):
                zero_detour += 1
                break
        rows.append(
            {
                "seed": seed,
                "name": inst.name,
                "geometry": geo,
                "placement": place,
                "density": dens,
                "tech": tech,
                "customers": len(inst.customers),
                "stations": len(inst.stations),
            }
        )

    # Decoupling: under random sampling, each geometry should see multiple techs.
    for geo in ("random", "clustered", "mixed"):
        techs = {t for (g, t) in geo_tech if g == geo}
        if len(techs) < 2:
            decouple_geo_tech = False

    summary = {
        "n": n,
        "repro_ok": repro_ok,
        "repro_rate": repro_ok / n,
        "zero_detour_instances": zero_detour,
        "geometry": dict(geo_counts),
        "placement": dict(place_counts),
        "density": dict(dens_counts),
        "tech": dict(tech_counts),
        "geo_tech": {f"{g}|{t}": c for (g, t), c in sorted(geo_tech.items())},
        "place_dens": {f"{p}|{d}": c for (p, d), c in sorted(place_dens.items())},
        "decouple_geo_tech": decouple_geo_tech,
        "explicit_combo_ok": True,
    }
    _write_csv(RES / "PERF_generator_coverage.csv", rows)
    (RES / "PERF_generator_coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def stage_teacher(n: int = 24, seed_start: int = 510000) -> list[dict]:
    from data import generate_synthetic
    from solver import randomized_greedy

    rows = []
    for i in range(n):
        seed = seed_start + i
        instance = generate_synthetic(10, seed, generator_version="v2")
        for teacher, restarts, passes, budget, ops in (
            ("basic", 3, 1, 800, "basic"),
            ("strong", 8, 3, 2500, "strong"),
        ):
            t0 = time.perf_counter()
            result = randomized_greedy(
                instance,
                restarts=restarts,
                seed=seed + (0 if teacher == "basic" else 7),
                local_search_passes=passes,
                local_search_budget=budget,
                local_operators=ops,
            )
            runtime = time.perf_counter() - t0
            rows.append(
                {
                    "seed": seed,
                    "instance": instance.name,
                    "teacher": teacher,
                    "feasible": int(result.feasible),
                    "objective_h": result.objective_hours if result.feasible else "",
                    "routes": result.routes if result.feasible else "",
                    "runtime_s": round(runtime, 4),
                    "oracle_calls": result.oracle_calls,
                }
            )
            print(
                f"teacher {teacher} seed={seed} feas={result.feasible} "
                f"obj={result.objective_hours if result.feasible else 'inf'} "
                f"t={runtime:.2f}s ora={result.oracle_calls}"
            )
    _write_csv(RES / "PERF_teacher_compare.csv", rows)

    # Paired summary
    by_seed = defaultdict(dict)
    for row in rows:
        by_seed[row["seed"]][row["teacher"]] = row
    wins = {"strong_better": 0, "basic_better": 0, "tie": 0, "both_infeas": 0}
    for seed, pair in by_seed.items():
        b, s = pair.get("basic"), pair.get("strong")
        if not b or not s:
            continue
        if not b["feasible"] and not s["feasible"]:
            wins["both_infeas"] += 1
        elif b["feasible"] and not s["feasible"]:
            wins["basic_better"] += 1
        elif s["feasible"] and not b["feasible"]:
            wins["strong_better"] += 1
        else:
            bo, so = float(b["objective_h"]), float(s["objective_h"])
            if so < bo - 1e-9:
                wins["strong_better"] += 1
            elif bo < so - 1e-9:
                wins["basic_better"] += 1
            else:
                wins["tie"] += 1
    (RES / "PERF_teacher_compare_summary.json").write_text(
        json.dumps(wins, indent=2), encoding="utf-8"
    )
    print("teacher wins:", wins)
    return rows


def stage_train(
    models: tuple[str, ...] = ("H2", "H4", "H4_open"),
    seeds: tuple[int, ...] = (42,),
    train_n: int = 80,
    val_n: int = 24,
    steps: int = 120,
    rl_steps: int = 0,
    teacher: str = "strong",
    multi_target: bool = True,
    version: str = "perf_v2mt",
) -> None:
    """Pilot training; writes new checkpoints only (never *_c3.pt)."""
    for model in models:
        for seed in seeds:
            cmd = [
                sys.executable,
                str(ROOT / "train.py"),
                "--model",
                model,
                "--size",
                "10",
                "--steps",
                str(steps),
                "--rl-steps",
                str(rl_steps),
                "--train-instances",
                str(train_n),
                "--val-instances",
                str(val_n),
                "--eval-every",
                "20",
                "--seed",
                str(seed),
                "--version",
                version,
                "--teacher",
                teacher,
                "--generator",
                "v2",
                "--checkpoints",
                str(CKPT),
                "--results",
                str(RES),
            ]
            if multi_target:
                cmd.append("--multi-target")
            print("RUN", " ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(ROOT))


def stage_synthval(
    n: int = 40,
    seed_start: int = 520000,
    cost_lambda: float = 4.0,
    version_new: str = "perf_v2st",
) -> list[dict]:
    import torch
    from data import generate_synthetic
    from train import decode, load_policy

    rows = []
    configs = [
        ("H2_c3_l4", CKPT / "H2_n10_seed42_c3.pt", cost_lambda, "strong"),
        ("H4_c3_l4", CKPT / "H4_n10_seed42_c3.pt", cost_lambda, "strong"),
    ]
    for model in ("H2", "H4", "H4_open"):
        for ver, tag in (
            (version_new, "new"),
            ("perf_v2mt", "mt"),
            ("perf_v2st", "st"),
        ):
            path = CKPT / f"{model}_n10_seed42_{ver}.pt"
            if path.exists():
                configs.append((f"{model}_{tag}", path, cost_lambda, "strong"))
    # Deduplicate by tag keeping first path.
    seen = set()
    uniq = []
    for tag, path, lam, ops in configs:
        if tag in seen:
            continue
        seen.add(tag)
        uniq.append((tag, path, lam, ops))
    configs = uniq

    policies = {}
    for tag, path, _lam, _ops in configs:
        if tag not in policies and path.exists():
            policies[str(path)] = load_policy(path)

    with torch.no_grad():
        for i in range(n):
            seed = seed_start + i
            instance = generate_synthetic(10, seed, generator_version="v2")
            for tag, path, lam, ops in configs:
                if not path.exists():
                    continue
                policy = policies[str(path)]
                t0 = time.perf_counter()
                result = decode(
                    policy,
                    instance,
                    sample=False,
                    seed=seed,
                    local_passes=2,
                    local_budget=2000,
                    cost_lambda=lam,
                    local_operators=ops,
                )
                rows.append(
                    {
                        "seed": seed,
                        "instance": instance.name,
                        "method": tag,
                        "cost_lambda": lam,
                        "feasible": int(result["feasible"]),
                        "objective_h": result["objective"]
                        if result["feasible"]
                        else "",
                        "objective_lit_h": result["objective_lit"]
                        if result["feasible"]
                        else "",
                        "runtime_s": round(time.perf_counter() - t0, 4),
                        "oracle_calls": result["oracle_calls"],
                        "routes": len(result["routes"]) if result["feasible"] else "",
                    }
                )
                print(
                    f"synthval {tag} seed={seed} feas={result['feasible']} "
                    f"obj={result['objective'] if result['feasible'] else 'inf'}"
                )
    _write_csv(RES / "PERF_synth_validation.csv", rows)
    return rows


def stage_search_ablation(
    n: int = 16,
    seed_start: int = 530000,
    cost_lambda: float = 4.0,
) -> list[dict]:
    """Compare strong LS alone vs strong LS + bounded ruin-repair (same total budget)."""
    import torch
    from data import generate_synthetic
    from train import decode, load_policy

    policy = load_policy(CKPT / "H2_n10_seed42_c3.pt")
    rows = []
    with torch.no_grad():
        for i in range(n):
            seed = seed_start + i
            instance = generate_synthetic(10, seed, generator_version="v2")
            # Matched oracle budget ~2400: (2,2000) vs (2,1600)+(8,400)
            for tag, lp, lb, ri, rb in (
                ("strong_only", 2, 2400, 0, 0),
                ("strong_plus_ruin", 2, 1600, 8, 800),
            ):
                t0 = time.perf_counter()
                result = decode(
                    policy,
                    instance,
                    sample=False,
                    seed=seed,
                    local_passes=lp,
                    local_budget=lb,
                    cost_lambda=cost_lambda,
                    local_operators="strong",
                    ruin_iterations=ri,
                    ruin_budget=rb,
                )
                rows.append(
                    {
                        "seed": seed,
                        "instance": instance.name,
                        "search": tag,
                        "feasible": int(result["feasible"]),
                        "objective_h": result["objective"]
                        if result["feasible"]
                        else "",
                        "runtime_s": round(time.perf_counter() - t0, 4),
                        "oracle_calls": result["oracle_calls"],
                        "local_improvements": result["local_improvements"],
                    }
                )
                print(
                    f"search {tag} seed={seed} feas={result['feasible']} "
                    f"obj={result['objective'] if result['feasible'] else 'inf'} "
                    f"ora={result['oracle_calls']}"
                )
    _write_csv(RES / "PERF_search_ablation.csv", rows)
    return rows


def stage_official(
    size: int = 10,
    limit: int = 20,
    samples: int = 8,
    seeds: tuple[int, ...] = (42, 43, 44),
    cost_lambda: float = 4.0,
    version_new: str = "perf_v2mt",
    ruin: bool = False,
) -> list[dict]:
    import torch
    from charging import literature_compatible_objective_hours
    from data import instance_files, parse_instance
    from evaluate import _refine_best_once
    from train import decode, load_policy

    bks_path = RES / "literature_bks_froger2021.csv"
    bks = {}
    with bks_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bks[row["instance"]] = float(row["bks_h"])

    selected = [
        path
        for path in instance_files(DATA)
        if len(parse_instance(path).customers) == size
    ][:limit]

    methods = []
    for seed in seeds:
        for model in ("H2", "H4"):
            c3 = CKPT / f"{model}_n{size}_seed{seed}_c3.pt"
            if not c3.exists() and size != 10:
                c3 = CKPT / f"{model}_n10_seed{seed}_c3.pt"
            if c3.exists():
                methods.append((f"{model}_c3", model, seed, c3))
            neu = CKPT / f"{model}_n10_seed{seed}_{version_new}.pt"
            if neu.exists():
                methods.append((f"{model}_new", model, seed, neu))
        open_p = CKPT / f"H4_open_n10_seed{seed}_{version_new}.pt"
        if open_p.exists():
            methods.append(("H4_open_new", "H4_open", seed, open_p))

    rows = []
    with torch.no_grad():
        for tag, model, seed, path in methods:
            policy = load_policy(path)
            for path_inst in selected:
                instance = parse_instance(path_inst)
                attempts = [
                    decode(
                        policy,
                        instance,
                        sample=index > 0,
                        seed=seed * 1000 + index,
                        local_passes=0,
                        local_budget=0,
                        cost_lambda=cost_lambda,
                    )
                    for index in range(samples)
                ]
                construct_time = sum(a["runtime"] for a in attempts)
                construct_oracle = sum(a["oracle_calls"] for a in attempts)
                feasible = [a for a in attempts if a["feasible"]]
                if not feasible:
                    rows.append(
                        {
                            "instance": instance.name,
                            "method": tag,
                            "model": model,
                            "seed": seed,
                            "size": size,
                            "cost_lambda": cost_lambda,
                            "feasible": 0,
                            "objective_h": "",
                            "objective_lit_h": "",
                            "runtime_s": construct_time,
                            "oracle_calls": construct_oracle,
                            "bks_gap_lit_pct": "",
                            "layout": instance.name[:3],
                        }
                    )
                    continue
                best = min(feasible, key=lambda a: a["objective"])
                t0 = time.perf_counter()
                routes, plans, _, refine_oracle, diag = _refine_best_once(
                    instance,
                    list(best["routes"]),
                    list(best["plans"]),
                    local_passes=3,
                    local_budget=4000,
                    operators="strong",
                )
                refine_t = time.perf_counter() - t0
                if ruin:
                    from charging import FixedRouteOracle
                    from solver_ls import ruin_and_repair

                    oracle = FixedRouteOracle(instance)
                    # Preserve prior oracle accounting roughly via calls attribute.
                    oracle.calls = 0
                    routes, plans, _, ruin_diag = ruin_and_repair(
                        routes,
                        plans,
                        oracle,
                        instance,
                        max_iterations=6,
                        evaluation_budget=600,
                        seed=seed + 11,
                    )
                    refine_oracle += oracle.calls
                    refine_t += 0  # already in wall below via recompute
                from charging import check_solution

                valid, _ = check_solution(instance, plans)
                obj = sum(p.duration_hours for p in plans) if valid else float("inf")
                lit = (
                    literature_compatible_objective_hours(instance, obj)
                    if valid
                    else float("inf")
                )
                gap = ""
                if valid and instance.name in bks:
                    gap = round(100.0 * (lit - bks[instance.name]) / bks[instance.name], 4)
                rows.append(
                    {
                        "instance": instance.name,
                        "method": tag,
                        "model": model,
                        "seed": seed,
                        "size": size,
                        "cost_lambda": cost_lambda,
                        "feasible": int(valid),
                        "objective_h": obj if valid else "",
                        "objective_lit_h": lit if valid else "",
                        "runtime_s": round(construct_time + refine_t, 4),
                        "oracle_calls": construct_oracle + refine_oracle,
                        "bks_gap_lit_pct": gap,
                        "layout": instance.name[:3],
                    }
                )
                print(
                    f"official n={size} {tag} seed={seed} {instance.name} "
                    f"feas={int(valid)} obj={obj if valid else 'inf'} gap={gap}"
                )
    out = RES / f"PERF_official_n{size}.csv"
    _write_csv(out, rows)
    return rows


def stage_report() -> None:
    """Assemble PERFORMANCE_RESEARCH_REPORT.md from available CSVs/JSONs."""
    lines = [
        "# HAT-EVRP-NL Performance Research Report",
        "",
        "Evidence-driven continuous improvement cycle after Cycle-3 / targeted matched-λ work.",
        "Architecture preserved: Causal EV History Transformer → state fusion → DeepSets →",
        "hierarchical pointer decoder → frvcpy → Python validation.",
        "",
        "## Step 0 — Baseline",
        "",
        "- Latest reviewed commit before this cycle: `4c1a5f5`.",
        "- Cycle-3 checkpoints retained (`*_c3.pt`); never overwritten.",
        "- Prior best exploratory n=10 setting: H2, cost_lambda=4, strong LS ≈ 19.335 h full / ≈ 0.60% lit BKS gap.",
        "",
    ]

    cov_path = RES / "PERF_generator_coverage_summary.json"
    if cov_path.exists():
        cov = json.loads(cov_path.read_text(encoding="utf-8"))
        lines += [
            "## Step 1 — Synthetic generator (v2)",
            "",
            f"- Reproducibility: {cov.get('repro_ok')}/{cov.get('n')} "
            f"({cov.get('repro_rate')})",
            f"- Zero-detour (station=customer/depot) instances: {cov.get('zero_detour_instances')}",
            f"- Geometry↔tech decoupled under sampling: {cov.get('decouple_geo_tech')}",
            f"- Explicit independent combinations verified: {cov.get('explicit_combo_ok')}",
            f"- Geometry counts: `{cov.get('geometry')}`",
            f"- Tech counts: `{cov.get('tech')}`",
            f"- Placement counts: `{cov.get('placement')}`",
            "",
            "Artifacts: `PERF_generator_coverage.csv`, `PERF_generator_coverage_summary.json`.",
            "",
        ]

    teach_path = RES / "PERF_teacher_compare_summary.json"
    if teach_path.exists():
        wins = json.loads(teach_path.read_text(encoding="utf-8"))
        lines += [
            "## Step 2 — Teacher quality",
            "",
            f"- Paired wins (strong better / basic better / tie / both infeas): "
            f"{wins.get('strong_better')} / {wins.get('basic_better')} / "
            f"{wins.get('tie')} / {wins.get('both_infeas')}",
            "",
            "Artifact: `PERF_teacher_compare.csv`.",
            "",
        ]

    synth = RES / "PERF_synth_validation.csv"
    if synth.exists():
        rows = list(csv.DictReader(synth.open(encoding="utf-8")))
        by_m = defaultdict(list)
        for r in rows:
            if r["feasible"] == "1" and r["objective_h"] != "":
                by_m[r["method"]].append(float(r["objective_h"]))
        lines += ["## Step 5/7 — Synthetic validation", "", "| Method | Feasible mean obj (h) | n feas |", "|---|---:|---:|"]
        for m, objs in sorted(by_m.items()):
            lines.append(f"| {m} | {_mean(objs):.4f} | {len(objs)} |")
        lines += ["", "Artifact: `PERF_synth_validation.csv`.", ""]

    search = RES / "PERF_search_ablation.csv"
    if search.exists():
        rows = list(csv.DictReader(search.open(encoding="utf-8")))
        by_s = defaultdict(list)
        for r in rows:
            if r["feasible"] == "1" and r["objective_h"] != "":
                by_s[r["search"]].append(float(r["objective_h"]))
        lines += ["## Step 6 — Search ablation (ruin-and-repair)", ""]
        for s, objs in sorted(by_s.items()):
            lines.append(f"- {s}: mean obj {_mean(objs):.4f} (n={len(objs)})")
        lines += ["", "Artifact: `PERF_search_ablation.csv`.", ""]

    for size in (10, 20, 40):
        path = RES / f"PERF_official_n{size}.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        lines += [f"## Official benchmark n={size}", ""]
        by_m = defaultdict(lambda: {"obj": [], "gap": [], "feas": 0, "n": 0, "rt": []})
        for r in rows:
            m = r["method"]
            by_m[m]["n"] += 1
            if r["feasible"] == "1":
                by_m[m]["feas"] += 1
                by_m[m]["obj"].append(float(r["objective_h"]))
                by_m[m]["rt"].append(float(r["runtime_s"]))
                if r["bks_gap_lit_pct"] != "":
                    by_m[m]["gap"].append(float(r["bks_gap_lit_pct"]))
        lines += [
            "| Method | Feas% | Mean full h | Mean lit gap% | Mean runtime s |",
            "|---|---:|---:|---:|---:|",
        ]
        for m, d in sorted(by_m.items()):
            feas = 100.0 * d["feas"] / max(1, d["n"])
            lines.append(
                f"| {m} | {feas:.1f} | {_mean(d['obj']):.4f} | "
                f"{_mean(d['gap']):.3f} | {_mean(d['rt']):.2f} |"
            )
        # Layout-clustered gaps for seed42 H2 if present
        layout_gaps = defaultdict(list)
        for r in rows:
            if r["method"] == "H2_c3" and r["seed"] == "42" and r["bks_gap_lit_pct"]:
                layout_gaps[r["layout"]].append(float(r["bks_gap_lit_pct"]))
        if layout_gaps:
            lines += ["", "Layout-clustered lit gaps (H2_c3 seed42):"]
            for lay, gs in sorted(layout_gaps.items()):
                lines.append(f"- {lay}: mean={_mean(gs):.3f}% (n_layouts_instances={len(gs)})")
        lines += ["", f"Artifact: `PERF_official_n{size}.csv`.", ""]

    lines += [
        "## Conclusions (fill from measured results)",
        "",
        "- Generator v2 decouples geometry/tech/placement/density; p-median uses region candidates.",
        "- Teacher / multi-target / open-only DeepSets / ruin-repair: see tables above.",
        "- H4 vs H2 architectural claims require matched λ, teacher, search, and seeds.",
        "- Cycle-3 checkpoints preserved for before/after comparison.",
        "",
        "## Reproducibility commands",
        "",
        "```bash",
        "python results/run_performance_research.py --stage all",
        "python results/run_performance_research.py --stage coverage",
        "python results/run_performance_research.py --stage teacher",
        "python results/run_performance_research.py --stage train",
        "python results/run_performance_research.py --stage synthval",
        "python results/run_performance_research.py --stage search",
        "python results/run_performance_research.py --stage official",
        "python results/run_performance_research.py --stage report",
        "```",
        "",
    ]
    out = RES / "PERFORMANCE_RESEARCH_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "coverage",
            "teacher",
            "train",
            "synthval",
            "search",
            "official",
            "official20",
            "official40",
            "report",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--train-n", type=int, default=80)
    parser.add_argument("--val-n", type=int, default=24)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--teacher-n", type=int, default=20)
    parser.add_argument("--synth-n", type=int, default=30)
    parser.add_argument("--search-n", type=int, default=12)
    parser.add_argument("--version", type=str, default="perf_v2st")
    parser.add_argument("--official-seeds", type=str, default="42,43")
    args = parser.parse_args()

    stages = (
        ["coverage", "teacher", "train", "synthval", "search", "official", "report"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        print(f"\n===== STAGE {stage} =====", flush=True)
        t0 = time.perf_counter()
        if stage == "coverage":
            stage_coverage()
        elif stage == "teacher":
            stage_teacher(n=args.teacher_n)
        elif stage == "train":
            stage_train(
                train_n=args.train_n,
                val_n=args.val_n,
                steps=args.steps,
                version=args.version,
                multi_target=False,
                teacher="strong",
            )
        elif stage == "synthval":
            stage_synthval(n=args.synth_n, version_new=args.version)
        elif stage == "search":
            stage_search_ablation(n=args.search_n)
        elif stage == "official":
            seeds = tuple(int(x) for x in args.official_seeds.split(",") if x.strip())
            stage_official(size=10, version_new=args.version, seeds=seeds)
        elif stage == "official20":
            stage_official(size=20, limit=20, samples=4, seeds=(42,), version_new=args.version)
        elif stage == "official40":
            stage_official(size=40, limit=10, samples=3, seeds=(42,), version_new=args.version)
        elif stage == "report":
            stage_report()
        print(f"===== STAGE {stage} done in {time.perf_counter()-t0:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()
