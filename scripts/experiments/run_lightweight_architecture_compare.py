"""Lightweight architecture compare using existing checkpoints (no retraining).

Evaluates construction-only (P0) on official n=10 instances for available
checkpoints: H1/H2/H3/H4 Cycle-3 and H4_open if present.

Writes a versioned run directory. This does **not** train H2-prefix (requires
code+training); see experiment_manifest for the deferred prefix-history study.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_io import make_run_directory  # noqa: E402
from charging import (  # noqa: E402
    decompose_solution,
    literature_compatible_objective_hours,
)
from data import instance_files, parse_instance  # noqa: E402
from train import decode, load_policy  # noqa: E402

CKPTS = {
    "H1": ROOT / "checkpoints" / "H1_n10_seed42_c3.pt",
    "H2": ROOT / "checkpoints" / "H2_n10_seed42_c3.pt",
    "H3": ROOT / "checkpoints" / "H3_n10_seed42_c3.pt",
    "H4": ROOT / "checkpoints" / "H4_n10_seed42_c3.pt",
    "H4_open": ROOT / "checkpoints" / "H4_open_n10_seed42_perf_v2mt_s.pt",
}


def main() -> None:
    run_dir = make_run_directory(
        ROOT / "results" / "raw" / "architecture_compare", "lightweight_p0_n10"
    )
    paths = [p for p in instance_files(ROOT / "dataset") if "c10" in p.name]
    paths = sorted(paths)[:20]
    rows = []
    for method, ckpt in CKPTS.items():
        if not ckpt.exists():
            print(f"skip missing {ckpt.name}")
            continue
        model = load_policy(ckpt)
        for path in paths:
            inst = parse_instance(path)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = decode(model, inst, sample=False, seed=42, cost_lambda=0.0)
            rt = time.perf_counter() - t0
            feas = bool(out.get("feasible"))
            full = float(out["objective"]) if feas else float("nan")
            lit = literature_compatible_objective_hours(inst, full) if feas else float("nan")
            decomp = (
                decompose_solution(inst, out["plans"])
                if feas and out.get("plans")
                else None
            )
            rows.append(
                {
                    "protocol": "P0_construction_only",
                    "method": method,
                    "checkpoint": ckpt.name,
                    "instance": inst.name,
                    "size": 10,
                    "training_seed": 42,
                    "eval_seed": 42,
                    "feasible": int(feas),
                    "objective_h": full if feas else "",
                    "lit_h": lit if feas else "",
                    "runtime_s": round(rt, 4),
                    "travel_h": decomp["travel_h"] if decomp else "",
                    "service_h": decomp["service_h"] if decomp else "",
                    "charging_h": decomp["charging_h"] if decomp else "",
                    "n_routes": decomp["n_routes"] if decomp else "",
                    "identity_ok": decomp["identity_ok"] if decomp else "",
                }
            )
            print(method, inst.name, feas, full if feas else "inf", flush=True)

    out_csv = run_dir / "p0_n10_architecture.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    summary = run_dir / "summary.md"
    lines = ["# Lightweight P0 n=10 architecture compare", ""]
    lines.append("| method | feasible | mean Z_full |")
    lines.append("|---|---:|---:|")
    from collections import defaultdict

    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    for method, xs in by.items():
        feas = sum(int(r["feasible"]) for r in xs)
        objs = [float(r["objective_h"]) for r in xs if r["feasible"] == 1]
        mean = sum(objs) / len(objs) if objs else float("nan")
        lines.append(f"| {method} | {feas}/{len(xs)} | {mean:.4f} |")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.read_text(encoding="utf-8"))
    print("wrote", run_dir)


if __name__ == "__main__":
    main()
