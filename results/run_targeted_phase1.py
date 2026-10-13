"""Phase 1: matched cost_lambda H2 vs H4 under identical strong LS."""
from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import torch

from charging import check_solution, literature_compatible_objective_hours
from data import instance_files, parse_instance
from evaluate import _refine_best_once
from train import decode, load_policy

RES = Path("results")
CKPT = Path("checkpoints")
DATA = Path("dataset")
BKS = pd.read_csv(RES / "literature_bks_froger2021.csv").set_index("instance")


def eval_method(
    method: str,
    lam: float,
    seed: int,
    operators: str = "strong",
    route_lam: float = 0.0,
    size: int = 10,
    limit: int = 20,
    samples: int = 10,
) -> list[dict]:
    policy = load_policy(CKPT / f"{method}_n10_seed{seed}_c3.pt")
    selected = [
        path
        for path in instance_files(DATA)
        if len(parse_instance(path).customers) == size
    ][:limit]
    rows = []
    with torch.no_grad():
        for path in selected:
            instance = parse_instance(path)
            attempts = [
                decode(
                    policy,
                    instance,
                    sample=index > 0,
                    seed=seed * 1000 + index,
                    local_passes=0,
                    local_budget=0,
                    cost_lambda=lam,
                    route_lambda=route_lam,
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
                        "method": method,
                        "seed": seed,
                        "cost_lambda": lam,
                        "route_lambda": route_lam,
                        "operators": operators,
                        "feasible": 0,
                        "objective_h": "",
                        "objective_lit_h": "",
                        "runtime_s": construct_time,
                        "oracle_calls": construct_oracle,
                        "bks_gap_lit_pct": "",
                    }
                )
                continue
            best = min(feasible, key=lambda a: a["objective"])
            t0 = time.perf_counter()
            routes, plans, _, refine_oracle, diag = _refine_best_once(
                instance,
                list(best["routes"]),
                list(best["plans"]),
                3,
                4000,
                operators=operators,
            )
            refine_time = time.perf_counter() - t0
            valid, _ = check_solution(instance, plans)
            obj = sum(p.duration_hours for p in plans) if valid else ""
            lit = (
                literature_compatible_objective_hours(instance, float(obj))
                if valid
                else ""
            )
            gap = ""
            if valid and instance.name in BKS.index:
                bk = float(BKS.loc[instance.name, "bks_h"])
                gap = (float(lit) - bk) / bk * 100.0
            rows.append(
                {
                    "instance": instance.name,
                    "method": method,
                    "seed": seed,
                    "cost_lambda": lam,
                    "route_lambda": route_lam,
                    "operators": operators,
                    "feasible": int(valid),
                    "objective_h": obj if valid else "",
                    "objective_lit_h": lit if valid else "",
                    "runtime_s": construct_time + refine_time,
                    "oracle_calls": construct_oracle + refine_oracle,
                    "bks_gap_lit_pct": gap,
                    "ls_evals": diag.evaluations if valid else "",
                    "ls_ops": str(diag.as_dict()["operators"]) if valid else "",
                }
            )
            print(
                f"{method} lam={lam} route={route_lam} {operators} "
                f"seed={seed} {instance.name}: {obj if valid else 'FAIL'}"
            )
    return rows


def main() -> None:
    # Skip combinations already present from prior strong LS runs by merging later.
    rows = []
    for seed in (42, 43, 44):
        for method in ("H2", "H4"):
            for lam in (0.0, 2.0, 4.0):
                rows.extend(eval_method(method, lam, seed, operators="strong"))
    df = pd.DataFrame(rows)
    out = RES / "TARGETED_matched_lambda_strong_n10.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    for (method, lam), g in df.groupby(["method", "cost_lambda"]):
        f = g[g.feasible == 1]
        print(
            f"{method} lam={lam}: mean={f.objective_h.astype(float).mean():.6f} "
            f"lit_gap={pd.to_numeric(f.bks_gap_lit_pct, errors='coerce').mean():.2f}% "
            f"ora={g.oracle_calls.mean():.1f}"
        )


if __name__ == "__main__":
    main()
