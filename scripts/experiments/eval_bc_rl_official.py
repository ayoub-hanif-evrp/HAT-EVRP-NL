"""Official evaluation of one BC→RL run. Not a training phase.

P0 construction-only, P1 one-shot basic local search (1 pass, budget 800),
P2 hybrid insertion (classical ruin budget 600, then neural insert, 6 iterations).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import instance_files, parse_instance
from train import decode, hybrid_lns, load_policy


def instances_of_size(data_dir: Path, size: int) -> list[Path]:
    return [path for path in instance_files(data_dir) if len(parse_instance(path).customers) == size]


def evaluate_run(run_dir: Path, data_dir: Path, out_csv: Path, sizes=(10, 20, 40, 80)) -> None:
    rows = []
    for role, filename in (("bc_best", "bc_best.pt"), ("rl_best", "rl_best.pt")):
        model = load_policy(run_dir / filename)
        for size in sizes:
            paths = instances_of_size(data_dir, size)
            for protocol in ("P0", "P1", "P2"):
                for path in paths:
                    instance = parse_instance(path)
                    if protocol == "P0":
                        result = decode(model, instance, sample=False, local_passes=0, local_budget=0)
                    elif protocol == "P1":
                        result = decode(
                            model,
                            instance,
                            sample=False,
                            local_passes=1,
                            local_budget=800,
                            local_operators="basic",
                        )
                    else:
                        base = decode(model, instance, sample=False, local_passes=0, local_budget=0)
                        if base["feasible"]:
                            routes, plans, _imp, _diag = hybrid_lns(
                                model,
                                instance,
                                base["routes"],
                                base["plans"],
                                max_iterations=6,
                                seed=0,
                                classical_iterations=6,
                                classical_budget=600,
                            )
                            from charging import check_solution, literature_compatible_objective_hours

                            ok, reason = check_solution(instance, plans)
                            objective = sum(plan.duration_hours for plan in plans) if ok else float("inf")
                            result = {
                                "feasible": ok,
                                "objective": objective,
                                "objective_lit": literature_compatible_objective_hours(instance, objective) if ok else float("inf"),
                                "runtime": base["runtime"],
                                "oracle_calls": base["oracle_calls"],
                                "oracle_requests": base.get("oracle_requests", ""),
                                "oracle_cache_hits": base.get("oracle_cache_hits", ""),
                                "reason": reason,
                            }
                        else:
                            result = base
                    rows.append(
                        {
                            "role": role,
                            "size": size,
                            "n_available": len(paths),
                            "protocol": protocol,
                            "instance": instance.name,
                            "feasible": int(bool(result["feasible"])),
                            "objective_h": result["objective"] if result["feasible"] else "",
                            "objective_lit_h": result.get("objective_lit", "") if result["feasible"] else "",
                            "runtime_s": result.get("runtime", ""),
                            "oracle_calls": result.get("oracle_calls", ""),
                            "oracle_requests": result.get("oracle_requests", ""),
                            "oracle_cache_hits": result.get("oracle_cache_hits", ""),
                            "reason": result.get("reason", ""),
                        }
                    )
                    print({"role": role, "protocol": protocol, "instance": instance.name, "feasible": rows[-1]["feasible"]}, flush=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    evaluate_run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
