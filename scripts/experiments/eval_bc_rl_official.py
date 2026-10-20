"""Official evaluation of one BC→RL run. Not a training phase.

P0 construction-only, P1 one-shot basic local search (1 pass, budget 800),
P2 hybrid insertion (classical ruin budget 600, then neural insert, 6 iterations).

P2 oracle counts and runtime cover construction plus classical ruin plus neural
repair. ``oracle_calls`` is the number of frvcpy solver executions.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import instance_files, parse_instance
from train import decode, hybrid_lns, load_policy

P2_ACCOUNTING = "construction_plus_hybrid"


def instances_of_size(data_dir: Path, size: int) -> list[Path]:
    return [path for path in instance_files(data_dir) if len(parse_instance(path).customers) == size]


def _as_int(value) -> int:
    if value in ("", None):
        return 0
    return int(float(value))


def combine_oracle_counts(base: dict, extra: dict | None) -> dict[str, int]:
    """Add construction-oracle counts to a later search phase.

    ``oracle_calls`` follows decode(): frvcpy executions, not cache hits.
    The hybrid diagnostic uses ``oracle_solver_executions`` for that same count.
    """
    if not extra:
        return {
            "oracle_calls": _as_int(base.get("oracle_calls")),
            "oracle_requests": _as_int(base.get("oracle_requests")),
            "oracle_cache_hits": _as_int(base.get("oracle_cache_hits")),
        }
    return {
        "oracle_calls": _as_int(base.get("oracle_calls")) + _as_int(extra.get("oracle_solver_executions")),
        "oracle_requests": _as_int(base.get("oracle_requests")) + _as_int(extra.get("oracle_requests")),
        "oracle_cache_hits": _as_int(base.get("oracle_cache_hits")) + _as_int(extra.get("oracle_cache_hits")),
    }


def evaluate_neural_construction(run_dir: Path, data_dir: Path, out_csv: Path, sizes=(10, 20, 40, 80)) -> None:
    """Score BC and RL checkpoints by neural construction only.

    This is the primary evaluation. It does not call local search, ruin-and-repair,
    or hybrid insertion. Those remain in ``evaluate_run``.
    """
    rows = []
    for role, filename in (("bc_best", "bc_best.pt"), ("rl_best", "rl_best.pt")):
        model = load_policy(run_dir / filename)
        for size in sizes:
            paths = instances_of_size(data_dir, size)
            for path in paths:
                instance = parse_instance(path)
                result = decode(model, instance, sample=False)
                rows.append(
                    {
                        "role": role,
                        "size": size,
                        "n_available": len(paths),
                        "protocol": "neural-construction",
                        "instance": instance.name,
                        "feasible": int(bool(result["feasible"])),
                        "objective_h": result["objective"] if result["feasible"] else "",
                        "objective_lit_h": result.get("objective_lit", "") if result["feasible"] else "",
                        "runtime_s": result.get("runtime", ""),
                        "oracle_calls": result.get("oracle_calls", ""),
                        "oracle_requests": result.get("oracle_requests", ""),
                        "oracle_cache_hits": result.get("oracle_cache_hits", ""),
                        "reason": result.get("reason", ""),
                        "accounting": "decode",
                    }
                )
                print(
                    {"role": role, "protocol": "neural-construction", "instance": instance.name, "feasible": rows[-1]["feasible"]},
                    flush=True,
                )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
                        result = evaluate_p2(model, instance)
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
                            "accounting": "decode" if protocol != "P2" else P2_ACCOUNTING,
                        }
                    )
                    print({"role": role, "protocol": protocol, "instance": instance.name, "feasible": rows[-1]["feasible"]}, flush=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_p2(model, instance) -> dict:
    """Construction followed by hybrid insertion, with joint cost accounting."""
    from charging import check_solution, literature_compatible_objective_hours

    started = time.perf_counter()
    base = decode(model, instance, sample=False, local_passes=0, local_budget=0)
    if not base["feasible"]:
        base = dict(base)
        base["runtime"] = time.perf_counter() - started
        return base
    routes, plans, _imp, diag = hybrid_lns(
        model,
        instance,
        base["routes"],
        base["plans"],
        max_iterations=6,
        seed=0,
        classical_iterations=6,
        classical_budget=600,
    )
    ok, reason = check_solution(instance, plans)
    objective = sum(plan.duration_hours for plan in plans) if ok else float("inf")
    counts = combine_oracle_counts(base, diag)
    return {
        "feasible": ok,
        "objective": objective,
        "objective_lit": literature_compatible_objective_hours(instance, objective) if ok else float("inf"),
        "runtime": time.perf_counter() - started,
        "reason": reason,
        **counts,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refresh_p2_csv(run_dir: Path, data_dir: Path, csv_path: Path, mismatch_log: Path) -> int:
    """Recompute P2 rows in an existing official CSV. P0 and P1 stay untouched."""
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty eval csv: {csv_path}")
    pending = [
        index
        for index, row in enumerate(rows)
        if row["protocol"] == "P2" and row.get("accounting") != P2_ACCOUNTING
    ]
    if not pending:
        print({"skip": csv_path.name, "reason": "p2 already accounted"}, flush=True)
        return 0
    models = {}
    catalog: dict[int, dict[str, object]] = {}
    mismatches = 0
    for done, index in enumerate(pending, start=1):
        row = rows[index]
        role = row["role"]
        size = int(row["size"])
        if role not in models:
            models[role] = load_policy(run_dir / f"{role}.pt")
        if size not in catalog:
            loaded = {}
            for path in instances_of_size(data_dir, size):
                parsed = parse_instance(path)
                loaded[parsed.name] = parsed
            catalog[size] = loaded
        instance = catalog[size][row["instance"]]
        result = evaluate_p2(models[role], instance)
        previous = row.get("objective_h", "")
        new_obj = result["objective"] if result["feasible"] else ""
        if previous != "" and new_obj != "" and abs(float(previous) - float(new_obj)) > 1e-4:
            mismatches += 1
            with mismatch_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{csv_path.name},{row['role']},{row['size']},{row['instance']},{previous},{new_obj}\n"
                )
        row["feasible"] = str(int(bool(result["feasible"])))
        row["objective_h"] = new_obj if new_obj != "" else ""
        row["objective_lit_h"] = result.get("objective_lit", "") if result["feasible"] else ""
        row["runtime_s"] = result.get("runtime", "")
        row["oracle_calls"] = result.get("oracle_calls", "")
        row["oracle_requests"] = result.get("oracle_requests", "")
        row["oracle_cache_hits"] = result.get("oracle_cache_hits", "")
        row["reason"] = result.get("reason", "")
        row["accounting"] = P2_ACCOUNTING
        if done % 5 == 0 or done == len(pending):
            _ensure_accounting_field(rows)
            _write_rows(csv_path, rows)
            print({"csv": csv_path.name, "p2_done": done, "p2_total": len(pending), "mismatches": mismatches}, flush=True)
    _ensure_accounting_field(rows)
    _write_rows(csv_path, rows)
    return mismatches


def _ensure_accounting_field(rows: list[dict]) -> None:
    for row in rows:
        row.setdefault("accounting", "decode" if row.get("protocol") != "P2" else "")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--refresh-p2":
        root = Path(sys.argv[2])
        data_dir = Path(sys.argv[3])
        mismatch_log = root / "eval" / "p2_objective_mismatches.csv"
        if not mismatch_log.exists() or mismatch_log.stat().st_size == 0:
            mismatch_log.write_text("csv,role,size,instance,previous_h,rerun_h\n", encoding="utf-8")
        total = 0
        for csv_path in sorted((root / "eval").glob("H*_seed*.csv")):
            run_dir = root / "runs" / csv_path.stem
            print({"refresh": csv_path.name}, flush=True)
            total += refresh_p2_csv(run_dir, data_dir, csv_path, mismatch_log)
        print({"objective_mismatches": total}, flush=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--evaluation" and sys.argv[2] == "neural-construction":
        evaluate_neural_construction(Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    else:
        evaluate_run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
