"""One locked official evaluation. Construction only.

Greedy neural construction on every available Montoya instance at n=10, 20, 40,
and 80. Stochastic construction uses the same K=4 sample-seed schedule as the
validation audit, on n=10 and n=20. No local search, ruin-and-repair, or hybrid
insertion. Rows are appended, so a stopped run resumes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import parse_instance
from train import decode, load_policy

sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from eval_bc_rl_official import instances_of_size

K = 4
SAMPLE_SEED0 = 910_000
FIELDS = [
    "model",
    "seed",
    "role",
    "size",
    "instance_index",
    "instance",
    "protocol",
    "trial",
    "feasible",
    "objective_h",
    "objective_lit_h",
    "runtime_s",
    "oracle_calls",
    "oracle_requests",
    "oracle_cache_hits",
]


def _done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["model"], row["seed"], row["role"], row["size"], row["instance"], row["protocol"], row["trial"])
            for row in csv.DictReader(handle)
        }


def _row(model, seed, role, size, index, instance, protocol, trial, result) -> dict:
    feasible = bool(result.get("feasible"))
    return {
        "model": model,
        "seed": seed,
        "role": role,
        "size": size,
        "instance_index": index,
        "instance": instance.name,
        "protocol": protocol,
        "trial": trial,
        "feasible": int(feasible),
        "objective_h": result["objective"] if feasible else "",
        "objective_lit_h": result.get("objective_lit", "") if feasible else "",
        "runtime_s": result.get("runtime", ""),
        "oracle_calls": result.get("oracle_calls", ""),
        "oracle_requests": result.get("oracle_requests", ""),
        "oracle_cache_hits": result.get("oracle_cache_hits", ""),
    }


def evaluate(run_root: Path, out_csv: Path, models, seeds, greedy_sizes, sample_sizes) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out_csv)
    new_file = not out_csv.exists()
    data_dir = ROOT / "dataset"
    with out_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for model in models:
            for seed in seeds:
                run_dir = run_root / "runs" / f"{model}_seed{seed}"
                for role, filename in (("bc", "bc_best.pt"), ("rl", "rl_best.pt")):
                    path = run_dir / filename
                    if not path.exists():
                        print({"missing": str(path)}, flush=True)
                        continue
                    policy = load_policy(path)
                    for size in greedy_sizes:
                        paths = instances_of_size(data_dir, size)
                        for index, instance_path in enumerate(paths):
                            instance = parse_instance(instance_path)
                            key = (model, str(seed), role, str(size), instance.name, "greedy", "0")
                            if key not in done:
                                with torch.no_grad():
                                    result = decode(policy, instance, sample=False, seed=0)
                                writer.writerow(_row(model, seed, role, size, index, instance, "greedy", 0, result))
                                handle.flush()
                                done.add(key)
                            if size not in sample_sizes:
                                continue
                            for trial in range(K):
                                sample_key = (model, str(seed), role, str(size), instance.name, "sample", str(trial))
                                if sample_key in done:
                                    continue
                                with torch.no_grad():
                                    result = decode(
                                        policy,
                                        instance,
                                        sample=True,
                                        seed=SAMPLE_SEED0 + index * 16 + trial,
                                    )
                                writer.writerow(
                                    _row(model, seed, role, size, index, instance, "sample", trial, result)
                                )
                                handle.flush()
                                done.add(sample_key)
                    print({"evaluated": f"{model}_seed{seed}:{role}"}, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["H2", "H4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--greedy-sizes", nargs="+", type=int, default=[10, 20, 40, 80])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[10, 20])
    args = parser.parse_args()
    evaluate(args.run_root, args.out, args.models, args.seeds, args.greedy_sizes, args.sample_sizes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
