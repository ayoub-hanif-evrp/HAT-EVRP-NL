"""Extract BC/RL diagnostics from authoritative training CSVs (no fabricated rewards)."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting_paths import RAW, RESULTS  # noqa: E402

LOG_DIR = RAW / "training_logs"
OUT_DIR = RAW / "training_logs" / "derived"


def _f(row: dict, key: str) -> float:
    val = row.get(key, "")
    if val in ("", None):
        return float("nan")
    return float(val)


def summarize_log(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return {}
    # Parse filename: training_{MODEL}_n{SIZE}_seed{SEED}_{VERSION}.csv
    stem = path.stem.replace("training_", "")
    parts = stem.split("_")
    # H1_n10_seed42_c3 or H2_n10_seed42_perf_insert_v1
    model = parts[0]
    size = int(parts[1][1:]) if parts[1].startswith("n") else ""
    seed = int(parts[2].replace("seed", "")) if parts[2].startswith("seed") else ""
    version = "_".join(parts[3:]) if len(parts) > 3 else ""

    bc = [r for r in rows if r.get("stage") == "bc"]
    rl = [r for r in rows if r.get("stage") in ("rl", "rl_pomo", "rl_repair")]
    all_eval = [r for r in rows if r.get("validation_objective") not in ("", None)]

    def best_of(subset: list[dict]) -> dict | None:
        feas = [r for r in subset if _f(r, "feasible_rate") >= 1.0 - 1e-12]
        pool = feas or subset
        if not pool:
            return None
        return min(pool, key=lambda r: _f(r, "validation_objective"))

    bc_best = best_of(bc)
    overall_best = best_of(all_eval)
    bc_end = bc[-1] if bc else None
    rl_end = rl[-1] if rl else None

    # Infer whether best checkpoint likely came from RL (log-only; ckpt lacks step).
    best_from = ""
    if overall_best is not None:
        best_from = overall_best.get("stage", "")
        if best_from == "bc" and rl_end is not None:
            # If RL endpoint strictly better than BC best, selection would keep RL.
            if (
                _f(rl_end, "feasible_rate") >= 1.0 - 1e-12
                and _f(rl_end, "validation_objective")
                < _f(bc_best, "validation_objective") - 1e-12
            ):
                best_from = rl_end["stage"]
        if best_from.startswith("rl") and bc_best is not None:
            if _f(overall_best, "validation_objective") > _f(
                bc_best, "validation_objective"
            ) - 1e-12 and overall_best.get("stage", "").startswith("rl"):
                # RL logged but BC was better → deployed weights from BC.
                if _f(bc_best, "validation_objective") < _f(
                    overall_best, "validation_objective"
                ) - 1e-12:
                    best_from = "bc"

    # Correct inference: score = (feas, -val); best among ALL logged points.
    if overall_best is not None:
        best_from = overall_best["stage"]

    delta_rl = float("nan")
    if bc_best is not None and rl_end is not None:
        delta_rl = _f(rl_end, "validation_objective") - _f(
            bc_best, "validation_objective"
        )

    return {
        "log_file": path.name,
        "model": model,
        "size": size,
        "seed": seed,
        "version": version,
        "n_bc_rows": len(bc),
        "n_rl_rows": len(rl),
        "bc_end_val_h": _f(bc_end, "validation_objective") if bc_end else "",
        "bc_end_loss": _f(bc_end, "loss") if bc_end else "",
        "bc_end_entropy": _f(bc_end, "entropy") if bc_end else "",
        "bc_best_val_h": _f(bc_best, "validation_objective") if bc_best else "",
        "bc_best_step": bc_best["step"] if bc_best else "",
        "rl_end_val_h": _f(rl_end, "validation_objective") if rl_end else "",
        "rl_end_loss": _f(rl_end, "loss") if rl_end else "",
        "rl_end_entropy": _f(rl_end, "entropy") if rl_end else "",
        "rl_stage": rl_end["stage"] if rl_end else "",
        "rl_minus_bc_best_h": delta_rl if math.isfinite(delta_rl) else "",
        "rl_improved_vs_bc_best": (
            int(delta_rl < -1e-9) if math.isfinite(delta_rl) else ""
        ),
        "overall_best_val_h": _f(overall_best, "validation_objective")
        if overall_best
        else "",
        "overall_best_stage": overall_best["stage"] if overall_best else "",
        "overall_best_step": overall_best["step"] if overall_best else "",
        "likely_deployed_stage": best_from,
        "sampled_reward_logged": 0,  # historical logs lack reward columns
        "note": "Reward distributions unavailable in historical CSVs",
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_seed_curves(version: str = "c3") -> list[dict]:
    """Long-form per-step rows for plotting (BC loss and RL loss kept separate)."""
    out = []
    for path in sorted(LOG_DIR.glob(f"training_*_n10_seed*_{version}.csv")):
        if "CHARGE" in path.name:
            continue
        summary = summarize_log(path)
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        for r in rows:
            out.append(
                {
                    "model": summary["model"],
                    "seed": summary["seed"],
                    "version": version,
                    "stage": r["stage"],
                    "step": int(r["step"]),
                    "bc_loss": _f(r, "loss") if r["stage"] == "bc" else "",
                    "rl_loss": _f(r, "loss")
                    if r["stage"] in ("rl", "rl_pomo", "rl_repair")
                    else "",
                    "validation_objective": _f(r, "validation_objective"),
                    "feasible_rate": _f(r, "feasible_rate"),
                    "entropy": _f(r, "entropy"),
                    "overall_best_stage": summary["overall_best_stage"],
                    "overall_best_step": summary["overall_best_step"],
                    "overall_best_val_h": summary["overall_best_val_h"],
                }
            )
    return out


def bc_vs_rl_table(version: str = "c3") -> list[dict]:
    rows = []
    for path in sorted(LOG_DIR.glob(f"training_H*_n10_seed*_{version}.csv")):
        if "CHARGE" in path.name or "open" in path.name:
            continue
        s = summarize_log(path)
        if not s:
            continue
        rows.append(s)
    return rows


def aggregate_bc_rl(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        if r["version"] != "c3":
            continue
        by[r["model"]].append(r)
    out = []
    for model, group in sorted(by.items()):
        bc_best = [_f(r, "bc_best_val_h") for r in group]
        rl_end = [_f(r, "rl_end_val_h") for r in group]
        deltas = [_f(r, "rl_minus_bc_best_h") for r in group]
        improved = [int(r["rl_improved_vs_bc_best"] == 1 or r["rl_improved_vs_bc_best"] == "1") for r in group]
        deployed = [r["likely_deployed_stage"] for r in group]
        out.append(
            {
                "model": model,
                "n_seeds": len(group),
                "mean_bc_best_val_h": statistics.mean(bc_best),
                "std_bc_best_val_h": statistics.stdev(bc_best) if len(bc_best) > 1 else 0.0,
                "mean_rl_end_val_h": statistics.mean(rl_end),
                "std_rl_end_val_h": statistics.stdev(rl_end) if len(rl_end) > 1 else 0.0,
                "mean_rl_minus_bc_best_h": statistics.mean(deltas),
                "std_rl_minus_bc_best_h": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "seeds_rl_improved": sum(improved),
                "seeds_deployed_from_bc": sum(1 for d in deployed if d == "bc"),
                "seeds_deployed_from_rl": sum(
                    1 for d in deployed if str(d).startswith("rl")
                ),
                "protocol": "Cycle-3 synthetic validation (n=10, 12 val instances); log-based",
                "reward_traces": "unavailable in historical logs",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="c3")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = bc_vs_rl_table(args.version)
    write_rows(OUT_DIR / f"bc_vs_rl_per_seed_{args.version}.csv", summaries)
    write_rows(OUT_DIR / f"bc_vs_rl_aggregate_{args.version}.csv", aggregate_bc_rl(summaries))
    write_rows(OUT_DIR / f"learning_curves_long_{args.version}.csv", build_seed_curves(args.version))

    # Also summarize selected post-c3 experimental logs (no claim they are Cycle-3).
    exp_rows = []
    for path in sorted(LOG_DIR.glob("training_H*_n10_seed42_perf_*.csv")):
        if "open" in path.name:
            continue
        exp_rows.append(summarize_log(path))
    write_rows(OUT_DIR / "experimental_post_c3_training_summary.csv", exp_rows)
    print(f"wrote derived training summaries under {OUT_DIR} ({len(summaries)} c3 seeds)")


if __name__ == "__main__":
    main()
