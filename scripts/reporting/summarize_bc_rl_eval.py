"""Paired BC-versus-RL summary of the official five-seed evaluation.

The unit of variation is the seed. Each seed contributes one mean over the
instances of that size and protocol. The interval is a Student t 95% interval
on the five paired differences (RL mean − BC mean), df = 4.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presentation_names import method_label  # noqa: E402

EVAL = ROOT / "results" / "raw" / "bc_rl_full_v1" / "eval"
OUT = ROOT / "results" / "tables" / "bc_rl_full_v1_paired.csv"
T_95_DF4 = 2.7764451051977987
MODELS = ("H1", "H2", "H3", "H4")
SIZES = (10, 20, 40, 80)
PROTOCOLS = ("P0", "P1", "P2")


def seed_means() -> tuple[dict, dict]:
    quality = defaultdict(list)
    coverage = defaultdict(lambda: [0, 0])
    for path in sorted(EVAL.glob("H*_seed*.csv")):
        model, _seed = path.stem.split("_seed")
        buckets = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (model, row["role"], int(row["size"]), row["protocol"])
                coverage[key][1] += 1
                if row["feasible"] == "1" and row["objective_h"] != "":
                    buckets[key].append(float(row["objective_h"]))
                    coverage[key][0] += 1
        for key, values in buckets.items():
            quality[key].append(sum(values) / len(values))
    return quality, coverage


def rows() -> list[dict]:
    quality, coverage = seed_means()
    written = []
    for size in SIZES:
        for protocol in PROTOCOLS:
            for model in MODELS:
                bc = quality[(model, "bc_best", size, protocol)]
                rl = quality[(model, "rl_best", size, protocol)]
                deltas = [right - left for left, right in zip(bc, rl)]
                mean = statistics.mean(deltas)
                sd = statistics.stdev(deltas)
                se = sd / math.sqrt(len(deltas))
                feas, total = coverage[(model, "bc_best", size, protocol)]
                feas_rl, total_rl = coverage[(model, "rl_best", size, protocol)]
                written.append(
                    {
                        "model": method_label(model),
                        "model_key": model,
                        "size": size,
                        "protocol": protocol,
                        "bc_mean_h": round(statistics.mean(bc), 3),
                        "rl_mean_h": round(statistics.mean(rl), 3),
                        "delta_rl_minus_bc_h": round(mean, 3),
                        "delta_sd_h": round(sd, 3),
                        "ci95_low_h": round(mean - T_95_DF4 * se, 3),
                        "ci95_high_h": round(mean + T_95_DF4 * se, 3),
                        "rl_better_seeds": sum(delta < 0 for delta in deltas),
                        "n_seeds": len(deltas),
                        "feasible_rows": feas + feas_rl,
                        "n_rows": total + total_rl,
                    }
                )
    return written


def main() -> None:
    table = rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote {OUT}")
    for size in SIZES:
        print(f"\n### n={size}")
        print("| Model | Protocol | BC | RL | RL-BC | 95% interval | RL better |")
        print("|---|---|---:|---:|---:|---|---:|")
        for row in table:
            if row["size"] != size:
                continue
            interval = f"[{row['ci95_low_h']:+.3f}, {row['ci95_high_h']:+.3f}]"
            print(
                f"| {row['model']} | {row['protocol']} | {row['bc_mean_h']:.3f} | {row['rl_mean_h']:.3f} | "
                f"{row['delta_rl_minus_bc_h']:+.3f} | {interval} | {row['rl_better_seeds']}/5 |"
            )


if __name__ == "__main__":
    main()
