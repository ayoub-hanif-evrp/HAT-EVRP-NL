"""Seed-level tests and deployment budgets from the locked official rows.

Does not decode or train. The two central comparisons are the Route Transformer
sampled-mean differences at n=10 and n=20. Other cells are written as exploratory.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "results" / "raw" / "bc_rl_full_v1" / "eval_stochastic" / "official_samples.csv"
BKS = ROOT / "results" / "literature_bks_froger2021.csv"
OUT = ROOT / "results" / "tables" / "final"
FIG = ROOT / "results" / "final" / "figures"
T_CRIT = 2.7764451051977987
SEEDS = (42, 43, 44, 45, 46)
CENTRAL = (("H2", 10, "sampled_mean"), ("H2", 20, "sampled_mean"))


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def holm(pvalues: list[float]) -> list[float]:
    """Step-down Holm adjusted p-values, monotone in the ordered raw p-values."""
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def _tests(deltas: list[float]) -> dict:
    array = np.asarray(deltas, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = T_CRIT * sd / math.sqrt(len(array))
    t_result = stats.ttest_1samp(array, popmean=0.0, alternative="two-sided")
    negatives = int(np.sum(array < 0))
    positives = int(np.sum(array > 0))
    zeros = int(np.sum(np.abs(array) <= 1e-12))
    sign = stats.binomtest(positives, positives + negatives, 0.5, alternative="two-sided")
    wilcox = stats.wilcoxon(array, alternative="two-sided", method="exact", zero_method="wilcox")
    return {
        "n_seeds": len(deltas),
        "mean_h": mean,
        "sd_h": sd,
        "ci_low_h": mean - half,
        "ci_high_h": mean + half,
        "t_stat": float(t_result.statistic),
        "t_p_raw": float(t_result.pvalue),
        "n_negative": negatives,
        "n_positive": positives,
        "n_zero": zeros,
        "sign_p_raw": float(sign.pvalue),
        "wilcoxon_stat": float(wilcox.statistic),
        "wilcoxon_p_raw": float(wilcox.pvalue),
    }


def seed_tables(raw: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Return per-seed rows, test rows, and instance-level sampled means."""
    sample_groups: dict[tuple, list[dict]] = defaultdict(list)
    greedy_groups: dict[tuple, dict] = {}
    for row in raw:
        key = (row["model"], int(row["seed"]), row["role"], int(row["size"]), row["instance"])
        if row["protocol"] == "sample":
            sample_groups[key].append(row)
        else:
            greedy_groups[key] = row
    instance_means: dict[tuple, dict] = {}
    for key, trials in sample_groups.items():
        if len(trials) != 4:
            raise SystemExit(f"{key} has {len(trials)} samples")
        objectives = [float(trial["objective_h"]) for trial in trials]
        best = min(trials, key=lambda trial: float(trial["objective_h"]))
        instance_means[key] = {
            "mean_h": _mean(objectives),
            "best_h": float(best["objective_h"]),
            "best_lit_h": float(best["objective_lit_h"]),
            "sample_sd_h": float(np.std(objectives, ddof=1)),
            "runtime_s": sum(float(trial["runtime_s"]) for trial in trials),
            "one_sample_runtime_s": _mean([float(trial["runtime_s"]) for trial in trials]),
            "oracle_calls": sum(float(trial["oracle_calls"]) for trial in trials),
            "oracle_requests": sum(float(trial["oracle_requests"]) for trial in trials),
            "oracle_cache_hits": sum(float(trial["oracle_cache_hits"]) for trial in trials),
        }
    seed_rows = []
    for model in ("H1", "H2", "H3", "H4"):
        for size in (10, 20, 40):
            for protocol in ("sampled_mean", "best_of_4"):
                field = "mean_h" if protocol == "sampled_mean" else "best_h"
                for seed in SEEDS:
                    bc = []
                    rl = []
                    instances = sorted(
                        {
                            key[4]
                            for key in instance_means
                            if key[0] == model and key[1] == seed and key[3] == size and key[2] == "bc"
                        }
                    )
                    if len(instances) != 20:
                        raise SystemExit(f"{model} seed {seed} n={size} has {len(instances)} instances")
                    for instance in instances:
                        bc.append(instance_means[(model, seed, "bc", size, instance)][field])
                        rl.append(instance_means[(model, seed, "rl", size, instance)][field])
                    seed_rows.append(
                        {
                            "model": model,
                            "size": size,
                            "protocol": protocol,
                            "seed": seed,
                            "bc_h": _mean(bc),
                            "rl_h": _mean(rl),
                            "delta_h": _mean(rl) - _mean(bc),
                            "instances_rl_lower": sum(right < left - 1e-8 for left, right in zip(bc, rl)),
                            "status": "central" if (model, size, protocol) in CENTRAL else "exploratory",
                        }
                    )
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_rows:
        by_cell[(row["model"], row["size"], row["protocol"])].append(row)
    test_rows = []
    families = {
        "central_sign": [],
        "central_wilcoxon": [],
        "central_t": [],
    }
    for key, rows in by_cell.items():
        deltas = [float(row["delta_h"]) for row in rows]
        bc = _mean([float(row["bc_h"]) for row in rows])
        result = _tests(deltas)
        status = "central" if key in CENTRAL else "exploratory"
        record = {
            "model": key[0],
            "size": key[1],
            "protocol": key[2],
            "status": status,
            "bc_h": bc,
            "rl_h": bc + result["mean_h"],
            "percent": 100.0 * result["mean_h"] / bc,
            **result,
            "seed_deltas_h": "|".join(f"{delta:.6f}" for delta in deltas),
        }
        test_rows.append(record)
        if status == "central":
            families["central_sign"].append(record)
            families["central_wilcoxon"].append(record)
            families["central_t"].append(record)
    for family, field in (
        ("central_sign", "sign_p_raw"),
        ("central_wilcoxon", "wilcoxon_p_raw"),
        ("central_t", "t_p_raw"),
    ):
        ordered = sorted(families[family], key=lambda row: (row["size"], row["protocol"]))
        adjusted = holm([row[field] for row in ordered])
        for row, value in zip(ordered, adjusted):
            row[field.replace("_raw", "_holm")] = value
    for row in test_rows:
        for name in ("sign_p_holm", "wilcoxon_p_holm", "t_p_holm"):
            row.setdefault(name, "")
    return seed_rows, test_rows, {"samples": instance_means, "greedy": greedy_groups}


def deployment_table(packed: dict, bks: dict[str, float]) -> list[dict]:
    rows = []
    samples = packed["samples"]
    greedy = packed["greedy"]
    for model in ("H2", "H4"):
        for size in (10, 20, 40):
            for option, role, kind in (
                ("BC greedy", "bc", "greedy"),
                ("RL greedy", "rl", "greedy"),
                ("BC best of 4", "bc", "best"),
                ("RL best of 4", "rl", "best"),
            ):
                seed_obj = []
                seed_gap = []
                seed_runtime = []
                seed_calls = []
                seed_requests = []
                seed_hits = []
                seed_feasible = []
                for seed in SEEDS:
                    instances = sorted(
                        key[4]
                        for key in (greedy if kind == "greedy" else samples)
                        if key[0] == model and key[1] == seed and key[2] == role and key[3] == size
                    )
                    objectives = []
                    gaps = []
                    runtime = calls = requests = hits = 0.0
                    feasible = 0
                    for instance in instances:
                        if kind == "greedy":
                            row = greedy[(model, seed, role, size, instance)]
                            objectives.append(float(row["objective_h"]))
                            lit = float(row["objective_lit_h"])
                            runtime += float(row["runtime_s"])
                            calls += float(row["oracle_calls"])
                            requests += float(row["oracle_requests"])
                            hits += float(row["oracle_cache_hits"])
                            feasible += int(row["feasible"])
                        else:
                            item = samples[(model, seed, role, size, instance)]
                            objectives.append(item["best_h"])
                            lit = item["best_lit_h"]
                            runtime += item["runtime_s"]
                            calls += item["oracle_calls"]
                            requests += item["oracle_requests"]
                            hits += item["oracle_cache_hits"]
                            feasible += 1
                        if instance in bks:
                            gaps.append(100.0 * (lit - bks[instance]) / bks[instance])
                    seed_obj.append(_mean(objectives))
                    seed_gap.append(_mean(gaps) if gaps else float("nan"))
                    seed_runtime.append(runtime)
                    seed_calls.append(calls)
                    seed_requests.append(requests)
                    seed_hits.append(hits)
                    seed_feasible.append(feasible / len(instances))
                rows.append(
                    {
                        "model": model,
                        "size": size,
                        "option": option,
                        "objective_h": _mean(seed_obj),
                        "seed_sd_h": float(np.std(seed_obj, ddof=1)),
                        "feasible_rate": _mean(seed_feasible),
                        "runtime_s": _mean(seed_runtime),
                        "oracle_calls": _mean(seed_calls),
                        "oracle_requests": _mean(seed_requests),
                        "oracle_cache_hits": _mean(seed_hits),
                        "reference_2019_gap_pct": "" if math.isnan(seed_gap[0]) else _mean(seed_gap),
                        "note": (
                            "cost is the sum of four sampled decodes; the objective is the selected route"
                            if kind == "best"
                            else "one greedy decode"
                        ),
                    }
                )
    return rows


def expectation_table(packed: dict) -> list[dict]:
    """Mean of four samples estimates one draw. Four evaluations buy precision."""
    rows = []
    samples = packed["samples"]
    for model in ("H2", "H4"):
        for size in (10, 20, 40):
            for role, label in (("bc", "BC"), ("rl", "RL")):
                seed_mean = []
                seed_one = []
                seed_four = []
                within = []
                for seed in SEEDS:
                    items = [
                        samples[key]
                        for key in samples
                        if key[0] == model and key[1] == seed and key[2] == role and key[3] == size
                    ]
                    seed_mean.append(_mean([item["mean_h"] for item in items]))
                    seed_one.append(sum(item["one_sample_runtime_s"] for item in items))
                    seed_four.append(sum(item["runtime_s"] for item in items))
                    within.append(_mean([item["sample_sd_h"] for item in items]))
                rows.append(
                    {
                        "model": model,
                        "size": size,
                        "policy": label,
                        "expected_one_draw_h": _mean(seed_mean),
                        "mean_within_instance_sample_sd_h": _mean(within),
                        "runtime_one_draw_s": _mean(seed_one),
                        "runtime_four_draws_used_to_estimate_s": _mean(seed_four),
                    }
                )
    return rows


def hierarchical(packed: dict, draws: int = 4000) -> list[dict]:
    """Cluster bootstrap over five seeds, then instances inside a seed.

    The four samples stay inside the instance mean. They are not resampled as
    training replicates. Five clusters make the interval exploratory.
    """
    rng = np.random.default_rng(91_201)
    rows = []
    samples = packed["samples"]
    for size in (10, 20):
        seed_maps = []
        for seed in SEEDS:
            instances = sorted(
                key[4]
                for key in samples
                if key[0] == "H2" and key[1] == seed and key[2] == "bc" and key[3] == size
            )
            deltas = np.array(
                [
                    samples[("H2", seed, "rl", size, instance)]["mean_h"]
                    - samples[("H2", seed, "bc", size, instance)]["mean_h"]
                    for instance in instances
                ]
            )
            seed_maps.append(deltas)
        replicates = []
        for _ in range(draws):
            chosen = rng.integers(0, 5, size=5)
            seed_means = []
            for index in chosen:
                draw = rng.integers(0, 20, size=20)
                seed_means.append(float(seed_maps[index][draw].mean()))
            replicates.append(float(np.mean(seed_means)))
        array = np.asarray(replicates)
        rows.append(
            {
                "model": "H2",
                "size": size,
                "protocol": "sampled_mean",
                "draws": draws,
                "mean_h": float(array.mean()),
                "p2_5_h": float(np.quantile(array, 0.025)),
                "p97_5_h": float(np.quantile(array, 0.975)),
                "fraction_negative": float(np.mean(array < 0)),
                "clusters": 5,
                "status": "exploratory",
            }
        )
    return rows


def figure(deployment: list[dict]) -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
    colors = {"BC greedy": "#0072B2", "RL greedy": "#56B4E9", "BC best of 4": "#D55E00", "RL best of 4": "#E69F00"}
    markers = {"BC greedy": "o", "RL greedy": "s", "BC best of 4": "^", "RL best of 4": "D"}
    chosen = [row for row in deployment if row["model"] == "H2"]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6), sharex=False, sharey=False)
    for column, size in enumerate((10, 20, 40)):
        for row_index, (xkey, xlabel) in enumerate(
            (("runtime_s", "Time for 20 instances (s)"), ("oracle_calls", "frvcpy executions"))
        ):
            ax = axes[row_index, column]
            for option in colors:
                item = next(row for row in chosen if row["option"] == option and row["size"] == size)
                ax.errorbar(
                    [item[xkey]],
                    [item["objective_h"]],
                    yerr=[item["seed_sd_h"]],
                    color=colors[option],
                    marker=markers[option],
                    linestyle="none",
                    label=option,
                    capsize=2,
                )
            ax.set_title(f"n = {size}")
            ax.set_xlabel(xlabel)
            if column == 0:
                ax.set_ylabel("Returned route (h)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_quality_vs_compute.png")
    plt.close(fig)


def main() -> None:
    raw = _rows(SAMPLES)
    bks = {row["instance"]: float(row["bks_h"]) for row in _rows(BKS)}
    seed_rows, test_rows, packed = seed_tables(raw)
    deployment = deployment_table(packed, bks)
    expectation = expectation_table(packed)
    boots = hierarchical(packed)
    _write(OUT / "central_seed_differences.csv", seed_rows)
    _write(OUT / "statistical_sensitivity.csv", test_rows)
    _write(OUT / "deployment_options.csv", deployment)
    _write(OUT / "sample_expectation_vs_selection.csv", expectation)
    _write(OUT / "hierarchical_bootstrap.csv", boots)
    figure(deployment)
    for row in test_rows:
        if row["status"] == "central":
            print(
                row["model"],
                row["size"],
                f"delta={row['mean_h']:.4f}",
                f"t_p={row['t_p_raw']:.4g}",
                f"sign={row['sign_p_raw']:.4g}",
                f"wilcox={row['wilcoxon_p_raw']:.4g}",
                f"holm_sign={row['sign_p_holm']:.4g}",
            )


if __name__ == "__main__":
    main()
