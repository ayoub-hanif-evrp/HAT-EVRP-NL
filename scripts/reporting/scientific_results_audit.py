"""Independently recompute main numerical claims from authoritative CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "SCIENTIFIC_RESULTS_AUDIT.md"
SERVICE = {10: 5.0, 20: 10.0, 40: 20.0, 80: 40.0}


def _gap_lit(obj_full: float, bks: float, n: int) -> float:
    return 100.0 * (obj_full - SERVICE[n] - bks) / bks


def section_a() -> list[str]:
    lines = ["## A. Nonlinear charging — fixed-route", ""]
    fr = pd.read_csv(RAW / "nonlinear_charging" / "fixed_route_charging_comparison.csv")
    # Prefer one sequence per instance (seed/checkpoint already fixed in study)
    for size in [10, 20, 40]:
        sub = fr[fr["size"] == size]
        lines.append(f"### size={size} (rows={len(sub)})")
        lines.append("")
        lines.append("| model | n | planner_ok | true_feas | plan_reject | false_feas | true_feas_rate |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for model in ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]:
            m = sub[sub["charging_model"] == model]
            # one row per instance: take first checkpoint/seed group if multiple
            if m.empty:
                continue
            # aggregate by instance (any sequence failure counts)
            # Study stores one planner outcome per instance×model typically with n_sequences
            n_inst = m["instance"].nunique()
            # Prefer rows that represent the sequence-level study: use unique instance
            g = m.groupby("instance").agg(
                planner=("feasible_under_planner", "max"),
                true=("feasible_under_true_physics", "max"),
            )
            # Actually for fixed-route comparison the CSV may have one row per instance×model
            # Recompute from raw rows if one-per-instance:
            if len(m) == n_inst:
                planner_ok = int(m["feasible_under_planner"].sum())
                true_feas = int(m["feasible_under_true_physics"].sum())
                plan_reject = int((m["feasible_under_planner"] == 0).sum())
                false_feas = int(
                    ((m["feasible_under_planner"] == 1) & (m["feasible_under_true_physics"] == 0)).sum()
                )
            else:
                # Multiple sequences: instance is true-feasible if ANY planned sequence is true-feasible?
                # User claim is about n=40 with 20 instances. Use majority: take first n_sequences group
                # Better: use per-instance: true if max true_physics among sequences where we evaluate the study's primary metric
                # Looking at protocol: typically one evaluation per instance. Deduplicate keeping all sequences then:
                # Count instances where at least one sequence is true-feasible under that model.
                by = m.groupby("instance")
                planner_ok = int((by["feasible_under_planner"].max() == 1).sum())
                true_feas = int((by["feasible_under_true_physics"].max() == 1).sum())
                plan_reject = int((by["feasible_under_planner"].max() == 0).sum())
                false_feas = int(
                    (
                        (by["feasible_under_planner"].max() == 1)
                        & (by["feasible_under_true_physics"].max() == 0)
                    ).sum()
                )
                n_inst = len(by)
            lines.append(
                f"| {model} | {n_inst} | {planner_ok} | {true_feas} | {plan_reject} | {false_feas} | {true_feas}/{n_inst} |"
            )
        lines.append("")
        # Detailed n40 failure modes from per-row if one row per instance
        if size == 40:
            lines.append("Per-instance n40 detail (deduped by instance, first row if multiples):")
            lines.append("")
            for model in ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]:
                m = sub[sub["charging_model"] == model].drop_duplicates("instance", keep="first")
                lines.append(f"**{model}** n={len(m)}")
                vc = m.apply(
                    lambda r: (
                        "true_feas"
                        if r["feasible_under_true_physics"] == 1
                        else (
                            "plan_reject"
                            if r["feasible_under_planner"] == 0
                            else "false_feas"
                        )
                    ),
                    axis=1,
                ).value_counts()
                lines.append(str(vc.to_dict()))
                # Also show OR across sequences if multi
                mm = sub[sub["charging_model"] == model]
                if len(mm) != len(m):
                    by = mm.groupby("instance")
                    cats = []
                    for _, g in by:
                        if g["feasible_under_true_physics"].max() == 1:
                            cats.append("true_feas")
                        elif g["feasible_under_planner"].max() == 0:
                            cats.append("plan_reject")
                        else:
                            cats.append("false_feas")
                    lines.append(f"  OR-across-sequences: {pd.Series(cats).value_counts().to_dict()}")
            lines.append("")
            # User-expected protocol: often counts are over 20 instances with a single planned route each
            # Check n_sequences and method
            lines.append(f"n40 unique methods: {sorted(sub['method'].unique())}")
            lines.append(f"n40 n_sequences values: {sorted(sub['n_sequences'].unique())}")
            lines.append(f"n40 rows per model: {sub.groupby('charging_model').size().to_dict()}")
            lines.append("")
    return lines


def section_a_exact() -> list[str]:
    """Match the published protocol: for fixed-route, evaluate the TRUE-NL feasible sequences.

    Looking at the study design: often for each instance they take customer sequences from
    TRUE-NL solutions and re-optimize charging under each model. Count should be 20 instances.
    """
    lines = ["## A2. Fixed-route n40 — protocol-aware recount", ""]
    fr = pd.read_csv(RAW / "nonlinear_charging" / "fixed_route_charging_comparison.csv")
    n40 = fr[fr["size"] == 40].copy()
    # Prefer method plan_* single sequence if available
    lines.append(f"Total n40 rows: {len(n40)}")
    lines.append(f"Instances: {n40['instance'].nunique()}")
    lines.append("")
    # If multiple sequences, the summary study typically reports whether the instance's
    # fixed customer order(s) are feasible. User expects:
    # TRUE-NL 20/20, SECANT 11/20 (8 reject + 1 false), EARLY 11/20 (0 reject + 9 false)
    # That implies 20 rows per model. Try filtering to one sequence index if present.
    for col in n40.columns:
        if "seq" in col.lower() or "route" in col.lower():
            pass
    # Check if dropping duplicates by instance×model gives 20
    d = n40.drop_duplicates(["instance", "charging_model"])
    lines.append(f"After drop_duplicates(instance,charging_model): {len(d)} rows")
    for model in ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]:
        m = d[d["charging_model"] == model]
        plan_reject = int((m["feasible_under_planner"] == 0).sum())
        false_feas = int(((m["feasible_under_planner"] == 1) & (m["feasible_under_true_physics"] == 0)).sum())
        true_feas = int((m["feasible_under_true_physics"] == 1).sum())
        lines.append(
            f"- {model}: true_feas={true_feas}/{len(m)}, plan_reject={plan_reject}, false_feas={false_feas}"
        )
    lines.append("")
    # Alternative: instance is counted true-feasible if ANY of its sequences is true-feasible
    lines.append("Alternative: ANY-sequence true-feasible per instance:")
    for model in ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]:
        m = n40[n40["charging_model"] == model]
        by = m.groupby("instance")
        true_feas = int((by["feasible_under_true_physics"].max() == 1).sum())
        # plan_reject: ALL sequences planner-rejected
        plan_reject = int((by["feasible_under_planner"].max() == 0).sum())
        # false_feas: planner accepts at least one, but none true-feasible
        false_feas = 0
        for _, g in by:
            if g["feasible_under_true_physics"].max() == 1:
                continue
            if g["feasible_under_planner"].max() == 1:
                false_feas += 1
        n = by.ngroups
        lines.append(
            f"- {model}: true_feas={true_feas}/{n}, plan_reject={plan_reject}, false_feas={false_feas}"
        )
    lines.append("")
    # Alternative: ALL-sequences must be true-feasible
    lines.append("Alternative: ALL-sequences true-feasible (conservative):")
    for model in ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]:
        m = n40[n40["charging_model"] == model]
        by = m.groupby("instance")
        true_feas = int((by["feasible_under_true_physics"].min() == 1).sum())
        plan_reject = int((by["feasible_under_planner"].min() == 0).sum())  # any reject?
        # Better categories per instance using worst outcome among sequences
        cats = {"true_feas": 0, "plan_reject": 0, "false_feas": 0}
        for _, g in by:
            if g["feasible_under_true_physics"].min() == 1:
                cats["true_feas"] += 1
            elif (g["feasible_under_planner"] == 0).any() and not (g["feasible_under_true_physics"] == 1).any():
                # if any plan reject and no true feas — need finer
                if (g["feasible_under_planner"] == 1).any() and (g["feasible_under_true_physics"] == 0).all():
                    cats["false_feas"] += 1
                elif (g["feasible_under_planner"] == 0).all():
                    cats["plan_reject"] += 1
                else:
                    # mixed: some reject, some accept but fail physics
                    if (g["feasible_under_true_physics"] == 1).any():
                        cats["true_feas"] += 1
                    elif (g["feasible_under_planner"] == 1).any():
                        cats["false_feas"] += 1
                    else:
                        cats["plan_reject"] += 1
            elif (g["feasible_under_planner"] == 1).any() and (g["feasible_under_true_physics"] == 0).all():
                cats["false_feas"] += 1
            else:
                cats["plan_reject"] += 1
        lines.append(f"- {model}: {cats} / {by.ngroups}")
    lines.append("")
    # Look at summary CSV
    summ = RAW / "nonlinear_charging" / "charging_study_summary.csv"
    if summ.exists():
        s = pd.read_csv(summ)
        lines.append("### charging_study_summary.csv")
        lines.append("")
        lines.append("```")
        lines.append(s.to_string(index=False))
        lines.append("```")
        lines.append("")
    return lines


def section_b() -> list[str]:
    lines = ["## B. Nonlinear charging — end-to-end", ""]
    e2e = pd.read_csv(RAW / "nonlinear_charging" / "end_to_end_charging_sensitivity.csv")
    for size in [10, 20, 40]:
        sub = e2e[e2e["size"] == size]
        lines.append(f"### size={size}")
        lines.append("| model | n | true_feas | plan_reject | false_feas | rate |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model in ["TRUE-NL", "LINEAR-SECANT", "LINEAR-EARLY-RATE"]:
            m = sub[sub["charging_model"] == model].drop_duplicates("instance", keep="first")
            n = len(m)
            true_feas = int((m["feasible_under_true_physics"] == 1).sum())
            plan_reject = int((m["feasible_under_planner"] == 0).sum())
            false_feas = int(((m["feasible_under_planner"] == 1) & (m["feasible_under_true_physics"] == 0)).sum())
            lines.append(f"| {model} | {n} | {true_feas} | {plan_reject} | {false_feas} | {true_feas}/{n} |")
        lines.append("")
    dc = pd.read_csv(RAW / "nonlinear_charging" / "end_to_end_decision_changes.csv")
    lines.append("### Decision changes vs TRUE-NL")
    lines.append("")
    for size in sorted(dc["size"].unique()):
        sub = dc[dc["size"] == size]
        for model in sorted(sub["charging_model"].unique()):
            m = sub[sub["charging_model"] == model]
            lines.append(
                f"- n={size} {model}: route_changed={int(m['route_changed'].sum())}/{len(m)} "
                f"({100*m['route_changed'].mean():.1f}%)"
            )
    lines.append("")
    return lines


def section_c() -> list[str]:
    lines = ["## C. n10 matched Cycle-3 fair-LS", ""]
    # Prefer per-instance CSVs
    candidates = [
        RAW / "cycle3" / "compare_size10_fairls_seed42_c3.csv",
        RAW / "cycle3" / "compare_size10_fairls_final_seed43_c3.csv",
        RAW / "cycle3" / "FINAL_paired_by_instance.csv",
        RAW / "cycle3" / "FINAL_n10_fairls.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            lines.append(f"### Source: `{p.relative_to(ROOT)}`")
            lines.append(f"shape={df.shape} cols={list(df.columns)}")
            lines.append("")
            # try to compute means
            method_col = "method" if "method" in df.columns else None
            obj_col = None
            for c in ["objective_h", "full_h", "mean_h", "duration_h", "obj_h"]:
                if c in df.columns:
                    obj_col = c
                    break
            if method_col and obj_col:
                if "seed" in df.columns:
                    # pool seeds
                    g = df.groupby(method_col)[obj_col].agg(["mean", "median", "count"])
                else:
                    g = df.groupby(method_col)[obj_col].agg(["mean", "median", "count"])
                lines.append(g.to_string())
            elif "mean_h" in df.columns:
                lines.append(df[["method", "mean_h", "median_h", "feasible"]].to_string(index=False))
            lines.append("")
    # Aggregate matched 3-seed fair-LS if present
    files = sorted((RAW / "cycle3").glob("compare_size10_fairls*c3.csv"))
    if files:
        parts = [pd.read_csv(f) for f in files]
        all_df = pd.concat(parts, ignore_index=True)
        lines.append(f"### Pooled fairls c3 files ({len(files)} files, {len(all_df)} rows)")
        if "method" in all_df.columns and "objective_h" in all_df.columns:
            g = all_df.groupby("method")["objective_h"].agg(["mean", "median", "count"])
            lines.append(g.to_string())
            lines.append("")
            lines.append(f"seeds: {sorted(all_df['seed'].unique()) if 'seed' in all_df.columns else 'n/a'}")
            lines.append(f"instances/file approx: {all_df['instance'].nunique() if 'instance' in all_df.columns else 'n/a'}")
        lines.append("")
    return lines


def section_d() -> list[str]:
    lines = ["## D. n40 recovered hybrid + BKS gap", ""]
    pipe = pd.read_csv(RAW / "regression_recovery" / "RECOVERY_n40_recovered_pipeline.csv")
    bks = pd.read_csv(ROOT / "results" / "literature_bks_froger2021.csv")
    bks_map = dict(zip(bks["instance"], bks["bks_h"]))
    # hybrid stage
    stages = sorted(pipe["stage"].unique())
    lines.append(f"stages: {stages}")
    hybrid = pipe[pipe["stage"].str.contains("hybrid", case=False, na=False)]
    if hybrid.empty:
        # take best stage per instance×method or final stage
        for st in ["hybrid", "hybrid_repair", "neural_hybrid", "final"]:
            pass
        # use last stage lexicographically or known name
        lines.append("No stage containing 'hybrid'. Stage means:")
        lines.append(pipe.groupby(["method", "stage"])["objective_h"].mean().to_string())
        lines.append("")
        # Prefer stage with lowest mean for recovered pipeline naming
        for preferred in [
            "hybrid_repair",
            "hybrid",
            "neural_repair",
            "strong_ls",
            "ls",
            "construct",
        ]:
            if preferred in set(pipe["stage"]):
                hybrid = pipe[pipe["stage"] == preferred]
                lines.append(f"Using stage={preferred}")
                break
        if hybrid.empty:
            # take min objective per instance×method across stages as 'best recovered'
            hybrid = (
                pipe.loc[pipe.groupby(["instance", "method"])["objective_h"].idxmin()]
                .copy()
            )
            lines.append("Using best-across-stages per instance×method")
    lines.append("")
    for method in ["H2", "H4"]:
        m = hybrid[hybrid["method"] == method].drop_duplicates("instance")
        mean_full = m["objective_h"].mean()
        gaps = []
        for _, r in m.iterrows():
            gaps.append(_gap_lit(r["objective_h"], bks_map[r["instance"]], 40))
        lines.append(
            f"- {method} n={len(m)} mean_full={mean_full:.4f}h "
            f"mean_lit_gap={np.mean(gaps):.4f}% median_lit_gap={np.median(gaps):.4f}% "
            f"feas={int(m['feasible'].sum())}/{len(m)}"
        )
    # Paired H4-H2
    h2 = hybrid[hybrid["method"] == "H2"].set_index("instance")["objective_h"]
    h4 = hybrid[hybrid["method"] == "H4"].set_index("instance")["objective_h"]
    common = h2.index.intersection(h4.index)
    diff = (h4.loc[common] - h2.loc[common]).astype(float)
    wins = int((diff < -1e-6).sum())
    losses = int((diff > 1e-6).sum())
    ties = int((diff.abs() <= 1e-6).sum())
    # paired SE over layouts (not treating as 60 independent)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    mean_d = diff.mean()
    ci = (mean_d - 1.96 * se, mean_d + 1.96 * se)
    lines.append("")
    lines.append(
        f"H4−H2 paired (n={len(diff)} layouts): mean={mean_d:.4f}h, "
        f"wins/ties/losses={wins}/{ties}/{losses}, "
        f"approx 95% layout-SE CI=[{ci[0]:.4f}, {ci[1]:.4f}]"
    )
    lines.append(f"CI includes zero: {ci[0] <= 0 <= ci[1]}")
    lines.append("")
    # Also report decomp if present
    decomp = RAW / "regression_recovery" / "RECOVERY_n40_objective_decomposition.csv"
    if decomp.exists():
        d = pd.read_csv(decomp)
        lines.append("### Objective decomposition source")
        lines.append(d.to_string(index=False))
        lines.append("")
    return lines


def section_e() -> list[str]:
    lines = ["## E. Controlled BC vs RL (official n10 construction-only)", ""]
    s = pd.read_csv(RAW / "reward_learning" / "bc_vs_rl_official_n10_summary.csv")
    lines.append(s.to_string(index=False))
    lines.append("")
    for model in ["H2", "H4"]:
        bc = s[(s["model"] == model) & (s["tag"] == "bc_only_rq4")].set_index("seed")["mean_full_h"]
        rl = s[(s["model"] == model) & (s["tag"] == "rl_from_bc_rq4")].set_index("seed")["mean_full_h"]
        common = bc.index.intersection(rl.index)
        improve = (rl.loc[common] < bc.loc[common] - 1e-9).sum()
        lines.append(
            f"- {model}: RL improves {int(improve)}/{len(common)} seeds; "
            f"paired deltas={(rl.loc[common]-bc.loc[common]).to_dict()}"
        )
    lines.append("")
    lines.append(
        "Protocol: official Montoya n10, construction-only inference, 20 instances/seed, "
        "3 training seeds. NOT hybrid/final routing."
    )
    lines.append("")
    return lines


def section_f() -> list[str]:
    lines = ["## F. Historical Cycle-3 RL synthetic validation", ""]
    # Look for cycle3 RL comparisons in training logs or reward study docs
    logs = RAW / "training_logs"
    # Parse best val from c3 training logs for H2/H4 with rl tags if any
    rl_logs = list(logs.glob("*rl*")) + list(logs.glob("*c3*"))
    # Check REWARD study summary if present
    for name in [
        "REWARD_AND_LEARNING_STUDY.md",
        "reward_learning_summary.csv",
        "cycle3_rl_summary.csv",
    ]:
        for p in [RAW / "reward_learning" / name, ROOT / "docs" / name, ROOT / "results" / name]:
            if p.exists() and p.suffix == ".csv":
                df = pd.read_csv(p)
                lines.append(f"### {p.relative_to(ROOT)}")
                lines.append(df.to_string(index=False))
                lines.append("")
            elif p.exists() and p.suffix == ".md":
                lines.append(f"Doc present: `{p.relative_to(ROOT)}`")
    # Extract from training logs: H4 c3 synthetic val at end vs BC
    # Many cycle3 models store validation in training_*_c3.csv
    rows = []
    for model in ["H2", "H4"]:
        for seed in [42, 43, 44]:
            for tag in [f"training_{model}_n10_seed{seed}_c3.csv", f"training_{model}_n10_seed{seed}.csv"]:
                p = logs / tag
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                # find validation objective columns
                val_cols = [c for c in df.columns if "val" in c.lower() or "obj" in c.lower()]
                rows.append((model, seed, tag, list(df.columns)[:12], len(df)))
    lines.append("Training log inventory (first cols):")
    for r in rows[:12]:
        lines.append(f"- {r}")
    lines.append("")
    # Try to find RL-specific cycle3 comparison CSVs
    for p in sorted((RAW / "reward_learning").glob("*.csv")):
        if "cycle" in p.name.lower() or "synth" in p.name.lower() or "histor" in p.name.lower():
            lines.append(f"Found {p.name}")
            lines.append(pd.read_csv(p).head().to_string())
    lines.append("")
    return lines


def section_competitiveness() -> list[str]:
    lines = ["## Competitiveness — multi-protocol BKS gaps", ""]
    bks = pd.read_csv(ROOT / "results" / "literature_bks_froger2021.csv")
    bks_map = dict(zip(bks["instance"], bks["bks_h"]))

    def summarize(df, method_col, obj_col, size, label, protocol):
        out = []
        for method, g in df.groupby(method_col):
            gaps = []
            for _, r in g.iterrows():
                inst = r["instance"]
                if inst not in bks_map:
                    continue
                gaps.append(_gap_lit(float(r[obj_col]), bks_map[inst], size))
            if not gaps:
                continue
            out.append(
                {
                    "protocol": protocol,
                    "label": label,
                    "method": method,
                    "size": size,
                    "n": len(gaps),
                    "mean_full_h": float(g[obj_col].mean()),
                    "mean_lit_gap_pct": float(np.mean(gaps)),
                    "median_lit_gap_pct": float(np.median(gaps)),
                    "feas": f"{int(g['feasible'].sum()) if 'feasible' in g.columns else 'n/a'}/{len(g)}",
                    "mean_runtime_s": float(g["runtime_s"].mean()) if "runtime_s" in g.columns else np.nan,
                    "mean_oracle": float(g["oracle_solver_executions"].mean())
                    if "oracle_solver_executions" in g.columns
                    else np.nan,
                }
            )
        return out

    rows = []
    # Cycle-3 fair-LS n10 per-instance
    for f in sorted((RAW / "cycle3").glob("compare_size10_fairls*c3.csv")):
        df = pd.read_csv(f)
        if "objective_h" not in df.columns:
            continue
        rows.extend(summarize(df, "method", "objective_h", 10, f.name, "Cycle-3 fair-LS"))
    # Insertion hybrid n10
    ins = RAW / "insertion_repair" / "INSERTION_official_n10.csv"
    if ins.exists():
        df = pd.read_csv(ins)
        # filter hybrid-like modes
        rows.extend(summarize(df, "method", "objective_h", 10, "INSERTION_official_n10", "insertion/hybrid modes"))
        if "mode" in df.columns:
            for mode, g in df.groupby("mode"):
                rows.extend(
                    summarize(g, "method", "objective_h", 10, f"INSERTION mode={mode}", f"insertion mode={mode}")
                )
    # n40 recovery hybrid
    pipe = pd.read_csv(RAW / "regression_recovery" / "RECOVERY_n40_recovered_pipeline.csv")
    for preferred in ["hybrid_repair", "hybrid", "neural_repair"]:
        if preferred in set(pipe["stage"]):
            h = pipe[pipe["stage"] == preferred]
            rows.extend(summarize(h, "method", "objective_h", 40, f"RECOVERY stage={preferred}", "n40 recovered"))
            break
    else:
        h = pipe.loc[pipe.groupby(["instance", "method"])["objective_h"].idxmin()]
        rows.extend(summarize(h, "method", "objective_h", 40, "RECOVERY best-stage", "n40 recovered best"))

    tab = pd.DataFrame(rows)
    lines.append("### Protocol-separated gap table")
    lines.append("")
    if not tab.empty:
        # collapse duplicate protocols by averaging isn't right — show unique key rows
        show = tab.drop_duplicates(["protocol", "label", "method", "size"])
        lines.append(
            show.to_string(
                index=False,
                columns=[
                    "protocol",
                    "size",
                    "method",
                    "n",
                    "mean_full_h",
                    "mean_lit_gap_pct",
                    "median_lit_gap_pct",
                    "feas",
                    "mean_runtime_s",
                    "mean_oracle",
                    "label",
                ],
            )
        )
        tab.to_csv(ROOT / "results" / "tables" / "benchmark_protocol_comparison.csv", index=False)
        lines.append("")
        lines.append("Wrote `results/tables/benchmark_protocol_comparison.csv`")
    lines.append("")
    lines.append(
        "**Critical:** Cycle-3 fair-LS n10 H2 ~5% lit gap is NOT the same experiment as "
        "INSERTION hybrid ~0.03% gap. The latter uses a more expensive insertion/hybrid repair protocol."
    )
    lines.append("")
    return lines


def main() -> None:
    parts = [
        "# Scientific Results Audit",
        "",
        "Independently recomputed from authoritative per-instance CSVs.",
        f"Repository root: `{ROOT}`",
        "",
        "Objective conventions:",
        "- `Z_full`: full route duration including fixed customer service time (hours).",
        "- `Z_lit = Z_full - n_customers × 0.5 h` (Montoya/Froger literature-compatible).",
        "- BKS gap: `100 × (Z_lit − BKS) / BKS`.",
        "",
    ]
    parts += section_a()
    parts += section_a_exact()
    parts += section_b()
    parts += section_c()
    parts += section_d()
    parts += section_e()
    parts += section_f()
    parts += section_competitiveness()
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
