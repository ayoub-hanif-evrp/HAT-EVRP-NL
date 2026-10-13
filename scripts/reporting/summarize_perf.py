"""Summarize PERF_* CSVs for the research report."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
RAW = RES / "raw"


def _find(name: str) -> Path:
    for base in (
        RES,
        RAW / "generator_research",
        RAW / "misc",
        RAW / "cycle3",
        RAW / "neural_repair",
    ):
        cand = base / name
        if cand.exists():
            return cand
    matches = list(RAW.rglob(name))
    return matches[0] if matches else RES / name


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def summarize_synth():
    path = _find("PERF_synth_validation.csv")
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by = defaultdict(lambda: {"obj": [], "n": 0, "f": 0})
    for r in rows:
        m = r["method"]
        by[m]["n"] += 1
        if r["feasible"] == "1":
            by[m]["f"] += 1
            by[m]["obj"].append(float(r["objective_h"]))
    return {
        m: {
            "feas_rate": d["f"] / max(1, d["n"]),
            "mean_obj": mean(d["obj"]),
            "n": d["n"],
            "n_feas": d["f"],
        }
        for m, d in sorted(by.items())
    }


def summarize_search():
    path = _find("PERF_search_ablation.csv")
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by = defaultdict(list)
    pair = {}
    for r in rows:
        if r["feasible"] == "1":
            by[r["search"]].append(float(r["objective_h"]))
            pair.setdefault(r["seed"], {})[r["search"]] = float(r["objective_h"])
    w = t = l = 0
    for p in pair.values():
        if "strong_plus_ruin" in p and "strong_only" in p:
            a, b = p["strong_only"], p["strong_plus_ruin"]
            if b < a - 1e-9:
                w += 1
            elif a < b - 1e-9:
                l += 1
            else:
                t += 1
    return {
        "means": {k: mean(v) for k, v in by.items()},
        "ruin_WTL": [w, t, l],
        "n_feasible_pairs": w + t + l,
    }


def summarize_official(size=10):
    path = _find(f"PERF_official_n{size}.csv")
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by = defaultdict(lambda: {"obj": [], "gap": [], "rt": [], "n": 0, "f": 0})
    layout = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m = r["method"]
        by[m]["n"] += 1
        if r["feasible"] == "1":
            by[m]["f"] += 1
            by[m]["obj"].append(float(r["objective_h"]))
            by[m]["rt"].append(float(r["runtime_s"]))
            if r["bks_gap_lit_pct"] != "":
                by[m]["gap"].append(float(r["bks_gap_lit_pct"]))
                layout[m][r["layout"]].append(float(r["bks_gap_lit_pct"]))
    out = {
        m: {
            "feas_pct": 100 * d["f"] / max(1, d["n"]),
            "mean_full_h": mean(d["obj"]),
            "mean_lit_gap_pct": mean(d["gap"]),
            "median_lit_gap_pct": statistics.median(d["gap"]) if d["gap"] else float("nan"),
            "mean_runtime_s": mean(d["rt"]),
            "n": d["n"],
        }
        for m, d in sorted(by.items())
    }
    # layout means for seed-pooled methods
    layout_out = {}
    for m, lays in layout.items():
        layout_out[m] = {lay: mean(gs) for lay, gs in sorted(lays.items())}
    return {"methods": out, "layout_gaps": layout_out}


def main():
    summary = {
        "synth": summarize_synth(),
        "search": summarize_search(),
        "official_n10": summarize_official(10),
        "official_n20": summarize_official(20),
    }
    path = RES / "PERF_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
