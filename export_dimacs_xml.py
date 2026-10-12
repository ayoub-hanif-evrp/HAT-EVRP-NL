"""Export a concrete hybrid solution as DIMACS E-VRP-NL XML."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from data import instance_files, parse_instance
from train import decode, load_policy


def export_solution_xml(
    instance_name: str,
    routes_stops: list[list[tuple[int, float | None]]],
    solver: str,
    output: Path,
) -> None:
    root = ET.Element("solution", instance=instance_name, solver=solver, optimal="false")
    for route_id, stops in enumerate(routes_stops):
        route = ET.SubElement(root, "route", id=str(route_id))
        for node, amount in stops:
            if node == 0:
                ET.SubElement(route, "depot", id="0")
            elif amount is not None:
                station = ET.SubElement(route, "station", id=str(node))
                ET.SubElement(station, "wait").text = "0"
                ET.SubElement(station, "charge").text = f"{float(amount):.10f}"
            else:
                ET.SubElement(route, "customer", id=str(node))
    runtime = ET.SubElement(root, "runtime")
    machine = ET.SubElement(runtime, "machine")
    ET.SubElement(machine, "language").text = "Python"
    ET.SubElement(runtime, "cputime").text = "0"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="tc0c10s2cf1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--fair-ls-passes", type=int, default=3)
    parser.add_argument("--fair-ls-budget", type=int, default=4000)
    args = parser.parse_args()
    path = next(p for p in instance_files(args.data) if p.stem == args.instance)
    instance = parse_instance(path)
    model = load_policy(args.checkpoint)
    result = decode(
        model,
        instance,
        sample=False,
        local_passes=args.fair_ls_passes,
        local_budget=args.fair_ls_budget,
    )
    if not result["feasible"]:
        raise SystemExit(result["reason"])
    export_solution_xml(
        instance.name,
        [list(plan.stops) for plan in result["plans"]],
        "HAT-EVRP-NL hybrid H4",
        args.output,
    )
    print(f"Wrote {args.output}; objective={result['objective']:.6f} h")


if __name__ == "__main__":
    main()
