"""Minimal parser for the Montoya et al. E-VRP-NL XML instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Node:
    id: int
    kind: int  # 0 depot, 1 customer, 2 charging station
    x: float
    y: float
    station_type: str | None = None


@dataclass(frozen=True)
class ChargingCurve:
    station_type: str
    times: tuple[float, ...]
    energies: tuple[float, ...]


@dataclass
class Instance:
    name: str
    path: Path
    nodes: tuple[Node, ...]
    depot: int
    customers: tuple[int, ...]
    stations: tuple[int, ...]
    service_times: tuple[float, ...]
    speed_kmph: float
    consumption_wh_per_km: float
    battery_wh: float
    max_route_hours: float
    decimals: int
    curves: dict[str, ChargingCurve]
    distances_km: tuple[tuple[float, ...], ...]
    travel_hours: tuple[tuple[float, ...], ...]
    energy_wh: tuple[tuple[float, ...], ...]


def _required_text(parent: ET.Element, path: str) -> str:
    element = parent.find(path)
    if element is None or element.text is None:
        raise ValueError(f"Missing XML field {path}")
    return element.text


def parse_instance(path: str | Path) -> Instance:
    """Parse one benchmark file without modifying or rounding its data."""
    path = Path(path)
    root = ET.parse(path).getroot()
    name = _required_text(root, "./info/name")

    parsed_nodes: list[Node] = []
    for element in root.findall("./network/nodes/node"):
        kind = int(element.attrib["type"])
        station_type = (
            _required_text(element, "./custom/cs_type") if kind == 2 else None
        )
        parsed_nodes.append(
            Node(
                id=int(element.attrib["id"]),
                kind=kind,
                x=float(_required_text(element, "cx")),
                y=float(_required_text(element, "cy")),
                station_type=station_type,
            )
        )
    parsed_nodes.sort(key=lambda node: node.id)
    if [node.id for node in parsed_nodes] != list(range(len(parsed_nodes))):
        raise ValueError(f"{name}: node IDs must be contiguous from zero")
    nodes = tuple(parsed_nodes)

    depots = tuple(node.id for node in nodes if node.kind == 0)
    if len(depots) != 1:
        raise ValueError(f"{name}: expected one depot, found {len(depots)}")
    customers = tuple(node.id for node in nodes if node.kind == 1)
    stations = tuple(node.id for node in nodes if node.kind == 2)

    vehicle = root.find("./fleet/vehicle_profile")
    if vehicle is None:
        raise ValueError(f"{name}: missing vehicle profile")
    speed = float(_required_text(vehicle, "speed_factor"))
    consumption = float(_required_text(vehicle, "./custom/consumption_rate"))
    battery = float(_required_text(vehicle, "./custom/battery_capacity"))
    max_route = float(_required_text(vehicle, "max_travel_time"))

    curves: dict[str, ChargingCurve] = {}
    for function in vehicle.findall("./custom/charging_functions/function"):
        station_type = function.attrib["cs_type"]
        points = [
            (
                float(_required_text(point, "charging_time")),
                float(_required_text(point, "battery_level")),
            )
            for point in function.findall("breakpoint")
        ]
        if len(points) < 2:
            raise ValueError(f"{name}: charging curve {station_type} has <2 points")
        times, energies = zip(*points)
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"{name}: non-increasing times in {station_type} curve")
        if any(b <= a for a, b in zip(energies, energies[1:])):
            raise ValueError(f"{name}: non-increasing energies in {station_type} curve")
        if abs(energies[0]) > 1e-12 or abs(energies[-1] - battery) > 1e-9:
            raise ValueError(f"{name}: {station_type} curve does not span [0, Q]")
        curves[station_type] = ChargingCurve(station_type, times, energies)

    for station in stations:
        station_type = nodes[station].station_type
        if station_type not in curves:
            raise ValueError(f"{name}: station {station} has no matching curve")

    service_times = [0.0] * len(nodes)
    for request in root.findall("./requests/request"):
        node_id = int(request.attrib["node"])
        service_times[node_id] = float(_required_text(request, "service_time"))
    if any(service_times[node] < 0 for node in customers):
        raise ValueError(f"{name}: negative service time")

    distances: list[tuple[float, ...]] = []
    travel: list[tuple[float, ...]] = []
    energy: list[tuple[float, ...]] = []
    for source in nodes:
        distance_row = tuple(
            math.hypot(source.x - target.x, source.y - target.y)
            for target in nodes
        )
        distances.append(distance_row)
        travel.append(tuple(distance / speed for distance in distance_row))
        energy.append(tuple(distance * consumption for distance in distance_row))

    return Instance(
        name=name,
        path=path,
        nodes=nodes,
        depot=depots[0],
        customers=customers,
        stations=stations,
        service_times=tuple(service_times),
        speed_kmph=speed,
        consumption_wh_per_km=consumption,
        battery_wh=battery,
        max_route_hours=max_route,
        decimals=int(_required_text(root, "./network/decimals")),
        curves=curves,
        distances_km=tuple(distances),
        travel_hours=tuple(travel),
        energy_wh=tuple(energy),
    )


def instance_files(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).rglob("*.xml"))


def generate_synthetic(size: int, seed: int, distribution: str | None = None) -> Instance:
    """Generate an independent Montoya-like instance for policy training."""
    rng = random.Random(seed)
    distribution = distribution or ("random", "clustered", "mixed")[seed % 3]
    depot_xy = (rng.uniform(10, 110), rng.uniform(10, 110))

    if distribution == "random":
        customer_xy = [(rng.uniform(0, 120), rng.uniform(0, 120)) for _ in range(size)]
    else:
        n_clusters = max(2, round(math.sqrt(size) / 1.5))
        centers = [(rng.uniform(8, 112), rng.uniform(8, 112)) for _ in range(n_clusters)]
        clustered = [
            (
                min(120, max(0, centers[k % n_clusters][0] + rng.gauss(0, 5))),
                min(120, max(0, centers[k % n_clusters][1] + rng.gauss(0, 5))),
            )
            for k in range(size)
        ]
        if distribution == "mixed":
            for k in range(size // 2):
                clustered[k] = (rng.uniform(0, 120), rng.uniform(0, 120))
        customer_xy = clustered

    station_count = max(2, round(size * 0.15))
    station_xy = [(rng.uniform(5, 115), rng.uniform(5, 115)) for _ in range(station_count)]
    technologies = ("fast", "normal", "slow")
    curve_times = {
        "fast": (0.0, 0.31, 0.39, 0.51),
        "normal": (0.0, 0.62, 0.77, 1.01),
        "slow": (0.0, 1.26, 1.54, 2.04),
    }
    curves = {
        technology: ChargingCurve(
            technology, curve_times[technology], (0.0, 13600.0, 15200.0, 16000.0)
        )
        for technology in technologies[: min(3, station_count)]
    }

    nodes = [Node(0, 0, *depot_xy)]
    nodes.extend(Node(k + 1, 1, *xy) for k, xy in enumerate(customer_xy))
    for k, xy in enumerate(station_xy):
        technology = tuple(curves)[k % len(curves)]
        nodes.append(Node(size + 1 + k, 2, *xy, station_type=technology))
    node_tuple = tuple(nodes)
    service = tuple(0.5 if node.kind == 1 else 0.0 for node in node_tuple)
    distances = tuple(
        tuple(math.hypot(a.x - b.x, a.y - b.y) for b in node_tuple)
        for a in node_tuple
    )
    travel = tuple(tuple(distance / 40.0 for distance in row) for row in distances)
    energy = tuple(tuple(distance * 125.0 for distance in row) for row in distances)
    return Instance(
        name=f"syn_{distribution}_n{size}_s{seed}",
        path=Path("<synthetic>"),
        nodes=node_tuple,
        depot=0,
        customers=tuple(range(1, size + 1)),
        stations=tuple(range(size + 1, len(node_tuple))),
        service_times=service,
        speed_kmph=40.0,
        consumption_wh_per_km=125.0,
        battery_wh=16000.0,
        max_route_hours=10.0,
        decimals=14,
        curves=curves,
        distances_km=distances,
        travel_hours=travel,
        energy_wh=energy,
    )
