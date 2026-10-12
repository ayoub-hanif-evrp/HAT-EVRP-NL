"""Minimal parser for the Montoya et al. E-VRP-NL XML instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
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
