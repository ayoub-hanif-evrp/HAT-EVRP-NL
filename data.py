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


def _pmedian_stations(
    points: list[tuple[float, float]], count: int, rng: random.Random
) -> list[tuple[float, float]]:
    """Greedy p-median-like station placement among candidate sites."""
    if count <= 0:
        return []
    candidates = points[:]
    rng.shuffle(candidates)
    selected = [candidates[0]]
    while len(selected) < min(count, len(candidates)):
        best = None
        best_cost = float("inf")
        for candidate in candidates:
            if candidate in selected:
                continue
            trial = selected + [candidate]
            cost = 0.0
            for px, py in points:
                cost += min(math.hypot(px - sx, py - sy) for sx, sy in trial)
            if cost < best_cost:
                best_cost = cost
                best = candidate
        if best is None:
            break
        selected.append(best)
    return selected


@dataclass(frozen=True)
class SyntheticConfig:
    """Independently controllable synthetic instance factors (v2 generator)."""

    geometry: str = "random"  # random | clustered | mixed
    station_placement: str = "uniform"  # uniform | pmedian_region
    density: float = 0.15
    tech_pool: tuple[str, ...] = ("fast", "normal")
    cluster_sigma: float = 5.0
    n_clusters: int | None = None
    station_count: int | None = None  # if set, overrides density rounding
    depot_bias: str = "center"  # center | offset (farther mean depot–customer)


def _station_count_from_density(size: int, density: float) -> int:
    """Map density to station count without banker's-rounding collapse.

    For n=10, densities in {0.10..0.25} all previously yielded 2 stations because
    ``round(2.5)==2``. Use ceil so 0.25 → 3, matching Montoya's 2/3 split.
    """
    return max(2, math.ceil(size * density - 1e-12))


def sample_synthetic_config(rng: random.Random) -> SyntheticConfig:
    """Independently sample geometry, placement, density, and technologies.

    Priors are rebalanced toward Montoya-like mass (fast+slow / normal+slow pairs,
    ~50% three-station instances at n=10) while remaining factor-independent and
    never copying official coordinates.
    """
    geometry = rng.choice(("random", "clustered", "mixed"))
    station_placement = rng.choice(("uniform", "pmedian_region"))
    # Explicit station counts: equal weight on 2 vs 3 for typical n≈10 training.
    station_count = rng.choice((2, 2, 3, 3, 4))
    # Density kept for naming / scaling; count is authoritative when set.
    density = station_count / 10.0
    # Weighted tech pools: emphasize Montoya pairs; keep rare singles for coverage.
    tech_weights = (
        (("fast", "slow"), 4),
        (("normal", "slow"), 2),
        (("fast", "normal"), 1),
        (("fast",), 1),
        (("fast", "normal", "slow"), 1),
        (("normal",), 0.5),
        (("slow",), 0.5),
    )
    techs, weights = zip(*tech_weights)
    tech_pool = rng.choices(techs, weights=weights, k=1)[0]
    cluster_sigma = rng.uniform(3.5, 8.0)
    depot_bias = rng.choice(("center", "offset", "offset"))
    return SyntheticConfig(
        geometry=geometry,
        station_placement=station_placement,
        density=density,
        tech_pool=tech_pool,
        cluster_sigma=cluster_sigma,
        station_count=station_count,
        depot_bias=depot_bias,
    )


def _region_pmedian_stations(
    count: int, rng: random.Random, demand: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """P-median on independently sampled region candidates (not customer copies)."""
    candidate_count = max(count * 8, 24)
    candidates = [
        (rng.uniform(2, 118), rng.uniform(2, 118)) for _ in range(candidate_count)
    ]
    # Reject near-duplicates.
    filtered: list[tuple[float, float]] = []
    for xy in candidates:
        if all(math.hypot(xy[0] - fx, xy[1] - fy) > 1.5 for fx, fy in filtered):
            filtered.append(xy)
    if len(filtered) < count:
        filtered.extend(
            (rng.uniform(2, 118), rng.uniform(2, 118))
            for _ in range(count - len(filtered))
        )
    return _pmedian_from_candidates(filtered, demand, count, rng)


def _pmedian_from_candidates(
    candidates: list[tuple[float, float]],
    demand: list[tuple[float, float]],
    count: int,
    rng: random.Random,
) -> list[tuple[float, float]]:
    selected: list[tuple[float, float]] = []
    remaining = list(candidates)
    rng.shuffle(remaining)
    for _ in range(min(count, len(remaining))):
        best = None
        best_cost = float("inf")
        for candidate in remaining:
            trial = selected + [candidate]
            cost = 0.0
            for px, py in demand:
                cost += min(math.hypot(px - sx, py - sy) for sx, sy in trial)
            if cost < best_cost:
                best_cost = cost
                best = candidate
        if best is None:
            break
        selected.append(best)
        remaining = [xy for xy in remaining if xy != best]
    while len(selected) < count:
        selected.append((rng.uniform(5, 115), rng.uniform(5, 115)))
    return selected


def generate_synthetic(
    size: int,
    seed: int,
    distribution: str | None = None,
    config: SyntheticConfig | None = None,
    generator_version: str = "v2",
) -> Instance:
    """Generate an independent Montoya-like instance for policy training.

    ``generator_version='v2'`` (default): geometry, station placement, density,
    and charging technologies are sampled independently via the RNG.
    ``generator_version='v1'`` restores the legacy seed-% coupled generator for
    ablation comparisons.
    """
    rng = random.Random(seed)
    if generator_version == "v1":
        return _generate_synthetic_v1(size, seed, distribution, rng)

    if config is None:
        config = sample_synthetic_config(rng)
        if distribution is not None:
            config = SyntheticConfig(
                geometry=distribution,
                station_placement=config.station_placement,
                density=config.density,
                tech_pool=config.tech_pool,
                cluster_sigma=config.cluster_sigma,
                n_clusters=config.n_clusters,
                station_count=config.station_count,
                depot_bias=config.depot_bias,
            )

    # Customers first so offset depots can sit away from the customer mass (tc1-like).
    geometry = config.geometry
    if geometry == "random":
        customer_xy = [(rng.uniform(0, 120), rng.uniform(0, 120)) for _ in range(size)]
    else:
        n_clusters = config.n_clusters or max(2, round(math.sqrt(size) / 1.5))
        centers = [(rng.uniform(8, 112), rng.uniform(8, 112)) for _ in range(n_clusters)]
        customer_xy = [
            (
                min(120, max(0, centers[k % n_clusters][0] + rng.gauss(0, config.cluster_sigma))),
                min(120, max(0, centers[k % n_clusters][1] + rng.gauss(0, config.cluster_sigma))),
            )
            for k in range(size)
        ]
        if geometry == "mixed":
            for k in range(size // 2):
                customer_xy[k] = (rng.uniform(0, 120), rng.uniform(0, 120))

    cx = sum(x for x, _ in customer_xy) / len(customer_xy)
    cy = sum(y for _, y in customer_xy) / len(customer_xy)
    if config.depot_bias == "offset":
        # Strong corner bias to raise mean depot–customer distance toward Montoya tc1.
        corner = rng.choice(((2, 2), (2, 118), (118, 2), (118, 118)))
        depot_xy = (
            min(118, max(2, 0.15 * cx + 0.85 * corner[0] + rng.uniform(-5, 5))),
            min(118, max(2, 0.15 * cy + 0.85 * corner[1] + rng.uniform(-5, 5))),
        )
    else:
        depot_xy = (
            min(110, max(10, cx + rng.uniform(-25, 25))),
            min(110, max(10, cy + rng.uniform(-25, 25))),
        )

    if config.station_count is not None:
        station_count = max(2, int(config.station_count))
        # Scale with problem size when config was sampled at n≈10 units.
        if size != 10:
            station_count = max(2, round(station_count * size / 10))
    else:
        station_count = _station_count_from_density(size, config.density)
    demand = customer_xy + [depot_xy]
    if config.station_placement == "pmedian_region":
        station_xy = _region_pmedian_stations(station_count, rng, demand)
    else:
        station_xy = []
        attempts = 0
        while len(station_xy) < station_count and attempts < station_count * 40:
            xy = (rng.uniform(2, 118), rng.uniform(2, 118))
            attempts += 1
            if all(math.hypot(xy[0] - sx, xy[1] - sy) > 1.0 for sx, sy in station_xy):
                station_xy.append(xy)
        while len(station_xy) < station_count:
            station_xy.append((rng.uniform(5, 115), rng.uniform(5, 115)))

    tech_pool = config.tech_pool
    curve_times = {
        "fast": (0.0, 0.31, 0.39, 0.51),
        "normal": (0.0, 0.62, 0.77, 1.01),
        "slow": (0.0, 1.26, 1.54, 2.04),
    }
    curves = {
        technology: ChargingCurve(
            technology, curve_times[technology], (0.0, 13600.0, 15200.0, 16000.0)
        )
        for technology in tech_pool
    }

    # Independent per-station technology assignment (not a fixed cycle only).
    tech_assignment = [rng.choice(tech_pool) for _ in range(station_count)]

    nodes = [Node(0, 0, *depot_xy)]
    nodes.extend(Node(k + 1, 1, *xy) for k, xy in enumerate(customer_xy))
    for k, xy in enumerate(station_xy):
        nodes.append(Node(size + 1 + k, 2, *xy, station_type=tech_assignment[k]))
    node_tuple = tuple(nodes)
    service = tuple(0.5 if node.kind == 1 else 0.0 for node in node_tuple)
    distances = tuple(
        tuple(math.hypot(a.x - b.x, a.y - b.y) for b in node_tuple)
        for a in node_tuple
    )
    travel = tuple(tuple(distance / 40.0 for distance in row) for row in distances)
    energy = tuple(tuple(distance * 125.0 for distance in row) for row in distances)
    tech_tag = "".join(t[0] for t in tech_pool)
    place_tag = config.station_placement.replace("_", "-")
    return Instance(
        name=(
            f"syn_v2_{geometry}_n{size}_{place_tag}"
            f"_d{int(config.density*100)}_{tech_tag}_s{seed}"
        ),
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


def _generate_synthetic_v1(
    size: int, seed: int, distribution: str | None, rng: random.Random
) -> Instance:
    """Legacy coupled generator retained for ablation only."""
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
    density = (0.10, 0.15, 0.20, 0.25)[seed % 4]
    station_count = max(2, round(size * density))
    if seed % 2 == 0:
        station_xy = _pmedian_stations(customer_xy + [depot_xy], station_count, rng)
        place = "pmed"
    else:
        station_xy = [
            (rng.uniform(5, 115), rng.uniform(5, 115)) for _ in range(station_count)
        ]
        place = "rand"
    technologies = ("fast", "normal", "slow")
    tech_pool = technologies[: 1 + (seed % 3)]
    curve_times = {
        "fast": (0.0, 0.31, 0.39, 0.51),
        "normal": (0.0, 0.62, 0.77, 1.01),
        "slow": (0.0, 1.26, 1.54, 2.04),
    }
    curves = {
        technology: ChargingCurve(
            technology, curve_times[technology], (0.0, 13600.0, 15200.0, 16000.0)
        )
        for technology in tech_pool
    }
    nodes = [Node(0, 0, *depot_xy)]
    nodes.extend(Node(k + 1, 1, *xy) for k, xy in enumerate(customer_xy))
    for k, xy in enumerate(station_xy):
        technology = tech_pool[k % len(tech_pool)]
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
        name=f"syn_v1_{distribution}_n{size}_{place}_s{seed}",
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
