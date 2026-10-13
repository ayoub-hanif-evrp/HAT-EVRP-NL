"""Compact H1-H4 policies for hybrid E-VRP-NL route construction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from charging import cumulative_charging_time
from data import Instance


VISIT_FEATURE_DIM = 8


@dataclass(frozen=True)
class PolicyConfig:
    history: bool
    deepsets: bool
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    deepsets_open_only: bool = False


def plan_visit_features(instance: Instance, stops) -> list[tuple[int, list[float]]]:
    """Forward-simulate an optimized plan into per-stop history features.

    Features are rebuilt whenever frvcpy revises the mutable charging plan.
    Customer-decision order remains separate in route['customers'].
    """
    if not stops:
        return []
    features: list[tuple[int, list[float]]] = []
    soc = instance.battery_wh
    elapsed = 0.0
    previous = None
    for index, (node, amount) in enumerate(stops):
        travel_energy = 0.0
        if previous is not None:
            travel_energy = instance.energy_wh[previous][node]
            soc = max(0.0, soc - travel_energy)
            elapsed += instance.travel_hours[previous][node]
        arrival_soc = soc
        arrival_time = elapsed
        charge_amount = 0.0
        charge_duration = 0.0
        if amount is not None and amount > 0:
            curve = instance.curves[instance.nodes[node].station_type]
            q_out = min(instance.battery_wh, soc + float(amount))
            charge_amount = q_out - soc
            charge_duration = (
                cumulative_charging_time(curve, q_out)
                - cumulative_charging_time(curve, soc)
            )
            elapsed += charge_duration
            soc = q_out
        service = instance.service_times[node]
        elapsed += service
        kind = instance.nodes[node].kind / 2.0
        features.append(
            (
                node,
                [
                    arrival_time / instance.max_route_hours,
                    elapsed / instance.max_route_hours,
                    arrival_soc / instance.battery_wh,
                    soc / instance.battery_wh,
                    travel_energy / instance.battery_wh,
                    charge_amount / instance.battery_wh,
                    charge_duration / instance.max_route_hours,
                    kind,
                ],
            )
        )
        previous = node
    return features


class HATPolicy(nn.Module):
    """Shared causal route encoder, dynamic fusion, and hierarchical pointers."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.node_encoder = nn.Sequential(nn.Linear(13, d), nn.ReLU(), nn.Linear(d, d))
        self.visit_encoder = nn.Sequential(
            nn.Linear(d + VISIT_FEATURE_DIM, d), nn.ReLU(), nn.Linear(d, d)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d, config.heads, 4 * d, batch_first=True, dropout=0.0
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, config.layers)
        self.start_token = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.position = nn.Parameter(torch.randn(512, d) / math.sqrt(d))
        self.state_fusion = nn.Sequential(
            nn.Linear(2 * d + 5, d), nn.ReLU(), nn.Linear(d, d)
        )
        self.fleet_phi = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.fleet_rho = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.global_fusion = nn.Sequential(
            nn.Linear(2 * d + 2, d), nn.ReLU(), nn.Linear(d, d)
        )
        self.new_route = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.route_score = nn.Sequential(
            nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1)
        )
        self.customer_query = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh())
        self.customer_key = nn.Linear(d, d)
        self.relative_score = nn.Sequential(
            nn.Linear(3, d // 2), nn.ReLU(), nn.Linear(d // 2, 1)
        )
        self.close_score = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _node_features(self, instance: Instance) -> torch.Tensor:
        xs = [node.x for node in instance.nodes]
        ys = [node.y for node in instance.nodes]
        x_mid, y_mid = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        station_order = {"fast": 0, "normal": 1, "slow": 2}
        rows = []
        for node in instance.nodes:
            kind = [float(node.kind == value) for value in range(3)]
            station = [0.0, 0.0, 0.0]
            curve_features = [0.0] * 4
            if node.kind == 2 and node.station_type is not None:
                station[station_order[node.station_type]] = 1.0
                curve = instance.curves[node.station_type]
                curve_features = [
                    cumulative_charging_time(curve, fraction * instance.battery_wh)
                    / max(curve.times[-1], 1e-9)
                    for fraction in (0.25, 0.5, 0.75, 1.0)
                ]
            rows.append(
                [
                    (node.x - x_mid) / scale,
                    (node.y - y_mid) / scale,
                    *kind,
                    instance.service_times[node.id] / instance.max_route_hours,
                    *station,
                    *curve_features,
                ]
            )
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _history_vector(
        self, instance: Instance, nodes: torch.Tensor, route: dict
    ) -> torch.Tensor:
        if not self.config.history:
            return torch.zeros(self.config.d_model, device=self.device)
        plan = route.get("plan")
        stops = plan.stops if plan is not None and plan.stops is not None else ()
        visit_rows = plan_visit_features(instance, stops)
        tokens = [self.start_token]
        for node_id, visit_feat in visit_rows:
            extras = torch.tensor(visit_feat, dtype=torch.float32, device=self.device)
            tokens.append(self.visit_encoder(torch.cat([nodes[node_id], extras])))
        sequence = torch.stack(tokens)
        sequence = sequence + self.position[: len(sequence)]
        causal = torch.triu(
            torch.full(
                (len(sequence), len(sequence)),
                float("-inf"),
                device=self.device,
            ),
            diagonal=1,
        )
        return self.history_encoder(sequence.unsqueeze(0), mask=causal)[0, -1]

    def encode_nodes(self, instance: Instance) -> torch.Tensor:
        """Cacheable static node embeddings for one instance."""
        return self.node_encoder(self._node_features(instance))

    def encode_state(
        self,
        instance: Instance,
        routes: list[dict],
        unserved: set[int],
        nodes: torch.Tensor | None = None,
        history_cache: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        if nodes is None:
            nodes = self.encode_nodes(instance)
        route_vectors = []
        for route in routes:
            customers = route["customers"]
            plan = route.get("plan")
            cache_key = (
                customers,
                plan.stops if plan is not None else None,
                self.config.history,
            )
            if history_cache is not None and cache_key in history_cache:
                history = history_cache[cache_key]
            else:
                history = self._history_vector(instance, nodes, route)
                if history_cache is not None:
                    history_cache[cache_key] = history
            last_node = customers[-1] if customers else instance.depot
            scalars = torch.tensor(
                [
                    route["duration"] / instance.max_route_hours,
                    route["terminal_soc"] / instance.battery_wh,
                    max(0.0, instance.max_route_hours - route["duration"])
                    / instance.max_route_hours,
                    float(route["closed"]),
                    len(customers) / max(1, len(instance.customers)),
                ],
                dtype=torch.float32,
                device=self.device,
            )
            route_vectors.append(
                self.state_fusion(torch.cat([history, nodes[last_node], scalars]))
            )
        route_tensor = (
            torch.stack(route_vectors)
            if route_vectors
            else torch.empty((0, self.config.d_model), device=self.device)
        )
        open_indices = [
            index for index, route in enumerate(routes) if not route["closed"]
        ]
        if self.config.deepsets and len(route_tensor):
            if self.config.deepsets_open_only:
                if open_indices:
                    fleet_input = route_tensor[open_indices]
                else:
                    fleet_input = torch.empty(
                        (0, self.config.d_model), device=self.device
                    )
            else:
                fleet_input = route_tensor
            if len(fleet_input):
                fleet = self.fleet_rho(self.fleet_phi(fleet_input).sum(dim=0))
            else:
                fleet = torch.zeros(self.config.d_model, device=self.device)
        else:
            fleet = torch.zeros(self.config.d_model, device=self.device)
        if unserved:
            unserved_pool = nodes[sorted(unserved)].mean(dim=0)
        else:
            unserved_pool = torch.zeros(self.config.d_model, device=self.device)
        counts = torch.tensor(
            [
                len(routes) / max(1, len(instance.customers)),
                len(unserved) / max(1, len(instance.customers)),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        context = self.global_fusion(torch.cat([fleet, unserved_pool, counts]))
        return nodes, route_tensor, open_indices, context

    def route_logits(
        self,
        route_vectors: torch.Tensor,
        open_indices: list[int],
        context: torch.Tensor,
    ) -> torch.Tensor:
        candidates = [route_vectors[index] for index in open_indices] + [self.new_route]
        return torch.stack(
            [
                self.route_score(torch.cat([candidate, context])).squeeze(-1)
                for candidate in candidates
            ]
        )

    def customer_logits(
        self,
        instance: Instance,
        nodes: torch.Tensor,
        selected_route: torch.Tensor | None,
        selected_last_node: int,
        context: torch.Tensor,
        unserved_order: list[int],
        allow_close: bool,
    ) -> torch.Tensor:
        route_vector = self.new_route if selected_route is None else selected_route
        query = self.customer_query(torch.cat([route_vector, context]))
        if unserved_order:
            relative = torch.tensor(
                [
                    [
                        instance.distances_km[selected_last_node][customer] / 170.0,
                        instance.travel_hours[selected_last_node][customer]
                        / instance.max_route_hours,
                        instance.energy_wh[selected_last_node][customer]
                        / instance.battery_wh,
                    ]
                    for customer in unserved_order
                ],
                dtype=torch.float32,
                device=self.device,
            )
            keys = self.customer_key(nodes[unserved_order])
            pointer = torch.matmul(keys, query) / math.sqrt(self.config.d_model)
            scores = pointer + self.relative_score(relative).squeeze(-1)
        else:
            scores = torch.empty(0, device=self.device)
        if allow_close:
            close = self.close_score(torch.cat([route_vector, context])).squeeze(-1)
            scores = torch.cat([scores, close.unsqueeze(0)])
        return scores


def physical_target_socs(
    instance: Instance,
    arrival_soc: float,
    station: int,
    required_target: float,
) -> list[float]:
    """Build absolute Wh charge targets from curve breakpoints and need levels.

    Coarse battery percentages remain as fallback anchors; primary candidates
    come from the station's nonlinear curve energies and the onward energy need.
    """
    battery = instance.battery_wh
    curve = instance.curves[instance.nodes[station].station_type]
    candidates = {min(battery, max(arrival_soc + 1.0, required_target))}
    for energy in curve.energies:
        if energy > arrival_soc + 1e-7:
            candidates.add(min(battery, float(energy)))
    for fraction in (0.25, 0.5, 0.75, 0.85, 0.95, 1.0):
        level = fraction * battery
        if level > arrival_soc + 1e-7:
            candidates.add(level)
    for buffer in (1.05, 1.15, 1.3):
        level = min(battery, required_target * buffer)
        if level > arrival_soc + 1e-7:
            candidates.add(level)
    ordered = sorted(candidates)
    # Cap cardinality for a compact head while keeping extremes.
    if len(ordered) <= 12:
        return ordered
    keep = {ordered[0], ordered[-1]}
    step = max(1, (len(ordered) - 1) // 10)
    keep.update(ordered[index] for index in range(0, len(ordered), step))
    return sorted(keep)


class DirectChargePolicy(nn.Module):
    """Secondary CHARGE/station/target-SOC policy for a fixed customer order."""

    # Legacy percentage anchors retained for diagnostics; scoring uses absolute Wh.
    targets = (0.25, 0.5, 0.75, 0.85, 0.95, 1.0)

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(10, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.action_head = nn.Linear(hidden, 2)  # GO or CHARGE
        self.station_score = nn.Sequential(
            nn.Linear(hidden + 8, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.quantity_score = nn.Sequential(
            nn.Linear(hidden + 8 + 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.state(state_features)

    def station_logits(
        self, state_embedding: torch.Tensor, station_features: torch.Tensor
    ) -> torch.Tensor:
        repeated = state_embedding.expand(len(station_features), -1)
        return self.station_score(
            torch.cat([repeated, station_features], dim=-1)
        ).squeeze(-1)

    def quantity_logits(
        self,
        state_embedding: torch.Tensor,
        selected_station: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        repeated = torch.cat([state_embedding, selected_station]).expand(
            len(target_features), -1
        )
        return self.quantity_score(
            torch.cat([repeated, target_features], dim=-1)
        ).squeeze(-1)
