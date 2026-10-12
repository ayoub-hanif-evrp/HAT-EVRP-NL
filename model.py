"""Compact H1-H4 policies for hybrid E-VRP-NL route construction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from charging import cumulative_charging_time
from data import Instance


@dataclass(frozen=True)
class PolicyConfig:
    history: bool
    deepsets: bool
    d_model: int = 128
    layers: int = 2
    heads: int = 4


class HATPolicy(nn.Module):
    """Shared causal route encoder, dynamic fusion, and hierarchical pointers."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.node_encoder = nn.Sequential(nn.Linear(13, d), nn.ReLU(), nn.Linear(d, d))
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
        self.relative_score = nn.Sequential(nn.Linear(3, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))
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

    def encode_state(
        self,
        instance: Instance,
        routes: list[dict],
        unserved: set[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        nodes = self.node_encoder(self._node_features(instance))
        route_vectors = []
        for route in routes:
            customers = route["customers"]
            if self.config.history and customers:
                sequence = torch.stack(
                    [self.start_token, *[nodes[customer] for customer in customers]]
                )
                sequence = sequence + self.position[: len(sequence)]
                causal = torch.triu(
                    torch.full(
                        (len(sequence), len(sequence)),
                        float("-inf"),
                        device=self.device,
                    ),
                    diagonal=1,
                )
                history = self.history_encoder(sequence.unsqueeze(0), mask=causal)[0, -1]
            elif self.config.history:
                history = self.start_token
            else:
                history = torch.zeros(self.config.d_model, device=self.device)
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
        if self.config.deepsets and len(route_tensor):
            fleet = self.fleet_rho(self.fleet_phi(route_tensor).sum(dim=0))
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
        open_indices = [index for index, route in enumerate(routes) if not route["closed"]]
        return nodes, route_tensor, open_indices, context

    def route_logits(
        self, route_vectors: torch.Tensor, open_indices: list[int], context: torch.Tensor
    ) -> torch.Tensor:
        candidates = [route_vectors[index] for index in open_indices] + [self.new_route]
        return torch.stack(
            [self.route_score(torch.cat([candidate, context])).squeeze(-1) for candidate in candidates]
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
        scores = []
        for customer in unserved_order:
            relative = torch.tensor(
                [
                    instance.distances_km[selected_last_node][customer] / 170.0,
                    instance.travel_hours[selected_last_node][customer] / instance.max_route_hours,
                    instance.energy_wh[selected_last_node][customer] / instance.battery_wh,
                ],
                dtype=torch.float32,
                device=self.device,
            )
            scores.append(
                torch.dot(query, self.customer_key(nodes[customer])) / math.sqrt(self.config.d_model)
                + self.relative_score(relative).squeeze(-1)
            )
        if allow_close:
            scores.append(self.close_score(torch.cat([route_vector, context])).squeeze(-1))
        return torch.stack(scores)


class DirectChargePolicy(nn.Module):
    """Secondary CHARGE/station/target-SOC policy for a fixed customer order."""

    targets = (0.25, 0.5, 0.75, 0.85, 0.95, 1.0)

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(8, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.action_head = nn.Linear(hidden, 2)  # GO or CHARGE
        self.station_score = nn.Sequential(
            nn.Linear(hidden + 8, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.quantity_head = nn.Sequential(
            nn.Linear(hidden + 8, hidden), nn.ReLU(), nn.Linear(hidden, len(self.targets))
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
        return self.station_score(torch.cat([repeated, station_features], dim=-1)).squeeze(-1)

    def quantity_logits(
        self, state_embedding: torch.Tensor, selected_station: torch.Tensor
    ) -> torch.Tensor:
        return self.quantity_head(torch.cat([state_embedding, selected_station]))
