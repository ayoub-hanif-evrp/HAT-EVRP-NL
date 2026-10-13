"""Synthetic teacher generation, H1-H4 training, and verified decoding."""

from __future__ import annotations

import argparse
import csv
import copy
from dataclasses import asdict
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from charging import (
    FixedRouteOracle,
    RoutePlan,
    charging_time,
    check_route,
    check_solution,
)
from data import Instance, generate_synthetic
from model import (
    DirectChargePolicy,
    HATPolicy,
    PolicyConfig,
    physical_target_socs,
)
from solver import randomized_greedy


VARIANTS = {
    "H1": PolicyConfig(history=False, deepsets=False),
    "H2": PolicyConfig(history=True, deepsets=False),
    "H3": PolicyConfig(history=False, deepsets=True),
    "H4": PolicyConfig(history=True, deepsets=True),
}


def _direct_features(
    instance: Instance,
    current: int,
    next_customer: int,
    soc: float,
    remaining: int,
    elapsed: float,
) -> torch.Tensor:
    node, target = instance.nodes[current], instance.nodes[next_customer]
    remaining_hours = max(0.0, instance.max_route_hours - elapsed)
    return torch.tensor(
        [
            node.x / 120.0,
            node.y / 120.0,
            target.x / 120.0,
            target.y / 120.0,
            soc / instance.battery_wh,
            instance.energy_wh[current][next_customer] / instance.battery_wh,
            remaining / max(1, len(instance.customers)),
            float(node.kind == 2),
            elapsed / instance.max_route_hours,
            remaining_hours / instance.max_route_hours,
        ],
        dtype=torch.float32,
    )


def _target_features(
    instance: Instance, arrival: float, station: int, target_soc: float
) -> torch.Tensor:
    duration = charging_time(
        instance.curves[instance.nodes[station].station_type], arrival, target_soc
    )
    return torch.tensor(
        [
            target_soc / instance.battery_wh,
            (target_soc - arrival) / instance.battery_wh,
            duration / instance.max_route_hours,
            float(instance.nodes[station].station_type == "fast"),
        ],
        dtype=torch.float32,
    )


def _station_features(instance: Instance, current: int) -> torch.Tensor:
    technology = {"fast": 0, "normal": 1, "slow": 2}
    rows = []
    for station in instance.stations:
        node = instance.nodes[station]
        one_hot = [0.0, 0.0, 0.0]
        one_hot[technology[node.station_type]] = 1.0
        rows.append(
            [
                node.x / 120.0,
                node.y / 120.0,
                instance.distances_km[current][station] / 170.0,
                instance.energy_wh[current][station] / instance.battery_wh,
                *one_hot,
                instance.curves[node.station_type].times[-1] / 2.04,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def direct_examples(
    instance: Instance, customer_routes: tuple[tuple[int, ...], ...]
) -> list[tuple]:
    """Extract GO/CHARGE, station, and absolute target-SOC labels from oracle plans."""
    oracle = FixedRouteOracle(instance)
    examples = []
    for customers in customer_routes:
        plan = oracle.solve(customers)
        if not plan.feasible:
            continue
        current = instance.depot
        soc = instance.battery_wh
        elapsed = 0.0
        customer_position = 0
        for node, amount in plan.stops[1:]:
            next_customer = (
                customers[customer_position]
                if customer_position < len(customers)
                else instance.depot
            )
            state = _direct_features(
                instance,
                current,
                next_customer,
                soc,
                len(customers) - customer_position,
                elapsed,
            )
            stations = _station_features(instance, current)
            travel = instance.travel_hours[current][node]
            if instance.nodes[node].kind == 2:
                arrival = soc - instance.energy_wh[current][node]
                target_soc = arrival + float(amount)
                required_target = instance.energy_wh[node][next_customer] + (
                    0.0
                    if next_customer == instance.depot
                    else min(
                        instance.energy_wh[next_customer][other]
                        for other in (instance.depot, *instance.stations)
                    )
                )
                candidates = physical_target_socs(
                    instance, arrival, node, required_target
                )
                quantity = min(
                    range(len(candidates)),
                    key=lambda index: abs(candidates[index] - target_soc),
                )
                target_feats = torch.stack(
                    [
                        _target_features(instance, arrival, node, candidate)
                        for candidate in candidates
                    ]
                )
                examples.append(
                    (
                        state,
                        stations,
                        1,
                        instance.stations.index(node),
                        quantity,
                        target_feats,
                    )
                )
                charge_dur = charging_time(
                    instance.curves[instance.nodes[node].station_type],
                    arrival,
                    target_soc,
                )
                elapsed += travel + charge_dur
                soc = target_soc
            else:
                examples.append((state, stations, 0, -1, -1, None))
                elapsed += travel + instance.service_times[node]
                soc -= instance.energy_wh[current][node]
                if node != instance.depot:
                    customer_position += 1
            current = node
    return examples


def _onward_energy_need(instance: Instance, node: int) -> float:
    if node == instance.depot:
        return 0.0
    return min(
        instance.energy_wh[node][other]
        for other in (instance.depot, *instance.stations)
    )


def _station_supports_next(
    instance: Instance, station: int, next_customer: int, full_soc: float
) -> bool:
    """True if a full battery at station can reach next, possibly via one more CS."""
    need = instance.energy_wh[station][next_customer] + _onward_energy_need(
        instance, next_customer
    )
    if need <= full_soc + 1e-7:
        return True
    for relay in instance.stations:
        if relay == station:
            continue
        if instance.energy_wh[station][relay] > full_soc + 1e-7:
            continue
        if (
            instance.energy_wh[relay][next_customer]
            + _onward_energy_need(instance, next_customer)
            <= full_soc + 1e-7
        ):
            return True
    return False


def decode_direct(
    model: DirectChargePolicy, instance: Instance, customers: tuple[int, ...]
) -> dict:
    """Execute the learned CHARGE policy without post-optimization."""
    current = instance.depot
    soc = instance.battery_wh
    elapsed = 0.0
    customer_position = 0
    stops: list[tuple[int, float | None]] = [(instance.depot, None)]
    max_decisions = 4 * (len(customers) + 1)
    reason = ""
    start = time.perf_counter()
    for _ in range(max_decisions):
        if customer_position == len(customers) and current == instance.depot:
            break
        next_customer = (
            customers[customer_position]
            if customer_position < len(customers)
            else instance.depot
        )
        state_features = _direct_features(
            instance,
            current,
            next_customer,
            soc,
            len(customers) - customer_position,
            elapsed,
        ).to(model.device)
        stations = _station_features(instance, current).to(model.device)
        state = model.encode(state_features)
        action_logits = model.action_head(state)
        energy_to_next = instance.energy_wh[current][next_customer]
        travel_to_next = instance.travel_hours[current][next_customer]
        service_next = instance.service_times[next_customer]
        arrival_at_next = soc - energy_to_next
        remaining_hours = instance.max_route_hours - elapsed
        if next_customer == instance.depot:
            onward_safe = True
            time_ok_go = travel_to_next <= remaining_hours + 1e-9
        else:
            onward_safe = arrival_at_next + 1e-7 >= _onward_energy_need(
                instance, next_customer
            )
            # Optimistic time lower bound: travel+service; return may include charging.
            time_ok_go = travel_to_next + service_next <= remaining_hours + 1e-9
        go_feasible = (
            energy_to_next <= soc + 1e-7 and onward_safe and time_ok_go
        )
        reachable = []
        for station in instance.stations:
            energy_to_station = instance.energy_wh[current][station]
            energy_ok = (
                energy_to_station <= soc + 1e-7
                and station != current
                and _station_supports_next(
                    instance, station, next_customer, instance.battery_wh
                )
            )
            if not energy_ok:
                reachable.append(False)
                continue
            arrival = soc - energy_to_station
            required_target = min(
                instance.battery_wh,
                instance.energy_wh[station][next_customer]
                + _onward_energy_need(instance, next_customer),
            )
            # If one hop is insufficient, charge toward full for a relay hop.
            if required_target > instance.battery_wh + 1e-7 or not (
                instance.energy_wh[station][next_customer]
                + _onward_energy_need(instance, next_customer)
                <= instance.battery_wh + 1e-7
            ):
                required_target = instance.battery_wh
            candidates = physical_target_socs(
                instance, arrival, station, required_target
            )
            travel_cs = instance.travel_hours[current][station]
            time_feasible = False
            for target_soc in candidates:
                if target_soc + 1e-7 < min(required_target, arrival + 1.0):
                    continue
                charge_dur = charging_time(
                    instance.curves[instance.nodes[station].station_type],
                    arrival,
                    target_soc,
                )
                after = (
                    travel_cs
                    + charge_dur
                    + instance.travel_hours[station][next_customer]
                    + instance.service_times[next_customer]
                )
                if after <= remaining_hours + 1e-9:
                    time_feasible = True
                    break
            reachable.append(time_feasible)
        action_mask = torch.tensor(
            [go_feasible, any(reachable)], dtype=torch.bool, device=model.device
        )
        if not bool(action_mask.any()):
            reason = "no feasible GO or CHARGE action"
            break
        action = int(action_logits.masked_fill(~action_mask, float("-inf")).argmax())
        if action == 0:
            elapsed += travel_to_next + service_next
            soc -= instance.energy_wh[current][next_customer]
            current = next_customer
            stops.append((current, None))
            if current != instance.depot:
                customer_position += 1
            continue

        station_logits = model.station_logits(state, stations)
        station_mask = torch.tensor(reachable, dtype=torch.bool, device=model.device)
        station_index = int(
            station_logits.masked_fill(~station_mask, float("-inf")).argmax()
        )
        station = instance.stations[station_index]
        arrival = soc - instance.energy_wh[current][station]
        direct_need = instance.energy_wh[station][next_customer] + _onward_energy_need(
            instance, next_customer
        )
        required_target = (
            min(instance.battery_wh, direct_need)
            if direct_need <= instance.battery_wh + 1e-7
            else instance.battery_wh
        )
        candidates = physical_target_socs(
            instance, arrival, station, required_target
        )
        target_feats = torch.stack(
            [
                _target_features(instance, arrival, station, candidate)
                for candidate in candidates
            ]
        ).to(model.device)
        quantity_logits = model.quantity_logits(
            state, stations[station_index], target_feats
        )
        travel_cs = instance.travel_hours[current][station]
        quantity_mask = []
        charge_durs = []
        for target_soc in candidates:
            if target_soc <= arrival + 1e-7:
                quantity_mask.append(False)
                charge_durs.append(float("inf"))
                continue
            if (
                direct_need <= instance.battery_wh + 1e-7
                and target_soc + 1e-7 < required_target
            ):
                quantity_mask.append(False)
                charge_durs.append(float("inf"))
                continue
            charge_dur = charging_time(
                instance.curves[instance.nodes[station].station_type],
                arrival,
                target_soc,
            )
            after = (
                travel_cs
                + charge_dur
                + instance.travel_hours[station][next_customer]
                + instance.service_times[next_customer]
            )
            quantity_mask.append(after <= remaining_hours + 1e-9)
            charge_durs.append(charge_dur)
        quantity_mask_t = torch.tensor(
            quantity_mask, dtype=torch.bool, device=model.device
        )
        if not bool(quantity_mask_t.any()):
            reason = "no positive target-SOC action"
            break
        masked_logits = quantity_logits.masked_fill(~quantity_mask_t, float("-inf"))
        if remaining_hours < 3.0:
            # Prefer the shortest feasible charge when time is scarce.
            best = None
            for index, ok in enumerate(quantity_mask):
                if not ok:
                    continue
                if best is None or charge_durs[index] < charge_durs[best] - 1e-12:
                    best = index
            quantity = int(best)
        else:
            quantity = int(masked_logits.argmax())
        target_soc = candidates[quantity]
        amount = target_soc - arrival
        charge_dur = charging_time(
            instance.curves[instance.nodes[station].station_type], arrival, target_soc
        )
        elapsed += travel_cs + charge_dur
        current, soc = station, target_soc
        stops.append((station, amount))
    concrete = tuple(stops)
    check = check_route(instance, concrete, customers)
    return {
        "feasible": check.feasible,
        "objective": check.duration_hours if check.feasible else float("inf"),
        "stops": concrete,
        "reason": reason or check.reason,
        "runtime": time.perf_counter() - start,
    }


def train_direct(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = make_teacher_data(
        args.size, args.train_instances, 30000 + args.seed * 100
    )
    examples = [
        example
        for instance, routes in teacher
        for example in direct_examples(instance, routes)
    ]
    model = DirectChargePolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    losses = []
    for step in range(1, args.steps + 1):
        example = examples[(step - 1) % len(examples)]
        state_features, station_features, action, station, quantity, target_feats = (
            example
        )
        state_features = state_features.to(device)
        station_features = station_features.to(device)
        state = model.encode(state_features)
        loss = F.cross_entropy(
            model.action_head(state).unsqueeze(0),
            torch.tensor([action], device=device),
        )
        if action == 1:
            loss = loss + F.cross_entropy(
                model.station_logits(state, station_features).unsqueeze(0),
                torch.tensor([station], device=device),
            )
            loss = loss + F.cross_entropy(
                model.quantity_logits(
                    state, station_features[station], target_feats.to(device)
                ).unsqueeze(0),
                torch.tensor([quantity], device=device),
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    version = getattr(args, "version", "c2")
    checkpoint = (
        args.checkpoints / f"CHARGE_n{args.size}_seed{args.seed}_{version}.pt"
    )
    torch.save(
        {"state_dict": model.state_dict(), "size": args.size, "version": version},
        checkpoint,
    )
    args.results.mkdir(parents=True, exist_ok=True)
    output = (
        args.results / f"training_CHARGE_n{args.size}_seed{args.seed}_{version}.csv"
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "elapsed_s", "device"])
        writer.writeheader()
        writer.writerow(
            {
                "step": args.steps,
                "loss": sum(losses[-min(50, len(losses)) :]) / min(50, len(losses)),
                "elapsed_s": time.perf_counter() - start,
                "device": str(device),
            }
        )
    print(
        f"trained CHARGE on {len(examples)} expert actions; "
        f"final loss={losses[-1]:.6f}; saved {checkpoint}"
    )


def route_state(instance: Instance, customers: tuple[int, ...], plan: RoutePlan) -> dict:
    check = check_route(instance, plan.stops, customers)
    if not check.feasible:
        raise RuntimeError(check.reason)
    return {
        "customers": customers,
        "plan": plan,
        "duration": plan.duration_hours,
        "terminal_soc": check.end_soc_wh,
        "closed": False,
    }


def make_teacher_data(
    size: int, count: int, seed_start: int
) -> list[tuple[Instance, tuple[tuple[int, ...], ...]]]:
    data = []
    candidate_seed = seed_start
    while len(data) < count and candidate_seed < seed_start + count * 40:
        instance = generate_synthetic(size, candidate_seed)
        result = randomized_greedy(
            instance,
            restarts=3,
            seed=candidate_seed,
            local_search_passes=1,
            local_search_budget=800,
        )
        if result.feasible and len(result.customer_routes) >= 1:
            data.append((instance, result.customer_routes))
        candidate_seed += 1
    if len(data) < count:
        raise RuntimeError(
            f"Only generated {len(data)}/{count} feasible teacher instances"
        )
    return data


def interleaved_actions(
    teacher_routes: tuple[tuple[int, ...], ...], rng: random.Random
) -> list[tuple[str, int | None, int | None]]:
    """Build a valid interleaved construction: preserve per-route order.

    Actions are ('extend', route_id, customer), ('close', route_id, None),
    or starting a route is implied by the first extend of an unstarted route.
    Multiple routes stay open while they still have remaining customers.
    """
    remaining = [list(route) for route in teacher_routes]
    started = [False] * len(teacher_routes)
    open_ids = set()
    actions: list[tuple[str, int | None, int | None]] = []
    while any(remaining[i] for i in range(len(remaining))) or open_ids:
        extendable = [
            i for i in range(len(remaining)) if remaining[i] and (started[i] or True)
        ]
        closable = [i for i in open_ids if not remaining[i]]
        choices: list[tuple[str, int]] = []
        for route_id in extendable:
            choices.append(("extend", route_id))
        for route_id in closable:
            choices.append(("close", route_id))
        if not choices:
            break
        kind, route_id = rng.choice(choices)
        if kind == "close":
            actions.append(("close", route_id, None))
            open_ids.remove(route_id)
            continue
        customer = remaining[route_id].pop(0)
        actions.append(("extend", route_id, customer))
        started[route_id] = True
        open_ids.add(route_id)
    for route_id in sorted(open_ids):
        actions.append(("close", route_id, None))
    return actions


def teacher_loss(
    model: HATPolicy,
    instance: Instance,
    teacher_routes: tuple[tuple[int, ...], ...],
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, int, dict]:
    """Imitation loss on an interleaved multi-open-route trajectory."""
    rng = rng or random.Random(0)
    oracle = FixedRouteOracle(instance)
    routes: list[dict] = []
    # Map expert route id -> live route index once started.
    live_index: dict[int, int] = {}
    unserved = set(instance.customers)
    losses = []
    decisions = 0
    multi_open_states = 0
    competing_states = 0
    total_states = 0

    for kind, expert_id, customer in interleaved_actions(teacher_routes, rng):
        nodes, route_vectors, open_indices, context = model.encode_state(
            instance, routes, unserved
        )
        total_states += 1
        if len(open_indices) >= 2:
            multi_open_states += 1
        route_scores = model.route_logits(route_vectors, open_indices, context)

        if kind == "extend":
            assert expert_id is not None and customer is not None
            if expert_id not in live_index:
                route_target = len(open_indices)  # NEW_ROUTE
                selected_vector = None
                last_node = instance.depot
                prefix: tuple[int, ...] = ()
                is_new = True
            else:
                active_index = live_index[expert_id]
                route_target = open_indices.index(active_index)
                selected_vector = route_vectors[active_index]
                prefix = routes[active_index]["customers"]
                last_node = prefix[-1]
                is_new = False
            losses.append(
                F.cross_entropy(
                    route_scores.unsqueeze(0),
                    torch.tensor([route_target], device=model.device),
                )
            )
            order = sorted(unserved)
            customer_scores = model.customer_logits(
                instance,
                nodes,
                selected_vector,
                last_node,
                context,
                order,
                allow_close=not is_new,
            )
            masks = [
                oracle.solve((*prefix, candidate)).feasible for candidate in order
            ]
            if not is_new:
                masks.append(True)
            if sum(masks[: len(order)]) >= 2 and len(open_indices) >= 1:
                competing_states += 1
            masked_scores = customer_scores.masked_fill(
                ~torch.tensor(masks, dtype=torch.bool, device=model.device),
                float("-inf"),
            )
            customer_target = order.index(customer)
            losses.append(
                F.cross_entropy(
                    masked_scores.unsqueeze(0),
                    torch.tensor([customer_target], device=model.device),
                )
            )
            plan = oracle.solve((*prefix, customer))
            if is_new:
                routes.append(route_state(instance, (customer,), plan))
                live_index[expert_id] = len(routes) - 1
            else:
                routes[live_index[expert_id]] = route_state(
                    instance, (*prefix, customer), plan
                )
            unserved.remove(customer)
            decisions += 1
            continue

        # CLOSE an expert route that has no remaining customers.
        assert expert_id is not None
        active_index = live_index[expert_id]
        route_target = open_indices.index(active_index)
        losses.append(
            F.cross_entropy(
                route_scores.unsqueeze(0),
                torch.tensor([route_target], device=model.device),
            )
        )
        order = sorted(unserved)
        prefix = routes[active_index]["customers"]
        customer_scores = model.customer_logits(
            instance,
            nodes,
            route_vectors[active_index],
            prefix[-1] if prefix else instance.depot,
            context,
            order,
            allow_close=True,
        )
        masks = [oracle.solve((*prefix, candidate)).feasible for candidate in order]
        masks.append(True)
        masked_scores = customer_scores.masked_fill(
            ~torch.tensor(masks, dtype=torch.bool, device=model.device),
            float("-inf"),
        )
        losses.append(
            F.cross_entropy(
                masked_scores.unsqueeze(0),
                torch.tensor([len(order)], device=model.device),
            )
        )
        routes[active_index]["closed"] = True
        decisions += 1

    stats = {
        "multi_open_frac": multi_open_states / max(1, total_states),
        "competing_frac": competing_states / max(1, total_states),
        "total_states": total_states,
    }
    return torch.stack(losses).mean(), decisions, stats


def diagnose_deepsets(model: HATPolicy, instance: Instance) -> dict:
    """Compact checks that DeepSets is permutation-invariant and informative."""
    if not model.config.deepsets:
        return {"deepsets": False}
    oracle = FixedRouteOracle(instance)
    routes = []
    for customer in instance.customers[:2]:
        plan = oracle.solve((customer,))
        if plan.feasible:
            routes.append(route_state(instance, (customer,), plan))
    if len(routes) < 2:
        return {"deepsets": True, "skipped": True}
    unserved = set(instance.customers) - {routes[0]["customers"][0], routes[1]["customers"][0]}
    model.zero_grad(set_to_none=True)
    _, route_vectors, open_indices, context = model.encode_state(
        instance, routes, unserved
    )
    logits = model.route_logits(route_vectors, open_indices, context)
    loss = logits.sum()
    loss.backward()
    phi_grad = 0.0
    for parameter in model.fleet_phi.parameters():
        if parameter.grad is not None:
            phi_grad += float(parameter.grad.abs().sum())
    with torch.no_grad():
        _, route_perm, open_perm, context_perm = model.encode_state(
            instance, list(reversed(routes)), unserved
        )
        fleet_a = model.fleet_rho(model.fleet_phi(route_vectors).sum(dim=0))
        fleet_b = model.fleet_rho(model.fleet_phi(route_perm).sum(dim=0))
        perm_error = float((fleet_a - fleet_b).abs().max())
        routes_perturbed = copy.deepcopy(routes)
        routes_perturbed[1]["duration"] = min(
            instance.max_route_hours, routes_perturbed[1]["duration"] + 1.5
        )
        _, rv2, oi2, ctx2 = model.encode_state(instance, routes_perturbed, unserved)
        logits2 = model.route_logits(rv2, oi2, ctx2)
        active_delta = float((logits - logits2).abs().max())
    return {
        "deepsets": True,
        "phi_grad": phi_grad,
        "perm_error": perm_error,
        "active_logit_delta": active_delta,
    }


def decode(
    model: HATPolicy,
    instance: Instance,
    sample: bool = False,
    seed: int = 0,
    local_passes: int = 0,
    local_budget: int = 0,
    cost_lambda: float = 0.0,
    route_lambda: float = 0.0,
    local_operators: str = "basic",
    route_cost_top_k: int = 3,
) -> dict:
    """Greedy/sampled hierarchical decode with optional oracle-aware scoring.

    When ``cost_lambda > 0``, customer pointer logits are adjusted by
    ``logit - cost_lambda * (D(R⊕j) - D(R)) / T_max``, using FRVCP durations
    already required for feasibility masks (cache hits; no redundant solves).

    When ``route_lambda > 0``, route/new-route logits are biased by the best
    feasible one-customer extension potential among a shortlist of top learned
    route candidates (plus NEW_ROUTE). ``route_lambda=0`` and ``cost_lambda=0``
    preserve Cycle-3 decode behavior.
    """
    from solver import local_search

    start = time.perf_counter()
    if sample:
        torch.manual_seed(seed)
    oracle = FixedRouteOracle(instance)
    routes: list[dict] = []
    unserved = set(instance.customers)
    log_probs: list[torch.Tensor] = []
    entropy = 0.0
    reason = ""
    max_decisions = 3 * len(instance.customers) + 5
    nodes = model.encode_nodes(instance)
    history_cache: dict = {}
    singleton_plans = {
        customer: oracle.solve((customer,)) for customer in instance.customers
    }
    singleton_ok = {
        customer: plan.feasible for customer, plan in singleton_plans.items()
    }

    for decision in range(max_decisions):
        if not unserved and all(route["closed"] for route in routes):
            break
        nodes, route_vectors, open_indices, context = model.encode_state(
            instance, routes, unserved, nodes=nodes, history_cache=history_cache
        )
        route_scores = model.route_logits(route_vectors, open_indices, context)
        new_feasible = bool(unserved) and any(
            singleton_ok[customer] for customer in unserved
        )
        route_mask = [True] * len(open_indices) + [new_feasible]
        if not any(route_mask):
            reason = "no route action can serve remaining customers"
            break
        route_scores = route_scores.masked_fill(
            ~torch.tensor(route_mask, dtype=torch.bool, device=model.device),
            float("-inf"),
        )
        if route_lambda != 0.0 and unserved:
            scale = max(instance.max_route_hours, 1e-6)
            adjusted_routes = route_scores.clone()
            finite = [
                index
                for index in range(len(open_indices) + 1)
                if bool(route_mask[index])
            ]
            # Shortlist: top-k open routes by learned score, always include NEW_ROUTE if feasible.
            open_finite = [index for index in finite if index < len(open_indices)]
            open_finite.sort(
                key=lambda index: float(route_scores[index].detach()), reverse=True
            )
            shortlist = open_finite[: max(0, route_cost_top_k)]
            if new_feasible:
                shortlist.append(len(open_indices))
            for route_pos in shortlist:
                if route_pos == len(open_indices):
                    potentials = [
                        plan.duration_hours
                        for customer, plan in singleton_plans.items()
                        if customer in unserved and plan.feasible
                    ]
                    if not potentials:
                        continue
                    potential = min(potentials)
                else:
                    route_index = open_indices[route_pos]
                    prefix = routes[route_index]["customers"]
                    base = float(routes[route_index]["duration"])
                    potentials = []
                    for customer in sorted(unserved):
                        plan = oracle.solve((*prefix, customer))
                        if plan.feasible:
                            potentials.append(plan.duration_hours - base)
                    if not potentials:
                        continue
                    potential = min(potentials)
                adjusted_routes[route_pos] = (
                    adjusted_routes[route_pos] - float(route_lambda) * potential / scale
                )
            route_scores = adjusted_routes.masked_fill(
                ~torch.tensor(route_mask, dtype=torch.bool, device=model.device),
                float("-inf"),
            )
        route_dist = torch.distributions.Categorical(logits=route_scores)
        route_action = route_dist.sample() if sample else torch.argmax(route_scores)
        log_probs.append(route_dist.log_prob(route_action))
        entropy += float(route_dist.entropy().detach())
        route_position = int(route_action)
        is_new = route_position == len(open_indices)
        selected_index = None if is_new else open_indices[route_position]
        selected_vector = None if is_new else route_vectors[selected_index]
        prefix = () if is_new else routes[selected_index]["customers"]
        last_node = instance.depot if is_new else prefix[-1]
        order = sorted(unserved)
        customer_scores = model.customer_logits(
            instance,
            nodes,
            selected_vector,
            last_node,
            context,
            order,
            allow_close=not is_new,
        )
        if is_new:
            candidate_plans = [singleton_plans[candidate] for candidate in order]
            masks = [plan.feasible for plan in candidate_plans]
            base_duration = 0.0
        else:
            candidate_plans = [
                oracle.solve((*prefix, candidate)) for candidate in order
            ]
            masks = [plan.feasible for plan in candidate_plans]
            masks.append(True)
            base_duration = float(routes[selected_index]["duration"])
        if not any(masks):
            reason = "selected route has no feasible action"
            break
        if cost_lambda != 0.0:
            adjusted = customer_scores.clone()
            scale = max(instance.max_route_hours, 1e-6)
            for index, plan in enumerate(candidate_plans):
                if not plan.feasible:
                    continue
                marginal = (plan.duration_hours - base_duration) / scale
                adjusted[index] = adjusted[index] - float(cost_lambda) * marginal
            customer_scores = adjusted
        customer_scores = customer_scores.masked_fill(
            ~torch.tensor(masks, dtype=torch.bool, device=model.device),
            float("-inf"),
        )
        customer_dist = torch.distributions.Categorical(logits=customer_scores)
        customer_action = (
            customer_dist.sample() if sample else torch.argmax(customer_scores)
        )
        log_probs.append(customer_dist.log_prob(customer_action))
        entropy += float(customer_dist.entropy().detach())
        action = int(customer_action)
        if not is_new and action == len(order):
            routes[selected_index]["closed"] = True
            continue
        customer = order[action]
        plan = (
            singleton_plans[customer]
            if is_new
            else oracle.solve((*prefix, customer))
        )
        if is_new:
            routes.append(route_state(instance, (customer,), plan))
        else:
            routes[selected_index] = route_state(
                instance, (*prefix, customer), plan
            )
        unserved.remove(customer)
    else:
        reason = "decision budget exceeded"

    customer_routes = [route["customers"] for route in routes]
    plans = [route["plan"] for route in routes]
    improvements = 0
    search_diag = None
    if local_passes > 0 and local_budget > 0 and not unserved:
        customer_routes, plans, improvements, search_diag = local_search(
            customer_routes,
            plans,
            oracle,
            local_passes,
            local_budget,
            operators=local_operators,
        )
    valid, validation_reason = check_solution(instance, plans) if not unserved else (
        False,
        f"{len(unserved)} customers unserved",
    )
    if not valid:
        reason = reason or validation_reason
    from charging import literature_compatible_objective_hours

    objective = sum(plan.duration_hours for plan in plans) if valid else float("inf")
    objective_lit = (
        literature_compatible_objective_hours(instance, objective)
        if valid
        else float("inf")
    )
    return {
        "feasible": valid,
        "objective": objective,
        "objective_lit": objective_lit,
        "routes": tuple(customer_routes),
        "plans": plans,
        "runtime": time.perf_counter() - start,
        "oracle_calls": oracle.calls,
        "reason": reason,
        "log_prob": torch.stack(log_probs).sum() if log_probs else None,
        "entropy": entropy / max(1, len(log_probs)),
        "decisions": decision + 1,
        "local_improvements": improvements,
        "cost_lambda": float(cost_lambda),
        "route_lambda": float(route_lambda),
        "search_diagnostics": None if search_diag is None else search_diag.as_dict(),
    }


def train_model(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version = getattr(args, "version", "c2")
    train_data = make_teacher_data(
        args.size, args.train_instances, 10000 + args.seed * 100
    )
    validation = make_teacher_data(
        args.size, args.val_instances, 20000 + args.seed * 100
    )
    model = HATPolicy(VARIANTS[args.model]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rows = []
    start = time.perf_counter()
    best_score = (-1.0, float("-inf"))
    best_state = copy.deepcopy(model.state_dict())
    multi_open_sum = 0.0
    train_rng = random.Random(args.seed + 17)

    diag = diagnose_deepsets(model, train_data[0][0])
    print({"diagnostics": diag})

    for step in range(1, args.steps + 1):
        instance, routes = train_data[(step - 1) % len(train_data)]
        optimizer.zero_grad()
        loss, decisions, stats = teacher_loss(
            model, instance, routes, rng=train_rng
        )
        multi_open_sum += stats["multi_open_frac"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                decoded = [decode(model, val_inst) for val_inst, _ in validation]
            feasible = [result for result in decoded if result["feasible"]]
            mean_objective = (
                sum(result["objective"] for result in feasible) / len(feasible)
                if feasible
                else float("nan")
            )
            row = {
                "stage": "bc",
                "step": step,
                "loss": float(loss.detach()),
                "feasible_rate": len(feasible) / len(decoded),
                "validation_objective": mean_objective,
                "entropy": sum(result["entropy"] for result in decoded)
                / len(decoded),
                "elapsed_s": time.perf_counter() - start,
                "device": str(device),
                "decisions": decisions,
                "multi_open_frac": multi_open_sum / step,
                "competing_frac": stats["competing_frac"],
            }
            rows.append(row)
            print(row)
            score = (row["feasible_rate"], -mean_objective)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            model.train()

    for rl_step in range(1, args.rl_steps + 1):
        instance, _ = train_data[(rl_step - 1) % len(train_data)]
        model.eval()
        with torch.no_grad():
            greedy = decode(model, instance, sample=False, seed=args.seed + rl_step)
        model.train()
        sampled = decode(model, instance, sample=True, seed=args.seed + rl_step)
        sampled_reward = -sampled["objective"] if sampled["feasible"] else -100.0
        greedy_reward = -greedy["objective"] if greedy["feasible"] else -100.0
        advantage = sampled_reward - greedy_reward
        if sampled["log_prob"] is not None and abs(advantage) > 1e-9:
            optimizer.zero_grad()
            rl_loss = -advantage * sampled["log_prob"]
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            rl_loss = torch.tensor(0.0)
        if rl_step == args.rl_steps:
            model.eval()
            with torch.no_grad():
                decoded = [
                    decode(model, val_instance) for val_instance, _ in validation
                ]
            feasible = [result for result in decoded if result["feasible"]]
            mean_objective = (
                sum(result["objective"] for result in feasible) / len(feasible)
                if feasible
                else float("nan")
            )
            row = {
                "stage": "rl",
                "step": rl_step,
                "loss": float(rl_loss.detach()),
                "feasible_rate": len(feasible) / len(decoded),
                "validation_objective": mean_objective,
                "entropy": sum(result["entropy"] for result in decoded)
                / len(decoded),
                "elapsed_s": time.perf_counter() - start,
                "device": str(device),
                "decisions": sampled["decisions"],
                "multi_open_frac": multi_open_sum / max(1, args.steps),
                "competing_frac": "",
            }
            rows.append(row)
            print(row)
            score = (row["feasible_rate"], -mean_objective)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final_diag = diagnose_deepsets(model, validation[0][0])
    print({"final_diagnostics": final_diag})
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        args.checkpoints / f"{args.model}_n{args.size}_seed{args.seed}_{version}.pt"
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(VARIANTS[args.model]),
            "model": args.model,
            "size": args.size,
            "seed": args.seed,
            "version": version,
            "diagnostics": final_diag,
        },
        checkpoint,
    )
    args.results.mkdir(parents=True, exist_ok=True)
    log_path = (
        args.results
        / f"training_{args.model}_n{args.size}_seed{args.seed}_{version}.csv"
    )
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"saved {checkpoint} and {log_path}; total {time.perf_counter() - start:.2f}s"
    )


def load_policy(path: str | Path, device: torch.device | None = None) -> HATPolicy:
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HATPolicy(PolicyConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_direct_policy(
    path: str | Path, device: torch.device | None = None
) -> DirectChargePolicy:
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DirectChargePolicy().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(*VARIANTS, "CHARGE"), required=True)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--rl-steps", type=int, default=10)
    parser.add_argument("--train-instances", type=int, default=12)
    parser.add_argument("--val-instances", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="c3")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    if args.model == "CHARGE":
        train_direct(args)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
